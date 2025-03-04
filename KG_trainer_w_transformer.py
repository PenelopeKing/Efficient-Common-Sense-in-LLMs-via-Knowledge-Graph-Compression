import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    BartConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from datasets import Dataset, load_from_disk

# -----------------------------------------------------------
# Import KG resources from your KG_utils file
# -----------------------------------------------------------
from KG_utils import (
    load_resources,
    load_cpnet,
    load_total_concepts,
    get_subgraph_for_message,
    concept2id,
    relation2id,
)

# Initialize KG resources.
load_resources()
load_cpnet()
load_total_concepts("data/eg")

# -----------------------------------------------------------
# Full Graph Extraction (Preprocessing)
# -----------------------------------------------------------
def get_graph_info(text):
    """
    Extracts the full KG subgraph information from the text.
    
    Returns:
        dict: Contains:
            - "concept_ids": tensor of shape (num_nodes,) with node IDs.
            - "edge_index": tensor of shape (2, num_edges) with edge indices.
            - "edge_type": tensor of shape (num_edges,) with edge type IDs.
    """
    subgraph = get_subgraph_for_message(text)
    
    nodes = list(subgraph.nodes())
    concept_ids = [concept2id.get(node, 0) for node in nodes]
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    edge_list = []
    edge_types = []
    for u, v, data in subgraph.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            relation = data.get("relation", "default")
            edge_types.append(relation2id.get(relation, 0))
    
    concept_ids = torch.tensor(concept_ids, dtype=torch.long)
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).transpose(0, 1)
        edge_type = torch.tensor(edge_types, dtype=torch.long)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    
    return {
        "concept_ids": concept_ids,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }

# -----------------------------------------------------------
# Custom GPS Layer: Alternating Local R-GCN and Global Transformer
# -----------------------------------------------------------
from torch_geometric.nn import RGCNConv

class GPSLayer(nn.Module):
    def __init__(self, hidden_dim, num_relations, nhead=8):
        """
        A single GPS layer that alternates between local (R-GCN) and global (Transformer)
        processing of node features.
        
        Args:
            hidden_dim (int): Dimension of node features.
            num_relations (int): Number of relation types.
            nhead (int): Number of attention heads.
        """
        super(GPSLayer, self).__init__()
        self.local = RGCNConv(hidden_dim, hidden_dim, num_relations)
        self.global_transformer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_type):
        # Local R-GCN update.
        local_out = self.local(x, edge_index, edge_type)
        x = x + local_out  # residual connection
        x = self.norm1(x)
        # Global transformer update.
        # Transformer expects input shape (num_nodes, batch_size, hidden_dim). Here batch_size=1.
        x_trans = self.global_transformer(x.unsqueeze(1)).squeeze(1)
        x = x + x_trans  # residual connection
        x = self.norm2(x)
        return x

# -----------------------------------------------------------
# New Graph Encoder with GPS and SAG Pooling for Node Scoring
# -----------------------------------------------------------
class GraphEncoderGPS(nn.Module):
    def __init__(self, hidden_size, num_concepts, num_layers, num_relations, compress_ratio=0.5, nhead=8):
        """
        Encodes a full KG subgraph using a stack of GPS layers for node feature refinement,
        then scores nodes for top-k selection via SAG pooling.
        
        Args:
            hidden_size (int): Embedding dimension.
            num_concepts (int): Size of the node vocabulary.
            num_layers (int): Number of stacked GPS layers.
            num_relations (int): Number of edge types.
            compress_ratio (float): Fraction of nodes to keep.
            nhead (int): Number of transformer heads.
        """
        super(GraphEncoderGPS, self).__init__()
        self.embedding = nn.Embedding(num_concepts, hidden_size)
        self.gps_layers = nn.ModuleList([
            GPSLayer(hidden_size, num_relations, nhead=nhead)
            for _ in range(num_layers)
        ])
        self.node_score = nn.Linear(hidden_size, 1)
        self.sag_attn = nn.Linear(hidden_size, 1)
        self.compress_ratio = compress_ratio

    def forward(self, concept_ids, edge_index, edge_type):
        """
        Args:
            concept_ids (torch.Tensor): (batch_size, num_nodes)
            edge_index (torch.Tensor): (batch_size, 2, num_edges)
            edge_type (torch.Tensor): (batch_size, num_edges)
            
        Returns:
            pooled: (batch_size, 1, hidden_size) compressed graph representation.
            compressed_nodes_list: list of selected node representations per graph.
        """
        x = self.embedding(concept_ids)  # (batch_size, num_nodes, hidden_size)
        batch_size, num_nodes, _ = x.size()
        
        updated_nodes = []
        for i in range(batch_size):
            x_i = x[i]
            for layer in self.gps_layers:
                x_i = layer(x_i, edge_index[i], edge_type[i])
            updated_nodes.append(x_i)
        updated_nodes = torch.stack(updated_nodes, dim=0)  # (batch_size, num_nodes, hidden_size)
        
        pooled_list = []
        compressed_nodes_list = []
        for i in range(batch_size):
            x_i = updated_nodes[i]  # (num_nodes, hidden_size)
            scores = self.node_score(x_i).squeeze(-1)  # (num_nodes,)
            k = max(1, int(num_nodes * self.compress_ratio))
            topk_values, topk_indices = scores.topk(k, dim=0)
            x_top = x_i[topk_indices]  # (k, hidden_size)
            compressed_nodes_list.append(x_top)
            attn_scores = self.sag_attn(x_top).squeeze(-1)  # (k,)
            attn_weights = F.softmax(attn_scores, dim=0)  # (k,)
            pooled_i = torch.sum(x_top * attn_weights.unsqueeze(-1), dim=0, keepdim=True)  # (1, hidden_size)
            pooled_list.append(pooled_i)
        pooled = torch.stack(pooled_list, dim=0)  # (batch_size, 1, hidden_size)
        return pooled, compressed_nodes_list

# -----------------------------------------------------------
# Modified Graph-Aware BART Model Using GPS-based GraphEncoder
# -----------------------------------------------------------
class BartGraphAwareForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartGraphAwareForConditionalGeneration, self).__init__(config)
        # Use the new GPS-based graph encoder.
        self.graph_encoder = GraphEncoderGPS(
            hidden_size=config.d_model,
            num_concepts=len(concept2id),
            num_layers=2,
            num_relations=len(relation2id),
            compress_ratio=0.2,
            nhead=8
        )
        self.graph_fusion_layer = nn.Linear(config.d_model, config.d_model)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        # Graph-related arguments:
        concept_ids=None,
        edge_index=None,
        edge_type=None,
        **kwargs,
    ):
        kwargs.pop("num_items_in_batch", None)

        if (concept_ids is not None) and (edge_index is not None) and (edge_type is not None):
            device = input_ids.device if input_ids is not None else next(self.parameters()).device
            if not isinstance(concept_ids, torch.Tensor):
                concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=device)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            if not isinstance(edge_type, torch.Tensor):
                edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)
            
            pooled_graph, _ = self.graph_encoder(concept_ids, edge_index, edge_type)
            graph_features = self.graph_fusion_layer(pooled_graph)
        else:
            graph_features = None

        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            **kwargs,
        )
        lm_loss = outputs.loss
        lm_logits = outputs.logits

        if (labels is not None) and (lm_loss is not None) and (graph_features is not None):
            dummy_target = torch.zeros_like(graph_features)
            aux_loss = ((graph_features - dummy_target) ** 2).mean()
            total_loss = lm_loss + 0.1 * aux_loss
        else:
            total_loss = lm_loss

        return Seq2SeqLMOutput(
            loss=total_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )

# -----------------------------------------------------------
# Data Collator Updated for New Graph Fields
# -----------------------------------------------------------
class GraphAwareDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        graph_keys = ["concept_ids", "edge_index", "edge_type"]
        graph_features = {}
        for key in graph_keys:
            if key in features[0]:
                graph_features[key] = [f.pop(key) for f in features]

        batch = super().__call__(features)

        if "concept_ids" in graph_features:
            concept_ids_list = [
                f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.long)
                for f in graph_features["concept_ids"]
            ]
            batch["concept_ids"] = nn.utils.rnn.pad_sequence(
                concept_ids_list, batch_first=True, padding_value=0
            )
        if "edge_index" in graph_features:
            edge_index_list = [
                f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.long)
                for f in graph_features["edge_index"]
            ]
            max_edges = max([e.size(1) for e in edge_index_list])
            padded_edge_index = []
            for e in edge_index_list:
                pad_amt = max_edges - e.size(1)
                if pad_amt > 0:
                    pad_tensor = torch.zeros((2, pad_amt), dtype=e.dtype)
                    padded_edge_index.append(torch.cat([e, pad_tensor], dim=1))
                else:
                    padded_edge_index.append(e)
            batch["edge_index"] = torch.stack(padded_edge_index)
        if "edge_type" in graph_features:
            edge_type_list = [
                f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.long)
                for f in graph_features["edge_type"]
            ]
            max_edges = max([e.size(0) for e in edge_type_list])
            padded_edge_type = []
            for e in edge_type_list:
                pad_amt = max_edges - e.size(0)
                if pad_amt > 0:
                    pad_tensor = torch.zeros((pad_amt,), dtype=e.dtype)
                    padded_edge_type.append(torch.cat([e, pad_tensor], dim=0))
                else:
                    padded_edge_type.append(e)
            batch["edge_type"] = torch.stack(padded_edge_type)
        return batch

# -----------------------------------------------------------
# KG-Aware Trainer Creation Function (with Dataset Caching)
# -----------------------------------------------------------
def get_KG_trainer(
    source_path: str,
    target_path: str,
    model_name: str = "facebook/bart-base",
    output_dir: str = "KG_finetuned_out",
    max_len: int = 128,
    epochs: int = 3,
    train_batch_size: int = 60,
):
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed_dataset")

    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:500]
        targets = [line.strip() for line in f_tgt][:500]
    raw_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]

    if os.path.exists(preprocessed_path):
        print("Loading preprocessed dataset from disk...")
        train_dataset = load_from_disk(preprocessed_path)
    else:
        print("Preprocessing dataset...")
        train_dataset = Dataset.from_list(raw_data)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            model_inputs = tokenizer(examples["source"], max_length=max_len, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["target"], max_length=max_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            graph_info = get_graph_info(examples["source"])
            model_inputs.update(graph_info)
            return model_inputs

        train_dataset = train_dataset.map(preprocess_function, batched=False, desc="Preprocessing data")
        print("Saving preprocessed dataset to disk...")
        train_dataset.save_to_disk(preprocessed_path)

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartGraphAwareForConditionalGeneration.from_pretrained(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    data_collator = GraphAwareDataCollator(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        save_steps=15,
        save_total_limit=3,
        logging_steps=15,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
