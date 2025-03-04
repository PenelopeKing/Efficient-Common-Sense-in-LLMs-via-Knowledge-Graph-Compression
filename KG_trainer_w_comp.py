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
    
    It returns:
      - All node concept IDs (without truncation or learned selection).
      - All edge information (as raw indices with respect to the full node list).
      
    Returns:
        dict: Contains:
            - "concept_ids": a tensor of shape (num_nodes,) with node IDs.
            - "edge_index": a tensor of shape (2, num_edges) with edge indices.
            - "edge_type": a tensor of shape (num_edges,) with edge type IDs.
    """
    subgraph = get_subgraph_for_message(text)
    
    # --- Process nodes (full list) ---
    nodes = list(subgraph.nodes())
    concept_ids = [concept2id.get(node, 0) for node in nodes]
    # Build a mapping from node to index (for all nodes)
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # --- Process edges (full set) ---
    edge_list = []
    edge_types = []
    for u, v, data in subgraph.edges(data=True):
        if u in node_to_idx and v in node_to_idx:
            edge_list.append([node_to_idx[u], node_to_idx[v]])
            relation = data.get("relation", "default")
            edge_types.append(relation2id.get(relation, 0))
    
    # Convert to tensors (note: the lengths will vary between examples)
    concept_ids = torch.tensor(concept_ids, dtype=torch.long)
    if len(edge_list) > 0:
        edge_index = torch.tensor(edge_list, dtype=torch.long).transpose(0, 1)  # shape: (2, num_edges)
        edge_type = torch.tensor(edge_types, dtype=torch.long)  # shape: (num_edges,)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_type = torch.zeros((0,), dtype=torch.long)
    
    return {
        "concept_ids": concept_ids,
        "edge_index": edge_index,
        "edge_type": edge_type,
    }


# -----------------------------------------------------------
# Simple GCN Layer Incorporating Edge Classes
# -----------------------------------------------------------
class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A simple GCN layer that aggregates messages from neighboring nodes,
        incorporating edge class information.
        
        Args:
            in_channels (int): Dimension of input node features.
            out_channels (int): Dimension of output node features.
        """
        super(GraphConvolutionalLayer, self).__init__()
        self.linear = nn.Linear(in_channels, out_channels)

    def forward(self, node_features, edge_index, edge_type, edge_embedding):
        """
        Args:
            node_features (torch.Tensor): (num_nodes, in_channels)
            edge_index (torch.Tensor): (2, num_edges) with rows [source, target].
            edge_type (torch.Tensor): (num_edges,)
            edge_embedding (nn.Embedding): Embedding module for edge types.
            
        Returns:
            torch.Tensor: Updated node features of shape (num_nodes, out_channels)
        """
        # Transform node features.
        node_features_transformed = self.linear(node_features)
        aggregated = torch.zeros_like(node_features_transformed)
        num_edges = edge_index.size(1)
        # Loop over each edge (acceptable for small graphs).
        for j in range(num_edges):
            src = edge_index[0, j]
            tgt = edge_index[1, j]
            # Get the embedding for the edge class.
            e_emb = edge_embedding(edge_type[j])
            # Message: transformed source node feature + edge embedding.
            message = node_features_transformed[src] + e_emb
            aggregated[tgt] += message
        # Include self–loop.
        aggregated += node_features_transformed
        return F.relu(aggregated)


# -----------------------------------------------------------
# Graph Scoring Layer (GCN-based) for Top-k Selection
# -----------------------------------------------------------
class GraphScoringLayer(nn.Module):
    def __init__(self, in_channels):
        """
        Computes a scalar score for each node using a small GCN followed by a linear layer.
        
        Args:
            in_channels (int): Dimension of node features.
        """
        super(GraphScoringLayer, self).__init__()
        self.gcn = GraphConvolutionalLayer(in_channels, in_channels)
        self.linear = nn.Linear(in_channels, 1)

    def forward(self, node_features, edge_index, edge_type, edge_embedding):
        """
        Args:
            node_features (torch.Tensor): (num_nodes, in_channels)
            edge_index (torch.Tensor): (2, num_edges)
            edge_type (torch.Tensor): (num_edges,)
            edge_embedding (nn.Embedding): Embedding module for edge types.
            
        Returns:
            torch.Tensor: Scores of shape (num_nodes,)
        """
        updated = self.gcn(node_features, edge_index, edge_type, edge_embedding)
        scores = self.linear(updated).squeeze(-1)
        return scores


# -----------------------------------------------------------
# Learned SAG Pooling Module (with Top-k Selection Inside)
# -----------------------------------------------------------
class LearnedSAGPooling(nn.Module):
    def __init__(self, in_channels, compress_ratio=0.5):
        """
        Self–attention pooling that first performs a learned top–k selection based on a GCN–based scoring layer,
        then applies attention pooling over the selected nodes.
        
        Args:
            in_channels (int): Dimension of node features.
            compress_ratio (float): Fraction of nodes to keep (e.g. 0.5 for top 50%).
        """
        super(LearnedSAGPooling, self).__init__()
        self.compress_ratio = compress_ratio
        self.scoring = GraphScoringLayer(in_channels)
        self.attn = nn.Linear(in_channels, 1)

    def forward(self, node_features, edge_index, edge_type, edge_embedding):
        """
        Args:
            node_features (torch.Tensor): (batch_size, num_nodes, in_channels)
            edge_index (torch.Tensor): (batch_size, 2, num_edges)
            edge_type (torch.Tensor): (batch_size, num_edges)
            edge_embedding (nn.Embedding): The edge embedding module.
            
        Returns:
            pooled (torch.Tensor): (batch_size, 1, in_channels) compressed graph representation.
            compressed_nodes_list (list): List (length=batch_size) of selected node features 
                                          (each of shape (k, in_channels)).
        """
        batch_size, num_nodes, in_channels = node_features.size()
        pooled_list = []
        compressed_nodes_list = []
        for i in range(batch_size):
            x = node_features[i]  # (num_nodes, in_channels)
            e_idx = edge_index[i]  # (2, num_edges)
            e_type = edge_type[i]  # (num_edges,)
            # Compute node scores using the GCN–based scoring layer.
            scores = self.scoring(x, e_idx, e_type, edge_embedding)  # (num_nodes,)
            # Determine k = max(1, floor(num_nodes * compress_ratio)).
            k = max(1, int(num_nodes * self.compress_ratio))
            # Select the top-k nodes.
            topk_values, topk_indices = scores.topk(k, dim=0)
            x_top = x[topk_indices]  # (k, in_channels)
            compressed_nodes_list.append(x_top)
            # Compute attention weights over the selected nodes.
            attn_scores = self.attn(x_top).squeeze(-1)  # (k,)
            attn_weights = F.softmax(attn_scores, dim=0)  # (k,)
            pooled_i = torch.sum(x_top * attn_weights.unsqueeze(-1), dim=0, keepdim=True)  # (1, in_channels)
            pooled_list.append(pooled_i)
        pooled = torch.stack(pooled_list, dim=0)  # (batch_size, 1, in_channels)
        return pooled, compressed_nodes_list


# -----------------------------------------------------------
# Graph Encoder with Learned Compression (Top-k inside SAG Pooling)
# -----------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, hidden_size, num_concepts, compress_ratio=0.5):
        """
        Encodes a full KG subgraph by:
          - Looking up node embeddings.
          - Updating node representations via a GCN that incorporates edge classes.
          - Compressing the updated node representations using a learned top–k selection
            (inside the pooling module) followed by SAG pooling.
        
        Args:
            hidden_size (int): Embedding dimension.
            num_concepts (int): Size of the node vocabulary.
            compress_ratio (float): Fraction of nodes to keep.
        """
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(num_concepts, hidden_size)
        # Instead of passing in num_edge_classes, we compute it as the length of relation2id.
        self.edge_embedding = nn.Embedding(len(relation2id), hidden_size)
        self.gcn = GraphConvolutionalLayer(hidden_size, hidden_size)
        self.sag_pool = LearnedSAGPooling(hidden_size, compress_ratio=compress_ratio)

    def forward(self, concept_ids, edge_index, edge_type):
        """
        Args:
            concept_ids (torch.Tensor): (batch_size, num_nodes)
            edge_index (torch.Tensor): (batch_size, 2, num_edges)
            edge_type (torch.Tensor): (batch_size, num_edges)
            
        Returns:
            pooled: (batch_size, 1, hidden_size) – the compressed graph representation.
            compressed_nodes: list of length batch_size containing the selected node representations.
        """
        # Look up node embeddings.
        x = self.embedding(concept_ids)  # (batch_size, num_nodes, hidden_size)
        batch_size, num_nodes, _ = x.size()
        
        # Update node features via a GCN (process each graph separately).
        updated_nodes = []
        for i in range(batch_size):
            node_feats = x[i]  # (num_nodes, hidden_size)
            sample_edge_index = edge_index[i]  # (2, num_edges)
            sample_edge_type = edge_type[i]    # (num_edges)
            updated = self.gcn(node_feats, sample_edge_index, sample_edge_type, self.edge_embedding)
            updated_nodes.append(updated)
        updated_nodes = torch.stack(updated_nodes, dim=0)  # (batch_size, num_nodes, hidden_size)
        
        # Use the Learned SAG Pooling (which applies top-k selection inside) to compress the graph.
        pooled, compressed_nodes = self.sag_pool(updated_nodes, edge_index, edge_type, self.edge_embedding)
        return pooled, compressed_nodes


# -----------------------------------------------------------
# Custom Graph-Aware BART Model
# -----------------------------------------------------------
class BartGraphAwareForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartGraphAwareForConditionalGeneration, self).__init__(config)
        self.graph_encoder = GraphEncoder(
            hidden_size=config.d_model,
            num_concepts=len(concept2id),
            compress_ratio=0.5, # Experiment with this?
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
            # Ensure tensors are on the correct device.
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

        # Auxillarly loss (TODO look into this more)
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
        # Remove graph-related keys so that the tokenizer's pad() function does not try to process them.
        graph_keys = ["concept_ids", "edge_index", "edge_type"]
        graph_features = {}
        for key in graph_keys:
            if key in features[0]:
                graph_features[key] = [f.pop(key) for f in features]

        # Let the default collator pad the remaining keys.
        batch = super().__call__(features)

        # Now, pad the graph features manually and add them back to the batch.
        if "concept_ids" in graph_features:
            # Ensure each is a tensor.
            concept_ids_list = [
                f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.long)
                for f in graph_features["concept_ids"]
            ]
            batch["concept_ids"] = nn.utils.rnn.pad_sequence(
                concept_ids_list, batch_first=True, padding_value=0
            )
        if "edge_index" in graph_features:
            # Each edge_index is expected to be a tensor of shape (2, num_edges).
            edge_index_list = [
                f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=torch.long)
                for f in graph_features["edge_index"]
            ]
            # Pad along dimension 1 (number of edges).
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
            # Each edge_type is expected to be a tensor of shape (num_edges,).
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
    num_points: int = 200,
):
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed_dataset")

    # Read and limit to num_points datapoints.
    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:num_points]
        targets = [line.strip() for line in f_tgt][:num_points]
    raw_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]

    # Load preprocessed dataset if available; otherwise, preprocess the raw_data.
    if os.path.exists(preprocessed_path):
        print("Loading preprocessed dataset from disk...")
        train_dataset = load_from_disk(preprocessed_path).select(range(num_points))
    else:
        print("Preprocessing dataset...")
        train_dataset = Dataset.from_list(raw_data)
        tokenizer = BartTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            model_inputs = tokenizer(examples["source"], max_length=max_len, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(examples["target"], max_length=max_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            # Compute full graph info (nodes and edges) without learned compression.
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
        save_steps=10000,
        save_total_limit=3,
        logging_steps=5,
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
