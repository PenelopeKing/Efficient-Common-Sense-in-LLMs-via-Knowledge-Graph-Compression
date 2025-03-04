import os
import re
import gc

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
    TrainerCallback,
)
from transformers.modeling_outputs import Seq2SeqLMOutput
from datasets import Dataset, load_from_disk

# Import torch_geometric modules.
from torch_geometric.nn import RGCNConv

# -----------------------------------------------------------
# EmptyCacheCallback to mitigate gradual slowdown
# -----------------------------------------------------------
class EmptyCacheCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        return control

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
    id2concept,  # Ensure id2concept is defined in KG_utils.
)

# Initialize KG resources.
load_resources()
load_cpnet()
load_total_concepts("data/eg")

# -----------------------------------------------------------
# Full Graph Extraction (Preprocessing)
# -----------------------------------------------------------
def get_graph_info(text):
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
    return {"concept_ids": concept_ids, "edge_index": edge_index, "edge_type": edge_type}

# -----------------------------------------------------------
# Batched RGCN Layer using torch_geometric's RGCNConv
# -----------------------------------------------------------
class GraphConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GraphConvolutionalLayer, self).__init__()
        self.rgcn = RGCNConv(in_channels, out_channels, num_relations)
    
    def forward(self, node_features, edge_index, edge_type):
        out = self.rgcn(node_features, edge_index, edge_type)
        return F.relu(out)

# -----------------------------------------------------------
# Helper: Batched RGCN Forward
# -----------------------------------------------------------
def batch_rgcn(rgcn_layer, node_features_batch, edge_index_batch, edge_type_batch):
    """
    node_features_batch: (B, N, C)
    edge_index_batch: (B, 2, E)
    edge_type_batch: (B, E)
    """
    B, N, C = node_features_batch.shape

    # Make contiguous before reshaping.
    x = node_features_batch.contiguous().view(B * N, C)

    edge_index_list = []
    edge_type_list = []
    for i in range(B):
        ei = edge_index_batch[i]
        ei = ei + i * N
        edge_index_list.append(ei)
        edge_type_list.append(edge_type_batch[i])
    edge_index_combined = torch.cat(edge_index_list, dim=1)  # (2, total_edges)
    edge_type_combined = torch.cat(edge_type_list, dim=0)    # (total_edges,)

    out = rgcn_layer(x, edge_index_combined, edge_type_combined)
    out = out.view(B, N, -1)  # (B, N, out_channels)
    return out

# -----------------------------------------------------------
# Batched Graph Scoring Layer
# -----------------------------------------------------------
class GraphScoringLayer(nn.Module):
    def __init__(self, in_channels, num_relations):
        super(GraphScoringLayer, self).__init__()
        self.gcn = GraphConvolutionalLayer(in_channels, in_channels, num_relations)
        self.linear = nn.Linear(in_channels, 1)
    
    def forward(self, node_features, edge_index, edge_type):
        updated = batch_rgcn(self.gcn, node_features, edge_index, edge_type)
        scores = self.linear(updated).squeeze(-1)
        return scores

# -----------------------------------------------------------
# Learned SAG Pooling Module (Batched Vectorized Version)
# -----------------------------------------------------------
class LearnedSAGPooling(nn.Module):
    def __init__(self, in_channels, compress_ratio=0.5, num_relations=None):
        super(LearnedSAGPooling, self).__init__()
        self.compress_ratio = compress_ratio
        self.scoring = GraphScoringLayer(in_channels, num_relations)
        self.attn = nn.Linear(in_channels, 1)
    
    def forward(self, node_features, edge_index, edge_type):
        B, N, C = node_features.size()
        scores = self.scoring(node_features, edge_index, edge_type)  # (B, N)
        k = max(1, int(N * self.compress_ratio))
        _, topk_indices = torch.topk(scores, k, dim=1)
        topk_indices_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, C)  # (B, k, C)
        x_top = torch.gather(node_features, 1, topk_indices_expanded)         # (B, k, C)
        attn_scores = self.attn(x_top).squeeze(-1)                           # (B, k)
        attn_weights = F.softmax(attn_scores, dim=1).unsqueeze(-1)           # (B, k, 1)
        pooled = torch.sum(x_top * attn_weights, dim=1, keepdim=True)        # (B, 1, C)
        return pooled, x_top, topk_indices

# -----------------------------------------------------------
# Graph Encoder with 1 RGCN -> 1 Transformer -> 1 RGCN -> 1 Transformer -> SAG
# -----------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, hidden_size, num_concepts, compress_ratio=0.5,
                 num_relations=None, transformer_nhead=8, transformer_num_layers=1):
        super(GraphEncoder, self).__init__()
        if num_relations is None:
            num_relations = len(relation2id)

        self.embedding = nn.Embedding(num_concepts, hidden_size)

        # RGCN layer 1
        self.rgcn1 = GraphConvolutionalLayer(hidden_size, hidden_size, num_relations)
        # Transformer layer 1
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=transformer_nhead)
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_num_layers)

        # RGCN layer 2
        self.rgcn2 = GraphConvolutionalLayer(hidden_size, hidden_size, num_relations)
        # Transformer layer 2
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=transformer_nhead)
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_num_layers)

        self.sag_pool = LearnedSAGPooling(hidden_size, compress_ratio=compress_ratio, num_relations=num_relations)
    
    def forward(self, concept_ids, edge_index, edge_type):
        # (B, N)
        x = self.embedding(concept_ids)  # (B, N, hidden_size)

        # RGCN layer 1 + residual
        x0 = x
        x1_local = batch_rgcn(self.rgcn1, x0, edge_index, edge_type)  # (B, N, hidden_size)
        x1 = x1_local + x0  # residual

        # Transformer layer 1 + residual
        # Transformer expects shape (N, B, hidden_size)
        x1_t = x1.permute(1, 0, 2).contiguous()   # (N, B, hidden_size)
        x1_global = self.transformer1(x1_t)       # (N, B, hidden_size)
        x1_global = x1_global.permute(1, 0, 2).contiguous()  # (B, N, hidden_size)
        x2 = x1_global + x1  # residual

        # RGCN layer 2 + residual
        x2_local = batch_rgcn(self.rgcn2, x2, edge_index, edge_type)  # (B, N, hidden_size)
        x3 = x2_local + x2

        # Transformer layer 2 + residual
        x3_t = x3.permute(1, 0, 2).contiguous()   # (N, B, hidden_size)
        x3_global = self.transformer2(x3_t)       # (N, B, hidden_size)
        x3_global = x3_global.permute(1, 0, 2).contiguous()  # (B, N, hidden_size)
        x4 = x3_global + x3

        # SAG pooling
        pooled, _, topk_indices = self.sag_pool(x4, edge_index, edge_type)
        concept_ids_exp = concept_ids.unsqueeze(-1)  # (B, N, 1)
        topk_concept_ids = torch.gather(concept_ids_exp, 1, topk_indices.unsqueeze(-1)).squeeze(-1)
        return pooled, topk_concept_ids

# -----------------------------------------------------------
# Custom Graph-Aware BART Model (Insert Concepts as Text after <KG>)
# -----------------------------------------------------------
class BartGraphAwareForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig, tokenizer: BartTokenizer):
        super(BartGraphAwareForConditionalGeneration, self).__init__(config)
        self.graph_encoder = GraphEncoder(
            hidden_size=config.d_model,
            num_concepts=len(concept2id),
            compress_ratio=0.2,
            num_relations=len(relation2id),
            transformer_nhead=8,
            transformer_num_layers=1,
        )
        self.tokenizer = tokenizer
        self.post_init()
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        concept_ids=None,
        edge_index=None,
        edge_type=None,
        **kwargs,
    ):
        device = input_ids.device if input_ids is not None else next(self.parameters()).device

        # If input_ids is None (happens during generation w/ cached encoder outputs), call parent
        if input_ids is None:
            return super().forward(
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
            )

        if (concept_ids is not None) and (edge_index is not None) and (edge_type is not None):
            if not isinstance(concept_ids, torch.Tensor):
                concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=device)
            if not isinstance(edge_index, torch.Tensor):
                edge_index = torch.tensor(edge_index, dtype=torch.long, device=device)
            if not isinstance(edge_type, torch.Tensor):
                edge_type = torch.tensor(edge_type, dtype=torch.long, device=device)

            if concept_ids.dim() == 1:
                concept_ids = concept_ids.unsqueeze(0)
            if edge_index.dim() == 2:
                edge_index = edge_index.unsqueeze(0)
            if edge_type.dim() == 1:
                edge_type = edge_type.unsqueeze(0)

            # run the extended graph encoder
            _, topk_concept_ids = self.graph_encoder(concept_ids, edge_index, edge_type)

            # convert top-k concept IDs to text
            concept_texts = []
            for instance in topk_concept_ids:
                tokens = [id2concept.get(int(id.item()), "") for id in instance]
                text = " ".join(tokens).strip()
                concept_texts.append("<KG> " + text)

            concept_tokenized = self.tokenizer(concept_texts, return_tensors="pt", padding=True, truncation=False)
            concept_tokenized = {k: v.to(device) for k, v in concept_tokenized.items()}

            # embed the concept text
            graph_token_embeds = self.model.encoder.embed_tokens(concept_tokenized["input_ids"])
            text_token_embeds = self.model.encoder.embed_tokens(input_ids)

            B, k_len, _ = graph_token_embeds.size()
            graph_attention_mask = torch.ones((B, k_len), dtype=attention_mask.dtype, device=attention_mask.device)

            inputs_embeds = torch.cat([graph_token_embeds, text_token_embeds], dim=1)
            extended_attention_mask = torch.cat([graph_attention_mask, attention_mask], dim=1)

            encoder_outputs = self.model.encoder(
                inputs_embeds=inputs_embeds,
                attention_mask=extended_attention_mask,
                return_dict=True,
            )
            outputs = self.model.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=extended_attention_mask,
                return_dict=True,
            )
            lm_logits = self.lm_head(outputs.last_hidden_state)
            loss = None
            if labels is not None:
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

            return Seq2SeqLMOutput(
                loss=loss,
                logits=lm_logits,
                past_key_values=outputs.past_key_values if hasattr(outputs, "past_key_values") else None,
                decoder_hidden_states=outputs.hidden_states if hasattr(outputs, "hidden_states") else None,
                decoder_attentions=outputs.attentions if hasattr(outputs, "attentions") else None,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states if hasattr(encoder_outputs, "hidden_states") else None,
                encoder_attentions=encoder_outputs.attentions if hasattr(encoder_outputs, "attentions") else None,
            )
        else:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                labels=labels,
                **kwargs,
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
def get_KG_transformer_trainer(
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

    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:num_points]
        targets = [line.strip() for line in f_tgt][:num_points]

    raw_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]
    if os.path.exists(preprocessed_path):
        train_dataset = load_from_disk(preprocessed_path).select(range(num_points))
    else:
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
        train_dataset.save_to_disk(preprocessed_path)

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartGraphAwareForConditionalGeneration.from_pretrained(model_name, tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_collator = GraphAwareDataCollator(tokenizer, model=model)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        fp16=True,
        save_steps=10000,
        save_total_limit=3,
        logging_steps=10,
        eval_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[EmptyCacheCallback()],
    )
    return trainer
