import os
import torch
import torch.nn as nn
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
)

# Initialize KG resources.
load_resources()
load_cpnet()
load_total_concepts("data/eg")


# -----------------------------------------------------------
# Define the KG-based graph information function.
# -----------------------------------------------------------
def get_graph_info(text):
    """
    Uses KG_utils to extract graph information for the given text.
    - Computes a subgraph from the input text using BFS.
    - Extracts concept IDs from the nodes in the subgraph.
    - Pads or truncates the resulting list to a fixed length (here, 5).
    
    Returns:
        dict: A dictionary with key "concept_ids" containing a tensor of integers.
    """
    subgraph = get_subgraph_for_message(text)
    concept_ids = []
    for node in subgraph.nodes():
        if node in concept2id:
            concept_ids.append(concept2id[node])
    fixed_length = 5
    if len(concept_ids) < fixed_length:
        concept_ids = concept_ids + [0] * (fixed_length - len(concept_ids))
    else:
        concept_ids = concept_ids[:fixed_length]
    # Convert to tensor immediately so that the cached dataset contains tensors.
    return {"concept_ids": torch.tensor(concept_ids, dtype=torch.long)}


# -----------------------------------------------------------
# Custom Graph Encoder
# -----------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, hidden_size, num_concepts):
        """
        A simple graph encoder that looks up embeddings for concept IDs.
        
        Args:
            hidden_size (int): The embedding dimension.
            num_concepts (int): The total number of concepts in your KG vocabulary.
        """
        super(GraphEncoder, self).__init__()
        self.embedding = nn.Embedding(num_concepts, hidden_size)

    def forward(self, concept_ids):
        """
        Args:
            concept_ids: Tensor of shape (batch_size, num_concepts_per_example)
        Returns:
            pooled: A pooled representation of shape (batch_size, 1, hidden_size)
            embedded: All concept embeddings of shape (batch_size, num_concepts_per_example, hidden_size)
        """
        embedded = self.embedding(concept_ids)  # shape: (batch_size, num_concepts, hidden_size)
        pooled = embedded.mean(dim=1, keepdim=True)  # simple mean pooling
        return pooled, embedded


# -----------------------------------------------------------
# Custom Graph-Aware BART Model
# -----------------------------------------------------------
class BartGraphAwareForConditionalGeneration(BartForConditionalGeneration):
    def __init__(self, config: BartConfig):
        super(BartGraphAwareForConditionalGeneration, self).__init__(config)
        # Use the actual size of the concept vocabulary.
        self.graph_encoder = GraphEncoder(hidden_size=config.d_model, num_concepts=len(concept2id))
        # An optional fusion layer to transform the pooled graph representation.
        self.graph_fusion_layer = nn.Linear(config.d_model, config.d_model)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        labels=None,
        # New graph-related argument:
        concept_ids=None,
        **kwargs,
    ):
        # Remove unwanted kwargs that might be passed by Trainer.
        kwargs.pop("num_items_in_batch", None)

        if concept_ids is not None:
            # Ensure concept_ids is on the same device as input_ids.
            if not torch.is_tensor(concept_ids):
                concept_ids = torch.tensor(concept_ids, dtype=torch.long, device=input_ids.device)
            pooled_graph, _ = self.graph_encoder(concept_ids)
            # Transform the pooled representation.
            graph_features = self.graph_fusion_layer(pooled_graph)  # shape: (batch_size, 1, d_model)
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

        # Optionally add an auxiliary loss from the graph features.
        if (labels is not None) and (lm_loss is not None) and (graph_features is not None):
            dummy_target = torch.zeros_like(graph_features)
            aux_loss = ((graph_features - dummy_target) ** 2).mean()
            total_loss = lm_loss + 0.1 * aux_loss
        else:
            total_loss = lm_loss  # When labels is None (e.g., during generation).

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
# Custom Data Collator for Graph Fields (Vectorized and Robust)
# -----------------------------------------------------------
class GraphAwareDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features):
        # Use the parent collator to batch standard fields (e.g., input_ids, labels)
        batch = super().__call__(features)
        if "concept_ids" in features[0]:
            # Ensure each concept_ids is a tensor.
            concept_ids_list = []
            for f in features:
                cid = f["concept_ids"]
                if not torch.is_tensor(cid):
                    cid = torch.tensor(cid, dtype=torch.long)
                concept_ids_list.append(cid)
            # Stack and perform a single device transfer.
            batch["concept_ids"] = torch.stack(concept_ids_list).to(batch["input_ids"].device)
        return batch


# -----------------------------------------------------------
# KG-Aware Trainer Creation Function (with dataset caching)
# -----------------------------------------------------------
def get_KG_trainer(
    source_path: str,
    target_path: str,
    model_name: str = "facebook/bart-base",
    output_dir: str = "KG_finetuned_out",
    max_len: int = 128,
    epochs: int = 3,
    train_batch_size: int = 60
):
    """
    Returns a Hugging Face Trainer instance for fine-tuning a graph-aware BART model.
    This function first checks for a saved preprocessed dataset; if found, it loads it,
    otherwise it preprocesses the raw data and saves the result for future uses.
    """
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed_dataset")

    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:1000]
        targets = [line.strip() for line in f_tgt][:1000]
    raw_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]
    
    if os.path.exists(preprocessed_path):
        print("Loading preprocessed dataset from disk...")
        train_dataset = load_from_disk(preprocessed_path)
    else:
        print("Preprocessing dataset...")
        train_dataset = Dataset.from_list(raw_data)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
        def preprocess_function(example):
            # Process one example at a time.
            model_inputs = tokenizer(example["source"], max_length=max_len, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(example["target"], max_length=max_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            # Compute and add graph info (concept_ids are precomputed as tensors)
            model_inputs.update(get_graph_info(example["source"]))
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
        save_steps=500,
        save_total_limit=3,
        logging_steps=100,
        eval_strategy="no",
        fp16=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer
