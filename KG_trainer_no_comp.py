import os
import torch
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
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
    id2concept,
)

# Initialize KG resources.
load_resources()
load_cpnet()
load_total_concepts("data/eg")


# -----------------------------------------------------------
# Define a function to get KG-derived text using all concepts.
# -----------------------------------------------------------
def get_graph_text(text):
    """
    Extracts graph information for the given text by:
      - Computing a subgraph using BFS.
      - Extracting all concept IDs available.
      - Converting each concept ID into text using id2concept.
      - Joining the concept texts with a space.
    
    Returns:
        A string containing the KG information.
    """
    subgraph = get_subgraph_for_message(text)
    concept_ids = []
    for node in subgraph.nodes():
        if node in concept2id:
            concept_ids.append(concept2id[node])
    # Convert each concept ID to its corresponding text.
    concepts = [id2concept.get(cid, "") for cid in concept_ids]
    # Filter out any empty strings.
    concepts = [c for c in concepts if c]
    # Join using a space between items.
    graph_text = " ".join(concepts)
    return graph_text


# -----------------------------------------------------------
# KG-Aware Trainer Creation Function (Prompt Injection Version)
# -----------------------------------------------------------
def get_KG_trainer(
    source_path: str,
    target_path: str,
    model_name: str = "facebook/bart-base",
    output_dir: str = "KG_finetuned_out",
    max_len: int = 1024,
    epochs: int = 3,
    train_batch_size: int = 60,
    num_points: int = 200,  # New parameter for datapoints count.
):
    os.makedirs(output_dir, exist_ok=True)
    preprocessed_path = os.path.join(output_dir, "preprocessed_dataset")

    with open(source_path, "r", encoding="utf-8") as f_src, open(target_path, "r", encoding="utf-8") as f_tgt:
        sources = [line.strip() for line in f_src][:num_points]
        targets = [line.strip() for line in f_tgt][:num_points]
    raw_data = [{"source": s, "target": t} for s, t in zip(sources, targets)]
    
    if os.path.exists(preprocessed_path):
        print("Loading preprocessed dataset from disk...")
        train_dataset = load_from_disk(preprocessed_path).select(range(num_points))
    else:
        print("Preprocessing dataset...")
        train_dataset = Dataset.from_list(raw_data)
        tokenizer = BartTokenizer.from_pretrained(model_name)
        
        def preprocess_function(example):
            graph_text = get_graph_text(example["source"])
            new_source = example["source"] + " <KG> " + graph_text if graph_text else example["source"]
            model_inputs = tokenizer(new_source, max_length=max_len, truncation=True)
            with tokenizer.as_target_tokenizer():
                labels = tokenizer(example["target"], max_length=max_len, truncation=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        train_dataset = train_dataset.map(preprocess_function, batched=False, desc="Preprocessing data")
        print("Saving preprocessed dataset to disk...")
        train_dataset.save_to_disk(preprocessed_path)

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        save_steps=5000,
        save_total_limit=5,
        logging_steps=500,
        eval_strategy="no",
        fp16=True
    )

    sample = train_dataset[0]
    print("Decoded input:", tokenizer.decode(sample['input_ids']))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    return trainer

def generate_explanation_no_comp(model, tokenizer, device, sentence: str) -> list:
    # Inject KG-derived text into the input sentence.
    graph_text = get_graph_text(sentence)
    if graph_text:
        sentence = sentence + " <KG> " + graph_text

    # Tokenize the modified input text
    inputs = tokenizer(sentence, return_tensors="pt", max_length=256, truncation=True).to(device)
    
    # Generate output using beam search:
    output_ids = model.generate(
        **inputs,
        max_length=60,
        num_beams=3,
        num_return_sequences=3,
        early_stopping=True
    )
    
    # Decode each of the generated token sequences
    explanations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return explanations

