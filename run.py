import argparse
import json
import os
import sys
import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge import Rouge
from bleu import Bleu
from corpus_diversity import eval_entropy_distinct
import time
from tqdm import tqdm

import networkx as nx
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch.nn import Linear
import random
import torch.optim as optim

# Load model configurations
with open("data-params.json", "r") as f:
    config = json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dataset, model_name):
    """Train a model based on dataset and model type."""
    model_config = config["datasets"][dataset]["models"][model_name]
    data_path = config["datasets"][dataset]["data_path"]

    source_path = f"{data_path}/train.source"
    target_path = f"{data_path}/train.target"
    test_source_path = f"{data_path}/test.source"
    test_target_path = f"{data_path}/test.target"
    
    def get_trainer(model_type):
        if model_type == "basic":
            from basic_trainer import get_basic_trainer
            return get_basic_trainer
        elif model_type == "kg_no_comp":
            from KG_trainer_no_comp import generate_explanation_no_comp
            from KG_trainer_no_comp import get_KG_trainer
            return get_KG_trainer
        elif model_type == "rgcn_comp":
            from KG_trainer_w_comp import get_KG_RGCN_trainer
            return get_KG_RGCN_trainer
        elif model_type == "transformer_comp":
            from KG_trainer_w_transformer_no_dropout import get_KG_transformer_trainer
            return get_KG_transformer_trainer
        elif model_type == "transformer_dropout":
            from KG_trainer_w_transformer_dropout import get_KG_transformer_trainer
            return get_KG_transformer_trainer
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    trainer_function = get_trainer(model_name)

    trainer = trainer_function(
        source_path=source_path,
        target_path=target_path,
        # test_source_path=test_source_path,
        # test_target_path=test_target_path,
        model_name=model_config["model_name"],
        output_dir=model_config["output_dir"],
        max_len=model_config["max_len"],
        epochs=model_config["epochs"],
        train_batch_size=model_config["train_batch_size"],
        num_points=model_config["num_points"],
    )

    print(f"Training {model_name} on {dataset}...")
    trainer.train()
    print(f"Training complete for {model_name} on {dataset}. Model saved at {model_config['output_dir']}")

def evaluate_model(dataset, model_name):
    """Evaluate a trained model and store results."""
    model_config = config["datasets"][dataset]["models"][model_name]
    data_path = config["datasets"][dataset]["data_path"]
    model_dir = os.path.join(model_config["output_dir"], model_config["checkpoint"])

    print(f"Evaluating {model_name} on {dataset} from checkpoint {model_config['checkpoint']}...")

        # Load model based on type
    if model_name in ["rgcn_comp", "transformer_comp"]:  # Graph-aware models
        from KG_trainer_w_comp import get_graph_info, BartGraphAwareForConditionalGeneration
        tokenizer = BartTokenizer.from_pretrained(model_dir)
        model = BartGraphAwareForConditionalGeneration.from_pretrained(
            model_dir, tokenizer=tokenizer
        ).to(device)
    else:  # Standard BART model (no KG, no Comp)

        model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
        tokenizer = BartTokenizer.from_pretrained(model_dir)
    # # Load trained model and tokenizer
    # model = BartForConditionalGeneration.from_pretrained(model_dir).to(device)
    # tokenizer = BartTokenizer.from_pretrained(model_dir)

    with open(f"{data_path}/test.source", "r", encoding="utf-8") as f_src,          open(f"{data_path}/test.target", "r", encoding="utf-8") as f_tgt:
        test_inputs = [line.strip() for line in f_src]
        test_targets = [line.strip().split("\t") for line in f_tgt]

    bleu = Bleu()
    rouge = Rouge()
    
    total_metrics = {
        "self_bleu_3": 0, "self_bleu_4": 0, "distinct_2": 0,
        "entropy_4": 0, "bleu_4": 0, "rouge_l": 0
    }
    num_samples = len(test_inputs)

    def generate_explanation(model, tokenizer, sentence):
        """Generate explanations for input text."""
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        output_ids = model.generate(
            **inputs, max_length=60, num_beams=3, num_return_sequences=3, early_stopping=True
        )
        return [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    
    def generate_explanation_RGCN(model, tokenizer, sentence: str, max_length=60):
        # Tokenize the input text and move to device
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Ensure inputs are on device
        
        # Extract graph info from the sentence and move graph tensors to device
        graph_info = get_graph_info(sentence)
        concept_ids = graph_info["concept_ids"].to(device).unsqueeze(0) if graph_info["concept_ids"].numel() > 0 else None
        edge_index = graph_info["edge_index"].to(device).unsqueeze(0) if graph_info["edge_index"].numel() > 0 else None
        edge_type = graph_info["edge_type"].to(device).unsqueeze(0) if graph_info["edge_type"].numel() > 0 else None

        # Generate outputs using beam search; passing the graph fields as additional arguments
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            concept_ids=concept_ids,
            edge_index=edge_index,
            edge_type=edge_type,
            max_length=max_length,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True,
        )

        # Decode the generated token sequences
        explanations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        return explanations
    # Select correct generation function based on model type
    if model_name == "basic":
        generation_function = generate_explanation
    elif model_name == "kg_no_comp":
        generation_function = generate_explanation_no_comp
    elif model_name in ["rgcn_comp", "transformer_comp"]:
        generation_function = generate_explanation_RGCN
    else:
        raise ValueError(f"Unknown model type: {model_name}")
    start_time = time.time()

    for input_text, ground_truths in tqdm(zip(test_inputs, test_targets), total=num_samples, desc="Evaluating", unit="sample"):
        
        explanations = generation_function(model, tokenizer, input_text)
        
        diversity_scores = eval_entropy_distinct(explanations)
        distinct_2 = diversity_scores["distinct_2"] * 100
        entropy_4 = diversity_scores["entropy_4"]

        self_bleu_3 = 0
        self_bleu_4 = 0
        for i, e in enumerate(explanations):
            res = {input_text: [e]}
            gts = {input_text: explanations[:i] + explanations[i+1:]}
            scores = bleu.compute_score(gts=gts, res=res)[0]
            self_bleu_3 += scores[2]
            self_bleu_4 += scores[3]

        self_bleu_3 = (self_bleu_3 / len(explanations)) * 100
        self_bleu_4 = (self_bleu_4 / len(explanations)) * 100

        bleu_4 = max([bleu.compute_score(gts={input_text: ground_truths}, res={input_text: [e]})[0][3] for e in explanations]) * 100
        rouge_l = max([rouge.compute_score(gts={input_text: ground_truths}, res={input_text: [e]})[0] for e in explanations]) * 100

        for key, value in zip(total_metrics.keys(), [self_bleu_3, self_bleu_4, distinct_2, entropy_4, bleu_4, rouge_l]):
            total_metrics[key] += value

    end_time = time.time()
    elapsed_time = end_time - start_time

    average_metrics = {key: total_metrics[key] / num_samples for key in total_metrics}

    results_path = "results.json"
    if os.path.exists(results_path):
        with open(results_path, "r") as f:
            results = json.load(f)
    else:
        results = {}

    if dataset not in results:
        results[dataset] = {}
    results[dataset][model_name] = {
        "metrics": average_metrics,
        "elapsed_time": elapsed_time,
        "checkpoint": model_config["checkpoint"]
    }

    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Evaluation complete for {model_name} on {dataset}. Results saved in {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or evaluate models")
    parser.add_argument("task", choices=["train", "evaluate"], help="Task to perform")
    parser.add_argument("dataset", choices=config["datasets"].keys(), help="Dataset to use (anlg or comve)")
    parser.add_argument("model", choices=config["datasets"]["anlg"]["models"].keys(), help="Model type")

    args = parser.parse_args()

    if args.task == "train":
        train_model(args.dataset, args.model)
    elif args.task == "evaluate":
        evaluate_model(args.dataset, args.model)
