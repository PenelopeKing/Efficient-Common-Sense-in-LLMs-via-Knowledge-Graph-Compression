import torch
from transformers import BartTokenizer, BartForConditionalGeneration
from datasets import Dataset
import bert_score
import os
import json
import sys
from KG_trainer_w_comp import get_graph_info, BartGraphAwareForConditionalGeneration
from KG_trainer_w_comp import get_KG_trainer as comp_trainer
from KG_trainer_no_comp import get_KG_trainer


from basic_trainer import get_basic_trainer
from evaluator import evaluate_model
import networkx as nx
import pickle
from torch_geometric.data import Data
from torch_geometric.nn import RGCNConv
import torch.nn.functional as F
from torch.nn import Linear
import random
import torch.optim as optim


def run_encoding():
     
    cpnet_path = "data/cpnet.graph"
    cpnet = None
    cpnet_simple = None

    # taken from quentin lol - changed a little
    def load_cpnet():
        global cpnet, cpnet_simple
        print("Loading cpnet...")
        with open(cpnet_path, "rb") as f:
            cpnet = pickle.load(f)
        print("Done")

        # Build an undirected version
        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data["weight"] if "weight" in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]["weight"] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        
        return cpnet_simple, cpnet.edges(data=True)

    # converting to pyG cause its directly usable for models like R-GCN
    def convert_to_pyg(cpnet):
        node_map = {node: i for i, node in enumerate(cpnet.nodes())}
        edge_index = []
        edge_attr = []
        rel_map = {}
        
        for u, v, data in cpnet.edges(data=True):
            u_idx, v_idx = node_map[u], node_map[v]
            edge_index.append([u_idx, v_idx])
            rel = data.get("rel", "generic")
            if rel not in rel_map:
                rel_map[rel] = len(rel_map)
            edge_attr.append(rel_map[rel])
        
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.long)
        
        return Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=len(node_map)), rel_map

    class RGCN(torch.nn.Module):
        def __init__(self, num_nodes, num_relations, in_channels, hidden_channels, out_channels):
            super(RGCN, self).__init__()
            self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations)
            self.conv2 = RGCNConv(hidden_channels, out_channels, num_relations)
            self.score_fn = Linear(out_channels * 2, 1)  # Scoring function for link prediction

        def forward(self, x, edge_index, edge_attr):
            x = self.conv1(x, edge_index, edge_attr)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_attr)
            return x
        
        def predict_link(self, x, edge_pairs):
            h1 = x[edge_pairs[:, 0]]
            h2 = x[edge_pairs[:, 1]]
            scores = self.score_fn(torch.cat([h1, h2], dim=1))
            return torch.sigmoid(scores).squeeze()

    # apparently the RGCN documentation said you should use negative sampling so this is what that does
    def generate_negative_edges(num_nodes, num_samples, existing_edges):
        neg_edges = set()
        while len(neg_edges) < num_samples:
            u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
            if (u, v) not in existing_edges and (v, u) not in existing_edges and u != v:
                neg_edges.add((u, v))
        return torch.tensor(list(neg_edges), dtype=torch.long)

    # train
    def train(model, data, x, epochs=100):
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        loss_fn = torch.nn.BCELoss()
        
        pos_edges = data.edge_index.t()
        neg_edges = generate_negative_edges(data.num_nodes, pos_edges.shape[0], set(map(tuple, pos_edges.tolist())))
        
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            embeddings = model(x, data.edge_index, data.edge_attr)
            pos_scores = model.predict_link(embeddings, pos_edges)
            neg_scores = model.predict_link(embeddings, neg_edges)
            
            labels = torch.cat([torch.ones(pos_scores.shape[0]), torch.zeros(neg_scores.shape[0])])
            scores = torch.cat([pos_scores, neg_scores])
            
            loss = loss_fn(scores, labels)
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")
    # Modify the load_cpnet function to return cpnet_simple
    def load_cpnet():
        global cpnet, cpnet_simple
        print("Loading cpnet...")
        with open(cpnet_path, "rb") as f:
            cpnet = pickle.load(f)
        print("Done")

        # Build an undirected version
        cpnet_simple = nx.Graph()
        for u, v, data in cpnet.edges(data=True):
            w = data["weight"] if "weight" in data else 1.0
            if cpnet_simple.has_edge(u, v):
                cpnet_simple[u][v]["weight"] += w
            else:
                cpnet_simple.add_edge(u, v, weight=w)
        
        return cpnet_simple, cpnet.edges(data=True)


    cpnet_simple, edge_types = load_cpnet()
    data, rel_map = convert_to_pyg(cpnet_simple)

    num_relations = len(rel_map)
    num_nodes = data.num_nodes

    model = RGCN(num_nodes, num_relations, in_channels=64, hidden_channels=128, out_channels=64)

    x = torch.randn((num_nodes, 64), dtype=torch.float)
    train(model, data, x, epochs=100)

    embeddings = model(x, data.edge_index, data.edge_attr).detach()
    torch.save(embeddings, "conceptnet_embeddings.pt")
    print("saved embeddings")
    model.eval()
    with torch.no_grad():
        embeddings = model(x, data.edge_index, data.edge_attr)
        print("Shape of computed embeddings:", embeddings.shape)  # Should be (num_nodes, out_channels)
        print("Max value in embeddings:", embeddings.max().item())
        print("Min value in embeddings:", embeddings.min().item())
        print("Any NaNs?", torch.isnan(embeddings).any().item())



def run_no_KG():
    MODEL_NAME_OR_PATH = "facebook/bart-base"

    tokenizer_base = BartTokenizer.from_pretrained(MODEL_NAME_OR_PATH)
    model_base = BartForConditionalGeneration.from_pretrained(MODEL_NAME_OR_PATH)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    model_base.to(device)

    def generate_explanation(model, tokenizer, sentence: str) -> list:
        # Tokenize the input text
        inputs = tokenizer(sentence, return_tensors="pt").to(device)

        # Generate output using beam search:
        output_ids = model.generate(
            **inputs,
            max_length=60,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True
        )

        # Decode each of the generated token sequences
        explanations = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        return explanations

    DATA_PATH = "data/eg"
    model_path = "basic_finetuned_out"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading the model.")
    else:
        trainer = get_basic_trainer(
            source_path=DATA_PATH+"/train.source",
            target_path=DATA_PATH+"/train.target",
            model_name="facebook/bart-base", 
            output_dir="basic_finetuned_out",
            max_len=128,
            epochs=3,
            train_batch_size=60
        )

        trainer.train()
        trainer.save_model(model_path)


    output_dir = "basic_finetuned_out"
    checkpoints = [name for name in os.listdir(output_dir) if name.startswith("checkpoint")]
    print("Available checkpoints:", checkpoints)

    FINE_TUNED_MODEL_DIR = "basic_finetuned_out/checkpoint-1281"
    tokenizer = BartTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model = BartForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    while True:
        # Ask for a sentence
        test_sentence = input("Enter a sentence for explanation (or 'exit' to stop): ")
        
        # Exit the loop if user types 'exit'
        if test_sentence.lower() == 'exit':
            print("Exiting the sentence input loop.")
            break
        
        explanations = generate_explanation(model, tokenizer, test_sentence)
        print("Generated Explanations:")
        for explanation in explanations:
            print(explanation)


def run_full_KG():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    DATA_PATH = "data/eg"
    model_path = "KG_finetuned_out"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading the model.")
    else:
        trainer = get_KG_trainer(
            source_path=f"{DATA_PATH}/train.source",
            target_path=f"{DATA_PATH}/train.target",
            model_name="facebook/bart-base", 
            output_dir="KG_finetuned_out",
            max_len=128,
            epochs=1,
            train_batch_size=60
        )

        trainer.train()
        trainer.save_model(model_path)


    FINE_TUNED_MODEL_DIR = "KG_finetuned_out/checkpoint-427"

    # Load the tokenizer and custom model.
    tokenizer = BartTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model = BartGraphAwareForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_DIR)

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_explanation_kg(sentence: str) -> list:
        # Tokenize the input sentence.
        inputs = tokenizer(sentence, return_tensors="pt").to(device)
        
        # Compute the graph features for the sentence.
        graph_info = get_graph_info(sentence)
        
        # Ensure the concept_ids tensor has a batch dimension.
        if graph_info["concept_ids"].dim() == 1:
            graph_info["concept_ids"] = graph_info["concept_ids"].unsqueeze(0)
        
        # Make sure graph features are on the same device.
        graph_info["concept_ids"] = graph_info["concept_ids"].to(device)
        
        # Generate outputs, passing the extra keyword argument "concept_ids".
        output_ids = model.generate(
            **inputs,
            concept_ids=graph_info["concept_ids"],
            max_length=60,
            num_beams=3,
            num_return_sequences=3,  # Added to return 3 distinct sequences
            early_stopping=True
        )
        
        explanations = [
            tokenizer.decode(ids, skip_special_tokens=True)
            for ids in output_ids
        ]
        return explanations
    while True:
        # Ask for a sentence
        test_sentence = input("Enter a sentence for explanation (or 'exit' to stop): ")
        
        # Exit the loop if user types 'exit'
        if test_sentence.lower() == 'exit':
            print("Exiting the sentence input loop.")
            break
        
        explanations = generate_explanation_kg(test_sentence)
        print("Generated Explanations:")
        for explanation in explanations:
            print(explanation)
    

def run_compressed_KG():
    from KG_trainer_w_comp import BartGraphAwareForConditionalGeneration
    from transformers import BartTokenizer
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    DATA_PATH = "data/eg"
    model_path = "KG_finetuned_out2"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading the model.")
    else:
        trainer = comp_trainer(
            source_path=f"{DATA_PATH}/train.source",
            target_path=f"{DATA_PATH}/train.target",
            model_name="facebook/bart-base", 
            output_dir="KG_finetuned_out2",
            max_len=128,
            epochs=1,
            train_batch_size=20
        )

        trainer.train()
        trainer.save_model(model_path)
    
    model_path = "KG_finetuned_out2/checkpoint-30"
    tokenizer = BartTokenizer.from_pretrained(model_path)
    model = BartGraphAwareForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval() 

    from KG_trainer_w_comp import get_graph_info, BartGraphAwareForConditionalGeneration
    from transformers import BartTokenizer
    import torch

    # Path to your fine-tuned model checkpoint
    FINE_TUNED_MODEL_DIR = "KG_finetuned_out2/checkpoint-30"

    # Load the tokenizer and custom model.
    tokenizer = BartTokenizer.from_pretrained(FINE_TUNED_MODEL_DIR)
    model = BartGraphAwareForConditionalGeneration.from_pretrained(FINE_TUNED_MODEL_DIR)

    # Move model to GPU if available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    def generate_explanation_kg_compressed(sentence: str) -> list:
        # Tokenize the input sentence explicitly (ensure input_ids are produced).
        inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=128)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        
        # Extract the full graph information from the sentence.
        graph_info = get_graph_info(sentence)
        # Ensure the graph tensors have a batch dimension.
        if graph_info["concept_ids"].dim() == 1:
            graph_info["concept_ids"] = graph_info["concept_ids"].unsqueeze(0)
        if graph_info["edge_index"].dim() == 2:
            graph_info["edge_index"] = graph_info["edge_index"].unsqueeze(0)
        if graph_info["edge_type"].dim() == 1:
            graph_info["edge_type"] = graph_info["edge_type"].unsqueeze(0)
            
        # Move the graph tensors to the correct device.
        graph_info["concept_ids"] = graph_info["concept_ids"].to(device)
        graph_info["edge_index"] = graph_info["edge_index"].to(device)
        graph_info["edge_type"] = graph_info["edge_type"].to(device)
        
        # Call generate() explicitly with input_ids and attention_mask.
        output_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            concept_ids=graph_info["concept_ids"],
            edge_index=graph_info["edge_index"],
            edge_type=graph_info["edge_type"],
            max_length=60,
            num_beams=3,
            num_return_sequences=3,
            early_stopping=True
        )
        
        # Decode the generated token IDs.
        explanations = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
        return explanations


    while True:
        # Ask for a sentence
        test_sentence = input("Enter a sentence for explanation (or 'exit' to stop): ")
        
        # Exit the loop if user types 'exit'
        if test_sentence.lower() == 'exit':
            print("Exiting the sentence input loop.")
            break
        
        explanations = generate_explanation_kg_compressed(test_sentence)
        print("Generated Explanations:")
        for explanation in explanations:
            print(explanation)



if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    if target == 'encode':
        run_encoding()
    elif target == "no_KG":
        run_no_KG()
    elif target == "full_KG":
        run_full_KG() 
    elif target == "compressed_KG":
        run_compressed_KG()
    else:
        run_encoding()
        run_no_KG()
    
