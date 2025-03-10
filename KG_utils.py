import json
import pickle
import networkx as nx
import os

# ----------------------------------------------------------------------
# 1) Define file paths
# ----------------------------------------------------------------------
concept_vocab_path = "data/concept.txt"
relation_vocab_path = "data/relation.txt"
cpnet_path         = "data/cpnet.graph"

data_path = "data/eg"

# ----------------------------------------------------------------------
# 2) Global dictionaries and graphs
# ----------------------------------------------------------------------
concept2id = {}
id2concept = {}
relation2id = {}
id2relation = {}

cpnet = None
cpnet_simple = None

# This will store the "allowed" concept IDs used for BFS filtering
allowed_nodes = set()

# ----------------------------------------------------------------------
# 3) Load vocabularies
# ----------------------------------------------------------------------
def load_resources():
    global concept2id, id2concept, relation2id, id2relation

    # Check if concept vocab file exists
    if not os.path.exists(concept_vocab_path):
        print("[DEBUG] load_resources: Concept vocabulary file missing:", concept_vocab_path)
    else:
        print("[DEBUG] load_resources: Loading concept vocabulary from:", concept_vocab_path)
    # Load concept vocabulary
    with open(concept_vocab_path, "r", encoding="utf8") as f:
        for line in f:
            concept = line.strip()
            cid = len(concept2id)
            concept2id[concept] = cid
            id2concept[cid] = concept
    print(f"[DEBUG] load_resources: Loaded {len(concept2id)} concepts.")

    # Check if relation vocab file exists
    if not os.path.exists(relation_vocab_path):
        print("[DEBUG] load_resources: Relation vocabulary file missing:", relation_vocab_path)
    else:
        print("[DEBUG] load_resources: Loading relation vocabulary from:", relation_vocab_path)
    # Load relation vocabulary
    with open(relation_vocab_path, "r", encoding="utf8") as f:
        for line in f:
            rel = line.strip()
            rid = len(relation2id)
            relation2id[rel] = rid
            id2relation[rid] = rel
    print(f"[DEBUG] load_resources: Loaded {len(relation2id)} relations.")

# ----------------------------------------------------------------------
# 4) Load the ConceptNet graph
# ----------------------------------------------------------------------
def load_cpnet():
    global cpnet, cpnet_simple
    if not os.path.exists(cpnet_path):
        print("[DEBUG] load_cpnet: CPNet file missing:", cpnet_path)
    else:
        print("[DEBUG] load_cpnet: Found CPNet file:", cpnet_path)
    print("[DEBUG] load_cpnet: Loading cpnet...")
    with open(cpnet_path, "rb") as f:
        cpnet = pickle.load(f)
    print("[DEBUG] load_cpnet: Finished loading cpnet.")

    # Build an undirected version
    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]["weight"] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)
    print(f"[DEBUG] load_cpnet: cpnet_simple has {cpnet_simple.number_of_nodes()} nodes and {cpnet_simple.number_of_edges()} edges.")

# ----------------------------------------------------------------------
# 5) Load "allowed" concepts from data
#    This replicates the original approach that collects 
#    all concepts from train/val files, then filters them to those in concept2id.
# ----------------------------------------------------------------------
def load_total_concepts(data_path):
    global allowed_nodes  # we'll store them in this set
    total_concepts = []
    print(f"[DEBUG] load_total_concepts: Loading allowed concepts from directory: {data_path}")

    for fname in ["train.concepts_nv.json", "val.concepts_nv.json"]:
        full_path = f"{data_path}/{fname}"
        if not os.path.exists(full_path):
            print(f"[DEBUG] load_total_concepts: File not found -> {full_path}")
            continue
        print(f"[DEBUG] load_total_concepts: Loading file: {full_path}")
        try:
            with open(full_path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    # 'qc' => question concepts, 'ac' => answer concepts
                    # The original code extends them together
                    total_concepts.extend(item['qc'] + item['ac'])
        except Exception as e:
            print(f"[DEBUG] load_total_concepts: Error reading {full_path}: {e}")

    # Unique
    total_concepts = list(set(total_concepts))
    print(f"[DEBUG] load_total_concepts: Found {len(total_concepts)} unique concepts in train/val files.")

    # Convert them to IDs, filter out unknowns
    valid_ids = []
    for cpt in total_concepts:
        if cpt in concept2id:
            valid_ids.append(concept2id[cpt])
        else:
            print(f"[DEBUG] load_total_concepts: Concept '{cpt}' not found in concept2id.")

    allowed_nodes = set(valid_ids)
    print(f"[DEBUG] load_total_concepts: Loaded {len(allowed_nodes)} allowed concept IDs from train/val data.")

# ----------------------------------------------------------------------
# 6) BFS-based triple extraction with filtering
#    This matches the original code's "find_neighbours_frequency" approach:
#    - BFS up to T hops
#    - At each hop, keep top max_B neighbors by frequency
#    - Filter expansions to "allowed_nodes"
# ----------------------------------------------------------------------
def get_triples_for_token_bfs(input_token, T=2, max_B=100):
    """
    Perform BFS from `input_token` (string) in the UNDIRECTED `cpnet_simple`.
    Restrict expansions to `allowed_nodes`.
    Return a list of (src_str, [rel_ids], tgt_str) edges.
    """
    print(f"[DEBUG] get_triples_for_token_bfs: Starting BFS for token '{input_token}'")
    if input_token not in concept2id:
        print(f"[DEBUG] get_triples_for_token_bfs: Token '{input_token}' not in concept2id.")
        return []

    start_id = concept2id[input_token]
    if start_id not in cpnet_simple:
        print(f"[DEBUG] get_triples_for_token_bfs: Start ID {start_id} not in cpnet_simple.")
        return []

    # We'll track how many hops from the starting token
    visited = {start_id: 0}
    frontier = [start_id]

    # Ets: a place to store discovered edges
    # Ets[tgt_node] = {src_node: [rel_ids], ...}
    Ets = {}

    for hop_num in range(T):
        neighbor_freq = {}  # track frequency of each neighbor in this hop
        print(f"[DEBUG] get_triples_for_token_bfs: Hop {hop_num + 1}, current frontier size: {len(frontier)}")
        for src_node in frontier:
            if src_node not in cpnet_simple:
                continue
            for nbr in cpnet_simple[src_node]:
                if nbr not in visited and nbr in allowed_nodes:
                    neighbor_freq[nbr] = neighbor_freq.get(nbr, 0) + 1
                    if cpnet.has_edge(src_node, nbr):
                        rel_ids = []
                        for key, edge_data in cpnet[src_node][nbr].items():
                            if "rel" in edge_data:
                                rel_ids.append(edge_data["rel"])
                        if nbr not in Ets:
                            Ets[nbr] = {src_node: rel_ids}
                        else:
                            Ets[nbr][src_node] = rel_ids
        sorted_nbrs = sorted(neighbor_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_nbrs = sorted_nbrs[:max_B]
        print(f"[DEBUG] get_triples_for_token_bfs: Hop {hop_num + 1} selected {len(sorted_nbrs)} neighbors.")
        new_frontier = []
        for nid, _freq in sorted_nbrs:
            visited[nid] = hop_num + 1
            new_frontier.append(nid)
        frontier = new_frontier

    final_triples = []
    for tgt_node, src_map in Ets.items():
        if tgt_node in visited:
            for src_node, rel_ids in src_map.items():
                if src_node in visited:
                    src_str = id2concept[src_node].replace("_", " ")
                    tgt_str = id2concept[tgt_node].replace("_", " ")
                    final_triples.append((src_str, rel_ids, tgt_str))
    print(f"[DEBUG] get_triples_for_token_bfs: Found {len(final_triples)} triples for token '{input_token}'.")
    return final_triples

def get_triples_for_message(message, T=2, max_B=100):
    """
    For each token in the input message that exists in the concept vocabulary,
    perform a BFS-based connection search in the ConceptNet graph and return the results.
    
    Parameters:
        message (str): The input message (e.g., sentence or paragraph).
        T (int): Number of BFS hops (default 2).
        max_B (int): Maximum number of neighbors per hop (default 100).
    
    Returns:
        dict: A dictionary mapping each token (that exists in concept2id) to its list of triples.
              Each triple is of the form (src_str, [rel_ids], tgt_str).
    """
    tokens = message.lower().split()
    print(f"[DEBUG] get_triples_for_message: Processing message with {len(tokens)} tokens.")
    message_triples = {}
    for token in tokens:
        if token in concept2id:
            triples = get_triples_for_token_bfs(token, T=T, max_B=max_B)
            if triples:
                message_triples[token] = triples
            else:
                print(f"[DEBUG] get_triples_for_message: No triples found for token '{token}'.")
        else:
            print(f"[DEBUG] get_triples_for_message: Token '{token}' not in concept2id, skipping.")
    return message_triples

def get_subgraph_for_message(message, T=2, max_B=100):
    """
    For an input message, perform BFS-based extraction for each token and build a subgraph.
    
    The subgraph is a NetworkX undirected graph containing all nodes and edges 
    (with relation information) that were extracted for the message.
    
    Parameters:
        message (str): The input message.
        T (int): Number of BFS hops.
        max_B (int): Maximum number of neighbors per hop.
    
    Returns:
        networkx.Graph: A subgraph containing the extracted nodes and edges.
                        Each edge has an attribute 'rel_ids' (a list of relation IDs).
    """
    print(f"[DEBUG] get_subgraph_for_message: Building subgraph for message: {message[:50]}...")
    message_triples = get_triples_for_message(message, T, max_B)
    
    subgraph = nx.Graph()
    for token, triples in message_triples.items():
        for src, rel_ids, tgt in triples:
            subgraph.add_node(src)
            subgraph.add_node(tgt)
            if subgraph.has_edge(src, tgt):
                existing_rel_ids = subgraph[src][tgt].get("rel_ids", [])
                merged = list(set(existing_rel_ids + rel_ids))
                subgraph[src][tgt]["rel_ids"] = merged
            else:
                subgraph.add_edge(src, tgt, rel_ids=rel_ids)
    print(f"[DEBUG] get_subgraph_for_message: Subgraph built with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    return subgraph
