import json
import pickle
import networkx as nx

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

    # Load concept vocabulary
    with open(concept_vocab_path, "r", encoding="utf8") as f:
        for line in f:
            concept = line.strip()
            cid = len(concept2id)
            concept2id[concept] = cid
            id2concept[cid] = concept

    # Load relation vocabulary
    with open(relation_vocab_path, "r", encoding="utf8") as f:
        for line in f:
            rel = line.strip()
            rid = len(relation2id)
            relation2id[rel] = rid
            id2relation[rid] = rel

# ----------------------------------------------------------------------
# 4) Load the ConceptNet graph
# ----------------------------------------------------------------------
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

# ----------------------------------------------------------------------
# 5) Load "allowed" concepts from data
#    This replicates the original approach that collects 
#    all concepts from train/val files, then filters them to those in concept2id.
# ----------------------------------------------------------------------
def load_total_concepts(data_path):
    global allowed_nodes  # we'll store them in this set
    total_concepts = []

    for fname in ["train.concepts_nv.json", "val.concepts_nv.json"]:
        full_path = f"{data_path}/{fname}"
        try:
            with open(full_path, "r", encoding="utf8") as f:
                for line in f.readlines():
                    item = json.loads(line)
                    # 'qc' => question concepts, 'ac' => answer concepts
                    # The original code extends them together
                    total_concepts.extend(item['qc'] + item['ac'])
        except FileNotFoundError:
            print(f"Warning: file not found -> {full_path}")

    # Unique
    total_concepts = list(set(total_concepts))

    # Convert them to IDs, filter out unknowns
    valid_ids = []
    for cpt in total_concepts:
        if cpt in concept2id:
            valid_ids.append(concept2id[cpt])

    allowed_nodes = set(valid_ids)
    print(f"Loaded {len(allowed_nodes)} allowed concept IDs from train/val data.")

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

    if input_token not in concept2id:
        return []

    start_id = concept2id[input_token]
    if start_id not in cpnet_simple:
        return []

    # We'll track how many hops from the starting token
    visited = {start_id: 0}
    frontier = [start_id]

    # Ets: a place to store discovered edges
    # Ets[tgt_node] = {src_node: [rel_ids], ...}
    Ets = {}

    for hop_num in range(T):
        neighbor_freq = {}  # track frequency of each neighbor in this hop

        for src_node in frontier:
            # If src_node not in cpnet_simple adjacency, skip
            if src_node not in cpnet_simple:
                continue

            # Expand neighbors
            for nbr in cpnet_simple[src_node]:
                # Must not be visited; must be in allowed nodes
                if nbr not in visited and nbr in allowed_nodes:
                    # Count how many times we see 'nbr'
                    neighbor_freq[nbr] = neighbor_freq.get(nbr, 0) + 1

                    # Retrieve relations from the directed graph `cpnet`
                    if cpnet.has_edge(src_node, nbr):
                        rel_ids = []
                        for key, edge_data in cpnet[src_node][nbr].items():
                            if "rel" in edge_data:
                                rel_ids.append(edge_data["rel"])

                        # Store them
                        if nbr not in Ets:
                            Ets[nbr] = {src_node: rel_ids}
                        else:
                            Ets[nbr][src_node] = rel_ids

        # Sort neighbors by frequency, keep top `max_B`
        sorted_nbrs = sorted(neighbor_freq.items(), key=lambda x: x[1], reverse=True)
        sorted_nbrs = sorted_nbrs[:max_B]

        # Mark them visited, set BFS frontier for the next hop
        new_frontier = []
        for nid, _freq in sorted_nbrs:
            visited[nid] = hop_num + 1
            new_frontier.append(nid)

        frontier = new_frontier

    # ---------------------------------------------
    # Collect final edges from Ets
    # Only keep edges if both ends are visited
    # ---------------------------------------------
    final_triples = []
    for tgt_node, src_map in Ets.items():
        if tgt_node in visited:
            for src_node, rel_ids in src_map.items():
                if src_node in visited:
                    src_str = id2concept[src_node].replace("_", " ")
                    tgt_str = id2concept[tgt_node].replace("_", " ")
                    final_triples.append((src_str, rel_ids, tgt_str))

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
    # Tokenize the message. Here, we simply split on whitespace and lower the case.
    # You can replace this with a more advanced tokenizer if needed.
    tokens = message.lower().split()
    
    message_triples = {}
    
    for token in tokens:
        if token in concept2id:
            # Use your existing function to get BFS triples for this token.
            triples = get_triples_for_token_bfs(token, T=T, max_B=max_B)
            if triples:
                message_triples[token] = triples
        #else:
            #print(f"Warning: Token '{token}' not found in the concept vocabulary. Skipping.")
    
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
    message_triples = get_triples_for_message(message, T, max_B)
    
    # Create an empty undirected graph for the subgraph
    subgraph = nx.Graph()
    
    # Iterate over all tokens and their associated triples, adding nodes and edges.
    for token, triples in message_triples.items():
        for src, rel_ids, tgt in triples:
            # Add the nodes (this is idempotent if they already exist)
            subgraph.add_node(src)
            subgraph.add_node(tgt)
            
            # If the edge already exists, merge the relation information.
            if subgraph.has_edge(src, tgt):
                existing_rel_ids = subgraph[src][tgt].get("rel_ids", [])
                # Merge without duplicates
                merged = list(set(existing_rel_ids + rel_ids))
                subgraph[src][tgt]["rel_ids"] = merged
            else:
                subgraph.add_edge(src, tgt, rel_ids=rel_ids)
    
    return subgraph

