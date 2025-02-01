def load_cpnet():
    global cpnet, concept2id, relation2id, id2relation, id2concept, cpnet_simple
    print("loading cpnet....")
    cpnet = nx.read_gpickle(config["paths"]["conceptnet_en_graph"])
    print("Done")

    cpnet_simple = nx.Graph()
    for u, v, data in cpnet.edges(data=True):
        w = data['weight'] if 'weight' in data else 1.0
        if cpnet_simple.has_edge(u, v):
            cpnet_simple[u][v]['weight'] += w
        else:
            cpnet_simple.add_edge(u, v, weight=w)