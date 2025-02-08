# Efficient Common Sense in LLMs via Knowledge Graph Compression

## Introduction

This repository contains the PyTorch implementation of our Data Science Capstone Project, which builds upon the methods introduced in the [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.37.pdf) paper "*Knowledge Graph Compression Enhances Diverse Commonsense Generation*".

The following data retrieval and preprocessing steps are based on the original implementations from the paper. 

## Create an environment

```
transformers==3.3.1
torch==1.7.0
nltk==3.4.5
networkx==2.1
spacy==2.2.1
torch-scatter==2.0.5+${CUDA}
psutil==5.9.0
```

-- For `torch-scatter`, `${CUDA}` should be replaced by either `cu101` `cu102` `cu110` or `cu111` depending on your PyTorch installation.


## Preprocess the data

-- Extract English ConceptNet and build graph.

```bash
cd data
wget https://s3.amazonaws.com/conceptnet/downloads/2018/edges/conceptnet-assertions-5.6.0.csv.gz
gzip -d conceptnet-assertions-5.6.0.csv.gz
cd ../preprocess
python extract_cpnet.py
python graph_construction.py
```

-- Preprocess multi-hop relational paths. Set `$DATA` to either `anlg` or `eg`.

```bash
export DATA=eg
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
```
