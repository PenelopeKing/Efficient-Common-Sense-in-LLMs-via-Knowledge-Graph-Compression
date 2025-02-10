# Efficient Common Sense in LLMs via Knowledge Graph Compression

## Introduction

This repository contains the PyTorch implementation of our Data Science Capstone Project, which builds upon the methods introduced in the [EMNLP 2023](https://aclanthology.org/2023.emnlp-main.37.pdf) paper "*Knowledge Graph Compression Enhances Diverse Commonsense Generation*".

The following data retrieval and preprocessing steps are based on the original implementations from the paper. 

## Create an environment

```
nltk==3.9.1
spacy==3.8.4
torch_scatter==2.1.2+${CUDA}
psutil==5.9.0
bert_score==0.3.13
datasets==3.2.0
networkx==3.2.1
pandas==2.2.3
torch==2.5.1+${CUDA}
tqdm==4.67.1
transformers==4.48.1
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

Replace **relation.txt** with our own relation.txt in this repo. It is modified from the original processing. 

## Graph Encoding
To get the R-GCN encodings, from the project root dir, run:
```python run.py encoding </code>```
This applies a graph encoder to subgraphs create meaningful node embeddings. This is a necessary step for models below that use the knowledge graph. 

## Run Models
```
# To run our basic model (with no knowledge graph) run:
python run.py no_KG
 
# To run our uncompressed full subgraph model run:
python run.py full_KG

#To run our compressed subgraph model run:
python run.py compressed_KG 
```
## Reference

EunJeong Hwang, Veronika Thost, Vered Shwartz, and Tengfei Ma. 2023. [Knowledge Graph Compression Enhances Diverse Commonsense Generation](https://aclanthology.org/2023.emnlp-main.37.pdf). In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 558–572, Singapore. Association for Computational Linguistics.

## Acknowledgements

Many thanks to the Github repository of EMNLP 2023 paper "[Knowledge Graph Compression Enhances Diverse Commonsense Generation](https://aclanthology.org/2023.emnlp-main.37.pdf)" ([implementation](https://github.com/eujhwang/KG-Compression)) and the paper it was based on: ACL 2022 paper "[Diversifying Content Generation for Commonsense Reasoning with Mixture of Knowledge Graph Experts](https://arxiv.org/abs/2203.07285)" 

Our codes are modified based on their codes.
