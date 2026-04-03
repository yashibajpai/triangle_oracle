# triangle-oracle: learning-augmented triangle counting

this project builds a prediction oracle for triangle counting in graphs, using machine learning models (mlp + transformer) and evaluates how these predictions improve algorithmic performance in a learning-augmented algorithms (laa) framework.

---

## 🧠 overview

triangle counting is a fundamental problem in graph analysis, especially in biological networks (e.g., protein-protein interaction graphs). 

traditional algorithms treat all edges equally.  

this project introduces a **prediction oracle** that estimates how “important” an edge is:

> edge heaviness = number of triangles containing that edge

these predictions are then used to guide a learning-augmented algorithm.

---

## pipeline

the full pipeline consists of 4 stages:

raw graph → dataset → model → predictions → laa evaluation

---

## 1. data preparation
### input
data/raw/ppi_edges.csv
### process
- load graph using `networkx`
- compute edge heaviness (triangle counts)
- generate edge-level features (degree, clustering, etc.)
- split into train / validation / test
### output
data/processed/
train_edges.csv
valid_edges.csv
test_edges.csv
### run
``` bash scripts/prepare_data.sh ```
 ---

## 2. model training
this project implements two oracle models:
mlp oracle (baseline)
- input: handcrafted edge features
- architecture: feedforward neural network
- purpose: fast baseline for comparison
### run
``` bash scripts/train_mlp.sh ```
transformer oracle (edge neighborhood encoder)
- input: tokenized local neighborhood of each edge
- architecture: transformer encoder (bidirectional attention)
- purpose: learn structural patterns directly from graph topology
key idea

each edge is converted into a sequence:

[EDGE] [U] u [V] v [SEP] [NU] neighbors(u) [SEP] [NV] neighbors(v)

the transformer uses self-attention to detect shared neighbors, which correspond to triangles.

### run

``` python -m triangle_oracle.cli.train_transformer_cli ```

---

### 3. prediction generation

after training, models output predictions:

outputs/predictions/run1/
  predictions.npz
  best_predictions.npz

these files contain:

- predicted edge heaviness
- ground truth values
### run
```bash scripts/predict.sh ```

---

### 4. learning-augmented evaluation
predictions are used to guide triangle counting:

- classify edges as “heavy” or “light”
- prioritize heavy edges
- compare against baseline algorithms
- metrics
- precision / recall for heavy edge detection
- effect of prediction error on performance

### run
``` bash scripts/eval_laa.sh ```

---

### models
mlp oracle
- uses engineered features
- fast and simple
- serves as baseline
transformer oracle
- uses edge-centered local neighborhoods
- tokenizes graph structure into sequences
- applies bidirectional self-attention
captures structural patterns such as shared neighbors

---
 
### important:
the oracle does not need perfect predictions, it only needs to rank edges effectively

---

### setup
1. create virtual environment

``` python -m venv .venv ```
``` source .venv/bin/activate ``
`
2. install dependencies
``` pip install -r requirements.txt ```

---

### full pipeline run
run everything step by step:

``` bash scripts/prepare_data.sh ```
``` bash scripts/train_mlp.sh ```
``` bash scripts/predict.sh ```
``` bash scripts/eval_laa.sh ```

---

### author
yashi bajpai

virginia tech '29

this project was made through the FY-FURF research program, in collaboration with Virginia Tech professor Dr. Ali Vakilian 