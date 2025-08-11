# Evaluating Re-ranking Methods

This repository contains code to evaluate three different re-ranking methods:

- **Cross-encoder**
- **SVM model**
- **Bi-encoder model**

The evaluation can be performed on the following datasets:

1. **Amazon Shopping Queries** – [Download here](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset)  
2. **MS MARCO Passage Re-ranking** – [Download here](https://huggingface.co/datasets/BeIR/msmarco) and [here](https://huggingface.co/datasets/BeIR/msmarco-qrels)  
3. **Private Theca dataset** (not included)

> **Note:** The datasets are not included in this repository. You must download them manually.

The script supports evaluation metrics: **NDCG@k**, **MRR@k**, and **MAP@k** (with customizable *k*).

---

## About the Project

This code was originally developed as part of a thesis project.  
For the thesis, we trained our own cross-encoder model. The trained model is **not** included here, but you can run `eval.py` with your own cross-encoder to test its performance.

---

## How to Run

1. **Download** the dataset you want to evaluate on.  
2. **Prepare** the dataset by running:  
   ```bash
   python datasetname_prepare.py
3. **Evaluate** the chosen model by running:
   ```bash
    python eval.py
At the top of eval.py, specify:

- The dataset

- The re-ranking model to evaluate
