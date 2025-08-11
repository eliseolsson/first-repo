# Evaluating re-ranking methods 

By running eval.py three different re-ranking methods can be evaluated. The methods are a cross-encoder, an SVM model and a bi-encoder model. The evaluation can be done on three datasets, Amazon's shopping queries dataset, MS MARCO passage re-ranking and a private dataset from Theca. The data is not included here but Amazon's can be downloaded from https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset and MS MARCO from https://huggingface.co/datasets/BeIR/msmarco and https://huggingface.co/datasets/BeIR/msmarco-qrels. Supports evaluation in terms of NDCG, MRR and MAP@k where k can be chosen. 


## How to run
1. Download the dataset you want to evaluate on
2. Run the prepare.py file for the correct dataset
3. Run the eval.py file and select the right dataset at the top as well as the perferred re-ranking model to evaluate. 
