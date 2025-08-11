import torch
import numpy as np
from datasets import Dataset
from datasets import load_from_disk
import pandas as pd
import faiss 
from sklearn.metrics import ndcg_score
import time

# corpus_ds: "product_id", "combined_text"
# query_ds:  "query", "query_id"
# qrel_ds: "query_id", "product_id"
# top_100_ds: "label", "score", "query_id", "product_id", "product_embedding", "corpus_embedding"

# TODO: kanske döpa om scores kolonnerna så både retrival scores och reranking scores sparas


RETRIEVAL = True
RERANK = True
device = "cuda"

# need to select what dataset,this determines the corpus_ds, qrel_ds, query_ds and top_100 retrieval dataset we use
dataset = "AMAZON"
#dataset = "THECA"
#dataset = "MSMARCO"

reranker = "CROSSENCODER"
#reranker = "SVM"
#reranker = "BIENCODER"

# here corpuse is the whole corpus, and qrel and query_ds is evaluation datasets, top_100 is initial retrival 
if dataset == "AMAZON":
    prestring = "amazon"
if dataset == "THECA":
    prestring = "theca"
if dataset == "MSMARCO":
    prestring = "msmarco"

corpus_ds = load_from_disk(prestring + "_corpus_ds")
qrel_ds = load_from_disk(prestring + "_qrel_test_ds")
query_ds = load_from_disk( prestring + "_query_test_ds")

# uncomment if we already have embeddings 
#corpus_embeddings = load_from_disk( prestring + "_corpus_embeddings")


# RETRIEVAL

if RETRIEVAL: 
    from sentence_transformers import SentenceTransformer
    bi_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v2.0", trust_remote_code=True, tokenizer_kwargs={"model_max_length" : 512})
    bi_model.to("cuda")

    #create corpus and query embeddings 
    corpus_embeddings = bi_model.encode(corpus_ds["combined_text"], show_progress_bar=True) 
    corpus_embeddings.save_to_disk(prestring+"_corpus_embeddings")
    query_embeddings = bi_model.encode(query_ds["query"], prompt_name="query", show_progress_bar=True)
    query_embeddings.save_to_disk(prestring+"_query_embeddings")

    # Create FAISS index (L2 normalized embeddings needed for dot product / cosine similarity)

    dimension = query_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product (equivalent to cosine similarity if normalized)
    faiss.normalize_L2(corpus_embeddings)
    index.add(corpus_embeddings)  # Add documents to index
    faiss.normalize_L2(query_embeddings)

    faiss.write_index(index, prestring+"_index.faiss") #save index to file

    k = 100  
    scores, I = index.search(query_embeddings, k)  # D = scores, I = indices of top-k docs

    # Add scores to dataset
    query_ids = query_ds["query_id"]  # List of all query IDs
    product_ids = corpus_ds["product_id"]  # List of all product IDs
    query_ids = np.array(query_ids)
    product_ids = np.array(product_ids)

    scores_flat = scores.flatten()  # Convert the tensor to a flattened NumPy array
    query_ids_flat = np.repeat(query_ids, k)
    product_ids_flat = []
    for ind in I.flatten():
        product_id = product_ids[ind]
        product_ids_flat.append(product_id)
    flattened_scores = pd.DataFrame({
        "query_id": query_ids_flat,
        "product_id": product_ids_flat,
        "score": scores_flat
    })

    scores_ds = Dataset.from_pandas(flattened_scores)

    def get_top_k_per_query(scores_ds: Dataset, k: int = 100) -> Dataset:
        scores_df = scores_ds.to_pandas()

        #Sort by 'query_id' and 'score' (descending order)
        scores_df = scores_df.sort_values(by=['query_id', 'score'], ascending=[True, False])

        # Group by 'query_id' and retain top k rows for each query
        top_k_scores_df = (
            scores_df.groupby('query_id')
            .head(k)
            .reset_index(drop=True)
        )

        top_k_scores_ds = Dataset.from_pandas(top_k_scores_df)
        return top_k_scores_ds

    top_100 = get_top_k_per_query(scores_ds,k=100)

    # Add labels to top_100 (1=relevant, 0=not relevant)
    def add_labels(top_100_dataset: Dataset, qrel: Dataset):
        new_col = []
        
        # Create a dictionary from qrel 
        qrel_dict = {}
        for row in qrel:
            qrel_dict[(row['query_id'], row['product_id'])] = 1  # Relevant products are labeled as 1
        
        # Iterate through the top 100 dataset and add labels
        iterat = top_100_dataset.iter(batch_size=1)
        for i in iterat:
            query_id = i["query_id"][0]
            product_id = i["product_id"][0]
            
            # Check if the (query_id, product_id) exists in qrel_dict
            if (query_id, product_id) in qrel_dict:
                new_col.append(1)  # Label 1 if relevant
            else:
                new_col.append(0)  # Label 0 if not relevant

        top_100_dataset = top_100_dataset.add_column("label", new_col)
        
        return top_100_dataset

    top_100_ds = add_labels(top_100, qrel_ds)

    # Add embeddings to top_100_ds
    # Convert embeddings to dictionaries for quick lookup
    query_id_to_embedding = {q_id: query_embeddings[i] for i, q_id in enumerate(query_ds["query_id"])}
    product_id_to_embedding = {p_id: corpus_embeddings[i] for i, p_id in enumerate(corpus_ds["product_id"])}

    # Retrieve embeddings for each row in top_100_ds
    query_emb_list = []
    product_emb_list = []

    for row in top_100_ds:
        query_emb_list.append(query_id_to_embedding[row["query_id"]])
        product_emb_list.append(product_id_to_embedding[row["product_id"]])

    query_emb_list = np.array(query_emb_list)
    product_emb_list = np.array(product_emb_list)

    # Add embeddings as new columns
    top_100_ds = top_100_ds.add_column("query_embedding", query_emb_list.tolist())
    top_100_ds = top_100_ds.add_column("product_embedding", product_emb_list.tolist())

else: #if we have already done retrieval but want to test another re-ranker, this is run
    top_100_ds = load_from_disk(prestring + "_top_ds")

# NDCG 
def ndcg_at_k(top_100_labels : Dataset, k, top_k):
    y_true = []
    y_score = []
    score_row = []
    true_row = []
    for i in range(top_100_labels.num_rows):
        score_row.append(top_100_labels[i]["score"])
        true_row.append(top_100_labels[i]["label"])
        if (i+1) % top_k == 0:
            y_true.append(true_row)
            y_score.append(score_row)
            score_row = []
            true_row = []

    y_score = np.array(y_score)
    y_true = np.array(y_true)
    return ndcg_score(y_true, y_score, k=k)  # Compute NDCG@10

retrieval_ndcg_score = ndcg_score(top_100_ds, 10, 100)
print(f"Retrieval NDCG@10: {retrieval_ndcg_score}")

if RERANK:
    if reranker == "CROSSENCODER":
        from transformers import AutoTokenizer, AutoModel 
        import torch.nn as nn

        class SnowflakeClassifier(nn.Module):
            def __init__(self, model_name, num_labels, dropout_value):
                super().__init__() 
                self.transformer = AutoModel.from_pretrained(model_name, add_pooling_layer=False, trust_remote_code=True)
                self.dropout = nn.Dropout(dropout_value)
                self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)
            
            def forward(self, input_ids, attention_mask, labels=None):
                outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs.last_hidden_state[:,0,:]
                cls_embedding = self.dropout(cls_embedding)
                logits = self.classifier(cls_embedding)
                if labels is not None:
                    loss_fn = torch.nn.CrossEntropyLoss()
                    loss = loss_fn(logits, labels)
                    return {"loss": loss, "logits": logits}
                return {"logits": logits}

        model_name = "Snowflake/snowflake-arctic-embed-m-v2.0"
        num_labels = 2
        dropout_value = 0.175
        model_cross = SnowflakeClassifier(model_name, num_labels, dropout_value)
        model_cross.to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        def crossencoder_reranker(model: nn.Module, tokenizer, top_k: Dataset, device): 
            t_start = time.time()
            query_product_text = []
            query_text = top_k["query"][0] 
            product_texts = top_k["product_text"]
            for p_text in product_texts:
                query_product_text.append([query_text, p_text])

            tokens = tokenizer(query_product_text, padding=True, truncation=True, return_tensors="pt")
            tokens.to(device)
            model.to(device)

            model.eval()
            with torch.no_grad():
                logits = model(**tokens)
                soft = nn.Softmax(dim=-1)
                scores = soft(logits)[:,1]
            t_stop = time.time()
            time = 1000*(t_stop-t_start)
            return scores,time
        
        # in rerank we do pairs for each query and its 100 products, one query at a time
        scores = []
        times = []
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = crossencoder_reranker(model_cross, tokenizer, one_top_k, device)
            scores.append(one_scores)
            times.append(one_time)
        
        # this replaces the retrieval scores with reranking scores
        reranked_top_100_ds = top_100_ds.remove_columns("score").add_column("score", scores)


    if reranker == "SVM":
        #TODO: 
        from sklearn import svm

        # load faiss index that we created during retrieval
        index = faiss.read_index(prestring+"_index.faiss")

        def svm_reranker(top_k: Dataset, faiss_index, corpus_embeddings): 
            t_start = time.time()
            query_embedding = top_k["query_embedding"][0] 
            product_embeddings = top_k["product_embedding"]

            #dimension = query_embedding.shape[1]
            #index = faiss.IndexFlatIP(dimension)
            #faiss.normalize_L2(corpus_embeddings)
            #index.add(corpus_embeddings)
            faiss.normalize_L2(query_embedding)
            k = 100
            _, I = index.search(-1*query_embedding, k)

            negative_embeddings = corpus_embeddings[I] 

            x_train = np.concatenate([query_embedding[None,...], negative_embeddings])
            x_rank = np.concatenate([query_embedding[None,...], product_embeddings])
            y = np.zeros(len(negative_embeddings)+1)
            y[0] = 1

            clf = svm.LinearSVC(class_weight='balanced', verbose=False, max_iter=10000, tol=1e-6, C=1, dual='auto', loss="hinge")
            clf.fit(x_train,y)

            scores = clf.decision_function(x_rank)[1:]
            t_stop = time.time()
            time = 1000*(t_stop-t_start)
            return scores, time
        
        scores = []
        times = []
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = svm_reranker(one_top_k, index, corpus_embeddings)

            scores.append(one_scores)
            times.append(one_time)

        reranked_top_100_ds = top_100_ds.remove_columns("score").add_column("score", scores)

    if reranker == "BIENCODER":
        from sentence_transformers import SentenceTransformer
        bi_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", trust_remote_code=True, tokenizer_kwargs={"model_max_length" : 512})
        bi_model.to(device)

        def biencoder_reranker(model, one_top_k, device):
            t_start = time.time()
            query_embedding = model.encode(one_top_k[0]["query"], prompt_name="query", show_progress_bar=True)
            corpus_embeddings = model.encode(corpus_ds["combined_text"], show_progress_bar=True)

            scores = bi_model.similarity(query_embedding, corpus_embeddings)
            t_stop = time.time()
            time = 1000*(t_stop-t_start)
            return scores, time

        scores = []
        times = []
        for i in range(0, len(top_100_ds), k):
            one_top_k = top_100_ds.select(range(i, min(i + 100, len(top_100_ds))))
            one_scores, one_time = biencoder_reranker(bi_model, one_top_k, device)
            scores.append(one_scores)
            times.append(one_time)

        reranked_top_100_ds = top_100_ds.remove_columns("score").add_column("score", scores)
        

    ndcg_score = ndcg_at_k(reranked_top_100_ds,k=10,top_k=100)
    print(f"Reranked NDCG@10: {ndcg_score}")
