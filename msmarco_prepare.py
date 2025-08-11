from datasets import load_dataset
from datasets import concatenate_datasets

corpus_ds = load_dataset("BeIR/msmarco", "corpus")
query_ds = load_dataset("BeIR/msmarco", "queries")
qrel_ds = load_dataset("BeIR/msmarco-qrels")

train = qrel_ds["train"]
test = qrel_ds["test"]
val = qrel_ds["validation"]

qrel_ds = concatenate_datasets([train,test, val])

corpus_ds =corpus_ds.rename_column("_id", "product_id")
corpus_ds =corpus_ds.rename_column("text", "combined_text")

qrel_ds =qrel_ds.rename_column("corpus-id", "product_id")
qrel_ds =qrel_ds.rename_column("query-id", "query_id")

query_ds = query_ds.rename_column("text", "query")
query_ds =query_ds.rename_column("_id", "query_id")

corpus_product_ids = set(corpus_ds['product_id'])

# REMOVE QUERIES WITH NO PRODUCTS
qrel_ds = qrel_ds.filter(lambda example: example['product_id'] in corpus_product_ids)
valid_query_ids = set(qrel_ds['query_id'])
query_ds = query_ds.filter(lambda example: example['query_id'] in valid_query_ids)

# Split into 80% train and 20% temp
split_query_1 = query_ds.train_test_split(test_size=0.2, seed=42)

# Step 2: Split the temp 20% into 50/50 → 10% val / 10% test
split_query_2 = split_query_1["test"].train_test_split(test_size=0.5, seed=42)

# Final splits
query_ds_train = split_query_1["train"]       # 80%
query_ds_test = split_query_2["train"]         # 10%
query_ds_val = split_query_2["test"]         # 10%

qrel_ds.save_to_disk("msmarco_qrel_ds")

query_ds_train.save_to_disk("msmarco_query_ds_train")
query_ds_test.save_to_disk("msmarco_query_ds_test")
query_ds_val.save_to_disk("msmarco_query_ds_val")

corpus_ds.save_to_disk("msmarco_corpus_ds")