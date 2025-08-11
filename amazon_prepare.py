import pandas as pd
import torch
import numpy as np
from datasets import Dataset

df_examples = pd.read_parquet('shopping_queries_dataset/shopping_queries_dataset_examples.parquet')
df_products = pd.read_parquet('shopping_queries_dataset/shopping_queries_dataset_products.parquet')
df_sources = pd.read_csv("shopping_queries_dataset/shopping_queries_dataset_sources.csv")


products = Dataset.from_pandas(df_products)
examples = Dataset.from_pandas(df_examples)

SMALL = False
products_ds = products.filter(lambda example: example['product_locale'] == "us")

examples_ds = examples.filter(lambda example: example['small_version'] == 1)
examples_ds = examples.filter(lambda example: example['product_locale'] == "us")

if SMALL == True:
    train_split, products_ds = products_ds.train_test_split(test_size=0.01, seed=42).values()

from html.parser import HTMLParser

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text = []

    def handle_data(self, data):
        self.text.append(data)

    def get_text(self):
        return " ".join(self.text).strip()

def strip_html(text):
    if text is None:
        return ""
    stripper = HTMLStripper()
    stripper.feed(text)
    return stripper.get_text()

def concatenate_fields(example):
    fields = [example["product_title"], example["product_color"], example["product_brand"], example["product_description"]]
    filtered_fields = [strip_html(f) for f in fields if f]  # Remove HTML and None values
    example["combined_text"] = " ".join(filtered_fields).strip()
    return example

products_ds = products_ds.map(concatenate_fields)

cols_to_remove = products_ds.column_names
cols_to_remove.remove("product_id")
cols_to_remove.remove("combined_text")
corpus_ds = products_ds.remove_columns(cols_to_remove)

examples_ds = examples_ds.remove_columns("small_version")
examples_ds = examples_ds.remove_columns("large_version")
examples_ds = examples_ds.remove_columns("__index_level_0__")

examples_rel = examples_ds.filter(lambda example: example['esci_label'] == "E")
examples_rel = examples_rel.remove_columns("esci_label")
examples_rel = examples_rel.remove_columns("example_id")
examples_rel= examples_rel.remove_columns("product_locale")
qrel_ds = examples_rel.remove_columns("query")

query_ds = examples_rel.remove_columns("product_id")

seen = set()
query_ds = query_ds.filter(lambda example: example["query_id"] not in seen and not seen.add(example["query_id"]))

corpus_product_ids = set(products_ds['product_id'])

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

# Split into 80% train and 20% temp
split_qrel_1 = qrel_ds.train_test_split(test_size=0.2, seed=42)

# Step 2: Split the temp 20% into 50/50 → 10% val / 10% test
split_qrel_2 = split_qrel_1["test"].train_test_split(test_size=0.5, seed=42)

# Final splits
qrel_ds_train = split_qrel_1["train"]       # 80%
qrel_ds_test = split_qrel_2["train"]         # 10%
qrel_ds_val = split_qrel_2["test"]         # 10%

qrel_ds_train.save_to_disk("amazon_qrel_ds_train")
qrel_ds_test.save_to_disk("amazon_qrel_ds_test")
qrel_ds_val.save_to_disk("amazon_qrel_ds_val")

query_ds_train.save_to_disk("amazon_query_ds_train")
query_ds_test.save_to_disk("amazon_query_ds_test")
query_ds_val.save_to_disk("amazon_query_ds_val")

corpus_ds.save_to_disk("amazon_corpus_ds")