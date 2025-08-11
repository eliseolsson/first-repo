import gzip
import json
import random
import pandas as pd
from datasets import load_dataset
from datasets import concatenate_datasets

clicks = load_dataset('json', data_files={
    'train': 'theca/swedish/ma_data/splits/clicks_train.jsonl.gz',
    'test': 'theca/swedish/ma_data/splits/clicks_test.jsonl.gz',
    'validation': 'theca/swedish/ma_data/splits/clicks_validation.jsonl.gz'
})
clicks_train = clicks["train"]
clicks_test = clicks["test"]
clicks_val = clicks["validation"]

queries = load_dataset('json', data_files={
    'train': 'theca/swedish/ma_data/splits/queries_train.jsonl.gz',
    'test': 'theca/swedish/ma_data/splits/queries_test.jsonl.gz',
    'validation': 'theca/swedish/ma_data/splits/queries_validation.jsonl.gz'
})
queries_train = queries["train"]
queries_test = queries["test"]
queries_val = queries["validation"]

items = load_dataset('json', data_files={
    'train': 'theca/swedish/ma_data/splits/items_train.jsonl.gz',
    'test': 'theca/swedish/ma_data/splits/items_test.jsonl.gz',
    'validation': 'theca/swedish/ma_data/splits/items_validation.jsonl.gz'
})
items_train = items["train"]
items_test = items["test"]
items_val = items["validation"]

# PRODUCTS
all_items = concatenate_datasets([items_train, items_test, items_val])
cat_items = all_items.map(lambda item: {'category': None if item["category"] and item["category"][0].startswith("varumarke") else item["category"]})

import random
import numpy as np

def random_cat(categories):
    if not categories:
        return None  # Handle empty lists

    n = len(categories)
    mid = (n - 1) / 2  # Middle index (float for symmetry)

    # Generate weights using a Gaussian centered at the middle
    weights = [np.exp(-((i - mid) ** 2) / (2 * (n / 4) ** 2)) for i in range(n)]

    # Normalize weights to sum to 1
    weights = np.array(weights) / sum(weights)

    return random.choices(categories, weights=weights, k=1)[0]

random_items = cat_items.map(lambda item: {'category': random_cat(item["category"]) if item["category"] else None})

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
    fields = [example["name"], example["category"], example["brand"], example["description"]]
    #fields = [example["product_title"], example["product_description"]]
    filtered_fields = [strip_html(f) for f in fields if f]  # Remove HTML and None values
    example["combined_text"] = " ".join(filtered_fields).strip()
    return example

products_ds = random_items.map(concatenate_fields)

# QUERIES 
all_queries = concatenate_datasets([queries_train, queries_test, queries_val])
seen = set()
unique_queries = all_queries.filter(lambda example: example["query_id"] not in seen and not seen.add(example["query_id"]))

# QREL
all_clicks = concatenate_datasets([clicks_train, clicks_test, clicks_val])

seen = set()
unique_clicks = all_clicks.filter(lambda item: item["query_id"]+str(item["item_id"]) not in seen and not seen.add(item["query_id"]+str(item["item_id"])))

qrel_ds = unique_clicks.remove_columns(["timestamp", "split"])
query_ds = unique_queries.remove_columns(["timestamp", "split"])
corpus_ds = products_ds.remove_columns(['name', 'description', 'brand', 'url', 'seller', 'price', 'price_currency', 'availability', 'category', 'mpn', 'sku', 'split'])

# remove long query
query_ds= query_ds.filter(lambda example: example["query_id"] != "01922f9f-3f41-bda5-2999-df9ade39b1ec")
qrel_ds = qrel_ds.filter(lambda example: example["query_id"] != "01922f9f-3f41-bda5-2999-df9ade39b1ec")

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

qrel_ds_train.save_to_disk("theca_qrel_ds_train")
qrel_ds_test.save_to_disk("theca_qrel_ds_test")
qrel_ds_val.save_to_disk("theca_qrel_ds_val")

query_ds_train.save_to_disk("theca_query_ds_train")
query_ds_test.save_to_disk("theca_query_ds_test")
query_ds_val.save_to_disk("theca_query_ds_val")

corpus_ds.save_to_disk("theca_corpus_ds")