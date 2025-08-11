from datasets import load_from_disk
from transformers import AutoTokenizer
import pickle

# Load dataset
prestring = "amazon"
data = load_from_disk(prestring + "_corpus_ds")

# Load tokenizer
MODEL_ID = "Snowflake/snowflake-arctic-embed-m-v2.0"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Setup for chunked processing
queries = data["combined_text"]
n = len(queries)
chunks = 10
chunk_size = n // chunks

ntokens = []

for i in range(chunks):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < chunks - 1 else n
    chunk = queries[start:end]
    
    tokens = tokenizer(chunk, padding=False, truncation=False, max_length=5192)
    input_ids = tokens["input_ids"]
    ntokens.extend([len(toks) for toks in input_ids])

# Save token counts
with open("number_tokens.pkl", "wb") as f:
    pickle.dump(ntokens, f)

# Print stats
print(f"Mean: {sum(ntokens)/len(ntokens)}")
print(f"Max: {max(ntokens)}")