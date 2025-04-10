import scann
import numpy as np
import pickle  # For saving the ScaNN searcher

# Normalize if using cosine similarity (same as FAISS with IndexFlatIP)
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)
query_embeddings = query_embeddings / np.linalg.norm(query_embeddings, axis=1, keepdims=True)

# Set search parameters
k = 100  # Number of nearest neighbors

# Build the ScaNN searcher
searcher = scann.scann_ops_pybind.builder(corpus_embeddings, k, "dot_product") \
    .score_ah(2, anisotropic_quantization_threshold=0.2) \
    .reorder(100) \
    .build()

# Save the searcher (to disk)
with open(prestring + "_scann_searcher.pkl", "wb") as f:
    pickle.dump(searcher, f)

# Perform the search
neighbors, scores = searcher.search_batched(query_embeddings)