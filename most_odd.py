import pandas as pd
import pyarrow.parquet as pq
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

table = pq.read_table("AtomicCards.parquet")
df = table.to_pandas()

names = df["name"].values
embeddings = df.iloc[:, 1:].values

matrix = cosine_similarity(embeddings)

avg_similarities = np.mean(matrix, axis=1)
least_similar = np.argsort(avg_similarities)[:25]
least_similar_names = names[least_similar]
least_similar_scores = avg_similarities[least_similar]

for name, score in zip(least_similar_names, least_similar_scores):
    print(f"{name}, {score}")