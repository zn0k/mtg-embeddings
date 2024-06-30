import argparse
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from create_embeddings import init_embedding_model, create_embeddings

def load_embeddings_from_parquet(file_path):
    table = pq.read_table(file_path)
    df = table.to_pandas()
    return df

def get_card_embedding(df, target_name):
    # Find the target item
    target_item = df[df['name'] == target_name]
    if target_item.empty:
        raise ValueError(f"Item with name '{target_name}' not found in the dataset.")    
    return target_item.iloc[0, 1:].values.reshape(1, -1) 

def find_most_similar_items(df, target_embedding, top_n=10, exclude_self=True):
    # Compute cosine similarity between the target item and all other items
    embeddings = df.iloc[:, 1:].values  # All embeddings
    similarity_scores = cosine_similarity(target_embedding, embeddings)[0]

    # Get the top N most similar items (excluding the target item itself)
    start = 1 if exclude_self else 0
    top_indices = np.argsort(similarity_scores)[::-1][start:top_n+1]
    top_items = df.iloc[top_indices]
    top_scores = similarity_scores[top_indices]

    return top_items, top_scores

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Find similar MTG cards.")
    parser.add_argument("--card", type=str, help="Exact name of the card. This or --query is required.")
    parser.add_argument("--query", type=str, help="Free form query. This or --card is required.")
    parser.add_argument("--parquet", type=str, required=True, help="Path to the Parquet file containing card embeddings.")
    parser.add_argument("--top", type=int, default=10, help="Top N items, defaults to 10.")
    args = parser.parse_args()

    # Load embeddings from Parquet file
    df = load_embeddings_from_parquet(args.parquet)

    if args.card:
        target_embedding = get_card_embedding(df, args.card)
        exclude_self = True
    elif args.query:
        # Initialize embedding model and tokenizer
        tokenizer, model = init_embedding_model()
        # Create query embedding
        target_embedding = create_embeddings(tokenizer, model, [args.query])
        exclude_self = False
    else:
        raise("--card or --query must be specified.")

    # Find the top 10 most similar items to "lightning bolt"
    top_items, top_scores = find_most_similar_items(df, target_embedding, args.top, exclude_self)
    
    # Print the results
    print(f"Top {args.top} items similar to '{args.card}':")
    for idx, (index, row) in enumerate(top_items.iterrows()):
        print(f"Name: {row['name']}, Similarity: {top_scores[idx]}")

if __name__ == "__main__":
    main()
