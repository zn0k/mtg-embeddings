import argparse
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import json
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm



# Load your structured JSON data
def load_json_data(file_path):
    with open(file_path, "r") as file:
        data = json.load(file)
    return list(data["data"].values())

# Extract the text data and names you want to embed
# The names are stored separately so they can be added as a column into the database file
# allowing for later retrieval of embeddings for a specific card
def extract_texts_and_names(data):
    def textify(card):
        fields = []
        for key in ["name", "text", "type", "manaCost", "convertedManaCost"]:
            if key in card:
                fields.append(f"{key}: {card[key]}")
        for key in ["colorIdentity", "subtypes", "supertypes"]:
            if key in card and len(card[key]):
                fields.append(f"{key}: " + "|".join(card[key]))
        text = ", ".join(fields)
        return text

    texts = [textify(item[0]) for item in tqdm(data, desc="Extracting card descriptions")] # extract texts to turn into embeddings
    names = [item[0]["name"] for item in tqdm(data, desc="Extracting card names as keys")]  # just the card names
    return texts, names

# Initialize the model and tokenizer
def init_embedding_model():
    tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    model = AutoModel.from_pretrained("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True)
    return tokenizer, model

# Normalize a vector
def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

# Create embeddings
def create_embeddings(tokenizer, model, texts):
    embeddings = []
    for text in tqdm(texts, desc="Creating embeddings"):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        pooled_embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        normalized_embedding = normalize_vector(pooled_embedding)
        embeddings.append(normalized_embedding)
    return embeddings

# Save embeddings to a Parquet file
def save_embeddings_to_parquet(names, embeddings, file_path):
    df = pd.DataFrame(embeddings)
    df.insert(0, "name", names)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, file_path)

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Create embeddings from mtgjson.com data and save to parquet.")
    parser.add_argument("--json", type=str, required=True, help="Path to the input JSON file. Required.")
    parser.add_argument("--parquet", type=str, help="Path to the output Parquet file. If not provided, the same base file name as for the JSON file will be used")
    args = parser.parse_args()
    if not args.parquet:
        args.parquet = args.json
        args.parquet.replace("json", "parquet")

    # Load data
    data = load_json_data(args.json)
    
    # Extract texts
    texts, names = extract_texts_and_names(data)
    
    # Initialize embedding model and tokenizer
    tokenizer, model = init_embedding_model()
    
    # Create embeddings
    embeddings = create_embeddings(tokenizer, model, texts)
    
    # Save to Parquet
    save_embeddings_to_parquet(names, embeddings, args.parquet)
    
if __name__ == "__main__":
    main()
