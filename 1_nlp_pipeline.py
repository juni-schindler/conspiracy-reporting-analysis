import pandas as pd
import re
import spacy
import time
import numpy as np

from tqdm import tqdm

from src.doc_embedding import DocumentEmbedder

# Load the English language model
nlp = spacy.load("en_core_web_lg")

def preprocess_text(text):
    """
    Preprocess the given text using spaCy and extract entities.
    """

    # Remove \r\n and other whitespace characters
    text = text.replace('\r', '').replace('\n', '').strip()

    # Remove additional whitespace
    text = re.sub(r'\s+', ' ', text) 

    # Process the text with spaCy
    doc = nlp(text)

    # Extract entities and store them in separate lists
    person_entities = []
    for ent in doc.ents:
        if ent.label_ == 'PERSON':
            person_entities.append(ent.text)


    # Remove stopwords and perform lemmatization
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    preprocessed_text = " ".join(tokens)

    # Create results dictionary
    result = {
        "preprocessed_text": preprocessed_text,
        "person_entities": list(set(person_entities))
    }

    return result

if __name__ == "__main__":

    # Load raw documents
    df = pd.read_csv("data/raw/alt_leg_data_v1.csv")

    # Run pre-processing pipeline
    tqdm.pandas(desc="Processing Documents")
    results = df['text_prep'].progress_apply(preprocess_text)

    # Store results in new columns
    df['preprocessed_text'] = results.apply(lambda x: x['preprocessed_text'])
    df['person_entities'] = results.apply(lambda x: x['person_entities'])
    
    # Save dataframe to csv
    df.to_csv("data/processed/alt_leg_data_v1_spacy_prep_ent.csv")

    # Extract preprocessed documents for embedding
    documents = list(df["preprocessed_text"])

    # Initialize embedder
    embedder = DocumentEmbedder(batch_size=32)

    # Generate embeddings
    start_time = time.time()
    embeddings = embedder.embed_documents(documents, strategy='mean')
    end_time = time.time()

    # Print results
    print(f"\nEmbedding dimension: {embedder.get_embedding_dimension()}")
    print(f"Number of documents processed: {len(documents)}")
    print(f"Shape of embeddings: {embeddings.shape}")
    print(f"Processing time: {end_time - start_time:.2f} seconds")

    # Store embeddings
    np.save("results/embeddings/alt_leg_data_v1_all-mpnet-base-v2_mean.npy", embeddings)