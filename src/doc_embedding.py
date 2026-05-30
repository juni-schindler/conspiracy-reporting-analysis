import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List
import nltk
from tqdm.auto import tqdm

# Download nltk data for sentence splitting
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class DocumentEmbedder:
    def __init__(self, model_name: str = "all-mpnet-base-v2", batch_size: int = 32):
        """
        Initialize the document embedder.

        Args:
            model_name: Name of the sentence-transformers model to use
            batch_size: Batch size for processing sentences
        """
        # Check if GPU is available (CUDA for NVIDIA, MPS for Apple Silicon)
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using device: {self.device}")

        # Load the model
        self.model = SentenceTransformer(model_name)
        self.model = self.model.to(self.device)

        self.batch_size = batch_size
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

    def embed_documents(
        self, documents: List[str], strategy: str = "mean", show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of documents.

        Args:
            documents: List of document texts
            strategy: Aggregation strategy ('mean' or 'max')
            show_progress: Whether to show progress bar

        Returns:
            numpy array of document embeddings
        """
        # Split documents into sentences
        all_sentences = []
        doc_sentence_counts = []

        for doc in documents:
            sentences = nltk.sent_tokenize(doc)
            all_sentences.extend(sentences)
            doc_sentence_counts.append(len(sentences))

        # Generate embeddings for all sentences
        sentence_embeddings = []

        # Process sentences in batches
        for i in tqdm(
            range(0, len(all_sentences), self.batch_size),
            disable=not show_progress,
            desc="Generating embeddings",
        ):
            batch = all_sentences[i : i + self.batch_size]
            batch_embeddings = self.model.encode(
                batch, convert_to_tensor=True, device=self.device
            )
            sentence_embeddings.append(batch_embeddings.cpu().numpy())

        # Concatenate all embeddings
        sentence_embeddings = np.vstack(sentence_embeddings)

        # Aggregate sentence embeddings into document embeddings
        doc_embeddings = []
        start_idx = 0

        for count in doc_sentence_counts:
            end_idx = start_idx + count
            doc_sentences = sentence_embeddings[start_idx:end_idx]

            if strategy == "mean":
                doc_embedding = np.mean(doc_sentences, axis=0)
            elif strategy == "max":
                doc_embedding = np.max(doc_sentences, axis=0)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            doc_embeddings.append(doc_embedding)
            start_idx = end_idx

        return np.array(doc_embeddings)

    def get_embedding_dimension(self) -> int:
        """Return the dimension of the embeddings."""
        return self.embedding_dim
