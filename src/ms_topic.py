"""Compute topic representation using c-TF-IDF and compute PMI."""

import numpy as np
import pandas as pd

from scipy.sparse import csr_array
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm


def compute_pmi(
    document_term_matrix,
    topic_representation_ids,
    topic_counts,
    n_documents=None,
    n_words_total=None,
):
    """
    Compute the Pointwise Mutual Information (PMI) score for a topic model.

    Parameters:
        document_term_matrix (csr_matrix): The document-term matrix.
        topic_representation_ids (ndarray): Array of top word indices for each topic.
        topic_counts (ndarray): Array of document counts for each topic.
        n_documents (int): Total number of documents.
        n_words_total (int): Total number of words in the document-term matrix.

    Returns:
        float: The PMI score for the topic model.
    """
    if n_words_total is None:
        # Compute the total number of words if not provided
        n_words_total = document_term_matrix.sum()
    if n_documents is None:
        # Compute the number of documents if not provided
        n_documents = document_term_matrix.shape[0]

    # Number of representative words per topic
    top_n = len(topic_representation_ids[0])

    # Restrict document-term matrix to the topic representation words
    topic_representation_ids_all = np.unique(topic_representation_ids)
    restricted_document_term_matrix = document_term_matrix[
        :, topic_representation_ids_all
    ]

    # Compute co-occurrence matrix
    cooccurrence = restricted_document_term_matrix.T @ restricted_document_term_matrix

    # Compute joint probability
    p_joint = cooccurrence.toarray() / n_words_total

    # Compute marginal probabilities
    p = cooccurrence.diagonal() / n_words_total

    # Compute PMI
    pmi = np.zeros_like(p_joint)
    for i in range(p_joint.shape[0]):
        for j in range(i + 1, p_joint.shape[0]):
            if p_joint[i, j] > 0:  # Avoid log of zero
                pmi[i, j] = np.log2(p_joint[i, j]) - np.log2(p[i]) - np.log2(p[j])
                pmi[j, i] = pmi[i, j]

    # Compute the PMI score for the topic model
    pmi_topics_score = 0
    for k, topic_count in enumerate(topic_counts):
        pmi_indices = np.argwhere(
            np.isin(topic_representation_ids_all, topic_representation_ids[k])
        ).flatten()
        pmi_topic_values = []

        for i in range(top_n):
            for j in range(i + 1, top_n):
                if i < len(pmi_indices) and j < len(
                    pmi_indices
                ):  # Ensure indices are valid
                    pmi_topic_values.append(pmi[pmi_indices[i], pmi_indices[j]])

        if pmi_topic_values:  # Avoid empty lists
            pmi_topics_score += (topic_count / n_documents) * np.median(
                pmi_topic_values
            )

    return pmi_topics_score


def ms_topic_representation(
    documents,
    ms_results,
    top_n=10,
    ngram_range=(1, 1),
    max_df=0.8,
    min_df=10,
    language="english",
    sort_topics=True,
):
    """
    Compute topic representation using c-TF-IDF for multiple sets of topic IDs and compute PMI.

    This function generates a representation of topics from a collection of documents
    by calculating the class-based Term Frequency-Inverse Document Frequency (c-TF-IDF).
    It identifies the most representative words for each topic and compiles topic-related
    information, including topic names, word representations, and counts. Additionally,
    it computes the Pointwise Mutual Information (PMI) score for each topic model.

    Parameters:
        documents (list of str): A list of text documents.
        ms_results (dict): A dictionary containing Markov stability results. It must include
                           a key "community_id" with a list of arrays, where each array
                           represents topic IDs for a different grouping.
        top_n (int, optional): The number of top words to include in the topic representation. Default is 10.
        ngram_range (tuple, optional): The range of n-grams to consider for tokenization. Default is (1, 1).
        max_df (float, optional): Maximum document frequency for filtering terms. Default is 0.8.
        min_df (int, optional): Minimum document frequency for filtering terms. Default is 10.
        language (str, optional): The language for stop words. Default is "english".
        sort_topics (bool, optional): If True, sort topics by size. Default is True.


    Returns:
        dict: The updated `ms_results` dictionary with the following keys added:
            - "topic_info": A list of DataFrames, where each DataFrame corresponds to a set of topic IDs
                            and contains the following columns:
                - "Topic": The topic ID.
                - "Count": The number of documents assigned to the topic.
                - "Name": A string representation of the topic, including the top 3 words.
                - "Representation": A list of the top `top_n` words for each topic.
            - "pmi": A list of PMI scores, where each score corresponds to a set of topic IDs.
    """
    # Extract topic IDs from Markov stability results
    topic_ids_unsorted = ms_results["community_id"]

    if sort_topics:
        # Sort topic IDs by size
        topic_ids = []

        for topic_id_unsorted in topic_ids_unsorted:
            n_topics = len(np.unique(topic_id_unsorted))
            topic_count = np.bincount(topic_id_unsorted)
            topic_new_id_dict = {
                np.argsort(-topic_count)[j]: j for j in range(n_topics)
            }
            topic_id_sorted = np.array(
                [topic_new_id_dict[j] for j in topic_id_unsorted]
            )
            topic_ids.append(topic_id_sorted)

        topic_ids = np.array(topic_ids)
        ms_results["community_id"] = topic_ids
    else:
        topic_ids = np.array(topic_ids_unsorted)

    # Number of documents
    n_documents = len(documents)

    # Compute document-term matrix
    vectorizer = CountVectorizer(
        stop_words=language, ngram_range=ngram_range, max_df=max_df, min_df=min_df
    )
    document_term_matrix = vectorizer.fit_transform(documents)
    words = vectorizer.get_feature_names_out()

    # Compute the total number of words
    n_words_total = document_term_matrix.sum()

    # Initialise topic info list and pmi
    topic_infos = []
    pmi_topic_scores = []

    # Iterate over each set of topic IDs
    for topic_id in tqdm(topic_ids):
        ################################
        # Compute topic representation #
        ################################

        # Number topics
        n_topics = len(np.unique(topic_id))

        # Define cluster / topic indicator matrix for topics
        row = np.arange(n_documents)
        col = topic_id
        data = np.ones(n_documents)
        topic_indicator_matrix = csr_array(
            (data, (row, col)), shape=(n_documents, n_topics), dtype=int
        )

        # Compute term frequency for topics
        ctf = topic_indicator_matrix.T @ document_term_matrix

        # Compute document (topic) frequency
        cdf = ctf.sum(axis=0)

        # Compute average number of words per topic
        avg_nr_samples = int(ctf.sum(axis=1).mean())
        cidf = np.log((avg_nr_samples / cdf) + 1)

        # Compute c-TF-IDF
        ctfidf = ctf * cidf

        # Find the top_n words with highest c-TF-IDF score per class
        topic_representation_ids = np.argsort(-ctfidf.toarray())[:, :top_n]
        topic_representation_words = words[topic_representation_ids]

        # Define topic names
        topic_names = [
            f"{j}_{topic_representation_words[j][0]}_{topic_representation_words[j][1]}_{topic_representation_words[j][2]}"
            for j in range(n_topics)
        ]

        # Get topics count
        topic_counts = topic_indicator_matrix.sum(axis=0)

        # Compile topic info
        topic_info = pd.DataFrame.from_dict(
            {
                "Topic": np.arange(n_topics),
                "Count": topic_counts,
                "Name": topic_names,
                "Representation": list(topic_representation_words),
            }
        )

        # Store the summary for the current topic_id array
        topic_infos.append(topic_info)

        ###############
        # Compute PMI #
        ###############
        pmi_topics_score = compute_pmi(
            document_term_matrix,
            topic_representation_ids,
            topic_counts,
            n_documents,
            n_words_total,
        )

        # Store the PMI score for the current topic_id array
        pmi_topic_scores.append(pmi_topics_score)

    # Append topic info and pmi to ms_results dictionary
    ms_results["topic_info"] = topic_infos
    ms_results["pmi"] = pmi_topic_scores

    return ms_results
