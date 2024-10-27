"""
This module contains optimized utility functions for document and term similarity analysis.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def analyze_document_similarity(
    similar_indices: list,
    docs: list,
    doc_labels: list,
    class_labels: list,
    similarity_matrix: np.ndarray,
    reference_idx: int,
    num_similar: int = 5
    ):
    """
    Analyze the similarity between the reference document and other documents.

    Parameters:
    - similar_indices: List of indices of the most similar documents.
    - docs: List of all documents.
    - doc_labels: List of labels corresponding to each document.
    - class_labels: List of class names corresponding to each label.
    - similarity_matrix: Cosine similarity matrix of the documents.
    - reference_idx: Index of the reference document to analyze.
    - num_similar: Number of most similar documents to display. Default is 5.
    """
    
    print(f"\nTop {num_similar} Similar Documents:\n{'-'*50}")

    for idx in similar_indices:
        print(f"\n{'='*50}")
        print(f"Analyzing Document {idx}")
        print(f"{'='*50}")
        print(f"Content:\n{docs[idx][:200]}\n")
        print(f"Document Class: {class_labels[doc_labels[idx]]}")
        print(f"Reference Document Class: {class_labels[doc_labels[reference_idx]]}")
        print(f"{'='*50}")

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def find_top_similar_terms(query_term, similarity_matrix, term_list, top_n=5):
    """
    Retrieve the most similar terms to a given query term using the similarity matrix.

    Parameters:
    - query_term (str): The term to find similar terms for.
    - similarity_matrix (np.ndarray): Precomputed similarity matrix.
    - term_list (list): List of terms corresponding to the rows/columns of the matrix.
    - top_n (int): Number of top similar terms to return. Default is 5.

    Returns:
    - top_terms (list): A list of tuples with the most similar terms and their similarity scores.
    """
    # Convert term_list to a list if it's a NumPy array
    if isinstance(term_list, np.ndarray):
        term_list = term_list.tolist()

    # Find the index of the query term
    try:
        term_idx = term_list.index(query_term)
    except ValueError:
        raise ValueError(f"The term '{query_term}' is not in the term list.")

    # Directly use the term vector from the matrix
    term_vector = similarity_matrix[term_idx]

    # Compute similarities and get the top N similar terms
    similarities = cosine_similarity(term_vector.reshape(1, -1), similarity_matrix)[0]
    top_indices = np.argsort(similarities)[::-1][1:top_n + 1]
    
    top_terms = [(term_list[i], similarities[i]) for i in top_indices]
    return top_terms

