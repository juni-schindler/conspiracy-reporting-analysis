import os
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pygenstability as pgs

from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

from src.ms_topic import ms_topic_representation


def _compute_CkNN(D, k=5, delta=1):
    """Computes CkNN graph."""
    # obtain rescaled distance matrix, see CkNN paper
    darray_n_nbrs = np.partition(D, k)[:, [k]]
    ratio_matrix = D / np.sqrt(darray_n_nbrs.dot(darray_n_nbrs.T))
    # threshold rescaled distance matrix by delta
    A = D * (ratio_matrix < delta)
    return A


if __name__ == "__main__":

    # load preprocessed documents
    df = pd.read_csv("data/processed/alt_leg_data_v1_spacy_prep_ent.csv")

    # load embeddings
    embeddings = np.load("results/embeddings/alt_leg_data_v1_all-mpnet-base-v2_mean.npy")

    # exclude left-leaning alternative outlets (out of scope for this study)
    considered_docs = np.array(~df["subplatform"].isin(["tpm", "occupyDems"]))
    # exclude empty doc with nan embedding
    embed_nan_ind = np.unique(np.argwhere(np.isnan(embeddings))[:, 0])
    considered_docs[embed_nan_ind] = False
    # restrict embeddings
    embeddings = embeddings[considered_docs]

    # normalize embeddings
    X = (embeddings.T / np.linalg.norm(embeddings, axis=1)).T

    # restrict documents
    df = df[considered_docs]
    documents = list(df["preprocessed_text"])

    # compute normalised distance matrix
    print("Compute distance matrix...")
    D = squareform(pdist(X, metric="euclidean"))
    np.save("results/distance/alt_leg_data_v1_all-mpnet-base-v2_mean_dist.npy", D)
    D_norm = D / np.amax(D)

    # compute normalised similarity matrix
    S = 1 - D_norm

    # compute minimum spanning tree for backbone construction
    print("Compute minimum spanning tree...")
    mst = minimum_spanning_tree(D_norm)
    np.save(
        "results/graphs/alt_leg_data_v1_all-mpnet-base-v2_mean_backbone_mst.npy",
        mst,
    )

    # Hyperparameter sweep over CkNN delta (connectivity threshold) and k
    # (neighbourhood size). Final values selected by PMI-based linear regression
    # in 3_results_analysis.ipynb.
    for delta in [1, 0.5, 0.6, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]:

        for k in range(7, 16):

            # sparsify distance matrix with CkNN
            print(f"Compute CkNN for k={k} and delta={delta}...")
            sparse = _compute_CkNN(D_norm, k, delta)
            sp.save_npz(
                f"results/graphs/alt_leg_data_v1_all-mpnet-base-v2_mean_backbone_cknn_k{k}_delta{delta}.npz",
                sp.csr_matrix(sparse),
            )

            # undirected distance backbone is given by sparse graph and MST
            backbone = np.array((mst + mst.T + sparse + sparse.T) > 0, dtype=int)

            # adjacency matrix has weights of similarity matrix
            A = sp.csr_matrix(S * backbone)

            # store sparse matrix
            sp.save_npz(
                f"results/graphs/alt_leg_data_v1_all-mpnet-base-v2_mean_cknn_k{k}_delta{delta}_mst.npz",
                A,
            )

            # parameters for MS analysis
            min_scale = -1
            max_scale = 2
            n_scale = 300
            n_workers = os.cpu_count()
            method = "leiden"
            result_file = f"results/ms/alt_leg_data_v1_all-mpnet-base-v2_mean_cknn-k{k}-delta{delta}-mst_ms{min_scale}min{max_scale}max{n_scale}n-{method}_topics_top10.pkl"

            # run MS analysis
            print(f"Run MS for k={k} and delta={delta}...")
            ms_results = pgs.run(
                A,
                min_scale=min_scale,
                max_scale=max_scale,
                n_scale=n_scale,
                n_workers=n_workers,
                with_optimal_scales=False,
                constructor="linearized",
                method=method,
                result_file=result_file,
            )

            # compute topic representation and PMI
            print(f"Compute PMI for k={k} and delta={delta}...")
            ms_results_topic = ms_topic_representation(documents, ms_results)

            # save results
            pgs.save_results(ms_results_topic, result_file)
