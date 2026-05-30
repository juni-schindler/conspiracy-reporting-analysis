# Tracing Characteristics and Dynamics of Conspiracy-Related Reporting in US Far-Right Hyperpartisan and Legacy Media

Code repository for the paper:

> Annett Heft, Juni Schindler, Kilian Buehling, Curd Benjamin Knüpfer.
> *Tracing Characteristics and Dynamics of Conspiracy-Related Reporting in US Far-Right Hyperpartisan and Legacy Media.*
> [DOI to be added]

---

## Overview

This repository provides the code for a multiscale topic modelling analysis of news articles from far-right hyperpartisan and legacy US media outlets (2011–2021). The pipeline produces document embeddings, constructs similarity graphs, runs Markov stability community detection across scales, and performs statistical comparisons between outlet types.

**Media outlets:**
- *Far-right Hyperpartisan:* InfoWars, Breitbart, The Daily Caller, The Gateway Pundit, The Blaze
- *Legacy:* The New York Times, The Washington Post, USA Today, The Wall Street Journal

---

## Repository structure

```
├── 1_nlp_pipeline.py        # Text preprocessing and document embedding
├── 2_ms_pipeline.py         # Graph construction and multiscale topic modelling
├── 3_results_analysis.ipynb # Scale selection, statistics, and figures
├── src/
│   ├── doc_embedding.py     # DocumentEmbedder class
│   └── ms_topic.py          # c-TF-IDF topic representation and PMI scoring
├── data/
│   ├── raw/                 # Raw corpus (not included, see Data section)
│   └── processed/           # Preprocessed corpus (not included, see Data section)
├── results/
│   ├── embeddings/          # Document embeddings (.npy)
│   ├── entities/            # Person entities (.xlsx)
│   ├── ms/                  # Multiscale analysis results (.pkl)
│   └── topics/              # Topic labels and significance tables (.xlsx)

└── figures/                 # Publication figures (SVG, HTML)
```

---

## Data

The raw corpus is not included in this repository. Data collection methodology and source information are documented on OSF: [link to be added upon publication]. This documentation can be used to reconstruct the dataset. 
Place the collected data at `data/raw/alt_leg_data_v1.csv` before running the pipeline.

---

## Installation

Requires Python 3.10+.

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

GPU acceleration is supported on NVIDIA (CUDA) and Apple Silicon (MPS) and will be used automatically if available.

---

## Running the pipeline

Run the three steps in order:

```bash
# 1. Preprocess text and generate document embeddings
python 1_nlp_pipeline.py

# 2. Build graphs and run multiscale topic modelling
#    (computationally intensive — uses all available CPU cores)
python 2_ms_pipeline.py

# 3. Scale selection, statistical analysis, and figures
#    Open and run 3_results_analysis.ipynb
```

**Note on reproducibility:** The Leiden community detection algorithm used in step 2 is stochastic. Results may differ slightly across runs. The analysis notebook (`3_results_analysis.ipynb`) was run on the specific set of results included in `results/ms/`.

---

## License

This code is released under the GNU General Public License v3.0. See [LICENSE](LICENSE) for details.
