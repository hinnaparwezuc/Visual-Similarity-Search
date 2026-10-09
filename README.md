# Visual Similarity Search

![Tests](https://github.com/hinnaparwezuc/Visual-Similarity-Search/actions/workflows/tests.yml/badge.svg)

Search a local image collection by text description or by example image, and
find near-duplicate photos, using [OpenCLIP](https://github.com/mlfoundations/open_clip)
embeddings and a [FAISS](https://github.com/facebookresearch/faiss) vector index.
Includes an HNSW approximate-search index written from scratch, with a
benchmark against FAISS. Use it from the command line or through a Streamlit
web app.

## How It Works

CLIP maps images and text into the same numerical embedding space, so
semantically related images and text descriptions produce vectors that sit close
together. Searching is then just a nearest-neighbour lookup.

1. `CLIPEmbedder` loads a pretrained OpenCLIP model and generates L2-normalized
   feature vectors for images or text.
2. `indexer.py` embeds every image in a directory and writes a FAISS index plus a
   metadata file mapping index positions back to file paths. The metadata also
   records which model built the index and each file's size and modification
   time, so later runs can update the index incrementally.
3. `search.py` embeds your query with that same model, searches the index by
   inner product (which equals cosine similarity for normalized vectors), and
   returns ranked results.
4. `duplicates.py` compares every indexed image against every other and groups
   the ones above a similarity threshold.
5. `app.py` wraps all of this in a Streamlit interface.
6. `hnsw.py` is a from-scratch implementation of Hierarchical Navigable Small
   World graphs, an approximate nearest-neighbour index, and
   `benchmarks/bench.py` measures it against exact search and FAISS's own HNSW.

## Requirements

- Python 3.9 or newer
- See `requirements.txt` for package versions

CUDA is used automatically when available; otherwise everything runs on CPU.

## Installation

```bash
git clone https://github.com/hinnaparwezuc/Visual-Similarity-Search.git
cd Visual-Similarity-Search

python -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Usage

### 1. Add images

Drop your images into `data/images/`. Subdirectories are searched recursively.
Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

### 2. Build the index

```bash
python src/indexer.py
```

After adding, editing or deleting images, update the index instead of
rebuilding it:

```bash
python src/indexer.py --update
```

This embeds only images that are new or have changed (detected by file size
and modification time) and drops deleted ones, so CLIP runs only on what
changed. If nothing changed, the model isn't even loaded.

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--images` | `data/images` | Directory containing the source images |
| `--output` | `data/index` | Where the index and metadata are written |
| `--batch-size` | `32` | Images processed per batch |
| `--model` | `ViT-B-32` | OpenCLIP model architecture |
| `--pretrained` | `laion2b_s34b_b79k` | OpenCLIP pretrained weights |
| `--update` | off | Only embed new or changed images and drop deleted ones |
| `--hnsw` | off | Also build an HNSW index for approximate search |
| `--hnsw-only` | off | Add an HNSW index to the existing index without re-running CLIP |
| `--hnsw-m` | `16` | HNSW neighbours per node |
| `--hnsw-ef-construction` | `200` | HNSW candidate pool size while building |

This writes `images.faiss` and `metadata.json` into the output directory.
Unreadable or corrupt images are skipped and listed at the end instead of
stopping the build.

With `--hnsw` or `--hnsw-only`, an `hnsw.pkl` graph is saved alongside them.
`--update` keeps an existing graph in sync: new images are added to it, and
since HNSW graphs can't delete nodes, a removal or edit rebuilds the graph from
the stored vectors (without re-running CLIP). A full rebuild without `--hnsw`
removes an old graph, since it would no longer match the images.

`--update` falls back to a full rebuild when there's no previous index, when
the previous index was built with a different model, or when it was built by
an older version of this project that didn't record file details.

### 3. Search from the command line

Search by text description:

```bash
python src/search.py --text "a dog running on the beach" --top-k 5
```

Search by example image:

```bash
python src/search.py --image path/to/query.jpg --top-k 5
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--text` | — | Text description to search for (mutually exclusive with `--image`) |
| `--image` | — | Query image to find visual matches for |
| `--top-k` | `5` | Number of results to return |
| `--index-directory` | `data/index` | Directory holding the index and metadata |
| `--backend` | `flat` | `flat` for exact search, `hnsw` for approximate search |
| `--ef-search` | `64` | HNSW candidates examined per query (higher is more accurate, slower) |
| `--include-self` | off | Keep the query image in results if it is part of the index |

Results are printed as JSON, each entry containing `image_path` and
`similarity_score` (higher is more similar; 1.0 is an exact match). The
index lookup time is printed after the results.

### 4. Find near-duplicates

```bash
python src/duplicates.py --threshold 0.95
```

Groups images that are nearly identical, such as resized copies, re-saved
exports and burst shots. Matches are chained, so if A matches B and B matches
C, all three land in one group. The output lists each group with every image's
similarity to the group's first image, and a summary of how many images could
be removed while keeping one of each. Lower the threshold to find looser
matches; `1.0` finds only identical images.

No CLIP model is needed: the vectors are read back out of the index.

### 5. Use the web app

```bash
streamlit run src/app.py
```

The app has three tabs: search by text, search by uploaded image, and find
duplicates. Results are shown as an image grid. From the sidebar you can set
the number of results and a minimum similarity score, and reload the index
after re-running `indexer.py`. If an HNSW index exists, the sidebar also lets
you switch between exact and approximate search and tune `efSearch`, and each
search shows how long the index lookup took.

## Benchmarks

Exact search compares a query against every vector, so its cost grows with the
collection. HNSW builds a layered proximity graph and searches by walking it,
touching only a small fraction of the vectors per query. `src/hnsw.py`
implements it from scratch, following Malkov & Yashunin (2016), including the
diverse-neighbour selection heuristic. Build it with `indexer.py --hnsw` and
search with it using `--backend hnsw` or the app's sidebar.

For small collections, exact search is faster: FAISS's flat index is highly
optimized C++, while `hnsw.py` is pure Python, and a graph only pays off once
there are too many vectors to compare against each one directly. The benchmark
shows where that crossover falls for your data.

After building the index, run:

```bash
python benchmarks/bench.py
```

The benchmark reads the stored embeddings back out of `data/index`, so CLIP does
not run again. It compares four approaches on the same queries: exact search in
NumPy, FAISS `IndexFlatIP` (what this project uses), FAISS `IndexHNSWFlat`, and
`hnsw.py`, sweeping the `efSearch` setting for both HNSW indexes.

| Column | Meaning |
| --- | --- |
| `build_time_s` | Time to build the index from the embeddings |
| `index_mb` | Serialized size of the finished index |
| `recall_at_k` | Share of the exact top-k the index returned (fidelity to exact search) |
| `label_acc_at_k` | Share of queries with a same-folder image in the top-k (only when images sit in class subfolders) |
| `p50_ms` / `p95_ms` | Median and 95th-percentile query latency |

Options: `--queries` (default 200), `--k` (default 3), `--ef-search` (default
`16 64 256`), `--threads` (FAISS threads, default 1 so all indexes are timed on
one core), and `--out` (default `benchmarks/results.json`).

Each query is an indexed image, evaluated leave-one-out: the query is removed
from its own results, as `search.py` does.

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

The tests cover the HNSW index (recall against exact search, saving and
loading, adding in batches), full and incremental indexing, both search
backends, duplicate grouping, and the Streamlit app, driven headlessly with
Streamlit's test runner.

They replace CLIP with a lightweight stand-in (`tests/fake_embedder.py`) that
embeds each image from its pixels, so identical images still get identical
vectors and edited copies get similar ones. That keeps the suite fast and means
it runs without PyTorch or downloading model weights. GitHub Actions runs it on
every push.

## Project Structure

```text
Visual-Similarity-Search/
├── .github/workflows/
│   └── tests.yml        # runs the tests on every push
├── benchmarks/
│   └── bench.py         # compares exact and approximate indexes
├── data/
│   ├── images/          # your source images (contents gitignored)
│   └── index/           # generated index files (contents gitignored)
├── src/
│   ├── __init__.py
│   ├── app.py           # Streamlit web interface
│   ├── duplicates.py    # near-duplicate finder
│   ├── embedder.py      # CLIP embedding engine
│   ├── hnsw.py          # HNSW index written from scratch
│   ├── indexer.py       # builds and updates the index
│   └── search.py        # queries the index
├── tests/
│   ├── conftest.py      # shared fixtures; swaps in the fake embedder
│   ├── fake_embedder.py # pixel-based stand-in for CLIP
│   └── test_*.py
├── pytest.ini
├── requirements.txt
├── requirements-dev.txt
├── .gitignore
└── README.md
```

## Features

- Image and text embedding generation using OpenCLIP, on CPU or CUDA
- Text-to-image and image-to-image search, from the command line and a
  Streamlit web app
- Batch indexing with a progress bar; corrupt images are skipped
- Incremental index updates that embed only new or changed images
- Near-duplicate detection with adjustable threshold
- HNSW approximate nearest-neighbour index, implemented from scratch, selectable
  as a search backend and kept in sync on updates
- Benchmark of recall, latency, build time and size across exact and HNSW
  indexes
- Test suite that runs without PyTorch, with continuous integration on GitHub
