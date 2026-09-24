# Visual Similarity Search

Search from a local image collection by text description or by example image, using
[OpenCLIP](https://github.com/mlfoundations/open_clip) embeddings and a
[FAISS](https://github.com/facebookresearch/faiss) vector index. Use it from the
command line or through a Streamlit web app.

## How It Works

CLIP maps images and text into the same numerical embedding space, so
semantically related images and text descriptions produce vectors that sit close
together. Searching is then just a nearest-neighbour lookup.

1. `CLIPEmbedder` loads a pretrained OpenCLIP model and generates L2-normalized
   feature vectors for images or text.
2. `indexer.py` embeds every image in a directory and writes a FAISS index plus a
   metadata file mapping index positions back to file paths. The metadata also
   records which model built the index.
3. `search.py` embeds your query with that same model, searches the index by
   inner product (which equals cosine similarity for normalized vectors), and
   returns ranked results.
4. `app.py` wraps the search engine in a Streamlit interface.

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

Create `data/images/` and drop your images into it. Subdirectories are searched
recursively. Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`.

### 2. Build the index

```bash
python src/indexer.py
```

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--images` | `data/images` | Directory containing the source images |
| `--output` | `data/index` | Where the index and metadata are written |
| `--batch-size` | `32` | Images processed per batch |
| `--model` | `ViT-B-32` | OpenCLIP model architecture |
| `--pretrained` | `laion2b_s34b_b79k` | OpenCLIP pretrained weights |

This writes `images.faiss` and `metadata.json` into the output directory.
Unreadable or corrupt images are skipped and listed at the end instead of
stopping the build.

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
| `--include-self` | off | Keep the query image in results if it is part of the index |

Results are printed as JSON, each entry containing `image_path` and
`similarity_score` (higher is more similar; 1.0 is an exact match).

### 4. Search in the browser

```bash
streamlit run src/app.py
```

The app offers text and image search tabs, shows results as an image grid, and
lets you adjust the number of results and a minimum similarity score from the
sidebar.

## Project Structure

```text
Visual-Similarity-Search/
├── data/
│   ├── images/          # your source images (gitignored)
│   └── index/           # generated FAISS index (gitignored)
├── src/
│   ├── __init__.py
│   ├── app.py           # Streamlit web interface
│   ├── embedder.py      # CLIP embedding engine
│   ├── indexer.py       # builds the FAISS index
│   └── search.py        # queries the index
├── requirements.txt
├── .gitignore
└── README.md
```

## Status

### Implemented

- Image and text embedding generation using OpenCLIP
- Shared embedding space for text and image comparison
- L2-normalized vectors for cosine-similarity search
- Automatic CPU or CUDA device selection
- Batch image processing with a progress bar
- Corrupt images skipped during indexing
- FAISS index construction with image and model metadata
- Text-to-image and image-to-image search from the command line
- Interactive Streamlit interface

### Planned

- Approximate nearest-neighbour indexes for larger collections
- Incremental index updates without a full rebuild
- Near-duplicate detection
