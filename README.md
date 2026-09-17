# Visual Similarity Search

Search a local image collection by text description or by example image, using
[OpenCLIP](https://github.com/mlfoundations/open_clip) embeddings and a
[FAISS](https://github.com/facebookresearch/faiss) vector index.

## How It Works

CLIP maps images and text into the same numerical embedding space, so
semantically related images and text descriptions produce vectors that sit close
together. Searching is then just a nearest-neighbour lookup.

1. `CLIPEmbedder` loads a pretrained OpenCLIP model and generates L2-normalized
   feature vectors for images or text.
2. `indexer.py` embeds every image in a directory and writes a FAISS index plus a
   metadata file mapping index positions back to file paths.
3. `search.py` embeds your query, searches the index by inner product (which
   equals cosine similarity for normalized vectors), and returns ranked results.

## Requirements

- Python 3.9 or newer
- See `requirements.txt` for package versions

CUDA is used automatically when available; otherwise everything runs on CPU.

## Installation

```bash
git clone https://github.com/<your-username>/Visual-Similarity-Search.git
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

Options:

| Flag | Default | Description |
| --- | --- | --- |
| `--images` | `data/images` | Directory containing the source images |
| `--output` | `data/index` | Where the index and metadata are written |
| `--batch-size` | `32` | Images processed per batch |

This writes `images.faiss` and `metadata.json` into the output directory.

### 3. Search

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

Results are printed as JSON, each entry containing `image_path` and
`similarity_score` (higher is more similar; 1.0 is an exact match).

## Project Structure

```text
Visual-Similarity-Search/
├── data/
│   ├── images/          # your source images (gitignored)
│   └── index/           # generated FAISS index (gitignored)
├── src/
│   ├── __init__.py
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
- Batch image processing
- FAISS index construction with image metadata
- Text-to-image and image-to-image search from the command line

### Planned

- Interactive Streamlit interface
- Approximate nearest-neighbour indexes for larger collections
- Incremental index updates without a full rebuild
