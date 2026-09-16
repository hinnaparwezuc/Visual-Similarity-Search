## Current Status

The project currently includes the core CLIP embedding engine.

### Implemented

- Image embedding generation using OpenCLIP
- Text embedding generation using OpenCLIP
- Shared embedding space for text and image comparison
- L2-normalized vectors for cosine-similarity search
- Automatic CPU or CUDA device selection
- Batch image processing
- Support for JPG, PNG, and other Pillow-compatible image formats

### In Progress

- FAISS image indexing
- Text-to-image similarity search
- Image-to-image similarity search
- Search result ranking and metadata
- Interactive user interface

## How It Works

CLIP maps images and text into the same numerical embedding space. Semantically related images and text descriptions produce vectors that are close together.

The current `CLIPEmbedder` class:

1. Loads a pretrained OpenCLIP model.
2. Preprocesses images or tokenizes text.
3. Generates numerical feature vectors.
4. Normalizes the vectors for cosine-similarity comparison.
5. Returns NumPy arrays that can be stored in a vector index.

## Project Structure

```text
visual-similarity-search/
├── data/
│   └── images/
├── src/
├── embedder.py
├── requirements.txt
├── .gitignore
└── README.md
