from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# app.py lives in src/; the index and images live under the repo root.
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from search import ImageSearchEngine


DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"


st.set_page_config(
    page_title="Visual Similarity Search",
    page_icon="🔍",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading CLIP model and index...")
def load_search_engine(index_directory: str) -> ImageSearchEngine:
    """Load the engine once and reuse it across reruns."""
    return ImageSearchEngine(Path(index_directory))


def resolve_image_path(stored_path: str) -> Path:
    """Metadata paths may be relative to the project root."""
    path = Path(stored_path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def render_results(
    results: list[dict],
    min_score: float = 0.0,
    columns_per_row: int = 4,
) -> None:
    results = [r for r in results if r["similarity_score"] >= min_score]
    if not results:
        st.warning("No results above the minimum similarity.")
        return

    for row_start in range(0, len(results), columns_per_row):
        row = results[row_start : row_start + columns_per_row]
        columns = st.columns(columns_per_row)

        for column, result in zip(columns, row):
            image_path = resolve_image_path(result["image_path"])
            score = result["similarity_score"]

            with column:
                if image_path.exists():
                    st.image(
                        str(image_path),
                        use_container_width=True,
                        caption=f"{score:.4f} — {image_path.name}",
                    )
                else:
                    st.error(f"Missing file:\n\n`{result['image_path']}`")


st.title("Visual Similarity Search")
st.caption("Search a local image collection by description or by example image.")

with st.sidebar:
    st.header("Settings")

    index_directory = st.text_input(
        "Index directory",
        value=str(DEFAULT_INDEX_DIRECTORY),
    )
    top_k = st.slider("Results to return", min_value=1, max_value=24, value=8)
    min_score = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
        help="Hide results scoring below this value.",
    )

try:
    search_engine = load_search_engine(index_directory)
except FileNotFoundError:
    st.error(
        "No index found. Build one first:\n\n"
        "```bash\npython src/indexer.py\n```"
    )
    st.stop()
except ValueError as error:
    st.error(f"The index could not be loaded: {error}")
    st.stop()

with st.sidebar:
    metadata = search_engine.metadata
    st.divider()
    st.metric("Indexed images", metadata["image_count"])
    st.write(f"**Model:** `{metadata.get('model_name', 'unknown')}`")
    st.write(f"**Weights:** `{metadata.get('pretrained', 'unknown')}`")
    st.write(f"**Dimensions:** {metadata['embedding_dimension']}")

text_tab, image_tab = st.tabs(["Search by text", "Search by image"])

with text_tab:
    query = st.text_input(
        "Describe what you're looking for",
        placeholder="a dog running on the beach",
    )

    if st.button("Search", key="text_search", type="primary"):
        if not query.strip():
            st.warning("Enter a description first.")
        else:
            with st.spinner("Searching..."):
                results = search_engine.search_text(query, top_k)
            render_results(results, min_score)

with image_tab:
    uploaded_file = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded_file is not None:
        preview_column, _ = st.columns([1, 3])
        with preview_column:
            st.image(uploaded_file, caption="Query image", use_container_width=True)

        if st.button("Find similar", key="image_search", type="primary"):
            suffix = Path(uploaded_file.name).suffix or ".png"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(uploaded_file.getbuffer())
                temporary_path = Path(handle.name)

            try:
                with st.spinner("Searching..."):
                    results = search_engine.search_image(temporary_path, top_k)
                render_results(results, min_score)
            finally:
                temporary_path.unlink(missing_ok=True)
