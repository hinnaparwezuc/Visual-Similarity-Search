from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st

# app.py lives in src/; the index and images live under the repo root.
SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from duplicates import DEFAULT_THRESHOLD, find_duplicate_groups
from search import ImageSearchEngine


DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
EXACT_MATCH_THRESHOLD = 0.999
MAX_DUPLICATE_GROUPS_SHOWN = 30


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


def show_timing(engine: ImageSearchEngine, backend: str) -> None:
    label = "HNSW" if backend == "hnsw" else "exact"
    st.caption(f"Index lookup: {engine.last_search_ms:.2f} ms ({label})")


def render_results(
    results: list[dict],
    min_score: float = 0.0,
    columns_per_row: int = 4,
    empty_message: str = "No results above the minimum similarity.",
) -> None:
    results = [r for r in results if r["similarity_score"] >= min_score]
    if not results:
        st.warning(empty_message)
        return

    for row_start in range(0, len(results), columns_per_row):
        row = results[row_start : row_start + columns_per_row]
        columns = st.columns(columns_per_row)

        for column, result in zip(columns, row):
            image_path = resolve_image_path(result["image_path"])
            score = result["similarity_score"]

            with column:
                if image_path.exists():
                    try:
                        st.image(
                            str(image_path),
                            width="stretch",
                            caption=f"{score:.4f} — {image_path.name}",
                        )
                    except Exception:
                        # A file changed or corrupted since indexing
                        # shouldn't take down the whole results grid.
                        st.error(f"Can't display:\n\n`{result['image_path']}`")
                else:
                    st.error(f"Missing file:\n\n`{result['image_path']}`")


st.title("Visual Similarity Search")
st.caption(
    "Search a local image collection by description or by example image, "
    "or find near-duplicates."
)

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

    # The engine is cached, so a rebuilt index isn't picked up automatically.
    if st.button("Reload index", help="Use after re-running indexer.py."):
        load_search_engine.clear()

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

    st.divider()
    backend_labels = {"flat": "Exact (FAISS)", "hnsw": "Approximate (HNSW)"}
    backend = st.radio(
        "Index type",
        options=list(backend_labels),
        format_func=backend_labels.get,
        disabled=not search_engine.has_hnsw,
        help="Exact compares against every image. HNSW walks a graph and "
        "checks only a fraction of them.",
    )
    ef_search = None

    if not search_engine.has_hnsw:
        backend = "flat"
        st.caption(search_engine.hnsw_unavailable_reason)
    elif backend == "hnsw":
        ef_search = st.slider(
            "efSearch",
            min_value=8,
            max_value=512,
            value=64,
            step=8,
            help="Candidates examined per query. Higher is more accurate "
            "but slower.",
        )

text_tab, image_tab, duplicates_tab = st.tabs(
    ["Search by text", "Search by image", "Find duplicates"]
)

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
                results = search_engine.search_text(
                    query, top_k, backend=backend, ef_search=ef_search
                )
            show_timing(search_engine, backend)
            render_results(results, min_score)

with image_tab:
    uploaded_file = st.file_uploader(
        "Upload a query image",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
    )

    if uploaded_file is not None:
        preview_column, _ = st.columns([1, 3])
        with preview_column:
            st.image(uploaded_file, caption="Query image", width="stretch")

        hide_exact = st.checkbox(
            "Hide exact matches",
            value=True,
            help="If this image is already in your collection, "
            "leave it out of its own results.",
        )

        if st.button("Find similar", key="image_search", type="primary"):
            suffix = Path(uploaded_file.name).suffix or ".png"

            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
                handle.write(uploaded_file.getbuffer())
                temporary_path = Path(handle.name)

            try:
                with st.spinner("Searching..."):
                    # Uploads arrive as temp files, so the path-based
                    # self-match filter can't recognize them. Drop
                    # near-perfect scores instead.
                    results = search_engine.search_image(
                        temporary_path,
                        top_k,
                        exclude_self=False,
                        backend=backend,
                        ef_search=ef_search,
                        exclude_above=EXACT_MATCH_THRESHOLD if hide_exact else None,
                    )
                show_timing(search_engine, backend)
                render_results(results, min_score)
            finally:
                temporary_path.unlink(missing_ok=True)

with duplicates_tab:
    st.write(
        "Groups images that are nearly identical: resized copies, "
        "re-saved exports, burst shots. Each image's score is its similarity "
        "to the first image in its group."
    )
    threshold = st.slider(
        "Similarity threshold",
        min_value=0.80,
        max_value=1.00,
        value=DEFAULT_THRESHOLD,
        step=0.01,
        help="Lower finds looser matches. 1.00 finds only identical images.",
    )

    if st.button("Find duplicates", key="duplicate_search", type="primary"):
        with st.spinner("Comparing every image against every other..."):
            groups = find_duplicate_groups(
                search_engine.index, search_engine.image_paths, threshold
            )

        if not groups:
            st.success("No near-duplicates found at this threshold.")
        else:
            removable = sum(len(g["images"]) - 1 for g in groups)
            st.write(
                f"**{len(groups)}** group(s) found. Keeping one image per "
                f"group would remove **{removable}**."
            )
            for number, group in enumerate(
                groups[:MAX_DUPLICATE_GROUPS_SHOWN], start=1
            ):
                st.subheader(f"Group {number} · {len(group['images'])} images")
                render_results(group["images"])
            if len(groups) > MAX_DUPLICATE_GROUPS_SHOWN:
                st.info(
                    f"Showing the first {MAX_DUPLICATE_GROUPS_SHOWN} groups. "
                    "Run `python src/duplicates.py` for the full list."
                )
