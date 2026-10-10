"""REST API for the image search engine.

Run locally:
    uvicorn src.api:app --reload
    python src/api.py

Then open http://localhost:8000/docs for interactive documentation.

The index to serve is read from the INDEX_DIRECTORY environment variable,
defaulting to data/index. If no index exists yet the API still starts:
/health reports why, and search endpoints return 503 until an index is
built and the server is restarted.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

SOURCE_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = SOURCE_DIRECTORY.parent
sys.path.insert(0, str(SOURCE_DIRECTORY))

from duplicates import DEFAULT_THRESHOLD, find_duplicate_groups  # noqa: E402
from search import ImageSearchEngine  # noqa: E402

DEFAULT_INDEX_DIRECTORY = PROJECT_ROOT / "data" / "index"
SEARCH_PAGE = SOURCE_DIRECTORY / "static" / "index.html"
EXACT_MATCH_THRESHOLD = 0.999
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TOP_K = 100

Backend = Literal["flat", "hnsw"]


# ----------------------------------------------------------------------
# Response and request shapes (these also drive the /docs page)
# ----------------------------------------------------------------------

class SearchResult(BaseModel):
    image_id: int = Field(description="Position of the image in the index.")
    image_path: str
    image_url: str = Field(description="Where to fetch the image from this API.")
    similarity_score: float = Field(description="Cosine similarity; higher is closer.")


class SearchResponse(BaseModel):
    results: list[SearchResult]
    backend: Backend
    lookup_ms: float = Field(description="Time spent in the index lookup only.")


class TextSearchRequest(BaseModel):
    model_config = {"json_schema_extra": {"examples": [
        {"query": "a big wild cat", "top_k": 5, "backend": "flat"}
    ]}}

    query: str = Field(min_length=1, max_length=500,
                       examples=["a dog running on the beach"])
    top_k: int = Field(5, ge=1, le=MAX_TOP_K)
    backend: Backend = "flat"
    ef_search: int | None = Field(
        None, ge=1, le=2048,
        description="HNSW only: candidates examined per query.",
    )


class DuplicateGroup(BaseModel):
    images: list[SearchResult]


class DuplicatesResponse(BaseModel):
    threshold: float
    groups: list[DuplicateGroup]
    removable: int = Field(description="Images that could be deleted while "
                                       "keeping one per group.")


class HealthResponse(BaseModel):
    status: Literal["ok"]
    image_count: int
    model_name: str
    pretrained: str
    hnsw_available: bool


# ----------------------------------------------------------------------
# App
# ----------------------------------------------------------------------

def create_app(index_directory: Path | None = None) -> FastAPI:
    """Build the API. Takes the index directory as an argument so tests can
    point each app at its own index."""

    if index_directory is None:
        index_directory = Path(
            os.environ.get("INDEX_DIRECTORY", DEFAULT_INDEX_DIRECTORY)
        )

    state: dict = {"engine": None, "error": None, "ids": {}}
    # The engine records timing and the HNSW search setting on itself, so
    # concurrent requests take turns. Lookups take well under a millisecond.
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            engine = ImageSearchEngine(Path(index_directory))
        except (FileNotFoundError, ValueError) as error:
            state["error"] = str(error)
        else:
            state["engine"] = engine
            state["ids"] = {path: i for i, path in enumerate(engine.image_paths)}
        yield

    app = FastAPI(
        title="Visual Similarity Search API",
        description=(
            "Search an image collection by text or by example image, using "
            "CLIP embeddings with exact (FAISS) or approximate (HNSW) "
            "nearest-neighbour search.\n\n"
            "Looking for the search page? It's at [the home page](/).\n\n"
            "Every result includes an `image_url` that this API serves. "
            "Choose `\"backend\": \"hnsw\"` for approximate search when the "
            "index was built with `--hnsw`."
        ),
        version="1.0.0",
        lifespan=lifespan,
        openapi_tags=[
            {"name": "Search", "description": "Find images by description or by example."},
            {"name": "Images", "description": "Fetch the images that results point to."},
            {"name": "Collection", "description": "Status and maintenance of the indexed collection."},
        ],
        # Hide the raw schema list at the bottom of /docs; each endpoint
        # already shows its own request and response shapes.
        swagger_ui_parameters={"defaultModelsExpandDepth": -1},
    )

    def engine_or_503() -> ImageSearchEngine:
        if state["engine"] is None:
            raise HTTPException(
                status_code=503,
                detail=f"{state['error']} Restart the API after building the index.",
            )
        return state["engine"]

    def to_result(result: dict) -> SearchResult:
        image_id = state["ids"][result["image_path"]]
        return SearchResult(
            image_id=image_id,
            image_path=result["image_path"],
            image_url=f"/images/{image_id}",
            similarity_score=result["similarity_score"],
        )

    def run_search(search) -> tuple[list[dict], float]:
        engine = engine_or_503()
        with lock:
            try:
                results = search(engine)
            except ValueError as error:          # e.g. no HNSW index built
                raise HTTPException(status_code=400, detail=str(error))
            return results, engine.last_search_ms

    # ---------------- endpoints ----------------

    @app.get("/", include_in_schema=False)
    def search_page():
        """The search page people use; it calls the endpoints below."""
        return FileResponse(SEARCH_PAGE, media_type="text/html")

    @app.get("/health", response_model=HealthResponse, tags=["Collection"],
             summary="Check status",
             responses={503: {"description": "No index loaded"}})
    def health():
        """Report whether an index is loaded and what it contains."""
        engine = engine_or_503()
        return HealthResponse(
            status="ok",
            image_count=len(engine.image_paths),
            model_name=engine.metadata.get("model_name", "unknown"),
            pretrained=engine.metadata.get("pretrained", "unknown"),
            hnsw_available=engine.has_hnsw,
        )

    @app.post("/search/text", response_model=SearchResponse, tags=["Search"],
              summary="Search by description",
              responses={400: {"description": "HNSW requested but not built"}})
    def search_text(request: TextSearchRequest):
        """Find images matching a text description."""
        results, lookup_ms = run_search(
            lambda engine: engine.search_text(
                request.query, request.top_k, request.backend, request.ef_search
            )
        )
        return SearchResponse(
            results=[to_result(r) for r in results],
            backend=request.backend,
            lookup_ms=round(lookup_ms, 3),
        )

    @app.post("/search/image", response_model=SearchResponse, tags=["Search"],
              summary="Search by example image",
              responses={400: {"description": "Empty or unreadable image"},
                         413: {"description": "Image larger than 20 MB"}})
    def search_image(
        file: Annotated[UploadFile, File(description="Query image.")],
        top_k: Annotated[int, Form(ge=1, le=MAX_TOP_K)] = 5,
        backend: Annotated[Backend, Form()] = "flat",
        ef_search: Annotated[int | None, Form(ge=1, le=2048)] = None,
        exclude_exact: Annotated[bool, Form(
            description="Leave out near-identical matches, such as the "
                        "uploaded image itself if it is in the collection."
        )] = True,
    ):
        """Find images that look like an uploaded image."""
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(status_code=413, detail="Image is larger than 20 MB.")
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        suffix = Path(file.filename or "").suffix or ".png"
        handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            handle.write(data)
            handle.close()
            try:
                results, lookup_ms = run_search(
                    lambda engine: engine.search_image(
                        Path(handle.name), top_k, exclude_self=False,
                        backend=backend, ef_search=ef_search,
                        exclude_above=EXACT_MATCH_THRESHOLD if exclude_exact else None,
                    )
                )
            except RuntimeError:                 # not a readable image
                raise HTTPException(status_code=400,
                                    detail="The uploaded file is not a readable image.")
        finally:
            handle.close()
            Path(handle.name).unlink(missing_ok=True)

        return SearchResponse(
            results=[to_result(r) for r in results],
            backend=backend,
            lookup_ms=round(lookup_ms, 3),
        )

    @app.get("/images/{image_id}", response_class=FileResponse, tags=["Images"],
             summary="Get an image",
             responses={200: {"content": {"image/*": {}},
                              "description": "The image file"},
                        404: {"description": "Unknown id or file missing"}})
    def get_image(image_id: int):
        """Return an indexed image by its id from a search result.

        Images are looked up by id rather than by path, so only files that
        are in the index can ever be served.
        """
        engine = engine_or_503()
        if not 0 <= image_id < len(engine.image_paths):
            raise HTTPException(status_code=404, detail="No image with that id.")
        stored = Path(engine.image_paths[image_id])
        path = stored if stored.is_absolute() else PROJECT_ROOT / stored
        if not path.is_file():
            raise HTTPException(status_code=404,
                                detail="The image file is no longer on disk.")
        return FileResponse(path)

    @app.get("/duplicates", response_model=DuplicatesResponse, tags=["Collection"],
             summary="Find near-duplicates")
    def duplicates(
        threshold: Annotated[float, Query(gt=0, le=1)] = DEFAULT_THRESHOLD,
    ):
        """Group near-identical images in the collection."""
        engine = engine_or_503()
        groups = find_duplicate_groups(engine.index, engine.image_paths, threshold)
        return DuplicatesResponse(
            threshold=threshold,
            groups=[DuplicateGroup(images=[to_result(r) for r in g["images"]])
                    for g in groups],
            removable=sum(len(g["images"]) - 1 for g in groups),
        )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")))
