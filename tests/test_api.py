"""Exercise the REST API end to end with FastAPI's test client."""

from __future__ import annotations

import io
import shutil

import pytest

fastapi_testclient = pytest.importorskip("fastapi.testclient")

import api  # noqa: E402
import indexer  # noqa: E402

TestClient = fastapi_testclient.TestClient


@pytest.fixture
def client(image_directory, index_directory):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "copy.png")
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    with TestClient(api.create_app(index_directory)) as test_client:
        yield test_client


@pytest.fixture
def client_without_index(tmp_path):
    with TestClient(api.create_app(tmp_path / "missing")) as test_client:
        yield test_client


# ---------------- health ----------------

def test_health_reports_the_index(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["image_count"] == 11
    assert body["hnsw_available"] is True


def test_missing_index_returns_503_with_instructions(client_without_index):
    response = client_without_index.get("/health")
    assert response.status_code == 503
    assert "indexer.py" in response.json()["detail"]

    search = client_without_index.post("/search/text", json={"query": "cat"})
    assert search.status_code == 503


def test_home_page_is_the_search_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    page = response.text
    assert "<title>Visual Similarity Search</title>" in page
    # The page talks to the API's own endpoints and links to the docs.
    for path in ("/health", "/search/text", "/search/image", "/duplicates", 'href="/docs"'):
        assert path in page


def test_home_page_loads_without_an_index(client_without_index):
    assert client_without_index.get("/").status_code == 200


def test_docs_group_endpoints_by_purpose(client):
    schema = client.get("/openapi.json").json()
    assert [t["name"] for t in schema["tags"]] == ["Search", "Images", "Collection"]
    tags = {path: next(iter(ops.values()))["tags"][0]
            for path, ops in schema["paths"].items()}
    assert tags == {
        "/health": "Collection", "/search/text": "Search",
        "/search/image": "Search", "/images/{image_id}": "Images",
        "/duplicates": "Collection",
    }
    assert "/" not in schema["paths"]


def test_openapi_schema_lists_every_endpoint(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/health", "/search/text", "/search/image",
            "/images/{image_id}", "/duplicates"} <= set(paths)


# ---------------- text search ----------------

def test_text_search_returns_ranked_results(client):
    response = client.post("/search/text", json={"query": "a dog", "top_k": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["backend"] == "flat"
    assert len(body["results"]) == 4
    scores = [r["similarity_score"] for r in body["results"]]
    assert scores == sorted(scores, reverse=True)
    first = body["results"][0]
    assert first["image_url"] == f"/images/{first['image_id']}"


def test_text_search_with_hnsw_matches_exact(client):
    exact = client.post("/search/text", json={"query": "sunset", "top_k": 5}).json()
    approx = client.post("/search/text", json={
        "query": "sunset", "top_k": 5, "backend": "hnsw", "ef_search": 64,
    }).json()
    assert approx["backend"] == "hnsw"
    # Compare scores rather than ids: the image and its identical copy tie,
    # and the two indexes may list tied images in either order.
    assert [r["similarity_score"] for r in approx["results"]] == pytest.approx(
        [r["similarity_score"] for r in exact["results"]], abs=1e-4)


@pytest.mark.parametrize("payload", [
    {"query": ""},
    {"query": "cat", "top_k": 0},
    {"query": "cat", "top_k": 1000},
    {"query": "cat", "backend": "annoy"},
    {"top_k": 3},
])
def test_invalid_text_requests_are_rejected(client, payload):
    assert client.post("/search/text", json=payload).status_code == 422


def test_hnsw_without_graph_is_a_clear_400(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    with TestClient(api.create_app(index_directory)) as client:
        response = client.post("/search/text",
                               json={"query": "cat", "backend": "hnsw"})
    assert response.status_code == 400
    assert "No HNSW index" in response.json()["detail"]


# ---------------- image search ----------------

def upload(client, path, **form):
    with open(path, "rb") as handle:
        return client.post(
            "/search/image",
            files={"file": (path.name, handle, "image/png")},
            data={key: str(value) for key, value in form.items()},
        )


def test_image_search_hides_exact_matches_by_default(client, image_directory):
    response = upload(client, image_directory / "dogs" / "image0.png", top_k=3)
    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 3
    # Both the original and its copy are identical to the query.
    assert all(r["similarity_score"] < 0.999 for r in results)


def test_image_search_can_include_exact_matches(client, image_directory):
    response = upload(client, image_directory / "dogs" / "image0.png",
                      top_k=2, exclude_exact="false", backend="hnsw")
    names = sorted(r["image_path"].rsplit("/", 1)[-1]
                   for r in response.json()["results"])
    assert names == ["copy.png", "image0.png"]


def test_unreadable_upload_is_a_400(client):
    response = client.post(
        "/search/image",
        files={"file": ("notes.png", io.BytesIO(b"not an image"), "image/png")},
    )
    assert response.status_code == 400
    assert "not a readable image" in response.json()["detail"]


def test_empty_upload_is_a_400(client):
    response = client.post(
        "/search/image", files={"file": ("empty.png", io.BytesIO(b""), "image/png")}
    )
    assert response.status_code == 400


def test_oversized_upload_is_a_413(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_BYTES", 10)
    response = client.post(
        "/search/image",
        files={"file": ("big.png", io.BytesIO(b"x" * 100), "image/png")},
    )
    assert response.status_code == 413


# ---------------- images and duplicates ----------------

def test_result_urls_serve_the_image(client):
    result = client.post("/search/text", json={"query": "cat", "top_k": 1}).json()
    url = result["results"][0]["image_url"]

    response = client.get(url)

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


@pytest.mark.parametrize("image_id", [-1, 11, 9999])
def test_unknown_image_id_is_a_404(client, image_id):
    assert client.get(f"/images/{image_id}").status_code == 404


def test_deleted_image_file_is_a_404(client, image_directory):
    (image_directory / "copy.png").unlink()
    ids = {r["image_path"].rsplit("/", 1)[-1]: r["image_id"]
           for r in client.get("/duplicates").json()["groups"][0]["images"]}
    assert client.get(f"/images/{ids['copy.png']}").status_code == 404


def test_duplicates_endpoint_groups_the_copy(client):
    body = client.get("/duplicates", params={"threshold": 0.99}).json()
    assert body["removable"] == 1
    names = sorted(r["image_path"].rsplit("/", 1)[-1]
                   for r in body["groups"][0]["images"])
    assert names == ["copy.png", "image0.png"]


@pytest.mark.parametrize("threshold", [0, -1, 1.5])
def test_invalid_duplicate_threshold_is_a_422(client, threshold):
    assert client.get("/duplicates",
                      params={"threshold": threshold}).status_code == 422
