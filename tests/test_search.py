from __future__ import annotations

import json

import pytest

import indexer
import search
from search import ImageSearchEngine


@pytest.fixture
def engine(image_directory, index_directory) -> ImageSearchEngine:
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    return ImageSearchEngine(index_directory)


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ImageSearchEngine(tmp_path)


def test_image_search_excludes_the_query_itself(engine, image_directory):
    query = image_directory / "cats" / "image1.png"

    results = engine.search_image(query, top_k=3)

    assert len(results) == 3
    assert not any(r["image_path"].endswith("cats/image1.png") for r in results)


def test_image_search_can_include_the_query(engine, image_directory):
    query = image_directory / "cats" / "image1.png"

    results = engine.search_image(query, top_k=3, exclude_self=False)

    assert results[0]["image_path"].endswith("cats/image1.png")
    assert results[0]["similarity_score"] == pytest.approx(1.0, abs=1e-3)


def test_results_are_sorted_by_score(engine):
    results = engine.search_text("a cat on a sofa", top_k=10)
    scores = [r["similarity_score"] for r in results]
    assert scores == sorted(scores, reverse=True)


def test_top_k_is_capped_at_collection_size(engine):
    assert len(engine.search_text("anything", top_k=50)) == 10


def test_top_k_must_be_positive(engine):
    with pytest.raises(ValueError):
        engine.search_text("anything", top_k=0)


def test_missing_query_image_raises(engine, tmp_path):
    with pytest.raises(FileNotFoundError):
        engine.search_image(tmp_path / "missing.png")


@pytest.mark.parametrize("query", ["a dog", "sunset over water", "red car"])
def test_hnsw_matches_exact_search(engine, query):
    exact = engine.search_text(query, top_k=5, backend="flat")
    approximate = engine.search_text(query, top_k=5, backend="hnsw", ef_search=64)
    assert [r["image_path"] for r in approximate] == [r["image_path"] for r in exact]
    assert [r["similarity_score"] for r in approximate] == pytest.approx(
        [r["similarity_score"] for r in exact], abs=1e-4
    )


def test_unknown_backend_is_rejected(engine):
    with pytest.raises(ValueError):
        engine.search_text("anything", backend="annoy")


def test_hnsw_unavailable_without_graph(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    engine = ImageSearchEngine(index_directory)

    assert not engine.has_hnsw
    assert engine.search_text("still works", top_k=2, backend="flat")
    with pytest.raises(ValueError, match="No HNSW index"):
        engine.search_text("anything", backend="hnsw")


def test_stale_hnsw_graph_is_ignored(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    stale_graph = (index_directory / "hnsw.pkl").read_bytes()
    (image_directory / "dogs" / "image0.png").unlink()
    indexer.build_index(image_directory, index_directory)
    (index_directory / "hnsw.pkl").write_bytes(stale_graph)

    engine = ImageSearchEngine(index_directory)

    assert not engine.has_hnsw
    assert "out of date" in engine.hnsw_unavailable_reason


def test_mismatched_metadata_is_rejected(engine, index_directory):
    metadata_path = index_directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["images"] = metadata["images"][:-1]
    metadata_path.write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="different"):
        ImageSearchEngine(index_directory)


def test_lookup_time_is_recorded(engine):
    engine.search_text("anything", top_k=3)
    assert engine.last_search_ms >= 0


def test_command_line(engine, index_directory, image_directory, capsys, monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["search.py", "--image", str(image_directory / "dogs" / "image4.png"),
         "--top-k", "2", "--backend", "hnsw",
         "--index-directory", str(index_directory)],
    )
    search.main()
    output = capsys.readouterr()

    results = json.loads(output.out)
    assert len(results) == 2
    assert "hnsw index lookup" in output.err
