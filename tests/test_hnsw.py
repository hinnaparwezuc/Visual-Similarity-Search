from __future__ import annotations

import numpy as np
import pytest

from hnsw import HNSW


def random_unit_vectors(count: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(count, dim)).astype(np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def exact_top_k(data: np.ndarray, query: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-(data @ query))[:k]


def mean_recall(graph: HNSW, data: np.ndarray, queries: np.ndarray, k: int) -> float:
    hits = 0
    for query in queries:
        hits += len(set(graph.search(query, k)) & set(exact_top_k(data, query, k)))
    return hits / (len(queries) * k)


@pytest.fixture(scope="module")
def data() -> np.ndarray:
    return random_unit_vectors(1500, 32)


@pytest.fixture(scope="module")
def graph(data: np.ndarray) -> HNSW:
    index = HNSW(dim=32, M=16, ef_construction=100, seed=0)
    index.add(data, show_progress=False)
    return index


def test_recall_matches_exact_search(graph, data):
    queries = random_unit_vectors(50, 32, seed=1)
    graph.ef_search = 64
    assert mean_recall(graph, data, queries, k=10) >= 0.95


def test_higher_ef_search_does_not_reduce_recall(graph, data):
    queries = random_unit_vectors(50, 32, seed=2)
    graph.ef_search = 8
    low = mean_recall(graph, data, queries, k=10)
    graph.ef_search = 128
    high = mean_recall(graph, data, queries, k=10)
    graph.ef_search = 64
    assert high >= low


def test_stored_vector_finds_itself(graph, data):
    for row in (0, 500, 1499):
        assert graph.search(data[row], 1)[0] == row


def test_graph_shape(graph):
    stats = graph.stats()
    assert stats["vectors"] == 1500
    # Every node is in layer 0, and higher layers thin out.
    assert stats["nodes_per_layer"][0] == 1500
    assert all(a >= b for a, b in zip(stats["nodes_per_layer"],
                                      stats["nodes_per_layer"][1:]))
    # Layer 0 keeps at most 2 * M edges per node.
    assert max(len(edges) for edges in graph.layers[0].values()) <= 32


def test_save_and_load_round_trip(graph, data, tmp_path):
    path = tmp_path / "graph.pkl"
    graph.save(path)
    loaded = HNSW.load(path)

    assert loaded.ntotal == graph.ntotal
    for query in random_unit_vectors(10, 32, seed=3):
        assert list(loaded.search(query, 5)) == list(graph.search(query, 5))


def test_adding_in_batches_keeps_ids_and_recall(data):
    index = HNSW(dim=32, M=16, ef_construction=100, seed=0)
    index.add(data[:1000], show_progress=False)
    index.add(data[1000:], show_progress=False)

    assert index.ntotal == 1500
    # Rows added in the second batch keep their position in the full array.
    assert index.search(data[1200], 1)[0] == 1200
    queries = random_unit_vectors(30, 32, seed=4)
    assert mean_recall(index, data, queries, k=10) >= 0.95


def test_empty_index_returns_nothing():
    assert len(HNSW(dim=8).search(np.ones(8, dtype=np.float32), 3)) == 0


def test_wrong_dimension_is_rejected():
    with pytest.raises(ValueError):
        HNSW(dim=8).add(np.ones((2, 4), dtype=np.float32), show_progress=False)
