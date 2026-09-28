"""Hierarchical Navigable Small World index, written from scratch.

An exhaustive search compares a query against every vector, so its cost grows
linearly with the collection. HNSW instead builds a layered proximity graph
and walks it, touching a small fraction of the data per query.

The structure is a stack of graphs. The top layer is sparse and long-range;
each layer down is denser and shorter-range, with layer 0 holding every point.
A search enters at the top, greedily descends toward the query, and only does
careful work at the bottom. Insertions pick a maximum layer at random with an
exponentially decaying probability, which is what produces that hierarchy
without any global coordination.

Reference: Malkov & Yashunin, "Efficient and robust approximate nearest
neighbor search using Hierarchical Navigable Small World graphs" (2016).

Vectors are assumed L2-normalized, as CLIPEmbedder produces them, so
cosine similarity is a dot product and distance is 1 - similarity.
"""

from __future__ import annotations

import heapq
import math
import pickle
from pathlib import Path

import numpy as np


class HNSW:
    def __init__(
        self,
        dim: int,
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        seed: int | None = None,
    ):
        """
        M                neighbours kept per node per layer. Layer 0 keeps 2M,
                         since the bottom layer carries the real work.
        ef_construction  size of the candidate pool while inserting. Higher
                         builds a better graph, more slowly.
        ef_search        size of the candidate pool while querying. This is
                         the recall/latency dial, tunable after the build.
        """
        self.dim = dim
        self.M = M
        self.M0 = 2 * M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        # Level assignment decays as exp(-level / m_L); this value of m_L is
        # the one the paper shows minimises expected search cost.
        self.m_L = 1.0 / math.log(M) if M > 1 else 1.0

        self.vectors: np.ndarray | None = None
        self.layers: list[dict[int, list[int]]] = []   # layer -> node -> neighbours
        self.entry_point: int | None = None
        self.max_layer = -1
        self._rng = np.random.default_rng(seed)

    # ---------------- internals ----------------

    def _distance(self, q: np.ndarray, ids) -> np.ndarray:
        """Cosine distance from one query to many stored points, in one
        batched dot product rather than a Python loop."""
        return 1.0 - (self.vectors[ids] @ q)

    def _random_level(self) -> int:
        return int(-math.log(self._rng.random() + 1e-12) * self.m_L)

    def _search_layer(self, q: np.ndarray, entry_points: list[int],
                      ef: int, layer: int) -> list[tuple[float, int]]:
        """Best-first search within one layer.

        Two heaps: `candidates` is a min-heap of places still worth visiting,
        `results` is a max-heap (negated) holding the best ef found so far.
        The walk stops when the nearest unvisited candidate is further away
        than the worst result already held, which means the frontier can no
        longer improve the answer.
        """
        graph = self.layers[layer]
        dists = self._distance(q, entry_points)

        candidates = [(d, p) for d, p in zip(dists, entry_points)]
        heapq.heapify(candidates)
        results = [(-d, p) for d, p in candidates]
        heapq.heapify(results)
        visited = set(entry_points)

        while candidates:
            dist, node = heapq.heappop(candidates)
            if dist > -results[0][0] and len(results) >= ef:
                break

            neighbours = [n for n in graph.get(node, ()) if n not in visited]
            if not neighbours:
                continue
            visited.update(neighbours)

            for d, n in zip(self._distance(q, neighbours), neighbours):
                if len(results) < ef:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(results, (-d, n))
                elif d < -results[0][0]:
                    heapq.heappush(candidates, (d, n))
                    heapq.heapreplace(results, (-d, n))

        return [(-nd, n) for nd, n in results]

    def _select_neighbours(self, candidates: list[tuple[float, int]],
                           M: int) -> list[int]:
        """Prune a candidate set down to M edges.

        Not simply the M closest. A candidate is dropped if it sits closer to
        an already-selected neighbour than to the query point, because such an
        edge is redundant: the graph can already reach it through that
        neighbour. Keeping diverse edges is what stops the graph collapsing
        into a cluster and leaving whole regions unreachable.
        """
        candidates = sorted(candidates)
        selected: list[int] = []

        for dist, cand in candidates:
            if len(selected) >= M:
                break
            if not selected:
                selected.append(cand)
                continue
            to_selected = self._distance(self.vectors[cand], selected)
            if dist < to_selected.min():
                selected.append(cand)

        # If the heuristic was too aggressive, top up with nearest remaining.
        if len(selected) < M:
            for _, cand in candidates:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= M:
                    break

        return selected

    def _link(self, node: int, neighbours: list[int], layer: int) -> None:
        """Add edges both ways, re-pruning any neighbour that overflows."""
        graph = self.layers[layer]
        max_edges = self.M0 if layer == 0 else self.M
        graph[node] = list(neighbours)

        for n in neighbours:
            edges = graph.setdefault(n, [])
            if node in edges:
                continue
            edges.append(node)
            if len(edges) > max_edges:
                cands = [(d, e) for d, e in
                         zip(self._distance(self.vectors[n], edges), edges)]
                graph[n] = self._select_neighbours(cands, max_edges)

    # ---------------- public API ----------------

    def add(self, vectors: np.ndarray, show_progress: bool = True) -> None:
        """Insert every vector, one at a time, building the graph as we go."""
        vectors = np.ascontiguousarray(vectors, dtype="float32")
        if vectors.shape[1] != self.dim:
            raise ValueError(
                f"Expected {self.dim}-dimensional vectors, got {vectors.shape[1]}."
            )

        start = 0 if self.vectors is None else len(self.vectors)
        self.vectors = vectors if self.vectors is None else np.vstack(
            [self.vectors, vectors]
        )

        iterator = range(start, len(self.vectors))
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="Building HNSW", unit="vec")
            except ImportError:
                pass

        for node in iterator:
            self._insert(node)

    def _insert(self, node: int) -> None:
        level = self._random_level()
        q = self.vectors[node]

        while len(self.layers) <= level:
            self.layers.append({})

        if self.entry_point is None:
            for l in range(level + 1):
                self.layers[l][node] = []
            self.entry_point = node
            self.max_layer = level
            return

        ep = [self.entry_point]

        # Descend the layers above this node's level, greedily, keeping only
        # the single closest point to enter the next layer with.
        for layer in range(self.max_layer, level, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]

        # At and below this node's level, search properly and wire it in.
        for layer in range(min(level, self.max_layer), -1, -1):
            candidates = self._search_layer(q, ep, self.ef_construction, layer)
            M = self.M0 if layer == 0 else self.M
            self._link(node, self._select_neighbours(candidates, M), layer)
            ep = [n for _, n in candidates]

        for layer in range(self.max_layer + 1, level + 1):
            self.layers[layer][node] = []

        if level > self.max_layer:
            self.max_layer = level
            self.entry_point = node

    def search(self, q: np.ndarray, k: int) -> np.ndarray:
        """Return the ids of the approximate k nearest neighbours."""
        if self.entry_point is None:
            return np.empty(0, dtype=int)

        q = np.ascontiguousarray(q, dtype="float32").reshape(-1)
        ep = [self.entry_point]

        for layer in range(self.max_layer, 0, -1):
            ep = [self._search_layer(q, ep, 1, layer)[0][1]]

        ef = max(self.ef_search, k)
        results = self._search_layer(q, ep, ef, 0)
        results.sort()
        return np.array([n for _, n in results[:k]], dtype=int)

    # ---------------- persistence ----------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "dim": self.dim,
                    "M": self.M,
                    "ef_construction": self.ef_construction,
                    "ef_search": self.ef_search,
                    "vectors": self.vectors,
                    "layers": self.layers,
                    "entry_point": self.entry_point,
                    "max_layer": self.max_layer,
                },
                f,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

    @classmethod
    def load(cls, path: str | Path) -> "HNSW":
        with open(path, "rb") as f:
            state = pickle.load(f)
        index = cls(
            dim=state["dim"],
            M=state["M"],
            ef_construction=state["ef_construction"],
            ef_search=state["ef_search"],
        )
        index.vectors = state["vectors"]
        index.layers = state["layers"]
        index.entry_point = state["entry_point"]
        index.max_layer = state["max_layer"]
        return index

    @property
    def ntotal(self) -> int:
        return 0 if self.vectors is None else len(self.vectors)

    def stats(self) -> dict:
        """Graph shape, useful when a recall number needs explaining."""
        return {
            "vectors": self.ntotal,
            "layers": len(self.layers),
            "nodes_per_layer": [len(l) for l in self.layers],
            "mean_degree_layer0": (
                float(np.mean([len(v) for v in self.layers[0].values()]))
                if self.layers and self.layers[0] else 0.0
            ),
        }
