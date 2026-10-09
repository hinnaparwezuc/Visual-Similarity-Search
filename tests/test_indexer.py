from __future__ import annotations

import json
import os

import faiss
import pytest

import fake_embedder
import indexer
from conftest import make_image
from hnsw import HNSW


def embedded() -> int:
    return fake_embedder.CLIPEmbedder.images_embedded


def read_metadata(index_directory):
    return json.loads((index_directory / "metadata.json").read_text())


def graph_count(index_directory) -> int:
    return HNSW.load(index_directory / "hnsw.pkl").ntotal


def touch_later(path):
    """Push the modification time forward so the change is always detected,
    even on filesystems with coarse timestamps."""
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 5_000_000_000))


# ---------------- full builds ----------------

def test_full_build_writes_index_and_metadata(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)

    metadata = read_metadata(index_directory)
    index = faiss.read_index(str(index_directory / "images.faiss"))

    assert metadata["image_count"] == 10 == index.ntotal
    assert len(metadata["signatures"]) == 10
    assert metadata["model_name"] == fake_embedder.DEFAULT_MODEL_NAME
    assert not (index_directory / "hnsw.pkl").exists()


def test_corrupt_images_are_skipped(image_directory, index_directory):
    (image_directory / "broken.jpg").write_bytes(b"not an image")

    summary = indexer.build_index(image_directory, index_directory)

    assert summary["skipped"] == 1
    metadata = read_metadata(index_directory)
    assert metadata["image_count"] == 10
    assert not any("broken" in path for path in metadata["images"])


def test_missing_or_empty_directory_raises(tmp_path, index_directory):
    with pytest.raises(FileNotFoundError):
        indexer.build_index(tmp_path / "nope", index_directory)

    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        indexer.build_index(tmp_path / "empty", index_directory)


def test_with_hnsw_builds_matching_graph(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)

    assert graph_count(index_directory) == 10
    assert read_metadata(index_directory)["hnsw"] == {"M": 16, "ef_construction": 200}


def test_full_rebuild_without_hnsw_removes_stale_graph(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    indexer.build_index(image_directory, index_directory)

    assert not (index_directory / "hnsw.pkl").exists()
    assert "hnsw" not in read_metadata(index_directory)


def test_hnsw_only_reuses_stored_vectors(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    before = embedded()

    indexer.add_hnsw_to_existing(index_directory)

    assert embedded() == before          # no images re-embedded
    assert graph_count(index_directory) == 10


# ---------------- incremental updates ----------------

def test_update_with_no_changes_embeds_nothing(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    before = embedded()

    summary = indexer.build_index(image_directory, index_directory, update=True)

    assert embedded() == before
    assert summary == {"added": 0, "removed": 0, "unchanged": 10, "skipped": 0}


def test_update_embeds_only_new_images(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    make_image(image_directory / "new" / "fresh.png", seed=100)
    before = embedded()

    summary = indexer.build_index(image_directory, index_directory, update=True)

    assert embedded() - before == 1
    assert summary["added"] == 1 and summary["unchanged"] == 10
    metadata = read_metadata(index_directory)
    assert metadata["image_count"] == 11
    assert metadata["images"][-1].endswith("new/fresh.png")


def test_update_drops_deleted_images(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    (image_directory / "dogs" / "image0.png").unlink()

    summary = indexer.build_index(image_directory, index_directory, update=True)

    assert summary["removed"] == 1 and summary["added"] == 0
    metadata = read_metadata(index_directory)
    assert metadata["image_count"] == 9
    assert not any(path.endswith("image0.png") for path in metadata["images"])
    assert faiss.read_index(str(index_directory / "images.faiss")).ntotal == 9


def test_update_re_embeds_modified_images(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    changed = make_image(image_directory / "cats" / "image1.png", seed=999)
    touch_later(changed)
    before = embedded()

    summary = indexer.build_index(image_directory, index_directory, update=True)

    assert embedded() - before == 1
    assert summary["added"] == 1 and summary["removed"] == 1
    assert read_metadata(index_directory)["image_count"] == 10


def test_update_keeps_vectors_identical_to_full_build(image_directory, tmp_path):
    """An updated index must search exactly like one built from scratch."""
    updated_directory = tmp_path / "updated"
    indexer.build_index(image_directory, updated_directory)
    make_image(image_directory / "new" / "fresh.png", seed=100)
    (image_directory / "dogs" / "image2.png").unlink()
    indexer.build_index(image_directory, updated_directory, update=True)

    fresh_directory = tmp_path / "fresh"
    indexer.build_index(image_directory, fresh_directory)

    def vectors_by_path(directory):
        index = faiss.read_index(str(directory / "images.faiss"))
        images = read_metadata(directory)["images"]
        vectors = index.reconstruct_n(0, index.ntotal)
        return {path: vectors[row].tolist() for row, path in enumerate(images)}

    updated, fresh = vectors_by_path(updated_directory), vectors_by_path(fresh_directory)
    assert updated.keys() == fresh.keys()
    for path in fresh:
        assert updated[path] == pytest.approx(fresh[path], abs=1e-6)


def test_update_extends_graph_when_images_are_only_added(
    image_directory, index_directory, monkeypatch
):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    make_image(image_directory / "new" / "fresh.png", seed=100)

    # Pure additions must append to the saved graph, never rebuild it.
    def fail(*args, **kwargs):
        raise AssertionError("graph was rebuilt instead of extended")
    monkeypatch.setattr(indexer, "build_hnsw", fail)

    indexer.build_index(image_directory, index_directory, update=True)

    graph = HNSW.load(index_directory / "hnsw.pkl")
    assert graph.ntotal == 11
    new_vector = graph.vectors[10]
    assert graph.search(new_vector, 1)[0] == 10


def test_update_rebuilds_graph_after_deletion(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory, with_hnsw=True)
    (image_directory / "cats" / "image3.png").unlink()
    before = embedded()

    indexer.build_index(image_directory, index_directory, update=True)

    assert embedded() == before          # graph rebuilt from stored vectors
    assert graph_count(index_directory) == 9


def test_update_with_hnsw_adds_missing_graph(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)

    indexer.build_index(image_directory, index_directory, update=True, with_hnsw=True)

    assert graph_count(index_directory) == 10


def test_update_without_previous_index_builds_from_scratch(image_directory, index_directory):
    summary = indexer.build_index(image_directory, index_directory, update=True)
    assert summary["added"] == 10


def test_update_on_old_metadata_rebuilds_once(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    metadata = read_metadata(index_directory)
    del metadata["signatures"]           # as written by older versions
    (index_directory / "metadata.json").write_text(json.dumps(metadata))
    before = embedded()

    indexer.build_index(image_directory, index_directory, update=True)

    assert embedded() - before == 10
    assert "signatures" in read_metadata(index_directory)


def test_update_with_different_model_rebuilds(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    before = embedded()

    indexer.build_index(
        image_directory, index_directory, update=True, model_name="ViT-L-14"
    )

    assert embedded() - before == 10
    assert read_metadata(index_directory)["model_name"] == "ViT-L-14"


def test_command_line_flags(image_directory, index_directory):
    indexer.main(["--images", str(image_directory), "--output", str(index_directory), "--hnsw"])
    make_image(image_directory / "new" / "fresh.png", seed=100)
    indexer.main(["--images", str(image_directory), "--output", str(index_directory), "--update"])

    assert read_metadata(index_directory)["image_count"] == 11
    assert graph_count(index_directory) == 11
