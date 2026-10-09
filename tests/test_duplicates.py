from __future__ import annotations

import json
import shutil

import faiss
import numpy as np
import pytest

import duplicates
import indexer
from conftest import make_near_copy


def build(image_directory, index_directory):
    indexer.build_index(image_directory, index_directory)
    return duplicates.load_index(index_directory)


def group_names(groups):
    return [sorted(item["image_path"].rsplit("/", 1)[-1] for item in g["images"])
            for g in groups]


def test_no_duplicates_in_unrelated_images(image_directory, index_directory):
    index, paths = build(image_directory, index_directory)
    assert duplicates.find_duplicate_groups(index, paths, threshold=0.95) == []


def test_exact_and_near_copies_are_grouped(image_directory, index_directory):
    original = image_directory / "dogs" / "image0.png"
    shutil.copy(original, image_directory / "dogs" / "exact_copy.png")
    make_near_copy(original, image_directory / "other" / "edited_copy.png")

    index, paths = build(image_directory, index_directory)
    groups = duplicates.find_duplicate_groups(index, paths, threshold=0.95)

    assert group_names(groups) == [["edited_copy.png", "exact_copy.png", "image0.png"]]
    scores = [item["similarity_score"] for item in groups[0]["images"]]
    assert scores[0] == pytest.approx(1.0, abs=1e-4)      # the reference
    assert all(score >= 0.95 for score in scores)


def test_separate_groups_stay_separate(image_directory, index_directory):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "a.png")
    shutil.copy(image_directory / "cats" / "image1.png", image_directory / "b.png")
    shutil.copy(image_directory / "cats" / "image1.png", image_directory / "c.png")

    index, paths = build(image_directory, index_directory)
    groups = duplicates.find_duplicate_groups(index, paths, threshold=0.99)

    # Largest group first.
    assert group_names(groups) == [["b.png", "c.png", "image1.png"],
                                   ["a.png", "image0.png"]]


def test_threshold_of_one_still_finds_identical_images(image_directory, index_directory):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "copy.png")
    index, paths = build(image_directory, index_directory)
    assert len(duplicates.find_duplicate_groups(index, paths, threshold=1.0)) == 1


def test_matches_are_chained_into_one_group():
    """A~B and B~C put A, B and C in one group even if A and C differ more."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    c = np.array([np.cos(0.5), np.sin(0.5)], dtype=np.float32)   # cos(A,C)≈0.88
    b = (a + c) / np.linalg.norm(a + c)                           # cos≈0.97 to both
    index = faiss.IndexFlatIP(2)
    index.add(np.stack([a, b, c]).astype(np.float32))

    groups = duplicates.find_duplicate_groups(index, ["a", "b", "c"], threshold=0.95)

    assert len(groups) == 1 and len(groups[0]["images"]) == 3


def test_batches_give_the_same_answer(image_directory, index_directory):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "copy.png")
    index, paths = build(image_directory, index_directory)

    whole = duplicates.find_duplicate_groups(index, paths, batch_size=1024)
    batched = duplicates.find_duplicate_groups(index, paths, batch_size=3)

    assert whole == batched


@pytest.mark.parametrize("threshold", [0.0, -0.5, 1.5])
def test_invalid_threshold_is_rejected(image_directory, index_directory, threshold):
    index, paths = build(image_directory, index_directory)
    with pytest.raises(ValueError):
        duplicates.find_duplicate_groups(index, paths, threshold=threshold)


def test_tiny_index_has_no_duplicates():
    index = faiss.IndexFlatIP(2)
    index.add(np.array([[1.0, 0.0]], dtype=np.float32))
    assert duplicates.find_duplicate_groups(index, ["only"]) == []


def test_command_line(image_directory, index_directory, capsys):
    shutil.copy(image_directory / "dogs" / "image0.png", image_directory / "copy.png")
    indexer.build_index(image_directory, index_directory)
    capsys.readouterr()                  # discard the indexer's progress output

    duplicates.main(["--index-directory", str(index_directory)])
    output = capsys.readouterr()

    assert len(json.loads(output.out)) == 1
    assert "1 image(s) could be removed" in output.err
