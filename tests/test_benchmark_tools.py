from __future__ import annotations

import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmarks"))

import bench  # noqa: E402
import prepare_pets  # noqa: E402


@pytest.mark.parametrize(
    "filename, breed",
    [
        ("Abyssinian_12.jpg", "abyssinian"),
        ("american_bulldog_3.jpg", "american_bulldog"),
        ("Egyptian_Mau_100.jpg", "egyptian_mau"),
        ("images/yorkshire_terrier_7.jpg", "yorkshire_terrier"),
    ],
)
def test_breed_from_filename(filename, breed):
    assert prepare_pets.breed_from_filename(filename) == breed


def make_archive(members: dict[str, bytes]) -> io.BytesIO:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    buffer.seek(0)
    return buffer


def test_extract_sorts_by_breed_and_skips_other_files(tmp_path):
    archive = make_archive({
        "images/Bengal_1.jpg": b"a",
        "images/Bengal_2.jpg": b"b",
        "images/pug_1.jpg": b"c",
        "images/Bengal_100.mat": b"not an image",
    })

    counts = prepare_pets.extract(archive, tmp_path, per_breed=None)

    assert counts == {"bengal": 2, "pug": 1}
    assert (tmp_path / "bengal" / "Bengal_2.jpg").read_bytes() == b"b"
    assert not list(tmp_path.rglob("*.mat"))


def test_extract_respects_per_breed_limit(tmp_path):
    archive = make_archive({f"images/pug_{i}.jpg": b"x" for i in range(10)})
    counts = prepare_pets.extract(archive, tmp_path, per_breed=3)
    assert counts == {"pug": 3}


def test_extract_cannot_write_outside_output(tmp_path):
    output = tmp_path / "out"
    archive = make_archive({"../../escaped_1.jpg": b"x"})

    prepare_pets.extract(archive, output, per_breed=None)

    assert not (tmp_path / "escaped_1.jpg").exists()
    assert (output / "escaped" / "escaped_1.jpg").exists()


def test_label_precision_counts_every_result():
    labels = ["cat", "cat", "cat", "dog", "dog"]
    # Query 0 gets two cats and a dog; query 3 gets one dog and two cats.
    got = [[1, 2, 3], [4, 0, 1]]
    assert bench.label_precision(got, [0, 3], labels) == pytest.approx(3 / 6)


def test_label_precision_ignores_self_and_padding():
    labels = ["cat", "cat", "dog"]
    assert bench.label_precision([[0, 1, -1]], [0], labels) == 1.0
