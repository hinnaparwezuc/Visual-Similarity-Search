"""
prepare_pets.py — download a labeled benchmark dataset, one folder per class.

Fetches the Oxford-IIIT Pet dataset (about 7,400 photos of 37 cat and dog
breeds; Parkhi et al., 2012) and extracts it straight into

    data/benchmark/images/<breed>/<photo>.jpg

The folder names give bench.py its ground-truth labels, so it can measure
how often a search returns the same breed as the query. Breeds make a hard
test: many look alike (Bengal vs Egyptian Mau, the terriers), so the score
says something real about the embeddings.

USAGE
  python benchmarks/prepare_pets.py                     # all ~7,400 images
  python benchmarks/prepare_pets.py --per-breed 50      # smaller, faster
  python benchmarks/prepare_pets.py --archive images.tar.gz
                                     # use a copy you downloaded yourself

Then:
  python src/indexer.py --images data/benchmark/images --output data/benchmark/index
  python benchmarks/bench.py --index-directory data/benchmark/index --k 10

The archive (~790 MB) is streamed and never saved, so only the extracted
images take up disk space.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "benchmark" / "images"

ARCHIVE_URL = "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
ARCHIVE_MD5 = "5c4f3ee8e5d25df40f4fd59a7f44e54c"


def breed_from_filename(name: str) -> str:
    """'Abyssinian_12.jpg' -> 'abyssinian', 'american_bulldog_3.jpg' ->
    'american_bulldog'. The dataset names every photo <breed>_<number>."""
    return Path(name).stem.rsplit("_", 1)[0].lower()


class _HashingReader:
    """Wraps a stream so the MD5 and a progress bar update as it is read."""

    def __init__(self, stream, total: int | None):
        self.stream = stream
        self.md5 = hashlib.md5()
        self.progress = None
        try:
            from tqdm import tqdm
            self.progress = tqdm(total=total, unit="B", unit_scale=True,
                                 desc="Downloading")
        except ImportError:
            pass

    def read(self, size: int = -1) -> bytes:
        chunk = self.stream.read(size)
        self.md5.update(chunk)
        if self.progress is not None:
            self.progress.update(len(chunk))
        return chunk

    def close(self) -> None:
        if self.progress is not None:
            self.progress.close()


def extract(stream, output: Path, per_breed: int | None) -> Counter:
    """Stream a .tar.gz and write each photo into its breed's folder.

    Only the base file name of each archive member is used, so nothing can
    be written outside `output`.
    """
    counts: Counter = Counter()
    with tarfile.open(fileobj=stream, mode="r|gz") as archive:
        for member in archive:
            name = Path(member.name).name
            if not member.isfile() or not name.lower().endswith(".jpg"):
                continue                     # skips the stray .mat files
            breed = breed_from_filename(name)
            if per_breed is not None and counts[breed] >= per_breed:
                continue
            data = archive.extractfile(member)
            if data is None:
                continue
            folder = output / breed
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data.read())
            counts[breed] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download Oxford-IIIT Pets into one folder per breed."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Where to put the breed folders.")
    parser.add_argument("--per-breed", type=int, default=None,
                        help="Keep at most this many photos per breed.")
    parser.add_argument("--archive", type=Path, default=None,
                        help="Use an already-downloaded images.tar.gz.")
    args = parser.parse_args(argv)

    if args.output.exists() and any(args.output.rglob("*.jpg")):
        print(f"{args.output} already has images; nothing to do. "
              "Delete the folder to prepare it again.")
        return 0

    if args.archive is not None:
        print(f"Reading {args.archive}...")
        raw = open(args.archive, "rb")
        total = args.archive.stat().st_size
    else:
        print(f"Downloading {ARCHIVE_URL}")
        raw = urllib.request.urlopen(ARCHIVE_URL)
        length = raw.headers.get("Content-Length")
        total = int(length) if length else None

    reader = _HashingReader(raw, total)
    try:
        counts = extract(reader, args.output, args.per_breed)
        # Drain anything tarfile left unread, so the checksum covers it all.
        while reader.read(1 << 20):
            pass
    finally:
        reader.close()
        raw.close()

    if reader.md5.hexdigest() != ARCHIVE_MD5:
        print("Warning: the archive's checksum doesn't match the published "
              "one. The images may still be fine, but the download could "
              "be incomplete.", file=sys.stderr)

    print(f"\nExtracted {sum(counts.values())} images in {len(counts)} breeds "
          f"to {args.output}")
    print("Next:\n"
          f"  python src/indexer.py --images {args.output.relative_to(PROJECT_ROOT) if args.output.is_relative_to(PROJECT_ROOT) else args.output} "
          "--output data/benchmark/index")
    return 0


if __name__ == "__main__":
    sys.exit(main())
