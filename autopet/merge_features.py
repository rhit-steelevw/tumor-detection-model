import argparse
import json
import os
from pathlib import Path

import numpy as np

from feature_checks import file_sha256, split_cases, read_chunk

ap = argparse.ArgumentParser()
ap.add_argument("directory")
ap.add_argument("num_chunks", type=int)
ap.add_argument("--split", required=True, help="The split JSON used for extraction")
args = ap.parse_args()

directory = Path(args.directory)
chunks = split_cases(args.split, args.num_chunks)
files = [directory / f"chunk_{i:03d}.npz" for i in range(args.num_chunks)]
if set(directory.glob("chunk_*.npz")) != set(files):
    raise ValueError("Missing or unexpected chunks; expected exactly the numbered extraction jobs")

parts = []
metadata = None
split_hash = file_sha256(args.split)
for path, expected_cases in zip(files, chunks):
    part, current = read_chunk(path, expected_cases, metadata)
    if current["split_sha256"] != split_hash or current["num_chunks"] != args.num_chunks:
        raise ValueError(f"{path}: split or chunk count differs from requested merge")
    if parts and part["features"].shape[1] != parts[0]["features"].shape[1]:
        raise ValueError(f"{path}: feature width differs between chunks")
    metadata = current
    parts.append(part)

out = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
out["metadata"] = np.array(json.dumps(metadata, sort_keys=True))
destination = directory / "encoder_features.npz"
temporary = directory / f"encoder_features.{os.getpid()}.tmp"
try:
    with temporary.open("wb") as file:
        np.savez(file, **out)
    os.replace(temporary, destination)
finally:
    if temporary.exists():
        temporary.unlink()
print("features:", out["features"].shape,
      {name: int((out["split"] == name).sum()) for name in ("train", "val", "test")})
