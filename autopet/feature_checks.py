"""Feature file checks."""

import hashlib
import json

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_cases(path, num_chunks):
    with open(path) as file:
        split = json.load(file)
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split must contain train, val and test")
    cases = sorted((case, name) for name in split for case in split[name])
    if not 1 <= num_chunks <= len(cases):
        raise ValueError("num_chunks must be between 1 and the number of cases")
    if len({case for case, _ in cases}) != len(cases):
        raise ValueError("Duplicate case IDs in split")
    patients = {}
    for case, name in cases:
        patient = "_".join(case.split("_")[:2])
        if patient in patients and patients[patient] != name:
            raise ValueError(f"Patient appears in multiple splits: {patient}")
        patients[patient] = name
    return [[cases[i] for i in indices]
            for indices in np.array_split(np.arange(len(cases)), num_chunks)]


def read_chunk(path, expected_cases, metadata=None):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "case_ids", "split", "tracer", "lesion", "metadata"}
        if not required.issubset(archive.files):
            raise ValueError(f"{path}: missing data/provenance; use a fresh output directory")
        data = {key: archive[key].copy() for key in required}
    actual_metadata = json.loads(data.pop("metadata").item())
    required_metadata = {"format_version", "checkpoint_name", "checkpoint_sha256",
                         "plans_sha256", "dataset_sha256", "split_sha256",
                         "extractor_sha256", "checks_sha256", "num_chunks", "pooling"}
    if not isinstance(actual_metadata, dict) or set(actual_metadata) != required_metadata:
        raise ValueError(f"{path}: invalid provenance metadata")
    if actual_metadata["format_version"] != 1:
        raise ValueError(f"{path}: unsupported feature format")
    if metadata is not None and actual_metadata != metadata:
        raise ValueError(f"{path}: checkpoint or extraction settings changed; use a fresh output directory")
    count = len(expected_cases)
    if any(data[key].shape != (count,) for key in ("case_ids", "split", "tracer", "lesion")):
        raise ValueError(f"{path}: inconsistent row counts")
    if list(zip(data["case_ids"].tolist(), data["split"].tolist())) != expected_cases:
        raise ValueError(f"{path}: case IDs/order/splits do not match the requested chunk")
    features = data["features"]
    if features.ndim != 2 or features.shape[0] != count or features.shape[1] == 0:
        raise ValueError(f"{path}: invalid feature dimensions")
    if not np.issubdtype(features.dtype, np.floating) or not np.isfinite(features).all():
        raise ValueError(f"{path}: features must be finite floating-point values")
    if not np.isin(data["lesion"], [0, 1]).all():
        raise ValueError(f"{path}: labels must be binary")
    if data["tracer"].tolist() != [case.split("_")[0] for case, _ in expected_cases]:
        raise ValueError(f"{path}: tracer does not match case ID")
    return data, actual_metadata
