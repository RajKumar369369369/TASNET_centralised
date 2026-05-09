from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    from scipy.io import whosmat
except ImportError:
    whosmat = None


def is_hdf5_mat(path: Path) -> bool:
    if h5py is None:
        return False
    try:
        with h5py.File(path, "r") as f:
            _ = list(f.keys())
        return True
    except Exception:
        return False


def inspect_h5(path: Path, max_depth: int = 3) -> Dict[str, Any]:
    """
    Inspect a v7.3 .mat file (HDF5-based) without loading large arrays.
    """
    out: Dict[str, Any] = {
        "file": str(path),
        "format": "hdf5_mat",
        "top_level_keys": [],
        "nodes": [],
        "warnings": [],
        "features_found": [],
        "labels_found": [],
    }

    def walk(name: str, obj: Any, depth: int):
        if depth > max_depth:
            return

        if isinstance(obj, h5py.Dataset):
            info = {
                "path": name,
                "kind": "dataset",
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
            }
            out["nodes"].append(info)

            # Heuristics for author compatibility
            if name.lower().endswith("features") or name.lower() == "features":
                out["features_found"].append(info)
            if name.lower().endswith("labels") or name.lower() == "labels":
                out["labels_found"].append(info)

        elif isinstance(obj, h5py.Group):
            info = {
                "path": name,
                "kind": "group",
                "children": list(obj.keys())[:20],
                "n_children": len(obj.keys()),
            }
            out["nodes"].append(info)

            for child_name, child_obj in obj.items():
                walk(f"{name}/{child_name}" if name else child_name, child_obj, depth + 1)

    with h5py.File(path, "r") as f:
        out["top_level_keys"] = list(f.keys())
        for k in f.keys():
            walk(k, f[k], 0)

    return out


def inspect_legacy_mat(path: Path) -> Dict[str, Any]:
    """
    Inspect a non-v7.3 MATLAB .mat file using whosmat so we don't load large arrays.
    """
    out: Dict[str, Any] = {
        "file": str(path),
        "format": "legacy_mat",
        "top_level_keys": [],
        "nodes": [],
        "warnings": [],
        "features_found": [],
        "labels_found": [],
    }

    if whosmat is None:
        out["warnings"].append("scipy.io.whosmat not available.")
        return out

    try:
        vars_info = whosmat(str(path))
        for name, shape, cls in vars_info:
            info = {
                "path": name,
                "kind": "variable",
                "shape": shape,
                "class": cls,
            }
            out["top_level_keys"].append(name)
            out["nodes"].append(info)

            if name.lower() == "features":
                out["features_found"].append(info)
            if name.lower() == "labels":
                out["labels_found"].append(info)

            # Some files may contain structs/cells instead of flat arrays
            if "struct" in str(cls).lower():
                out["warnings"].append(f"{name}: MATLAB struct found, may need custom parsing.")
            if "cell" in str(cls).lower():
                out["warnings"].append(f"{name}: MATLAB cell array found, may need custom parsing.")

    except Exception as e:
        out["warnings"].append(f"whosmat failed: {e}")

    return out


def file_summary(path: Path) -> Dict[str, Any]:
    """
    Return a compact summary for a single .mat file.
    """
    if hdf5_available_and_openable(path):
        return inspect_h5(path)
    return inspect_legacy_mat(path)


def hdf5_available_and_openable(path: Path) -> bool:
    return h5py is not None and is_hdf5_mat(path)


def get_subject_from_filename(name: str) -> str:
    # Example: 10_20131130.mat -> subject "10"
    stem = Path(name).stem
    if "_" in stem:
        return stem.split("_", 1)[0]
    return "unknown"


def shape_to_tuple(shape: Any) -> Tuple[int, ...]:
    try:
        return tuple(int(x) for x in shape)
    except Exception:
        return tuple()


def pretty_shape(shape: Any) -> str:
    try:
        return "x".join(str(int(x)) for x in shape)
    except Exception:
        return str(shape)


def analyze_summaries(
    summaries: List[Dict[str, Any]],
    expected_feature_dim: int = 192,
    expected_trials_per_subject: int | None = 15,
) -> Dict[str, Any]:
    """
    Compare your files against what TAS-Net main.py / evaluate.py expects.
    """
    report: Dict[str, Any] = {
        "global_warnings": [],
        "per_file_issues": [],
        "subject_counts": {},
        "feature_dims": [],
        "files_with_features": 0,
        "files_with_labels": 0,
        "files_compatible_with_author_structure": 0,
    }

    subject_counter = Counter()

    for s in summaries:
        fname = Path(s["file"]).name
        subject = get_subject_from_filename(fname)
        if subject != "unknown":
            subject_counter[subject] += 1

        issues = []

        features = s.get("features_found", [])
        labels = s.get("labels_found", [])

        if features:
            report["files_with_features"] += 1
        if labels:
            report["files_with_labels"] += 1

        # Author code expects trial-wise groups with features + labels
        # If both are present at some path, that's a good sign.
        author_like = bool(features and labels)

        if author_like:
            report["files_compatible_with_author_structure"] += 1
        else:
            issues.append("No clear 'features' + 'labels' structure found like the author's HDF5 dataset.")

        # Check feature dimension if we can infer it
        feature_dim_found = None
        feature_shapes = []
        for feat in features:
            shp = feat.get("shape", None)
            if shp:
                feature_shapes.append(tuple(shp))
                if len(shp) >= 2:
                    feature_dim_found = shp[-1]

        if feature_shapes:
            if feature_dim_found is not None:
                report["feature_dims"].append(feature_dim_found)
                if feature_dim_found != expected_feature_dim:
                    issues.append(
                        f"Feature dimension looks like {feature_dim_found}, not author's expected {expected_feature_dim}."
                    )

            # If there is a sequence-like first dimension, it should not be tiny
            first_shape = feature_shapes[0]
            if len(first_shape) >= 1 and first_shape[0] < 2:
                issues.append("Feature sequence length looks too small.")

        # If labels exist, check shape consistency heuristically
        label_shapes = []
        for lab in labels:
            shp = lab.get("shape", None)
            if shp:
                label_shapes.append(tuple(shp))
                if len(shp) == 0:
                    issues.append("Labels dataset has scalar shape; author code expects trial-wise labels or arrays.")
                if len(shp) == 1 and shp[0] < 2:
                    issues.append("Labels sequence is extremely short.")

        # File-level heuristics for SEED-style folder
        if s["format"] == "legacy_mat" and not features and not labels:
            issues.append("Legacy .mat file with no flat 'features'/'labels' variables; likely not directly compatible.")

        if s["format"] == "hdf5_mat" and len(s.get("top_level_keys", [])) == 0:
            issues.append("Empty HDF5 .mat file.")

        report["per_file_issues"].append(
            {
                "file": fname,
                "subject": subject,
                "issues": issues,
                "summary_format": s["format"],
                "top_level_keys": s.get("top_level_keys", []),
                "feature_shapes": [list(x.get("shape", [])) for x in features],
                "label_shapes": [list(x.get("shape", [])) for x in labels],
            }
        )

    report["subject_counts"] = dict(subject_counter)

    # Global issues related to the repo code
    if len(summaries) > 0 and report["files_compatible_with_author_structure"] == 0:
        report["global_warnings"].append(
            "None of your files clearly match the author's expected trial structure (features + labels)."
        )

    if expected_trials_per_subject is not None:
        bad_subjects = {
            subj: cnt for subj, cnt in subject_counter.items() if cnt != expected_trials_per_subject
        }
        if bad_subjects:
            report["global_warnings"].append(
                f"Some subjects do not have the expected {expected_trials_per_subject} files/trials: {bad_subjects}"
            )

    # Major structural mismatch with author's main.py
    report["global_warnings"].append(
        "The author code loads one HDF5 file containing many trial keys; a folder of standalone .mat files is a different structure."
    )
    report["global_warnings"].append(
        "If your files are session-level .mat files, you will likely need a conversion step into a single HDF5 file or adapt main.py."
    )

    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to the folder containing .mat files.",
    )
    parser.add_argument(
        "--expected_feature_dim",
        type=int,
        default=192,
        help="Author's expected feature dimension from TAS-Net code.",
    )
    parser.add_argument(
        "--expected_trials_per_subject",
        type=int,
        default=15,
        help="Author's assumed number of trials per subject in main.py.",
    )
    parser.add_argument(
        "--out_json",
        type=str,
        default="mat_audit_report.json",
        help="Where to save the full report.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")

    mat_files = sorted([p for p in folder.iterdir() if p.suffix.lower() == ".mat"])
    if not mat_files:
        print("No .mat files found.")
        return

    summaries = []
    for p in mat_files:
        print(f"\n=== Inspecting: {p.name} ===")
        try:
            s = file_summary(p)
            summaries.append(s)

            print(f"Format: {s['format']}")
            print(f"Top-level keys: {s.get('top_level_keys', [])}")

            if s.get("features_found"):
                for feat in s["features_found"]:
                    print(f"  features -> shape={feat.get('shape')}, dtype={feat.get('dtype', feat.get('class', 'unknown'))}")
            if s.get("labels_found"):
                for lab in s["labels_found"]:
                    print(f"  labels   -> shape={lab.get('shape')}, dtype={lab.get('dtype', lab.get('class', 'unknown'))}")

            if s.get("warnings"):
                for w in s["warnings"]:
                    print(f"  warning: {w}")

        except Exception as e:
            print(f"  FAILED to inspect: {e}")
            summaries.append(
                {
                    "file": str(p),
                    "format": "error",
                    "top_level_keys": [],
                    "nodes": [],
                    "warnings": [str(e)],
                    "features_found": [],
                    "labels_found": [],
                }
            )

    report = analyze_summaries(
        summaries,
        expected_feature_dim=args.expected_feature_dim,
        expected_trials_per_subject=args.expected_trials_per_subject,
    )

    print("\n\n==================== SUMMARY ====================")
    print(f"Total .mat files: {len(mat_files)}")
    print(f"Files with features-like variable: {report['files_with_features']}")
    print(f"Files with labels-like variable:   {report['files_with_labels']}")
    print(f"Files that look author-compatible:  {report['files_compatible_with_author_structure']}")
    print(f"Subjects found: {report['subject_counts']}")

    print("\n--- Global warnings ---")
    for w in report["global_warnings"]:
        print(f"- {w}")

    print("\n--- Possible problems per file ---")
    for item in report["per_file_issues"]:
        if item["issues"]:
            print(f"{item['file']}:")
            for issue in item["issues"]:
                print(f"  - {issue}")

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summaries": summaries,
                "analysis": report,
            },
            f,
            indent=2,
            default=str,
        )

    print(f"\nFull report saved to: {args.out_json}")


if __name__ == "__main__":
    main()