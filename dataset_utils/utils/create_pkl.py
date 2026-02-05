#!/usr/bin/env python3
import os
import argparse
import pickle
import random
import shutil
from pathlib import Path

def is_nonempty_file(p: Path) -> bool:
    return p.is_file() and p.stat().st_size > 0

def safe_symlink(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def safe_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py_dir", type=Path, required=True)
    ap.add_argument("--stl_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--mode", choices=["symlink", "copy"], default="symlink")
    ap.add_argument("--val_size", type=int, default=0)   # 0 => only pairs.pkl
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_code_len", type=int, default=0)  # 0 => no filter
    args = ap.parse_args()

    py_dir = args.py_dir.resolve()
    stl_dir = args.stl_dir.resolve()
    out_dir = args.out_dir.resolve()

    assert py_dir.is_dir(), py_dir
    assert stl_dir.is_dir(), stl_dir
    out_py = out_dir / "py"
    out_stl = out_dir / "stl"
    out_dir.mkdir(parents=True, exist_ok=True)

    copier = safe_symlink if args.mode == "symlink" else safe_copy

    # index STLs by stem
    stl_map = {p.stem: p for p in stl_dir.glob("*.stl") if is_nonempty_file(p)}

    pairs = []
    skipped_no_stl = 0
    skipped_empty_py = 0
    skipped_len = 0

    for py in sorted(py_dir.glob("*.py")):
        if not is_nonempty_file(py):
            skipped_empty_py += 1
            continue

        if args.max_code_len > 0:
            try:
                code = py.read_text(encoding="utf-8", errors="ignore")
                if len(code) > args.max_code_len:
                    skipped_len += 1
                    continue
            except Exception:
                skipped_len += 1
                continue

        stl = stl_map.get(py.stem)
        if stl is None:
            skipped_no_stl += 1
            continue

        # materialize in out_dir with consistent names
        py_dst = out_py / py.name
        stl_dst = out_stl / (py.stem + ".stl")
        copier(py.resolve(), py_dst)
        copier(stl.resolve(), stl_dst)

        pairs.append((str(py_dst.resolve()), str(stl_dst.resolve())))

    print(f"Found pairs: {len(pairs)}")
    print(f"Skipped: no_stl={skipped_no_stl}, empty_py={skipped_empty_py}, too_long={skipped_len}")

    # write pairs.pkl
    pairs_pkl = out_dir / "pairs.pkl"
    with open(pairs_pkl, "wb") as f:
        pickle.dump(pairs, f)
    print("Wrote:", pairs_pkl)

    # optional split
    if args.val_size > 0:
        assert args.val_size < len(pairs), "val_size must be < number of pairs"
        rnd = random.Random(args.seed)
        idx = list(range(len(pairs)))
        rnd.shuffle(idx)
        val_idx = idx[:args.val_size]
        train_idx = idx[args.val_size:]

        train_pairs = [pairs[i] for i in train_idx]
        val_pairs = [pairs[i] for i in val_idx]

        with open(out_dir / "train.pkl", "wb") as f:
            pickle.dump(train_pairs, f)
        with open(out_dir / "val.pkl", "wb") as f:
            pickle.dump(val_pairs, f)

        print("Wrote:", out_dir / "train.pkl", "size", len(train_pairs))
        print("Wrote:", out_dir / "val.pkl", "size", len(val_pairs))

if __name__ == "__main__":
    main()