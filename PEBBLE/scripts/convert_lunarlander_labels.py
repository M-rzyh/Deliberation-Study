#!/usr/bin/env python3
"""Convert Lunar Lander human_labels.pkl + human_labels.csv into the NPZ + CSV
format that PEBBLE's offline label loader expects.

Input:
  - human_labels.pkl: contains sa_t_1, sa_t_2, labels, len_1, len_2
  - human_labels.csv: contains pair_index, batch_idx, pair_idx, label, time_sec

Output (in --output-dir):
  - batches/batch_000.npz, batch_001.npz, ...
  - human_labels_converted.csv with columns: query_id, label, decision_time
"""

import argparse
import csv
import os
import pickle

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pkl", required=True, help="Path to human_labels.pkl")
    parser.add_argument("--csv", required=True, help="Path to human_labels.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=25, help="Pairs per NPZ batch")
    args = parser.parse_args()

    with open(args.pkl, "rb") as f:
        pkl = pickle.load(f)

    sa_t_1 = pkl["sa_t_1"]  # (N, seg_len, obs+act)
    sa_t_2 = pkl["sa_t_2"]
    pkl_labels = pkl["labels"].flatten()  # (N,)
    pkl_len1 = pkl.get("len_1", np.full(len(pkl_labels), sa_t_1.shape[1], dtype=np.int32))
    pkl_len2 = pkl.get("len_2", np.full(len(pkl_labels), sa_t_2.shape[1], dtype=np.int32))
    if hasattr(pkl_len1, 'flatten'):
        pkl_len1 = pkl_len1.flatten().astype(np.int32)
        pkl_len2 = pkl_len2.flatten().astype(np.int32)

    rows = []
    with open(args.csv, "r") as f:
        for row in csv.DictReader(f):
            rows.append(row)

    print(f"PKL: {len(pkl_labels)} label pairs, CSV: {len(rows)} rows")

    # The PKL was built by filtering out None/skipped labels from the CSV,
    # and converting -1 (equal) to -1. We need to match them.
    # PKL has 985 entries (1000 - 15 skipped), CSV has 1000.
    # Build mapping: pkl_index -> csv_index
    pkl_idx = 0
    csv_to_pkl = {}
    for csv_idx, row in enumerate(rows):
        lbl = row["label"]
        if lbl == "None":
            continue
        csv_to_pkl[csv_idx] = pkl_idx
        pkl_idx += 1

    print(f"Mapped {len(csv_to_pkl)} CSV rows to PKL entries (PKL has {len(pkl_labels)})")

    os.makedirs(os.path.join(args.output_dir, "batches"), exist_ok=True)

    # Create NPZ batches and CSV
    csv_rows_out = []
    batch_sa1, batch_sa2, batch_qids, batch_len1, batch_len2 = [], [], [], [], []
    batch_count = 0

    for csv_idx, row in enumerate(rows):
        lbl = row["label"]
        if lbl == "None":
            continue

        pkl_i = csv_to_pkl[csv_idx]
        qid = f"ll_pair_{csv_idx:06d}"

        # Map label: CSV has 0, 1, -1. PEBBLE expects: 0=A, 1=B, -1=tie
        if lbl == "0":
            pebble_label = "A"
        elif lbl == "1":
            pebble_label = "B"
        elif lbl == "-1":
            pebble_label = "tie"
        else:
            continue

        time_sec = row.get("time_sec", "")

        csv_rows_out.append({
            "query_id": qid,
            "label": pebble_label,
            "choice": pebble_label,
            "decision_time": time_sec,
        })

        batch_sa1.append(sa_t_1[pkl_i])
        batch_sa2.append(sa_t_2[pkl_i])
        batch_qids.append(qid)
        batch_len1.append(pkl_len1[pkl_i])
        batch_len2.append(pkl_len2[pkl_i])

        if len(batch_sa1) >= args.batch_size:
            npz_path = os.path.join(args.output_dir, "batches", f"batch_{batch_count:03d}.npz")
            np.savez(
                npz_path,
                sa_t_1=np.array(batch_sa1, dtype=np.float32),
                sa_t_2=np.array(batch_sa2, dtype=np.float32),
                query_ids=np.array(batch_qids),
                len_1=np.array(batch_len1, dtype=np.int32),
                len_2=np.array(batch_len2, dtype=np.int32),
            )
            batch_count += 1
            batch_sa1, batch_sa2, batch_qids, batch_len1, batch_len2 = [], [], [], [], []

    # Save remaining
    if batch_sa1:
        npz_path = os.path.join(args.output_dir, "batches", f"batch_{batch_count:03d}.npz")
        np.savez(
            npz_path,
            sa_t_1=np.array(batch_sa1, dtype=np.float32),
            sa_t_2=np.array(batch_sa2, dtype=np.float32),
            query_ids=np.array(batch_qids),
            len_1=np.array(batch_len1, dtype=np.int32),
            len_2=np.array(batch_len2, dtype=np.int32),
        )
        batch_count += 1

    # Save CSV
    csv_path = os.path.join(args.output_dir, "human_labels_converted.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "label", "choice", "decision_time"])
        writer.writeheader()
        writer.writerows(csv_rows_out)

    print(f"Created {batch_count} NPZ batches in {args.output_dir}/batches/")
    print(f"Created CSV with {len(csv_rows_out)} rows: {csv_path}")


if __name__ == "__main__":
    main()
