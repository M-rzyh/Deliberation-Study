#!/usr/bin/env python3
import argparse
import glob
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd

import utils
from reward_model import RewardModel


def parse_label(value):
    if pd.isna(value):
        return None
    s = str(value).strip().lower()
    if s in {'a', '0', 'left'}:
        return 0
    if s in {'b', '1', 'right'}:
        return 1
    if s in {'-1', 'tie', 'equal', 'same'}:
        return -1
    return None


def load_labeled_queries(query_dir, labels_csv):
    df = pd.read_csv(labels_csv)
    if 'query_id' not in df.columns or 'label' not in df.columns:
        raise ValueError('labels CSV must contain columns: query_id,label')

    label_map = {}
    for _, row in df.iterrows():
        qid = str(row['query_id'])
        lbl = parse_label(row['label'])
        if lbl is not None:
            label_map[qid] = lbl

    sa1_all, sa2_all, y_all = [], [], []

    npz_files = sorted(glob.glob(os.path.join(query_dir, 'batches', '*.npz')))
    for f in npz_files:
        data = np.load(f, allow_pickle=True)
        query_ids = data['query_ids']
        sa_t_1 = data['sa_t_1']
        sa_t_2 = data['sa_t_2']

        for i in range(len(query_ids)):
            qid = str(query_ids[i])
            if qid in label_map:
                sa1_all.append(sa_t_1[i])
                sa2_all.append(sa_t_2[i])
                y_all.append(label_map[qid])

    if len(y_all) == 0:
        raise RuntimeError('No labeled queries matched query_id entries in saved batches.')

    sa1 = np.asarray(sa1_all, dtype=np.float32)
    sa2 = np.asarray(sa2_all, dtype=np.float32)
    y = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)
    return sa1, sa2, y


def main():
    parser = argparse.ArgumentParser(description='Train PEBBLE reward model from offline human labels.')
    parser.add_argument('--env', type=str, default='walker_walk')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--query-dir', type=str, required=True)
    parser.add_argument('--labels-csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--ensemble-size', type=int, default=3)
    parser.add_argument('--reward-lr', type=float, default=3e-4)
    parser.add_argument('--segment', type=int, default=50)
    parser.add_argument('--reward-update', type=int, default=200)
    parser.add_argument('--label-margin', type=float, default=0.0)
    parser.add_argument('--capacity', type=int, default=500000)
    parser.add_argument('--save-step', type=str, default='human_offline')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = SimpleNamespace(env=args.env, seed=args.seed)
    env = utils.make_env(cfg)
    ds = int(env.observation_space.shape[0])
    da = int(env.action_space.shape[0])

    reward_model = RewardModel(
        ds,
        da,
        ensemble_size=args.ensemble_size,
        lr=args.reward_lr,
        size_segment=args.segment,
        label_margin=args.label_margin,
        capacity=args.capacity,
    )

    sa1, sa2, labels = load_labeled_queries(args.query_dir, args.labels_csv)
    reward_model.put_queries(sa1, sa2, labels)

    use_soft = bool(np.any(labels < 0) or args.label_margin > 0)
    for _ in range(args.reward_update):
        if use_soft:
            reward_model.train_soft_reward()
        else:
            reward_model.train_reward()

    reward_model.save(args.output_dir, args.save_step)
    print(f'Saved reward model to {args.output_dir} (step tag: {args.save_step})')
    print(f'Used {len(labels)} human-labeled queries.')


if __name__ == '__main__':
    main()
