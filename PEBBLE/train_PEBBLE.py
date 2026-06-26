#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.expanduser("~/compare_utils"))
from compare_logger import algo_dir, get_clock, Timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
#import os
#import sys
import time
import csv
import glob
import pickle as pkl
import tqdm
import imageio

from logger import Logger
from replay_buffer import ReplayBuffer
from reward_model import RewardModel
from collections import deque

import utils
import hydra
from human_query_logger import HumanQueryLogger

class Workspace(object):
    def __init__(self, cfg):
        # self.work_dir = os.getcwd()
        # print(f'workspace: {self.work_dir}')
        
        print("\n" + "="*60)
        print("ACTUAL CONFIG BEING USED:")
        print("="*60)
        print(f"env: {cfg.env}")
        print(f"device: {cfg.device}")
        print(f"num_train_steps: {cfg.num_train_steps}")
        print(f"num_unsup_steps: {cfg.num_unsup_steps}")
        print(f"num_interact: {cfg.num_interact}")
        print(f"max_feedback: {cfg.max_feedback}")
        print(f"reward_batch: {cfg.reward_batch}")
        print(f"reward_update: {cfg.reward_update}")
        print(f"feed_type: {cfg.feed_type}")
        print("="*60 + "\n")
        
        # Put PEBBLE logs under $COMPARE_RUN_DIR/pebble
        self.work_dir = algo_dir("pebble")
        print(f'workspace: {self.work_dir}')
        
        self.cfg = cfg
        self.logger = Logger(
            self.work_dir,
            save_tb=cfg.log_save_tb,
            log_frequency=cfg.log_frequency,
            agent="sac")

        self.clock = get_clock()  # shared TB at $COMPARE_RUN_DIR/common_tb
        
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.log_success = False
        
        # make env
        if 'metaworld' in cfg.env:
            self.env = utils.make_metaworld_env(cfg)
            self.log_success = True
        else:
            self.env = utils.make_env(cfg)
        
        agent_cfg = cfg.agent.agent if 'agent' in cfg.agent else cfg.agent
        agent_params = agent_cfg.params if 'params' in agent_cfg else agent_cfg
        agent_params.obs_dim = self.env.observation_space.shape[0]
        agent_params.action_dim = self.env.action_space.shape[0]
        agent_params.action_range = [
            float(self.env.action_space.low.min()),
            float(self.env.action_space.high.max())
        ]
        self.agent = hydra.utils.instantiate(agent_cfg)

        self.replay_buffer = ReplayBuffer(
            self.env.observation_space.shape,
            self.env.action_space.shape,
            int(cfg.replay_buffer_capacity),
            self.device)
        
        # for logging
        self.total_feedback = 0
        self.labeled_feedback = 0
        self.step = 0
        self.last_pref_sec = 0.0
        self.last_pref_pairs = 0
        self.pref_time_so_far_sec = 0.0
        self.last_logged_labeled_feedback = 0

        # instantiating the reward model
        self.query_logger = None
        # offline human-label loading state (must exist regardless of query collection mode)
        # We cache all matched offline labels once, then inject them gradually in mb_size chunks
        # to mimic online preference collection/update timing.
        self.offline_human_labels_prepared = False
        self.offline_human_loaded_count = 0
        self.offline_human_next_index = 0
        self.offline_sa1_all = None
        self.offline_sa2_all = None
        self.offline_y_all = None
        self.online_human_labels_csv = str(getattr(cfg, 'online_human_labels_csv', '') or '').strip()
        self.online_human_poll_interval_sec = float(getattr(cfg, 'online_human_poll_interval_sec', 5.0))
        self.online_human_timeout_sec = float(getattr(cfg, 'online_human_timeout_sec', 0.0))
        if getattr(cfg, 'collect_human_queries', False):
            requested_job_id = str(getattr(cfg, 'human_query_job_id', '') or '').strip()
            env_job_key = str(getattr(cfg, 'human_query_job_id_env', 'SLURM_JOB_ID'))
            env_job_id = str(os.environ.get(env_job_key, '')).strip()
            active_job_id = requested_job_id or env_job_id

            base_query_dir = str(getattr(cfg, 'human_query_dir', 'human_queries'))
            effective_query_dir = os.path.join(base_query_dir, active_job_id) if (
                getattr(cfg, 'human_query_separate_by_job_id', False) and active_job_id
            ) else base_query_dir

            base_video_dir = str(getattr(cfg, 'human_query_video_dir', os.path.join(base_query_dir, 'videos')))
            if (
                getattr(cfg, 'human_query_separate_by_job_id', False)
                and active_job_id
                and base_video_dir.startswith(base_query_dir)
            ):
                suffix = base_video_dir[len(base_query_dir):].lstrip('/\\')
                effective_video_dir = os.path.join(effective_query_dir, suffix) if suffix else effective_query_dir
            else:
                effective_video_dir = base_video_dir

            print(f"[HumanQueryLogger] base_dir={base_query_dir} effective_dir={effective_query_dir}")
            if active_job_id:
                print(f"[HumanQueryLogger] job_id={active_job_id}")

            self.query_logger = HumanQueryLogger(
                save_dir=base_query_dir,
                ds=self.env.observation_space.shape[0],
                job_id=active_job_id,
                separate_by_job_id=getattr(cfg, 'human_query_separate_by_job_id', False),
                save_videos=getattr(cfg, 'human_query_save_videos', False),
                video_dir=effective_video_dir,
                video_env_name=getattr(cfg, 'human_query_video_env', 'walker_walk'),
                video_fps=getattr(cfg, 'human_query_video_fps', 30),
                video_size=(getattr(cfg, 'human_query_video_width', 1280), getattr(cfg, 'human_query_video_height', 720)),
            )

            if getattr(cfg, 'use_online_human_labels', False):
                if not self.online_human_labels_csv:
                    self.online_human_labels_csv = str(self.query_logger.labels_template_csv)
                elif not os.path.isabs(self.online_human_labels_csv):
                    self.online_human_labels_csv = os.path.join(str(self.query_logger.save_dir), self.online_human_labels_csv)
                print(f"[online_human] labels_csv={self.online_human_labels_csv}")

        self.reward_model = RewardModel(
            self.env.observation_space.shape[0],
            self.env.action_space.shape[0],
            ensemble_size=cfg.ensemble_size,
            size_segment=cfg.segment,
            activation=cfg.activation, 
            lr=cfg.reward_lr,
            mb_size=cfg.reward_batch, 
            large_batch=cfg.large_batch, 
            label_margin=cfg.label_margin, 
            teacher_beta=cfg.teacher_beta, 
            teacher_gamma=cfg.teacher_gamma, 
            teacher_eps_mistake=cfg.teacher_eps_mistake, 
            teacher_eps_skip=cfg.teacher_eps_skip, 
            teacher_eps_equal=cfg.teacher_eps_equal,
            query_logger=self.query_logger)

        # optional: save last few training episodes as reference videos
        self.save_last_train_episode_videos = bool(getattr(cfg, 'save_last_train_episode_videos', False))
        self.last_train_video_count = int(getattr(cfg, 'last_train_video_count', 5))
        self.last_train_video_fps = int(getattr(cfg, 'last_train_video_fps', 30))
        self.last_train_video_dir = os.path.join(
            self.work_dir,
            str(getattr(cfg, 'last_train_video_dir', 'train_episode_videos')),
        )
        self._current_episode_frames = []
        self._capture_current_episode = False
        self._max_episode_steps = utils.get_env_horizon(self.env, default=1000)
        self._video_capture_start_step = max(
            0,
            int(self.cfg.num_train_steps) - self.last_train_video_count * self._max_episode_steps,
        )
        if self.save_last_train_episode_videos:
            os.makedirs(self.last_train_video_dir, exist_ok=True)
            print(
                f"[train_video] enabled: saving last {self.last_train_video_count} training episodes "
                f"to {self.last_train_video_dir}"
            )

    def _get_env_frame(self):
        try:
            frame = self.env.render(mode='rgb_array')
        except TypeError:
            frame = self.env.render()
        except Exception:
            return None

        if frame is None:
            return None
        return np.asarray(frame)

    def _append_episode_frame_if_needed(self):
        if not self._capture_current_episode:
            return
        frame = self._get_env_frame()
        if frame is not None:
            self._current_episode_frames.append(frame)

    def _save_episode_video_if_needed(self, episode_idx):
        if not self._capture_current_episode:
            return
        if len(self._current_episode_frames) == 0:
            return

        video_path = os.path.join(
            self.last_train_video_dir,
            f"episode_{int(episode_idx):06d}_step_{int(self.step):07d}.mp4",
        )
        try:
            imageio.mimsave(video_path, self._current_episode_frames, fps=self.last_train_video_fps)
            print(f"[train_video] saved: {video_path}")
        except Exception as e:
            print(f"[train_video] WARNING: failed to save video {video_path}: {e}")

    @staticmethod
    def _parse_human_label(value):
        s = str(value).strip().lower()
        if s in ['a', '0', 'left']:
            return 0
        if s in ['b', '1', 'right']:
            return 1
        if s in ['-1', 'tie', 'equal', 'same']:
            return -1
        return None

    def _prepare_offline_human_labels_once(self):
        if self.offline_human_labels_prepared:
            return self.offline_human_loaded_count

        query_dir = str(self.cfg.offline_human_query_dir)
        labels_csv = str(self.cfg.offline_human_labels_csv)

        offline_job_id = str(getattr(self.cfg, 'offline_human_job_id', '') or '').strip()
        if offline_job_id:
            query_dir = os.path.join(query_dir, offline_job_id)
            labels_basename = os.path.basename(labels_csv)
            labels_csv = os.path.join(query_dir, labels_basename)
            print(f"[offline_human] using job-scoped query_dir={query_dir}")
            print(f"[offline_human] using labels_csv={labels_csv}")

        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"offline_human_labels_csv not found: {labels_csv}")

        label_map = {}
        time_map = {}
        with open(labels_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            if 'query_id' not in reader.fieldnames or 'label' not in reader.fieldnames:
                raise ValueError("offline_human_labels_csv must include columns: query_id,label")
            for row in reader:
                qid = str(row.get('query_id', '')).strip()
                lbl = self._parse_human_label(row.get('label', ''))
                if qid and lbl is not None:
                    label_map[qid] = lbl
                    dt = row.get('decision_time', row.get('time_sec', ''))
                    try:
                        time_map[qid] = float(dt)
                    except (ValueError, TypeError):
                        time_map[qid] = None

        if len(label_map) == 0:
            raise RuntimeError("No valid labels found in offline_human_labels_csv")

        npz_files = sorted(glob.glob(os.path.join(query_dir, 'batches', '*.npz')))
        if len(npz_files) == 0:
            raise RuntimeError(f"No query batch files found under: {query_dir}/batches")

        seg_size = self.reward_model.size_segment
        sa1_all, sa2_all, y_all, t_all, len1_all, len2_all = [], [], [], [], [], []
        for npz_path in npz_files:
            data = np.load(npz_path, allow_pickle=True)
            query_ids = data['query_ids']
            sa_t_1 = data['sa_t_1']
            sa_t_2 = data['sa_t_2']
            npz_len1 = data['len_1'] if 'len_1' in data else None
            npz_len2 = data['len_2'] if 'len_2' in data else None

            for i in range(len(query_ids)):
                qid = str(query_ids[i])
                if qid in label_map:
                    sa1_all.append(sa_t_1[i])
                    sa2_all.append(sa_t_2[i])
                    y_all.append(label_map[qid])
                    t_all.append(time_map.get(qid))
                    len1_all.append(int(npz_len1[i]) if npz_len1 is not None else seg_size)
                    len2_all.append(int(npz_len2[i]) if npz_len2 is not None else seg_size)

        if len(y_all) == 0:
            raise RuntimeError("No labeled query_id matched any saved query batches")

        sa1 = np.asarray(sa1_all, dtype=np.float32)
        sa2 = np.asarray(sa2_all, dtype=np.float32)
        labels = np.asarray(y_all, dtype=np.float32).reshape(-1, 1)

        # Compute time-based weights
        strategy = str(getattr(self.cfg, 'time_weight_strategy', 'none')).strip().lower()
        if strategy != 'none' and any(t is not None for t in t_all):
            times = np.array([t if t is not None else np.nan for t in t_all], dtype=np.float32)
            valid = ~np.isnan(times)
            if valid.sum() > 0:
                median_t = np.median(times[valid])
                if strategy == 'linear':
                    raw_w = times / median_t
                elif strategy == 'sqrt':
                    raw_w = np.sqrt(times) / np.sqrt(median_t)
                elif strategy == 'log':
                    raw_w = np.log1p(times) / np.log1p(median_t)
                else:
                    raw_w = np.ones_like(times)
                raw_w[~valid] = 1.0
                weights = (raw_w / np.mean(raw_w[valid])).reshape(-1, 1)
                print(f"[time_weight] strategy={strategy}, median_time={median_t:.2f}s, "
                      f"weight range=[{weights.min():.3f}, {weights.max():.3f}]")
            else:
                weights = np.ones((len(y_all), 1), dtype=np.float32)
                print("[time_weight] no valid decision times found, using uniform weights")
        else:
            weights = np.ones((len(y_all), 1), dtype=np.float32)
            if strategy != 'none':
                print(f"[time_weight] strategy={strategy} but no timing data in CSV")

        len1 = np.array(len1_all, dtype=np.int32)
        len2 = np.array(len2_all, dtype=np.int32)

        self.offline_sa1_all = sa1
        self.offline_sa2_all = sa2
        self.offline_y_all = labels
        self.offline_weights_all = weights
        self.offline_len1_all = len1
        self.offline_len2_all = len2
        self.offline_human_labels_prepared = True
        self.offline_human_loaded_count = len(labels)
        self.offline_human_next_index = 0

        print(f"Loaded offline human labels (pool): {self.offline_human_loaded_count}")
        return self.offline_human_loaded_count

    def _inject_offline_human_label_batch(self):
        self._prepare_offline_human_labels_once()

        if self.offline_human_next_index >= self.offline_human_loaded_count:
            return 0

        batch_size = int(max(1, self.reward_model.mb_size))
        offline_budget = int(getattr(self.cfg, 'offline_human_max_labels', 0) or 0)
        if offline_budget > 0:
            budget_remaining = max(0, offline_budget - self.offline_human_next_index)
            if budget_remaining <= 0:
                return 0
            batch_size = min(batch_size, budget_remaining)

        start = self.offline_human_next_index
        end = min(start + batch_size, self.offline_human_loaded_count)

        self.reward_model.put_queries(
            self.offline_sa1_all[start:end],
            self.offline_sa2_all[start:end],
            self.offline_y_all[start:end],
            weights=self.offline_weights_all[start:end],
            len_1=self.offline_len1_all[start:end],
            len_2=self.offline_len2_all[start:end],
        )
        self.offline_human_next_index = end

        injected = end - start
        remaining = self.offline_human_loaded_count - self.offline_human_next_index
        print(
            f"Injected offline human labels batch: {injected} "
            f"(total injected: {self.offline_human_next_index}/{self.offline_human_loaded_count}, "
            f"remaining: {remaining})"
        )
        return injected

    def _sample_synthetic_labels(self, first_flag=0):
        if first_flag == 1:
            return self.reward_model.uniform_sampling(train_step=self.step)

        if self.cfg.feed_type == 0:
            return self.reward_model.uniform_sampling(train_step=self.step)
        elif self.cfg.feed_type == 1:
            return self.reward_model.disagreement_sampling(train_step=self.step)
        elif self.cfg.feed_type == 2:
            return self.reward_model.entropy_sampling(train_step=self.step)
        elif self.cfg.feed_type == 3:
            return self.reward_model.kcenter_sampling(train_step=self.step)
        elif self.cfg.feed_type == 4:
            return self.reward_model.kcenter_disagree_sampling(train_step=self.step)
        elif self.cfg.feed_type == 5:
            return self.reward_model.kcenter_entropy_sampling(train_step=self.step)
        else:
            raise NotImplementedError

    def _get_query_strategy(self, first_flag=0):
        if first_flag == 1:
            return 'uniform'

        if self.cfg.feed_type == 0:
            return 'uniform'
        elif self.cfg.feed_type == 1:
            return 'disagreement'
        elif self.cfg.feed_type == 2:
            return 'entropy'
        elif self.cfg.feed_type == 3:
            return 'kcenter'
        elif self.cfg.feed_type == 4:
            return 'kcenter_disagree'
        elif self.cfg.feed_type == 5:
            return 'kcenter_entropy'
        else:
            raise NotImplementedError

    def _sample_online_human_labels(self, first_flag=0):
        if self.query_logger is None:
            raise RuntimeError('use_online_human_labels=true requires collect_human_queries=true')
        if not self.online_human_labels_csv:
            raise RuntimeError('online_human_labels_csv is required for online human labeling mode')

        strategy = self._get_query_strategy(first_flag=first_flag)
        sa_t_1, sa_t_2, query_ids = self.reward_model.sample_queries_for_human(
            strategy=strategy,
            train_step=self.step,
        )

        if len(query_ids) != len(sa_t_1):
            print(
                f"[online_human] WARNING: query_id count mismatch "
                f"({len(query_ids)} ids for {len(sa_t_1)} queries); skipping this batch"
            )
            return 0

        pending_ids = set(query_ids)
        collected = {}
        wait_start = time.time()
        next_log_time = 0.0

        while len(pending_ids) > 0:
            if os.path.exists(self.online_human_labels_csv):
                with open(self.online_human_labels_csv, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames and 'query_id' in reader.fieldnames and 'label' in reader.fieldnames:
                        for row in reader:
                            qid = str(row.get('query_id', '')).strip()
                            if qid not in pending_ids:
                                continue
                            lbl = self._parse_human_label(row.get('label', ''))
                            if lbl is None:
                                continue
                            collected[qid] = lbl

                for qid in list(pending_ids):
                    if qid in collected:
                        pending_ids.remove(qid)

            if len(pending_ids) == 0:
                break

            elapsed = time.time() - wait_start
            if self.online_human_timeout_sec > 0 and elapsed >= self.online_human_timeout_sec:
                print(
                    f"[online_human] timeout after {elapsed:.1f}s; "
                    f"received {len(collected)}/{len(query_ids)} labels"
                )
                break

            if elapsed >= next_log_time:
                print(
                    f"[online_human] waiting for labels in {self.online_human_labels_csv}: "
                    f"{len(collected)}/{len(query_ids)} received"
                )
                next_log_time += 30.0

            time.sleep(max(0.1, self.online_human_poll_interval_sec))

        selected_indices = []
        selected_labels = []
        ties_or_invalid = 0
        for idx, qid in enumerate(query_ids):
            if qid not in collected:
                continue
            lbl = int(collected[qid])
            if lbl not in (0, 1):
                ties_or_invalid += 1
                continue
            selected_indices.append(idx)
            selected_labels.append(lbl)

        if ties_or_invalid > 0:
            print(f"[online_human] skipped {ties_or_invalid} tie/invalid labels (expected 0 or 1)")

        if len(selected_indices) == 0:
            return 0

        sa_selected_1 = sa_t_1[selected_indices]
        sa_selected_2 = sa_t_2[selected_indices]
        y_selected = np.asarray(selected_labels, dtype=np.float32).reshape(-1, 1)

        self.reward_model.put_queries(sa_selected_1, sa_selected_2, y_selected)
        print(f"[online_human] injected {len(selected_indices)} human labels")
        return len(selected_indices)
        
    def evaluate(self):
        average_episode_reward = 0
        average_true_episode_reward = 0
        success_rate = 0
        
        for episode in range(self.cfg.num_eval_episodes):
            obs = self.env.reset()
            # self.agent.reset()  # Not needed
            done = False
            episode_reward = 0
            true_episode_reward = 0
            if self.log_success:
                episode_success = 0

            while not done:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=False)
                obs, reward, done, extra = self.env.step(action)
                
                episode_reward += reward
                true_episode_reward += reward
                if self.log_success:
                    episode_success = max(episode_success, extra['success'])
                
            average_episode_reward += episode_reward
            average_true_episode_reward += true_episode_reward
            if self.log_success:
                success_rate += episode_success
            
        average_episode_reward /= self.cfg.num_eval_episodes
        average_true_episode_reward /= self.cfg.num_eval_episodes
        if self.log_success:
            success_rate /= self.cfg.num_eval_episodes
            success_rate *= 100.0
        
        self.logger.log('eval/episode_reward', average_episode_reward,
                        self.step)
        self.logger.log('eval/true_episode_reward', average_true_episode_reward,
                        self.step)
        if self.log_success:
            self.logger.log('eval/success_rate', success_rate,
                    self.step)
            self.logger.log('train/true_episode_success', success_rate,
                        self.step)
        self.logger.dump(self.step)
    
    def learn_reward(self, first_flag=0):
        # --- preference query generation + labeling timing ---
        labeled_queries, noisy_queries = 0, 0
        used_synthetic = False
        if getattr(self.cfg, 'use_online_human_labels', False):
            with Timer() as t_pref:
                labeled_queries = int(self._sample_online_human_labels(first_flag=first_flag))
            pref_sec = t_pref.dt

            self.total_feedback += int(self.reward_model.mb_size)
            self.labeled_feedback += int(labeled_queries)
        elif getattr(self.cfg, 'use_offline_human_labels', False):
            with Timer() as t_pref:
                newly_loaded = self._inject_offline_human_label_batch()
                if newly_loaded > 0:
                    labeled_queries = int(newly_loaded)
                elif getattr(self.cfg, 'offline_human_continue_with_synthetic', False):
                    labeled_queries = int(self._sample_synthetic_labels(first_flag=first_flag))
                    used_synthetic = True
                    if labeled_queries > 0:
                        print(f"[offline_human] exhausted/budget-reached, switching to synthetic labels: {labeled_queries}")
            pref_sec = t_pref.dt

            if newly_loaded > 0:
                self.total_feedback += int(newly_loaded)
                self.labeled_feedback += int(newly_loaded)
            elif used_synthetic:
                self.total_feedback += self.reward_model.mb_size
                self.labeled_feedback += labeled_queries
        else:
            with Timer() as  t_pref:
                labeled_queries = int(self._sample_synthetic_labels(first_flag=first_flag))

            pref_sec = t_pref.dt
        
        # # get feedbacks
        # labeled_queries, noisy_queries = 0, 0
        # if first_flag == 1:
        #     # if it is first time to get feedback, need to use random sampling
        #     labeled_queries = self.reward_model.uniform_sampling()
        # else:
        #     if self.cfg.feed_type == 0:
        #         labeled_queries = self.reward_model.uniform_sampling()
        #     elif self.cfg.feed_type == 1:
        #         labeled_queries = self.reward_model.disagreement_sampling()
        #     elif self.cfg.feed_type == 2:
        #         labeled_queries = self.reward_model.entropy_sampling()
        #     elif self.cfg.feed_type == 3:
        #         labeled_queries = self.reward_model.kcenter_sampling()
        #     elif self.cfg.feed_type == 4:
        #         labeled_queries = self.reward_model.kcenter_disagree_sampling()
        #     elif self.cfg.feed_type == 5:
        #         labeled_queries = self.reward_model.kcenter_entropy_sampling()
        #     else:
        #         raise NotImplementedError
        
        if (
            (not getattr(self.cfg, 'use_offline_human_labels', False))
            and (not getattr(self.cfg, 'use_online_human_labels', False))
        ):
            self.total_feedback += self.reward_model.mb_size
            self.labeled_feedback += labeled_queries
        query_number = self.total_feedback // self.cfg.reward_batch
        print(f"Total preference labels so far: {self.total_feedback}, Query number: {query_number}")
        
        seg_len = int(self.cfg.segment)             # L
        pairs = int(labeled_queries)               # number of preference labels created now
        segment_steps = 2 * seg_len * pairs        # 2*L per preference
        
        self.last_pref_sec = float(pref_sec)
        self.last_pref_pairs = int(pairs)
        self.pref_time_so_far_sec = (self.last_pref_sec / max(1, self.last_pref_pairs)) * self.labeled_feedback
        
        # self.clock.log_scalar("clock/pebble_pref_batch_seconds", pref_sec, self.step)
        # self.clock.log_scalar("samples/pebble_pref_pairs", float(pairs), self.step)
        # self.clock.log_scalar("samples/pebble_segment_len", float(seg_len), self.step)
        # self.clock.log_scalar("samples/pebble_pref_segment_steps", float(segment_steps), self.step)

        # self.clock.log_scalar("true_reward/sample", self.true_episode_reward,(pref_sec / max(1, pairs))*len(self.labeled_feedback))
        # self.clock.log_scalar("reward/sample", self.episode_reward,(pref_sec / max(1, pairs))*len(self.labeled_feedback))

        # self.clock.log_scalar(
        #     "clock/pebble_pref_sec_per_segment_step",
        #     pref_sec / max(1, segment_steps),
        #     self.step
        # )
        self.clock.flush()
        
        train_acc = 0
        total_acc = 0.0
        if self.labeled_feedback > 0:
            # update reward
            for epoch in range(self.cfg.reward_update):
                if self.cfg.label_margin > 0 or self.cfg.teacher_eps_equal > 0:
                    train_acc = self.reward_model.train_soft_reward()
                else:
                    train_acc = self.reward_model.train_reward()
                total_acc = np.mean(train_acc)
                
                if total_acc > 0.97:
                    break;
                    
        print("Reward function is updated!! ACC: " + str(total_acc))

    def run(self):
        episode, episode_reward, done = 0, 0, True
        episode_step = 0
        if self.log_success:
            episode_success = 0
        true_episode_reward = 0
        
        # store train returns of recent 10 episodes
        avg_train_true_return = deque([], maxlen=10)
        avg_episode_length = deque([], maxlen=10)
        start_time = time.time()
        env_time_acc = 0.0
        env_steps_acc = 0
        ENV_LOG_EVERY = 1000
        HEARTBEAT_EVERY = 1000

        interact_count = 0
        while self.step < self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    self.logger.log('train/duration', time.time() - start_time, self.step)
                    start_time = time.time()
                    self.logger.dump(
                        self.step, save=(self.step > self.cfg.num_seed_steps))

                # evaluate agent periodically
                if self.step > 0 and self.step % self.cfg.eval_frequency == 0:
                    self.logger.log('eval/episode', episode, self.step)
                    self.evaluate()
                
                self.logger.log('train/episode_reward', episode_reward, self.step)
                self.logger.log('train/true_episode_reward', true_episode_reward, self.step)
                
                # rough clock timing
                if self.labeled_feedback > self.last_logged_labeled_feedback:
                    print("\nLogging to TB: total feedback", self.total_feedback, "labeled feedback", self.labeled_feedback, "pref time so far (sec)", self.pref_time_so_far_sec)
                    x_ms = int(self.pref_time_so_far_sec * 1000)

                    self.clock.log_scalar("1: true_reward/sample", true_episode_reward, x_ms)
                    self.clock.log_scalar("1: reward/sample", episode_reward, x_ms)
                    self.clock.flush()

                    self.last_logged_labeled_feedback = self.labeled_feedback
                
                self.clock.log_scalar("2: true_reward/sample", true_episode_reward,int(((self.last_pref_sec / max(1, self.last_pref_pairs))*(self.labeled_feedback))))
                self.clock.log_scalar("2: reward/sample", episode_reward,int(((self.last_pref_sec / max(1, self.last_pref_pairs))*(self.labeled_feedback))))
                
                self.logger.log('train/total_feedback', self.total_feedback, self.step)
                self.logger.log('train/labeled_feedback', self.labeled_feedback, self.step)
                
                if self.log_success:
                    self.logger.log('train/episode_success', episode_success,
                        self.step)
                    self.logger.log('train/true_episode_success', episode_success,
                        self.step)

                if self.step > 0:
                    self._save_episode_video_if_needed(episode)
                    self._current_episode_frames = []
                    self._capture_current_episode = False
                
                obs = self.env.reset()
                # self.agent.reset()  # Not needed
                done = False
                episode_reward = 0
                avg_train_true_return.append(true_episode_reward)
                avg_episode_length.append(episode_step)
                true_episode_reward = 0
                if self.log_success:
                    episode_success = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

                if self.save_last_train_episode_videos and self.step >= self._video_capture_start_step:
                    self._capture_current_episode = True
                    self._append_episode_frame_if_needed()
                        
            # sample action for data collection
            if self.step < self.cfg.num_seed_steps:
                action = self.env.action_space.sample()
            else:
                with utils.eval_mode(self.agent):
                    action = self.agent.act(obs, sample=True)

            # run training update                
            if self.step == (self.cfg.num_seed_steps + self.cfg.num_unsup_steps):
                # update schedule
                if self.cfg.reward_schedule == 1:
                    frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                    if frac == 0:
                        frac = 0.01
                elif self.cfg.reward_schedule == 2:
                    frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                else:
                    frac = 1
                self.reward_model.change_batch(frac)
                
                # update margin --> not necessary / will be updated soon
                _horizon = np.mean(avg_episode_length) if len(avg_episode_length) > 0 else max(1, int(self.cfg.segment))
                _horizon = max(_horizon, 1)
                new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / _horizon)
                self.reward_model.set_teacher_thres_skip(new_margin)
                self.reward_model.set_teacher_thres_equal(new_margin)
                
                # first learn reward
                self.learn_reward(first_flag=1)
                
                # relabel buffer
                self.replay_buffer.relabel_with_predictor(self.reward_model)
                
                # reset Q due to unsuperivsed exploration
                self.agent.reset_critic()
                
                # update agent
                self.agent.update_after_reset(
                    self.replay_buffer, self.logger, self.step, 
                    gradient_update=self.cfg.reset_update, 
                    policy_update=True)
                
                # reset interact_count
                interact_count = 0
            elif self.step > self.cfg.num_seed_steps + self.cfg.num_unsup_steps:
                # update reward function
                if self.total_feedback < self.cfg.max_feedback:
                    if interact_count == self.cfg.num_interact:
                        # update schedule
                        if self.cfg.reward_schedule == 1:
                            frac = (self.cfg.num_train_steps-self.step) / self.cfg.num_train_steps
                            if frac == 0:
                                frac = 0.01
                        elif self.cfg.reward_schedule == 2:
                            frac = self.cfg.num_train_steps / (self.cfg.num_train_steps-self.step +1)
                        else:
                            frac = 1
                        self.reward_model.change_batch(frac)
                        
                        # update margin --> not necessary / will be updated soon
                        _horizon = np.mean(avg_episode_length) if len(avg_episode_length) > 0 else max(1, int(self.cfg.segment))
                        _horizon = max(_horizon, 1)
                        new_margin = np.mean(avg_train_true_return) * (self.cfg.segment / _horizon)
                        self.reward_model.set_teacher_thres_skip(new_margin * self.cfg.teacher_eps_skip)
                        self.reward_model.set_teacher_thres_equal(new_margin * self.cfg.teacher_eps_equal)
                        
                        # corner case: new total feed > max feed
                        if self.reward_model.mb_size + self.total_feedback > self.cfg.max_feedback:
                            self.reward_model.set_batch(self.cfg.max_feedback - self.total_feedback)
                            
                        self.learn_reward()
                        self.replay_buffer.relabel_with_predictor(self.reward_model)
                        interact_count = 0
                        
                self.agent.update(self.replay_buffer, self.logger, self.step, 1)
                
            # unsupervised exploration
            elif self.step > self.cfg.num_seed_steps:
                self.agent.update_state_ent(self.replay_buffer, self.logger, self.step, 
                                            gradient_update=1, K=self.cfg.topK)
                
            # next_obs, reward, done, extra = self.env.step(action)
            t0 = time.perf_counter()
            next_obs, reward, done, extra = self.env.step(action)
            dt = time.perf_counter() - t0
            env_time_acc += dt
            env_steps_acc += 1
            if env_steps_acc >= ENV_LOG_EVERY:
                self.clock.log_scalar("clock/env_step_sec", env_time_acc / env_steps_acc, self.step)
                self.clock.flush()
                env_time_acc = 0.0
                env_steps_acc = 0
                
            reward_hat = self.reward_model.r_hat(np.concatenate([obs, action], axis=-1))

            # allow infinite bootstrap
            done = float(done)
            if self._max_episode_steps is None:
                done_no_max = done
            else:
                done_no_max = 0 if episode_step + 1 == self._max_episode_steps else done
            episode_reward += reward_hat
            true_episode_reward += reward
            
            if self.log_success:
                episode_success = max(episode_success, extra['success'])
                
            # adding data to the reward training data
            self.reward_model.add_data(obs, action, reward, done)
            self.replay_buffer.add(
                obs, action, reward_hat, 
                next_obs, done, done_no_max)

            obs = next_obs
            episode_step += 1
            self.step += 1
            interact_count += 1
            self._append_episode_frame_if_needed()

            if self.step % HEARTBEAT_EVERY == 0:
                print(
                    f"[heartbeat] step={self.step} episode={episode} "
                    f"ep_step={episode_step} total_feedback={self.total_feedback} "
                    f"labeled_feedback={self.labeled_feedback}",
                    flush=True,
                )

        if self._capture_current_episode and len(self._current_episode_frames) > 0:
            self._save_episode_video_if_needed(episode)
            
        self.agent.save(self.work_dir, self.step)
        self.reward_model.save(self.work_dir, self.step)
        if self.query_logger is not None:
            self.query_logger.close()
    
@hydra.main(config_path='config/train_PEBBLE.yaml', strict=True)
def main(cfg):
    workspace = Workspace(cfg)
    workspace.run()

if __name__ == '__main__':
    main()
