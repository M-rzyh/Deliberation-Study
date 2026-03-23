import csv
import os
from datetime import datetime
from pathlib import Path

import numpy as np


class HumanQueryLogger:
    """Logs sampled preference queries so they can be labeled by humans later."""

    def __init__(
        self,
        save_dir,
        ds,
        job_id=None,
        separate_by_job_id=False,
        save_videos=False,
        video_dir=None,
        video_env_name='Walker2d-v4',
        video_fps=30,
        video_size=(800, 600),
    ):
        self.job_id = str(job_id).strip() if job_id is not None else ''
        self.separate_by_job_id = bool(separate_by_job_id)

        base_dir = Path(save_dir)
        if self.separate_by_job_id and self.job_id:
            self.save_dir = base_dir / self.job_id
        else:
            self.save_dir = base_dir

        self.ds = int(ds)
        self.save_videos = bool(save_videos)
        self.video_dir = Path(video_dir) if video_dir is not None else self.save_dir / 'videos'

        # If a custom video_dir is under the base dir, mirror it under the job subdir.
        if self.separate_by_job_id and self.job_id and video_dir is not None:
            try:
                rel_video = Path(video_dir).relative_to(base_dir)
                self.video_dir = self.save_dir / rel_video
            except Exception:
                pass

        self.batches_dir = self.save_dir / 'batches'
        self.batches_dir.mkdir(parents=True, exist_ok=True)
        self.video_dir.mkdir(parents=True, exist_ok=True)

        self.manifest_csv = self.save_dir / 'query_manifest.csv'
        self.labels_template_csv = self.save_dir / 'human_labels_template.csv'
        self._init_csv_files()

        info_path = self.save_dir / 'run_info.json'
        if not info_path.exists():
            with open(info_path, 'w') as f:
                f.write('{\n')
                f.write(f'  "save_dir": "{self.save_dir}",\n')
                f.write(f'  "job_id": "{self.job_id}",\n')
                f.write(f'  "separate_by_job_id": {str(self.separate_by_job_id).lower()}\n')
                f.write('}\n')

        self.batch_counter = 0

        self.video_generator = None
        if self.save_videos:
            try:
                from trajectory_video_generator import TrajectoryVideoGenerator

                self.video_generator = TrajectoryVideoGenerator(
                    env_name=video_env_name,
                    video_fps=video_fps,
                    video_size=video_size,
                )
            except Exception as e:
                # Common cluster case: Gym has Walker2d-v2/v3 but not v4
                fallback_envs = ["walker_walk", "Walker2d-v3", "Walker2d-v2"]
                initialized = False
                for env_name_try in fallback_envs:
                    try:
                        from trajectory_video_generator import TrajectoryVideoGenerator

                        self.video_generator = TrajectoryVideoGenerator(
                            env_name=env_name_try,
                            video_fps=video_fps,
                            video_size=video_size,
                        )
                        print(
                            f"[HumanQueryLogger] INFO: Video generator fallback activated: {env_name_try}"
                        )
                        initialized = True
                        break
                    except Exception:
                        pass

                if not initialized:
                    print(
                        f"[HumanQueryLogger] WARNING: Could not initialize video generation ({e}). "
                        "Continuing with query tensor logging only."
                    )
                    self.video_generator = None

    def _init_csv_files(self):
        if not self.manifest_csv.exists():
            with open(self.manifest_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'query_id',
                    'batch_id',
                    'query_index_in_batch',
                    'train_step',
                    'strategy',
                    'npz_path',
                    'video_a_path',
                    'video_b_path',
                    'segment_length',
                ])

        if not self.labels_template_csv.exists():
            with open(self.labels_template_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'query_id',
                    'label',
                    'confidence',
                    'notes',
                ])

    def log_batch(self, sa_t_1, sa_t_2, train_step=None, strategy='unknown'):
        """
        Save one batch of sampled query pairs.

        Args:
            sa_t_1: np.array [N, L, ds+da]
            sa_t_2: np.array [N, L, ds+da]
            train_step: current environment step
            strategy: query sampling strategy name
        """
        if sa_t_1 is None or sa_t_2 is None:
            return
        if len(sa_t_1) == 0:
            return

        self.batch_counter += 1
        step_str = str(train_step) if train_step is not None else 'na'
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        batch_id = f"step{step_str}_{strategy}_{ts}_{self.batch_counter:06d}"

        n_queries = int(sa_t_1.shape[0])
        seg_len = int(sa_t_1.shape[1])
        query_ids = np.array([f"{batch_id}_q{i:04d}" for i in range(n_queries)])

        npz_path = self.batches_dir / f"{batch_id}.npz"
        np.savez_compressed(
            npz_path,
            query_ids=query_ids,
            sa_t_1=sa_t_1.astype(np.float32),
            sa_t_2=sa_t_2.astype(np.float32),
            train_step=np.array([train_step if train_step is not None else -1], dtype=np.int64),
            strategy=np.array([strategy]),
        )

        manifest_rows = []
        label_rows = []

        for i, qid in enumerate(query_ids):
            qid = str(qid)
            video_a_rel = ''
            video_b_rel = ''

            if self.video_generator is not None:
                seg_a = sa_t_1[i]
                seg_b = sa_t_2[i]

                states_a = seg_a[:, :self.ds]
                actions_a = seg_a[:, self.ds:]
                states_b = seg_b[:, :self.ds]
                actions_b = seg_b[:, self.ds:]

                frames_a, _ = self.video_generator.render_trajectory(states_a, actions_a)
                video_a_abs = self.video_dir / f"{qid}_A.mp4"
                self.video_generator.save_video(frames_a, str(video_a_abs))

                self.video_generator.env.reset()

                frames_b, _ = self.video_generator.render_trajectory(states_b, actions_b)
                video_b_abs = self.video_dir / f"{qid}_B.mp4"
                self.video_generator.save_video(frames_b, str(video_b_abs))

                video_a_rel = os.path.relpath(video_a_abs, self.save_dir)
                video_b_rel = os.path.relpath(video_b_abs, self.save_dir)

            manifest_rows.append([
                qid,
                batch_id,
                i,
                train_step if train_step is not None else '',
                strategy,
                os.path.relpath(npz_path, self.save_dir),
                video_a_rel,
                video_b_rel,
                seg_len,
            ])

            label_rows.append([qid, '', '', ''])

        with open(self.manifest_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(manifest_rows)

        with open(self.labels_template_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(label_rows)

    def close(self):
        if self.video_generator is not None:
            self.video_generator.close()
            self.video_generator = None
