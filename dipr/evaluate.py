import os
import sys
import glob
import tqdm
import argparse
import numpy as np
import termcolor

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import utils as u

from tqdm import tqdm
from pathlib import Path
from scipy.spatial.transform import Rotation
from dataset import StatesArray, CnnPreds, ImuArray, ArcturusDataset
from cnn_backend import TorchScriptBackend
from imu_fallback import ImuFallback

from typing import Tuple, List, Any, Union, Optional


def _concat_states_with_nans(states: Tuple[Union[StatesArray, CnnPreds]]) -> Optional[StatesArray]:
    if None in states:
        return None

    sep = np.ones_like(states[0][:1].data) * np.nan
    return type(states[0])(np.r_[tuple(e for s in states for e in [s.data, sep])[:-1]])


class Evaluator(object):

    minimal_required_imu_history_seconds = 5  # fixing this to filter out segments at very beginning for test dataset

    def __init__(self, config: argparse.Namespace):
        self.config = config

        test_data_folder = os.path.join(config.data_folder, 'test_synthetic')
        self.hdf5_files = sorted(glob.glob(os.path.join(test_data_folder, '*.hdf5')))
        assert len(self.hdf5_files) > 0, 'test data folder is empty, please check paths in command line parameters'

        self.backend = TorchScriptBackend(config.model_path)
        self.window_time = self.backend.window_size / self.backend.target_imu_rate

        from noise_utils import NoiseModel
        self.noise_model = NoiseModel()

    def plot_cnn_preds(self, dateset_name, cnn_preds: CnnPreds) -> List[np.ndarray]:
        beg, end = cnn_preds.times[[0, -1]]
        xbeg, xend = 0, (end - beg) + 5  # let's pad 5 seconds for legend
        figsize, dpi = (33, 12), 150
        t = cnn_preds.times - beg

        colors, ltype = dict(gt_vel='r', cnn_vel='b', cnn_error='y', cnn_3sigma='m'), dict(cnn_error='--', cnn_3sigma='--')
        renames = dict(cnn_vel='CNN velocity', gt_vel='GT velocity')

        selectors = dict(
            gt_vel=lambda x: x.gt, cnn_vel=lambda x: x.vel,
            cnn_error=lambda x: x.gt - x.vel, cnn_3sigma=lambda x: 3 * x.sigma,
            cnn_3sigma_max=lambda x: x.vel + 3 * x.sigma,
            cnn_3sigma_min=lambda x: x.vel - 3 * x.sigma,
        )
        up, down = cnn_preds.vel + 3 * cnn_preds.sigma, cnn_preds.vel - 3 * cnn_preds.sigma

        title = 'CNN Prediction (velocity in local frame)'
        fig = plt.figure(num=title, figsize=figsize, dpi=dpi)
        plt.suptitle(f'{title}, {dateset_name}', fontsize=22)

        for i, coo in enumerate('xyz'):
            ax = plt.subplot(3, 1, i + 1)
            ax.yaxis.tick_right()
            plt.fill_between(t, up[:, i], down[:, i], color='g', alpha=0.1)

            for n in ['gt_vel', 'cnn_vel']:
                plt.plot(t, selectors[n](cnn_preds)[:, i], label=f'{renames.get(n, n)} {coo} ', color=colors[n], linestyle=ltype.get(n, '-'), linewidth=1)

            plt.xlim(xbeg, xend)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
            plt.legend(loc='upper right')
            plt.grid(True)
        plt.xlabel('t(sec) from the beginning of the first fallback', fontsize=16)
        plt.tight_layout()
        return [u.fig2numpy(fig)]

    def plot_states(self, dateset_name, plot_eulers: bool = False, **plots) -> List[np.ndarray]:
        plots = {k: p for k, p in plots.items() if p is not None}
        time_beg = np.min([p.times[0] for p in plots.values()])
        time_end = np.max([p.times[-1] for p in plots.values()])
        x_beg, x_end = 0, (time_end - time_beg) + 5  # let's pad 5 seconds for legend
        figsize, dpi = (33, 12), 150

        colors, ltype = dict(gt='r', imu_only='g', fallback='b'), dict(imu_only='--')
        renames = dict(imu_only='IMU integr.', fallback='Fallback')

        selectors = dict(
            poses=lambda x: x.poses[:, :3, 3], velocity=lambda x: x.vel[:, :3],
            gravity=lambda x: x.gravity, gyro_bias=lambda x: x.biases[:, :3], acc_bias=lambda x: x.biases[:, 3:],
            rotvec=lambda x: Rotation.from_matrix(x.poses[:, :3, :3]).as_rotvec(),
            euler=lambda x: np.rad2deg(u.unwrap_eulers_nan_delimiter(Rotation.from_matrix(x.poses[:, :3, :3]).as_euler('zxy'))),
        )

        images, beg, end = [], int(not plot_eulers), 3
        for j, text_name in enumerate(['euler', 'velocity', 'poses', 'gyro_bias', 'acc_bias', 'gravity'][beg:end]):
            fig = plt.figure(num=text_name, figsize=figsize, dpi=dpi)
            plt.suptitle(f'{text_name} {dateset_name if not j else ""}', fontsize=22)

            for i, coo in enumerate('xyz'):
                ax = plt.subplot(3, 1, i + 1)
                ax.yaxis.tick_right()
                for n, p in plots.items():
                    t, xyz = (p.times - time_beg), selectors[text_name](p)
                    plt.plot(t, xyz[:, i], label=f'{text_name}_{coo} {renames.get(n, n)}', color=colors[n], linestyle=ltype.get(n, '-'), linewidth=1)

                plt.xlim(x_beg, x_end)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
                plt.legend(loc='upper right')
                plt.grid(True)
            plt.xlabel('t(sec) from the beginning of the first fallback', fontsize=16)
            plt.tight_layout()
            images.append(u.fig2numpy(fig))

        return images

    def run_segment(self, data: ArcturusDataset, beg, end, data_window_time, enable_imu_only: bool = True):
        # get imu samples we need for tracking, need window_time in past to have enought data for CNN
        i0 = u.bisect_right(data.imu.times, beg - data_window_time) - 1
        i1 = u.bisect_left(data.imu.times, end)
        assert i0 >= 0, 'not enough IMU history to run current segment with requested window size'
        data_imu: ImuArray = data.imu[i0:i1]

        self.noise_model.add_imu_noise(data_imu.data[:, 1:])

        # split into history and tracking part, history we need to run CNN at the beginning
        ib = u.bisect_left(data_imu.times, beg)
        imu_history, imu_segment = data_imu[:ib].data, data_imu[ib:].data

        # interpolate start state
        start_state = data.gt.states_at(beg)
        update_step = 1.0 / config.update_rate
        self.noise_model.add_noise_to_init_state(start_state)

        # init fallback with start state, and some history
        tracker = ImuFallback(start_state.data, imu_history, update_step)
        tracker.backend = self.backend

        # perform fallback tracking by feeding imu samples sequentally
        for imu_sample in imu_segment:
            tracker.on_new_imu(imu_sample)

        fallback_poses = StatesArray(np.array(tracker.states_at_update_times))

        gt = data.gt.states_at(fallback_poses.times)
        zxy = Rotation.from_matrix(gt.poses[:, :3, :3]).as_euler('zxy')
        yaw = Rotation.from_euler('y', zxy[:, 2]).as_matrix()

        cnn_gt = np.einsum('nji,nj->ni', yaw, gt.vel[:, :3])  # yaw transposed
        cnn_preds = CnnPreds(np.c_[np.array(tracker.cnn_predictions), cnn_gt[1:]])

        imu_only = None
        if enable_imu_only:
            # extra run with skipping CNN updates to get IMU only trajectory
            tracker = ImuFallback(start_state.data, imu_history, update_step, skip_updates=True)
            tracker.backend = None  # we don't need backend in skip_updates mode
            for imu_sample in imu_segment:
                tracker.on_new_imu(imu_sample)
            imu_only = StatesArray(np.array(tracker.states_at_update_times))

        return imu_only, fallback_poses, gt, cnn_preds

    def save_plots(self, data_folder, data_name, images) -> str:
        plots_file = Path(data_folder) / f'_results/{data_name}.png'
        plots_file.parent.mkdir(parents=True, exist_ok=True)

        import cv2
        zeros = np.array([0, 255, 0], dtype=np.uint8) * np.ones_like(images[0][:5], dtype=np.uint8)
        image = cv2.vconcat([e for i in images for e in [i, zeros]][:-1])
        cv2.imwrite(str(plots_file), image[..., ::-1])  # opencv uses BGR format
        return str(plots_file)

    def run(self):
        metrics = dict()
        for hdf5_file in tqdm(self.hdf5_files, unit='files', desc='Processing datasets', file=sys.stdout):
            data = ArcturusDataset.load(hdf5_file)
            durations = ', '.join([f'{d:.01f}' for d in (data.segments[:, 1] - data.segments[:, 0])])
            tqdm.write(f'Dataset {termcolor.colored(data.name, "cyan")}, segments durations [{durations}] sec')

            results: List[Tuple[StatesArray, StatesArray, StatesArray, CnnPreds]] = []
            for beg, end in data.segments:
                # skip segments that doesn't have enough IMU history
                if beg - self.minimal_required_imu_history_seconds < data.imu.times[0]:
                    continue
                data_window_time = self.window_time + 0.007  # add safety margin to time
                results.append(self.run_segment(data, beg, end, data_window_time, enable_imu_only=self.config.plot_imu_only))

            vel_mae_cnn = [np.abs(cnn.vel - cnn.gt).sum(axis=1).mean() for _, ekf, gt, cnn in results]
            vel_mae_fallback = [np.abs(ekf.vel[:, :3] - gt.vel[:, :3]).sum(axis=1).mean() for _, ekf, gt, cnn in results]
            pose_mae_fallback = [np.abs(ekf.poses[:, :3, 3] - gt.poses[:, :3, 3]).sum(axis=1).mean() for _, ekf, gt, cnn in results]
            metrics[data.name] = dict(vel_mae_cnn=vel_mae_cnn, vel_mae_fallback=vel_mae_fallback, pose_mae_fallback=pose_mae_fallback)

            imu_only, fallback, gt, cnn_preds = tuple(map(_concat_states_with_nans, zip(*results)))
            cnn_preds: CnnPreds

            if not config.dont_save_plots:
                images = []
                images.extend(self.plot_cnn_preds(data.name, cnn_preds=cnn_preds))
                images.extend(self.plot_states(data.name, plot_eulers=False, imu_only=imu_only, fallback=fallback, gt=gt))

                plot_file = self.save_plots(config.data_folder, data.name, images)
                metrics[data.name].update(**dict(plots_file=plot_file))
                u.open_plots(filename=plot_file)

        # compute accumulated averaged metrics
        all_vel_mae_cnn = np.mean([m for v in metrics.values() for m in v['vel_mae_cnn']])
        all_vel_mae_final_ekf = np.mean([m for v in metrics.values() for m in v['vel_mae_fallback']])
        all_pose_mae_final_ekf = np.mean([m for v in metrics.values() for m in v['pose_mae_fallback']])
        metrics.update(all_vel_mae_cnn=all_vel_mae_cnn, all_vel_mae_fallback=all_vel_mae_final_ekf, all_pose_mae_fallback=all_pose_mae_final_ekf)

        print(f'all_vel_mae_cnn {all_vel_mae_cnn * 100:0.02f}cm/s')
        print(f'all_vel_mae_final_ekf {all_vel_mae_final_ekf * 100:0.02f}cm/s')
        print(f'all_pose_mae_final_ekf {all_pose_mae_final_ekf * 100:0.02f}cm')

        import json
        with open(Path(config.data_folder) / '_results/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4, sort_keys=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Inertial Prediction: sample evaluation script')

    parser.add_argument('--data_folder', '-df', type=str, default=None, required=True, help='Data type to generate')
    parser.add_argument('--update_rate', type=int, default=20, help='EKF update rate or CNN call step')
    parser.add_argument('--model_path', type=str, default=None, help='Path to CNN model to use')

    parser.add_argument('--dont_save_plots', '-dsp', action='store_true', help='Show IMU only plot')
    parser.add_argument('--plot_imu_only', '-pio', action='store_true', help='Show IMU only plot')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable more logging')
    parser.add_argument('--seed', '-s', type=int, default=777, help='Random seed for noise generation')

    config: argparse.Namespace = parser.parse_args()

    u.set_random_seed(config.seed)
    np.set_printoptions(linewidth=600, suppress=True)

    config.plot_imu_only = True
    if config.model_path is None:
        config.model_path = config.data_folder + '/pretrained/model_v1.scripted.pt'

    evaluator = Evaluator(config)
    evaluator.run()
