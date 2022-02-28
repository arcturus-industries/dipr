import pathlib
import sys
import math
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
import quaternion as q

from dataset import ArcturusDataset
import utils as u


class DatasetBuilder(object):
    train_data_folder = 'train_synthetic_from_real_v1'

    def __init__(self, config: argparse.Namespace):
        self.config = config

        self.hdf5_files = sorted(Path(self.config.data_folder, self.train_data_folder).glob('*.hdf5'))
        assert len(self.hdf5_files) > 0, 'train data folder is empty, please check paths in command line parameters'
        self.samples = []

    def prepare(self):
        train_data_cache_folder = Path(self.config.data_folder) / '_cache_train_data'
        train_data_cache_folder.mkdir(parents=True, exist_ok=True)

        durations, array_list = [], []
        for hdf5 in tqdm(self.hdf5_files, unit='files', desc='Loading training data', file=sys.stdout):
            name = hdf5.name[:-len('.hdf5')]
            __cache = train_data_cache_folder / f'{name}_{config.target_imu_rate}hz.pickle'
            cache_array, duration = u.pickle_try_load(str(__cache), (None, None))
            if cache_array is None:
                data = ArcturusDataset.load(hdf5)
                np.testing.assert_allclose(data.gt.gravity.mean(axis=0), np.array([0, 9.81, 0]), atol=1e-13)
                tqdm.write(f'{data.duration/60:.01f}min {data.name}')

                # resample IMU to requested rate
                t0, tn = data.imu.times[0], data.imu.times[-1]
                n = int(math.ceil((tn - t0) * config.target_imu_rate))
                times = np.array([(t0 + i / config.target_imu_rate) for i in range(n)])
                data.imu.data = u.sorted_interp1d(data.imu.times, data.imu.data)(times)

                # create cached arrays
                cache_array = self._create_cache_arrays(data)
                u.pickle_save(str(__cache), (cache_array, data.duration))
                tqdm.write(f'Cached data saved to {__cache}', file=sys.stdout)
                duration = data.duration

            durations.append(duration)
            array_list.append(cache_array)

            import gc
            gc.collect()

        self.samples = []
        for arr in array_list:
            self.samples.extend(self._generate_samples(arr))

        print(f'Durations: {np.array(durations)}')
        print(f'Total, h: {np.array(durations).sum() / 60 / 60:.02f}')
        return self.samples

    def _create_cache_arrays(self, data: ArcturusDataset) -> np.ndarray:
        if len(data.imu.times) < self.config.window_size:
            return np.array([]).reshape(0, 13)

        gyro = data.imu.gyro
        acc = data.imu.acc  # inverse of body acc in m/s2
        gt_states_at_imu = data.gt.states_at(data.imu.times)

        # convert IMU to global
        glob_gt_poses = gt_states_at_imu.poses
        glob_gt_vels = gt_states_at_imu.vel[:, :3]
        glob_gyro = np.einsum('bij,bj->bi', glob_gt_poses[:, :3, :3], gyro)
        glob_acc = np.einsum('bij,bj->bi', glob_gt_poses[:, :3, :3], acc)

        # rotations from IMU to global for each IMU sample
        wxyz = q.as_float_array(q.from_rotation_matrix(glob_gt_poses[:, :3, :3], nonorthogonal=False))
        return np.c_[glob_gyro, glob_acc, glob_gt_vels, wxyz]

    def _generate_samples(self, cache_array: np.ndarray):
        if len(cache_array) == 0:
            return tuple([]), None

        # gx gy gz | ax ay az | gt_x gt_y gt_z | wxyz
        glob_imu, glob_gt_vels, wxyz = np.split(cache_array, [6, 9], axis=1)

        # not used in this sample, we used GT rotations for some augmentations
        rotations = q.as_rotation_matrix(q.from_float_array(wxyz))

        # note that the arrays below are shared between samples (windows of IMU data that overlap a lot), thus memory efficient
        shared = glob_imu, glob_gt_vels, rotations

        samples = []
        for i in range(0, len(glob_gt_vels) - self.config.window_size, self.config.window_step):
            beg, end = i, i + self.config.window_size
            gt = glob_gt_vels[end]
            samples.append((glob_imu[beg:end], gt))

        return samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deep Inertial Prediction: sample dataset builder script')

    parser.add_argument('--data_folder', '-df', type=str, default=None, required=True, help='Data type to generate')
    parser.add_argument('--target_imu_rate', type=int, default=100, help='IMU data rate CNN accepts')
    parser.add_argument('--window_size', type=int, default=100, help='IMU window size in number of IMU measurements that CNN accepts')
    parser.add_argument('--window_step', type=int, default=10, help='Sliding window step in number of IMU measurements')

    config: argparse.Namespace = parser.parse_args()

    np.set_printoptions(linewidth=600, suppress=True)

    builder = DatasetBuilder(config)
    samples = builder.prepare()

    import torch.utils.data
    loader = torch.utils.data.DataLoader(samples, batch_size=128, shuffle=True)

    print('Start training')
    iter, max_iters = 0, 5
    for imu, gt in loader:
        tqdm.write(f'{iter} {imu.shape} {gt.shape}')
        iter += 1
        if iter >= max_iters:
            break
