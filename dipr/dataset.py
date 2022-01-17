import os
import re
import bisect
import datetime
import numpy as np
import utils as u

from typing import Any


class StatesArray(object):
    # t | isometry 1x16 | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
    # 0   1               17      20      23       26       29      32
    data: np.ndarray

    def __init__(self, states: np.ndarray) -> None:
        self.data = u.ensure_unpacked_quaternion(states)

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def __getitem__(self, key):
        return StatesArray(self.data[key])

    @property
    def duration(self) -> float:
        return self.times[-1] - self.times[0]

    @property
    def rate(self) -> float:
        return len(self.times) / (self.times[-1] - self.times[0])

    @property
    def times(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def poses(self) -> np.ndarray:
        """ transform from device to world """
        return self.data[:, 1:17].reshape(-1, 4, 4)

    @property
    def vel(self) -> np.ndarray:
        """ linear velocity in world and angular velocity in world """
        return self.data[:, 17:23].reshape(-1, 6)

    @property
    def biases(self) -> np.ndarray:
        """ gyro bias and acc bias """
        return self.data[:, 23:29].reshape(-1, 6)

    @property
    def gravity(self) -> np.ndarray:
        """ gravity in world """
        return self.data[:, 29:32].reshape(-1, 3)

    def decimate_(self, step: int) -> None:
        self.data = self.data[::step].copy()

    def transform_device_(self, isometry: np.ndarray) -> None:
        new_poses = np.einsum('bij,jk->bik', self.poses, isometry)
        new_vel = u.transform_velocities_to_target(new_poses, self.poses, self.vel)
        self.poses[:], self.vel[:] = new_poses, new_vel

    def transform_world_(self, isometry: np.ndarray) -> None:
        self.poses[:] = np.einsum('ij,njk->nik', isometry, self.poses)
        self.vel[:] = np.einsum('ij,nj->ni', isometry[:3, :3], self.vel.reshape(-1, 3)).reshape(-1, 6)
        self.gravity[:] = np.einsum('ij,nj->ni', isometry[:3, :3], self.gravity)

    def states_at(self, times: np.ndarray):
        times = np.atleast_1d(np.array(times).squeeze())
        i = np.searchsorted(self.times, times).clip(1, len(self.data) - 1).astype(np.int32)
        t0, t1 = self.times[i - 1], self.times[i]
        alpha = np.array((times - t0) / (t1 - t0))

        import quaternion as q
        r0, r1 = self.poses[i - 1, :3, :3], self.poses[i, :3, :3]
        r0 = q.from_rotation_matrix(r0, nonorthogonal=False)
        r1 = q.from_rotation_matrix(r1, nonorthogonal=False)
        wxyz = q.as_float_array(q.slerp(r0, r1, 0, 1, alpha))

        d0, d1 = self.data[i - 1], self.data[i]
        states = StatesArray(d0 + alpha[:, None] * (d1 - d0))
        states.poses[:, :3, :3] = q.as_rotation_matrix(q.from_float_array(wxyz))

        invalid = np.logical_or(times < self.times[0], times > self.times[-1])
        states.data[invalid] = np.full_like(states.data[0], np.nan)
        return states

    def __add__(self, other):
        output = StatesArray(self.data)
        output.data = np.r_[output.data, other.data]
        return output


class ImuArray(object):
    # t | gyro xyz | acc xyz | END
    # 0   1          4         7
    data: np.ndarray

    def __init__(self, data: np.ndarray) -> None:
        self.data = data

    def __iter__(self):
        return iter(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, key):
        return ImuArray(self.data[key])

    @property
    def rate(self) -> int:
        return int(len(self.data) / (self.times[-1] - self.times[0]))

    @property
    def times(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def gyro(self) -> np.ndarray:
        return self.data[:, 1:4]

    @property
    def acc(self) -> np.ndarray:
        return self.data[:, 4:7]

    def imu_at(self, times: np.ndarray) -> Any:
        return ImuArray(data=u.sorted_interp1d(self.times, self.data)(times))


class CnnPreds(object):

    # t | vel xyz | uncertainty xyz | optional_gt xyz | END
    # 0   1          4                7                 10
    data: np.ndarray

    def __init__(self, data: np.ndarray) -> None:
        assert data.shape[-1] in [7, 10]
        self.data = data

    def __getitem__(self, key):
        return ImuArray(self.data[key])

    @property
    def times(self) -> np.ndarray:
        return self.data[:, 0]

    @property
    def vel(self) -> np.ndarray:
        return self.data[:, 1:4]

    @property
    def uncertainty(self) -> np.ndarray:
        return self.data[:, 4:7]

    @property
    def sigma(self) -> np.ndarray:
        return np.exp(self.uncertainty)

    @property
    def gt(self) -> np.ndarray:
        return self.data[:, 7:10]


class ArcturusDataset(object):
    gt: StatesArray
    tracked: StatesArray
    imu: ImuArray

    def __init__(self):
        self.gt = self.imu = None
        self.hdf5_file = self.name = None
        self.is_synthetic = None
        self.tracked = self.segments = None

    @staticmethod
    def load(hdf5_file: str):
        import h5py

        with h5py.File(hdf5_file, 'r') as f:
            data = ArcturusDataset()
            data.is_synthetic = f['is_synthetic'][()]
            data.imu = ImuArray(f['imu'][:])

            if 'gt' in f.keys():
                gt_format = f['gt_format'][()].decode('utf-8')
                if gt_format == 't_wxyz_translation_vel_angular':
                    gt = f['gt'][:]
                    biases, gravity = np.zeros_like(gt[:, :6]), [0, 9.81, 0]
                    data.gt = StatesArray(u.unpack_quaternion(np.c_[gt, biases, np.tile(gravity, [len(gt), 1])]))
                elif gt_format == 't_wxyz_translation_vel_angular_gb_ab_gravity':
                    data.gt = StatesArray(u.unpack_quaternion(f['gt'][:]))
                else:
                    assert False, f'Unsupported gt format {gt_format}'

            if 'tracked' in f.keys():
                tracked_format = f['tracked_format'][()].decode('utf-8')
                assert tracked_format == 't_wxyz_translation_vel_angular_gb_ab_gravity', 'Only supported format'
                data.tracked = StatesArray(u.unpack_quaternion(f['tracked'][:]))

            if 'segments' in f.keys():
                data.segments = f['segments'][:]

            data.hdf5_file = hdf5_file
            data.name = os.path.basename(hdf5_file)[:-len('.hdf5')]

        """ We agreed on GOLD standard to use inverse of body acc in m/s2 """
        if True:  # magnitude and accel do not necessarily satisfy the criteria
            magnitude = np.linalg.norm(data.imu.data[:, 4:], axis=1).mean()
            assert magnitude > 4, f'Required inverse of body in m/s2 units, magnitude {magnitude:.02f}'

        return data

    @property
    def duration(self) -> float:
        times = tuple(t.times for t in (self.gt, self.imu) if t is not None)
        if all(len(t) > 1 for t in times):
            min_time, max_time = max(t[0] for t in times), min(t[-1] for t in times)
            return float(max_time - min_time)
        return 0

    @property
    def datetime(self) -> datetime.datetime:
        pattern = r'OpenVR_(\d\d\d\d-\d\d-\d\d_\d\d-\d\d-\d\d)'
        m = re.match(pattern, re.sub(r'[-_]synthetic', '', self.name))
        return datetime.datetime.strptime(m.group(1), '%Y-%m-%d_%H-%M-%S')

    def clone(self):
        from copy import deepcopy
        return deepcopy(self)

    def align_tracker_to_gt_at_(self, time_hint):
        timestamp = max(time_hint, self.gt.times[0], self.tracked.times[0])
        index = bisect.bisect_left(self.tracked.times, timestamp)

        tracked0 = self.tracked.poses[index]
        gt0 = self.gt.states_at(self.tracked.times[index]).poses[0]

        aligner = gt0 @ u.invert_isometry(tracked0)
        if np.sum(aligner - np.eye(4)) < 1e-12:
            return self

        self.tracked.transform_world_(aligner)
        return self

