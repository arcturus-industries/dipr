import numpy as np
import quaternion as q
from dataset import StatesArray


def _rand11_like(tensor_like: np.array) -> np.array:
    return 2 * np.random.rand(*tensor_like.shape) - 1


def _rand_like(tensor_like: np.array) -> np.array:
    return np.random.rand(*tensor_like.shape)


def _randn_like(tensor_like: np.array) -> np.array:
    return np.random.randn(*tensor_like.shape)


class ImuNoiseCalibration(object):
    deg2rad = np.pi / 180

    def __init__(self) -> None:
        self.sampling_rate = 1000.  # Hz
        self.gyro_noise_density_std = 0.004  # deg/s * 1/sqrt(Hz)  ~= 0.126 deg/s at 1000Hz
        self.acc_noise_density_std = 9.81e-04  # m/s2 * 1/sqrt(Hz) ~= 0.031 m/s^2 at 1000Hz

        # From the AD curve and at first glance seems that
        # a value of gyroChangeStd in (deg/s2) * 1/sqrt(Hz) about 4e-04 at tau=1s
        # and for accelChangeStd in (m/s3) * 1/sqrt(Hz)  about 3e-05 at tau=1s are reasonable

        self.gyro_bias_change_std = 4e-04  # (deg/s) * sqrt(Hz)
        self.acc_bias_change_std = 3e-05  # (m/s2) * sqrt(Hz)

        # C++ parameter used in SW & UKF
        self.gyro_bias_std_per_minute = 0.0018  # (deg/s) / min
        self.accel_bias_std_per_minute = 0.0145  # (m/s^2) / min

    @property
    def gyro_discrete_noise_std(self):
        """ deg/s * 1/sqrt(Hz) * sqrt(Hz) * rad/deg  = rad/s """
        return self.gyro_noise_density_std * np.sqrt(self.sampling_rate) * self.deg2rad

    @property
    def acc_discrete_noise_std(self):
        """ m/s2 * 1/sqrt(Hz) * sqrt(Hz)  = m/s2 """
        return self.acc_noise_density_std * np.sqrt(self.sampling_rate)

    @property
    def gyro_discrete_bias_rw_std(self):  # random walk
        """ (deg/s2) * 1/sqrt(Hz) / sqrt(hz) * rad/deg = rad/s """
        return self.gyro_bias_change_std * self.deg2rad

    @property
    def acc_discrete_bias_rw_std(self):  # random walk
        """ (m/s3) * 1/sqrt(Hz) / sqrt(hz) * rad/deg = m/s2 """
        return self.acc_bias_change_std


class NoiseModel(ImuNoiseCalibration):
    scale_factor_magnitude = 0.005
    cross_axis_magnitude = np.pi * 0.0001

    init_tilt_noise = False
    init_tilt_angle = np.deg2rad(1)

    init_position_noise_magnitude = 0.
    init_velocity_noise_magnitude = 0.01
    init_gravity_noise_magnitude = 0.01

    init_gyro_bias_noise = np.deg2rad(0.2)  # rad/sec
    init_acc_bias_noise = 0.01  # m/s2

    def add_imu_noise(self, imu: np.ndarray):
        # scale noise
        imu *= 1 + _rand11_like(imu[:1]) * self.scale_factor_magnitude

        # cross-axis of gyro and accelerometer errors. Error is different for gyro and accelerometer
        cross_axis = self.cross_axis_magnitude * _rand11_like(imu[:1]).reshape(2, 3)
        rots = q.as_rotation_matrix(q.from_rotation_vector(cross_axis)).reshape(2, 3, 3)

        # transpose matrix vector product
        imu[:] = np.einsum('tij, ntj->nti', rots, imu.reshape(-1, 2, 3)).reshape(-1, 6)

        # gaussian noise
        imu[:, :3] += _randn_like(imu[:, :3]) * self.gyro_discrete_noise_std
        imu[:, 3:] += _randn_like(imu[:, 3:]) * self.acc_discrete_noise_std

    def add_noise_to_init_state(self, state: StatesArray):
        if self.init_tilt_noise:
            # attitude with error
            attitudes = state.poses[:, :3, :3]
            angle = 2 * np.pi * _rand_like(attitudes[:, 0, 0])
            hvec = np.c_[np.cos(angle), np.zeros_like(attitudes[:, 0, 0]), np.sin(angle)]
            half_magnitude = _rand_like(attitudes[:, 0, 0]) * self.init_tilt_angle / 2

            wxyz = q.from_float_array(np.c_[np.cos(half_magnitude), np.sin(half_magnitude) * hvec])
            state.poses[:, :3, :3] = np.einsum('nij,njk->nik', q.as_rotation_matrix(wxyz), attitudes)
        else:
            # ideal initial attitude but gravity noise
            state.gravity[:] += _rand11_like(state.gravity) * self.init_gravity_noise_magnitude

        state.poses[:, :3, 3] += 0
        state.vel[:, :3] += _randn_like(state.vel[:, :3]) * self.init_velocity_noise_magnitude

        state.biases[:, :3] += _randn_like(state.biases[:, :3]) * self.init_gyro_bias_noise
        state.biases[:, 3:] += _randn_like(state.biases[:, 3:]) * self.init_acc_bias_noise
