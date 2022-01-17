import utils
import numpy as np
import quaternion as q


from typing import Optional, Tuple, Callable, Union, Any

from numba.experimental import jitclass
from numba import jit, njit, float64, int32, objmode


class SO3(object):

    @staticmethod
    def as_wyxz(attitude):
        return q.as_float_array(q.from_rotation_matrix(attitude))

    @staticmethod
    def hat(w: np.ndarray) -> np.ndarray:
        w = w.reshape(3)
        return np.array([0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0]).reshape(3, 3)

    @staticmethod
    def exp(phi: np.ndarray) -> np.ndarray:
        phi = phi.reshape(3)
        angle = np.linalg.norm(phi)
        # Near phi == 0, use first order Taylor expansion
        if np.isclose(angle, 0.):
            return np.identity(3) + SO3.hat(phi)

        axis = phi / angle
        s = np.sin(angle)
        c = np.cos(angle)
        return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * SO3.hat(axis)

    @staticmethod
    def extract_yaw(R: np.ndarray, euler_mode: str = 'YXZ') -> Tuple[np.ndarray, Any, str]:
        """
                R = Ry @ Rz @ Rx
                or
                R = Ry @ Rx @ Rz

        Decompose matrix so that last intrinsic rotation is about Y (vertical axis in world frame) and return Ry
        For more details about intrinsic/extrinsics rotations: https://en.wikipedia.org/wiki/Euler_angles
        """
        assert euler_mode in ['YZX', 'YXZ']

        ex = ey = ez = np.nan
        if euler_mode == 'YZX':
            # assert np.abs(R[1, 0]) < np.sin(np.deg2rad(87)), f'gimbal lock case | {np.abs(R[1, 0])}'
            ez = np.arcsin(R[1, 0])
            ey = -np.arctan2(R[2, 0] / np.cos(ez), R[0, 0] / np.cos(ez))
        elif euler_mode == 'YXZ':
            # assert np.abs(R[1, 2]) < np.sin(np.deg2rad(87)), f'gimbal lock case | {np.abs(R[1, 2])}'
            ex = -np.arcsin(R[1, 2])
            ey = np.arctan2(R[0, 2] / np.cos(ex), R[2, 2] / np.cos(ex))
        else:
            assert False, 'unsupported euler decomposition mode'

        # Constructing Yaw rotation matrix, note that signs of sines swapped
        # (in contrast to pitch and roll matrices), that is correct for Y
        # because of right-hand rule of rotation definition
        Ry = np.array([np.cos(ey), 0, np.sin(ey),
                       0, 1, 0,
                       -np.sin(ey), 0, np.cos(ey)]).reshape(3, 3)

        return Ry, (ex, ey, ez), euler_mode


@jit(nopython=True)
def hat(w: np.ndarray) -> np.ndarray:
    w = w.reshape(3)
    return np.array([0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0]).reshape(3, 3)


@jit(nopython=True)
def exp(phi: np.ndarray) -> np.ndarray:
    phi = phi.reshape(3)
    angle = np.linalg.norm(phi)
    # Near phi == 0, use first order Taylor expansion
    if np.abs(angle) < 1e-10:
        return np.identity(3) + hat(phi)

    axis = phi / angle
    s = np.sin(angle)
    c = np.cos(angle)
    return c * np.identity(3) + (1 - c) * np.outer(axis, axis) + s * hat(axis)


class Ekf(object):
    StateType = Union[Any]  # union for mypy workaround
    ErrorStateType = Union[np.ndarray]
    StateCovarianceType = Union[np.ndarray]
    TransitionMatrixType = Union[np.ndarray]
    NoiseMatrixType = Union[np.ndarray]
    MeasurementType = Union[np.ndarray]
    MeasurementJacobianType = Union[np.ndarray]

    PredictionMapping = Callable[[StateType], Tuple[StateType, TransitionMatrixType, NoiseMatrixType]]
    MeasurementMapping = Callable[[StateType], Tuple[MeasurementType, MeasurementJacobianType]]
    NormalizationMapping = Callable[[StateType, StateCovarianceType, Optional[ErrorStateType]], None]

    verbose: bool = False
    chi2_threshold: float = 11.345

    def __init__(self, nominal_state: StateType, init_covariance: np.ndarray) -> None:
        """ error state mean is always zero for error-state EKF, thus not used """
        self.state = nominal_state
        self.error_cov = init_covariance

    def predict(self, prediction_mapping: PredictionMapping, process_noise: np.ndarray,
                normalization_mapping: NormalizationMapping = lambda state, cov, error: None) -> None:

        self.state, A, B = prediction_mapping(self.state)
        _Q = process_noise if B is None else (B @ process_noise @ B.T)
        self.error_cov = A @ self.error_cov @ A.T + _Q
        normalization_mapping(self.state, None, self.error_cov)

    def update(self, measurement: Any, measurement_covariance: np.ndarray,
               measurement_mapping: MeasurementMapping, normalization_mapping: NormalizationMapping, mode: str = 'joseph') -> bool:
        assert mode in ['simple', 'symmetric', 'joseph']
        predicted_measurement, H = measurement_mapping(self.state)  # (3, ), (3, state.size)

        measurement_error = measurement.squeeze() - predicted_measurement
        innovation_covariance = H @ self.error_cov @ H.T + measurement_covariance

        gain = np.linalg.solve(innovation_covariance, H @ self.error_cov.T).T  # (state.size, 3)

        # now we have non-zero error_mean which to be injected into nominal state next
        error_mean = gain @ measurement_error[:, None]

        if mode == 'simple':
            self.error_cov = self.error_cov - gain @ H @ self.error_cov
        elif mode == 'symmetric':
            self.error_cov = self.error_cov - gain @ innovation_covariance @ gain.T
        elif mode == 'joseph':
            mult = np.eye(error_mean.size) - gain @ H
            self.error_cov = mult @ self.error_cov @ mult.T + gain @ measurement_covariance @ gain.T

        normalization_mapping(self.state, error_mean, self.error_cov)
        return True


class FusionState(object):
    StateSize = 18
    NoiseSize = 12

    time: np.float64  # in microseconds
    attitude: np.ndarray  # world from_omu_rotation
    translation: np.ndarray  # world_from_imu_translation
    vel: np.ndarray  # linear_velocity_in_world
    angular: np.ndarray  # angular_velocity_in_world
    gb: np.ndarray  # gyro bias, rad/sec
    ab: np.ndarray  # acc bias, m/s2
    gravity: np.ndarray  # gravity estimate, m/s2

    def __init__(self, init_state: np.ndarray) -> None:
        assert init_state.size == 32
        # t | isometry 1x16 | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
        # 0   1               17      20      23       26       29      32
        _splits = np.split(init_state.squeeze(), [1, 17, 20, 23, 26, 29])
        _, pose, self.vel, self.angular, self.gb, self.ab, self.gravity = _splits
        self.time, pose = init_state.squeeze()[0], pose.reshape(4, 4)
        self.attitude, self.translation = pose[:3, :3].copy(), pose[:3, 3].copy()

    @property
    def pose(self) -> np.ndarray:
        pose = np.eye(4)
        pose[:3, :3] = self.attitude
        pose[:3, 3] = self.translation
        return pose

    @property
    def wxyz(self) -> np.ndarray:
        return SO3.as_wyxz(self.attitude)

    @property
    def t_wyxz_biases(self) -> np.ndarray:
        return np.r_[[self.time], self.wxyz, self.gb, self.ab]

    def numpy(self) -> np.ndarray:
        return np.r_[[self.time], self.pose.reshape(-1), self.vel, self.angular, self.gb, self.ab, self.gravity]

    def __str__(self) -> str:
        return f'{self.time:.03f} -> q{self.wxyz} t{self.translation} v{self.vel} gb{self.gb} ab{self.ab}'

    @staticmethod
    def create_covariance_matrix(theta_std, t_std, v_std, gb_std, ab_std, g_std) -> np.ndarray:
        covariance = np.zeros(shape=(FusionState.StateSize, FusionState.StateSize), dtype=np.float64)
        diag_std = np.concatenate([theta_std, t_std, v_std, gb_std, ab_std, g_std])
        return np.diag(np.square(diag_std))


@jit(nopython=True)
def prediction_fn_numba(imu0: np.ndarray, imu1: np.ndarray, curr_att, curr_pos, curr_vel, curr_gb, curr_ab,
                        curr_gravity) -> Tuple[FusionState, np.ndarray, Optional[np.ndarray]]:
    # For tabletop lying device without movement, inverseOfBodyAcceleration after conversion to world is gravity vector
    # i.e. inverseOfBodyAcceleration = +9.81 on X before rotation to world, and +9.81 on Y after rotation to world
    _, gyro0, inverseOfBodyAcc0 = np.split(imu0, [1, 4])
    _, gyro1, inverseOfBodyAcc1 = np.split(imu1, [1, 4])
    time0, time1 = imu0[0], imu1[0]

    # acc = bodyAcc = -inverseOfBodyAcc
    acc0 = -inverseOfBodyAcc0
    acc1 = -inverseOfBodyAcc1

    # assert np.abs(time0 - state.time) < 1e-9, f'{time}, {state.time}'
    dt = time1 - time0
    rate1 = 0.5 * (gyro0 + gyro1)
    # s1 = hat(gyro1) @ gyro0
    # s2 = hat(gyro1) @ s1
    # rate1 += -1/12 * s1 * dt + 1/240 * s2 * dt * dt

    glob0 = (acc0 - curr_ab)
    glob1 = exp((rate1 - curr_gb) * dt) @ (acc1 - curr_ab)
    next_att = curr_att @ exp((rate1 - curr_gb) * dt)

    # Comments:
    # 1. the glob0/glob1 weights come we assume a(t) is not constant, but linearly interpolated
    # between IMU measurements, after double integration of a(t) you will estimate coefficients 2/3 and 1/3
    #
    # 2. glob0 and glob1 accelerations are summed with weights in current coordinate frame,
    # to need to right multiply by attitude to get acceleration in world

    acc_v = (glob0 + glob1) / 2
    acc_t = (2 * glob0 + glob1) / 3

    next_pos = curr_pos + curr_vel * dt + 0.5 * (curr_att @ acc_t + curr_gravity) * dt * dt
    next_vel = curr_vel + (curr_att @ acc_v + curr_gravity) * dt

    state_time = time1
    state_attitude = next_att
    state_translation = next_pos
    state_vel = next_vel
    state_gb = curr_gb
    state_ab = curr_ab
    state_gravity = curr_gravity

    zero = np.zeros(shape=(3, 3))
    eye = np.identity(3)
    Rk = curr_att

    dThdGb = -Rk * dt
    dPdTh = -hat(curr_att @ acc_t) * 0.5 * dt * dt
    dPdV = eye * dt
    dPdAb = -Rk * 0.5 * dt * dt
    dVdTh = -hat(curr_att @ acc_v) * dt
    dVdAb = -Rk * dt
    dVdGb = Rk * 0.5 * hat(curr_att @ acc_v) * dt * dt
    dPdG = eye * dt * dt / 2
    dVdG = eye * dt

    transition_A = np.concatenate((
        #      R    pos    v      gb     ab   g
        np.concatenate((eye, zero, zero, dThdGb, zero, zero), axis=1),  # d_r
        np.concatenate((dPdTh, eye, dPdV, zero, dPdAb, dPdG), axis=1),  # d_p
        np.concatenate((dVdTh, zero, eye, dVdGb, dVdAb, dVdG), axis=1),  # d_v
        np.concatenate((zero, zero, zero, eye, zero, zero), axis=1),  # d_gb
        np.concatenate((zero, zero, zero, zero, eye, zero), axis=1),  # d_ab
        np.concatenate((zero, zero, zero, zero, zero, eye), axis=1),  # d_g
    ), axis=0)
    ret = state_time, state_attitude, state_translation, state_vel, state_gb, state_ab, state_gravity
    return ret, transition_A


class ImuFallback(object):
    StateSize = FusionState.StateSize

    init_theta_std = np.deg2rad([0., 0., 0.])  # rad
    init_t_std = np.ones(3) * 0.  # m

    verbose: bool = False
    process_noise_mult: float = 1.

    skip_updates: bool
    ekf: Ekf
    Q: np.ndarray
    update_step: float
    next_update_time: float

    backend: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]

    def __init__(self, init_state: np.ndarray, imu_history: np.ndarray, update_step, skip_updates: bool = False) -> None:
        from noise_utils import NoiseModel
        noise_calib = NoiseModel()
        init_v_std = np.ones(3) * noise_calib.init_velocity_noise_magnitude  # m/s
        init_gb_std = np.ones(3) * noise_calib.init_gyro_bias_noise  # rad/s
        init_ab_std = np.ones(3) * noise_calib.init_acc_bias_noise  # m/s2
        init_g_std = np.ones(3) * noise_calib.init_gravity_noise_magnitude  # m/s2

        inits = self.init_theta_std, self.init_t_std, init_v_std, init_gb_std, init_ab_std, init_g_std
        init_covariance = FusionState.create_covariance_matrix(*inits)
        self.ekf = Ekf(FusionState(init_state), init_covariance)

        self.update_step = update_step
        self.skip_updates = skip_updates
        self.next_update_time = self.ekf.state.time + update_step

        self.imu_history = list(imu_history)
        self.states_at_update_times = [init_state.squeeze()]
        self.cnn_predictions = []

        self.backend: Any = None

        gyro_noise_var = np.square(np.ones(3) * noise_calib.gyro_discrete_noise_std)  # rad2 / s2
        acc_noise_var = np.square(np.ones(3) * noise_calib.acc_discrete_noise_std)  # m2 / s4
        gyro_bias_rw_var = np.square(np.ones(3) * noise_calib.gyro_discrete_bias_rw_std)  # (rad/s)2 / s
        accel_bias_rw_var = np.square(np.ones(3) * noise_calib.acc_discrete_bias_rw_std)  # (m/s2)2 / s
        process_noise_vars = np.r_[gyro_noise_var, acc_noise_var, gyro_bias_rw_var, accel_bias_rw_var]

        eye, zero = np.eye(3), np.zeros(shape=(3, 3))
        noise_perturbation_matrix_B = np.r_[
            #      gn   an   gbrw  abrw
            np.c_[eye, zero, zero, zero],  # dTheta
            np.c_[zero, zero, zero, zero],  # dp
            np.c_[zero, eye, zero, zero],  # dv
            np.c_[zero, zero, eye, zero],  # dgb
            np.c_[zero, zero, zero, eye],  # dab
            np.c_[zero, zero, zero, zero],  # dg
        ]
        self.Q = np.diag(noise_perturbation_matrix_B @ np.diag(process_noise_vars) @ noise_perturbation_matrix_B.T)

    def on_new_imu(self, imu1: np.ndarray) -> Optional[np.ndarray]:
        # time | gyro | acc
        if imu1[0] <= self.ekf.state.time:
            self.imu_history.append(imu1)
            return None

        imu0 = self.imu_history[-1]
        if imu0[0] < self.ekf.state.time:
            alpha = (self.ekf.state.time - imu0[0]) / (imu1[0] - imu0[0])
            imu0 = imu0 * (1 - alpha) + imu1 * alpha
            self.imu_history.append(imu0)

        if self.verbose:
            print(f'Propagate {self.ekf.state.time:.04f} ==> {imu1[0]:.04f}')

        propagate = [imu0, imu1]
        if self.next_update_time < imu1[0]:
            alpha = (self.next_update_time - imu0[0]) / (imu1[0] - imu0[0])
            imuU = imu0 * (1 - alpha) + imu1 * alpha
            propagate = [imu0, imuU, imu1]
            self.imu_history.append(imuU)
            if self.verbose:
                print(f'Update at : {self.next_update_time}')

        for i, (imu_p, imu_n) in enumerate(zip(propagate, propagate[1:])):
            dt = imu_n[0] - self.ekf.state.time
            noise_integration_time_multipliers = np.array([
                dt * dt, dt * dt, dt * dt,
                0, 0, 0,
                dt * dt, dt * dt, dt * dt,
                dt, dt, dt,  # in C++ UKF here is square for biases
                dt, dt, dt,  # because of different noise bias model
                0, 0, 0,
            ])

            process_noise_covariance = np.diag(self.Q * noise_integration_time_multipliers) * self.process_noise_mult
            self.ekf.predict(lambda state: self.prediction_fn(imu_p, imu_n, state), process_noise_covariance, self.norm_fn)

            if len(propagate) == 3 and i == 0:  # two integration steps with measurement in the middle
                self.update_with_cnn_prediction()
                self.next_update_time = self.ekf.state.time + self.update_step
                self.states_at_update_times.append(self.ekf.state.numpy())

        self.imu_history.append(imu1)
        return self.ekf.state.pose

    def update_with_cnn_prediction(self):
        if self.skip_updates:
            return

        assert self.backend is not None, 'backend is not set'

        window_size = self.backend.window_size
        sample_step = 1.0 / self.backend.target_imu_rate

        start = self.ekf.state.time - window_size * sample_step
        cnn_samples_times = np.array([(start + i * sample_step) for i in range(window_size)])  # (N,)

        assert self.ekf.state.time == self.imu_history[-1][0], 'current state time should match last imu time in history'
        curr_q = q.from_rotation_matrix(self.ekf.state.attitude, nonorthogonal=False).reshape(1)
        
        index = utils.bisect_right(self.imu_history, cnn_samples_times[0], key=lambda x: x[0]) - 1
        hist_times = np.array([imu[0] for imu in self.imu_history[index:]])
        hist_gyro = np.array([imu[1:4] for imu in self.imu_history[index:]])
        assert index >= 0, 'should have enough IMU history, otherwise look at init code'

        # back-integrate
        delta_time = hist_times[:-1] - hist_times[1:]  # negative
        mid_gyro_unbiased = 0.5 * (hist_gyro[1:] + hist_gyro[:-1]) - self.ekf.state.gb
        delta_wxyz = q.from_rotation_vector(delta_time[:, None] * mid_gyro_unbiased)
        hist_attitudes = np.cumprod(np.r_[delta_wxyz, curr_q][::-1])[::-1]

        # interpolate
        i = np.searchsorted(hist_times.squeeze(), cnn_samples_times).clip(1, len(hist_times) - 1).astype(np.int32)
        t0, r0 = hist_times[i - 1], hist_attitudes[i - 1]
        t1, r1 = hist_times[i], hist_attitudes[i]
        alpha = (cnn_samples_times - t0) / (t1 - t0)
        attitudes = q.as_rotation_matrix(q.slerp(r0, r1, 0, 1, alpha))  # (N, 3, 3)

        # t | gyro | acc
        imu_times, imu_samples = np.split(np.array(self.imu_history), [1], axis=1)
        cnn_samples = utils.sorted_interp1d(imu_times, imu_samples)(cnn_samples_times)  # (N, 6)
        assert not np.isnan(cnn_samples.sum()), 'nans in imu samples'

        # calibrate samples
        cnn_samples[:, :3] -= self.ekf.state.gb  # gyro biases
        cnn_samples[:, 3:] += self.ekf.state.ab  # acc biases should be added to inverse_of_body_acc

        # rotate to global frame
        cnn_samples = np.einsum('bij,bkj->bki', attitudes, cnn_samples.reshape(-1, 2, 3)).reshape(-1, 6)

        # Decompose matrix R = Ry @ Rz @ Rx, and perform yaw subtraction
        Ry, _, _ = SO3.extract_yaw(self.ekf.state.attitude, euler_mode='YXZ')

        # rotate to yaw subtracted frame since yaw is unobservable
        cnn_samples = np.einsum('ij,bkj->bki', Ry.T, cnn_samples.reshape(-1, 2, 3)).reshape(-1, 6)

        accel_y = cnn_samples[:, 4].mean(axis=0)
        magnitude = np.linalg.norm(cnn_samples[:, 3:], axis=1).mean()
        assert magnitude > 4 and accel_y > 5, f'Required inverse of body in m/s2 units, magnitude {magnitude:.02f}, accel_y = {accel_y:.02f}'

        v, u = self.backend(cnn_samples)
        assert v.size == 3 and v.ndim == 1 and u.size == 3 and u.ndim == 1
        self.cnn_predictions.append(np.r_[[self.ekf.state.time], v, u])

        measurement_covariance = np.diag(np.exp(2 * u))
        self.ekf.update(v.reshape(3), measurement_covariance, self.measurement_fn, self.norm_fn)

    @staticmethod
    def prediction_fn(imu0: np.ndarray, imu1: np.ndarray, state: FusionState) -> Tuple[FusionState, np.ndarray, Optional[np.ndarray]]:
        if True:
            curr_att, curr_pos, curr_vel, curr_gb, curr_ab, curr_gravity = state.attitude, state.translation, state.vel, state.gb, state.ab, state.gravity

            # For tabletop lying device without movement, inverseOfBodyAcceleration after conversion to world is gravity vector
            # i.e. inverseOfBodyAcceleration = +9.81 on X before rotation to world, and +9.81 on Y after rotation to world
            time0, gyro0, inverseOfBodyAcc0 = np.split(imu0, [1, 4])
            time1, gyro1, inverseOfBodyAcc1 = np.split(imu1, [1, 4])

            # acc = bodyAcc = -inverseOfBodyAcc
            acc0 = -inverseOfBodyAcc0
            acc1 = -inverseOfBodyAcc1

            dt = time1 - time0
            rate1 = 0.5 * (gyro0 + gyro1)
            # s1 = hat(gyro1) @ gyro0
            # s2 = hat(gyro1) @ s1
            # rate1 += -1/12 * s1 * dt + 1/240 * s2 * dt * dt

            glob0 = (acc0 - curr_ab)
            glob1 = SO3.exp((rate1 - curr_gb) * dt) @ (acc1 - curr_ab)
            next_att = curr_att @ SO3.exp((rate1 - curr_gb) * dt)

            # Comments:
            # 1. the glob0/glob1 weights come we assume a(t) is not constant, but linearly interpolated
            # between IMU measurements, after double integration of a(t) you will estimate coefficients 2/3 and 1/3
            #
            # 2. glob0 and glob1 accelerations are summed with weights in current coordinate frame,
            # to need to right multiply by attitude to get acceleration in world

            acc_v = (glob0 + glob1) / 2
            acc_t = (2 * glob0 + glob1) / 3

            next_pos = curr_pos + curr_vel * dt + 0.5 * (curr_att @ acc_t + curr_gravity) * dt * dt
            next_vel = curr_vel + (curr_att @ acc_v + curr_gravity) * dt

            state.time = time1[0]
            state.attitude = next_att
            state.translation = next_pos
            state.vel = next_vel
            state.gb = curr_gb
            state.ab = curr_ab
            state.gravity = curr_gravity

            zero = np.zeros(shape=[3, 3])
            eye = np.identity(3)
            Rk = curr_att

            dThdGb = -Rk * dt
            dPdTh = -SO3.hat(curr_att @ acc_t) * 0.5 * dt * dt
            dPdV = eye * dt
            dPdAb = -Rk * 0.5 * dt * dt
            dVdTh = -SO3.hat(curr_att @ acc_v) * dt
            dVdAb = -Rk * dt
            dVdGb = Rk * 0.5 * SO3.hat(curr_att @ acc_v) * dt * dt
            dPdG = eye * dt * dt / 2
            dVdG = eye * dt

            transition_A = np.r_[
                #      R    pos    v      gb     ab   g
                np.c_[eye, zero, zero, dThdGb, zero, zero],  # d_r
                np.c_[dPdTh, eye, dPdV, zero, dPdAb, dPdG],  # d_p
                np.c_[dVdTh, zero, eye, dVdGb, dVdAb, dVdG],  # d_v
                np.c_[zero, zero, zero, eye, zero, zero],  # d_gb
                np.c_[zero, zero, zero, zero, eye, zero],  # d_ab
                np.c_[zero, zero, zero, zero, zero, eye],  # d_g
            ]
        else:
            (state.time, state.attitude, state.translation, state.vel, state.gb, state.ab, state.gravity), transition_A \
                = prediction_fn_numba(imu0, imu1, state.attitude, state.translation, state.vel, state.gb, state.ab, state.gravity)

        perturbation_B = None  # that means already incorporated into process_noise: Q = B @ Cov @ B.T
        return state, transition_A, perturbation_B

    @staticmethod
    def norm_fn(state: FusionState, error: np.ndarray, covariance: np.ndarray) -> None:
        # ensure symmetrical
        covariance[:] = (covariance + covariance.T) / 2

        if error is not None:
            delta_rvec, delta_pos, delta_vel, delta_gb, delta_ab, delta_g = np.split(error.squeeze(), 6)

            state.attitude = SO3.exp(delta_rvec) @ state.attitude
            state.translation = state.translation + delta_pos
            state.vel = state.vel + delta_vel
            state.gb = state.gb + delta_gb
            state.ab = state.ab + delta_ab
            state.gravity = state.gravity + delta_g

            # reset operation for covariance impacts only on rotations
            G = np.eye(error.size)
            G[:3, :3] = np.eye(3) + 0.5 * SO3.hat(delta_rvec)
            covariance[:] = G @ covariance @ G.T

            # G33 = np.eye(3) + 0.5 * SO3.hat(delta_rvec)
            # covariance[:3, :3] = G33 @ covariance[:3, :3] @ G33.T
            # covariance[3:, :3] = covariance[3:, :3] @ G33.T
            # covariance[:3, 3:] = G33 @ covariance[:3, 3:]

    @staticmethod
    def measurement_fn(state: FusionState) -> Tuple[np.ndarray, np.ndarray]:
        Size = state.StateSize
        Ri = state.attitude
        v = state.vel

        Ry, (ex, ey, ez), euler_mode = SO3.extract_yaw(Ri, euler_mode='YXZ')

        if euler_mode == 'YXZ':
            Hy = np.array([0, 0, 0,
                           np.sin(ey) * np.tan(ex), 1, np.cos(ey) * np.tan(ex),
                           0, 0, 0]).reshape(3, 3)
        else:
            assert False, 'unsupported euler decomposition mode'

        # Note that local gravity-aligned frame is the same for CNN input and UKF measurement
        h = Ry.T @ v
        H = np.zeros(shape=(3, Size))
        H[:, 0:3] = Ry.T @ SO3.hat(v) @ Hy
        H[:, 6:9] = Ry.T
        return h, H
