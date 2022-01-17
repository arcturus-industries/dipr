import bisect
from typing import Sequence, Any, Optional, Callable

import numpy as np
import quaternion as q


def bisect_left(a: Sequence[Any], x: Any, key: Optional[Callable[[Any], Any]] = None, lo: int = 0, hi: Optional[int] = None) -> int:
    if key is None:
        return bisect.bisect_left(a, x, lo, hi or len(a))

    if lo < 0:
        raise ValueError('lo must be non-negative')
    hi = hi or len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if key(a[mid]) < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


def bisect_right(a: Any, x: Any, key: Optional[Callable[[Any], Any]] = None, lo: int = 0, hi: Optional[int] = None) -> int:
    if key is None:
        return bisect.bisect_right(a, x, lo, hi or len(a))

    if lo < 0:
        raise ValueError('lo must be non-negative')

    hi = hi or len(a)
    while lo < hi:
        mid = (lo + hi) // 2
        if x < key(a[mid]):
            hi = mid
        else:
            lo = mid + 1
    return lo


def invert_isometry(transform):
    transform = transform.squeeze()
    R, t = transform[:3, :3], transform[:3, 3:4]
    result = np.eye(4, 4, dtype=transform.dtype)
    result[:3, 0:3] = R.T
    result[:3, 3:4] = R.T @ -t
    return result


def unwrap_eulers(eulers: np.ndarray, degrees: bool = False) -> np.ndarray:
    assert not np.isnan(eulers.sum()), 'this function doesn\'t support nans'

    threshold, ring = (300, 360) if degrees else (5.23, 2 * np.pi)

    unwrapped = np.zeros_like(eulers)
    unwrapped[0, :] = eulers[0, :]

    diff = eulers[1:, :] - eulers[0:-1, :]
    diff[diff > threshold] = diff[diff > threshold] - ring

    diff[diff < -threshold] = diff[diff < -threshold] + ring
    unwrapped[1:, :] = unwrapped[0, :] + np.cumsum(diff, axis=0)
    return unwrapped


def unwrap_eulers_nan_delimiter(eulers: np.ndarray, degrees: bool = False) -> np.ndarray:
    nans = np.isnan(eulers.sum(axis=1)).nonzero()[0].squeeze()
    splits = np.concatenate([np.array([0]), nans.reshape(-1), np.array([len(eulers) - 1])])

    unwrapped = eulers.copy()
    for beg, end in zip(splits, splits[1:]):
        if end - beg > 1:
            unwrapped[beg + 1:end] = unwrap_eulers(eulers[beg + 1:end], degrees)

    return unwrapped


def transform_linear_velocity(target_in_world: np.ndarray, source_in_world: np.ndarray,
                              linear_velocity_of_source_in_world: np.ndarray, angular_velocity_in_world: np.ndarray) -> np.ndarray:
    """
    Equation for computing velocity at point N on a rigid body given
    the velocity at point P and the angular velocity (w) of the rigid body.
    N, P, and w are in the fixed reference frame (the world).

          vel_at_N = vel_at_P + cross( w , N - P )

    w = rotation rate of the body == rotation rate at N == rotation rate at P by addition theorem for angular velocity
    """
    assert target_in_world.shape[-1] == 3 and source_in_world.shape[-1] == 3, 'should be a vector'
    assert angular_velocity_in_world.shape[-1] == 3 and linear_velocity_of_source_in_world.shape[-1] == 3, 'should be a vector'

    linear_velocity_of_target_in_world = linear_velocity_of_source_in_world + np.cross(angular_velocity_in_world, target_in_world - source_in_world)
    return linear_velocity_of_target_in_world


def transform_velocities_to_target(target_poses: np.ndarray, source_poses: np.ndarray, source_vel_in_world: np.ndarray) -> np.ndarray:
    sp, tp = source_poses[:, :3, 3], target_poses[:, :3, 3]
    vel, w = source_vel_in_world[:, :3], source_vel_in_world[:, 3:]
    target_vel = g.transform_linear_velocity_simple(tp, sp, vel, w)
    return np.concatenate([target_vel, w], axis=-1)


def sorted_interp1d(x: np.ndarray, y: np.ndarray, kind: str = 'linear', axis: int = 0, range_outside_value: Optional[Any] = None) -> Any:
    from scipy.interpolate import interp1d
    if range_outside_value is None:
        return interp1d(x.squeeze(), y, kind, axis, copy=False, assume_sorted=True)
    else:
        return interp1d(x.squeeze(), y, kind, axis, copy=False, assume_sorted=True, bounds_error=False, fill_value=range_outside_value)


def extract_wxyz_biases(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states.squeeze())
    assert states.shape[-1] in [23, 32]

    if states.shape[-1] == 32:
        # t | isometry 1x16 | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
        # 0   1               17      20      23       26       29      32
        times, poses, _, biases, _ = np.split(states, [1, 17, 23, 29], axis=1)
        wxyz = q.from_rotation_matrix(poses.reshape(-1, 4, 4)[:, :3, :3], nonorthogonal=False)
        return np.c_[times, q.as_float_array(wxyz), biases].squeeze()
    elif states.shape[-1] == 23:
        # t | q wxyz | t xyz | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
        # 0   1        5       8       11      14       17       20      23
        times, wxyz, _, biases, _ = np.split(states, [1, 5, 14, 20], axis=1)
        return np.c_[times, wxyz, biases].squeeze()
    else:
        assert False, f'Unsupported states format (shape={states.shape})'


def pack_quaternion(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states.squeeze())
    assert states.shape[-1] == 32

    # t | isometry 1x16 | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
    # 0   1               17      20      23       26       29      32
    times, poses, tail = np.split(states, [1, 17], axis=1)
    poses = poses.reshape(-1, 4, 4)
    wxyz = q.as_float_array(q.from_rotation_matrix(poses[:, :3, :3], nonorthogonal=False))
    a = np.c_[times, wxyz, poses[:, :3, 3], tail]
    return a


def unpack_quaternion(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states.squeeze())
    assert states.shape[-1] == 23

    # t | q wxyz | t xyz | v xyz | w xyz | gb xyz | ab xyz | g xyz | END
    # 0   1        5       8       11      14       17       20      23
    times, wxyz, xyz, tail = np.split(states, [1, 5, 8], axis=1)

    poses = np.empty(shape=(len(times), 4, 4), dtype=np.float64)
    poses[:, :3, :3] = q.as_rotation_matrix(q.from_float_array(wxyz))
    poses[:, :3, 3] = xyz
    poses[:, 3] = np.array([0, 0, 0, 1])
    return np.c_[times, poses.reshape(-1, 16), tail]


def ensure_packed_quaternion(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states.squeeze())
    if states.shape[-1] == 23:
        return states
    elif states.shape[-1] == 32:
        return pack_quaternion(states)
    else:
        assert False, f'Unsupported states format (shape={states.shape})'


def ensure_unpacked_quaternion(states: np.ndarray) -> np.ndarray:
    states = np.atleast_2d(states.squeeze())
    if states.shape[-1] == 23:
        return unpack_quaternion(states)
    elif states.shape[-1] == 32:
        return states
    else:
        assert False, f'Unsupported states format (shape={states.shape})'


def fig2numpy(figure: Any, close: bool = True) -> np.ndarray:
    import matplotlib.figure
    import matplotlib.axes
    import seaborn

    if isinstance(figure, matplotlib.figure.Figure):
        pass
    elif isinstance(figure, matplotlib.axes.SubplotBase):
        figure = figure.figure
    elif isinstance(figure, seaborn.axisgrid.FacetGrid):
        figure = figure.fig
    else:
        assert False, 'Which class you passed to fig2numpy?'

    figure.canvas.draw()
    s, (w, h) = figure.canvas.print_to_buffer()
    buf = np.fromstring(s, np.uint8).reshape((h, w, 4))
    if close:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return buf[..., :-1]  # RGBA -> RGB


def open_plots(filename: str):
    import os, sys, subprocess

    if sys.platform == 'win32':
        os.startfile(filename)

    elif sys.platform == 'darwin':
        subprocess.Popen(['open', filename])

    else:
        try:
            subprocess.Popen(['xdg-open', filename])
        except OSError:
            pass


def fig2numpy(figure: Any, close: bool = True) -> np.ndarray:
    import matplotlib.figure
    import matplotlib.axes
    import seaborn

    if isinstance(figure, matplotlib.figure.Figure):
        pass
    elif isinstance(figure, matplotlib.axes.SubplotBase):
        figure = figure.figure
    elif isinstance(figure, seaborn.axisgrid.FacetGrid):
        figure = figure.fig
    else:
        assert False, 'Which class you passed to fig2numpy?'

    figure.canvas.draw()
    s, (w, h) = figure.canvas.print_to_buffer()
    buf = np.fromstring(s, np.uint8).reshape((h, w, 4))
    if close:
        import matplotlib.pyplot as plt
        plt.close(figure)
    return buf[..., :-1]  # RGBA -> RGB


def set_random_seed(seed: int) -> None:
    import numpy as np
    np.random.seed(seed)
    seeds = np.random.random_integers(2 ** 24, size=10)

    import random
    random.seed(seeds[0])

    try:
        import torch
        torch.manual_seed(seeds[1])
    except ImportError:
        pass

    try:
        import tensorflow as tf
        tf.compat.v1.set_random_seed(seeds[2])  # must be before graph and session
    except ImportError:
        pass
