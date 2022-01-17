import json
import torch.jit
import numpy as np

from typing import Tuple, Union, List, Any


def to_numpy(*tensors) -> Any:
    return [t.detach().cpu().numpy() for t in tensors] if len(tensors) > 1 else tensors[0].detach().cpu().numpy()


class TorchScriptBackend(object):
    target_imu_rate: int
    window_size: int

    def __init__(self, model_path: str) -> None:
        assert model_path is not None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = torch.jit.load(model_path, map_location=self.device)

        json_file = model_path.replace('.scripted.pt', '.json')
        with open(json_file, 'r') as f:
            model_config = json.load(f)

        self.window_size = model_config.get('window_size', 100)
        self.target_imu_rate = model_config.get('target_imu_rate', 100)

    def __call__(self, imu_window: np.ndarray) -> Tuple[Any, ...]:
        # torch script model provided by Arcturus takes (1, 15, window_size) blob, feel free to adjust for your model
        imu_window = np.c_[imu_window, np.zeros_like(imu_window), np.zeros_like(imu_window[:, :3])]  # (N, 15)

        imu_window = imu_window.T.reshape(1, 15, -1)  # (1, 15, window_size)
        input_tensor = torch.as_tensor(imu_window, dtype=torch.float32, device=self.device)
        return to_numpy(*[t.reshape(3) for t in self.model(input_tensor)])


class OpenCVDnnBackend(object):

    def __init__(self, model_path: str) -> None:
        # need to build model graph in python and load weights only
        pass

    def __call__(self, imu_window: np.ndarray):
        return 0, 0


class OnnxBackend(object):

    def __init__(self, model_path: str) -> None:
        pass

    def __call__(self, imu_window: np.ndarray):
        return 0, 0


class FrozenTensorFlowBackend(object):

    def __init__(self, model_path: str) -> None:
        pass

    def __call__(self, imu_window: np.ndarray):
        return 0, 0


class TorchBackend(object):

    def __init__(self, model_path: str) -> None:
        # for torch need to build model graph in python and load just weights
        pass

    def __call__(self, imu_window: np.ndarray):
        return 0, 0
