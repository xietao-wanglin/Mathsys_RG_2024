import math

import torch
from botorch.test_functions.base import ConstrainedBaseTestProblem
from botorch.utils.transforms import unnormalize
from torch import Tensor


class ConstrainedBranin(ConstrainedBaseTestProblem):
    _bounds = [(-5.0, 10.0), (0.0, 15.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = (
                X_tf[..., 1]
                - 5.1 / (4 * math.pi ** 2) * X_tf[..., 0] ** 2
                + 5 / math.pi * X_tf[..., 0]
                - 6
        )
        t2 = 10 * (1 - 1 / (8 * math.pi)) * torch.cos(X_tf[..., 0])
        return t1 ** 2 + t2 + 10

    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return 50 - (X_tf[..., 0:1] - 2.5).pow(2) - (X_tf[..., 1:2] - 7.5).pow(2)

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack_true(X)
        return torch.concat([y, c1], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 1, "Maximum of 2 Outputs allowed (task_index <= 1)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.evaluate_true(X)
        elif task_index == 1:
            return self.evaluate_slack_true(X)
        else:
            print("Error evaluate_task")
            raise


class ConstrainedFunc3(ConstrainedBaseTestProblem):
    _bounds = [(0.0, 1.0), (0.0, 1.0)]

    def __init__(self, noise_std=0.0, negate=False):
        self.dim = 2
        super().__init__(noise_std=noise_std, negate=negate)
        self._bounds = torch.tensor(self._bounds, dtype=torch.float).transpose(-1, -2)

    def evaluate_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        t1 = ( X_tf[..., 0]-1) ** 2
        t2 = (X_tf[..., 1]-0.5) ** 2
        return -t1 ** 2 -t2**2
    
    def evaluate_slack_true(self, X: Tensor) -> Tensor:
        pass
        
    def evaluate_slack1_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return (X_tf[..., 0] - 3)**2 + (X_tf[..., 1] + 2)**2 - 12
    
    def evaluate_slack2_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return 10*X_tf[..., 0]  + X_tf[..., 1] -7
    
    def evaluate_slack3_true(self, X: Tensor) -> Tensor:
        X_tf = unnormalize(X, self._bounds)
        return (X_tf[..., 0] - 0.5)**2 + (X_tf[..., 1] -0.5)**2 - 0.2

    def evaluate_black_box(self, X: Tensor) -> Tensor:
        y = self.evaluate_true(X).reshape(-1, 1)
        c1 = self.evaluate_slack1_true(X)
        c2 = self.evaluate_slack2_true(X)
        c3 = self.evaluate_slack3_true(X)
        return torch.concat([y, c1, c2, c3], dim=1)

    def evaluate_task(self, X: Tensor, task_index: int) -> Tensor:
        assert task_index <= 3, "Maximum of 4 Outputs allowed (task_index <= 3)"
        assert task_index >= 0, "No negative values for task_index allowed"
        if task_index == 0:
            return self.evaluate_true(X)
        elif task_index == 1:
            return self.evaluate_slack1_true(X)
        elif task_index == 2:
            return self.evaluate_slack2_true(X)
        elif task_index == 3:
            return self.evaluate_slack3_true(X)
        else:
            print("Error evaluate_task")
            raise
