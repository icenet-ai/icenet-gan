import torch
import torchmetrics

from torchmetrics import Metric
from torchmetrics.functional.regression.mae import _mean_absolute_error_update
from torchmetrics.functional.regression.mse import _mean_squared_error_update


class BinaryAccuracy(Metric):
    """Binary accuracy metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """    

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Custom loss/metric for binary accuracy in classifying SIC>15% for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first six days in accuracy computation or
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        self.add_state("weighted_score", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("possible_score", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        preds = (preds > 0.15).long() # torch.Size([2, 432, 432, 7])
        target = (target > 0.15).long() # torch.Size([2, 432, 432, 7])
        base_score = preds[:, :, :, self.leadtimes_to_evaluate] == target[:, :, :, self.leadtimes_to_evaluate]
        self.weighted_score += torch.sum(base_score * sample_weight[:, :, :, self.leadtimes_to_evaluate])
        self.possible_score += torch.sum(sample_weight[:, :, :, self.leadtimes_to_evaluate])

    def compute(self):
        return self.weighted_score.float() / self.possible_score * 100.0


class SIEError(Metric):
    """
    Sea Ice Extent error metric (in km^2) for use at multiple leadtimes.
    """

    # Set class properties
    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = True

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Construct an SIE error metric (in km^2) for use at multiple leadtimes.
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider six days in computation or
                e.g., [0] to only look at the first day
                e.g., [5] to only look at the sixth day
        """
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)
        self.add_state("pred_sie", default=torch.tensor(0.), dist_reduce_fx="sum")
        self.add_state("true_sie", default=torch.tensor(0.), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, sample_weight: torch.Tensor):
        # preds and target are shape (b, h, w, t)
        preds = (preds > 0.15).long()
        target = (target > 0.15).long()
        self.pred_sie += preds[:, :, :, self.leadtimes_to_evaluate].sum()
        self.true_sie += target[:, :, :, self.leadtimes_to_evaluate].sum()

    def compute(self):
        return (self.pred_sie - self.true_sie) * 25**2 # each pixel is 25x25 km


class MAE(torchmetrics.MeanAbsoluteError):
    """MAE metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted MAE metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first six days in accuracy computation or
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        # Pass `squared=False` to get RMSE instead of MSE
        super().__init__()
        self.leadtimes_to_evaluate = leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)

    def update(self, preds, target, sample_weight: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        predictions = (preds*sample_weight)[..., self.leadtimes_to_evaluate]
        targets = (target*sample_weight)[..., self.leadtimes_to_evaluate]
        sum_abs_error, num_obs = _mean_absolute_error_update(predictions, targets)

        self.sum_abs_error += sum_abs_error
        self.total += num_obs


class RMSE(torchmetrics.MeanSquaredError):
    """RMSE metric for use at multiple leadtimes.

    Reference: https://lightning.ai/docs/torchmetrics/stable/pages/implement.html
    """

    def __init__(self, leadtimes_to_evaluate: list = None):
        """Weighted RMSE metric for multiple leadtimes.

        Args:
            leadtimes_to_evaluate: A list of leadtimes to consider
                e.g., [0, 1, 2, 3, 4, 5] to consider first six days in accuracy computation or
                e.g., [0] to only look at the first day's accuracy
                e.g., [5] to only look at the sixth day's accuracy
        """
        # Pass `squared=False` to get RMSE instead of MSE
        super().__init__(squared=False)
        self.leadtimes_to_evaluate = leadtimes_to_evaluate if leadtimes_to_evaluate is not None else slice(None)

    def update(self, preds, target, sample_weight: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        predictions = (preds*sample_weight)[..., self.leadtimes_to_evaluate]
        targets = (target*sample_weight)[..., self.leadtimes_to_evaluate]
        sum_squared_error, num_obs = _mean_squared_error_update(predictions,
                                                                targets,
                                                                num_outputs=self.num_outputs
                                                                )

        self.sum_squared_error += sum_squared_error
        self.total += num_obs
