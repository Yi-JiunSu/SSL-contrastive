"""
  dice_coeff, multiclass_dice_coeff, & dice_loss were downloaded from
     https://github.com/milesial/Pytorch-UNet/blob/master/utils/dice_score.py

  added tversky_coeff, multiclass_tversky_coeff, tversky_loss & focal_tversky_loss by Yi-Jiun Su 
  the latest update in April 2023 by Yi-Jiun Su
"""
import torch

def dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    """
    Average of Dice coefficient for all batches, or for a single mask
    """

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    """
   for 2D -1 means 1, -2 mean 0; for 3D -1 mean 2, -2 mean 1, -3 mean 0
    """
    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    """
    Average of Dice coefficient for all classes
    """
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)
    """
    flatten(start_dim, end_dim) a contiguous range of dims into a tensor
       first dim to flattern (default = 1);  last dim to flatten (default = -1)
    """

def dice_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False):
    """
    Dice loss (objective to minimize) between 0 and 1
    """
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def tversky_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, alpha: float = 0.5, beta: float = 0.5, epsilon: float = 1e-6):
    """
      alpha = 0.5 & beta = 0.5 equivalent to Dice Score
    """
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first
    """
    Average of Tversky coefficient for all batches, or for a single mask
    """

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)
    """
    for 2D -1 means 1, -2 mean 0; for 3D -1 mean 2, -2 mean 1, -3 mean 0
    """
    TP =(input * target).sum(dim=sum_dim)
    FP = ((1-target)*input).sum(dim=sum_dim)
    FN = (target*(1-input)).sum(dim=sum_dim)

    tversky = (TP + epsilon) / (TP + alpha*FP +beta*FN + epsilon)
    return tversky.mean()

def multiclass_tversky_coeff(input: torch.Tensor, target: torch.Tensor, reduce_batch_first: bool = False, alpha: float = 0.5, beta: float=0.5,epsilon: float = 1e-6):
    """
    Average of Tversky coefficient for all classes
    """
    return tversky_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)
    """
    flatten(start_dim, end_dim) a contiguous range of dims into a tensor
       first dim to flattern (default = 1);  last dim to flatten (default = -1)
    """

def tversky_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False, alpha: float=0.5, beta: float=0.5):
    """
    Tversky loss (objective to minimize) 
    """
    fn = multiclass_tversky_coeff if multiclass else tversky_coeff
    return 1 - fn(input, target, reduce_batch_first=True, alpha=alpha, beta=beta)

def focal_tversky_loss(input: torch.Tensor, target: torch.Tensor, multiclass: bool = False, alpha: float=0.5, beta: float=0.5, gamma: float=2.):
    """
    Focal Tversky Loss (objective to minimize)
    """
    fn = multiclass_tversky_coeff if multiclass else tversky_coeff
    return (1 - fn(input, target, reduce_batch_first=True, alpha=alpha, beta=beta))**gamma

