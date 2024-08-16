import cv2 as cv
import numpy as np
import torch
from torch import nn
from scipy.ndimage import distance_transform_edt as edt
"""
   Extract euclidean distance transform
     the index of the closest background element is returned along the first axis of the result
   edt(input, sampling=None, return_distance=True, return_indexes=False, distance=None, Indcies=None)
     input - input data to transform. will be convertedinto binary 1 True 0 elsewhere
   scipy.ndimage.morphology.distance_transform_edt is deprecated
"""
from scipy.ndimage import convolve
"""
   multidimensional convolution
   convolve(input, weights, output=None, mode='reflect', cval=0.0, origin=0)
     weights array has the same number of dimensions as input
     mode {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
     reflect - half-sample symmetric - reflect about the edge of the last pixel
          (dcba | abcd | dcba)
     constant - filll all values beyond the edge with the same constant value
          (kkkk | abcd | kkkk)
     cval - value fill pas edge of input mode "constant" default=0.0
     nearest - the input is extended by replicating the last pixel
          (aaaa | abcd | dddd)
     mirro - whole-sample symmetric - reflecting about the center of the last pixel
          (dcb | abcd | cba)
     wrap - wrapping around the opposite edge
          (abcd | abcd | abcd )

Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf

copy pasted from - all credit goes to original authors:
https://github.com/SilmarilBearer/HausdorffLoss

There are three methords discussed in the paper
(1) Estimation of the Hausdorff Distance based on Distance Transformation 
    HausdorffDTLoss
(2) Estimation of the Hausdorff Distance using Morphological Operations (Erosion)
    HausdorffERLoss (using 3x3 as 2D Kernel; 3x3x3 as 3d Kernel)
(3) Estimation of the Hausdorff Distance using Convolutions with Circular/Spherical Kernels
    not described here

    The Hausdorff classes were downloaded from https://github.com/PatRyg99/HausdorffLoss
"""

def hausdorff_distance(pred, ref):
    """ 
    Compute symmetric point-wise distance between two binary masks

    args: pred: predicted ndarray
          ref: reference ndarray
    Return:
       Dict[str, float]: keys are 'mean', 'median', 'std', 'max'

       It's not in the original file. 
       Added for test metrics
       Yi-Jiun Su 2023/03/28
    """
    pred_distance = edt(np.logical_not(pred))
    ref_distance = edt(np.logical_not(ref))
    ref2pred = pred_distance[np.nonzero(ref)]
    pred2ref = ref_distance[np.nonzero(pred)]
    all_distance = np.concatenate((ref2pred,pred2ref),axis=None)
    all_distance[all_distance <= 0.0] = 0.0
    std = np.std(all_distance)
    c95 = 1.96*std/np.sqrt(len(all_distance))
    return {
        "mean": np.mean(all_distance),
        "median": np.median(all_distance),
        "std": std,
        "c95": c95,
        "var": np.var(all_distance),
        "max": np.max(all_distance),
    }

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

#        pred = torch.sigmoid(pred)

        pred_dt = torch.from_numpy(self.distance_field(pred.cpu().detach().numpy())).float()
        target_dt = torch.from_numpy(self.distance_field(target.cpu().detach().numpy())).float()
        pred_dt = pred_dt.to(device=0)
        target_dt = target_dt.to(device=0)

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = pred_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss

class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha  
        self.erosions = erosions 
        """
        alpha - specified how strongly penalizing larger segmentation error
        erosions -  total numbers of erosions
            increasing the numbers of erosions increase computational cost
            however, needs to be large enough because all part that remain after
        the numbers of erosions will be weighted equally
        """
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        """
        cross = [ [ 0, 1, 0], [1, 1, 1], [0, 1, 0]] a cross in a 2D 3x3 array
        """
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        """
        make 2D Kernel sumation of all elements to be 1
        """
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)
        """
        3D Kernel of size 3 with the center element and its 6-neighbors set to 1/7
            and the remaining 20 elements are set to zero
        """
    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
            Uses one binary channel: 1 - fg, 0 - bg
            pred: (b, 1, x, y, z) or (b, 1, x, y)
            target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

#        pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().detach().numpy(), target.cpu().detach().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().detach().numpy(), target.cpu().detach().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss
