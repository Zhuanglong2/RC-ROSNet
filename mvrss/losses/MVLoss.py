import torch
import torch.nn as nn


class MVLoss(nn.Module):
    """
    Compute the Unsupervised Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(MVLoss, self).__init__()
        self.global_weight = global_weight
        self.MVLoss = torch.nn.HuberLoss(reduction ='mean')
        #self.MVLoss = nn.MSELoss() #instead?
    def forward(self, rd_input: torch.Tensor,
                ra_input: torch.Tensor) -> torch.Tensor:
        """Forward pass to compute the loss between the two predicted view masks"""
        rd_softmax = nn.Softmax(dim=1)(rd_input)
        ra_softmax = nn.Softmax(dim=1)(ra_input)
        rd_range_probs = torch.max(rd_softmax, dim=3, keepdim=True)[0]
        # Rotate RD Range vect to match zero range
        rd_range_probs = torch.rot90(rd_range_probs, 2, [2, 3])
        ra_range_probs = torch.max(ra_softmax, dim=3, keepdim=True)[0]
        loss = self.MVLoss(rd_range_probs, ra_range_probs)
        loss = self.global_weight*loss
        return loss


import torch
import torch.nn as nn


class MSELoss(nn.Module):
    """
    Compute the Unsupervised Coherence Loss

    PARAMETERS
    ----------
    global_weight: float
        Global weight to apply on the entire loss
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(MSELoss, self).__init__()
        self.global_weight = global_weight
        self.MSELoss = torch.nn.MSELoss()
        #self.MVLoss = nn.MSELoss() #instead?
    def forward(self, ra_input, true):

        loss = self.MSELoss(ra_input, true)
        loss = self.global_weight*loss
        return loss
