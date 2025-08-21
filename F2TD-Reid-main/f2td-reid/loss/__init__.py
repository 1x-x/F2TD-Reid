from __future__ import absolute_import

from .triplet import TripletLoss
from .triplet_xbm import TripletLossXBM
from .crossentropy import CrossEntropyLabelSmooth
from .F2TDReid_loss import DivLoss, BridgeFeatLoss, BridgeProbLoss

__all__ = [
    'DivLoss',
    'BridgeFeatLoss',
    'BridgeProbLoss',
    'TripletLoss',
    'TripletLossXBM',
    'CrossEntropyLabelSmooth',
]
