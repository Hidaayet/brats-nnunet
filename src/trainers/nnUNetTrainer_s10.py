"""nnU-Net trainers with a resumable checkpoint every 10 epochs.

Default nnU-Net writes checkpoint_latest.pth only every 50 epochs, so a dead free-tier
session can lose up to 50 epochs. These variants lose at most 10.
"""
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_100ep_s10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 100
        self.save_every = 10


class nnUNetTrainer_250ep_s10(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 250
        self.save_every = 10
