import torch
import argparse
import numpy as np
import random

def set_config(args):
    pass


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
class Config:
    def __init__(self):
        
        # Device settings
        self.device = 'cuda'
        self.sync_bn = False

        # General settings
        self.seed = 60
        self.num_gpus = 2
        self.num_workers = 0

        # Dataset settings
        self.dataset = 'PBVS'  # Options: 'PBVS', 'NIR'
        self.mean_shift = False  # Apply mean shift
        self.psnr_up = 0.0
        self.real_data = False  # MR->HR
        self.data_range = 1  # LR Patch
        self.cached = True

        # Data augmentation
        self.repeated_aug = False
        self.show_every = 20.0
        self.print_freq = 10
        self.rgb_norm = False

        # Optimizer settings
        self.lr = 3e-4
        self.opt = 'AdamW'
        self.loss = '1*L1'
        self.hdelta = 1.0  # HuberLoss

        # Training settings
        self.epochs = 100
        self.sched = 'multistep'  # Options: 'cosine', 'multistep'
        self.weight_decay = 0.0
        self.warmup_epochs = 0
        self.cooldown_epochs = 10
        self.min_lr = 1e-5
        self.warmup_lr = 1e-5
        self.decay_rate = 0.5
        self.decay_epochs = '50'

        # Training Stats
        self.resume = False
        self.test_only = False
        self.start_epoch = 0
        self.checkpoint_hist = 10
        self.load_name = 'model_best.pth'

        # Model settings
        self.scale = 2
        self.batch_size = 10
        self.patch_size = 4  # LR Patch
        self.val_batch_size = 1
        self.in_channels = 1
        self.model_name = 'Base2'  # Options: 'NAFNet', 'SWinIR', 'UFormer', etc.
        self.test_name = 'val'

        # UFormer settings
        self.embed_dim = 20
        self.win_size = 8
        self.pre_trained = False

        # Residual learning
        self.no_res = False
        self.light_model = False
        self.model_path = ''  # For SwinIR

        # MixUp settings
        self.mix_up = False
        self.mix_alpha = 0.1

        # Self-ensemble
        self.self_ensemble = False
        self.ensemble_mode = 'mean'

        # TLC enhancement
        self.tlc_enhance = False

        # Distributed training
        self.dist_url = 'env://'
        self.world_size = 1
        self.local_rank = 0

        # Save results
        self.save_result = False
        self.file_name = ''
        set_random_seed(self.seed)

    def __repr__(self):
        return str(vars(self))

CONFIG = Config()

