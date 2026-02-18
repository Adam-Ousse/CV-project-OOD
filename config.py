import torch

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

BATCH_SIZE = 128
NUM_WORKERS = 4

NUM_CLASSES = 100
NUM_EPOCHS = 200
LEARNING_RATE = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4

LR_MILESTONES = [60, 120, 160]
LR_GAMMA = 0.2

DATA_DIR = './data'
CHECKPOINT_DIR = './checkpoints'
RESULTS_DIR = './results'

TEMPERATURE = 1.0
MAHALANOBIS_LAYERS = ['layer4']
