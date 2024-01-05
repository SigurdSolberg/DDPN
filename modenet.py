import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet

import numpy as np
import pandas as pd

from src.distributed_homology import DistributedHomology
from src.models import DSSN, PersLay_DDPN, PersLay, Norm
from src.training_tools import TrainingLoop, DHDataset, TrainingLoopWithSampling, SamplingDataset
from src.normalization import normalize_size, normalize_size_dwise

from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader

device = torch.device('cpu')


K = 1000
M = 1
DIM = 256

pre_transform = T.NormalizeScale()
transform = T.SamplePoints(10000)

train_dataset = ModelNet(
    root="data/ModelNet10", name='10', train=True,
    transform=transform, pre_transform=pre_transform
)
val_dataset = ModelNet(
    root="data/ModelNet10", name='10', train=False,
    transform=transform, pre_transform=pre_transform
)

val_data = []
val_labels = []
for i in val_dataset:
    val_data.append(i.pos.numpy())
    val_labels.append(i.y.numpy()[0])

train_data = []
train_labels = []
for i in train_dataset:
    train_data.append(i.pos.numpy())
    train_labels.append(i.y.numpy()[0])

val_labels = pd.get_dummies(val_labels).to_numpy()
train_labels = pd.get_dummies(train_labels).to_numpy()

# Compute the distributed homology
'''dh = DistributedHomology()
val_data = dh.get_subsets(val_data, m = M, k = K)
print(f'Val data shape: {val_data.shape}')'''

train_dataset = DHDataset(np.array(train_data), train_labels)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = DHDataset(np.array(val_data), val_labels)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True) # Use batch size 1 to verify the that the batchnorm works


class PermEqui2_max(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(PermEqui2_max, self).__init__()
    self.Gamma = nn.Linear(in_dim, out_dim)
    self.Lambda = nn.Linear(in_dim, out_dim, bias=False)

  def forward(self, x):
    xm, _ = x.max(-2, keepdim=True)
    xm = self.Lambda(xm) 
    x = self.Gamma(x)
    x = x - xm
    return x

# Create a DDPN model
'''inner_rho = nn.Sequential(
        nn.Linear(3, 12),
        nn.LeakyReLU(),
        nn.Linear(12, 16),
        ).to(device)'''

inner_rho = nn.Sequential(
    PermEqui2_max(3, DIM),
    nn.ELU(inplace=True),
    PermEqui2_max(DIM, DIM),
    nn.ELU(inplace=True),
    PermEqui2_max(DIM, DIM),
    nn.ELU(inplace=True),
)

outer_rho = nn.Sequential(
    #nn.Linear(DIM, DIM),
    #nn.LeakyReLU(),
    #nn.Linear(DIM, DIM),
    ).to(device)

downstream_network = nn.Sequential( 
            #nn.BatchNorm1d(DIM),
            nn.Dropout(0.5),
            nn.Linear(DIM, DIM),
            nn.ReLU(),
            #nn.BatchNorm1d(DIM),
            nn.Dropout(0.5),
            nn.Linear(DIM, 10),
            ).to(device)

model = DSSN(
    inner_transform = inner_rho,#PersLay_DDPN(rho=inner_rho, phi=inner_phi), 
    outer_transform = outer_rho, 
    downstream_network = downstream_network,
    device = device,
    norm=False
    )


# Train the model
lr = 1e-02
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = ExponentialLR(optimizer=optimizer, gamma=0.99)

training_loop = TrainingLoopWithSampling(model=model, optimizer=optimizer, loss_function=loss, device=device, scheduler=scheduler, k = K, m = M)
training_loop.train(train_loader=train_loader, val_loader=val_loader, epochs=1000, verbose=True)

model.visualize_outer(train_dataset.x, train_dataset.y)