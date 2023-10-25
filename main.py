import numpy as np

from src.distributed_homology import DistributedHomology
from src.models import DSSN, PersLay_DDPN
from src.training_tools import TrainingLoop, DHDataset
from src.normalization import normalize_size

from torch.optim.lr_scheduler import ExponentialLR
import torch.nn as nn
import torch.optim as optim
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

device = torch.device('cpu')

if __name__ == '__main__':

    # Load the data
    X = np.load('data/X_clouds.npy', allow_pickle=True)[::1]
    Y = torch.tensor(np.load('data/Y.npy'), dtype=torch.float)[::1]
    print(f'Number of pointclouds: {len(X)}')

    # Compute the distributed homology
    dh = DistributedHomology()
    data = dh.fit(X, m = 10 , k = 50, normalization=[normalize_size], max_featues=150)
    print(f'data shape: {data.shape}')

    # Create a dataloader
    train_data, val_data, train_labels, val_labels = train_test_split(data, Y, test_size=0.3, random_state=42)
    train_dataset = DHDataset(train_data, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataset = DHDataset(val_data, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True) # Use batch size 1 to verify the that the batchnorm works


    # Create a DDPN model
    inner_rho = nn.Sequential(
            nn.Linear(6, 12),
            nn.LeakyReLU(),
            nn.Linear(12, 16),
            ).to(device)
    
    inner_phi = nn.Sequential(
            nn.Linear(3, 16, bias = False),
            ).to(device)

    outer_rho = nn.Sequential(
        nn.Linear(16, 64),
        nn.LeakyReLU(),
        nn.Linear(64, 128),
        ).to(device)

    downstream_network = nn.Sequential( 
                nn.BatchNorm1d(128),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 5),
                ).to(device)

    model = DSSN(
        inner_transform = PersLay_DDPN(rho=inner_rho, phi=inner_phi), 
        outer_transform = outer_rho, 
        downstream_network = downstream_network,
        device = 'cpu'
        )
    

    # Train the model
    lr = 1e-02
    loss = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.98)
    
    training_loop = TrainingLoop(model=model, optimizer=optimizer, loss_function=loss, device=device, scheduler=scheduler)
    training_loop.train(train_loader=train_loader, val_loader=val_loader, epochs=500, verbose=True)
    

