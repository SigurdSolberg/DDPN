import torch.nn as nn
import torch

class DSSN(nn.Module):
    """
        Deep Set of Set Network

        Args:
        - inner_transform (nn.Module):  DeepSet network applied to each element of the outerset
        - outer_transform (nn.Module):  DeepSet network applied to the outer set
        - downstream_network (nn.Module):  Network applied to the output of the outer_transform for downstream tasks
    """

    def __init__(self, inner_transform, outer_transform, downstream_network, norm = True, device = 'cpu'):

        super(DSSN, self).__init__()

        self.device = device
        self.inner_transform = inner_transform
        self.outer_transform = outer_transform
        self.downstream_network = downstream_network
        if norm:
            self.norm = nn.Identity()
        else:
            self.norm = SetBatchNorm()

    def forward(self:nn.Module, x) -> torch.Tensor: 
        x = torch.sum(self.inner_transform(x), dim = -2)
        x = self.norm(x)
        x = torch.mean(self.outer_transform(x), dim = -2)
        x = self.downstream_network(x)
        return x
     
class SetBatchNorm(nn.Module):
    """
    Batch normalization specifically tailored for set data.

    Note:
        Expands the input tensor to apply 2D batch normalization and then squeezes it back.
        Operates on 4D tensor, but casting fixes this.
    """

    def __init__(self, ) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.norm(x)
        x = x.squeeze(1)
        return x

class PersLay(nn.Module):
    
        def __init__(self, rho, phi = None) -> None:
            super(PersLay, self).__init__()
    
            self.rho = rho      # Vectorization of elements, use DeepSet.Module
            if phi is None:
                self.phi = nn.Identity()
            else:
                self.phi = phi      # Weighting of elements, use DeepSet.Module
    
        def forward(self, x:torch.Tensor) -> torch.Tensor:
            # Only operate on the o-h-e og feature dimension
            return self.rho(x)*self.phi(x)

class PersLay_DDPN(PersLay):

    def __init__(self, rho, phi = None) -> None:
        super(PersLay_DDPN, self).__init__(rho, phi)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # Only operate on the o-h-e og feature dimension
        return self.rho(x)*self.phi(x[..., 3:])