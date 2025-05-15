#rmsnorm
import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self,dim:int,eps:float=1e-6):
        """
        Initializes the RMSNorm module.

        Args:
            dim (int): The dimensionality of the input feature space.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.
        """
        super().__init__()
        self.eps=eps
        self.w=nn.Parameter(torch.ones(dim))
    
    def norm(self,x:torch.Tensor):
        """
        Computes the root mean square normalization of the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return x * torch.rsqrt(torch.mean(x**2,-1, keepdim=True) + self.eps)
    def forward(self,x:torch.Tensor):
        """
        Forward pass of the RMSNorm module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.
        """
        return self.w * self.norm(x.float()).type_as(x)


