import pytest
from utils import RMSNorm
import torch

def test_output_shape():
    x = torch.randn(8, 16)
    norm = RMSNorm(dim=16)
    out = norm(x)
    assert out.shape == x.shape, "Output shape should match input shape"

def test_known_input():
    x = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
    norm = RMSNorm(dim=2, eps=0.0)
    norm.w.data = torch.tensor([1.0, 1.0])
    out = norm(x)
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
    expected = x / rms
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)

def test_gradient_flow():
    x = torch.randn(10, 32, requires_grad=True)
    norm = RMSNorm(dim=32)
    out = norm(x)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradients should be propagated through RMSNorm"
    assert not torch.isnan(x.grad).any(), "No NaNs should appear in the gradient"

# def test_eps_effect():
#     x = torch.ones(2, 4) * 1000  # large magnitude to test eps
#     norm_small_eps = RMSNorm(dim=4, eps=1e-12)
#     norm_large_eps = RMSNorm(dim=4, eps=1e-1)
#     out_small = norm_small_eps(x)
#     out_large = norm_large_eps(x)
#     assert not torch.allclose(out_small, out_large), "Outputs should differ due to different eps values"

def test_weight_initialization():
    norm = RMSNorm(dim=10)
    assert torch.allclose(norm.w.data, torch.ones(10)), "Weights should initialize to ones"