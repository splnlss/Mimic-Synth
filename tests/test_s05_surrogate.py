import torch
import pytest
import os
import numpy as np
import pandas as pd
from s05_surrogate.model import Surrogate

def test_surrogate_shapes():
    input_dim = 10 + 1 # 10 params + 1 note
    model = Surrogate(input_dim=input_dim, output_dim=128)
    
    params = torch.randn(8, 10)
    note = torch.randn(8)
    
    out = model(params, note)
    assert out.shape == (8, 128)

def test_surrogate_gradients():
    input_dim = 5 + 1
    model = Surrogate(input_dim=input_dim)
    params = torch.randn(1, 5, requires_grad=True)
    note = torch.tensor([0.5], requires_grad=True)
    
    out = model(params, note)
    loss = out.sum()
    loss.backward()
    
    assert params.grad is not None
    assert note.grad is not None
    assert not torch.isnan(params.grad).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocast_compatibility():
    model = Surrogate(input_dim=11).cuda()
    params = torch.randn(8, 10).cuda()
    note = torch.randn(8).cuda()
    
    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(params, note)
    
    assert out.dtype in (torch.float32, torch.float16) 
    assert out.shape == (8, 128)
