import torch
import pytest
import os
import numpy as np
import pandas as pd
from mimic_synth.s04_surrogate.model import Surrogate, SurrogateMRSTFTHead, MRSTFT_DIM

# ── Existing tests (legacy path, use_film=False) ──────────────────────────────

def test_surrogate_shapes():
    input_dim = 10 + 1  # 10 params + 1 note
    model = Surrogate(input_dim=input_dim, output_dim=128)

    params = torch.randn(8, 10)
    note   = torch.randn(8)

    out = model(params, note)
    assert out.shape == (8, 128)

def test_surrogate_gradients():
    input_dim = 5 + 1
    model  = Surrogate(input_dim=input_dim)
    params = torch.randn(1, 5, requires_grad=True)
    note   = torch.tensor([0.5], requires_grad=True)

    out  = model(params, note)
    loss = out.sum()
    loss.backward()

    assert params.grad is not None
    assert note.grad is not None
    assert not torch.isnan(params.grad).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_autocast_compatibility():
    model  = Surrogate(input_dim=11).cuda()
    params = torch.randn(8, 10).cuda()
    note   = torch.randn(8).cuda()

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(params, note)

    assert out.dtype in (torch.float32, torch.float16)
    assert out.shape == (8, 128)

# ── FiLM architecture tests ────────────────────────────────────────────────────

def test_film_surrogate_shapes():
    model  = Surrogate(input_dim=11, hidden_dim=1024, use_film=True)
    params = torch.randn(8, 10)
    note   = torch.rand(8)
    out    = model(params, note)
    assert out.shape == (8, 128)

def test_film_note_changes_output():
    model  = Surrogate(input_dim=11, hidden_dim=256, use_film=True)
    model.eval()
    params = torch.rand(1, 10)
    note_a = torch.tensor([0.2])
    note_b = torch.tensor([0.8])

    with torch.no_grad():
        out_a = model(params, note_a)
        out_b = model(params, note_b)
    assert not torch.allclose(out_a, out_b), "FiLM: same params, different note should change output"

    params_a = torch.rand(1, 10)
    params_b = torch.rand(1, 10)
    with torch.no_grad():
        out_pa = model(params_a, note_a)
        out_pb = model(params_b, note_a)
    assert not torch.allclose(out_pa, out_pb), "FiLM: different params should change output"

def test_film_surrogate_gradients():
    model  = Surrogate(input_dim=11, hidden_dim=256, use_film=True)
    params = torch.rand(1, 10, requires_grad=True)
    note   = torch.tensor([0.5], requires_grad=True)

    out  = model(params, note)
    loss = out.sum()
    loss.backward()

    assert params.grad is not None, "FiLM: grads must flow to params"
    assert note.grad is not None,   "FiLM: grads must flow to note"
    assert not torch.isnan(params.grad).any()
    assert not torch.isnan(note.grad).any()

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_film_autocast_compatibility():
    model  = Surrogate(input_dim=11, hidden_dim=1024, use_film=True).cuda()
    params = torch.randn(8, 10).cuda()
    note   = torch.rand(8).cuda()

    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
        out = model(params, note)

    assert out.dtype in (torch.float32, torch.float16)
    assert out.shape == (8, 128)

# ── New feature tests ──────────────────────────────────────────────────────────

def test_clap_output_dim():
    model  = Surrogate(input_dim=11, output_dim=512, use_film=True)
    params = torch.randn(8, 10)
    note   = torch.rand(8)
    out    = model(params, note)
    assert out.shape == (8, 512)

def test_forward_features_shape():
    hidden_dim = 256
    for use_film in (True, False):
        model   = Surrogate(input_dim=11, hidden_dim=hidden_dim, use_film=use_film)
        params  = torch.randn(8, 10)
        note    = torch.rand(8)
        feats   = model.forward_features(params, note)
        assert feats.shape == (8, hidden_dim), \
            f"forward_features shape wrong for use_film={use_film}: {feats.shape}"

def test_mrstft_head():
    hidden_dim  = 1024
    batch       = 8
    model       = Surrogate(input_dim=11, hidden_dim=hidden_dim, use_film=True)
    mrstft_head = SurrogateMRSTFTHead(hidden_dim)

    params = torch.randn(batch, 10)
    note   = torch.rand(batch)
    hidden = model.forward_features(params, note)
    out    = mrstft_head(hidden)
    assert out.shape == (batch, MRSTFT_DIM), \
        f"MRSTFT head output shape wrong: {out.shape}, expected ({batch}, {MRSTFT_DIM})"

def test_n_params_attribute():
    for use_film in (True, False):
        model = Surrogate(input_dim=31, use_film=use_film)
        assert model.n_params == 30, \
            f"n_params wrong for use_film={use_film}: {model.n_params}"
