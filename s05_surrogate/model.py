import torch
import torch.nn as nn
import torch.nn.functional as F

MRSTFT_DIM = 1924  # 129+257+513+1025 (four FFT sizes in mrstft_feats)


class FiLMResBlock(nn.Module):
    """Residual block with LayerNorm and per-layer FiLM note conditioning.

    h' = (1 + γ) * LayerNorm(h + net(h)) + β
    where γ, β are linear projections of the note scalar.
    The (1+γ) form keeps γ near zero at init, preserving training stability.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net  = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.norm = nn.LayerNorm(dim)
        self.film = nn.Linear(1, 2 * dim)  # note [B,1] → γ,β each [B,dim]

    def forward(self, x: torch.Tensor, note: torch.Tensor) -> torch.Tensor:
        h = self.norm(x + self.net(x))
        gamma, beta = self.film(note.unsqueeze(-1)).chunk(2, dim=-1)
        return (1 + gamma) * h + beta


class Surrogate(nn.Module):
    """Forward model mapping (params, note) -> target_latent.

    Architecture (use_film=False, legacy):
      Input: [B, N_params + 1] (param vector [0,1] + normalized midi_note [0,1])
      4-layer MLP with GELU activations
      Output: [B, output_dim]

    Architecture (use_film=True):
      params  → Linear → GELU → LayerNorm → 3×FiLMResBlock(note) → Linear
      Output: [B, output_dim]
    """

    def __init__(self, input_dim: int, output_dim: int = 128,
                 hidden_dim: int = 512, use_film: bool = False):
        super().__init__()
        self.use_film  = use_film
        self.n_params  = input_dim - 1  # excludes the note dimension
        self.output_dim = output_dim

        if use_film:
            self.input_proj = nn.Linear(input_dim - 1, hidden_dim)
            self.input_norm = nn.LayerNorm(hidden_dim)
            self.blocks     = nn.ModuleList([FiLMResBlock(hidden_dim) for _ in range(3)])
            self.out_proj   = nn.Linear(hidden_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, output_dim),
            )

    def forward_features(self, params: torch.Tensor, note: torch.Tensor) -> torch.Tensor:
        """Return the pre-projection hidden state [B, hidden_dim].

        Used by SurrogateMRSTFTHead during training to compute the auxiliary
        spectral loss without modifying the saved inference weights.
        """
        if self.use_film:
            x = F.gelu(self.input_norm(self.input_proj(params)))
            for block in self.blocks:
                x = block(x, note)
            return x
        else:
            x = torch.cat([params, note.unsqueeze(-1)], dim=-1)
            for layer in list(self.net)[:-1]:  # all layers except final Linear
                x = layer(x)
            return x

    def forward(self, params: torch.Tensor, note: torch.Tensor) -> torch.Tensor:
        if self.use_film:
            return self.out_proj(self.forward_features(params, note))
        else:
            x = torch.cat([params, note.unsqueeze(-1)], dim=-1)
            return self.net(x)


class SurrogateMRSTFTHead(nn.Module):
    """Auxiliary training head predicting MRSTFT spectral features.

    Only instantiated during training when --mrstft-features is supplied.
    Never saved to state_dict.pt — excluded explicitly in train.py.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.head = nn.Linear(hidden_dim, MRSTFT_DIM)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.head(hidden)


class SurrogateDataset(torch.utils.data.Dataset):
    """Dataset for training the surrogate model.

    Reads parameter vectors from parquet and pre-computed embeddings.
    Optionally loads MRSTFT auxiliary features.
    """

    def __init__(self, params: torch.Tensor, notes: torch.Tensor,
                 latents: torch.Tensor, mrstft: torch.Tensor | None = None):
        self.params  = params
        self.notes   = notes
        self.latents = latents
        self.mrstft  = mrstft

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        if self.mrstft is not None:
            return self.params[idx], self.notes[idx], self.latents[idx], self.mrstft[idx]
        return self.params[idx], self.notes[idx], self.latents[idx]
