import torch
import torch.nn as nn

class Surrogate(nn.Module):
    """Forward model mapping (params, note) -> target_latent.
    
    Architecture:
    - Input: [B, N_params + 1] (param vector [0,1] followed by normalized midi_note [0,1])
    - Hidden: 4-layer MLP with GELU activations
    - Output: [B, 128] (mean-pooled EnCodec latent)
    """
    def __init__(self, input_dim: int, output_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, params: torch.Tensor, note: torch.Tensor) -> torch.Tensor:
        # Concatenate note (normalized [0,1]) to the param vector
        x = torch.cat([params, note.unsqueeze(-1)], dim=-1)
        return self.net(x)

class SurrogateDataset(torch.utils.data.Dataset):
    """Dataset for training the surrogate model.
    
    Reads parameter vectors from parquet and pre-computed EnCodec embeddings.
    """
    def __init__(self, params: torch.Tensor, notes: torch.Tensor, latents: torch.Tensor):
        self.params = params
        self.notes = notes
        self.latents = latents

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.notes[idx], self.latents[idx]
