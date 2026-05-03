"""
models.py — All 10 neural architectures evaluated in the paper.

Architectures:
  CNN family:      SEResNet, ResNet, ConvNeXt
  CNN-RNN hybrid:  CNN_GRU, CNN_BiLSTM
  RNN:             GRU_DAS
  SSM/Mamba:       MambaDAS
  GNN:             DAS_GNN
  Transformer:     DASViT
  CNN-Transformer: Conformer

Input shape: (B, 1, 361, 2400) — single-channel DAS gather
             B=batch, 1=channel, 361=fiber channels, 2400=time samples
Output:      (B, 2) — binary logits [noise, event]

Usage:
    from architectures.models import SEResNet, Conformer
    model = SEResNet().to(device)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════
# SHARED BUILDING BLOCKS
# ══════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al. 2018)."""
    def __init__(self, ch, r=16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(ch, ch//r), nn.ReLU(True),
            nn.Linear(ch//r, ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.se(x).view(x.size(0), -1, 1, 1)


class SEResBlock(nn.Module):
    """SE-ResNet residual block with Dropout2d."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch))
        self.se  = SEBlock(ch)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.se(self.block(x)) + x)


class ResBlock(nn.Module):
    """Vanilla residual block (no SE attention)."""
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.GELU(), nn.Dropout2d(0.1),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch))
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.block(x) + x)


def _downsample(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(co), nn.GELU())


def _shared_stem():
    """Shared stem used by SEResNet, ResNet, ConvNeXt, CNN-RNN, Conformer."""
    return nn.Sequential(
        nn.Conv2d(1, 32, (7,15), stride=(2,4), padding=(3,7), bias=False),
        nn.BatchNorm2d(32), nn.GELU(),
        nn.MaxPool2d((3,5), stride=(2,2), padding=(1,2)))


def _shared_head(in_ch=256, mid=128):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.5),
        nn.Linear(in_ch, mid), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(mid, 2))


# ══════════════════════════════════════════════════════════════
# 1. SE-RESNET (primary backbone for URDI)
# ══════════════════════════════════════════════════════════════

class SEResNet(nn.Module):
    """
    SE-ResNet with squeeze-and-excitation channel attention.
    Achieves FORGE F1 = 0.697±0.022 (5 seeds, zero-shot).
    Selected as URDI backbone due to lowest seed variance.
    """
    def __init__(self):
        super().__init__()
        self.stem = _shared_stem()
        self.body = nn.Sequential(
            SEResBlock(32), SEResBlock(32), _downsample(32, 64),
            SEResBlock(64), SEResBlock(64), _downsample(64, 128),
            SEResBlock(128), SEResBlock(128), _downsample(128, 256),
            SEResBlock(256), SEResBlock(256))
        self.head = _shared_head()

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ══════════════════════════════════════════════════════════════
# 2. RESNET (vanilla, no SE attention)
# ══════════════════════════════════════════════════════════════

class ResNet(nn.Module):
    """
    Vanilla ResNet without channel attention.
    FORGE F1 = 0.696±0.026 (5 seeds).
    """
    def __init__(self):
        super().__init__()
        self.stem = _shared_stem()
        self.body = nn.Sequential(
            ResBlock(32), ResBlock(32), _downsample(32, 64),
            ResBlock(64), ResBlock(64), _downsample(64, 128),
            ResBlock(128), ResBlock(128), _downsample(128, 256),
            ResBlock(256), ResBlock(256))
        self.head = _shared_head()

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ══════════════════════════════════════════════════════════════
# 3. CONVNEXT
# ══════════════════════════════════════════════════════════════

class ConvNeXtBlock(nn.Module):
    """Simplified ConvNeXt block (Liu et al. 2022)."""
    def __init__(self, ch):
        super().__init__()
        self.dwconv  = nn.Conv2d(ch, ch, 7, padding=3, groups=ch, bias=False)
        self.norm    = nn.LayerNorm(ch)
        self.pwconv1 = nn.Linear(ch, 4*ch)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4*ch, ch)

    def forward(self, x):
        res = x
        x = self.dwconv(x)
        x = x.permute(0,2,3,1)          # (B,H,W,C) for LayerNorm
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        return (x.permute(0,3,1,2) + res)


class ConvNeXt(nn.Module):
    """
    ConvNeXt DAS detector.
    FORGE F1 = 0.716±0.145 (high variance — unsuitable for safety-critical use).
    """
    def __init__(self):
        super().__init__()
        self.stem = _shared_stem()
        self.body = nn.Sequential(
            ConvNeXtBlock(32), ConvNeXtBlock(32), _downsample(32, 64),
            ConvNeXtBlock(64), ConvNeXtBlock(64), _downsample(64, 128),
            ConvNeXtBlock(128), ConvNeXtBlock(128), _downsample(128, 256),
            ConvNeXtBlock(256), ConvNeXtBlock(256))
        self.head = _shared_head()

    def forward(self, x):
        return self.head(self.body(self.stem(x)))


# ══════════════════════════════════════════════════════════════
# 4. CNN-GRU
# ══════════════════════════════════════════════════════════════

class CNN_GRU(nn.Module):
    """
    CNN spatial front-end + bidirectional GRU temporal modeling.
    FORGE F1 = 0.676±0.000 (degenerate: majority-class predictor on all 5 seeds).
    Structural failure under 78.3%→51.0% event-rate prior shift.
    """
    def __init__(self, hidden=256):
        super().__init__()
        self.stem = _shared_stem()
        self.cnn  = nn.Sequential(
            SEResBlock(32), _downsample(32, 64),
            SEResBlock(64), _downsample(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, None))  # collapse spatial dim
        self.gru  = nn.GRU(128, hidden//2, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Dropout(0.5), nn.Linear(hidden, 2))

    def forward(self, x):
        x = self.cnn(self.stem(x))              # (B, 128, H', T')
        x = self.pool(x).squeeze(2)             # (B, 128, T')
        x = x.permute(0, 2, 1)                  # (B, T', 128)
        out, _ = self.gru(x)                    # (B, T', hidden)
        x = out.mean(1)                          # global average
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# 5. CNN-BILSTM
# ══════════════════════════════════════════════════════════════

class CNN_BiLSTM(nn.Module):
    """CNN spatial front-end + bidirectional LSTM."""
    def __init__(self, hidden=256):
        super().__init__()
        self.stem = _shared_stem()
        self.cnn  = nn.Sequential(
            SEResBlock(32), _downsample(32, 64),
            SEResBlock(64), _downsample(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.lstm = nn.LSTM(128, hidden//2, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden, 2))

    def forward(self, x):
        x = self.cnn(self.stem(x))
        x = self.pool(x).squeeze(2).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.head(out.mean(1))


# ══════════════════════════════════════════════════════════════
# 6. GRU (pure RNN, no CNN front-end)
# ══════════════════════════════════════════════════════════════

class GRU_DAS(nn.Module):
    """
    Pure bidirectional GRU: each DAS channel = one time step.
    FORGE F1 = 0.000 (complete collapse under event-rate prior shift).
    """
    def __init__(self, in_ch=2400, hidden=256):
        super().__init__()
        self.gru  = nn.GRU(in_ch, hidden//2, num_layers=2,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(hidden, 2))

    def forward(self, x):
        # x: (B, 1, 361, 2400) → treat channels as time steps
        x = x.squeeze(1)                        # (B, 361, 2400)
        out, _ = self.gru(x)
        return self.head(out.mean(1))


# ══════════════════════════════════════════════════════════════
# 7. MAMBA-DAS (SSM)
# ══════════════════════════════════════════════════════════════

class SimplifiedMambaBlock(nn.Module):
    """
    Simplified selective SSM block approximating Mamba (Gu & Dao 2023).
    Full Mamba requires the mamba-ssm package; this approximation uses
    a gated CNN with selective depth-wise convolution.
    """
    def __init__(self, ch):
        super().__init__()
        self.norm  = nn.LayerNorm(ch)
        self.gate  = nn.Linear(ch, ch*2)
        self.dconv = nn.Conv1d(ch, ch, 4, padding=3, groups=ch, bias=False)
        self.proj  = nn.Linear(ch, ch)

    def forward(self, x):
        # x: (B, T, C)
        res = x
        x = self.norm(x)
        gate, val = self.gate(x).chunk(2, dim=-1)
        val = val.permute(0,2,1)               # (B, C, T)
        val = self.dconv(val)[:,:,:x.size(1)].permute(0,2,1)
        x = F.silu(gate) * val
        return self.proj(x) + res


class MambaDAS(nn.Module):
    """
    Mamba-DAS: selective SSM on temporal features from CNN front-end.
    FORGE F1 = 0.172 (collapse under prior shift).
    """
    def __init__(self):
        super().__init__()
        self.stem = _shared_stem()
        self.cnn  = nn.Sequential(
            SEResBlock(32), _downsample(32, 64),
            SEResBlock(64), _downsample(64, 128))
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.ssm  = nn.Sequential(*[SimplifiedMambaBlock(128) for _ in range(4)])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(128, 2))

    def forward(self, x):
        x = self.cnn(self.stem(x))              # (B, 128, H', T')
        x = self.pool(x).squeeze(2)             # (B, 128, T')
        x = x.permute(0, 2, 1)                  # (B, T', 128)
        x = self.ssm(x)
        x = x.permute(0, 2, 1)                  # (B, 128, T')
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# 8. DAS-GNN (graph neural network)
# ══════════════════════════════════════════════════════════════

class GraphConvLayer(nn.Module):
    """Simple mean-aggregation graph convolution."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
        self.bn  = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x, adj):
        # x: (B, N, in_ch), adj: (N, N) normalized adjacency
        x = torch.bmm(adj.unsqueeze(0).expand(x.size(0),-1,-1), x)
        x = self.lin(x)
        B, N, C = x.shape
        x = self.bn(x.view(-1, C)).view(B, N, C)
        return self.act(x)


class DAS_GNN(nn.Module):
    """
    DAS-GNN: models 361 fiber channels as spatial graph (k=5 NN edges).
    FORGE F1 = 0.677 (single run, exploratory).
    """
    def __init__(self, n_ch=361, k=5, t_pool=64):
        super().__init__()
        self.t_pool  = t_pool
        self.in_proj = nn.Linear(t_pool, 128)
        self.gcn1    = GraphConvLayer(128, 256)
        self.gcn2    = GraphConvLayer(256, 256)
        self.head    = nn.Sequential(nn.Dropout(0.5), nn.Linear(256, 2))
        self.k       = k
        self._adj    = None   # built lazily on first forward pass

    def _build_adj(self, n_ch, device):
        """k-NN adjacency on 1D channel index (spatial proximity along fiber)."""
        idx = torch.arange(n_ch, device=device).float().unsqueeze(1)
        dist = (idx - idx.T).abs()
        _, nn_idx = dist.topk(self.k+1, dim=1, largest=False)
        adj = torch.zeros(n_ch, n_ch, device=device)
        adj.scatter_(1, nn_idx[:,1:], 1.0)
        deg = adj.sum(1, keepdim=True).clamp(min=1)
        return adj / deg                        # row-normalize

    def forward(self, x):
        B, C_in, n_ch, T = x.shape             # C_in=1
        x = x.squeeze(1)                        # (B, 361, T)
        # Pool time dimension
        x = F.adaptive_avg_pool1d(x, self.t_pool)  # (B, 361, t_pool)
        x = self.in_proj(x)                    # (B, 361, 128)
        if self._adj is None or self._adj.device != x.device:
            self._adj = self._build_adj(n_ch, x.device)
        x = self.gcn1(x, self._adj)
        x = self.gcn2(x, self._adj)
        x = x.mean(1)                           # (B, 256) global mean
        return self.head(x)


# ══════════════════════════════════════════════════════════════
# 9. DASViT (Vision Transformer)
# ══════════════════════════════════════════════════════════════

class DASViT(nn.Module):
    """
    DASViT: ViT with 19×50 patch embedding on 361×2400 DAS image.
    FORGE F1 = 0.008 (near-total collapse).
    """
    def __init__(self, patch_h=19, patch_w=50, embed=192, depth=4, heads=4):
        super().__init__()
        # 361/19=19, 2400/50=48 → ~19×48=912 patches
        self.patch_emb = nn.Conv2d(1, embed, (patch_h, patch_w),
                                   stride=(patch_h, patch_w))
        n_patches = (361//patch_h) * (2400//patch_w)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed) * 0.02)
        self.pos_emb   = nn.Parameter(torch.randn(1, n_patches+1, embed) * 0.02)
        layer = nn.TransformerEncoderLayer(embed, heads, embed*4,
                                           dropout=0.1, batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed)
        self.head = nn.Sequential(nn.Dropout(0.5), nn.Linear(embed, 2))

    def forward(self, x):
        x = self.patch_emb(x)                  # (B, embed, nH, nW)
        B, C, nH, nW = x.shape
        x = x.flatten(2).transpose(1,2)        # (B, N, embed)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1) + self.pos_emb
        x   = self.norm(self.transformer(x))
        return self.head(x[:, 0])              # CLS token


# ══════════════════════════════════════════════════════════════
# 10. CONFORMER (CNN-Transformer hybrid)
# ══════════════════════════════════════════════════════════════

class ConformerBlock(nn.Module):
    """Feed-forward + attention + Conv + feed-forward sandwich."""
    def __init__(self, ch, n_heads=4, ff_mult=4, kernel=31):
        super().__init__()
        self.ff1   = nn.Sequential(nn.LayerNorm(ch),
                                    nn.Linear(ch, ch*ff_mult), nn.SiLU(),
                                    nn.Dropout(0.1), nn.Linear(ch*ff_mult, ch))
        self.attn  = nn.MultiheadAttention(ch, n_heads, dropout=0.1,
                                            batch_first=True)
        self.attn_norm = nn.LayerNorm(ch)
        pad = (kernel-1)//2
        self.conv  = nn.Sequential(
            nn.LayerNorm(ch),
            nn.Conv1d(ch, ch*2, 1), nn.GLU(dim=1),
            nn.Conv1d(ch, ch, kernel, padding=pad, groups=ch, bias=False),
            nn.BatchNorm1d(ch), nn.SiLU(),
            nn.Conv1d(ch, ch, 1), nn.Dropout(0.1))
        self.ff2   = nn.Sequential(nn.LayerNorm(ch),
                                    nn.Linear(ch, ch*ff_mult), nn.SiLU(),
                                    nn.Dropout(0.1), nn.Linear(ch*ff_mult, ch))

    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        n = self.attn_norm(x)
        a, _ = self.attn(n, n, n)
        x = x + a
        c = self.conv[0](x)
        c = c.permute(0,2,1)
        for layer in self.conv[1:]:
            c = layer(c)
        x = x + c.permute(0,2,1)
        return x + 0.5 * self.ff2(x)


class Conformer(nn.Module):
    """
    Conformer: CNN spatial stem + Conformer temporal encoder.
    Achieves highest mean FORGE F1 = 0.775±0.092 but highest variance.
    """
    def __init__(self, embed=128, depth=4):
        super().__init__()
        self.stem = _shared_stem()
        self.proj = nn.Sequential(
            SEResBlock(32), _downsample(32, 64), SEResBlock(64))
        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.lin  = nn.Linear(64, embed)
        self.layers = nn.Sequential(*[ConformerBlock(embed) for _ in range(depth)])
        self.norm   = nn.LayerNorm(embed)
        self.head   = nn.Sequential(nn.Dropout(0.5), nn.Linear(embed, 2))

    def forward(self, x):
        x = self.proj(self.stem(x))             # (B, 64, H', T')
        x = self.pool(x).squeeze(2)             # (B, 64, T')
        x = self.lin(x.permute(0,2,1))          # (B, T', embed)
        x = self.norm(self.layers(x))
        return self.head(x.mean(1))


# ══════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════

MODELS = {
    "SE-ResNet":   SEResNet,
    "ResNet":      ResNet,
    "ConvNeXt":    ConvNeXt,
    "CNN-GRU":     CNN_GRU,
    "CNN-BiLSTM":  CNN_BiLSTM,
    "GRU":         GRU_DAS,
    "Mamba-DAS":   MambaDAS,
    "DAS-GNN":     DAS_GNN,
    "ViT":         DASViT,
    "Conformer":   Conformer,
}


def get_model(name: str) -> nn.Module:
    """
    Instantiate a model by name.
    Args:
        name: one of 'SE-ResNet', 'ResNet', 'ConvNeXt', 'CNN-GRU',
              'CNN-BiLSTM', 'GRU', 'Mamba-DAS', 'DAS-GNN', 'ViT', 'Conformer'
    Returns:
        Instantiated nn.Module (not yet on device)
    """
    if name not in MODELS:
        raise ValueError(f"Unknown model '{name}'. Choose from: {list(MODELS.keys())}")
    return MODELS[name]()


if __name__ == "__main__":
    # Quick sanity check — verify all models run on a dummy DAS batch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(2, 1, 361, 2400).to(device)   # batch=2

    print(f"Input: {x.shape}  Device: {device}")
    print()
    for name, cls in MODELS.items():
        try:
            model = cls().to(device)
            model.eval()
            with torch.no_grad():
                out = model(x)
            n_params = sum(p.numel() for p in model.parameters()) / 1e6
            print(f"  ✓ {name:<14} output={tuple(out.shape)}  params={n_params:.1f}M")
            del model
        except Exception as e:
            print(f"  ✗ {name:<14} ERROR: {e}")
