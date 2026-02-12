import pytest
import torch

from tests.utils import bypass_not_implemented

from mojo_opset import MojoRoPE
from mojo_opset import MojoGridRoPE
from mojo_opset.utils.platform import get_platform


@pytest.mark.parametrize("bs", [8, 32, 55])
@pytest.mark.parametrize("seqlen", [128, 512, 3345, 4985, 6688])
@pytest.mark.parametrize(
    "q_heads, k_heads",
    [
        (32, 32),
        (32, 8),
        (16, 1),
    ],
)
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_pos_emb(bs, seqlen, q_heads, k_heads, head_dim, dtype):
    device = get_platform()
    # [B, S, N, D]
    q = torch.randn(bs, seqlen, q_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(bs, seqlen, k_heads, head_dim, device=device, dtype=dtype)

    rope = MojoRoPE()
    rope_ref = MojoRoPE._registry.get("torch")()

    # Mock real inference memory layout: [B, N, S, D]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=q.device, dtype=torch.float32) / head_dim))
    t = torch.arange(seqlen, device=q.device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)

    # [1, 1, S, D]
    cos = emb.cos()[None, None, :, :]
    sin = emb.sin()[None, None, :, :]

    rope.forward_diff_with(rope_ref, q, k, cos, sin)


@pytest.mark.parametrize(
    "bs, grid, heads, head_dim, pad",
    [
        (4, (2, 4, 8), 8, 64, 10),
        (2, (1, 8, 8), 16, 128, 5),
        (3, (4, 4, 4), 4, 64, 3),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@bypass_not_implemented
def test_grid_pos_emb(bs, grid, heads, head_dim, pad, dtype):
    device = get_platform()
    f, h, w = grid
    seq_len = f * h * w
    L = seq_len + pad

    x = torch.randn(bs, L, heads, head_dim, device=device, dtype=dtype)

    grid_sizes = torch.tensor([grid] * bs, device=device, dtype=torch.int64)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
    freqs_scalar = torch.einsum("i,j->ij", t, inv_freq)  # [seq_len, head_dim/2]
    cos = freqs_scalar.cos()[:, None, :]  # [seq_len, 1, head_dim/2]
    sin = freqs_scalar.sin()[:, None, :]  # [seq_len, 1, head_dim/2]
    freqs = torch.complex(cos, sin)  # complex64
    freqs_list = [freqs for _ in range(bs)]

    rope = MojoGridRoPE()
    rope_ref = MojoGridRoPE._registry.get("torch")()

    rope.forward_diff_with(rope_ref, x, grid_sizes, freqs_list)
