# SPDX-License-Identifier: Apache-2.0
"""CPU tensor checks against hand-encoded NVFP4 records; requires PyTorch."""

import pytest
import torch
from attention_ts_scaled_mla_fixtures import decode_cache, quantize_nvfp4


@pytest.mark.parametrize("dim", [512, 576])
def test_known_token_record(dim):
    """Check nibble order and both public scale layouts independently of the decoder."""
    scale = torch.arange(1, 33).to(torch.float8_e4m3fn).float()
    magnitudes = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.5, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    values = (scale[:, None] * magnitudes).flatten()
    if dim == 576:
        values = torch.cat((values, torch.arange(64).to(torch.float8_e4m3fn).float()))
    packed = quantize_nvfp4(values[None])
    payload = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x99, 0xBA, 0xDC, 0xFE], dtype=torch.uint8
    ).repeat(32)
    assert torch.equal(packed[0, :256], payload)
    scale_bytes = scale.to(torch.float8_e4m3fn).view(torch.uint8)
    if dim == 576:
        # V32 is four groups of eight scales: s0,s4,...,s28; s1,s5,... .
        canonical_order = [group + 4 * i for group in range(4) for i in range(8)]
        assert torch.equal(packed[0, 320:], scale_bytes[canonical_order])
        assert torch.equal(
            packed[0, 256:320], values[512:].to(torch.float8_e4m3fn).view(torch.uint8)
        )
    else:
        assert torch.equal(packed[0, 256:], scale_bytes)
    assert torch.equal(decode_cache(packed, fp8_mma=False)[0], values)
    assert torch.equal(
        decode_cache(packed, fp8_mma=True)[0], values.to(torch.float8_e4m3fn).float()
    )


def test_midpoints_round_to_even_codes():
    """Max magnitude six fixes scale=1; expected bytes pin ties and signed underflow."""
    positive = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    values = torch.cat((positive, -positive)).repeat(32)[None]
    expected = torch.tensor(
        [0x20, 0x42, 0x64, 0x76, 0xA8, 0xCA, 0xEC, 0xFE], dtype=torch.uint8
    ).repeat(32)
    assert torch.equal(quantize_nvfp4(values)[0, :256], expected)
