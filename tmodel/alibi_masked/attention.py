import math

import torch

from tmodel.maskedVanilla.masks import bucket_dim, build_head_mask_bias

MAX_RANGE = 10

def get_slopes(n):
    """The first ``n // 2`` ALiBi slopes for ``n`` heads, repeated for the second half.

    So the first and second half of the heads (the causal and the future-only ones)
    get the same set of slopes. With an odd ``n`` the middle head keeps its own slope.
    """
    def get_slopes_power_of_2(n):
        ratio = 2**(-2**-(math.log2(n)-3))
        start = ratio * 0.25    # scales every slope by 0.25; the ratio between heads is unchanged
        return [start*ratio**i for i in range(n)]

    def all_slopes(n):
        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        closest_power_of_2 = 2**math.floor(math.log2(n))
        return get_slopes_power_of_2(closest_power_of_2) + all_slopes(2*closest_power_of_2)[0::2][:n-closest_power_of_2]

    slopes = all_slopes(n)
    half = slopes[:n // 2]
    return half + slopes[n // 2:n // 2 + n % 2] + half


def symmetric_alibi_bias(num_heads: int, seq_len: int, device=None, dtype=None) -> torch.Tensor:
    """Symmetric ALiBi bias, ``[num_heads, seq_len, seq_len]``: ``-slope_h * |i - j|``.

    The "symmetric" encoder option of
    https://github.com/ofirpress/attention_with_linear_biases/issues/5, added to the
    attention scores before the softmax. Built at the length asked for rather than
    stored at a maximum length and sliced: the values are the same, and a stored
    ``[heads, maxpos, maxpos]`` plane would be gigabytes at the eval lengths.
    """
    context_position = torch.arange(seq_len, device=device)[:, None]
    memory_position = torch.arange(seq_len, device=device)[None, :]
    relative_position = memory_position - context_position 
    relative_position = torch.abs(relative_position).unsqueeze(0).expand(num_heads, -1, -1)
    relative_position = relative_position.clamp(max=MAX_RANGE)

    slopes = torch.tensor(get_slopes(num_heads), device=device, dtype=dtype) * -1
    return slopes.unsqueeze(1).unsqueeze(1) * relative_position


def causal_future_block(num_heads: int, seq_len: int, device=None, dtype=None,
                        key_padding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Additive 4C4F head mask
    """
    spec = "C" * (num_heads // 2) + "F" * (num_heads - num_heads // 2)
    bias = build_head_mask_bias(spec, num_heads, bucket_dim(seq_len), device, dtype)
    bias = bias[..., :seq_len, :seq_len]
    if key_padding_mask is not None:
        rows = key_padding_mask.to(torch.bool).view(-1, 1, seq_len, 1)
        bias = bias.masked_fill(rows, 0.0)    # not in place: the cached plane is shared
    return bias
