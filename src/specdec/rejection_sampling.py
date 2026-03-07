"""Speculative decoding rejection sampling (Leviathan et al. 2023 / Chen et al. 2023).

Pure tensor math, model-agnostic: takes draft/target probability distributions and
returns which draft tokens are accepted plus a resampled/bonus token, with the
guarantee that the marginal distribution of the output token equals the target
model's own distribution (this is what makes speculative decoding lossless).
"""

from dataclasses import dataclass

import torch


@dataclass
class RejectionResult:
    accepted_tokens: torch.Tensor   # (num_accepted,) draft tokens that were accepted
    num_accepted: int               # how many of the K draft tokens were accepted
    bonus_token: torch.Tensor       # (1,) extra token sampled from target dist, always present
    all_accepted: bool              # True if every draft token was accepted


def sample_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Sample one token id from a single position's logits."""
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def speculative_rejection_sample(
    draft_tokens: torch.Tensor,       # (K,) token ids proposed by the draft model
    draft_probs: torch.Tensor,        # (K, vocab) draft model's per-position probabilities
    target_probs: torch.Tensor,       # (K+1, vocab) target model's per-position probabilities
                                       # position K is the "bonus" position after all K drafts
) -> RejectionResult:
    """Standard speculative decoding accept/reject loop.

    For each drafted token x_i, accept with probability min(1, p_i(x_i) / q_i(x_i)).
    On the first rejection at position j, resample from the residual distribution
    normalize(max(0, p_j - q_j)) and discard the remaining draft tokens. If all K are
    accepted, sample one additional "bonus" token from the target's distribution at
    position K — this is the standard free extra token from the algorithm.
    """
    K = draft_tokens.shape[0]
    device = draft_tokens.device

    for i in range(K):
        token_id = draft_tokens[i].item()
        p_i = target_probs[i, token_id]
        q_i = draft_probs[i, token_id]
        accept_prob = torch.clamp(p_i / q_i, max=1.0)
        if torch.rand((), device=device) < accept_prob:
            continue

        residual = torch.clamp(target_probs[i] - draft_probs[i], min=0.0)
        residual_sum = residual.sum()
        if residual_sum <= 0:
            resampled = target_probs[i].argmax(dim=-1, keepdim=True)
        else:
            residual = residual / residual_sum
            resampled = torch.multinomial(residual, num_samples=1)
        return RejectionResult(
            accepted_tokens=draft_tokens[:i],
            num_accepted=i,
            bonus_token=resampled,
            all_accepted=False,
        )

    bonus_token = torch.multinomial(target_probs[K], num_samples=1)
    return RejectionResult(
        accepted_tokens=draft_tokens,
        num_accepted=K,
        bonus_token=bonus_token,
        all_accepted=True,
    )
