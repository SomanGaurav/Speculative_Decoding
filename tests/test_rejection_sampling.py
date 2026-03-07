import torch

from specdec.rejection_sampling import speculative_rejection_sample


def _uniform_probs(k_plus_1: int, vocab: int) -> torch.Tensor:
    return torch.full((k_plus_1, vocab), 1.0 / vocab)


def test_all_accepted_when_draft_equals_target():
    torch.manual_seed(0)
    K, vocab = 4, 10
    draft_tokens = torch.tensor([1, 2, 3, 4])
    probs = _uniform_probs(K + 1, vocab)
    result = speculative_rejection_sample(draft_tokens, probs[:K], probs)
    assert result.all_accepted
    assert result.num_accepted == K
    assert torch.equal(result.accepted_tokens, draft_tokens)


def test_always_rejects_when_target_assigns_zero_prob():
    torch.manual_seed(0)
    K, vocab = 3, 5
    draft_tokens = torch.tensor([0, 1, 2])
    draft_probs = _uniform_probs(K, vocab)
    target_probs = torch.zeros((K + 1, vocab))
    # target puts all mass away from the drafted tokens at every position
    for i in range(K + 1):
        target_probs[i, (i + 3) % vocab] = 1.0
    result = speculative_rejection_sample(draft_tokens, draft_probs, target_probs)
    assert not result.all_accepted
    assert result.num_accepted == 0
    # resampled token must come from target's support at position 0
    assert result.bonus_token.item() == 3


def test_marginal_distribution_matches_target_statistically():
    """Core correctness guarantee of speculative decoding: over many trials, the
    distribution of the final output token equals the target model's own distribution,
    regardless of what the draft model proposed."""
    torch.manual_seed(42)
    vocab = 4
    K = 1
    target_dist = torch.tensor([0.1, 0.6, 0.2, 0.1])
    draft_dist = torch.tensor([0.4, 0.1, 0.4, 0.1])  # deliberately different from target

    trials = 20000
    outcomes = torch.zeros(vocab)
    for _ in range(trials):
        draft_token = torch.multinomial(draft_dist, num_samples=1)
        draft_probs = draft_dist.unsqueeze(0)
        target_probs = target_dist.unsqueeze(0).repeat(K + 1, 1)
        result = speculative_rejection_sample(draft_token, draft_probs, target_probs)
        if result.all_accepted:
            outcomes[draft_token.item()] += 1
        else:
            outcomes[result.bonus_token.item()] += 1

    empirical = outcomes / trials
    assert torch.allclose(empirical, target_dist, atol=0.02)


def test_partial_acceptance_stops_at_first_rejection():
    torch.manual_seed(1)
    K, vocab = 3, 6
    draft_tokens = torch.tensor([0, 1, 2])
    draft_probs = _uniform_probs(K, vocab)
    target_probs = _uniform_probs(K + 1, vocab).clone()
    # force rejection at position 1 by zeroing target prob for the drafted token there
    target_probs[1, 1] = 0.0
    target_probs[1] = target_probs[1] / target_probs[1].sum()

    accepted_at_least_once = False
    rejected_before_end = False
    for _ in range(200):
        result = speculative_rejection_sample(draft_tokens, draft_probs, target_probs)
        if result.num_accepted >= 1:
            accepted_at_least_once = True
        if result.num_accepted < K:
            rejected_before_end = True
    assert accepted_at_least_once
    assert rejected_before_end
