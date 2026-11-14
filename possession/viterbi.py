"""Segment per-frame class probabilities into possessions.

States H, A, D. Each state is expanded into a chain of `min_len` sub-states so a label, once
entered, has to be held for at least that many frames (a semi-Markov minimum duration, done with
an ordinary Viterbi). Leaving a state costs a penalty: one for a team change with the ball alive
(H<->A), one for going between alive and dead.

Emissions are posteriors divided by the training prior to the power `prior_power`, i.e. scaled
likelihoods, so a class that was common in training is not favoured for that reason alone.
"""

import numpy as np

H, A, D = 0, 1, 2


def decode(proba, team_cost=8.0, dead_cost=8.0, min_len=(25, 25, 25), prior=None,
           prior_power=0.0, eps=1e-6):
    """Most likely label path, int8 (T,). proba is (T, 3) for H, A, D."""
    T = len(proba)
    logp = np.log(np.clip(proba, eps, 1.0))
    if prior is not None and prior_power:
        logp = logp - prior_power * np.log(np.asarray(prior))

    # Switch cost from state i to state j.
    cost = np.array([[0, team_cost, dead_cost],
                     [team_cost, 0, dead_cost],
                     [dead_cost, dead_cost, 0]], dtype=np.float64)

    k = [max(1, int(m)) for m in min_len]
    # Flatten the chains: state s owns sub-states offs[s] .. offs[s]+k[s]-1.
    offs = np.concatenate(([0], np.cumsum(k)))[:-1]
    S = int(np.sum(k))
    owner = np.concatenate([[s] * k[s] for s in range(3)])
    first = offs
    last = offs + np.array(k) - 1

    score = np.full(S, -np.inf)
    score[first] = logp[0]
    back = np.zeros((T, S), dtype=np.int32)

    prev_of = np.arange(S) - 1  # in-chain predecessor
    in_chain = np.ones(S, dtype=bool)
    in_chain[first] = False

    for t in range(1, T):
        new = np.full(S, -np.inf)
        bp = np.zeros(S, dtype=np.int32)
        # advance along a chain
        new[in_chain] = score[prev_of[in_chain]]
        bp[in_chain] = prev_of[in_chain]
        # stay in the last sub-state
        for s in range(3):
            j = last[s]
            if k[s] == 1 or score[j] > new[j]:
                if score[j] > new[j] or k[s] == 1:
                    new[j] = score[j]
                    bp[j] = j
        # enter a state's first sub-state from another state's last sub-state
        for s in range(3):
            cand = score[last] - cost[:, s]
            cand[s] = -np.inf
            best = int(np.argmax(cand))
            val = cand[best]
            j = first[s]
            if k[s] == 1:
                # first == last: compare with staying
                if val > new[j]:
                    new[j] = val
                    bp[j] = last[best]
            elif val > new[j]:
                new[j] = val
                bp[j] = last[best]
        new += logp[t][owner]
        score = new
        back[t] = bp

    path = np.empty(T, dtype=np.int32)
    path[-1] = int(np.argmax(score))
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return owner[path].astype(np.int8)
