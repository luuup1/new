"""Behavior Cloning (BC) warm-start for the 5G-TSN PPO agent.

Route C of the project plan: pre-train the actor on expert (min-load)
demonstrations collected on the *random-order* env, so the agent starts
from a near-greedy min-load policy instead of a cold random one, then let
PPO fine-tune on the RL reward.

Why random-order demonstrations?
--------------------------------
The DRL MDP uses order_mode="random" (evaluated "on the baseline", see
EXPERIMENT_PROTOCOL.md). If we cloned the EDF-order `greedy` trajectory
as the teacher, the demo distribution would NOT match the DRL MDP and the
warm-start would be off-distribution. So the teacher here is the *min-load
placement rule* applied step-by-step on the same random-order env the agent
faces. The label for each step is simply: among feasible (masked) actions,
pick the one with the smallest post-placement cell load.

Usage (driven from train_ppo.py --bc-pretrain):
    demo = generate_demonstrations(env, num_episodes)
    bc_train(agent, demo, epochs=..., lr=...)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from .ppo import PPOAgent


def generate_demonstrations(env, num_episodes: int) -> dict:
    """Roll out the min-load teacher on `env` for `num_episodes` episodes.

    `env` should be the SAME env the agent trains on (random-order,
    full obs). With multi_scenario=True the teacher sees diverse traffic
    patterns, giving more generalizable demos.

    Returns a dict with stacked arrays:
        obs:    (N, obs_dim)      float32
        mask:   (N, n_actions)    bool
        action: (N,)              int64   (min-load action index)
    """
    obs_list, mask_list, act_list = [], [], []
    for _ in range(num_episodes):
        state, _ = env.reset()
        mask = env.action_masks()
        while True:
            a = env.expert_min_load_action()
            obs_list.append(state.astype(np.float32))
            mask_list.append(mask.copy())
            act_list.append(int(a))
            state, _, done, _, _ = env.step(a)
            if done:
                break
            mask = env.action_masks()
    return {
        "obs": np.stack(obs_list),
        "mask": np.stack(mask_list),
        "action": np.array(act_list, dtype=np.int64),
    }


def bc_train(
    agent: PPOAgent,
    demo: dict,
    *,
    epochs: int = 30,
    lr: float = 1e-3,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict:
    """Supervised pre-training of the actor on (obs, mask, action) demos.

    Uses a masked categorical cross-entropy: invalid actions are masked to
    -inf before softmax, so the loss only grades the agent on feasible
    choices (matching how the policy is used during RL). The critic is left
    untouched (PPO will train it from scratch during fine-tuning).

    Returns a small stats dict.
    """
    device = agent.device
    obs = torch.from_numpy(demo["obs"]).float().to(device)
    masks = torch.from_numpy(demo["mask"]).bool().to(device)
    actions = torch.from_numpy(demo["action"]).long().to(device)
    N = len(actions)

    opt = Adam(agent.actor.parameters(), lr=lr)
    idx = np.arange(N)
    last_acc = 0.0
    for ep in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, N, batch_size):
            mb = idx[s:s + batch_size]
            if len(mb) == 0:
                continue
            logits = agent.actor(obs[mb])
            masked = PPOAgent._mask_logits(logits, masks[mb])
            logp = F.log_softmax(masked, dim=-1)
            loss = -logp.gather(-1, actions[mb].unsqueeze(-1)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        # greedy match rate on the demo set (sanity check)
        with torch.no_grad():
            logits = agent.actor(obs)
            masked = PPOAgent._mask_logits(logits, masks)
            pred = masked.argmax(dim=-1)
            last_acc = float((pred == actions).float().mean().item())

        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs - 1):
            print(f"  [BC] epoch {ep+1:>3d}/{epochs}  "
                  f"loss~  ce  greedy_match={last_acc:.3f}")

    return {"bc_epochs": epochs, "bc_samples": N, "greedy_match": last_acc}
