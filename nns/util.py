from collections import namedtuple
import torch
import numpy as np

StateSpace = namedtuple('StateSpace', ['A', 'B', 'C', 'Bw', 'W', 'V'])


def seiler_state_space():
    A = np.array([[1.1052, 0.1105], [0, 1.1052]])
    B = np.array([[0.0053, 0.1052]]).T
    Bw = np.array([[0.1105, 0.1052]]).T
    C = np.array([[1.0, 0.0]])
    W = np.full((1, 1), 1e3, dtype=np.float64)
    V = np.ones((1, 1), dtype=np.float64)

    ss = StateSpace(A, B, C, Bw, W, V)

    Q = np.full((2, 2), 1e3, dtype=np.float64)
    R = np.ones((1, 1), dtype=np.float64)

    return ss, Q, R


def simulate(controller, env, N, noise=0):
    rewards = []

    controller.reset()
    obs = env.reset()

    for _ in range(N):
        u = controller(obs)
        # u, obs, and total loss blow up if the the below is uncommented
        # u += (torch.rand(u.shape) * 2 - 1) * noise
        obs, reward, _, _ = env.step(u)
        rewards.append(reward)

    total_loss = - torch.stack(rewards)
    # print(total_loss)
    return total_loss #.mean()


def train_model(controller, env, optimizer, n_epochs, n_steps, noise=0, v=False):
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        steps = n_steps
        loss_val = np.nan

        loss = simulate(controller, env, steps, noise=noise).mean(dim=1).mean()
        loss_val = loss.detach().cpu().numpy().item()
        if v:
            print(loss_val)

        losses.append(loss_val)

        loss.backward()
        optimizer.step()
    return losses


def eval_model(controller, env, n_epochs, n_steps):
    losses = []
    for epoch in range(n_epochs):
        loss = simulate(controller, env, n_steps)
        losses.append(loss.detach().cpu().numpy().item())
    return losses
