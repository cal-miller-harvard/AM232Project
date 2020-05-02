from collections import namedtuple
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


def simulate(controller, env, N):
    total_reward = 0.

    controller.reset()
    obs = env.reset()

    for _ in range(N):
        u = controller(obs)
        obs, reward, _, _ = env.step(u)
        total_reward += reward

    return - total_reward.mean() / N


def train_model(controller, env, optimizer, n_epochs, n_steps):
    losses = []
    for epoch in range(n_epochs):
        optimizer.zero_grad()

        loss = simulate(controller, env, n_steps)
        losses.append(loss.detach().cpu().numpy().item())

        loss.backward()
        optimizer.step()
    return losses


def eval_model(controller, env, n_epochs, n_steps):
    losses = []
    for epoch in range(n_epochs):
        loss = simulate(controller, env, n_steps)
        losses.append(loss.detach().cpu().numpy().item())
    return losses
