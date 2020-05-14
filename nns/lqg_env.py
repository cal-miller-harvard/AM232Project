import numpy as np
import torch

from torch.distributions.multivariate_normal import MultivariateNormal

import gym
from gym import spaces
from util import StateSpace


def Rn(shape):
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)


class LQG(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, ss, Q, R, batch_size=1, observe_x=False):
        super(LQG, self).__init__()

        self.ss = StateSpace(*[torch.tensor(x) for x in ss])
        A, B, C, Bw, W, V = self.ss

        self.wt = MultivariateNormal(torch.tensor([0.]), W)
        self.vt = MultivariateNormal(torch.tensor([0.]), V)

        self.Q = torch.tensor(Q)
        self.R = torch.tensor(R)

        self.x_shape = (A.shape[0], batch_size)
        self.y_shape = (C.shape[0], batch_size)

        self.action_space = Rn((batch_size, B.shape[1]))
        self.observation_space = Rn(self.x_shape if observe_x else self.y_shape)

        self.observe_x = observe_x
        self.x = None

    def LQR_cost(self, x, u):
        return (x.t() @ self.Q @ x + u.t() @ self.R @ u).diag()

    def step(self, u):
        # Execute one time step within the environment
        A, B, C, Bw = self.ss[:4]
        self.x = torch.mm(A, self.x) + torch.mm(B, u) + torch.mm(Bw, self.wt.sample([u.shape[1]]).t())
        y = torch.mm(C, self.x) + self.vt.sample([u.shape[1]]).t()

        obs = self.x if self.observe_x else y.view(self.y_shape)
        reward = - self.LQR_cost(self.x, u)

        return obs, reward, False, {}

    def reset(self):
        self.x = torch.zeros(self.x_shape)
        return self.x if self.observe_x else torch.zeros(self.y_shape)

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...
