import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

import gym
from gym import spaces
from util import StateSpace


def Rn(shape):
    return spaces.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float64)


# batch mv multiplication
def bmv(A, B):
    return torch.einsum('bij,bj->bi', A, B)

# batch vector squaring
def b2(x):
    return torch.einsum('bi,bi->b', x, x)


class Nonlinear(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, A, B, Z, C1, C2, W=None, batch_size=1):
        super(Nonlinear, self).__init__()

        self.wt = MultivariateNormal(torch.tensor([0.]), W) if W is not None else None
        self.b = batch_size
        self.get_ts = lambda x: [v(x) for v in [A, B, Z, C1, C2]]
        self.x_shape = (batch_size, B(0).shape[-2])
        self.action_space = Rn((batch_size, B(0).shape[-1]))
        self.observation_space = Rn(self.x_shape)
        self.x = None

    def step(self, u):
        # Execute one time step within the environment
        A, B, Z, C1, C2 = self.get_ts(self.x)
        noise = self.wt.sample([u.shape[0]]) if self.wt is not None else 0

        self.x = bmv(A, Z) + bmv(B, u) + noise
        z1 = bmv(C1, Z)
        z2 = bmv(C2, Z) + u
        reward = - (b2(z1) + b2(z2))
        return self.x, reward, False, {}

    def reset(self):
        self.x = torch.rand(self.x_shape)
        return self.x

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...
