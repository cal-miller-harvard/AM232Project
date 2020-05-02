import torch
import torch.nn as nn
import control


def get_KL(ss, Q, R):
    A, B, C, Bw, W, V = ss
    K = control.dare(A, B, Q, R)[2]
    L = control.dare(A.T, C.T, Bw @ Bw.T * W, V)[2]
    return torch.tensor(K), torch.tensor(L)


# the optimal controller determined from the DARE
class OptimalController:
    def __init__(self, ss, Q, R):
        K, L = get_KL(ss, Q, R)
        self.K = K

    def __call__(self, x):
        return -self.K @ x

    def reset(self):
        pass


class RNNController(nn.Module):
    def __init__(self, dim_y, dim_u, dim_h=10, layers=2, param_limits=(-0.2, 0.2)):
        super(RNNController, self).__init__()
        self.rnn = nn.RNN(dim_y, dim_h, num_layers=layers, nonlinearity='relu').double()
        self.out = nn.Linear(dim_h, dim_u).double()
        self.param_limits = param_limits
        self.hidden = None

    def forward(self, y):
        y = y.t().unsqueeze(0)
        y1, self.hidden = self.rnn(y, self.hidden)
        return self.out(y1[0]).t()

    def init_weights(self):
        a, b = self.param_limits
        for name, param in self.named_parameters():
              torch.nn.init.uniform_(param, a=a, b=b)

    def reset(self):
        # Called at the beginning of a simulation
        self.hidden = None

