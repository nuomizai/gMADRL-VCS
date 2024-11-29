import torch.distributions
# from torchrl.modules import OneHotCategorical as _OneHotCategorical

from util import *
from methods.net_tools import *


class Catigorical_Generator(nn.Module):
    def __init__(self, input_size, choice_num):
        super().__init__()
        self.input_size = input_size
        self.output_size = choice_num

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)
        self.linear = nn.Sequential(
            init_(nn.Linear(self.input_size, self.output_size)),
            nn.Softmax(dim=-1)
        )

    def forward(self, x, mask=None):
        x = self.linear(x)
        if mask is not None:
            x = torch.mul(x, mask)
        dist = torch.distributions.Categorical(probs=x)
        return dist



class DiagGaussian(nn.Module):
    def __init__(self, input_size, output_size):
        super(DiagGaussian, self).__init__()
        init_ = lambda m: init(m,
                               init_normc_,
                               lambda x: nn.init.constant_(x, 0))
        self.fc_mean = nn.Sequential(
            init_(nn.Linear(input_size, output_size)),
        )
        self.logstd = AddBias(torch.zeros(output_size, dtype=torch.float32))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_mean = torch.tanh(action_mean)
        zeros = torch.zeros(action_mean.size(), device=x.device)
        action_logstd = self.logstd(zeros)
        action_logstd = torch.tanh(action_logstd)
        return torch.distributions.Normal(action_mean, action_logstd.exp())
