import torch
import torch.nn as nn
from methods.net_tools import *

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class DoubleCritic(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=256,
            activation='mish'
    ):
        super(DoubleCritic, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU

        self.q1_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.q2_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, hidden_dim),
            _act(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, obs):
        obs = obs.reshape(obs.size(0), -1)
        return self.q1_net(obs), self.q2_net(obs)

    def q_min(self, obs):
        obs = obs.reshape(obs.size(0), -1)
        return torch.min(*self.forward(obs))
    

class ActionMLP(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            hidden_dim=128,
            t_dim=16,
            model_output='logistic_pars',
            use_res=False,
            activation='mish'
    ):
        super(ActionMLP, self).__init__()
        _act = nn.Mish if activation == 'mish' else nn.ReLU
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        self.action_dim = action_dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(t_dim),
            init_(nn.Linear(t_dim, t_dim * 2)),
            _act(),
            init_(nn.Linear(t_dim * 2, t_dim))
        )
        self.model_output = model_output
        if self.model_output == 'logistic_pars':
            output_dim = 2
        else:
            output_dim = self.action_dim
        self.mid_layer = nn.Sequential(
            init_(nn.Linear(state_dim + action_dim + t_dim, hidden_dim)),
            # init_(nn.Linear(action_dim + t_dim, hidden_dim)),
            _act(),
            init_(nn.Linear(hidden_dim, hidden_dim)),
            _act(),
            # init_(nn.Linear(hidden_dim, 2 * action_dim))
            init_(nn.Linear(hidden_dim, output_dim))
        )

        self.use_res = use_res
        # self.final_layer = nn.Tanh()
        self.training = True


    def max_gradient(self):
        max_gradient = 0
        for n, p in self.named_parameters():
            max_gradient = max(max_gradient, p.grad.data.mean().item())
        return max_gradient

    def print_grad(self,):
        print(' in print grad...')
        for n, p in self.named_parameters():
            if p.requires_grad:
                print(n, p.grad.data.mean())

    def forward(self, input, time, state):
        norm_input = input / self.action_dim
        input = input.squeeze(1)
        t = self.time_mlp(time)
        x_onehot = F.one_hot(input, num_classes=self.action_dim).to(input.device)
        # state = state.reshape(state.size(0), -1)
        x = torch.cat([x_onehot, t, state], dim=1)
        # x = torch.cat([x_onehot, t], dim=1)
        x = self.mid_layer(x)
        if self.model_output == 'logistic_pars':
            loc, log_scale = torch.chunk(x, 2, dim=1)
            if self.use_res:
                # print('111:', loc.shape, norm_input.shape)
                out = torch.tanh(loc + norm_input), log_scale
                # print(out[1].shape)
            else:
                out = torch.tanh(loc), log_scale
        else:
            if self.use_res:
                out = x + x_onehot
            else:
                out = x
            out = out.unsqueeze(1)
        return out
        # return self.final_layer(x)
