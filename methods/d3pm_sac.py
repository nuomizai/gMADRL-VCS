import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from copy import deepcopy
from typing import Any, Dict, List, Type, Optional, Union
from torch.optim.lr_scheduler import CosineAnnealingLR
from methods.d3pm.D3PM import CategoricalDiffusion
from methods.d3pm.model import ActionMLP, DoubleCritic
import os
from util import soft_update_params, my_logger
import ml_collections

def config_dict(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)

def get_diffusion_betas(spec):
    """Get betas from the hyperparameters."""
    if spec.type == 'linear':
        # Used by Ho et al. for DDPM, https://arxiv.org/abs/2006.11239.
        # To be used with Gaussian diffusion models in continuous and discrete
        # state spaces.
        # To be used with transition_mat_type = 'gaussian'
        return torch.linspace(spec.start, spec.stop, spec.num_timesteps)
    elif spec.type == 'cosine':
        # Schedule proposed by Hoogeboom et al. https://arxiv.org/abs/2102.05379
        # To be used with transition_mat_type = 'uniform'.
        steps = (
                np.arange(spec.num_timesteps + 1, dtype=np.float64) /
                spec.num_timesteps)
        alpha_bar = np.cos((steps + 0.008) / 1.008 * np.pi / 2)
        betas = torch.from_numpy(np.minimum(1 - alpha_bar[1:] / alpha_bar[:-1], 0.999))
        return betas
    elif spec.type == 'jsd':  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        # Proposed by Sohl-Dickstein et al., https://arxiv.org/abs/1503.03585
        # To be used with absorbing state models.
        # ensures that the probability of decaying to the absorbing state
        # increases linearly over time, and is 1 for t = T-1 (the final time).
        # To be used with transition_mat_type = 'absorbing'
        return 1. / torch.linspace(spec.num_timesteps, 1., spec.num_timesteps)
    else:
        raise NotImplementedError(spec.type)


class D3PMSAC(object):
    """
    Implementation of diffusion-based discrete soft actor-critic policy.
    """

    def __init__(
            self,
            ugv_net_cfg,
            actor_cfg,
            actor_lr,
            wd,
            critic_target_tau,
            critic_lr,
            alpha: float = 0.05,
            gamma: float = 0.95,
            reward_normalization: bool = False,
            lr_decay: bool = False,
            num_timesteps: int = 5,
            lr_maxt: int = 1000,
            **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        assert 0.0 <= alpha <= 1.0, "alpha should be in [0, 1]"
        # assert 0.0 <= tau <= 1.0, "tau should be in [0, 1]"
        assert 0.0 <= gamma <= 1.0, "gamma should be in [0, 1]"
        actor = ActionMLP(**actor_cfg)

        diffusion_args = config_dict(
            betas=get_diffusion_betas(config_dict(
                type='linear',
                # start, stop only relevant for linear, power, jsdtrunc schedules.
                start=1e-4,  # 1e-4 gauss, 0.02 uniform
                stop=0.02,  # 0.02, gauss, 1. uniform
                num_timesteps=num_timesteps,
            )),
            model_prediction='x_start',
            # model_output='logistic_pars',
            model_output=actor_cfg['model_output'],
            transition_mat_type='uniform',
            transition_bands=None,
            loss_type='hybrid',
            hybrid_coeff=0.001,
            num_pixel_vals=ugv_net_cfg['action_dim']
        )
        self._diffusion = CategoricalDiffusion(**diffusion_args)

        actor_optim = torch.optim.Adam(
            actor.parameters(),
            lr=actor_lr,
            weight_decay=wd
        )

        critic = DoubleCritic(**ugv_net_cfg)
        critic_optim = torch.optim.Adam(
            critic.parameters(),
            lr=critic_lr,
            weight_decay=wd
        )

        if actor is not None and actor_optim is not None:
            self._actor: torch.nn.Module = actor
            self._target_actor = deepcopy(actor)
            self._target_actor.eval()
            self._actor_optim: torch.optim.Optimizer = actor_optim

        if critic is not None and critic_optim is not None:
            self._critic: torch.nn.Module = critic
            self._target_critic = deepcopy(critic)
            self._target_critic.eval()
            self._critic_optim: torch.optim.Optimizer = critic_optim

        if lr_decay:
            self._actor_lr_scheduler = CosineAnnealingLR(
                self._actor_optim, T_max=lr_maxt, eta_min=0.)
            self._critic_lr_scheduler = CosineAnnealingLR(
                self._critic_optim, T_max=lr_maxt, eta_min=0.)

        self._dist_fn = torch.distributions.Categorical
        self._alpha = alpha
        # self._tau = tau
        self._gamma = gamma
        self._rew_norm = reward_normalization
        # self._n_step = estimation_step
        self._lr_decay = lr_decay
        self._critic_target_tau = critic_target_tau
        self.training = False
        self._max_grad_norm = 10

    def train(self):
        self.training = True
        self._actor.training = True 
        self._critic.training = True
        self._target_critic.training = True
        self._target_actor.training = True
        self._diffusion.training = True

    def eval(self):
        self.training = False
        self._actor.training = False 
        self._critic.training = False
        self._target_critic.training = False
        self._target_actor.training = False
        self._diffusion.training = False

    def to(self, device):
        self._critic.to(device)
        self._target_critic.to(device)
        self._actor.to(device)
        self._target_actor.to(device)
        self._diffusion.to(device)

    def load_model(self, model_path, device):
        my_logger.log('successfully load model from {}.'.format(model_path))
        critic = torch.load(os.path.join(model_path, 'cur_ugv_critic.pth'), map_location=device)
        critic_tar = torch.load(os.path.join(model_path, 'cur_ugv_tar_critic.pth'), map_location=device)
        actor = torch.load(os.path.join(model_path, 'cur_ugv_actor.pth'), map_location=device)
        self._critic.load_state_dict(critic)
        self._critic.to(device).eval()
        self._target_critic.load_state_dict(critic_tar)
        self._target_critic.to(device).eval()
        self._actor.load_state_dict(actor)
        self._actor.to(device).eval()


    def load_best_model(self, model_path, device):
        my_logger.log('successfully load model from {}.'.format(model_path))
        critic = torch.load(os.path.join(model_path, 'best_ugv_critic.pth'), map_location=device)
        critic_tar = torch.load(os.path.join(model_path, 'best_ugv_tar_critic.pth'), map_location=device)
        actor = torch.load(os.path.join(model_path, 'best_ugv_actor.pth'), map_location=device)
        self._critic.load_state_dict(critic)
        self._critic.to(device).eval()
        self._target_critic.load_state_dict(critic_tar)
        self._target_critic.to(device).eval()
        self._actor.load_state_dict(actor)
        self._actor.to(device).eval()


    def _target_q(self, obs_next_) -> torch.Tensor:
        probs, _, _ = self.forward(obs_next_, model="target_actor")
        target_q = probs * self._target_critic.q_min(obs_next_)
        return target_q.sum(dim=-1)

    def save_model(self, model_path, model_name):
        torch.save(self._actor.state_dict(), os.path.join(model_path, f"{model_name}.pth"))
        my_logger.log('successfully save model to {}.'.format(os.path.join(model_path, f"{model_name}.pth")))

    def choose_action(self, obs_):
        _, action, _ = self.forward(obs_)
        action = action.cpu().numpy().squeeze()
        return action

    def update(
            self,
            use_grad_norm,
            replay_buffer,
    ):
        loss_dict = dict()
        # self.updating = True
        self.training = True
        self._actor.training = True
        state, action, reward, next_state = replay_buffer.sample()
        critic_loss_dict = self._update_critic(state, action, reward, next_state, use_grad_norm)
        loss_dict.update(critic_loss_dict)
        pg_loss_dict = self._update_policy(state, use_grad_norm)
        loss_dict.update(pg_loss_dict)
        self._update_targets()
        if self._lr_decay:
            self._actor_lr_scheduler.step()
            self._critic_lr_scheduler.step()
        # self.updating = False
        self.training = False
        return loss_dict

    def forward(
            self,
            obs_,
            model: str = "actor"
    ):
        model_ = self._actor if model == "actor" else self._target_actor
        shape = (obs_.shape[0], 1)
        acts, probs = self._diffusion.p_sample_loop(model_, obs_, shape, None)
        dist = self._dist_fn(probs=probs)
        return probs, acts, dist


    def _update_critic(self, obs_, acts_, rew_, obs_next_, use_grad_norm) -> torch.Tensor:
        target_q = self._target_q(obs_next_)
        target_q = rew_ + self._gamma * target_q
        target_q = target_q.unsqueeze(-1)
        current_q1, current_q2 = self._critic(obs_)
        critic_loss = F.mse_loss(current_q1.gather(1, acts_), target_q) \
                      + F.mse_loss(current_q2.gather(1, acts_), target_q)
        self._critic_optim.zero_grad()
        critic_loss.backward()
        if use_grad_norm:
            max_gradient = self._critic.max_gradient()
            nn.utils.clip_grad_norm_(self._critic.parameters(), self._max_grad_norm)
        else:
            max_gradient = 0
        self._critic_optim.step()
        loss_dict = {
            "critic_loss": critic_loss.item(),
            "critic_gradient": max_gradient
        }
        return loss_dict

    def _update_policy(self, obs_, use_grad_norm) -> torch.Tensor:
        probs, _, dist = self.forward(obs_)
        entropy = dist.entropy()
        # entropy = dist.entropy()
        with torch.no_grad():
            q = self._critic.q_min(obs_)
        # pg_loss = -(self._alpha * entropy + (dist.probs * q).sum(dim=-1)).mean()

        pg_loss = -(self._alpha * entropy + (probs * q).sum(dim=-1)).mean()
        self._actor_optim.zero_grad()
        pg_loss.backward()
        if use_grad_norm:
            max_gradient = self._actor.max_gradient()
            nn.utils.clip_grad_norm_(self._actor.parameters(), self._max_grad_norm)
        else:
            max_gradient = 0
        self._actor_optim.step()
        loss_dict = {
            "pg_loss": pg_loss.item(),
            'policy_gradient': max_gradient
        }
        return loss_dict

    def _update_targets(self):
        soft_update_params(self._critic, self._target_critic, self._critic_target_tau)
        soft_update_params(self._actor, self._target_actor, self._critic_target_tau)