from util import *
from einops import rearrange


class Rollout(object):
    def __init__(self, mini_batch_size, uav_n, device, gamma, tau, use_gae):
        self.device = 'cuda:' + str(device)
        self.mini_batch_size = mini_batch_size
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae
        self.m = uav_n

        self.obses = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.returns = []

    def reset(self):
        self.obses = []
        self.actions = []
        self.values = []
        self.rewards = []
        self.log_probs = []
        self.returns = []

    def add_value(self, value):
        self.values.append(copy.deepcopy(value))

    def add(self, obs, act, rew, value, log_prob):
        self.obses.append(copy.deepcopy(obs))
        self.actions.append(copy.deepcopy(act))
        self.rewards.append(copy.deepcopy(rew))
        self.values.append(copy.deepcopy(value))
        self.log_probs.append(copy.deepcopy(log_prob))

    def compute_returns(self, st, ed):
        self.values = np.array(self.values)
        self.rewards = np.array(self.rewards)
        gae = 0
        if self.use_gae:
            for step in reversed(range(st, ed)):
                delta = self.rewards[step] + self.gamma * self.values[step + 1] - self.values[step]
                gae = delta + self.gamma * self.tau * gae
                self.returns.insert(st, gae + self.values[step])
        else:
            next_return = self.values[-1]
            for step in reversed(range(st, ed)):
                next_return = self.rewards[step] + next_return * self.gamma
                self.returns.insert(st, next_return.tolist())

        self.values = self.values.tolist()
        self.rewards = self.rewards.tolist()
        self.values.pop(-1)

    def minibatch_generator(self, advantage):

        obses = torch.from_numpy(np.array(self.obses, dtype=np.float32)).to(self.device)
        actions = torch.from_numpy(np.array(self.actions, dtype=np.float32)).to(self.device)
        values = torch.from_numpy(np.array(self.values, dtype=np.float32)).to(self.device)
        log_probs = torch.from_numpy(np.array(self.log_probs, dtype=np.float32)).to(self.device)
        returns = torch.from_numpy(np.array(self.returns, dtype=np.float32)).to(self.device)
        advantages = torch.from_numpy(advantage).to(self.device)

        obses = rearrange(obses, 'N m ... -> (N m) ...')
        actions = rearrange(actions, 'N m ... -> (N m) ...')
        values = rearrange(values, 'N m ... -> (N m) ...')
        log_probs = rearrange(log_probs, 'N m ... -> (N m) ...')
        returns = rearrange(returns, 'N m ... -> (N m) ...')
        advantages = rearrange(advantages, 'N m ... -> (N m) ...')

        N = actions.size(0)
        sampler = BatchSampler(SubsetRandomSampler(range(N)), self.mini_batch_size, drop_last=False)

        for indices in sampler:
            obs_batch = obses[indices]
            act_batch = actions[indices]
            value_batch = values[indices]
            return_batch = returns[indices]
            log_probs_batch = log_probs[indices]
            advantage_batch = advantages[indices]

            yield obs_batch, act_batch, value_batch, return_batch, log_probs_batch, advantage_batch
