from util import *
from methods.net_tools import *
from methods.network import UAV_Network


class PPO:
    def __init__(self, ppo_epoch, clip_param, use_clipped_value_loss, value_loss_coef, entropy_coef, max_grad_norm, uav_net_cfg):
        self.ppo_epoch = ppo_epoch
        self.clip_param = clip_param
        self.use_clipped_value_loss = use_clipped_value_loss
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        self.ac = UAV_Network(**uav_net_cfg)
        # self.ac = ac
        self.ac.train()
        self.optimizer = self.ac.optimizer

    def train(self):
        self.ac.train()

    def to(self, device):
        self.ac.to(device)

    def eval(self):
        self.ac.eval()

    def gen_cnn_feature(self, uav_full_obs):
        """
        uavs_full_obs: [uav_num, obs]
        """
        obs_feat = self.ac.gen_cnn_feature(uav_full_obs)
        return obs_feat

    def get_value(self, obs_feat):
        value = self.ac.get_value(obs_feat)
        value = value.squeeze(-1)
        return value.cpu().numpy()

    def choose_action(self, obs_feat):
        action, log_probs = self.ac.get_hybrid_action(obs_feat)
        log_probs = log_probs.squeeze(-1)
        return action.cpu().numpy(), log_probs.cpu().numpy()

    def save_model(self, model_path, model_name, verbose=False):
        _model_path = os.path.join(model_path, model_name + '.pth')
        torch.save(self.ac.state_dict(), _model_path)
        if verbose:
            my_logger.log('model has been saved to {}'.format(_model_path))

    def load_model(self, pretrained_model_path, device):
        ac = torch.load(pretrained_model_path, map_location=device)
        self.ac.load_state_dict(ac)
        self.ac.to(device)
        self.ac.eval()

    def update(self, rollout):
        returns = np.array(rollout.returns)
        values = np.array(rollout.values)

        advantage = returns - values
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

        # role = self.ac.__class__.__name__

        total_loss = []
        total_action_loss = []
        total_value_loss = []
        total_entropy_loss = []

        for _ in range(self.ppo_epoch):
            data_generator = rollout.minibatch_generator(advantage)

            for sample_mini_batch in data_generator:

                obs_batch, action_batch, old_value_batch, return_batch, \
                old_action_logp_batch, advantage_batch = sample_mini_batch
                values, action_logp, entropy = self.ac.evaluate_action(obs_batch, action_batch)

                # actor loss
                ratio = torch.exp(action_logp - old_action_logp_batch)
                surr1 = ratio * advantage_batch
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param,
                                    1.0 + self.clip_param) * advantage_batch
                action_loss = - torch.min(surr1, surr2)
                action_loss = action_loss.mean()

                # value loss
                if self.use_clipped_value_loss:
                    value_pred_clipped = old_value_batch + \
                                         (values - old_value_batch).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * F.mse_loss(return_batch, values)

                entropy_loss = entropy.mean()
                self.optimizer.zero_grad()
                loss = value_loss * self.value_loss_coef + action_loss - entropy_loss * self.entropy_coef
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.optimizer.step()

                total_loss.append(loss.item())
                total_action_loss.append(action_loss.item())
                total_value_loss.append(value_loss.item())
                total_entropy_loss.append(entropy_loss.item())

        total_loss = np.array(total_loss).mean()
        total_action_loss = np.array(total_action_loss).mean()
        total_value_loss = np.array(total_value_loss).mean()
        total_entropy_loss = np.array(total_entropy_loss).mean()

        loss_dict = dict(
            total_loss=total_loss,
            action_loss=total_action_loss,
            value_loss=total_value_loss,
            entropy_loss=total_entropy_loss,

        )
        return loss_dict
