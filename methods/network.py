from methods.distributions import *
from methods.net_tools import *
from util import UGV_STOP




class UGV_Critic(nn.Module):
    def __init__(self, ugv_o_dim, ugv_g_dim, ugv_a_dim, ugv_num, latent_dim=64):
        super(UGV_Critic, self).__init__()
        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))

        self.q1 = nn.Sequential(
            init_(nn.Linear(ugv_o_dim + ugv_g_dim + ugv_num * ugv_a_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, 1)),
        )
        self.q2 = nn.Sequential(
            init_(nn.Linear(ugv_o_dim + ugv_g_dim + ugv_num * ugv_a_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, latent_dim)),
            nn.ReLU(),
            init_(nn.Linear(latent_dim, 1)),
        )

    def forward(self, state, action):
        action = action.float()
        input = torch.cat([state, action], dim=-1)

        q1 = self.q1(input)
        q2 = self.q2(input)
        return q1.flatten(), q2.flatten()

    def print_data(self):
        for n, p in self.named_parameters():
            print('param:{}, value:{}'.format(n, p.data.mean()))

    def print_grad(self):
        print('-------------- print grad ---------------')
        for n, p in self.named_parameters():
            print('param:{}, shape:{}, value:{}'.format(n, p.grad.size(), p.grad.mean().item()))



class UAV_Network(nn.Module):
    def __init__(self, uav_loc_obs_channel_num, lr, eps, hidden_size):
        super(UAV_Network, self).__init__()

        init_ = lambda m: init(m,
                               nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0))
        cnn_init_ = lambda m: init(m,
                                   nn.init.orthogonal_,
                                   lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain('relu'))

        self.CNN4loc_obs = nn.Sequential(
            cnn_init_(nn.Conv2d(uav_loc_obs_channel_num, 32, 8, stride=4, padding=2)),
            nn.ReLU(),
            cnn_init_(nn.Conv2d(32, 32, 5, stride=2, padding=1)),
            nn.ReLU(),
            cnn_init_(nn.Conv2d(32, 32, 3, stride=1, padding=1)),
            nn.ReLU(),
            Flatten(),
            cnn_init_(nn.Linear(32 * 4 * 4, hidden_size)),
            nn.ReLU()
        )

        self.critic_linear = nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size // 2)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size // 2, 1)),
        )

        self.actor_public_linear = torch.nn.Sequential(
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ReLU(),
        )

        self.actor_return = Catigorical_Generator(hidden_size, 2)
        self.actor_flight = DiagGaussian(hidden_size, 2)

        self.lr = lr
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr, eps=eps, weight_decay=1e-6)

    def gen_cnn_feature(self, loc_obs_s):
        feature = self.CNN4loc_obs(loc_obs_s)
        return feature

    def get_value(self, value_feature):
        return self.critic_linear(value_feature)

   
    def get_hybrid_action(self, actor_feature):
        actor_feature = self.actor_public_linear(actor_feature)
        flight_dist = self.actor_flight(actor_feature)
        flight_choice = flight_dist.sample()
        flight_log_prob = flight_dist.log_prob(flight_choice).sum(dim=-1, keepdim=True)  

        return_dist = self.actor_return(actor_feature)
        return_choice = return_dist.sample()
        actions = torch.cat([return_choice, flight_choice.squeeze(0)])
        log_probs = flight_log_prob.squeeze(0)
        log_probs = torch.sum(log_probs, dim=-1, keepdim=True)

        return actions, log_probs

    def evaluate_action(self, loc_obs_s, action_s):
        action_features = value_features = self.CNN4loc_obs(loc_obs_s)
        values = self.critic_linear(value_features)

        action_flight = action_s[:, UAV_FLIGHT:]  # (bs, 2)

        action_features = self.actor_public_linear(action_features)

        flight_dists = self.actor_flight(action_features)
        flight_entropys = flight_dists.entropy()
        actions_flight_log_prob = flight_dists.log_prob(action_flight).sum(dim=-1, keepdim=True)


        actions_log_prob = actions_flight_log_prob
        dists_entropy = flight_entropys

        actions_log_prob = torch.sum(actions_log_prob, dim=-1, keepdim=True)
        dists_entropy = torch.sum(dists_entropy, dim=-1, keepdim=True)
        values = values.squeeze(-1)
        actions_log_prob = actions_log_prob.squeeze(-1)
        return values, actions_log_prob, dists_entropy

    def print_grad(self):
        print('-------------- print grad ---------------')
        for n, p in self.named_parameters():
            if p.requires_grad:
                if 'return' in n:
                    print('param:{} has no grad'.format(n))
                else:
                    print('param:{}, shape:{}, value:{}'.format(n, p.grad.size(), p.grad.mean().item()))
            else:
                print(f'param {n} does not need grad.')

    def print_data(self):
        print('-------------- print data ---------------')
        for n, p in self.named_parameters():
            print('param:{}, value:{}'.format(n, p.data.mean()))
