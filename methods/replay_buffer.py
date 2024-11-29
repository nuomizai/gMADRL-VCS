from util import *


class ReplayBufferHer:
    def __init__(self, device_id, capacity, batch_size, state_dim, ugv_n):
        self.device = 'cuda:' + str(device_id)
        self.capacity = capacity
        self.batch_size = batch_size
        self.n = 0
        self.tt_num = 0
        self.state = np.zeros((capacity*ugv_n, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity*ugv_n, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity*ugv_n, 1), dtype=np.int32)
        self.reward = np.zeros((capacity*ugv_n,), dtype=np.float32)
        self.tmp_buffer = {
            "state": [],
            "next_state": [],
            "obs": [],
            "next_obs": [],
            "action": [],
            "ach_goal": [],
            "goal": [],
            "move_dis": [],
            "dc": [],
            "reward": []
        }
        self.ugv_n = ugv_n

    def update_buffer(self):
        for t in range(len(self.tmp_buffer['state'])):
            for ugv_id in range(self.ugv_n):
                self.state[self.n] = self.tmp_buffer['state'][t][ugv_id]
                self.next_state[self.n] = self.tmp_buffer['next_state'][t][ugv_id]
                self.action[self.n] = self.tmp_buffer['action'][t][ugv_id]
                self.reward[self.n] = self.tmp_buffer['reward'][t][ugv_id]
            
                self.n  = (self.n + 1) % self.capacity
                self.tt_num  = min(self.tt_num + 1, self.capacity)
        for key in self.tmp_buffer:
            self.tmp_buffer[key] = []

    def her_augmentation(self, env):
        size = len(self.tmp_buffer['action'])
        for index in range(size):
            state_list = []
            next_state_list = []
            action_list = []
            reward_list = []
            for ugv_id in range(self.ugv_n):
                obs = self.tmp_buffer['obs'][index][ugv_id]
                next_obs = self.tmp_buffer['next_obs'][index][ugv_id]
                dc = self.tmp_buffer['dc'][index][ugv_id]
                move_dis = self.tmp_buffer['move_dis'][index][ugv_id]
                action = self.tmp_buffer['action'][index][ugv_id]
                ach_goal = self.tmp_buffer['ach_goal'][index]

                if index == size - 1:
                    her_goal = self.tmp_buffer['ach_goal'][index]
                else:
                    future_idx = np.random.choice(range(index + 1, size))
                    her_goal = self.tmp_buffer['ach_goal'][future_idx]


                reward = env.compute_ugv_reward(her_goal, ach_goal, dc, move_dis)

                her_state = np.concatenate([obs, her_goal], axis=-1)
                her_next_state = np.concatenate([next_obs, her_goal], axis=-1)
                state_list.append([her_state])
                next_state_list.append([her_next_state])
                reward_list.append([reward])
                action_list.append([action])
        
            state_list = np.concatenate(state_list, axis=0)
            next_state_list = np.concatenate(next_state_list, axis=0)
            reward_list = np.concatenate(reward_list, axis=0)
            action_list = np.concatenate(action_list, axis=0)
            self.tmp_buffer['state'].append(state_list)
            self.tmp_buffer['next_state'].append(next_state_list)
            self.tmp_buffer['reward'].append(reward_list)
            self.tmp_buffer['action'].append(action_list)


    def tmp_add(self, obs, next_obs, action, goal, ach_goal, move_dis, dc, rewards):
        self.tmp_buffer["obs"].append(copy.deepcopy(obs))
        self.tmp_buffer["next_obs"].append(copy.deepcopy(next_obs))
        self.tmp_buffer["goal"].append(copy.deepcopy(goal))
        self.tmp_buffer["ach_goal"].append(copy.deepcopy(ach_goal)) 
        repeat_goal = np.repeat([goal], self.ugv_n, axis=0)
        state = np.concatenate([obs, repeat_goal], axis=-1)
        next_state = np.concatenate([next_obs, repeat_goal], axis=-1)
        self.tmp_buffer["state"].append(copy.deepcopy(state))
        self.tmp_buffer["next_state"].append(copy.deepcopy(next_state))
        self.tmp_buffer["action"].append(copy.deepcopy(action))
        self.tmp_buffer["move_dis"].append(copy.deepcopy(move_dis))
        self.tmp_buffer["dc"].append(copy.deepcopy(dc))
        self.tmp_buffer["reward"].append(copy.deepcopy(rewards))



    def sample(self):
        N = self.tt_num
        indices = np.random.choice(range(N), self.batch_size)

        state_batch = torch.from_numpy(self.state[indices]).to(self.device)
        action_batch = torch.from_numpy(self.action[indices]).to(self.device).long()
        reward_batch = torch.from_numpy(self.reward[indices]).to(self.device)
        next_state_batch = torch.from_numpy(self.next_state[indices]).to(self.device)
        return state_batch, action_batch, reward_batch, next_state_batch
