from util import *


class MainLog:
    loss_dict = {}
    envs_info = {}

    def __init__(self, model_path, mode='train'):
        self.mode = mode
        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.log_path = ''
        writer_path = 'log'
        # self.writer = SummaryWriter(log_dir=writer_path)


    def save_model(self, model_name, model, verbose=False):
        work_dir = Path.cwd()
        _model_path = os.path.join(work_dir, self.model_path, model_name + '.pth')
        torch.save(model.state_dict(), _model_path)
        if verbose:
            my_logger.log('model has been saved to {}'.format(_model_path))

    def load_envs_info(self, mode='train'):
        episode_metrics_result_list = []
        env_num = 1
        # if mode == 'test':
        #     env_num = self.method_conf['test_num']
        for env_id in range(env_num):
            episode_metrics_result_list.append(
                np.load(os.path.join(self.log_path, 'process_' + str(env_id), 'episode_metrics_result.npy'),
                        allow_pickle=True)[()])
        for key in episode_metrics_result_list[0]:
            self.envs_info[key] = [episode_metrics_result_list[env_id][key] for env_id in range(env_num)]
        for key in self.envs_info:
            self.envs_info[key] = np.concatenate(
                [np.expand_dims(np.array(info_list), axis=1) for info_list in self.envs_info[key]], axis=1)

    def save_envs_info(self):
        envs_info_path = os.path.join(self.log_path, 'envs_info.npy')
        np.save(envs_info_path, self.envs_info)

    def record_report(self, report_str):
        report_path = os.path.join(self.log_path, 'report.txt')
        with open(report_path, 'a') as f:
            f.writelines(report_str + '\n')

    def load_sub_rollout_dict(self, env_id):
        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict.npy')
        sub_rollout_dict = np.load(sub_rollout_dict_path, allow_pickle=True)[()]
        return sub_rollout_dict

    def load_sub_rollout_dict_ugv(self, env_id):
        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict_ugv.npy')
        sub_rollout_dict = np.load(sub_rollout_dict_path, allow_pickle=True)[()]
        return sub_rollout_dict

    def delete_sub_rollout_dict(self, env_id):
        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict.npy')
        if os.path.exists(sub_rollout_dict_path):
            os.remove(sub_rollout_dict_path)

        sub_rollout_dict_path = os.path.join(self.log_path, 'process_' + str(env_id), 'sub_rollout_dict_ugv.npy')
        if os.path.exists(sub_rollout_dict_path):
            os.remove(sub_rollout_dict_path)

    def record_loss(self, loss_dict, role, iter_id):
        if role not in self.loss_dict:
            self.loss_dict[role] = {}
            for key in loss_dict.keys():
                self.loss_dict[role][key] = []

        for key in loss_dict.keys():
            if key not in self.loss_dict[role].keys():
                self.loss_dict[role][key] = []
            self.loss_dict[role][key].append(loss_dict[key])

        loss_dict_path = os.path.join(self.log_path, 'loss_dict.npy')
        np.save(loss_dict_path, self.loss_dict)


    def record_env_info(self, metrics, penalty, iter_id):
        eff, fairness, dcr, ecr, cor, cor1, cor2 = metrics
        hit, hover, fly = penalty
        self.writer.add_scalars('env_metric', {'eff': eff,
                                               'fair': fairness,
                                               'dcr': dcr,
                                               'ecr': ecr,
                                               'cor': cor,
                                               'cor1': cor1,
                                               'cor2': cor2,
                                               }, iter_id)
        self.writer.add_scalars('env_penalty', {'hit': hit,
                                                'hover': hover,
                                                'fly': fly}, iter_id)

