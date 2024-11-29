import copy
import os.path
from methods.network import *
from methods.replay_buffer import ReplayBufferHer
from util import schedule, setup_my_logger, my_logger, Path
from methods.rollout import Rollout
from env.env import Env


@hydra.main(config_path='../cfgs', config_name='config')
def train(args):
    ugv_step_interval = args.ugv_step_interval
    max_step_num = args.max_step_num
    uav_num = args.uav_n
    fix_random_seed(args.seed)
    work_dir = Path.cwd()
    setup_my_logger(str(work_dir), log_dir=str(work_dir))
    fig_path = work_dir / 'fig'
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    log_path = work_dir / "save_data.npz"

    my_logger.log("Models are saved to {}".format(os.path.join(work_dir, args.model_path)))
    model_path = os.path.join(work_dir, args.model_path)
    if not os.path.exists(model_path):
        os.mkdir(model_path)
    device = "cuda:{}".format(args.device)


    # ----------------------- create env --------------------------------------
    env = Env(**args.env_cfg)
    # ----------------------- create uav agent --------------------------------------
    agent_cfg = args.agent
    uav_agent = hydra.utils.instantiate(agent_cfg)
    uav_agent.to(device)
    # ----------------------- create ugv agent --------------------------------------
    ugv_agent_cfg = args.ugv_agent
    goal_dim = len(env.crucial_stops)
    state_dim = 2 * len(env.crucial_stops) + 2 * args.ugv_n + goal_dim
    ugv_agent_cfg.ugv_net_cfg.state_dim = state_dim
    ugv_agent_cfg.ugv_net_cfg.action_dim = len(env.crucial_stops)
    ugv_agent_cfg.actor_cfg.state_dim = state_dim
    ugv_agent_cfg.actor_cfg.action_dim = len(env.crucial_stops)

    ugv_agent = hydra.utils.instantiate(ugv_agent_cfg)
    ugv_agent.to(device)

    # ----------------------- create rollout for uav--------------------------------------
    rollout = Rollout(**args.rollout_cfg)

    # ----------------------- create replay buffer for ugv--------------------------------------
    args.replay_buffer_cfg['state_dim'] = state_dim
    replay_buffer = ReplayBufferHer(**args.replay_buffer_cfg)


    train_step = 0
    log_infos = dict()

    assert max_step_num % ugv_step_interval == 0
    while train_step < args.train_iter:
        rollout.reset()

        ugv_state =env.reset()
        step_id = 0
        infos = dict()

        # ----------------------- set goal --------------------------------------
        global_init_goal = env.global_init_goal
        remain_data_ratio = schedule(args.goal_schedule, train_step)
        goal = global_init_goal * (np.random.rand(len(env.crucial_stops)) * remain_data_ratio)
        env.setup_goal(goal)
        while step_id < max_step_num:
            # ----------------------- ugv 1 step --------------------------------------
            ugv_actions = []
            repeat_goal = np.repeat([goal], env.ugv_n, axis=0)
            full_ugv_state = np.concatenate([ugv_state, repeat_goal], axis=-1)
            full_ugv_state = torch.from_numpy(full_ugv_state).to(device).float()
            ugv_actions = ugv_agent.choose_action(full_ugv_state)
            ugv_actions = ugv_actions.tolist()
            next_ugv_state, obses, ugv_move_dis = env.step_ugv(copy.deepcopy(ugv_actions))


            # ----------------------- uav 10 steps --------------------------------------
            for k in range(ugv_step_interval + 1):
                action_logps = []
                actions = []
                with torch.no_grad():
                    obs_tensor = torch.tensor(obses).to(device)
                    feature_batch = uav_agent.gen_cnn_feature(obs_tensor)

                    values = uav_agent.get_value(feature_batch)
                    for uav_id in range(uav_num):
                        uav_actor_feature = torch.index_select(feature_batch, 0,
                                                               torch.tensor([uav_id]).to(device))
                        action, action_log_prob = uav_agent.choose_action(uav_actor_feature)
                        action_logps.append(action_log_prob.tolist())
                        actions.append(action)

                values = values.tolist()

                if k < ugv_step_interval:
                    schedule_action = env.action_scheduler(actions)
                    next_obses, rewards, infos = env.step(schedule_action)

                if k == ugv_step_interval:
                    rollout.add_value(values)
                else:
                    rollout.add(obses, actions, rewards, values, action_logps)
                    obses = next_obses

            rollout.compute_returns(step_id, step_id + ugv_step_interval)
            # ---------------------------- replay buffer add ----------------------------
            ach_goal = env.get_global_ach_goal()
            ugv_dc = env.get_ugv_dc(ugv_step_interval)
            ugv_rewards = []
            for ugv_id in range(env.ugv_n):
                ugv_rewards.append(env.compute_ugv_reward(goal, ach_goal, ugv_dc[ugv_id], ugv_move_dis[ugv_id]))
            replay_buffer.tmp_add(ugv_state, next_ugv_state, ugv_actions, goal, ach_goal, ugv_move_dis, ugv_dc, ugv_rewards)

            ugv_state = next_ugv_state
            step_id += ugv_step_interval


        # ---------------------------- replay buffer her augmentation ----------------------------
        replay_buffer.her_augmentation(env)
        replay_buffer.update_buffer()

        for keys in infos:
            if keys in log_infos:
                log_infos[keys].append(infos[keys])
            else:
                log_infos[keys] = [infos[keys]]


        if train_step % 100 == 0:
            np.savez(log_path, **log_infos)


        # ---------------------------- uav train ----------------------------
        uav_agent.train()
        uav_agent.update(rollout)
        uav_agent.eval()

        # ---------------------------- ugv train ----------------------------   
        ugv_agent.train()
        for _ in range(args.update_times):
            ugv_agent.update(args.use_grad_norm, replay_buffer)
        ugv_agent.eval()


        if (train_step + 1) % 1000 == 0:
            uav_agent.save_model(model_path, model_name="ppo_{}".format(train_step+1))
            ugv_agent.save_model(model_path, model_name="ugv_{}".format(train_step+1))

        fairness = infos['fairness']
        dcr = infos['dcr']
        ecr = infos['ecr']
        eff = infos['eff']
        report_str = '#' * 100 \
                     + '\n' \
                     + 'iter: ' + str(train_step) \
                     + '\n\t' \
                     + ' fairness: ' + str(np.round(fairness, 5)) \
                     + ' dcr: ' + str(np.round(dcr, 5)) \
                     + ' ecr: ' + str(np.round(ecr, 5)) \
                     + ' eff: ' + str(np.round(eff, 5)) \
                     + '\n\t'
       
        my_logger.log(report_str)
        train_step += 1

if __name__ == '__main__':
    train()