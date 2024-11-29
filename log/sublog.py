from util import *
from matplotlib.transforms import Bbox
import time
from matplotlib.patches import Circle

class SubLog:
    def __init__(self, coordx_max, coordy_max, dataset_path, zone_id, ball_id, lon_min, lat_min,
                 coordx_per_lon, coordy_per_lat, max_move_dist, move_energy_ratio, poi_value_max, poi_num, ugv_step_interval, ugv_n, uav_n, max_step_num, uav_sensing_range):
        self.coordx_max = coordx_max
        self.coordy_max = coordy_max
        self.dataset_path = dataset_path
        self.zone_id = zone_id
        self.ball_id = ball_id
        self.lon_min = lon_min
        self.lat_min = lat_min
        self.coordx_per_lon = coordx_per_lon
        self.coordy_per_lat = coordy_per_lat
        self.max_move_dist = max_move_dist
        self.move_energy_ratio = move_energy_ratio
        self.poi_value_max = poi_value_max
        self.poi_num = poi_num
        self.ugv_n = ugv_n
        self.uav_n = uav_n
        self.uav_sensing_range = uav_sensing_range


        self.episode_metrics_result = {}
        self.episode_metrics_result['eff'] = []
        self.episode_metrics_result['fairness'] = []
        self.episode_metrics_result['fairness2'] = []
        self.episode_metrics_result['dcr'] = []
        self.episode_metrics_result['hit'] = []
        # self.episode_metrics_result['fly'] = []
        self.episode_metrics_result['hover'] = []
        self.episode_metrics_result['ec'] = []
        self.episode_metrics_result['ecr'] = []
        self.episode_metrics_result['cor'] = []
        self.episode_metrics_result['cor1'] = []
        self.episode_metrics_result['cor2'] = []
        self.episode_metrics_result['cor3'] = []
        self.episode_metrics_result['inference_time'] = []
        # self.episode_metrics_result['pa_cor'] = []
        # self.episode_metrics_result['pa_cor2'] = []
        self.episode_metrics_result['ugv_avgR'] = []
        self.episode_metrics_result['uav_avgR'] = []

        self.log_path = 'process_0'
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)

        # self.episode_log_info_dict_dir_path = os.path.join(self.log_path, 'episode_log_info_dict')
        self.episode_log_info_dict_dir_path = os.path.join(self.log_path, 'episode_log_info_dict')
        if not os.path.exists(self.episode_log_info_dict_dir_path):
            os.makedirs(self.episode_log_info_dict_dir_path)

        # for draw
        # self.color_list = ['C' + str(color_id) for color_id in range(10)]
        self.color_list = ['red', 'green', 'purple', 'blue', 'black', 'coral', 'yellow', 'greenyellow']
        self.marker_list = ['o', '.', 'v', '^', '<', '>', '1', '2', '3', '4']
        self.ugv_step_interval = ugv_step_interval
        self.max_step_num = max_step_num

        # plt.ion()  # Turn on interactive mode
        # self.figure, self.ax = plt.subplots(figsize=(10, 10))  # Create a figure and axis object for updates
        # self.ax.set_xlim(0, self.coordx_max)  # Set the x-axis limits
        # self.ax.set_ylim(0, self.coordy_max)  # Set the y-axis limits

    def record_sub_rollout_dict(self, sub_rollout_manager):
        sub_rollout_dict_path = os.path.join(self.log_path, 'sub_rollout_dict.npy')
        np.save(sub_rollout_dict_path, sub_rollout_manager.sub_rollout_dict)

    def record_sub_buffer_dict(self, sub_buffer_manager):
        sub_buffer_dict_path = os.path.join(self.log_path, 'sub_rollout_dict_ugv.npy')
        np.save(sub_buffer_dict_path, sub_buffer_manager.sub_rollout_dict)

    def record_episode_log_info_dict(self, iter_id, env):
        # episode_log_info_dict = {'UGV': {},
        #                          'UAV': {}}
        #
        # for ugv_id in range(env.ugv_n):
        #     episode_log_info_dict['UGV'][ugv_id] = copy.deepcopy(env.UGV_list[ugv_id].episode_log_info_dict)
        # todo: rm ugv
        episode_log_info_dict = {'UAV': {}}
        for uav_id in range(env.uav_n):
            episode_log_info_dict['UAV'][uav_id] = copy.deepcopy(env.UAV_list[uav_id].episode_log_info_dict)

        episode_log_info_dict_path = os.path.join(self.episode_log_info_dict_dir_path,
                                                  'episode_log_info_dict_' + str(iter_id) + '.npy')
        np.save(episode_log_info_dict_path, episode_log_info_dict)

    def record_errors(self, report_str):
        self._report_path = self.log_path + '/errors.txt'
        f = open(self._report_path, 'a')
        f.writelines(report_str + '\n')
        f.close()

    def gen_metrics_result(self, iter_id, env, inference_time=0.0):
        if len(self.episode_metrics_result['inference_time']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['inference_time'].append(inference_time)

        # fairness
        fairness = 0.0
        final_poi_visit_time = np.clip(env.episode_log_info_dict['final_poi_visit_time'][-1], 0, 2)
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        if len(self.episode_metrics_result['fairness']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['fairness'].append(fairness)

        fairness2 = 0.0
        collect_poi_ratio = env.poi_cur_value_array / env.poi_init_vals
        square_of_sum = np.square(np.sum(collect_poi_ratio))
        sum_of_square = np.sum(np.square(collect_poi_ratio))
        if sum_of_square > 1e-5:
            fairness2 = square_of_sum / sum_of_square / collect_poi_ratio.shape[0]
        if len(self.episode_metrics_result['fairness2']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['fairness2'].append(fairness2)

        # data_collection_ratio (dcr)
        dcr = np.sum(env.poi_init_vals - env.poi_cur_value_array) / np.sum(env.poi_init_vals)
        if len(self.episode_metrics_result['dcr']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['dcr'].append(dcr)

        # # UGV UAV cooperation (cor)
        cor = np.mean(env.episode_log_info_dict['cor'])
        if len(self.episode_metrics_result['cor']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['cor'].append(cor)

        # move_dis = 0
        # for ugv in env.UGV_list:
        #     move_dis += ugv.move_dis
        #
        # if len(self.episode_metrics_result['cor1']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor1'].append(100 * dcr/move_dis)

        # hit
        hit = env.final_total_hit
        if len(self.episode_metrics_result['hit']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['hit'].append(hit)

        # # hover
        # hover = env.final_total_hover
        # if len(self.episode_metrics_result['hover']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['hover'].append(hover)

        # # fly
        # fly = env.final_total_fly_time
        # if len(self.episode_metrics_result['fly']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['fly'].append(fly)

        # energy_consumption (ec)
        ec = env.final_energy_consumption
        if len(self.episode_metrics_result['ec']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['ec'].append(ec)

        # energy_consumption_ratio (ecr)
        ec_upper_bound = env.uav_n * self.max_move_dist * self.max_step_num * self.move_energy_ratio
        ecr = ec / ec_upper_bound
        # another version (adopted)
        # if env.final_total_relax_time == 0:
        #     env.final_total_relax_time = env.uav_n
        # ecr4 = ec / (self.env_conf['uav_init_energy'] * env.final_total_relax_time)
        if len(self.episode_metrics_result['ecr']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['ecr'].append(ecr)

        # eff
        eff = 0.0
        # if ecr4 > min_value:
        #     eff = fairness * dcr * pa_cor / ecr4
        # todo: change
        if ecr > min_value:
            eff = fairness * dcr / ecr
        if len(self.episode_metrics_result['eff']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['eff'].append(eff)

        # if len(self.episode_metrics_result['cor2']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor2'].append(100 * eff / move_dis)

        ugv_move_dis = np.sum(env.episode_log_info_dict['move_dis'], axis=0)
        ugv_dcr = np.sum(env.episode_log_info_dict['ugv_dcr'], axis=0)
        # group_uav_n = env.group_uav_n
        cor1 = np.min(ugv_dcr / ugv_move_dis)
        if len(self.episode_metrics_result['cor1']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['cor1'].append(cor1)

        cor2 = np.mean(ugv_dcr / ugv_move_dis)
        if len(self.episode_metrics_result['cor2']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['cor2'].append(cor2)

        cor3 = 0.0
        # cor3 = np.min(ugv_dcr / group_uav_n / ugv_move_dis)
        # if len(self.episode_metrics_result['cor3']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor3'].append(cor3)

        # cor4 = np.mean(ugv_dcr / group_uav_n / ugv_move_dis)
        # if len(self.episode_metrics_result['cor4']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor4'].append(cor4)
        #
        # ugv_move_dis = np.sum(env.episode_log_info_dict['move_dis'], axis=1)
        # ugv_dcr = np.sum(env.episode_log_info_dict['ugv_dcr'], axis=1)
        # cor5 = np.min(ugv_dcr / ugv_move_dis)
        # if len(self.episode_metrics_result['cor5']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor5'].append(cor5)
        # cor6 = np.min((ugv_dcr / len(env.UAV_list)) / (ugv_move_dis / len(env.UGV_list)))
        # if len(self.episode_metrics_result['cor6']) != iter_id:
        #     report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
        #     print(report_str)
        #     self.record_errors(report_str)
        # else:
        #     self.episode_metrics_result['cor6'].append(cor6)

    def gen_rewards_result(self, sub_rollout_manager, sub_buffer_manager, iter_id, env):
        ugv_eps_reward = 0.
        if sub_buffer_manager is not None:
            for ugv_id in range(env.ugv_n):
                for sub_episode_buffer in sub_buffer_manager.sub_rollout_dict[str(ugv_id)].sub_episode_buffer_list:
                    if len(sub_episode_buffer['reward']) == 0:
                        ugv_eps_reward += 0
                    else:
                        ugv_eps_reward += np.mean(np.array(sub_episode_buffer['reward']))

            ugv_eps_reward /= env.ugv_n

        # ugv_eps_reward /= env.ugv_n
        if len(self.episode_metrics_result['ugv_avgR']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['ugv_avgR'].append(ugv_eps_reward)
        uav_eps_reward = 0.
        mean_uav_charge_time = 0.
        mean_uav_needless_charge_time = 0.
        for uav_id in range(env.uav_n):
            uav_sub_rollout_id = str(uav_id)
            if uav_sub_rollout_id in sub_rollout_manager.sub_rollout_dict['UAV']:
                for sub_episode_buffer in sub_rollout_manager.sub_rollout_dict['UAV'][
                    uav_sub_rollout_id].sub_episode_buffer_list:
                    uav_eps_reward += np.mean(np.array(sub_episode_buffer['reward_s']))

        uav_eps_reward /= env.uav_n
        if len(self.episode_metrics_result['uav_avgR']) != iter_id:
            report_str = str(datetime.now()).split('.')[0] + ' episode_metrics_result ERROR!'
            print(report_str)
            self.record_errors(report_str)
        else:
            self.episode_metrics_result['uav_avgR'].append(uav_eps_reward)

        for uav in env.UAV_list:
            mean_uav_charge_time += uav.charge_time
            mean_uav_needless_charge_time += uav.needless_charge_time
        mean_uav_charge_time /= env.uav_n
        mean_uav_needless_charge_time /= env.uav_n
        # self.episode_metrics_result['uav_chrg_time'].append(mean_uav_charge_time)
        # self.episode_metrics_result['uav_ndls_chrg_time'].append(mean_uav_needless_charge_time)

    def record_metrics_result(self):
        np.save(self.log_path + '/episode_metrics_result.npy', self.episode_metrics_result)

    def judge_same(self, loc1, loc2):
        if loc1[0] == loc2[0] and loc1[1] == loc2[1]:
            return True
        else:
            return False

    def load_trace(self, iter_id, env, mode='train'):
        Fig, ax = plt.subplots(figsize=(10, 10))
        plt.xlim(xmin=0, xmax=self.coordx_max)
        plt.ylim(ymin=0, ymax=self.coordy_max)
        cm = plt.cm.get_cmap('Blues')

        data_path = os.path.join(self.log_path, "trajectory", "episode_{}.npy".format(iter_id))
        print('data path:', data_path)
        assert os.path.exists(data_path)
        episode_dict = np.load(data_path, allow_pickle=True)[()]
        # load poi data
        poi_coordxy_array = episode_dict['poi_coordxy_array']
        poi_coordx_array = poi_coordxy_array[:, 0]
        poi_coordy_array = poi_coordxy_array[:, 1]
        poi_value = episode_dict['poi_value']

        # load uav trajectory
        all_uav_pos_list = episode_dict['all_uav_pos_list']

        # load ugv trajectory
        all_ugv_stop_list = episode_dict['all_ugv_stop_list']

        obstacles_file_path = os.path.join(self.dataset_path, 'obstacles.shp')
        obstacles_file = shapefile.Reader(obstacles_file_path)

        border_shape = obstacles_file
        border = border_shape.shapes()
        obstacle_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon

            coordys = (lats - self.lat_min) * self.coordy_per_lat
            obstacle_coord_list.append([coordxs, coordys])
            # plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        border_file_path = os.path.join(self.dataset_path, 'border.shp')
        border_file = shapefile.Reader(border_file_path)
        border_shape = border_file
        border = border_shape.shapes()
        boarder_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon
            coordys = (lats - self.lat_min) * self.coordy_per_lat
            boarder_coord_list.append([coordxs, coordys])

        stop_vis_dict = {}
        for stop_id in env.stops_net_dict:
            stop_vis_dict[stop_id] = 0

        uav_coordx_list = [[] for _ in range(self.uav_n)]
        uav_coordy_list = [[] for _ in range(self.uav_n)]

        pre_stop_list = []
        for ugv_id in range(self.ugv_n):
            ugv = env.UGV_list[ugv_id]
            # draw UGV trace
            cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][0]
            pre_stop_list.append(cur_stop_id)
        marker_width = 3

        for step_id in range(self.max_step_num):

            for uav_id in range(self.uav_n):
                uav_coordx_list[uav_id].append(all_uav_pos_list[uav_id][step_id][0])
                uav_coordy_list[uav_id].append(all_uav_pos_list[uav_id][step_id][1])
            last_step = (step_id + 1) % self.ugv_step_interval == 0 or step_id + 1 == self.max_step_num
            if last_step:
                plt.clf()
                filtered_uav_coordx_list, filtered_uav_coordy_list = self.filter_uav_trajectory(uav_coordx_list,
                                                                                                uav_coordy_list)
                # draw obstacle
                for obstacle_coord in obstacle_coord_list:
                    plt.plot(obstacle_coord[0], obstacle_coord[1], color='#6666ff', label='fungis', alpha=0.2,
                             linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

                # draw roads
                for road_node_id in env.roads_net_dict:
                    road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
                    next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
                    for next_node_id in next_node_list:
                        next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                        plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                                 color='grey', label='fungis', linewidth=10, alpha=0.1)
                # draw border
                for border_coord in boarder_coord_list:
                    plt.plot(border_coord[0], border_coord[1], color='grey', linewidth=1,
                             linestyle='--')  # x横坐标 y纵坐标 ‘k-’线性为黑色

                # draw poi
                poi_value_norm = np.array(poi_value[step_id]) / poi_value_max
                plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                            zorder=2, alpha=0.7)

                # draw stop
                for stop_id in env.crucial_stops:
                    stop_node_coordx = env.stops_net_dict[stop_id]['coordxy'][0]
                    stop_node_coordy = env.stops_net_dict[stop_id]['coordxy'][1]
                    plt.scatter(stop_node_coordx, stop_node_coordy, marker="*", c='green', alpha=1, s=80)

                # draw ugv trace
                for ugv_id in range(self.ugv_n):
                    color = self.color_list[ugv_id % len(self.color_list)]
                    # draw UGV trace
                    cur_stop_id = all_ugv_stop_list[ugv_id][(step_id + 1) // self.ugv_step_interval - 1]
                    stop_vis_dict[cur_stop_id] += 1
                    cur_stop_id_list = all_ugv_stop_list[ugv_id][
                                       0:(step_id + 1) // self.ugv_step_interval]
                    plt.scatter(env.stops_net_dict[cur_stop_id]['coordxy'][0],
                                env.stops_net_dict[cur_stop_id]['coordxy'][1], marker="^", edgecolor=color, alpha=1,
                                s=200,
                                zorder=13, facecolor='white', linewidths=marker_width)

                    for item_id, cur_stop_id in enumerate(cur_stop_id_list[:-1]):
                        start_stop_id = cur_stop_id
                        goal_stop_id = cur_stop_id_list[item_id + 1]
                        stops_net_SP_key = str(start_stop_id) + '_' + str(goal_stop_id)
                        shortest_path = env.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
                        for sub_item_id, stop_id in enumerate(shortest_path[:-1]):
                            ugv_coordx_list = []
                            ugv_coordy_list = []
                            ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                            ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                            if shortest_path[sub_item_id + 1] != stop_id:
                                for mid_road_node_id in env.stops_net_dict[stop_id]['next_node2mid_road_id_list_dict'][
                                    shortest_path[sub_item_id + 1]]:
                                    road_node_coordx, road_node_coordy = env.roads_net_dict[mid_road_node_id]['coordxy']
                                    ugv_coordx_list.append(road_node_coordx)
                                    ugv_coordy_list.append(road_node_coordy)
                            ugv_coordx_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][0])
                            ugv_coordy_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][1])
                            # todo: ugv
                            plt.plot(ugv_coordx_list, ugv_coordy_list,
                                     color=color,
                                     linewidth=4, alpha=0.6, linestyle='--')

                # for artist in plt.gca().get_children():
                #     if isinstance(artist, plt.Annotation):
                #         artist.remove()

                recall_list = [False for _ in range(self.uav_n)]
                for ugv_id in range(self.ugv_n):
                    ugv = env.UGV_list[ugv_id]
                    cur_stop_id = all_ugv_stop_list[ugv_id][
                        (step_id + 1) // self.ugv_step_interval - 1]
                    if (step_id + 1) // self.ugv_step_interval < len(ugv.episode_log_info_dict['cur_stop_id_list']):
                        next_stop_id = all_ugv_stop_list[ugv_id][
                            (step_id + 1) // self.ugv_step_interval]
                        recall = next_stop_id != cur_stop_id
                    else:
                        recall = False
                    if recall:
                        for uav_id in ugv.init_uav_list:
                            recall_list[uav_id] = True

                # draw uav trace
                for uav_id in range(self.uav_n):
                    uav = env.UAV_list[uav_id]
                    ugv_belong_list = uav.episode_log_info_dict['ugv_belong_to']
                    ugv_belong = ugv_belong_list[step_id]
                    # todo: plot uav
                    color = self.color_list[ugv_belong % len(self.color_list)]
                    trajectory_num = len(filtered_uav_coordx_list[uav_id])
                    for filter_i in range(trajectory_num):
                        plt.plot(filtered_uav_coordx_list[uav_id][filter_i], filtered_uav_coordy_list[uav_id][filter_i],
                                 color=color, linewidth=2, zorder=4)
                        if filter_i < trajectory_num - 1:
                            plt.scatter(filtered_uav_coordx_list[uav_id][filter_i][-1],
                                        filtered_uav_coordy_list[uav_id][filter_i][-1],
                                        color=color, s=200, marker="v", zorder=14, facecolor='white',
                                        linewidths=marker_width)
                    if recall_list[uav_id]:
                        plt.scatter(filtered_uav_coordx_list[uav_id][-1][-1],
                                    filtered_uav_coordy_list[uav_id][-1][-1],
                                    color=color, s=200, marker="v", zorder=14, facecolor='white',
                                    linewidths=marker_width)
                if (step_id + 1) % 20 == 0:

                    for ugv_id in range(self.ugv_n):
                        ugv = env.UGV_list[ugv_id]
                        uav_list = ugv.init_uav_list
                        cur_stop_id = all_ugv_stop_list[ugv_id][
                            (step_id + 1) // self.ugv_step_interval - 1]
                        for uav_id in uav_list:
                            uav_coordx_list[uav_id] = filtered_uav_coordx_list[uav_id][-1]
                            uav_coordy_list[uav_id] = filtered_uav_coordy_list[uav_id][-1]

                            if (step_id + 1) // self.ugv_step_interval < len(
                                    ugv.episode_log_info_dict['cur_stop_id_list']):
                                next_stop_id = all_ugv_stop_list[ugv_id][(step_id + 1) // self.ugv_step_interval]
                                change_stop = next_stop_id != cur_stop_id

                            else:
                                change_stop = False
                            if change_stop:
                                uav_coordx_list[uav_id] = []
                                uav_coordy_list[uav_id] = []
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                my_axis = plt.gca()
                my_axis.spines['top'].set_linewidth(20)
                my_axis.spines['bottom'].set_linewidth(10)
                my_axis.spines['left'].set_linewidth(10)
                my_axis.spines['right'].set_linewidth(10)
                # plt.tight_layout()
                # if mode == 'train':
                fig_dir_path = os.path.join(self.log_path, 'trace_{}'.format(mode))
                if not os.path.exists(fig_dir_path):
                    os.makedirs(fig_dir_path)
                fig_path = os.path.join(fig_dir_path, '{}-{}.png'.format(iter_id, step_id + 1))
                plt.tight_layout()
                # todo: 5
                bbox = Bbox([[0.5, 0.5], [9.5, 9.5]])  # 设置边界框的大小
                Fig.savefig(fig_path, bbox_inches=bbox, pad_inches=0.0)

        plt.close()

    def filter_uav_trajectory(self, full_uav_coordx_list, full_uav_coordy_list):
        filter_uav_coordx_list = [[] for _ in range(self.uav_n)]
        filter_uav_coordy_list = [[] for _ in range(self.uav_n)]
        for uav_id in range(self.uav_n):
            full_x = full_uav_coordx_list[uav_id]
            full_y = full_uav_coordy_list[uav_id]
            pre_x = full_x[0]
            pre_y = full_y[0]
            full_len = len(full_x)
            idx_list = [0]
            idx = 0
            for _x, _y in zip(full_x[1:], full_y[1:]):
                dis = np.sqrt((pre_x - _x) ** 2 + (pre_y - _y) ** 2)
                pre_x = _x
                pre_y = _y
                if dis > 100:
                    idx_list.append(idx + 1)
                    # print("{}/{}: {:.2f}".format(idx, full_len, dis))
                idx += 1
            idx_list.append(full_len)
            pre_idx = idx_list[0]
            for idx in idx_list[1:]:
                filter_uav_coordx_list[uav_id].append(full_uav_coordx_list[uav_id][pre_idx:idx])
                filter_uav_coordy_list[uav_id].append(full_uav_coordy_list[uav_id][pre_idx:idx])
                pre_idx = idx
        return filter_uav_coordx_list, filter_uav_coordy_list

    # draw image
    def record_trace(self, iter_id, env, mode='train', subp_id=None):
        # todo: 4
        # mpl.style.use('default')
        Fig = plt.figure(figsize=(int(10), int(10)))

        plt.xlim(xmin=0, xmax=self.coordx_max)
        plt.ylim(ymin=0, ymax=self.coordy_max)
        # plt.grid(True, linestyle='-.', color='r')

        # cm = plt.cm.get_cmap('RdYlBu_r')
        cm = plt.cm.get_cmap('Blues')

        obstacles_file_path = os.path.join(self.dataset_path, 'obstacles.shp')
        obstacles_file = shapefile.Reader(obstacles_file_path)

        border_shape = obstacles_file
        border = border_shape.shapes()
        obstacle_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon

            coordys = (lats - self.lat_min) * self.coordy_per_lat
            obstacle_coord_list.append([coordxs, coordys])
            # plt.plot(coordxs, coordys, color='#6666ff', label='fungis', linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

        border_file_path = os.path.join(self.dataset_path, 'border.shp')
        border_file = shapefile.Reader(border_file_path)
        border_shape = border_file
        border = border_shape.shapes()
        boarder_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon
            coordys = (lats - self.lat_min) * self.coordy_per_lat
            boarder_coord_list.append([coordxs, coordys])

        poi_coordxy_array = np.zeros([self.poi_num, 2], dtype=np.float32)
        for poi_id in range(self.poi_num):
            poi_coordxy_array[poi_id][0] = env.poi2coordxy_value_dict[poi_id]['coordxy'][0]
            poi_coordxy_array[poi_id][1] = env.poi2coordxy_value_dict[poi_id]['coordxy'][1]

        poi_coordx_array = env.poi_coordxy_array[:, 0]
        poi_coordy_array = env.poi_coordxy_array[:, 1]

        poi_value = env.episode_log_info_dict['poi_cur_value_array']
        episode_dict = {}
        episode_dict['poi_value'] = poi_value
        episode_dict['poi_coordxy_array'] = env.poi_coordxy_array

        stop_vis_dict = {}
        for stop_id in env.stops_net_dict:
            stop_vis_dict[stop_id] = 0

        all_uav_pos_list = []
        for uav_id in range(self.uav_n):
            uav = env.UAV_list[uav_id]
            all_uav_pos_list.append(uav.episode_log_info_dict['final_pos'])
        episode_dict['all_uav_pos_list'] = all_uav_pos_list

        all_ugv_stop_list = []
        for ugv_id in range(self.ugv_n):
            ugv = env.UGV_list[ugv_id]
            all_ugv_stop_list.append(ugv.episode_log_info_dict['cur_stop_id_list'])
        episode_dict['all_ugv_stop_list'] = all_ugv_stop_list

        if mode == 'test':
            trajectory_path = os.path.join(self.log_path, "trajectory")
            if not os.path.exists(trajectory_path):
                os.mkdir(trajectory_path)
            data_path = os.path.join(trajectory_path, "episode_{}.npy".format(iter_id))
            assert not os.path.exists(data_path)
            np.save(data_path, episode_dict)
            print('Save trajectory data into {}.'.format(data_path))
        uav_coordx_list = [[] for _ in range(self.uav_n)]
        uav_coordy_list = [[] for _ in range(self.uav_n)]

        pre_stop_list = []
        for ugv_id in range(self.ugv_n):
            ugv = env.UGV_list[ugv_id]
            # draw UGV trace
            cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][0]
            pre_stop_list.append(cur_stop_id)
        marker_width = 3

        for step_id in range(self.max_step_num):

            for uav_id in range(self.uav_n):
                uav_coordx_list[uav_id].append(all_uav_pos_list[uav_id][step_id][0])
                uav_coordy_list[uav_id].append(all_uav_pos_list[uav_id][step_id][1])
            last_step = (step_id + 1) % self.ugv_step_interval == 0 or step_id + 1 == self.max_step_num
            if last_step:
                plt.clf()
                filtered_uav_coordx_list, filtered_uav_coordy_list = self.filter_uav_trajectory(uav_coordx_list,
                                                                                                uav_coordy_list)
                # draw obstacle
                for obstacle_coord in obstacle_coord_list:
                    plt.plot(obstacle_coord[0], obstacle_coord[1], color='#6666ff', label='fungis', alpha=0.2,
                             linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色

                # draw roads
                for road_node_id in env.roads_net_dict:
                    road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
                    next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
                    for next_node_id in next_node_list:
                        next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                        plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                                 color='grey', label='fungis', linewidth=10, alpha=0.1)
                # draw border
                for border_coord in boarder_coord_list:
                    plt.plot(border_coord[0], border_coord[1], color='grey', linewidth=1,
                             linestyle='--')  # x横坐标 y纵坐标 ‘k-’线性为黑色

                # draw poi
                poi_value_norm = np.array(poi_value[step_id]) / self.poi_value_max
                plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                            zorder=2, alpha=0.7)

                # draw stop
                for stop_id in env.crucial_stops:
                    stop_node_coordx = env.stops_net_dict[stop_id]['coordxy'][0]
                    stop_node_coordy = env.stops_net_dict[stop_id]['coordxy'][1]
                    plt.scatter(stop_node_coordx, stop_node_coordy, marker="*", c='green', alpha=1, s=80)

                # draw ugv trace
                for ugv_id in range(self.ugv_n):
                    color = self.color_list[ugv_id % len(self.color_list)]
                    ugv = env.UGV_list[ugv_id]
                    # draw UGV trace
                    cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
                        (step_id + 1) // self.ugv_step_interval - 1]
                    stop_vis_dict[cur_stop_id] += 1
                    cur_stop_id_list = ugv.episode_log_info_dict['cur_stop_id_list'][
                                       0:(step_id + 1) // self.ugv_step_interval]
                    plt.scatter(env.stops_net_dict[cur_stop_id]['coordxy'][0],
                                env.stops_net_dict[cur_stop_id]['coordxy'][1], marker="^", edgecolor=color, alpha=1,
                                s=200,
                                zorder=13, facecolor='white', linewidths=marker_width)

                    for item_id, cur_stop_id in enumerate(cur_stop_id_list[:-1]):
                        start_stop_id = cur_stop_id
                        goal_stop_id = cur_stop_id_list[item_id + 1]
                        stops_net_SP_key = str(start_stop_id) + '_' + str(goal_stop_id)
                        shortest_path = env.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
                        for sub_item_id, stop_id in enumerate(shortest_path[:-1]):
                            ugv_coordx_list = []
                            ugv_coordy_list = []
                            ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
                            ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
                            if shortest_path[sub_item_id + 1] != stop_id:
                                for mid_road_node_id in env.stops_net_dict[stop_id]['next_node2mid_road_id_list_dict'][
                                    shortest_path[sub_item_id + 1]]:
                                    road_node_coordx, road_node_coordy = env.roads_net_dict[mid_road_node_id]['coordxy']
                                    ugv_coordx_list.append(road_node_coordx)
                                    ugv_coordy_list.append(road_node_coordy)
                            ugv_coordx_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][0])
                            ugv_coordy_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][1])
                            # todo: ugv
                            plt.plot(ugv_coordx_list, ugv_coordy_list,
                                     color=color,
                                     linewidth=4, alpha=0.6, linestyle='--')

                # for artist in plt.gca().get_children():
                #     if isinstance(artist, plt.Annotation):
                #         artist.remove()

                recall_list = [False for _ in range(self.uav_n)]
                for ugv_id in range(self.ugv_n):
                    ugv = env.UGV_list[ugv_id]
                    cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
                        (step_id + 1) // self.ugv_step_interval - 1]
                    if (step_id + 1) // self.ugv_step_interval < len(ugv.episode_log_info_dict['cur_stop_id_list']):
                        next_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
                            (step_id + 1) // self.ugv_step_interval]
                        recall = next_stop_id != cur_stop_id
                    else:
                        recall = False
                    if recall:
                        for uav_id in ugv.init_uav_list:
                            recall_list[uav_id] = True

                # draw uav trace
                for uav_id in range(self.uav_n):
                    uav = env.UAV_list[uav_id]
                    ugv_belong_list = uav.episode_log_info_dict['ugv_belong_to']
                    ugv_belong = ugv_belong_list[step_id]
                    # todo: plot uav
                    color = self.color_list[ugv_belong % len(self.color_list)]
                    trajectory_num = len(filtered_uav_coordx_list[uav_id])
                    for filter_i in range(trajectory_num):
                        plt.plot(filtered_uav_coordx_list[uav_id][filter_i], filtered_uav_coordy_list[uav_id][filter_i],
                                 color=color, linewidth=2, zorder=4)
                        if filter_i < trajectory_num - 1:
                            plt.scatter(filtered_uav_coordx_list[uav_id][filter_i][-1],
                                        filtered_uav_coordy_list[uav_id][filter_i][-1],
                                        color=color, s=200, marker="v", zorder=14, facecolor='white',
                                        linewidths=marker_width)
                    if recall_list[uav_id]:
                        plt.scatter(filtered_uav_coordx_list[uav_id][-1][-1],
                                    filtered_uav_coordy_list[uav_id][-1][-1],
                                    color=color, s=200, marker="v", zorder=14, facecolor='white',
                                    linewidths=marker_width)
                if (step_id + 1) % 20 == 0:

                    for ugv_id in range(self.ugv_n):
                        ugv = env.UGV_list[ugv_id]
                        uav_list = ugv.init_uav_list
                        cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
                            (step_id + 1) // self.ugv_step_interval - 1]
                        for uav_id in uav_list:
                            uav_coordx_list[uav_id] = filtered_uav_coordx_list[uav_id][-1]
                            uav_coordy_list[uav_id] = filtered_uav_coordy_list[uav_id][-1]

                            if (step_id + 1) // self.ugv_step_interval < len(
                                    ugv.episode_log_info_dict['cur_stop_id_list']):
                                next_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
                                    (step_id + 1) // self.ugv_step_interval]
                                change_stop = next_stop_id != cur_stop_id

                            else:
                                change_stop = False
                            if change_stop:
                                uav_coordx_list[uav_id] = []
                                uav_coordy_list[uav_id] = []
                plt.axis('off')
                plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                my_axis = plt.gca()
                # my_axis.spines['top'].set_linewidth(20)
                # my_axis.spines['bottom'].set_linewidth(10)
                # my_axis.spines['left'].set_linewidth(10)
                # my_axis.spines['right'].set_linewidth(10)
                # plt.tight_layout()
                # if mode == 'train':
                fig_dir_path = os.path.join(self.log_path, 'trace_{}'.format(mode))
                if not os.path.exists(fig_dir_path):
                    os.makedirs(fig_dir_path)
                fig_path = os.path.join(fig_dir_path, '{}-{}.png'.format(iter_id, step_id + 1))
                # todo: 5
                plt.tight_layout()
                bbox = Bbox([[0.5, 0.5], [9.5, 9.5]])  # 设置边界框的大小
                Fig.savefig(fig_path, bbox_inches=bbox, pad_inches=0.0)

                # im = plt.imread(fig_path)
                # plt.close()
                # scale = 0.2
                # width = im.shape[0]
                # height = im.shape[1]
                # print(width, height)
                # im = im[int(scale*width):width-int(scale*width), int(scale*height):height-int(scale*height)]
                # plt.axis('off')
                # # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
                # # my_axis = plt.gca()
                # # my_axis.spines['top'].set_linewidth(20)
                # # my_axis.spines['bottom'].set_linewidth(10)
                # # my_axis.spines['left'].set_linewidth(10)
                # # my_axis.spines['right'].set_linewidth(10)
                # plt.imshow(im)
                # plt.tight_layout()
                #
                # fig_path = os.path.join(fig_dir_path, '{}-{}.png'.format(iter_id, step_id + 1))
                # plt.savefig(fig_path)
        plt.close()

    # show image
    def show_trace(self, iter_id, env, mode='train', subp_id=None):
        plt.clf()
        Fig, ax = plt.subplots(figsize=(10, 10))

        plt.xlim(xmin=0, xmax=self.coordx_max)
        plt.ylim(ymin=0, ymax=self.coordy_max)

        cm = plt.cm.get_cmap('Blues')

        obstacles_file_path = os.path.join(self.dataset_path, 'obstacles.shp')
        obstacles_file = shapefile.Reader(obstacles_file_path)

        border_shape = obstacles_file
        border = border_shape.shapes()
        obstacle_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon

            coordys = (lats - self.lat_min) * self.coordy_per_lat
            obstacle_coord_list.append([coordxs, coordys])

        border_file_path = os.path.join(self.dataset_path, 'border.shp')
        border_file = shapefile.Reader(border_file_path)
        border_shape = border_file
        border = border_shape.shapes()
        boarder_coord_list = []
        for bd in border:
            border_points = bd.points
            x, y = zip(*border_points)
            lats, lons = utm.to_latlon(np.array(x), np.array(y), self.zone_id,
                                       self.ball_id)
            coordxs = (lons - self.lon_min) * self.coordx_per_lon
            coordys = (lats - self.lat_min) * self.coordy_per_lat
            boarder_coord_list.append([coordxs, coordys])

        poi_coordxy_array = np.zeros([self.poi_num, 2], dtype=np.float32)
        for poi_id in range(self.poi_num):
            poi_coordxy_array[poi_id][0] = env.poi2coordxy_value_dict[poi_id]['coordxy'][0]
            poi_coordxy_array[poi_id][1] = env.poi2coordxy_value_dict[poi_id]['coordxy'][1]

        poi_coordx_array = poi_coordxy_array[:, 0]
        poi_coordy_array = poi_coordxy_array[:, 1]

        poi_value = env.episode_log_info_dict['poi_cur_value_array']
        episode_dict = {}
        episode_dict['poi_value'] = poi_value
        episode_dict['poi_coordxy_array'] = poi_coordxy_array

        stop_vis_dict = {}
        for stop_id in env.stops_net_dict:
            stop_vis_dict[stop_id] = 0

        all_uav_pos_list = []
        for uav_id in range(self.uav_n):
            uav = env.UAV_list[uav_id]
            all_uav_pos_list.append(uav.episode_log_info_dict['final_pos'])
        episode_dict['all_uav_pos_list'] = all_uav_pos_list

        all_ugv_stop_list = []
        for ugv_id in range(self.ugv_n):
            ugv = env.UGV_list[ugv_id]
            all_ugv_stop_list.append(ugv.episode_log_info_dict['cur_stop_id_list'])
        episode_dict['all_ugv_stop_list'] = all_ugv_stop_list

        # draw obstacle
        for obstacle_coord in obstacle_coord_list:
            plt.plot(obstacle_coord[0], obstacle_coord[1], color='#000000', label='fungis', alpha=1.0,
                     linewidth=0.5)
        # draw roads
        for road_node_id in env.roads_net_dict:
            road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
            next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
            for next_node_id in next_node_list:
                next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
                plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
                         color='grey', label='fungis', linewidth=10, alpha=0.1)
        # draw border
        for border_coord in boarder_coord_list:
            plt.plot(border_coord[0], border_coord[1], color='grey', linewidth=1,
                     linestyle='--')

        # draw poi
        poi_value_norm = np.array(poi_value[-1]) / self.poi_value_max
        plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
                    zorder=2, alpha=0.7)

        # draw stop
        for stop_id in env.crucial_stops:
            stop_node_coordx = env.stops_net_dict[stop_id]['coordxy'][0]
            stop_node_coordy = env.stops_net_dict[stop_id]['coordxy'][1]
            plt.scatter(stop_node_coordx, stop_node_coordy, marker="*", c='green', alpha=1, s=80)

        # draw UAV pos
        uav_color = ['red', 'green']
        print('--------------> env.UAV_list[0].final_pos[0:2]:', env.UAV_list[0].final_pos[0:2])
        for uav_id in range(self.uav_n):
            pos = env.UAV_list[uav_id].final_pos[0:2]

            circle = Circle(pos, self.max_move_dist, color='gray', fill=True, linewidth=2, alpha=0.5)
            ax.add_patch(circle)
            plt.scatter(pos[0], pos[1], marker="^", c=uav_color[uav_id], alpha=1, s=80)

        plt.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        print('save trace under tmp_fig.jpg!!!!!!!!!1')
        plt.savefig("tmp_fig.jpg", bbox_inches='tight', pad_inches=0)
        plt.close()

        # self.figure.canvas.draw()  # Redraw the canvas
        # self.figure.canvas.flush_events()  # Process GUI events

        # uav_coordx_list = [[] for _ in range(self.uav_n)]
        # uav_coordy_list = [[] for _ in range(self.uav_n)]
        #
        # pre_stop_list = []
        # for ugv_id in range(self.ugv_n):
        #     ugv = env.UGV_list[ugv_id]
        #     # draw UGV trace
        #     cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][0]
        #     pre_stop_list.append(cur_stop_id)
        # marker_width = 3
        #
        # for step_id in range(self.max_step_num):
        #
        #     for uav_id in range(self.uav_n):
        #         uav_coordx_list[uav_id].append(all_uav_pos_list[uav_id][step_id][0])
        #         uav_coordy_list[uav_id].append(all_uav_pos_list[uav_id][step_id][1])
        #     last_step = (step_id + 1) % self.ugv_step_interval == 0 or step_id + 1 == self.max_step_num
        #     if last_step:
        #         plt.clf()
        #         filtered_uav_coordx_list, filtered_uav_coordy_list = self.filter_uav_trajectory(uav_coordx_list,
        #                                                                                         uav_coordy_list)
        #         # draw obstacle
        #         for obstacle_coord in obstacle_coord_list:
        #             plt.plot(obstacle_coord[0], obstacle_coord[1], color='#6666ff', label='fungis', alpha=0.2,
        #                      linewidth=0.5)  # x横坐标 y纵坐标 ‘k-’线性为黑色
        #
        #         # draw roads
        #         for road_node_id in env.roads_net_dict:
        #             road_node_coordx, road_node_coordy = env.roads_net_dict[road_node_id]['coordxy']
        #             next_node_list = env.roads_net_dict[road_node_id]['next_node_list']
        #             for next_node_id in next_node_list:
        #                 next_road_node_coordx, next_road_node_coordy = env.roads_net_dict[next_node_id]['coordxy']
        #                 plt.plot([road_node_coordx, next_road_node_coordx], [road_node_coordy, next_road_node_coordy],
        #                          color='grey', label='fungis', linewidth=10, alpha=0.1)
        #         # draw border
        #         for border_coord in boarder_coord_list:
        #             plt.plot(border_coord[0], border_coord[1], color='grey', linewidth=1,
        #                      linestyle='--')  # x横坐标 y纵坐标 ‘k-’线性为黑色
        #
        #         # draw poi
        #         poi_value_norm = np.array(poi_value[step_id]) / self.poi_value_max
        #         plt.scatter(poi_coordx_array, poi_coordy_array, c=poi_value_norm, vmin=0, vmax=1, cmap=cm, s=100,
        #                     zorder=2, alpha=0.7)
        #
        #         # draw stop
        #         for stop_id in env.crucial_stops:
        #             stop_node_coordx = env.stops_net_dict[stop_id]['coordxy'][0]
        #             stop_node_coordy = env.stops_net_dict[stop_id]['coordxy'][1]
        #             plt.scatter(stop_node_coordx, stop_node_coordy, marker="*", c='green', alpha=1, s=80)
        #
        #         # draw ugv trace
        #         for ugv_id in range(self.ugv_n):
        #             color = self.color_list[ugv_id % len(self.color_list)]
        #             ugv = env.UGV_list[ugv_id]
        #             # draw UGV trace
        #             cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                 (step_id + 1) // self.ugv_step_interval - 1]
        #             stop_vis_dict[cur_stop_id] += 1
        #             cur_stop_id_list = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                                0:(step_id + 1) // self.ugv_step_interval]
        #             plt.scatter(env.stops_net_dict[cur_stop_id]['coordxy'][0],
        #                         env.stops_net_dict[cur_stop_id]['coordxy'][1], marker="^", edgecolor=color, alpha=1,
        #                         s=200,
        #                         zorder=13, facecolor='white', linewidths=marker_width)
        #
        #             for item_id, cur_stop_id in enumerate(cur_stop_id_list[:-1]):
        #                 start_stop_id = cur_stop_id
        #                 goal_stop_id = cur_stop_id_list[item_id + 1]
        #                 stops_net_SP_key = str(start_stop_id) + '_' + str(goal_stop_id)
        #                 shortest_path = env.stops_net_SP_dict[stops_net_SP_key]['shortest_path']
        #                 for sub_item_id, stop_id in enumerate(shortest_path[:-1]):
        #                     ugv_coordx_list = []
        #                     ugv_coordy_list = []
        #                     ugv_coordx_list.append(env.stops_net_dict[stop_id]['coordxy'][0])
        #                     ugv_coordy_list.append(env.stops_net_dict[stop_id]['coordxy'][1])
        #                     if shortest_path[sub_item_id + 1] != stop_id:
        #                         for mid_road_node_id in env.stops_net_dict[stop_id]['next_node2mid_road_id_list_dict'][
        #                             shortest_path[sub_item_id + 1]]:
        #                             road_node_coordx, road_node_coordy = env.roads_net_dict[mid_road_node_id]['coordxy']
        #                             ugv_coordx_list.append(road_node_coordx)
        #                             ugv_coordy_list.append(road_node_coordy)
        #                     ugv_coordx_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][0])
        #                     ugv_coordy_list.append(env.stops_net_dict[shortest_path[sub_item_id + 1]]['coordxy'][1])
        #                     # todo: ugv
        #                     plt.plot(ugv_coordx_list, ugv_coordy_list,
        #                              color=color,
        #                              linewidth=4, alpha=0.6, linestyle='--')
        #
        #         # for artist in plt.gca().get_children():
        #         #     if isinstance(artist, plt.Annotation):
        #         #         artist.remove()
        #
        #         recall_list = [False for _ in range(self.uav_n)]
        #         for ugv_id in range(self.ugv_n):
        #             ugv = env.UGV_list[ugv_id]
        #             cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                 (step_id + 1) // self.ugv_step_interval - 1]
        #             if (step_id + 1) // self.ugv_step_interval < len(ugv.episode_log_info_dict['cur_stop_id_list']):
        #                 next_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                     (step_id + 1) // self.ugv_step_interval]
        #                 recall = next_stop_id != cur_stop_id
        #             else:
        #                 recall = False
        #             if recall:
        #                 for uav_id in ugv.init_uav_list:
        #                     recall_list[uav_id] = True
        #
        #         # draw uav trace
        #         for uav_id in range(self.uav_n):
        #             uav = env.UAV_list[uav_id]
        #             ugv_belong_list = uav.episode_log_info_dict['ugv_belong_to']
        #             ugv_belong = ugv_belong_list[step_id]
        #             # todo: plot uav
        #             color = self.color_list[ugv_belong % len(self.color_list)]
        #             trajectory_num = len(filtered_uav_coordx_list[uav_id])
        #             for filter_i in range(trajectory_num):
        #                 plt.plot(filtered_uav_coordx_list[uav_id][filter_i], filtered_uav_coordy_list[uav_id][filter_i],
        #                          color=color, linewidth=2, zorder=4)
        #                 if filter_i < trajectory_num - 1:
        #                     plt.scatter(filtered_uav_coordx_list[uav_id][filter_i][-1],
        #                                 filtered_uav_coordy_list[uav_id][filter_i][-1],
        #                                 color=color, s=200, marker="v", zorder=14, facecolor='white',
        #                                 linewidths=marker_width)
        #             if recall_list[uav_id]:
        #                 plt.scatter(filtered_uav_coordx_list[uav_id][-1][-1],
        #                             filtered_uav_coordy_list[uav_id][-1][-1],
        #                             color=color, s=200, marker="v", zorder=14, facecolor='white',
        #                             linewidths=marker_width)
        #         if (step_id + 1) % 20 == 0:
        #
        #             for ugv_id in range(self.ugv_n):
        #                 ugv = env.UGV_list[ugv_id]
        #                 uav_list = ugv.init_uav_list
        #                 cur_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                     (step_id + 1) // self.ugv_step_interval - 1]
        #                 for uav_id in uav_list:
        #                     uav_coordx_list[uav_id] = filtered_uav_coordx_list[uav_id][-1]
        #                     uav_coordy_list[uav_id] = filtered_uav_coordy_list[uav_id][-1]
        #
        #                     if (step_id + 1) // self.ugv_step_interval < len(
        #                             ugv.episode_log_info_dict['cur_stop_id_list']):
        #                         next_stop_id = ugv.episode_log_info_dict['cur_stop_id_list'][
        #                             (step_id + 1) // self.ugv_step_interval]
        #                         change_stop = next_stop_id != cur_stop_id
        #
        #                     else:
        #                         change_stop = False
        #                     if change_stop:
        #                         uav_coordx_list[uav_id] = []
        #                         uav_coordy_list[uav_id] = []
        #         plt.axis('off')
        #         plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
        #         my_axis = plt.gca()
        #         fig_dir_path = os.path.join(self.log_path, 'trace_{}'.format(mode))
        #         if not os.path.exists(fig_dir_path):
        #             os.makedirs(fig_dir_path)
        #         fig_path = os.path.join(fig_dir_path, '{}-{}.png'.format(iter_id, step_id + 1))
        #         # todo: 5
        #         plt.tight_layout()
        #         bbox = Bbox([[0.5, 0.5], [9.5, 9.5]])  # 设置边界框的大小
        #         plt.show(block=False)
        #
        #         # Fig.savefig(fig_path, bbox_inches=bbox, pad_inches=0.0)

        # plt.close()