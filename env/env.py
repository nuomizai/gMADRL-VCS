import random

from util import *


class UGV:
    def __init__(self, ugv_id, uav_num_each_group):
        self.ugv_id = ugv_id
        self.pre_stop_id = None
        self.cur_stop_id = None
        self.init_uav_n = 0
        self.init_uav_list = []


    def add_init_uav(self, uav_id):
        self.init_uav_list.append(uav_id)
        self.init_uav_n += 1



class UAV:
    def __init__(self, uav_id):
        self.uav_id = uav_id
        self.stop_belong_to = None
        # uav的位置
        self.pre_pos = None
        self.final_pos = None
        # uav的当前能量与能量损耗
        self.final_energy = None
        self.final_energy_consumption = None
        self.charge_time = None
        self.needless_charge_time = None

        # uav碰撞障碍以及飞出ugv范围的次数
        self.final_hit = None
        self.final_hover = None
        self.final_out_of_ugv = None
        # 每个poi采集的数据量，以及采集的次数
        self.final_data_collection = None
        self.final_poi_visit_time = None
        self.final_collect_data_time = None
        self.whether_flight_collect = None
        self.episode_log_info_dict = {}

        self.ugv_belong_to = None

        self.last_status = None
        self.last_status_length = None
        self.next_status = None


    def add_log_info(self):
        self.episode_log_info_dict['pre_pos'].append(copy.deepcopy(self.pre_pos))
        self.episode_log_info_dict['final_pos'].append(copy.deepcopy(self.final_pos))
        self.episode_log_info_dict['final_energy'].append(self.final_energy)
        self.episode_log_info_dict['final_energy_consumption'].append(self.final_energy_consumption)
        self.episode_log_info_dict['final_hit'].append(self.final_hit)
        self.episode_log_info_dict['final_hover'].append(self.final_hover)
        self.episode_log_info_dict['final_out_of_ugv'].append(self.final_out_of_ugv)
        self.episode_log_info_dict['final_data_collection'].append(copy.deepcopy(self.final_data_collection))
        self.episode_log_info_dict['final_poi_visit_time'].append(copy.deepcopy(self.final_poi_visit_time))
        self.episode_log_info_dict['final_collect_data_time'].append(self.final_collect_data_time)
        self.episode_log_info_dict['ugv_belong_to'].append(self.ugv_belong_to)



class Env:
    def __init__(self, ugv_n, uav_n, mode, stop_poi_max_dis, collect_speed_poi,
                 global_horizon, ego_horizon, poi_value_min, poi_value_max, grid_size, uav_move_check_hit_gap,
                 uav_sensing_range, stop_gap, uav_init_energy, max_move_dist, move_energy_ratio,
                 hover_energy, dataset_path, coordx_max, coordy_max, ugv_move_norm, norm_her_reward, max_step_num):
        self.ugv_n = ugv_n
        self.uav_n = uav_n
        self.stop_poi_max_dis = stop_poi_max_dis
        self.collect_speed_poi = collect_speed_poi
        self.poi_value_max = poi_value_max
        self.grid_size = grid_size
        self.uav_move_check_hit_gap = uav_move_check_hit_gap
        self.uav_sensing_range = uav_sensing_range
        self.uav_init_energy = uav_init_energy
        self.max_move_dist = max_move_dist
        self.move_energy_ratio = move_energy_ratio
        self.hover_energy = hover_energy
        self.coordx_max = coordx_max
        self.coordy_max = coordy_max
        self.global_horizon = global_horizon
        self.ego_horizon = ego_horizon
        self.ugv_move_norm = ugv_move_norm
        self.norm_her_reward = norm_her_reward
        self.max_step_num = max_step_num
        # 0129 gen ugvs and uavs
        self.UAV_list = []
        self.UGV_list = []
        self.UAV_UGV_MAP = []
        UAV_UGV_INDEX = []
        for uav_id in range(self.uav_n):
            self.UAV_list.append(UAV(uav_id))
            ugv_id = uav_id % ugv_n
            UAV_UGV_INDEX.append(ugv_id)
        uav_ugv_arr = np.array(UAV_UGV_INDEX)

        for ugv_id in range(self.ugv_n):
            total_uav_in_ugv = np.sum(uav_ugv_arr == ugv_id)
            self.UGV_list.append(UGV(ugv_id, total_uav_in_ugv))

        for uav_id in range(self.uav_n):
            ugv_id = UAV_UGV_INDEX[uav_id]
            ugv = self.UGV_list[ugv_id]
            ugv.add_init_uav(uav_id)
            
            self.UAV_UGV_MAP.append(ugv)
        
        self.max_uav_n = 0
        for ugv_id in range(self.ugv_n):
            ugv = self.UGV_list[ugv_id]
            self.max_uav_n = max(self.max_uav_n, ugv.init_uav_n)

        self.mode = mode

        # load the position and initial value of point-of-interests (poi)
        poi2coordxy_value_dict = \
            np.load(os.path.join(dataset_path, 'poi2coordxy_value_dict_' + str(
                poi_value_min) + '_' + str(self.poi_value_max) + '.npy'), allow_pickle=True)[()]
        self.poi2coordxy_value_dict = poi2coordxy_value_dict
        self.poi_num = len(poi2coordxy_value_dict.keys())


        # initialize the value of PoIs at the beginning
        self.poi_init_vals = np.zeros(self.poi_num, dtype=np.float32)
        for poi_id in range(self.poi_num):
            self.poi_init_vals[poi_id] = poi2coordxy_value_dict[poi_id]['value']

        self.poi2cell_dict = np.load(os.path.join(dataset_path,
                                                  'poi2cell_dict_' + str(
                                                      self.grid_size) + '_' + str(
                                                      self.uav_sensing_range) + '.npy'), allow_pickle=True)[
            ()]
        # 采集所有uav可以飞行到的cell，划去建筑物占用空间
        self.uav_cellset = np.load(os.path.join(dataset_path,
                                                'uav_cellset_' + str(self.grid_size) + '.npy'),
                                   allow_pickle=True)[()]
        # 在uav采集范围内的cell以及与poi的对应关系
        self.uav_cell2poi_dict = np.load(os.path.join(dataset_path,
                                                      'uav_cell2poi_dict_' + str(
                                                          self.grid_size) + '_' + str(
                                                          self.uav_sensing_range) + '.npy'),
                                         allow_pickle=True)[()]

        self.occupied_matrix = self.gen_occupied_matrix(global_horizon)
        self.grid_init_vals, self.poi_grid_pos = self.gen_grid_pois(global_horizon)
        self.uav_poi_mask = np.ones(
            (self.uav_n, self.grid_init_vals.shape[0], self.grid_init_vals.shape[1]))
        
        # 每个道路节点的坐标以及邻接点
        self.roads_net_dict = \
            np.load(os.path.join(dataset_path, 'roads_net_dict.npy'), allow_pickle=True)[()]
        # 每个stop节点的坐标以及邻接点，所在道路段以及位置pos
        self.stops_net_dict = \
            np.load(os.path.join(dataset_path,
                                 'stops_net_dict_' + str(stop_gap) + '.npy'), allow_pickle=True)[()]
        self.stop_num = len(self.stops_net_dict)

        self.stops_pois_AdjMatrix = np.load(os.path.join( dataset_path, 'stops_pois_AdjMatrix_' + str(
            stop_gap) + '_' + str(stop_poi_max_dis) + '.npy'), allow_pickle=True)
        self.crucial_stops = np.load(
            os.path.join(dataset_path, 'crucial_stops_{}.npy'.format(stop_poi_max_dis)))
        self.crucial_stops_map = {}
        for stop_idx, stop in enumerate(self.crucial_stops):
            self.crucial_stops_map[stop] = stop_idx

        self.crucial_stop_pois_adj_num = np.sum(self.stops_pois_AdjMatrix[self.crucial_stops], axis=1)

        crucial_stop_pois_adj_num, crucial_stops = zip(
            *sorted(zip(self.crucial_stop_pois_adj_num, self.crucial_stops), reverse=True))
        self.UGV_init_stop_id_list = []
        crucial_stop_num = len(self.crucial_stops)
        for i in range(self.ugv_n):
            ugv_stop_idx = i % crucial_stop_num
            self.UGV_init_stop_id_list.append(crucial_stops[ugv_stop_idx])

        self.stops_net_SP_dict = np.load(os.path.join(dataset_path,
                                                      'stops_net_SP_dict_' + str(stop_gap) + '.npy'),
                                         allow_pickle=True)[()]
        self.stops_net_SP_Matrix = np.ones([self.stop_num, self.stop_num], dtype=np.float32)
        self.stops_net_SP_Matrix *= 1e3
        for key in self.stops_net_SP_dict:
            start_stop_id = int(key.split('_')[0])
            goal_stop_id = int(key.split('_')[1])
            self.stops_net_SP_Matrix[start_stop_id, goal_stop_id] = self.stops_net_SP_dict[key]['min_dis']

        self.crucial_stop_dis_Matrix = np.ones([crucial_stop_num, crucial_stop_num], dtype=np.float32)
        for i in range(len(self.crucial_stops)):
            stop_i = self.crucial_stops[i]
            for j in range(len(self.crucial_stops)):
                stop_j = self.crucial_stops[j]
                stop_key = str(stop_i) + '_' + str(stop_j)
                if i == j:
                    self.crucial_stop_dis_Matrix[i][j] = 1
                else:
                    self.crucial_stop_dis_Matrix[i][j] = np.ceil(self.stops_net_SP_dict[stop_key]['min_dis'])

        self.global_init_goal = np.sum(self.poi_init_vals * self.stops_pois_AdjMatrix, axis=1)[
                                self.crucial_stops] / self.poi_value_max
        # self.episode_log_info_dict = {}
    
    def setup_goal(self, goal):
        self.goal = goal
    
    def gen_occupied_matrix(self, global_horizon):
        # 涉及grid大小转换，减小cnn输入矩阵大小，保显存
        # 通道1：可飞行的范围 0/1
        occupied_matrix = np.zeros([global_horizon, global_horizon], dtype=np.float32)
        for i in range(global_horizon):
            cell_x = int((i + 0.5) / global_horizon * self.grid_size)
            for j in range(global_horizon):
                cell_y = int((j + 0.5) / global_horizon * self.grid_size)
                cell_key = str(cell_x) + '_' + str(cell_y)
                if cell_key in self.uav_cellset:
                    occupied_matrix[i, j] = 1
        return occupied_matrix

    def get_energy_value_list(self, ):
        all_energy_values = [self.UAV_list[uav_id].final_energy for uav_id in range(self.uav_n)]
        return all_energy_values

    def get_uav_global_pos_list(self, ):
        all_uav_global_pos_list = [self.UAV_list[uav_id].final_pos for uav_id in range(self.uav_n)]
        return all_uav_global_pos_list

    def pos2grid_xy(self, pos, global_horizon):
        # 坐标-uav_grid转换
        grid_x = int(pos[0] / self.coordx_max * global_horizon)
        grid_y = int(pos[1] / self.coordy_max * global_horizon)
        grid_xy = (grid_x, grid_y)
        return grid_xy

    def compute_observations(self, uav_id):
        uav = self.UAV_list[uav_id]

        global_poi_matrix = copy.deepcopy(self.grid_cur_vals)

        occupied_matrix = copy.deepcopy(self.occupied_matrix)

        all_energy_value_list = self.get_energy_value_list()
        all_global_pos_list = self.get_uav_global_pos_list()

        ego_energy_value = all_energy_value_list[uav_id]
        ego_pos = all_global_pos_list[uav_id]
        ego_grid_pos = self.pos2grid_xy(ego_pos, self.global_horizon)
        ego_energy_matrix = np.zeros((self.global_horizon, self.global_horizon), dtype=np.float32)
        ego_energy_matrix[ego_grid_pos[0]][ego_grid_pos[1]] = ego_energy_value / self.uav_init_energy

        other_energy_matrix = np.zeros((self.global_horizon, self.global_horizon), dtype=np.float32)
        for other_uav_id in range(self.uav_n):
            if other_uav_id == uav_id:
                continue
            other_energy_value = all_energy_value_list[other_uav_id]

            other_pos = all_global_pos_list[other_uav_id]
            other_grid_pos = self.pos2grid_xy(other_pos, self.global_horizon)

            other_energy_matrix[other_grid_pos[0]][other_grid_pos[1]] = other_energy_value / self.uav_init_energy

        center_grid_xy = self.pos2grid_xy(uav.final_pos, self.global_horizon)

        uav_global_obs = np.concatenate(
            [[global_poi_matrix], [occupied_matrix], [ego_energy_matrix], [other_energy_matrix]], axis=0)
        uav_loc_obs = self.glb2loc(uav_global_obs, center_grid_xy, self.ego_horizon)
        return uav_loc_obs

    def glb2loc(self, glb_obs, center_grid_xy, loc_obs_shape):
        obs_channel_num = glb_obs.shape[0]
        glb_obs_shape = glb_obs.shape[1]
        loc_obs = np.zeros([obs_channel_num, loc_obs_shape, loc_obs_shape], dtype=np.float32)

        half_loc_obs_shape = int(loc_obs_shape / 2)

        glb_obs_x_min = center_grid_xy[0] - half_loc_obs_shape
        glb_obs_y_min = center_grid_xy[1] - half_loc_obs_shape
        glb_obs_x_max = glb_obs_x_min + loc_obs_shape
        glb_obs_y_max = glb_obs_y_min + loc_obs_shape

        glb_obs_x_min = np.clip(glb_obs_x_min, 0, glb_obs_shape - 1)
        glb_obs_y_min = np.clip(glb_obs_y_min, 0, glb_obs_shape - 1)
        glb_obs_x_max = np.clip(glb_obs_x_max, 1, glb_obs_shape)
        glb_obs_y_max = np.clip(glb_obs_y_max, 1, glb_obs_shape)

        loc_obs_x_min = half_loc_obs_shape - center_grid_xy[0]
        loc_obs_y_min = half_loc_obs_shape - center_grid_xy[1]
        loc_obs_x_min = np.clip(loc_obs_x_min, 0, loc_obs_shape - 1)
        loc_obs_y_min = np.clip(loc_obs_y_min, 0, loc_obs_shape - 1)

        loc_obs_x_max = loc_obs_x_min + glb_obs_x_max - glb_obs_x_min
        loc_obs_y_max = loc_obs_y_min + glb_obs_y_max - glb_obs_y_min
        loc_obs_x_max = np.clip(loc_obs_x_max, 1, loc_obs_shape)
        loc_obs_y_max = np.clip(loc_obs_y_max, 1, loc_obs_shape)

        # 不足大小的部分用0填充
        loc_obs[:, loc_obs_x_min:loc_obs_x_max, loc_obs_y_min:loc_obs_y_max] = glb_obs[:, glb_obs_x_min:glb_obs_x_max,
                                                                               glb_obs_y_min:glb_obs_y_max]
        return loc_obs

    def gen_grid_pois(self, global_horizon):
        # 通道3：计算 (400, 400)
        poi_grid_point_vals = np.zeros([global_horizon, global_horizon], dtype=np.float32)
        poi_grid_point = {}
        # poi2cell_dict 对于是对于每个poi的点集，点集中的每个点代表着无人机可以在这里采集到这个poi数据
        for poi_id in self.poi2cell_dict:
            poi_grid_point[poi_id] = set()
            for cell_id in self.poi2cell_dict[poi_id]:
                cell_x = float(cell_id.split('_')[0])
                cell_y = float(cell_id.split('_')[1])
                grid_x = int((cell_x + 0.5) / self.grid_size * global_horizon)
                grid_y = int((cell_y + 0.5) / self.grid_size * global_horizon)
                grid_pos = str(grid_x) + '_' + str(grid_y)
                poi_grid_point[poi_id].add(grid_pos)
        poi_grid_point_pos = {}
        # 构造broadcast数组，用来扫描poi范围内的每个点
        for poi_id in poi_grid_point:
            poi_grid_point_pos[poi_id] = [[], []]
            for grid_pos in poi_grid_point[poi_id]:
                grid_x = int(grid_pos.split('_')[0])
                grid_y = int(grid_pos.split('_')[1])
                poi_grid_point_pos[poi_id][0].append(grid_x)
                poi_grid_point_pos[poi_id][1].append(grid_y)
        # 计算每个位置可采集的poi总和
        # poi2glb_obs_grid_dict[poi_id][0]与poi2glb_obs_grid_dict[poi_id][1]的每个对应位置构成一个poi_id范围内的grid
        for poi_id in poi_grid_point_pos:
            poi_grid_point_vals[poi_grid_point_pos[poi_id][0], poi_grid_point_pos[poi_id][1]] += \
                self.poi2coordxy_value_dict[poi_id]['value'] / self.poi_value_max
        return poi_grid_point_vals, poi_grid_point_pos


    def reset(self):
        # self.cur_step = 0
        # 初始poi值与实时poi值
        self.poi_last_value_array = copy.deepcopy(self.poi_init_vals)
        self.poi_cur_value_array = copy.deepcopy(self.poi_init_vals)
        # 计算评价指标
        self.final_poi_visit_time = np.zeros(self.poi_num, dtype=np.float32)
        self.final_total_hit = 0
        self.final_energy_consumption = 0


        # 对每个无人车以及车上的所有无人机
        for ugv in self.UGV_list:
            ugv.episode_log_info_dict = {}
            ugv.episode_log_info_dict['status'] = []
            # 当前路径记录
            ugv.episode_log_info_dict['pre_stop_id_list'] = []
            ugv.episode_log_info_dict['cur_stop_id_list'] = []
            ugv.cur_stop_id = ugv.pre_stop_id = self.UGV_init_stop_id_list[ugv.ugv_id]

            for uav_id in ugv.init_uav_list:
                uav = self.UAV_list[uav_id]
                uav.episode_log_info_dict = {}
                uav.episode_log_info_dict['pre_pos'] = []
                uav.episode_log_info_dict['final_pos'] = []
                uav.episode_log_info_dict['final_energy'] = []
                uav.episode_log_info_dict['final_energy_consumption'] = []
                uav.episode_log_info_dict['final_hit'] = []
                uav.episode_log_info_dict['final_hover'] = []
                uav.episode_log_info_dict['final_out_of_ugv'] = []
                uav.episode_log_info_dict['final_data_collection'] = []
                uav.episode_log_info_dict['final_poi_visit_time'] = []
                uav.episode_log_info_dict['final_collect_data_time'] = []
                uav.episode_log_info_dict['ugv_belong_to'] = []
                uav.final_pos = uav.pre_pos = self.stops_net_dict[ugv.cur_stop_id]['coordxy']
                uav.final_energy = self.uav_init_energy
                uav.final_energy_consumption = 0
                uav.final_hit = 0
                uav.final_hover = 0
                uav.charge_time = 0
                uav.needless_charge_time = 0
                uav.final_out_of_ugv = 0
                uav.final_data_collection = np.zeros(self.poi_num, dtype=np.float32)
                uav.final_poi_visit_time = np.zeros(self.poi_num, dtype=np.float32)
                uav.final_collect_data_time = 0
                uav.whether_flight_collect = False
                uav.stop_belong_to = ugv.cur_stop_id
                uav.ugv_belong_to = ugv.ugv_id
                uav.add_log_info()

        self.grid_cur_vals = copy.deepcopy(self.grid_init_vals)

        self.stop_around_poi_value = np.sum(self.poi_init_vals * self.stops_pois_AdjMatrix, axis=1)

        state = []
        for ugv_id in range(self.ugv_n):
            state.append(self.compute_state_ugv(ugv_id))
        state = np.concatenate(state, axis=0)
        return state

    def road_pos2pos(self, road_pos):
        start_road_node_coordxy = self.roads_net_dict[road_pos['start_road_node_id']]['coordxy']
        end_road_node_coordxy = self.roads_net_dict[road_pos['end_road_node_id']]['coordxy']
        pos_coordx = start_road_node_coordxy[0] * (1 - road_pos['progress']) + end_road_node_coordxy[0] * road_pos[
            'progress']
        pos_coordy = start_road_node_coordxy[1] * (1 - road_pos['progress']) + end_road_node_coordxy[1] * road_pos[
            'progress']
        pos = (pos_coordx, pos_coordy)
        return pos

    def check_whether_hit(self, start_pos, move_vector):
        whether_hit_flag = False
        move_dis = vector_length_counter(move_vector)
        check_point_num = int(move_dis / self.uav_move_check_hit_gap) + 1
        for check_point_id in range(check_point_num):
            check_point_move_dis = move_dis - check_point_id * self.uav_move_check_hit_gap
            check_point_pos = (start_pos[0] + move_vector[0] * check_point_move_dis / move_dis,
                               start_pos[1] + move_vector[1] * check_point_move_dis / move_dis)
            check_point_cell_id = self.pos2cell_id(check_point_pos)
            if check_point_cell_id not in self.uav_cellset:
                whether_hit_flag = True
                break
        return whether_hit_flag

    def pos2cell_id(self, pos):
        # [0-1000]_[0:1000]
        cell_x = int(pos[0] / self.coordx_max * self.grid_size)
        cell_y = int(pos[1] / self.coordy_max * self.grid_size)
        cell_id = str(cell_x) + '_' + str(cell_y)
        return cell_id

    def uav_move_and_collect_N(self, uav, action4UAV, uav_stop_id):
        # uav is alive (has energy)
        hit_flag = False
        data_collection = 0
        energy_consumption = 0

        if abs(uav.final_energy) > 1e-5:
            uav_move_dis_capability = min(uav.final_energy / self.move_energy_ratio, self.max_move_dist)
            origin_move_vector = (action4UAV[0], action4UAV[1])
            origin_move_dis = vector_length_counter(origin_move_vector)
            move_dis = min(uav_move_dis_capability, origin_move_dis)
            move_vector = (action4UAV[0] / origin_move_dis * move_dis, action4UAV[1] / origin_move_dis * move_dis)
            hit_flag = self.check_whether_hit(uav.final_pos, move_vector)
            if hit_flag:
                uav.final_hit += 1
                self.final_total_hit += 1
                # 不合法的动作，此步相当于悬停
                uav.final_energy -= self.hover_energy
                uav.final_energy_consumption += self.hover_energy
                self.final_energy_consumption += self.hover_energy
                energy_consumption += self.hover_energy
            else:
                # move and consume energy
                uav.pre_pos = uav.final_pos
                uav_final_pos_tmp = (uav.final_pos[0] + move_vector[0], uav.final_pos[1] + move_vector[1])
                uav.final_pos = uav_final_pos_tmp
                uav.final_energy -= move_dis * self.move_energy_ratio
                uav.final_energy_consumption += move_dis * self.move_energy_ratio
                self.final_energy_consumption += move_dis * self.move_energy_ratio
                energy_consumption += move_dis * self.move_energy_ratio

                if abs(uav.final_energy) < 1e-5:
                    uav.final_energy = 0

                whether_collect_data_flag = False
                uav_cell_id = self.pos2cell_id(uav.final_pos)

                assigned_poi_list = np.nonzero(self.stops_pois_AdjMatrix[uav_stop_id])[0]
                if uav_cell_id in self.uav_cell2poi_dict:
                    available_poi_list = self.uav_cell2poi_dict[uav_cell_id]

                    self.final_poi_visit_time[available_poi_list] += 1
                    # 采集uav所有可采集范围内的poi
                    for poi_id in available_poi_list:
                        if self.poi_cur_value_array[poi_id] > 0:
                            whether_collect_data_flag = True
                        if self.poi_cur_value_array[poi_id] < self.collect_speed_poi:
                            if poi_id in assigned_poi_list:
                                uav.final_data_collection[poi_id] += self.poi_cur_value_array[poi_id]
                                data_collection += self.poi_cur_value_array[poi_id]
                                uav.final_poi_visit_time[poi_id] += 1
                            self.poi_cur_value_array[poi_id] = 0
                        else:
                            if poi_id in assigned_poi_list:
                                uav.final_data_collection[poi_id] += self.collect_speed_poi
                                data_collection += self.collect_speed_poi
                                uav.final_poi_visit_time[poi_id] += 1
                            self.poi_cur_value_array[poi_id] -= self.collect_speed_poi

                if whether_collect_data_flag:
                    uav.final_collect_data_time += 1
                    uav.whether_flight_collect = True
        return hit_flag, data_collection, energy_consumption

    def uav_back_to_ugv(self, uav, ugv):
        uav.pre_pos = uav.final_pos
        uav.final_pos = self.stops_net_dict[ugv.cur_stop_id]['coordxy']
        uav.final_energy = self.uav_init_energy
        uav.stop_belong_to = ugv.cur_stop_id

    def uav_charge(self, uav):
        uav.charge_time += 1
        if uav.final_energy >= 0.5 * self.uav_init_energy:
            uav.needless_charge_time += 1
        uav.final_energy = self.uav_init_energy

    def action_scheduler(self, actions_list):

        uav_actions = []
        for uav_id in range(self.uav_n):
            single_uav_actions = [actions_list[uav_id][0]]
            single_uav_actions.append(actions_list[uav_id][-2:])
            uav_actions.append(single_uav_actions)
        return uav_actions

    # human design
    def compute_reward(self, data_collection, fairness, energy_consumption, hit_flag,
                       verbose=False):
        if energy_consumption > 0.0:
            reward_collect = (0.3 * fairness * data_collection) / (
                    energy_consumption + min_value)
        else:
            reward_collect = 0.0
        reward_collect_clipped = np.clip(reward_collect, 0, 5.0)
        flight_penalty = - 0.25 * hit_flag
        reward = reward_collect_clipped + flight_penalty
        return reward, {}

    def compute_metric(self,):
        dcr = np.sum(self.poi_init_vals - self.poi_cur_value_array) / np.sum(self.poi_init_vals)

        fairness = 0.0
        final_poi_visit_time = np.clip(self.final_poi_visit_time, 0, 2)
        square_of_sum = np.square(np.sum(final_poi_visit_time))
        sum_of_square = np.sum(np.square(final_poi_visit_time))
        if sum_of_square > 1e-5:
            fairness = square_of_sum / sum_of_square / final_poi_visit_time.shape[0]
        
        ecr = self.final_energy_consumption / ( self.uav_n * self.max_move_dist * self.max_step_num * self.move_energy_ratio)
        eff = 0.0
        if ecr > min_value:
            eff = fairness * dcr / ecr
        infos = {
            'dcr': dcr,
            'ecr': ecr,
            'fairness': fairness,
            'eff':  eff
        }
        return infos
    
    def step(self, uav_actions_list):
        self.poi_last_value_array = copy.deepcopy(self.poi_cur_value_array)
        reward_list = []
        data_collection_list = []
        energy_consumption_list = []
        hit_flag_list = []
        for uav_id, action4UAV in enumerate(uav_actions_list):
            uav = self.UAV_list[uav_id]
            ugv = self.UAV_UGV_MAP[uav_id]
            # flight branch
            hit_flag, data_collection, energy_consumption = self.uav_move_and_collect_N(uav,
                                                                                        action4UAV[UAV_FLIGHT] *
                                                                                        self.max_move_dist,
                                                                                        uav.stop_belong_to)

            uav_move_dis_capability = min(uav.final_energy / self.move_energy_ratio,
                                          self.max_move_dist)
            if uav_move_dis_capability < 1.0:
                self.uav_back_to_ugv(uav, ugv)
            uav.add_log_info()
            data_collection_list.append(data_collection)
            energy_consumption_list.append(energy_consumption)
            hit_flag_list.append(hit_flag)

        fairness_list = []
        for uav_id in range(self.uav_n):
            uav = self.UAV_list[uav_id]
            uav_cur_stop = uav.stop_belong_to
            uav_stop_mask = np.nonzero(self.stops_pois_AdjMatrix[uav_cur_stop])[0]
            poi_vis_time = self.final_poi_visit_time[uav_stop_mask]
            square_of_sum = np.square(np.sum(poi_vis_time))
            sum_of_square = np.sum(np.square(poi_vis_time))
            if sum_of_square > 1e-5:
                fairness = square_of_sum / sum_of_square / poi_vis_time.shape[0]
            else:
                fairness = 0.0
            fairness_list.append(fairness)

        infos = self.compute_metric()
    
        reward_components = dict()
        for uav_id in range(self.uav_n):
            reward, reward_component = self.compute_reward(data_collection_list[uav_id], fairness_list[uav_id],
                                                           energy_consumption_list[uav_id], hit_flag_list[uav_id])
            for key in reward_component:
                if key in reward_components:
                    reward_components[key].append(reward_component[key])
                else:
                    reward_components[key] = [reward_component[key]]
            reward_list.append(reward)

        for key in reward_components:
            reward_components[key] = np.mean(reward_components[key])
        infos['reward_components'] = reward_components

        poi_delta_value_array = self.poi_last_value_array - self.poi_cur_value_array
        for poi_id in np.nonzero(poi_delta_value_array)[0]:
            self.grid_cur_vals[
                self.poi_grid_pos[poi_id][0], self.poi_grid_pos[poi_id][1]] -= \
                poi_delta_value_array[poi_id] / self.poi_value_max

        obs = []
        for uav_id in range(self.uav_n):
            obs.append([self.compute_observations(uav_id)])
        obs = np.concatenate(obs)
        return obs, reward_list, infos

    
    def step_ugv(self, ugv_actions_list):
        move_dis_list = []
        for ugv_id, stop_id in enumerate(ugv_actions_list):
            ugv = self.UGV_list[ugv_id]
            pre_stop = ugv.cur_stop_id
            ugv.pre_stop_id = ugv.cur_stop_id
            target_stop_id = self.crucial_stops[stop_id]
            ugv.cur_stop_id = target_stop_id
            cur_stop = ugv.cur_stop_id
            move_dis = max(1, self.stops_net_SP_Matrix[pre_stop][cur_stop])
            move_dis_list.append(move_dis)
            ugv_cur_pos = self.stops_net_dict[cur_stop]['coordxy']
            ugv_dc = 0
            for uav_id in ugv.init_uav_list:
                uav = self.UAV_list[uav_id]
                if target_stop_id != pre_stop or dis_p2p(uav.final_pos,
                                                     ugv_cur_pos) > self.stop_poi_max_dis and self.mode == "train":
                    self.uav_back_to_ugv(uav, ugv)
                
                ugv_dc += np.sum(uav.final_data_collection[-1])
            ugv.pre_data_collect = ugv_dc


        self.stop_around_poi_value = np.sum(self.poi_cur_value_array * self.stops_pois_AdjMatrix, axis=1)

        state = []
        for ugv_id in range(self.ugv_n):
            state.append(self.compute_state_ugv(ugv_id))
        state = np.concatenate(state, axis=0)
        self.update_uav_poi_mask()

        obs = []
        for uav_id in range(self.uav_n):
            obs.append([self.compute_observations(uav_id)])
        obs = np.concatenate(obs)
        obs = obs.tolist()
        return state, obs, move_dis_list

    def update_uav_poi_mask(self):
        for uav_id in range(self.uav_n):
            uav = self.UAV_list[uav_id]
            uav_belong_stop = uav.stop_belong_to
            poi_id_list = np.nonzero(self.stops_pois_AdjMatrix[uav_belong_stop])[0]
            self.uav_poi_mask[uav_id] = 0
            for poi_id in poi_id_list:
                self.uav_poi_mask[uav_id][self.poi_grid_pos[poi_id][0],self.poi_grid_pos[poi_id][1]] = 1

    def get_global_ach_goal(self):
        global_achieved_goal = np.sum(self.stops_pois_AdjMatrix[self.crucial_stops] * self.poi_cur_value_array,
                                      axis=1) / self.poi_value_max
        return global_achieved_goal

    def get_ugv_dc(self, uav_actual_step_num):
        
        data_collection_list = []
        for ugv_id, ugv in enumerate(self.UGV_list):
            ugv_dc = 0
            for uav_id in ugv.init_uav_list:
                uav = self.UAV_list[uav_id]
                if len(uav.episode_log_info_dict['final_data_collection']) == 0:
                    dc = 0
                else:
                    dc = np.sum(uav.episode_log_info_dict['final_data_collection'][-1]) - np.sum(uav.episode_log_info_dict['final_data_collection'][-uav_actual_step_num])
                ugv_dc += dc
            data_collection_list.append(ugv_dc / self.poi_value_max / self.max_uav_n)
        return data_collection_list
    
    def compute_state_ugv(self, ugv_id):
        cur_stop_idx_in_crucial = []
        for ugv in self.UGV_list:
            cur_stop = ugv.cur_stop_id
            cur_stop_idx_in_crucial.append(self.crucial_stops_map[cur_stop])
        ugv_onehot_idx = np.eye(self.ugv_n)[ugv_id].tolist()
        common_obs = self.stop_around_poi_value[self.crucial_stops] / self.poi_value_max
        common_obs = common_obs.tolist()
        obs4ugv = common_obs + ugv_onehot_idx + cur_stop_idx_in_crucial
        ugv_cur_stop = self.UGV_list[ugv_id].cur_stop_id
        idx_in_crucial = self.crucial_stops_map[ugv_cur_stop]
        move_dis = 1.0 * self.crucial_stop_dis_Matrix[idx_in_crucial] / self.ugv_move_norm
        move_dis = move_dis.tolist()
        obs4ugv = obs4ugv + move_dis
        return [obs4ugv]

    def compute_ugv_reward(self, goal, ach_goal, ugv_dc, move_dis):
        delta_goal = np.abs(ach_goal - goal)
        goal_distance = delta_goal / self.global_init_goal
        similarity = (1 - goal_distance) ** 2
        similarity = np.mean(similarity)
        reward = 5 * ugv_dc * similarity / move_dis
        if similarity < 0.98:
            reward -= self.norm_her_reward
        reward = min(5, reward)
        return reward
