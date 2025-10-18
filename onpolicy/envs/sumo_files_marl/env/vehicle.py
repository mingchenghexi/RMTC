import sys

import numpy as np


class VehicleAgent:
    def __init__(self, vehicle_id, env, all_tls):
        """
        初始化车agent的基本信息。
        :param env: 仿真环境
        :param vehicle_id: 车辆ID
        """

        self._env = env
        self._vehicle_id = vehicle_id

        # print(vehicle_id, "vehicle_id")
        # print(self._env.sim.vehicle.getNextTLS(self._vehicle_id), "getNextTLS(self._vehicle_id)")
        if len(self._env.sim.vehicle.getNextTLS(self._vehicle_id)) > 0:
            self._tl_id = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][0]
        self._all_tls = all_tls


        self._last_action_time = 0  # 上次动作的时间
        self._last_position = self._env.sim.vehicle.getPosition(self._vehicle_id)  # 车辆的上次位置
        self._total_travel_time = 0  # 自上次动作以来的累计行驶时间
        self._vehicle_type = self._env.sim.vehicle.getTypeID(vehicle_id)
        # 动态获取当前车道信息
        self._current_lane = self._env.sim.vehicle.getRoadID(self._vehicle_id)
        self._previous_lane = ""
        # 获取车辆的当前速度traci.vehicle.getLaneIndex
        self._speed = self._env.sim.vehicle.getSpeed(self._vehicle_id)
        # self._current_position = self._env.sim.vehicle.getLanePosition(self._vehicle_id)
        # self._current_lane_length = self._env.sim.lane.getLength(self._current_lane)
        # 获取即将到来的交通信号灯索引
        if len(self._env.sim.vehicle.getNextTLS(self._vehicle_id)) > 0:
            self._current_phase_index = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][3]
        # # 获取即将到来的交叉路口接近的方向（N/S/E/W）
        # self._approaching_direction = self._env.sim.intersection.getApproachingDirection(self._current_lane)
        # 获取目的地交叉路口和目的地车道方向
        self._destination_road = self._env.sim.vehicle.getRoute(self._vehicle_id)[-1]
        self._destination_road_id = self._env.sim.vehicle.getRoute(self._vehicle_id)[-1]
        # self._destination_intersection = self._env.sim.edge.getFromJunction(self._destination_road_id)
        # print(self._destination_road_id)
        # print(self._env._crosses[self._destination_intersection].outgoing_sequence.index(self._destination_road_id.split('_')[0]))
        # sys.exit()
        
        self._do_action = False

        self._route = self._env.sim.simulation.findRoute(self._current_lane, self._destination_road_id).edges
        for edge_id in reversed(self._route):
            to_node = self._env.sim.edge.getFromJunction(edge_id)  # 返回该 edge 的出口 node（类型为 str）
            if to_node in self._env.all_tls:
                self._destination_intersection = to_node
                self._destination_road_id = edge_id
                break
        else:
            self._destination_intersection = None  # 没找到

        self._unava_index = []
        self._travel_time = -1
        self._obs_change = False
        self._wait_time = 0


    def update_vehicle_info(self):
        """更新车的信息：车速，当前交通信号灯状态，当前接近的交叉路口方向和当前车道"""
        # 动态获取当前车道信息
        self._current_lane = self._env.sim.vehicle.getCurrentLane(self._vehicle_id)

        # 获取车辆的当前速度
        self._speed = self._env.sim.vehicle.getSpeed(self._vehicle_id)

        # 获取即将到来的交通信号灯索引
        self._current_phase_index = self._env.sim.trafficlight.getUpcomingSignal(self._current_lane)

        # 获取即将到来的交叉路口接近的方向（N/S/E/W）
        self._approaching_direction = self._env.sim.intersection.getApproachingDirection(self._current_lane)

        # 获取目的地交叉路口和目的地车道方向
        self._destination_intersection, self._destination_lane_direction = self._get_destination_info()

    def _get_destination_info(self):
        """
        根据车辆当前所在车道和目标路径，动态获取目的地交叉路口和目的地车道方向编码。
        :return: 目的地交叉路口，目的地车道的方向编码
        """
        # 从仿真环境中获取目标交叉路口及其对应的车道方向
        destination_info = self._env.sim.getVehicleDestinationInfo(self._vehicle_id)

        # 假设返回的destination_info是一个元组：(目的地交叉路口, 目的地车道的方向编码)
        destination_intersection = destination_info[0]
        destination_lane_direction = destination_info[1]
        return destination_intersection, destination_lane_direction

    # def get_state(self):
    #     """获取车的状态，包括车速、即将到来的信号灯索引、接近的交叉路口方向和目的地信息"""
    #     state = {
    #                 'current_intersection': 0,  # 当前车道
    #                 'current_intersection_direction': 0,
    #                 'current_phase': 0,
    #                 'destination_intersection': 0,
    #                 'destination_intersection_direction': 0

    #                 }
    #     if self._obs_change == False:
    #         self._travel_time += 1
    #         # if self._env.sim.vehicle.isStopped(self._vehicle_id):

    #     else:
    #         self._travel_time = 0
    #         # self._wait_time = 0
    #     self._current_lane = self._env.sim.vehicle.getRoadID(self._vehicle_id)
    #     self._obs_change = False
    #     if self._current_lane not in self._all_tls and ':' not in self._current_lane:
    #         if len(self._env.sim.vehicle.getNextTLS(self._vehicle_id)) > 0:
    #             self._tl_id = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][0]
    #             if self._current_lane in self._env._crosses[self._tl_id].entering_sequence:                    
    #                 x = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][-1]
    #                 tmp = 0
    #                 if x in ['g', "G"]:
    #                     tmp = 1
    #                 elif x == 'r':
    #                     tmp = 0

    #                 state = {
    #                     'current_intersection': self._all_tls.index(self._tl_id),  # 当前车道
    #                     'current_lane': self._env._crosses[self._tl_id].entering_sequence_NWSE.index(
    #                         self._env._crosses[self._tl_id].entering_sequence.index(self._current_lane)),
    #                     # 'current_phase': tmp,
    #                     'destination_intersection': self._all_tls.index(self._destination_intersection),
    #                     # 'destination_lane': self._env.conversion_dict[nwse]
    #                     'destination_lane': self._env._crosses[self._destination_intersection].outgoing_sequence_NWSE.index(
    #                         self._env._crosses[self._destination_intersection].outgoing_sequence.index(
    #                             self._destination_road_id))
    #                 }
    #                 if self._previous_lane != self._current_lane:
    #                     self._obs_change = True
    #                     self._unava_index = []
    #                     self._previous_lane = self._current_lane
    #                 else:
    #                     self._wait_time = self._env.sim.vehicle.getWaitingTime(self._vehicle_id)
    #         else:
    #             if self._current_lane == self._destination_road_id:
    #                 state = {
    #                     'current_intersection': self._all_tls.index(self._destination_intersection),
    #                     # 'current_intersection':self._env._all_junction.index(self._destination_intersection),  # 当前车道
    #                     # 'current_lane': self._env.conversion_dict[nwse],
    #                     'current_lane': self._env._crosses[self._destination_intersection].outgoing_sequence_NWSE.index(
    #                         self._env._crosses[self._destination_intersection].outgoing_sequence.index(
    #                             self._destination_road_id)),
    #                     # 'current_phase': -1,
    #                     'destination_intersection': self._all_tls.index(self._destination_intersection),
    #                     # 'destination_lane': self._env.conversion_dict[nwse]
    #                     'destination_lane': self._env._crosses[
    #                         self._destination_intersection].outgoing_sequence_NWSE.index(
    #                         self._env._crosses[self._destination_intersection].outgoing_sequence.index(
    #                             self._destination_road_id))
    #                 }
    #                 if self._previous_lane != self._current_lane:
    #                     self._obs_change = True
    #                     self._unava_index = []
    #                     self._previous_lane = self._current_lane
    #                 else:
    #                     self._wait_time = self._env.sim.vehicle.getWaitingTime(self._vehicle_id)
    #             # x = self._env.sim.vehicle.getNextTLS(self._vehicle_id)
    #             # nwse = self._env._crosses[self._destination_out_intersection].outgoing_sequence_NWSE.index(
    #             #     self._env._crosses[self._destination_out_intersection].outgoing_sequence.index(
    #             #         self._destination_road_id))


    #     return state

    
    def get_state(self):
        """获取车的状态，包括车速、即将到来的信号灯索引、接近的交叉路口方向和目的地信息"""
        state = {
                    'current_intersection': 0,  # 当前车道
                    'current_intersection_direction': 0,
                    'current_phase': 0,
                    'destination_intersection': 0,
                    'destination_intersection_direction': 0

                    }
        if self._obs_change == False:
            self._travel_time += 1
            # if self._env.sim.vehicle.isStopped(self._vehicle_id):

        else:
            self._travel_time = 0
            # self._wait_time = 0
        if self._destination_intersection == None:
            return state
        self._current_lane = self._env.sim.vehicle.getRoadID(self._vehicle_id)
        self._obs_change = False
        if self._current_lane not in self._all_tls and ':' not in self._current_lane:
            if len(self._env.sim.vehicle.getNextTLS(self._vehicle_id)) > 0:
                self._tl_id = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][0]
                if self._current_lane in self._env._crosses[self._tl_id].entering_sequence:                    
                    x = self._env.sim.vehicle.getNextTLS(self._vehicle_id)[0][-1]
                    tmp = 0
                    if x in ['g', "G"]:
                        tmp = 1
                    elif x == 'r':
                        tmp = 0

                    state = {
                        'current_intersection': self._all_tls.index(self._tl_id),  # 当前车道
                        'current_lane': self._env._crosses[self._tl_id].entering_sequence_NWSE.index(
                            self._env._crosses[self._tl_id].entering_sequence.index(self._current_lane)),
                        # 'current_phase': tmp,
                        'destination_intersection': self._all_tls.index(self._destination_intersection),
                        # 'destination_lane': self._env.conversion_dict[nwse]
                        'destination_lane': self._env._crosses[self._destination_intersection].outgoing_sequence_NWSE.index(
                            self._env._crosses[self._destination_intersection].outgoing_sequence.index(
                                self._destination_road_id))
                    }
                    if self._previous_lane != self._current_lane:
                        self._obs_change = True
                        self._unava_index = []
                        self._previous_lane = self._current_lane
                    else:
                        self._wait_time = self._env.sim.vehicle.getWaitingTime(self._vehicle_id)
            else:
                if self._current_lane == self._destination_road_id:
                    state = {
                        'current_intersection': self._all_tls.index(self._destination_intersection),
                        # 'current_intersection':self._env._all_junction.index(self._destination_intersection),  # 当前车道
                        # 'current_lane': self._env.conversion_dict[nwse],
                        'current_lane': self._env._crosses[self._destination_intersection].outgoing_sequence_NWSE.index(
                            self._env._crosses[self._destination_intersection].outgoing_sequence.index(
                                self._destination_road_id)),
                        # 'current_phase': -1,
                        'destination_intersection': self._all_tls.index(self._destination_intersection),
                        # 'destination_lane': self._env.conversion_dict[nwse]
                        'destination_lane': self._env._crosses[
                            self._destination_intersection].outgoing_sequence_NWSE.index(
                            self._env._crosses[self._destination_intersection].outgoing_sequence.index(
                                self._destination_road_id))
                    }
                    if self._previous_lane != self._current_lane:
                        self._obs_change = True
                        self._unava_index = []
                        self._previous_lane = self._current_lane
                    else:
                        self._wait_time = self._env.sim.vehicle.getWaitingTime(self._vehicle_id)
                # x = self._env.sim.vehicle.getNextTLS(self._vehicle_id)
                # nwse = self._env._crosses[self._destination_out_intersection].outgoing_sequence_NWSE.index(
                #     self._env._crosses[self._destination_out_intersection].outgoing_sequence.index(
                #         self._destination_road_id))


        return state  


    def get_reward(self, reward_type):
        """计算奖励：自上次动作以来的累计行驶时间"""
        # current_position = self._env.sim.vehicle.getPosition(self._vehicle_id)
        # # 计算自上次动作以来的行驶时间
        # time_elapsed = self._env.sim.getTimeStep() - self._last_action_time
        # # 计算自上次动作以来的行驶距离
        # distance_travelled = np.linalg.norm(np.array(current_position) - np.array(self._last_position))
        # # 累积行驶时间（基于时间和距离，适当缩放）
        # self._total_travel_time += time_elapsed
        #
        # # 更新上次动作的时间和位置
        # self._last_action_time = self._env.sim.getTimeStep()
        # self._last_position = current_position
        #
        # # 奖励可以基于行驶时间和距离的函数
        # reward = self._total_travel_time
        reward = {}
        if 'queue_len' in reward_type:
            reward['queue_len'] = 0
        if 'car_number' in reward_type:
            reward['car_number'] = 0
        if 'wait_time' in reward_type:
            reward['wait_time'] = -self._wait_time / 500
        if 'delay_time' in reward_type:
            reward['delay_time'] = 0
        if 'arrive_des' in reward_type:
            if self._current_lane == self._destination_road_id:
                reward['arrive_des'] = 1
            else:
                reward['arrive_des'] = 0
        if 'travel_time' in reward_type:
            reward['travel_time'] = -self._travel_time / 1000

        return reward

    def take_action(self, action):
        """根据选择的动作（左转、右转、直行）执行相应操作"""
        if not self._do_action:
            # if self._vehicle_id == '24':
            #     print(self._env.sim.vehicle.getRoadID(self._vehicle_id))
            self._env.sim.vehicle.setRoute(self._vehicle_id, action)
            self._route = action
            self._do_action = True
            # print(self._vehicle_id,action,self._env.sim.vehicle.getRoute(self._vehicle_id))

    def get_possible_actions(self):
        """获取可能的动作（左转、右转、直行）"""
        # 基于当前车道和交通情况，判断哪些动作是有效的
        possible_actions = ['left', 'right', 'straight']
        return possible_actions


