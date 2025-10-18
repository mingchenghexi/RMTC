#!/usr/bin/env python3
# encoding: utf-8

import copy
import os
import subprocess
import sys
import time
import math
import xml.etree.ElementTree
import numpy as np
import libsumo
import traci
import sumolib
from sumolib import checkBinary
from onpolicy.envs.sumo_files_marl.env.intersection import Intersection
from onpolicy.envs.sumo_files_marl.env.vehicle import VehicleAgent
# from intersection import Intersection
# from vehicle import VehicleAgent
# from CoEMV.onpolicy.envs.sumo_files_marl.env.Graph import HeterogeneousGraph
from torch_geometric.data import HeteroData
import torch
from sumolib.net import readNet
import pandas as pd

class TSCSimulator:
    def __init__(self,args, config, port, not_default=True, output_path=None):
        self.not_default = not_default
        self.args = args
        self.port = port
        self.name = config.get("name")
        self.seed = config.get('seed', 777)
        self.agent = config.get('agent')
        self.is_libsumo = config.get('is_libsumo', True)
        self._yellow_duration = config.get("yellow_duration")
        self.iter_duration = config.get("iter_duration")
        # self.init_data(is_record, record_stats, output_path)
        self.is_record = config.get("is_record")
        self.config = config
        self.use_pressure = False
        self.use_gat = False
        # self.num_vehicle_agents = 600
        # self.num_emv_agents = 1
        self.num_vehicle_agents = args.num_vehicle_agents
        self.num_emv_agents = args.num_emv_agents
        if output_path is not None:
            self.output_path = output_path
        else:
            self.output_path = config.get("output_path")

        self.reward_type = config.get("reward_type")
        self.reward_type_ve = config.get("reward_type_ve")
        self.reward_type_emv = config.get("reward_type_emv")
        self.is_neighbor_reward = config.get("is_neighbor_reward", False)
        self.cfg = config
        self.step_num = config.get("step_num", 1)
        self.step_tl = 0
        self.p = config.get("p", 1)  # for trip.xml output dir
        self._current_time = 1
        self._init_sim(config.get("sumocfg_file"), self.seed,
                       self.config.get("episode_length_time"), config.get("gui"))
        self.all_tls = list(self.sim.trafficlight.getIDList())
        self.infastructure = self._infastructure_extraction1(config.get("sumocfg_file"))
        # delete invalid tl part 1
        rm_num = 0
        ora_num = len(self.all_tls)
        for i in self.infastructure:
            if self.infastructure[i]['entering_lanes_pos'] == []:
                rm_num += 1
                self.all_tls.remove(i)
                print("no infastrurct: {}".format(i))
        for rtl in ['1673431902', '8996', '9153', '9531', '9884', '7223736528']:
            if rtl in self.all_tls:
                rm_num += 1
                self.all_tls.remove(rtl)
        rm_tl = []
        # print(self.sim.junction.getIDList())
        self._all_junction = list(self.sim.junction.getIDList()[192:])
        self.conversion_dict = {0: 2, 1: 3, 2: 0, 3: 1}
        self._crosses = {}
        self.state_key = config['state_key']
        self.ve_state_key = config['ve_state_key']
        self.all_reward = {}
        self.all_ve_reward = {}
        self.all_emv_reward = {}
        self.tl_phase_index = []
        for tl in self.all_tls:
            self._crosses[tl] = Intersection(tl, self.infastructure[tl],
                                             self, self.state_key, self.not_default)
            tl_ava = self._crosses[tl].get_tl_ava()
            if not tl_ava:
                rm_tl.append(tl)
                rm_num += 1
                print("Not ava: {}".format(tl))
        # delete invalid tl part 2
        for tl in rm_tl:
            del self._crosses[tl]
            self.all_tls.remove(tl)
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl].update_timestep()
            self.tl_phase_index.append(self._crosses[tl].get_phase_index())
        for ve_id in range(self.num_vehicle_agents):
            self.all_ve_reward[str(ve_id)] = {k: 0 for k in self.reward_type_ve}
        for emv_id in range(self.num_emv_agents):
            self.all_ve_reward["emergency_"+str(emv_id)] = {k: 0 for k in self.reward_type_emv}

        print("Remove {} tl, percent: {}".format(rm_num, rm_num / ora_num))

        self.is_adjacency_remove = config.get("is_adjacency_remove", True)
        self.adjacency_top_k = config.get("adjacency_top_k", 5)

        self.infastructure = self._infastructure_extraction2(config.get("sumocfg_file"),
                                                             self.infastructure,
                                                             config.get("is_dis", False))

        self.nextEdges = self._find_adjacent_edge(config.get("sumocfg_file"))
        self.action_type = config.get('action_type')

        self.vehicle_info = {}

        
        self.vehicle_agent = {}
        self.emv_agent = {}
        self.all_ves = []
        self.all_emv = []
        self.vehicle_num_agents = 0
        self.emv_num_agents = 0
        self.hetero_graph = HeteroData()
        self.action_space = config.get("vehicle_num_actions")
        self.vehicle_depart_time = {}
        self.vehicle_arrival_time = {}
        self.vehicle_travel_time = {}
        # 用于缓存信号灯抢占切换阶段状态
        self.emv_signal_buffer = {}
        self.data = {
        "time": [], 
        "vehicle_id": [],
        "speed": [],
        "lane_id": [],
        "signal_color":[],
        "phase":[]
        }   
        self.terminate()


    def _init_sim(self, sumocfg_file, seed, episode_length_time, gui=False):
        self.episode_length_time = episode_length_time
        if gui:
            app = 'sumo-gui'
        else:
            app = 'sumo'

        # if sumocfg_file.split("/")[-1] in ['grid4x4.sumocfg', 'arterial4x4.sumocfg']:
        # route_name = "fenglin_y2z_t"
        route_name = sumocfg_file.split("/")[-1][:-8]   #fenglin
        net_file = "/".join(sumocfg_file.split("/")[:-1] + [route_name + '.net.xml'])
        self.net = readNet(net_file)
        #     route = "/".join(sumocfg_file.split("/")[:-1] + [route_name])
        #     command = [
        #         checkBinary(app), '-c', net, '-r', route + '_' + str(self.step_num) + '.trip.xml']
        #     command += ['-a', "/".join(sumocfg_file.split("/")[:-1] + ['e1.add.xml']) + ", " +
        #                 "/".join(sumocfg_file.split("/")[:-1] + ['e2.add.xml'])]
        # else:
        #     command = [checkBinary(app), '-c', sumocfg_file]
        command = [checkBinary(app), '-c', sumocfg_file]
        # command += ['--seed',str(seed)]
        command += ['--random']
        # command += ['--remote-port', str(self.port)]
        command += ['--no-step-log', 'True']
        if self.name != 'real_net':
            command += ['--time-to-teleport',
                        '600']  # long teleport for safety
        else:
            command += ['--time-to-teleport', '300']
        command += ['--no-warnings', 'True']
        command += ['--duration-log.disable', 'True']
        # collect trip info if necessary
        if self.is_record:
            if not os.path.exists(self.output_path):
                try:
                    os.mkdir(self.output_path)
                except:
                    pass
            command += ['--tripinfo-output',
                        self.output_path + ('%s_%s_%s_trip.xml' % (
                            self.name, sumocfg_file.split("/")[-1], self.p)),
                        '--tripinfo-output.write-unfinished']
        # subprocess.Popen(command)
        if self.is_libsumo:
            libsumo.start(command)
            self.sim = libsumo
        else:
            command += ['--remote-port', str(self.port)]
            subprocess.Popen(command)
            time.sleep(5)
            self.sim = traci.connect(port=self.port, numRetries=1000)
        self.step_num += 1
        self.p += 1
        # wait 2s to establish the traci server
        # time.sleep(5)
        # self.sim = traci.connect(port=self.port, numRetries=1000)

        # if sumocfg_file.split("/")[-1] == 'ingolstadt21.sumocfg':
        #     self._current_time = 57600 - 1
        #     self.episode_length_time += self._current_time
        #     self.sim.simulationStep(self._current_time)
        # elif sumocfg_file.split("/")[-1] == 'cologne8.sumocfg':
        #     self._current_time = 25200 - 1
        #     self.episode_length_time += self._current_time
        #     self.sim.simulationStep(self._current_time)
        # self.sim = traci.start(command)

    def terminate(self):
        self.sim.close()



    def _do_action_vehicle(self, action):
        action_ve = action[:self.num_vehicle_agents]
        action_emv = action[self.num_vehicle_agents:]
        for i, a in enumerate(action_ve):
            if str(i) in self.all_ves:
                self.vehicle_agent[str(i)]._do_action = False
                if self.vehicle_agent[str(i)]._obs_change == True:
                    self._determine_next_edge(self.nextEdges, str(i), a)
        for i, a in enumerate(action_emv):
            if 'emergency_'+str(i) in self.all_emv:
                self.emv_agent['emergency_'+str(i)]._do_action = False
                if self.emv_agent['emergency_'+str(i)]._obs_change == True:
                    self._determine_next_edge(self.nextEdges, 'emergency_'+str(i), a, "emergency")
        self.sim.simulationStep()

    def do_action_emv(self):
        emv_phase_changes = {}  # {tl_id: [(phase_index, duration)]}
        default_duration = 1  # 默认推进时长为1秒

        # === 第一阶段：识别黄灯缓存，或检测新的EMV ===
        for tl in self.all_tls:
            # --- Step 1: 若在黄灯等待期间，检查是否要切换到绿灯 ---
            if tl in self.emv_signal_buffer:
                target_phase, remaining_yellow = self.emv_signal_buffer[tl]
                if remaining_yellow <= 1:
                    emv_phase_changes[tl] = [(target_phase, 10)]  # 切绿灯
                    del self.emv_signal_buffer[tl]
                else:
                    self.emv_signal_buffer[tl] = (target_phase, remaining_yellow - 1)
                continue  # 本轮已处理

            # --- Step 2: 检测是否有紧急车辆 ---
            have_emv = False
            emv_id = None
            emv_lane = None
            remaining = self.sim.trafficlight.getNextSwitch(tl) - self._current_time

            for lane in self._crosses[tl]._incoming_lanes:

                for veh in self.sim.lane.getLastStepVehicleIDs(lane):
                    if self.sim.vehicle.getTypeID(veh) == "emergency":
                        have_emv = True
                        emv_id = veh
                        emv_lane = lane
                        break
                if have_emv:
                    break

            if have_emv:
                speed = self.sim.vehicle.getSpeed(emv_id)
                next_tls = self.sim.vehicle.getNextTLS(emv_id)
                if speed >= 15 and next_tls and next_tls[0][2] < 50:
                    tl_id = next_tls[0][0]
                    dist = next_tls[0][2]
                    current_phase = self.sim.trafficlight.getPhase(tl_id)

                    # --- 若当前相位不是紧急车通行方向，则尝试切黄灯 ---
                    if not self._is_emv_direction(tl_id, current_phase, emv_lane):
                        yellow_phase = current_phase + 1
                        target_phase = self._get_emv_green_phase(tl_id, emv_lane)
                        if current_phase % 2 == 0:
                            emv_phase_changes[tl_id] = [(yellow_phase, 5)]  # 切黄灯
                            self.emv_signal_buffer[tl_id] = (target_phase, 5)
                        else:
                            if tl_id not in self.emv_signal_buffer:
                                self.emv_signal_buffer[tl_id] = (target_phase, 2)
                    else:
                        # 当前相位允许EMV通行，则延长绿灯
                        extend_time = (dist / speed) + 3  # 多等3秒缓冲
                        if extend_time > remaining:
                            emv_phase_changes[tl_id] = [(current_phase, extend_time)]
            else:
                # 没有EMV，不做操作，由仿真推进统一控制
                pass

        # === 第二阶段：应用相位变更 ===
        for tl_id, phase_list in emv_phase_changes.items():
            for phase_index, duration in phase_list:
                self._crosses[tl_id].set_phase_by_index(phase_index, duration)

        # === 第三阶段：推进仿真 ===
        self._current_time += 1
        # self.sim.simulationStep(self._current_time)

    def _is_emv_direction(self, tl_id, phase_index, emv_lanes):
        """
        判断当前相位是否允许EMV方向通行
        """
        # 获取信号灯的当前控制逻辑
        logic = self.sim.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[1]
        phase = logic.getPhases()[phase_index]


        controlled_lanes = self.sim.trafficlight.getControlledLanes(tl_id)

        for idx, light in enumerate(phase.state):
            if light == "G" or light == 's':
                lane = controlled_lanes[idx]
                if lane in emv_lanes:
                    return True
        return False

    def _get_emv_green_phase(self, tl_id, emv_lanes):
        """
        获取使EMV通行的信号灯相位索引
        """
        logic = self.sim.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        controlled_lanes = self.sim.trafficlight.getControlledLanes(tl_id)

        for i, phase in enumerate(logic.getPhases()):
            for idx, light in enumerate(phase.state):
                if light == "G":
                    lane = controlled_lanes[idx]
                    if lane in emv_lanes:
                        return i
        # 没有找到就默认返回第0相位（需谨慎）
        return 0

    def _do_action_tls(self, action):
        if self.action_type == 'select_phase':
            # for tl, a in action.items():
            #     assert a / 2 not in self._crosses[tl].unava_index, "{}-{}-{}".format(tl, a,
            #                                                                          self._crosses[tl].unava_index)
            #     current_phase_index = self._crosses[tl].getCurrentPhaseIndex()
            #     if a == current_phase_index:
            #         continue
            #     else:
            #         yellow_phase = current_phase_index + 1
            #         self._crosses[tl].set_phase_by_index(yellow_phase, self._yellow_duration)
            # self._current_time += self._yellow_duration
            # self.sim.simulationStep(self._current_time)
            # self.sim.simulationStep()
            # for tl, a in action.items():
            #     self._crosses[tl].set_phase_by_index(a, self.iter_duration)
            # self._current_time += self.iter_duration
            # self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                assert a / 2 not in self._crosses[tl].unava_index, "{}-{}-{}".format(tl, a,
                                                                                     self._crosses[tl].unava_index)
                self._crosses[tl].set_phase_by_index(a, self.iter_duration)
            # self._current_time += 1
            # # self._current_time += 1
            # self.sim.simulationStep()


        elif self.action_type == 'change':
            for tl, a in action.items():
                if a:
                    current_phase_index = self._crosses[tl].getCurrentPhaseIndex()
                    yellow_phase = current_phase_index + 1
                    self._crosses[tl].set_phase_by_index(yellow_phase, self._yellow_duration)
            self._current_time += self._yellow_duration
            # self._current_time += 1
            self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                if a:
                    next_phase = self._crosses[tl].getCurrentPhaseIndex() + 2
                    if next_phase > self._crosses[tl].green_phases[-1]:
                        next_phase = 0
                else:
                    next_phase = self._crosses[tl].getCurrentPhaseIndex()
                self._crosses[tl].set_phase_by_index(next_phase, self.iter_duration)
            self._current_time += self.iter_duration
            # self._current_time += 1
            self.sim.simulationStep(self._current_time)

        elif self.action_type == 'generate':
            for tl, a in action.items():
                current_phase_str = self._crosses[tl].getCurrentPhase()
                if a != current_phase_str:
                    yellow = self._crosses[tl].getCurrentPhaseYellow(current_phase_str)
                    self._crosses[tl].set_phase(yellow, self._yellow_duration)
            self._current_time += self._yellow_duration
            # self._current_time += 1
            self.sim.simulationStep(self._current_time)

            for tl, a in action.items():
                self._crosses[tl].set_phase(a, self.iter_duration)
            self.sim.simulationStep(self._current_time)
        else:
            raise NotImplemented

    def _get_reward_ve(self):
        list_reward = {ve: self.vehicle_agent[ve].get_reward(self.reward_type_ve) for ve in self.all_ves}

        return list_reward

    def _get_reward_emv(self):
        list_reward = {ve: self.emv_agent[ve].get_reward(self.reward_type_emv) for ve in self.all_emv}

        return list_reward
    def _get_reward(self):

        list_reward = {tl: self._crosses[tl].get_reward(self.reward_type, self.step_tl) for tl in self.all_tls}

        return list_reward

    def get_neighbors_of_traffic_lights(tl_id):
        """
        获取给定信号灯的邻居信号灯ID
        :param tl_id: 信号灯的ID
        :return: 邻居信号灯的ID列表
        """
        # 获取该信号灯所处的交叉口
        junction_id = traci.trafficlight.getLinkedJunction(tl_id)

        # 获取该交叉口的所有信号灯
        all_tl_ids = traci.trafficlight.getIDList()

        # 获取邻近信号灯的ID
        neighbors = []
        for other_tl_id in all_tl_ids:
            if other_tl_id != tl_id:  # 排除当前信号灯
                # 获取邻近信号灯的交叉口ID
                other_junction_id = traci.trafficlight.getLinkedJunction(other_tl_id)
                # 判断是否属于相邻的交叉口
                if other_junction_id != junction_id:
                    neighbors.append(other_tl_id)

        return neighbors


    def step(self, action_tl, action_ve):

        # all_obs, ve_reward, emv_reward, done, all_reward, self.hetero_graph = self.step_ve(action_ve)
        # print(self.sim.simulation.getMinExpectedNumber(),"traci.simulation.getMinExpectedNumber()")
        if self.step_tl % 15==0:
            self._do_action_tls(action_tl)
        self._do_action_vehicle(action_ve)
        # if "emergency_0" in self.sim.vehicle.getIDList():
        #     print(self.sim.vehicle.getRoadID("emergency_0"))
        self.log()
        self._current_time += 1
        self.step_tl+=1

        new_ves,new_emv = self.get_vehicle_agent()
        for tl in self.all_tls:
            self._crosses[tl].update_timestep()
        done = False

        obs, ve_obs, emv_obs, done_ve, done_emv, obs_change_ve, obs_change_emv = self._get_state()
        # if (self.step_tl + 1) % 15 == 0:
        obs = self.batch(obs, self.state_key, self.all_tls)

        all_ve_obs = np.full((self.num_vehicle_agents, len(self.ve_state_key)), 0, dtype=int)
        all_emv_obs = np.full((self.num_emv_agents, len(self.ve_state_key)), 0, dtype=int)
        ve_reward = {}
        emv_reward = {}
        reward_ = self._get_reward()

        if len(ve_obs) > 0:
            all_ve_obs = self.ve_batch(ve_obs, self.ve_state_key, self.all_ves)
            ve_reward = self._get_reward_ve()
            for ve, v in ve_reward.items():
                if obs_change_ve[ve] == True:
                    for k, r in v.items():
                        self.all_ve_reward[ve][k] += r
        if len(emv_obs) > 0:
            all_emv_obs = self.emv_batch(emv_obs, self.ve_state_key, self.all_emv)
            emv_reward = self._get_reward_emv()
            for emv, v in emv_reward.items():
                if obs_change_emv[emv] == True:
                    for k, r in v.items():
                        self.all_ve_reward[emv][k] += r
        if len(ve_obs) > 0 or len(emv_obs) > 0:
            all_obs = np.vstack((all_ve_obs, all_emv_obs))
        else:
            all_obs = np.full((self.num_vehicle_agents + self.num_emv_agents, len(self.ve_state_key)), 0)

        self.build_hetero_graph(obs, all_ve_obs, all_emv_obs)

        # multi-agent reward sum with neighbor
        if self.is_neighbor_reward:
            reward = {}
            for tl in self.all_tls:
                reward[tl] = {}
                for k in reward_[tl]:
                    reward[tl][k] = reward_[tl][k]
                    count = 0
                    tmp = 0
                    for nei in self._crosses[tl].nearset_inter[0][0]:
                        if nei != -1:
                            count += 1
                            tmp += reward_[self.all_tls[nei]][k]
                    if count > 0:
                        # reward[tl][k] += tmp / count
                        reward[tl][k] += tmp / 4

        else:
            reward = reward_
        if self.step_tl % 15 == 0:
            for tl, v in reward.items():
                for k, r in v.items():
                    self.all_reward[tl][k] += r
        # if self._current_time >= self.episode_length_time:
        #     done = True
        #     print(self.sim.vehicle.getIDList(),"now vehicle")
        #     self.terminate()
        # if len(self.sim.vehicle.getIDList()) <= 5  and self._current_time >= 5000:
        self.update_vehicle_travel_time()
        # print(self.sim.vehicle.getIDList())
        if len(self.sim.vehicle.getIDList()) == 0 or self._current_time >= 50000:
            done = True
            print(len(self.sim.vehicle.getIDList()), "now vehicle")
            # pd.DataFrame(self.data).to_csv("/home/PhD/ChenM/experiment/CoEMV/onpolicy/envs/sumo_files_marl/cologne8_3.csv", index=False)
            output_dir = "/home/PhD/ChenM/experiment/CoEMV/onpolicy/envs/sumo_files_marl"
    
            try:
                import datetime
                import os

                # 获取当前时间并格式化为字符串
                time_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

                # 组合文件名
                main_path = os.path.join(output_dir, f"trajectory_data_{time_str}.csv")

                # 分离相位数据单独保存
                if "phase" in self.data:
                    import json
                    phase_data = self.data.pop("phase")  # 取出相位数据
                    
                    # 保存为JSON文件（适合字典结构）
                    # phase_path = f"{output_dir}/phase_data.json"
                    phase_path = os.path.join(output_dir, f"phase_data_{time_str}.csv")
                    with open(phase_path, "a") as f:
                        json.dump({
                            "sim_time": self._current_time,
                            "phases": phase_data
                        }, f)
                        f.write("\n")  # 换行分隔记录
                
                # 保存剩余数据到主CSV文件
                if self.data:  # 确保有数据再保存
                    main_path = os.path.join(output_dir, f"trajectory_data_{time_str}.csv")
                    pd.DataFrame(self.data).to_csv(
                        main_path,
                        mode='a',  # 追加模式
                        header=not os.path.exists(main_path),  # 如果文件不存在则写表头
                        index=False
                    )
                    
                print(f"数据已保存到 {output_dir}")
                
            except Exception as e:
                print(f"数据保存失败: {str(e)}")
                # 可以添加失败数据备份逻辑
                with open(f"{output_dir}/error_backup.txt", "a") as f:
                    f.write(f"{self._current_time}: {str(self.data)}\n")
            self.terminate()
        # if "emergency_7" in self.sim.vehicle.getIDList():
        #     # 获取车辆位置 (x, y 坐标)
        #     position = self.sim.vehicle.getPosition("emergency_7")
        #     print(f"Emergency vehicle 7 current position: {position}")
        #
        #     # 获取即将到达的红绿灯状态
        #     traffic_light = self.sim.vehicle.getNextTLS("emergency_7")
        #     # if traffic_light:
        #     #     traffic_light_status = self.sim.trafficlight.getPhaseName(traffic_light[0])
        #     print(f"Next traffic light status for emergency vehicle 7: {traffic_light[0]}")
        emv = [int(agent_id.split('_')[-1]) for agent_id in self.all_emv]
        ve = list(map(int, self.all_ves))

        return obs, reward, done, done_ve, done_emv, self.all_reward, all_obs, ve_reward, emv_reward, done, self.all_ve_reward, self.hetero_graph, obs_change_ve, obs_change_emv, new_ves,new_emv, emv, ve, self.vehicle_travel_time

    def update_vehicle_travel_time(self):
        current_time = self.sim.simulation.getTime()

        # 获取当前在网车辆
        vehicle_ids = self.sim.vehicle.getIDList()

        # 记录新进入车辆的时间
        for veh_id in vehicle_ids:
            if veh_id not in self.vehicle_depart_time:
                self.vehicle_depart_time[veh_id] = current_time

        # 记录已经离开的车辆的到达时间
        arrived_vehicles = self.sim.simulation.getArrivedIDList()
        for veh_id in arrived_vehicles:
            self.vehicle_arrival_time[veh_id] = current_time
            depart_time = self.vehicle_depart_time.get(veh_id, None)
            if depart_time is not None:
                self.vehicle_travel_time[veh_id] = current_time - depart_time



    def default_step(self):
        done = False
        self._current_time += 1
        self.sim.simulationStep(self._current_time)
        for tl in self.all_tls:
            self._crosses[tl].update_timestep()
        reward = self._get_reward()
        for tl, v in reward.items():
            for k, r in v.items():
                self.all_reward[tl][k] += r
        if self._current_time >= self.episode_length_time:
            done = True
            self.terminate()

        return done, self.all_reward

    def reset(self):
        """have to terminate previous sim before calling reset"""
        self._current_time = 0
        self.step_tl = 0
        self._init_sim(self.cfg.get("sumocfg_file"), self.seed, self.config.get("episode_length_time"),
                       self.cfg.get("gui"))
        self.step_num = self.step_num % 1400 + 1
        # self.infastructure = self._infastructure_extraction1(self.cfg.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        self.all_ve_reward = {}
        self.all_emv_reward = {}
        self.vehicle_depart_time = {}
        self.vehicle_arrival_time = {}
        self.vehicle_travel_time = {}


        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key, self.not_default)
            self._crosses[tl].update_timestep()

        for ve_id in range(self.num_vehicle_agents):
            self.all_ve_reward[str(ve_id)] = {k: 0 for k in self.reward_type_ve}
        for emv_id in range(self.num_emv_agents):
            self.all_ve_reward["emergency_" + str(emv_id)] = {k: 0 for k in self.reward_type_emv}

        self.vehicle_agent = {}
        self.emv_agent = {}
        self.all_ves = []
        self.all_emv = []
        self.vehicle_num_agents = 0
        self.emv_num_agents = 0
        self.hetero_graph = HeteroData()
        obs, ve_obs, emv_obs, done_ve, done_emv, obs_change_ve, obs_change_emv = self._get_state()
        obs = self.batch(obs, self.state_key, self.all_tls)
        all_ve_obs = np.full((self.num_vehicle_agents, len(self.ve_state_key)), 0, dtype=int)
        all_emv_obs = np.full((self.num_emv_agents, len(self.ve_state_key)), 0, dtype=int)
        self.build_hetero_graph(obs, all_ve_obs, all_emv_obs)
        self.data = {
            "time": [], 
            "vehicle_id": [],
            "speed": [],
            "lane_id": [],
            "signal_color":[],
            "phase":[]
            }  
        return obs, self.hetero_graph
        return 0, 0

    def reset_default(self):
        self._current_time = 0
        self._init_sim(self.cfg.get("sumocfg_file"), self.seed, self.config.get("episode_length_time"),
                       self.cfg.get("gui"))
        self.infastructure = self._infastructure_extraction1(self.cfg.get("sumocfg_file"))
        self.vehicle_info.clear()
        self.all_reward = {}
        for tl in self.all_tls:
            self.all_reward[tl] = {k: 0 for k in self.reward_type}
            self._crosses[tl] = Intersection(tl, self.infastructure[tl], self, self.state_key,
                                             self.not_default)
            self._crosses[tl].update_timestep()

    def get_unava(self, turn_info, id, index, vehicle_type = "passenger"):
        if vehicle_type == "emergency":
            agent = self.emv_agent[id]
        else:
            agent = self.vehicle_agent[id]
        destination = agent._destination_road_id
        choices = {0: "s", 1: "r", 2: "l"}
        edge = agent._current_lane
        direction = choices[index]
        if direction in list(turn_info[edge].keys()):
            next_edge = turn_info[edge][direction]
            # if len(self.sim.simulation.findRoute(next_edge, destination).edges) == 0:
            #     return False
            # else:
            #     return True
            #需修改
            edges = self.sim.simulation.findRoute(next_edge, destination).edges
            is_u_turn = any(self.is_u_turn_edge_pair(e1, e2) for e1, e2 in zip(edges, edges[1:]))
            if len(edges) == 0 or is_u_turn:
                return False
            else:
                # print(self.sim.simulation.findRoute(next_edge, destination).edges)
                return True
        else:
            return False
        #     if direction in list(turn_info[edge].keys()):
        #     next_edge = turn_info[edge][direction]
        #     if len(self.sim.simulation.findRoute(next_edge, destination).edges) == 0:
        #         return False
        #     else:
        #         print(self.sim.simulation.findRoute(next_edge, destination).edges)
        #         return True
        # else:
        #     return False
    def is_u_turn_edge_pair(self, edge1, edge2):
        edge1 = self.net.getEdge(edge1)
        edge2 = self.net.getEdge(edge2)

        # 如果 edge1 的 fromNode 是 edge2 的 toNode，且反过来也成立
        if edge1.getFromNode().getID() == edge2.getToNode().getID() and \
        edge1.getToNode().getID() == edge2.getFromNode().getID():          
            return True
        return False  
        # return edge1.split("_")[0] == edge2.split("_")[1] and edge1.split("_")[1] == edge2.split("_")[0]    
        # def base_id(edge):
        #     return edge.split("#")[0]
        
        # return base_id(edge1) == "-" + base_id(edge2) or base_id(edge2) == "-" + base_id(edge1)



    def _get_state(self):
        # if self.step_tl % 15 == 0:
        ts_states = {}
        ve_states = {}
        emv_states = {}
        done_ve = {}
        done_emv = {}
        obs_change_ve = {}
        obs_change_emv = {}


        for veid in self.all_ves:
            ve_states[veid] = self.vehicle_agent[veid].get_state()
            done_ve[veid] = self.vehicle_agent[veid]._do_action
            obs_change_ve[veid] = self.vehicle_agent[veid]._obs_change
            if obs_change_ve[veid]:
                for i in range(self.action_space):
                    ava = self.get_unava(self.nextEdges, veid, i)
                    if not ava:
                        self.vehicle_agent[veid]._unava_index.append(i)
            # x = self.sim.vehicle.getLaneID(veid)
            # y = self.sim.lanearea.getLastStepVehicleIDs('top3D3_0')
            # z = self.sim.lanearea.getLastStepVehicleNumber('top3D3_0')
        for veid in self.all_emv:
            emv_states[veid] = self.emv_agent[veid].get_state()
            done_emv[veid] = self.emv_agent[veid]._do_action
            obs_change_emv[veid] = self.emv_agent[veid]._obs_change
            if obs_change_emv[veid]:
                for i in range(self.action_space):
                    ava = self.get_unava(self.nextEdges, veid, i, "emergency")
                    if not ava:
                        self.emv_agent[veid]._unava_index.append(i)
            # print(veid, emv_states[veid])
        for tid in self.all_tls:
            ts_states[tid] = self._crosses[tid].get_state(self.step_tl)
        return ts_states, ve_states, emv_states, done_ve, done_emv, obs_change_ve, obs_change_emv

    def _infastructure_extraction1(self, sumocfg_file):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()

        traffic_light_node_dict = {}
        for tl in net.findall("tlLogic"):
            if tl.attrib['id'] not in traffic_light_node_dict.keys():
                node_id = tl.attrib['id']
                traffic_light_node_dict[node_id] = {'leaving_lanes': [], 'entering_lanes': [],
                                                    'leaving_lanes_pos': [], 'entering_lanes_pos': [],
                                                    # "total_inter_num": None,
                                                    'adjacency_row': None}
                traffic_light_node_dict[node_id]["phases"] = [child.attrib["state"] for child in tl]

        # for index, item in enumerate(traffic_light_node_dict):
        #     traffic_light_node_dict[item]['total_inter_num'] = total_inter_num

        for edge in net.findall("edge"):
            if not edge.attrib['id'].startswith(":"):
                if edge.attrib['from'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes'].append(
                                child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['from']]['leaving_lanes_pos'].append(
                                child.attrib['shape'])
                if edge.attrib['to'] in traffic_light_node_dict.keys():
                    for child in edge:
                        if "id" in child.keys() and child.attrib['index'] == "0":
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes'].append(child.attrib['id'])
                            traffic_light_node_dict[edge.attrib['to']]['entering_lanes_pos'].append(
                                child.attrib['shape'])

        for junction in net.findall("junction"):
            if junction.attrib['id'] in traffic_light_node_dict.keys():
                traffic_light_node_dict[junction.attrib['id']]['location'] = \
                    {'x': float(junction.attrib['x']), 'y': float(junction.attrib['y'])}
        # print(traffic_light_node_dict,"trffic light nodes")
        return traffic_light_node_dict

    def bfs_find_neighbor(self, queue, edge_dict, net_lib, tl):
        seen = set()
        seen.add(tl)
        parents = {tl: None}
        while len(queue) > 0:
            if edge_dict[queue[0]].attrib['from'] != tl:
                if edge_dict[queue[0]].attrib['from'] not in self.all_tls:
                    next_node = edge_dict[queue[0]].attrib['from']
                    if next_node not in seen:
                        seen.add(next_node)
                        next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                        next_edges = {}
                        for nte in next_tmp_edges:
                            if edge_dict[nte._id].attrib['from'] == edge_dict[queue[0]].attrib['to']:
                                continue
                            # net_lib.getShortestPath(SC, CE)
                            c1 = net_lib.getNode(edge_dict[nte._id].attrib['from']).getCoord()
                            c2 = net_lib.getNode(next_node).getCoord()
                            c1 = {"x": c1[0], "y": c1[1]}
                            c2 = {"x": c2[0], "y": c2[1]}
                            next_edges[nte._id] = self._cal_distance(c1, c2)
                        if len(next_edges) > 0:
                            next_edges = sorted(next_edges.items(), key=lambda item: item[1])
                            next_edges = [ne[0] for ne in next_edges]
                        queue.extend(next_edges)
                        parents[next_node] = edge_dict[queue[0]].attrib['to']
                    del queue[0]
                else:
                    assert edge_dict[queue[0]].attrib['from'] in self.all_tls and edge_dict[queue[0]].attrib[
                        'from'] != tl, "{}-{}".format(edge_dict[queue[0]].attrib['from'], tl)
                    parents[edge_dict[queue[0]].attrib['from']] = edge_dict[queue[0]].attrib['to']
                    return True, parents, edge_dict[queue[0]].attrib['from']
            else:
                del queue[0]
        return False, None, None
        # length += self._cal_distance(
        #     traffic_light_node_dict[
        #         edge_dict[entering_sequence[es]].attrib['from']]['location'],
        #     traffic_light_node_dict[
        #         edge_dict[entering_sequence[es]].attrib['to']]['location']
        # )

    def _infastructure_extraction2(self, sumocfg_file, traffic_light_node_dict, dis=False):
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        network_file = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)
        net = xml.etree.ElementTree.parse(network_file).getroot()
        net_lib = sumolib.net.readNet(network_file)

        # all_tls is deleted
        all_traffic_light = self.all_tls
        total_inter_num = len(self.all_tls)
        if dis:
            top_k = self.adjacency_top_k
            for i in range(total_inter_num):
                if 'location' not in traffic_light_node_dict[all_traffic_light[i]]:
                    continue
                location_1 = traffic_light_node_dict[all_traffic_light[i]]['location']
                row = np.array([0] * total_inter_num)
                for j in range(total_inter_num):
                    if 'location' not in traffic_light_node_dict[all_traffic_light[j]]:
                        row[j] = 1e8
                        continue
                    location_2 = traffic_light_node_dict[all_traffic_light[j]]['location']
                    dist = self._cal_distance(location_1, location_2)
                    row[j] = dist
                if len(row) == top_k:
                    adjacency_row_unsorted = np.argpartition(row, -1)[:top_k].tolist()
                elif len(row) > top_k:
                    adjacency_row_unsorted = np.argpartition(row, top_k)[:top_k].tolist()
                else:
                    adjacency_row_unsorted = list(range(total_inter_num))

                if self.is_adjacency_remove:
                    adjacency_row_unsorted.remove(i)

                adjacency_row_unsorted = [j for j in adjacency_row_unsorted]
                traffic_light_node_dict[all_traffic_light[i]]['adjacency_row'] = \
                    [[adjacency_row_unsorted, row[adjacency_row_unsorted]], []]
        else:
            # adjacency_row is [entering NWSE neighbor, outgoing NWSE neighbor]
            #  NWSE neighbor contain: 1. neighbor tl name 2. edge distance

            edge_dict = {}
            for edge in net.findall("edge"):
                edge_dict[edge.attrib['id']] = edge

            junction_dict = {}
            for jun in net.findall("junction"):
                junction_dict[jun.attrib["id"]] = jun

            for i in self.all_tls:
                entering_sequence_NWSE = self._crosses[i].entering_sequence_NWSE
                entering_sequence = self._crosses[i].entering_sequence
                outgoing_sequence_NWSE = self._crosses[i].outgoing_sequence_NWSE
                outgoing_sequence = self._crosses[i].outgoing_sequence
                adjacency_row_entering = []
                adjacency_distance_entering = []
                adjacency_row_outgoing = []
                adjacency_distance_outgoing = []
                for es in entering_sequence_NWSE:
                    if es != -1:
                        queue = [entering_sequence[es]]
                        flag, parents, last_node = self.bfs_find_neighbor(queue, edge_dict, net_lib, i)
                        # while edge_dict[queue[0]].attrib['from'] not in all_traffic_light:
                        #     next_node = edge_dict[entering_sequence[es]].attrib['from']
                        #     next_tmp_edges = net_lib.getNode(next_node).getIncoming()
                        #     for index, nte in enumerate(next_tmp_edges):
                        #         if edge_dict[nte].attrib['from'] == i:
                        #             rm_index = index
                        #             break
                        #     del next_tmp_edges[rm_index]
                        #     queue.extend(next_tmp_edges)
                        if flag:
                            length = 0
                            last_node_ = last_node
                            while parents[last_node] != None:
                                coor1 = net_lib.getNode(last_node).getCoord()
                                coor2 = net_lib.getNode(parents[last_node]).getCoord()
                                c1 = {"x": coor1[0], "y": coor1[1]}
                                c2 = {"x": coor2[0], "y": coor2[1]}
                                length += self._cal_distance(c1, c2)
                                if length > 700:
                                    break
                                last_node = parents[last_node]
                            if length > 700:
                                no_neigh = True
                            else:
                                no_neigh = False
                        else:
                            no_neigh = True
                    else:
                        no_neigh = True
                    if no_neigh:
                        adjacency_row_entering.append(-1)
                        adjacency_distance_entering.append(1e8)
                    else:
                        adjacency_row_entering.append(
                            self.all_tls.index(last_node_))
                        adjacency_distance_entering.append(length / 100)
                for os_ in outgoing_sequence_NWSE:
                    if os_ != -1 and edge_dict[outgoing_sequence[os_]].attrib['to'] in all_traffic_light:
                        adjacency_row_outgoing.append(
                            self.all_tls.index(edge_dict[outgoing_sequence[os_]].attrib['to']))
                        adjacency_distance_outgoing.append(self._cal_distance(
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['from']]['location'],
                            traffic_light_node_dict[
                                edge_dict[outgoing_sequence[os_]].attrib['to']]['location']
                        ) / 100)
                    else:
                        adjacency_row_outgoing.append(-1)
                        adjacency_distance_outgoing.append(1e8)
                traffic_light_node_dict[i]['adjacency_row'] = \
                    [(adjacency_row_entering, adjacency_distance_entering),
                     (adjacency_row_outgoing, adjacency_distance_outgoing)]
        return traffic_light_node_dict

    @staticmethod
    def _cal_distance(loc_dict1, loc_dict2):
        a = np.array((loc_dict1['x'], loc_dict1['y']))
        b = np.array((loc_dict2['x'], loc_dict2['y']))
        return np.sqrt(np.sum((a - b) ** 2))

    @staticmethod
    def _coordinate_sequence(list_coord_str):
        import re
        list_coordinate = [re.split(r'[ ,]', lane_str) for lane_str in list_coord_str]
        # x coordinate
        x_all = np.concatenate(list_coordinate).astype('float64')
        west = np.int(np.argmin(x_all) / 2)

        y_all = np.array(list_coordinate, dtype=float)[:, [1, 3]]

        south = np.int(np.argmin(y_all) / 2)

        east = np.int(np.argmax(x_all) / 2)
        north = np.int(np.argmax(y_all) / 2)

        list_coord_sort = [west, north, east, south]
        return list_coord_sort

    @staticmethod
    def _sort_lane_id_by_sequence(ids, sequence=[2, 3, 0, 1]):
        result = []
        for i in sequence:
            result.extend(ids[i * 3: i * 3 + 3])
        return result

    @staticmethod
    def get_actual_lane_id(lane_id_list):
        actual_lane_id_list = []
        for lane_id in lane_id_list:
            if not lane_id.startswith(":"):
                actual_lane_id_list.append(lane_id)
        return actual_lane_id_list

    def get_vehicle_agent(self):
        self.vehicle_num_agents = 0
        self.emv_num_agents = 0
        self.all_ves = []
        self.all_emv = []
        new_ves = []
        new_emv = []


        for tl in self.all_tls:

            self._crosses[tl].get_vehicles_in_range()
            # if len(self._crosses[tl].vehicles_in_range) > 0:
            #     for vehicle_id in self._crosses[tl].vehicles_in_range:
        for vehicle_id in self.sim.vehicle.getIDList():
            if self.sim.vehicle.getTypeID(vehicle_id) != "emergency":
                # if len(self.sim.vehicle.getNextTLS(vehicle_id)) != 0:
                self.vehicle_num_agents += 1
                self.all_ves.append(vehicle_id)
                if vehicle_id not in self.vehicle_agent:     
                    self.vehicle_agent[vehicle_id] = VehicleAgent(vehicle_id, self,  self.all_tls)
                    new_ves.append(vehicle_id)
                        
            else:
                # if len(self.sim.vehicle.getRoute(vehicle_id)) != 0:
                self.all_emv.append(vehicle_id)
                self.emv_num_agents += 1
                if vehicle_id not in self.emv_agent:
                    self.emv_agent[vehicle_id] = VehicleAgent(vehicle_id, self, self.all_tls)
                    new_emv.append(vehicle_id)
        return new_ves, new_emv
        #该函数需要修改-这个代码有问题！！！
    # def get_vehicle_agent(self):
    #     self.vehicle_num_agents = 0
    #     self.emv_num_agents = 0
    #     self.all_ves = []
    #     self.all_emv = []
    #     new_ves = []
    #     new_emv = []
    #     all = self.sim.vehicle.getIDList()
    #     for i in all:
    #         if self.sim.vehicle.getTypeID(i) == "emergency":
    #             self.all_emv.append(i)
    #             self.emv_num_agents += 1
    #         else:
    #             self.all_ves.append(i)
    #             self.vehicle_num_agents += 1

    #     for tl in self.all_tls:

    #         self._crosses[tl].get_vehicles_in_range()
    #         if len(self._crosses[tl].vehicles_in_range) > 0:
    #             for vehicle_id in self._crosses[tl].vehicles_in_range:
    #                 if self.sim.vehicle.getTypeID(vehicle_id) != "emergency":
    #                     if vehicle_id not in self.vehicle_agent:
    #                         self.vehicle_agent[vehicle_id] = VehicleAgent(vehicle_id, self,  self.all_tls)
    #                         new_ves.append(vehicle_id)
    #                 else:
    #                     if vehicle_id not in self.emv_agent:
    #                         self.emv_agent[vehicle_id] = VehicleAgent(vehicle_id, self, self.all_tls)
    #                         new_emv.append(vehicle_id)
    #     return new_ves, new_emv
        # for vehicle_id in list(self.vehicle_agent):
        #     if vehicle_id not in self.all_ves:
        #         # del self.vehicle_agent[vehicle_id]
        #         self.vehicle_num_agents -= 1
        # for vehicle_id in list(self.emv_agent):
        #     if vehicle_id not in self.all_emv:
        #         # del self.emv_agent[vehicle_id]
        #         self.emv_num_agents -= 1
        # if len(self.sim.vehicle.getIDList()) > 0:
            # print(self.sim.vehicle.getIDList(),"current_vehicle")
            # print(self.sim.vehicle.getPosition("0"),"0_position")
            # print(self.vehicle_agent["0"]._route,"0_route")



    def reset_hetero_graph(self):
        self.hetero_graph['signal_light'].x = []
        self.hetero_graph['vehicle'].x = []
        self.hetero_graph['emergency'].x = []
        self.hetero_graph['signal_light', 'contral', 'vehicle'].edge_index = []
        self.hetero_graph['signal_light', 'important', 'emergency'].edge_index = []
        self.hetero_graph['vehicle', 'similar_to', 'emergency'].edge_index = []

        neighbor_tl = {}
        for i, tl in enumerate(self.all_tls):
            neighbor_tl[tl] = []
            for tl_id in self.infastructure[tl]['adjacency_row'][0][0]:
                if tl_id != -1:
                    neighbor_tl[tl].append(self.all_tls[tl_id])
        edges_set = set()
        # 生成信号灯间的边（无向）
        for signal, neigh_list in neighbor_tl.items():
            for neighbor in neigh_list:
                signal_id = self.all_tls.index(signal)  # 获取信号灯 ID 在列表中的索引
                neighbor_id = self.all_tls.index(neighbor)  # 获取邻居信号灯的 ID 索引

                # 将边按（较小的索引，较大的索引）顺序添加到集合中，保证无向
                if signal_id < neighbor_id:
                    edges_set.add((signal_id, neighbor_id))
                else:
                    edges_set.add((neighbor_id, signal_id))

        # 将去重后的边集合转换为张量
        signal_to_signal_edges = torch.tensor(list(edges_set)).t()
        self.hetero_graph['signal_light', 'connects', 'signal_light'].edge_index = signal_to_signal_edges

    def build_hetero_graph(self, obs, ve_obs, emv_obs):
        # 遍历所有交通信号灯
        self.reset_hetero_graph()
        self.hetero_graph['signal_light'].x = obs  # 信号灯的特征
        self.hetero_graph['emergency'].x = emv_obs    #信号灯特征
        self.hetero_graph['vehicle'].x = ve_obs
        neighbor_tl_ve = {}
        if len(ve_obs) > 0:
            for i, tl in enumerate(self.all_tls):
                if len(self._crosses[tl].vehicles_in_range) > 0:
                    neighbor_tl_ve[tl] = []
                    for vehicle_id in self._crosses[tl].vehicles_in_range:
                        neighbor_tl_ve[tl].append(vehicle_id)
                # for j, vehicle in enumerate(self.all_ves):
                edges = []
                edges_emv = []
                vehicle_tl_map = {}  # 存储每辆车对应的信号灯 ID
                emergency_tl_map = {}

                for signal, vehicles in neighbor_tl_ve.items():
                    for vehicle in vehicles:
                        if self.sim.vehicle.getTypeID(vehicle) == "emergency":
                            emergency_tl_map[vehicle] = self.all_tls.index(signal)
                            edges_emv.append([self.all_tls.index(signal), int(vehicle.split('_')[-1])])
                        else:
                            vehicle_tl_map[vehicle] = self.all_tls.index(signal)
                            edges.append([self.all_tls.index(signal), int(vehicle)])

                self.hetero_graph['signal_light', 'contral', 'vehicle'].edge_index = torch.tensor(edges).t()
                if len(edges_emv) > 0:
                    self.hetero_graph['signal_light', 'important', 'emergency'].edge_index = torch.tensor(edges_emv).t()

            # **新增：计算普通车辆与紧急车辆的相似度，并构建边**
            if len(ve_obs) > 0 and len(emv_obs) > 0:
                # 1. 先筛选出连接到同一个信号灯的普通车辆 & 紧急车辆
                same_tl_pairs = []
                for ve, tl_id in vehicle_tl_map.items():
                    for emv, emv_tl_id in emergency_tl_map.items():
                        if tl_id == emv_tl_id:  # 只有同一个信号灯的车辆才计算
                            same_tl_pairs.append((int(ve), int(emv.split('_')[-1])))

                    if same_tl_pairs:
                        # 4. 生成边索引
                        vehicle_to_emergency_edges = torch.tensor(same_tl_pairs).t()
                        self.hetero_graph['vehicle', 'similar_to', 'emergency'].edge_index = vehicle_to_emergency_edges


    def ve_batch_clean(self, env_output, use_keys, all_ves):
        obs_batch = {}

        for i in use_keys:
            obs_batch[i] = []
        for agent_id in all_ves:
            state = env_output[agent_id]
            tmp_dict = {k: np.zeros(5) for k in use_keys}

            for s in use_keys:
                tmp_dict[s] = state.get(s)
            for k in use_keys:
                obs_batch[k].append(tmp_dict[k])

        values = list(obs_batch.values())
        obs_values = np.hstack([np.array(v).reshape(-1, 1) for v in values])

        return obs_values


    def ve_batch(self, env_output, use_keys, all_ves):
        # 使用 NumPy 数组初始化 emv_obs，形状为 (num_emv_agents, len(use_keys))
        emv_obs = np.full((self.num_vehicle_agents, len(use_keys)), 0, dtype=int)

        # 遍历所有的 agent，填充对应的数据
        for i, agent_id in enumerate(all_ves):
            state = env_output[agent_id]
            # 根据 use_keys 提取指定字段的数据，若字段不存在，使用默认值 -1
            emv_obs[int(agent_id), :] = np.array([state.get(key, 0) for key in use_keys], dtype=int)
        return emv_obs

    def emv_batch(self, env_output, use_keys, all_emv):
        # 使用 NumPy 数组初始化 emv_obs，形状为 (num_emv_agents, len(use_keys))
        emv_obs = np.full((self.num_emv_agents, len(use_keys)), 0, dtype=int)

        # 遍历所有的 agent，填充对应的数据
        for i, agent_id in enumerate(all_emv):
            state = env_output[agent_id]
            # 根据 use_keys 提取指定字段的数据，若字段不存在，使用默认值 ''
            emv_obs[int(agent_id.split('_')[-1]), :] = np.array([state.get(key, 0) for key in use_keys], dtype=int)

        return emv_obs

    def batch(self, env_output, use_keys, all_tl):
        """Transform agent-wise env_output to batch format."""
        if all_tl == ['gym_test']:
            return torch.tensor([env_output])
        obs_batch = {}
        for i in use_keys + ['mask', 'neighbor_index', 'neighbor_dis']:
            obs_batch[i] = []
        for agent_id in all_tl:
            out = env_output[agent_id]
            tmp_dict = {k: np.zeros(8) for k in use_keys}
            state, mask, neight_msg = out
            for i in range(len(state)):
                for s in use_keys:
                    tmp_dict[s][i] = state[i].get(s, 0)
            for k in use_keys:
                obs_batch[k].append(tmp_dict[k])
            obs_batch['mask'].append(mask)
            obs_batch['neighbor_index'].append(neight_msg[0][0])
            obs_batch['neighbor_dis'].append(neight_msg[0][1])

        for key, val in obs_batch.items():
            if key not in ['current_phase', 'mask', 'neighbor_index']:
                obs_batch[key] = torch.FloatTensor(np.array(val))
            else:
                obs_batch[key] = torch.LongTensor(np.array(val))

        ###### 把 pressure 放在字典最后面

        if self.use_pressure:
            obs_batch['pressure'] = obs_batch.pop('pressure')
        else:
            obs_batch.pop('pressure')

        if self.use_gat:
            obs_batch['neighbor_index'] = obs_batch.pop('neighbor_index')
            obs_batch['neighbor_dis'] = obs_batch.pop('neighbor_dis')
        else:
            obs_batch.pop('neighbor_index')
            obs_batch.pop('neighbor_dis')

        self.obs_keys = list(obs_batch.keys())
        obs_values = np.hstack(list(obs_batch.values()))  ### 25, 56
        return obs_values


    def _find_adjacent_edge(self, sumocfg_file):
        """
        obtain the adjacent edges of each edge, which are stored in a dictionary.
        for example:

        {'A0A1': {'r': 'A1B1', 's': 'A1A2', 'l': 'A1left1', 't': 'A1A0'},
        'A0B0': {'r': 'B0bottom1', 's': 'B0D5', 'l': 'B0B1', 't': 'B0A0'},
        'A0bottom0': {'t': 'bottom0A0'},
        'A0left0': {'t': 'left0A0'},
        'A1A0': {'r': 'A0left0', 's': 'A0bottom0', 'l': 'A0B0', 't': 'A0A1'}}

        'r', 's', 'l', 't' represents the direction of the adjacent edges.

        """
        e = xml.etree.ElementTree.parse(sumocfg_file).getroot()
        network_file_name = e.find('input/net-file').attrib['value']
        file_name = os.path.join(os.path.split(sumocfg_file)[0], network_file_name)

        # file_name = xml.etree.ElementTree.parse(network_file).getroot()
        attr_connection = {'connection': ['from', 'to', 'fromLane', 'toLane', 'via', 'tl', 'linkIndex', 'dir', 'state']}
        attr_edge = {'edge': ['id', 'from', 'to', 'priority']}

        # obtain all the edges in the network
        turn_info = dict()
        edges = list(sumolib.xml.parse(file_name, 'net', attr_edge))[0]
        for edge in edges.edge:
            if edge.id[0] != ":":
                next_edges = dict()
                turn_info[edge.id] = next_edges

        # obtain the adjacent edges of each edge
        connections = list(sumolib.xml.parse(file_name, 'net', attr_connection))[0]
        for connection in connections.connection:
            if connection.attr_from[0] != ':':
                edge = connection.attr_from
                direction = connection.dir
                if direction == "L" or direction == "R":
                    direction = "s"
                next_edge = connection.to
                turn_info[edge][direction] = next_edge
        return turn_info

    def _determine_next_edge(self, turn_info, id, index, vehicle_type="passenger"):
        """
        reroute all the vehicle detected by the sumo induction loop
        :param loop_name: the name of the induction
        :return:
        """
        if vehicle_type == "emergency":
            agent = self.emv_agent[id]
        else:
            agent = self.vehicle_agent[id]
        destination = agent._destination_road_id


        choices = {0: "s", 1: "r", 2: "l"}

        edge = agent._current_lane

        if edge == destination:
            new_route = [edge]
            agent.take_action(new_route)
            agent._do_action = False

        else:
            direction = choices[index]
            if direction in list(turn_info[edge].keys()):
                next_edge = turn_info[edge][direction]
                destination_route = self.sim.simulation.findRoute(next_edge, destination).edges
                if len(destination_route) > 0:
                    destination_route = destination_route[1:]
                new_route = [edge] + [next_edge] + list(destination_route)
                agent.take_action(new_route)
            else:
                # next_edge = turn_info[edge][list(turn_info[edge].keys())[0]]
                agent._do_action = False

    # def _determine_next_edge(self, turn_info, id, index, vehicle_type="passenger"):
    #     """
    #     reroute all the vehicle detected by the sumo induction loop
    #     :param loop_name: the name of the induction
    #     :return:
    #     """
    #     if vehicle_type == "emergency":
    #         agent = self.emv_agent[id]
    #     else:
    #         agent = self.vehicle_agent[id]
    #     destination = agent._destination_road_id


    #     choices = {0: "s", 1: "r", 2: "l"}

    #     edge = agent._current_lane

    #     if edge == destination:
    #         if len(self.sim.simulation.findRoute(destination, agent._destination_road).edges) > 1:
    #             new_route = [edge]+ list(self.sim.simulation.findRoute(destination, agent._destination_road).edges[1:])
    #         else:
    #             new_route = [edge]
    #         agent.take_action(new_route)
    #         agent._do_action = False

    #     else:
    #         direction = choices[index]
    #         if direction in list(turn_info[edge].keys()):
    #             next_edge = turn_info[edge][direction]
    #             destination_route = self.sim.simulation.findRoute(next_edge, destination).edges
    #             if len(destination_route) > 0:
    #                 destination_route = destination_route[1:]
    #             if len(self.sim.simulation.findRoute(destination, agent._destination_road).edges) > 1:
    #                 new_route = [edge] + [next_edge] + list(self.sim.simulation.findRoute(next_edge, agent._destination_road).edges[1:])
    #             else:
    #                 new_route = [edge] + [next_edge] + list(destination_route)
    #             agent.take_action(new_route)
    #         else:
    #             # next_edge = turn_info[edge][list(turn_info[edge].keys())[0]]
    #             agent._do_action = False

    def log(self):
        current_time = self.sim.simulation.getTime()

        for veh_id in self.sim.vehicle.getIDList():
            self.data["time"].append(current_time)
            self.data["vehicle_id"].append(veh_id)
            self.data["speed"].append(self.sim.vehicle.getSpeed(veh_id))

            lane_id = self.sim.vehicle.getLaneID(veh_id)
            self.data["lane_id"].append(lane_id)

            

            if len(self.sim.vehicle.getNextTLS(veh_id)) >0:
                # 获取当前灯状态字符串，比如 'rrGGrr'
                tls_id = self.sim.vehicle.getNextTLS(veh_id)[0][0]
                light_state = self.sim.trafficlight.getRedYellowGreenState(tls_id)

                # 获取 tls_id 控制的 lane-link 列表
                links = self.sim.trafficlight.getControlledLinks(tls_id)

                # 查找车辆 lane_id 所对应的 link index
                signal_color = None
                for idx, link_list in enumerate(links):
                    for link in link_list:  # 每个 link 是一个 tuple: (fromLane, toLane, via)
                        if link[0] == lane_id:
                            signal_color = light_state[idx]  # 红绿灯状态字符
                            break
                    if signal_color is not None:
                        break
            else:
                signal_color = 'n'  # not under any light

            self.data["signal_color"].append(signal_color)  # r/g/y/n

        tl_phase = []
        for tls_id in self.all_tls:
            tl_phase.append({
                "id": tls_id,
                "state": self.sim.trafficlight.getRedYellowGreenState(tls_id),
            })
        self.data["phase"].append(tl_phase)
        # # 记录信号灯状态（可选）
        # for tl_id in self.sim.trafficlight.getIDList():
        #     print(f"Signal {tl_id} state: {self.sim.trafficlight.getRedYellowGreenState(tl_id)}")


if __name__ == '__main__':
    config = {
        "name": "colight",
        "agent": "",
        # "sumocfg_file": "rl-tsc/sumo_files/scenarios/nanshan/osm.sumocfg",
        "sumocfg_file": "/home/PhD/ChenM/experiment/CoEMV/onpolicy/envs/sumo_files_marl/scenarios/resco_envs/grid4x4_trip/grid4x4.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/cologne8/cologne8.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/sumo_fenglin_base_road/base.sumocfg",
        # "sumocfg_file": "sumo_files/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg",

        "action_type": "select_phase",
        "gui": False,
        "yellow_duration": 5,
        "iter_duration": 10,
        "episode_length_time": 3600,
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'state_key': ['current_phase', 'vehicle_num', 'queue_length', "occupancy", 'flow', 'stop_vehicle_num',
                      'pressure'],
        've_state_key': ['speed', 'current_lane', 'current_phase_index', 'destination_road_id',
                         'destination_intersection', 'do_action'],
        've_state_key': ['current_intersection', 'current_intersection_direction', 'destination_intersection',
                         'destination_intersection_direction'],
        'reward_type': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'reward_type_ve': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score'],
        'reward_type_emv': ['queue_len', 'wait_time', 'delay_time', 'pressure', 'speed score']

    }
    env = TSCSimulator(config, 1235)
    env.reset()
    all_tl = env.all_tls
    # sumoBinary = checkBinary('sumo')
    # # #
    # traci.start([sumoBinary, "-c", "/home/PhD/ChenM/experiment/CoEMV/onpolicy/envs/sumo_files_marl/scenarios/resco_envs/grid4x4_trip/grid4x4.sumocfg", "--tripinfo-output",
    #              "grid4x4.net.xml",])  # 多次调用
    # for step in range(0, 3600):
    #     while traci.simulation.getMinExpectedNumber() > 0:
    #         traci.simulationStep()
    #         run()
    # traci.close()
    # sys.stdout.flush()
    
    data = {
        "time": [], 
        "vehicle_id": [],
        "speed": [],
        "lane_id": []
    }
    x = 0
    # for step in range(100):
    #     env.sim.simulationStep()
    #     print(env.sim.trafficlight.getPhase("A0"))
    #     print("step")

    # sys.exit()
    for step in range(1000):
        env.default_step()
        if '24' in env.sim.vehicle.getIDList():
            print(env.sim.vehicle.getRoadID('24'))
        # current_time = traci.simulation.getTime()
        
        # # 记录所有车辆数据
        # for veh_id in traci.vehicle.getIDList():
        #     data["time"].append(current_time)
        #     data["vehicle_id"].append(veh_id)
        #     data["speed"].append(traci.vehicle.getSpeed(veh_id))
        #     data["lane_id"].append(traci.vehicle.getLaneID(veh_id))
        
        # # 记录信号灯状态（可选）
        # for tl_id in traci.trafficlight.getIDList():
        #     print(f"Signal {tl_id} state: {traci.trafficlight.getRedYellowGreenState(tl_id)}")

    # 保存为CSV
    pd.DataFrame(data).to_csv("/home/PhD/ChenM/experiment/CoEMV/onpolicy/envs/sumo_files_marl/vehicle_data.csv", index=False)
    traci.close()
    # for i in range(3600):
    #     # print(env.sim.vehicle.getLaneID("0"))
    #     # print(env.sim.junction.getIDList())
    #     # vector = env.sim.junction.getIDList()
    #     # start_index = vector.index('A0')
    #     # print(start_index)
    #     # print(vector[start_index:])
    #     # destination_road_id = env.sim.vehicle.getRoute('0')[-1]
    #     # print(env.sim.edge.getToJunction(destination_road_id))
    #     # sys.exit()
    #     # route = traci.vehicle.getLaneID("0")
    #     # print(route)
    #     # print("Current route:", traci.vehicle.getRoadID("0").replace(':','').split('_')[:-1])
    #     # print(traci.simulation.findRoute("nt4_np5","np13_nt13").edge)
    #     # sys.exit()
    #     # if i >460:
    #     #     if "305" in traci.vehicle.getIDList():
    #     #         print("305 exit")
    #     #         sys.exit()
    #     # print(i,"step",env.sim.trafficlight.getRedYellowGreenState('A0'))
    #     print(i,"step")
    #     # for lanes in env._crosses['D3']._incoming_lanes:
    #     #     print(env.sim.lanearea.getLastStepVehicleIDs(lanes),'lanes:',lanes)
    #     # print(traci.vehicle.getIDList())
    #     # # if i >= 0:
    #     # #     traci.vehicle.setStop( "0", "top3D3", pos=7.0, laneIndex=0, duration=-1073741824.0, flags=0, startPos=-1073741824.0,
    #     # #             until=-1073741824.0)

    #     # # if "0" in traci.vehicle.getIDList():
    #     # print(traci.vehicle.getLanePosition("0"))
    #     #     print(i,"step",env.sim.vehicle.getNextTLS('1471'))
    #     traci.simulationStep(i)
    # traci.terminate()
        # print(i,"lane",env.sim.vehicle.getLaneID("0"))
        # while traci.simulation.getMinExpectedNumber() > 0:
        # if "0" in traci.vehicle.getIDList() and i == 2:
        #     traci.vehicle.setRoute("0", ["top3D3","D3right3"])
        # if traci.vehicle.getRoadID("0") == "D3right3":
        #     traci.vehicle.setRoute("0", ["D3right3"])
        # print(env.sim.vehicle.getPosition("emergency_0"))
        # x += 25
        # env.sim.simulationStep(x)
        # print(i, env.sim.lanearea.getLastStepVehicleNumber("left0A0_1"))
        # sys.exit()
        # x += 5
        # traci.simulationStep()
        # print(traci.vehicle.getIDList())
        # if "0" in traci.vehicle.getIDList():
        #     print(traci.vehicle.getLastActionTime("0"))
            # print(traci.vehicle.isStoppedParking("0"))
            # print(traci.vehicle.getAccumulatedWaitingTime("0"))
            # print(len(traci.simulation.findRoute("left0A0","left0A0").edges))


        # if i == 9:
        #     env._determine_next_edge(env.nextEdges, "0", 1)

    # for i in range(100):
    #     tl_action_select = {}
    #     print(i)
    #
    #     for tl in all_tl:
    #         # env._crosses[tl].get_vehicles_in_range()
    #         # for id in env._crosses[tl]._env.sim.vehicle.getIDList():
    #         #     print(env._crosses[tl]._env.sim.vehicle.getLaneID(id))
    #         # print(env._crosses[tl]._lane_vehicle_dict)
    #         # print(env._current_time)
    #         # print(env._crosses[tl].vehicles_in_range)
    #         # print(env._crosses[tl].get_phase_index())
    #         a = np.random.choice(env._crosses[tl].green_phases)
    #         while a / 2 in env._crosses[tl].unava_index:
    #             a = np.random.choice(env._crosses[tl].green_phases)
    #         tl_action_select[tl] = a
    #     next_obs, reward, done, _ = env.step(tl_action_select)
    #
    # traci.close()