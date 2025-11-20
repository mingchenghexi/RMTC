import random

# import gfootball.env as football_env
from onpolicy.envs.sumo_files_marl.env.sim_env import TSCSimulator
from onpolicy.envs.sumo_files_marl.config import config

from gym import spaces
import numpy as np

import torch
import copy
import os, sys
import traci

# output_path


class SUMOEnv(object):
    '''Wrapper to make Google Research Football environment compatible'''

    # 定义 SUMO 环境类，用于包装并适配 SUMO 交通仿真环境

    def __init__(self, args, rank):
        
        self.args = args
        id = args.seed + np.random.randint(0, 2023) + rank
        self.set_seed(id)
        
        # make env
        self.env_config = config['environment']
        # sumo_envs_num = len(env_config['sumocfg_files'])
        sumo_cfg = args.sumocfg_files
        sumo_cfg = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + '/' + sumo_cfg

        self.env_config = copy.deepcopy(self.env_config)
        self.env_config['sumocfg_file'] = sumo_cfg
        port = args.port_start + id
        
        # print('------------------------', port )
        print('----port--', port, '----sumo_cfg--', sumo_cfg)
        
        output_path = config.get("environment").get("output_path")
        output_path = output_path + self.env_config['sumocfg_files'][0].split('/')[-2] + '/trial_' + str(id) + '/'

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.env = TSCSimulator(self.args, self.env_config, port, output_path=output_path)
        
        self.unava_phase_index = []
        for i in self.env.all_tls:
            self.unava_phase_index.append(self.env._crosses[i].unava_index)


        self.type_agents = 3  # TL EMV NV

        self.num_vehicle_agents = args.num_vehicle_agents
        self.num_ve_agents = args.num_ve_agents
        self.num_emv_agents = args.num_emv_agents
        self.num_agents = len(self.env.all_tls)
        self.unava_vehicle_index = np.full((self.num_vehicle_agents, 1), -1)
        self.unava_emv_index = np.full((self.num_emv_agents,1), -1)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        self.action_space_vehicle = self.env_config["vehicle_num_actions"]

        for idx in range(self.num_agents):
            self.action_space.append(spaces.Discrete(n=self.env_config['num_actions']))
            self.share_observation_space.append(spaces.Box(-float('inf'), float('inf'), [self.env_config['obs_shape']*self.num_agents], dtype=np.float32))
            self.observation_space.append(spaces.Box(-float('inf'), float('inf'), [self.env_config['obs_shape']], dtype=np.float32))

        self.vehicle_action_space = []
        self.vehicle_observation_space = []
        self.vehicle_share_observation_space = []

        for idx in range(self.num_vehicle_agents):
            self.vehicle_action_space.append(spaces.Discrete(n=self.env_config['vehicle_num_actions']))
            self.vehicle_share_observation_space.append(
                spaces.Box(-float('inf'), float('inf'), [self.env_config['vehicle_obs_shape'] * self.num_vehicle_agents],
                           dtype=np.float32))
            self.vehicle_observation_space.append(
                spaces.Box(-float('inf'), float('inf'), [self.env_config['vehicle_obs_shape']], dtype=np.float32))

        self.ve_share_observation_space = []
        for idx in range(self.num_ve_agents):
            self.ve_share_observation_space.append(spaces.Box(-float('inf'), float('inf'), [self.env_config['vehicle_obs_shape'] * self.num_ve_agents],
                           dtype=np.float32))

        self.emv_action_space = []
        self.emv_observation_space = []
        self.emv_share_observation_space = []

        for idx in range(self.num_emv_agents):
            self.emv_action_space.append(spaces.Discrete(n=self.env_config['vehicle_num_actions']))
            self.emv_share_observation_space.append(
                spaces.Box(-float('inf'), float('inf'), [self.env_config['vehicle_obs_shape'] * self.num_emv_agents],
                           dtype=np.float32))
            self.emv_observation_space.append(
                spaces.Box(-float('inf'), float('inf'), [self.env_config['vehicle_obs_shape']], dtype=np.float32))

    def get_unava_index(self):
        unava_vehicle_index = {}
        unava_emv_index = {}
        for vid in self.env.all_ves:
            unava_vehicle_index[vid] = self.env.vehicle_agent[vid]._unava_index
        for vid in self.env.all_emv:
            unava_emv_index[vid] = self.env.emv_agent[vid]._unava_index
        return unava_vehicle_index, unava_emv_index


    def get_unava_phase_index(self):
        # print(self.unava_phase_index,"get_unava_phase_index")
        #需要修改
        return self.unava_phase_index
        # return np.array(self.unava_phase_index)
    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        return

    def get_reward(self, reward, all_tls):
        ans = []
        for i in all_tls:
            ans.append(sum(reward[i].values()))
        return np.array(ans)

    def get_reward_ve(self, ve_reward, emv_reward, vehicle_num, emv_num):
        ans = np.full((vehicle_num+emv_num, 1),np.nan)
        if len(ve_reward) > 0:
            for k, v in ve_reward.items():
                ans[int(k)]=sum(v.values())
        if len(emv_reward) > 0:
            for k, v in emv_reward.items():
                ans[vehicle_num+int(k.split('_')[-1])] = sum(v.values())

        return ans

    def get_done_ve(self, ve_reward, emv_reward, vehicle_num, emv_num):
        ans = np.full((vehicle_num+emv_num, 1),None)
        if len(ve_reward) > 0:
            for k, v in ve_reward.items():
                ans[int(k)]=v
        if len(emv_reward) > 0:
            for k, v in emv_reward.items():
                ans[vehicle_num+int(k.split('_')[-1])] = v

        return ans


    def batch(self, env_output, use_keys, all_tl):
        """Transform agent-wise env_output to batch format."""
        if all_tl == ['gym_test']:
            return torch.tensor([env_output])
        obs_batch = {}
        for i in use_keys+['mask', 'neighbor_index', 'neighbor_dis']:
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
        
        if self.args.use_pressure:
            obs_batch['pressure'] = obs_batch.pop('pressure')
        else:
            obs_batch.pop('pressure')
            
        if self.args.use_gat:
            obs_batch['neighbor_index'] = obs_batch.pop('neighbor_index')
            obs_batch['neighbor_dis'] = obs_batch.pop('neighbor_dis')
        else:
            obs_batch.pop('neighbor_index')
            obs_batch.pop('neighbor_dis')
        
        self.obs_keys = list(obs_batch.keys())
        obs_values = np.hstack(list(obs_batch.values())) ### 25, 56
        
        
        return obs_values

    # def reset_ve(self):
    #     obs = self.env.reset_ve()
    #     obs_values = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
    #     obs_values = self._obs_wrapper(obs_values)
    #     return obs_values

    def reset(self):
        obs, graph = self.env.reset()
        # obs_values = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs_values = self._obs_wrapper(obs)
        return obs_values, graph

    def step(self, action_tl, action_ve):

        tl_action_select = {}
        for tl_index in range(len(self.env.all_tls)):
            tl_action_select[self.env.all_tls[tl_index]] = \
                    (self.env._crosses[self.env.all_tls[tl_index]].green_phases)[action_tl[tl_index]]
        obs, reward, done_tl, do_vehicle, do_emv, all_reward, all_obs, ve_reward, emv_reward, done_ve, all_ve_reward, hetero_graph, obs_change_ve, obs_change_emv, new_ves,new_emv, emv, ve, travel_info  = self.env.step(tl_action_select, action_ve)
            # obs = self.batch(obs, config['environment']['state_key'], self.env.all_tls)
        obs = self._obs_wrapper(obs)
        reward = self.get_reward(reward, self.env.all_tls)
        reward = reward.reshape(self.num_agents, 1)
            # if self.share_reward:
            #     global_reward = np.sum(reward)
            #     reward = [[global_reward]] * self.num_agents

            # info['individual_reward'] = reward
        ve_reward = self.get_reward_ve(ve_reward, emv_reward, self.env.num_vehicle_agents, self.env.num_emv_agents)
        # done_ve = np.array([done_ve] * (self.env.num_vehicle_agents+self.env.num_emv_agents))
        done = np.array([done_tl] * self.num_agents)
        done_ve = np.array([done_ve] * (self.env.num_vehicle_agents + self.env.num_emv_agents))
        do_action = self.get_done_ve(do_vehicle, do_emv, self.env.num_vehicle_agents, self.env.num_emv_agents)
        obs_change = self.get_done_ve(obs_change_ve, obs_change_emv, self.env.num_vehicle_agents, self.env.num_emv_agents)
            # info = self._info_wrapper(info)
        return obs, reward, done, do_action, all_reward, all_obs, ve_reward, done_ve, all_ve_reward, hetero_graph, obs_change, new_ves,new_emv, emv, ve, travel_info
        # else:
        #     obs, ve_reward, emv_reward, done, info, graph = self.env.step_ve(action_ve)
        #
        #     obs = self._obs_wrapper(obs)
        #     reward = self.get_reward_ve(ve_reward, emv_reward, self.env.num_vehicle_agents, self.env.num_emv_agents)
        #     done = np.array([done] * (self.env.num_vehicle_agents+self.env.num_emv_agents))
        #     return obs, reward, done, info, graph
        # done = np.array([done] * self.num_agents)
        # return ve_obs, ve_reward, done, all_ve_reward



    def seed(self, seed=None):
        if seed is None:
            random.seed(1)
        else:
            random.seed(seed)

    def close(self):
        self.env.terminate()

    def _obs_wrapper(self, obs):
        if self.num_agents == 1:
            return obs[np.newaxis, :]
        else:
            return obs

