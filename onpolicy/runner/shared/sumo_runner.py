import copy
import time
import numpy as np
import torch
from CoEMV_graph.onpolicy.runner.shared.base_runner import Runner
import wandb
import imageio
import copy
import sys

def _t2n(x):
    #将PyTorch张量转换为NumPy数组
    return x.detach().cpu().numpy()

def _n2t(x,device):
    return torch.from_numpy(x).to(device)

class SUMORunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""
    def __init__(self, config):
        super(SUMORunner, self).__init__(config)

    def run(self):
        #设置 ε-greedy 策略：定义初始的探索率和逐步减小的探索率
        self.epsilon = self.all_args.epsilon
        self.anneal_epsilon = (self.all_args.epsilon - self.all_args.min_epsilon) / self.all_args.anneal_steps
        #调用 warmup 方法：进行环境的预热，通常用于初始化
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            #线性学习率衰减
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)
            self.trainer.policy.lr_decay_gnn(episode, episodes)
            self.trainer_ve.policy.lr_decay_gnn(episode, episodes)
            self.reset_buffer()
            # self.step_ve = np.zeros((1, self.num_vehicle_agents))
            # self.step_emv = np.zeros((1, self.num_emv_agents))
            for step in range(10000000):
                print('-----step', step)
                # step_ = np.full((1, 1), step)
                values_tl, actions_tl, action_log_probs_tl, rnn_states_tl, rnn_states_critic_tl, actions_env_tl = self.collect(step//15)
                values_ve, actions_ve, action_log_probs_ve, rnn_states_ve, rnn_states_critic_ve, actions_env_ve = self.collect_ve(step)
                # values_emv, actions_emv, action_log_probs_emv, rnn_states_emv, rnn_states_critic_emv, actions_env_emv = self.collect_emv(step)
                # actions_v = np.concatenate((actions_vehicle, actions_emv), axis=1)
                obs_tl, rewards_tl, dones_tl, return_ve, infos_tl, obs_ve, rewards_ve, dones_ve, infos_ve, graph_ve, obs_change_ve, new_ves, new_emv, emv, ve  = self.envs.step(actions_tl.astype(np.int64)[:, :, 0], actions_ve.astype(np.int64)[:, :, 0])
                # graph_ve = self.trainer_ve.policy.get_state(graph_ve)
                obs_vehicle, obs_emv, rewards_vehicle, rewards_emv, dones_vehicle, dones_emv, return_vehicle, return_emv, obs_change_vehicle, obs_change_emv = self.separate(
                    obs_ve, rewards_ve, dones_ve, return_ve, obs_change_ve)
                # if episode % 25 == 0:
                #     graph_tl, graph_ve = self.get_graph_state(graph_ve)
                #     if step % 30 ==0:
                #         self.train_gnn(graph_tl, graph_ve, obs_tl, obs_vehicle, obs_emv)
                # else:
                with torch.no_grad():
                    graph_tl, graph_ve = self.get_graph_state(graph_ve)
                # intrinsic_reward_tl = self.compute_reward(graph_tl['signal_light'],graph_tl['emergency'], _n2t(rewards_emv, self.device).squeeze(0))
                # intrinsic_reward_ve = self.compute_reward(graph_ve['vehicle'], graph_ve['emergency'], _n2t(rewards_emv, self.device).squeeze(0))
                # rewards_vehicle = rewards_vehicle + intrinsic_reward_ve
                # rewards_ve = np.concatenate((rewards_vehicle, rewards_emv), axis=1)
                # rewards_tl = rewards_tl + intrinsic_reward_tl
                if (step+1) % 15 == 0:
                    self.ava = self.envs.get_unava_phase_index()
                    available_actions_tl = self.get_ava_actions(self.ava)
                    data_tl = obs_tl, rewards_tl, dones_tl, infos_tl, values_tl, actions_tl, action_log_probs_tl, rnn_states_tl, rnn_states_critic_tl, available_actions_tl, graph_tl
                    self.insert_tl_0(data_tl, step)
                if step % 15 == 0:
                    self.ava = self.envs.get_unava_phase_index()
                    available_actions_tl = self.get_ava_actions(self.ava)
                    data_tl = obs_tl, rewards_tl, dones_tl, infos_tl, values_tl, actions_tl, action_log_probs_tl, rnn_states_tl, rnn_states_critic_tl, available_actions_tl
                    self.insert_tl_1(data_tl)
                available_actions_vehicle, available_actions_emv = self.get_ava_actions_ve()
                available_actions_ve = np.concatenate((available_actions_vehicle, available_actions_emv),axis=1)
                # data_vehicle = obs_vehicle, rewards_vehicle, dones_vehicle, infos_ve, values_vehicle, actions_vehicle, action_log_probs_vehicle, rnn_states_vehicle, rnn_states_critic_vehicle, available_actions_vehicle, return_vehicle, obs_change_ve,graph_ve, obs_ve
                # data_emv = obs_emv, rewards_emv, dones_emv, infos_ve, values_emv, actions_emv, action_log_probs_emv, rnn_states_emv, rnn_states_critic_emv, available_actions_emv, return_emv, obs_change_emv,graph_emv, obs_ve
                data_ve = obs_ve, rewards_ve, dones_ve, infos_ve, values_ve, actions_ve, action_log_probs_ve, rnn_states_ve, rnn_states_critic_ve, available_actions_ve, return_ve, obs_change_ve, graph_ve
                self.insert_ve_0(data_ve, self.num_ve_agents, step)
                self.insert_ve_1(data_ve, self.num_ve_agents)
                # self.insert_ve_0(data_emv, self.num_emv_agents, step,"emv")
                # self.insert_ve_1(data_emv, self.num_emv_agents,"emv")
                # if step == 20:
                #     self.trainer_ve.train_feature(torch.tensor(self.buffer.graph_obs[self.buffer.step]).to(self.device), torch.tensor(self.buffer.obs[self.buffer.step]).to(self.device))

                ##### episide decay
                self.all_args.epsilon = self.all_args.epsilon - self.anneal_epsilon if self.all_args.epsilon > self.all_args.min_epsilon else self.all_args.epsilon
                if dones_ve.all() == True:
                    break
            self.warmup()
            # compute return and update network
            # 原来的sumo代码中有self.warmup()
            if episode % 1 == 0:
                self.compute()
                train_infos, train_infos_ve = self.train()
                # sys.exit()
                # post process
                total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

                # save model
                # if (total_num_steps % self.save_interval == 0 or episode == episodes - 1):
                if (episode % self.save_interval == 0 or episode == episodes - 1):
                    self.save(episode)

                # log information
                # if total_num_steps % self.log_interval == 0:
                if episode % self.log_interval == 0:
                    end = time.time()
                    print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                            .format(self.all_args.scenario_name,
                                    self.algorithm_name,
                                    self.experiment_name,
                                    episode,
                                    episodes,
                                    total_num_steps,
                                    self.num_env_steps,
                                    int(total_num_steps / (end - start))))

                    if self.env_name == "SUMO":
                        env_infos_tl = {}
                        env_infos_ve = {}
                        env_infos_emv = {}
                        for info in infos_tl:
                            for k, v in infos_tl[info].items():
                                if k not in env_infos_tl:
                                    env_infos_tl[k] = []
                                env_infos_tl[k].append(v)
                        # for info in infos_ve:
                        for agent_id in range(self.num_vehicle_agents+self.num_emv_agents):
                            if agent_id < self.num_vehicle_agents:
                                for k, v in infos_ve[str(agent_id)].items():
                                    if k not in env_infos_ve:
                                        env_infos_ve[k] = []
                                    env_infos_ve[k].append(v)
                            else:
                                for k, v in infos_ve["emergency_"+str(agent_id-self.num_vehicle_agents)].items():
                                    if k not in env_infos_emv:
                                        env_infos_emv[k] = []
                                    env_infos_emv[k].append(v)

                    train_infos["average_episode_rewards"] = np.sum(self.buffer.rewards) / (np.sum(self.buffer.step) * self.num_agents)
                    train_infos_ve["average_episode_rewards"] = np.sum(self.ve_buffer.rewards) / np.sum(self.ve_buffer.step_ve[(self.ve_buffer.step_ve != 1) & (self.ve_buffer.step_ve != 0)]-1)
                    # train_infos_emv["average_episode_rewards"] = np.sum(self.emv_buffer.rewards) / np.sum(self.emv_buffer.step_ve[(self.emv_buffer.step_ve != 1) & (self.emv_buffer.step_ve != 0)]-1)
                    # print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                    self.log_train_tl(train_infos, total_num_steps)
                    self.log_train_vehicle(train_infos_ve, total_num_steps)
                    # self.log_train_emv(train_infos_emv, total_num_steps)

                    self.log_env_tl(env_infos_tl, total_num_steps)
                    self.log_env_vehicle(env_infos_ve, total_num_steps)
                    self.log_env_emv(env_infos_emv, total_num_steps)
                # self.log

            # eval
            # if total_num_steps % self.eval_interval == 0 and self.use_eval:
                if episode % self.eval_interval == 0 and self.use_eval:
                    self.eval(total_num_steps)

    def get_graph_state(self, graph):
        graph_tl = self.trainer.policy.get_state(graph)
        graph_ve = self.trainer_ve.policy.get_state(graph)
        # graph_emv = self.trainer_emv.policy.get_state(graph)
        return graph_tl, graph_ve

    def train_gnn(self, graph_tl, graph_ve, obs_tl, obs_ve, obs_emv):
        masks_tl = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks_ve = np.ones((self.n_rollout_threads, self.num_vehicle_agents, 1), dtype=np.float32)
        masks_emv = np.ones((self.n_rollout_threads, self.num_emv_agents, 1), dtype=np.float32)

        out_tl, rnn_state_tl = self.trainer.policy.rnn(_n2t(obs_tl.squeeze(0), self.device), self.rnn_state_tl.to(self.device), _n2t(masks_tl.squeeze(0),self.device))
        out_ve, rnn_state_ve = self.trainer_ve.policy.rnn(_n2t(obs_ve.squeeze(0), self.device), self.rnn_state_ve.to(self.device), _n2t(masks_ve.squeeze(0), self.device))
        out_emv, rnn_state_emv = self.trainer_ve.policy.rnn(_n2t(obs_emv.squeeze(0), self.device), self.rnn_state_emv.to(self.device), _n2t(masks_emv.squeeze(0), self.device))

        self.trainer.train_gnn(graph_tl['signal_light'], graph_tl['emergency'], rnn_state_tl)
        self.trainer_ve.train_gnn_emv(graph_ve['vehicle'], graph_ve['emergency'], rnn_state_ve, rnn_state_emv)
        # self.trainer_emv.train_gnn_emv(graph_emv['emergency'], rnn_state_emv)
        self.rnn_state_tl = rnn_state_tl.detach()
        self.rnn_state_ve = rnn_state_ve.detach()
        self.rnn_state_emv = rnn_state_emv.detach()

        # self.trainer_emv.train_gnn(graph_emv['emergency'], obs_state_emv, rnn_state_emv)

    def reset_buffer(self):
        self.ve_buffer.reset()
        self.buffer.reset_tl()
        # self.emv_buffer.reset()

    def warmup(self):
        # reset env
        # self.envs.reset()
        obs, graph = self.envs.reset()
        graph_obs = self.trainer.policy.get_state(graph)
        # out, rnn_state_gnn = self.trainer.policy.rnn(obs)
        # self.rnn_state = rnn_state_gnn
        # out, rnn_state_gnn = self.trainer.policy.get_hidden_states(obs)
        # graph_tl = self.policy.get_dict(graph_obs['signal_light']).to(self.device)
        # obs_tl = torch.tensor(obs, dtype=torch.float32).squeeze(0).to(self.device)
        #
        # self.trainer.train_feature(graph_tl, obs_tl)
        # obs_ve = self.envs.reset_ve()
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs
        self.buffer.graph_obs[0] = graph_obs['signal_light'].detach()
        # self.buffer.rnn_states_gnn[0] = out
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
        self.ava = self.envs.get_unava_phase_index()
        # print("***********")
        # self.trainer.train_feature(self.buffer.graph_obs[0].squeeze(0).to(self.device),
        #                            torch.tensor(self.buffer.obs[0].squeeze(0)).to(self.device))
        available_actions = self.get_ava_actions(self.ava)
        self.buffer.available_actions[0] = available_actions.copy()
        # for i in range(self.num_agents):
        #     self.buffer.graph[0][0][i] = copy.deepcopy(graph)

        #################### fillin actor features：：

    def get_ava_actions(self, ava):
        available_actions = np.ones((self.all_args.n_rollout_threads, self.all_args.num_agents, self.all_args.num_actions))
        if len(ava.shape) == 2:
            for i in range(self.all_args.num_agents):
                for j in range(self.all_args.n_rollout_threads):
                    available_actions[j, i, ava[j][i]] = 0
        elif ava is not None and ava.shape[-1] != 0:
            for i in range(self.all_args.n_rollout_threads):
                for j in range(self.all_args.num_agents):
                    available_actions[i, j, ava[i][j][0]] = 0

        return available_actions

    def get_ava_actions_ve(self):
        available_actions_ve = np.ones((self.all_args.n_rollout_threads, self.all_args.num_vehicle_agents, self.all_args.num_ve_actions))
        available_actions_emv = np.ones((self.all_args.n_rollout_threads, self.all_args.num_emv_agents, self.all_args.num_ve_actions))
        unava = self.envs.get_unava_index()
        unava_vehicle = unava[0][0]
        unava_emv = unava[0][1]
        for k, v in unava_vehicle.items():
            for i in v:
                available_actions_ve[0, int(k), i] = 0
        for k, v in unava_emv.items():
            for i in v:
                available_actions_emv[0, int(k.split('_')[-1]), i] = 0
        return available_actions_ve, available_actions_emv

    def get_ava_actions_ve_eval(self):
        available_actions_ve = np.ones((self.all_args.n_rollout_threads, self.all_args.num_vehicle_agents, self.all_args.num_ve_actions))
        available_actions_emv = np.ones((self.all_args.n_rollout_threads, self.all_args.num_emv_agents, self.all_args.num_ve_actions))
        unava = self.eval_envs.get_unava_index()
        unava_vehicle = unava[0][0]
        unava_emv = unava[0][1]
        for k, v in unava_vehicle.items():
            for i in v:
                available_actions_ve[0, int(k), i] = 0
        for k, v in unava_emv.items():
            for i in v:
                available_actions_emv[0, int(k.split('_')[-1]), i] = 0
        return available_actions_ve, available_actions_emv

    def separate(self, obs_ve, rewards_ve, dones_ve, return_ve, obs_change):
        return obs_ve[:,:self.num_vehicle_agents,:], obs_ve[:,self.num_vehicle_agents:,:], rewards_ve[:,:self.num_vehicle_agents,:], rewards_ve[:,self.num_vehicle_agents:,:], dones_ve[:,:self.num_vehicle_agents], dones_ve[:,self.num_vehicle_agents:], return_ve[:,:self.num_vehicle_agents], return_ve[:,self.num_vehicle_agents:], obs_change[:, :self.num_vehicle_agents], obs_change[:, self.num_vehicle_agents:]

    def collect_ve(self, step):
        self.trainer_ve.prep_rollout()
        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer_ve.policy.get_actions(
            np.concatenate([self.ve_buffer.share_obs[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            np.concatenate([self.ve_buffer.obs[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            np.concatenate([self.ve_buffer.rnn_states[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            np.concatenate([self.ve_buffer.rnn_states_critic[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            np.concatenate([self.ve_buffer.masks[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            np.concatenate([_t2n(self.ve_buffer.graph_obs[self.ve_buffer.step_ve[i]-1,:,i,:]) for i in range(self.num_ve_agents)]),
            available_actions=np.concatenate([self.ve_buffer.available_actions[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)]),
            trans_masks = np.concatenate([self.ve_buffer.trans_masks[self.ve_buffer.step_ve[i]-1,:,i,:] for i in range(self.num_ve_agents)])
        )
        # print(np.concatenate(self.ve_buffer.obs[step]),"ve_buffer_obs")
        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        if self.envs.vehicle_action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.vehicle_action_space[0].shape):
                uc_actions_env = np.eye(self.envs.vehicle_action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.vehicle_action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.vehicle_action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()


        values, actions, action_log_probs, rnn_states, rnn_states_critic = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
            np.concatenate(_t2n(self.buffer.graph_obs[step])),
            available_actions=np.concatenate(self.buffer.available_actions[step]),
            trans_masks=np.concatenate(self.buffer.trans_masks[step])
        )

        values = np.array(np.split(_t2n(values), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(actions), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_probs), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        # actions_env = [actions[idx, :, 0] for idx in range(self.n_rollout_threads)]
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':

            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert_ve_0(self, data, num_ve_agents, step, ve_type="vehicle"):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, return_ve, obs_change_ve, graph = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size),
                                             dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, num_ve_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, num_ve_agents, 1), dtype=np.float32)
        graph_ve = torch.cat((graph['vehicle'], graph['emergency']), dim = 0)

        if ve_type=="vehicle":
            if self.use_centralized_V:
                share_obs = obs.reshape(self.n_rollout_threads, -1)
                share_obs = np.expand_dims(share_obs, 1).repeat(self.num_ve_agents, axis=1)
            else:
                share_obs = obs
            rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.ve_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
            self.ve_buffer.insert_ve_0(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, obs_change_ve, graph_ve.detach(),available_actions=available_actions, active_masks=active_masks)
        # else:
        #     if self.use_centralized_V:
        #         share_obs = obs.reshape(self.n_rollout_threads, -1)
        #         share_obs = np.expand_dims(share_obs, 1).repeat(self.num_emv_agents, axis=1)
        #     else:
        #         share_obs = obs
        #     rnn_states_critic[dones == True] = np.zeros(
        #         ((dones == True).sum(), *self.emv_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        #     self.emv_buffer.insert_ve_0(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values,
        #                                rewards, masks, obs_change_ve, graph['emergency'].detach(), available_actions=available_actions, active_masks=active_masks)


    def insert_ve_1(self, data, num_ve_agents, ve_type = "vehicle"):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, return_ve, obs_change_ve, graph = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, num_ve_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_ve_agents, axis=1)
        else:
            share_obs = obs
        if ve_type=="vehicle":
            rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.ve_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
            self.ve_buffer.insert_ve_1(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, return_ve, available_actions=available_actions)
        # else:
        #     rnn_states_critic[dones == True] = np.zeros(
        #         ((dones == True).sum(), *self.emv_buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        #     self.emv_buffer.insert_ve_1(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, return_ve, available_actions=available_actions)

    def insert_tl_1(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert_1(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, available_actions=available_actions)

    def insert_tl_0(self, data, step):
        # self.trainer_tl.eval_gnn()
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic, available_actions, graph = data
        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert_0(share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks, graph['signal_light'].detach(), available_actions=available_actions, active_masks = active_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards_tl = []
        eval_episode_rewards_vehicle = np.zeros((self.n_rollout_threads, self.num_vehicle_agents, 1), dtype=np.float32)
        eval_episode_rewards_emv = np.zeros((self.n_rollout_threads, self.num_emv_agents, 1), dtype=np.float32)

        eval_obs_tl = self.eval_envs.reset()[0]
        eval_rnn_states_tl = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks_tl = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        self.ava = self.eval_envs.get_unava_phase_index()
        available_actions_tl = self.get_ava_actions(self.ava)
        graph = self.eval_envs.reset()[1]
        graph_obs = self.trainer.policy.get_state(graph)
        eval_graph_tl = _t2n(graph_obs['signal_light'].to(torch.float))

        eval_obs_vehicle = np.zeros((self.n_rollout_threads, *self.ve_buffer.obs.shape[2:]), dtype=np.float32)
        eval_rnn_states_vehicle = np.zeros((self.n_rollout_threads, *self.ve_buffer.rnn_states.shape[2:]),dtype=np.float32)
        eval_masks_vehicle = np.ones((self.n_rollout_threads, self.num_vehicle_agents, 1), dtype=np.float32)
        eval_graph_vehicle = np.zeros(self.ve_buffer.graph_obs.shape[2:], dtype=np.float32)

        eval_obs_emv = np.zeros((self.n_rollout_threads, *self.emv_buffer.obs.shape[2:]), dtype=np.float32)
        eval_rnn_states_emv = np.zeros((self.n_rollout_threads, *self.emv_buffer.rnn_states.shape[2:]),
                                           dtype=np.float32)
        eval_masks_emv = np.ones((self.n_rollout_threads, self.num_emv_agents, 1), dtype=np.float32)
        eval_graph_emv = np.zeros(self.emv_buffer.graph_obs.shape[2:], dtype=np.float32)
        available_actions_vehicle = np.ones(
            (self.all_args.n_rollout_threads, self.all_args.num_vehicle_agents, self.all_args.num_ve_actions))
        available_actions_emv = np.ones(
            (self.all_args.n_rollout_threads, self.all_args.num_emv_agents, self.all_args.num_ve_actions))

        for eval_step in range(10000000):
            self.trainer.prep_rollout()
            print(eval_step)

            eval_action_tl, rnn_states_tl = self.trainer.policy.act(np.concatenate(eval_obs_tl),
                                                np.concatenate(eval_rnn_states_tl),
                                                np.concatenate(eval_masks_tl),
                                                eval_graph_tl,
                                                available_actions = np.concatenate(available_actions_tl),
                                                deterministic=True)
            eval_action_vehicle, rnn_states_vehicle = self.trainer_ve.policy.act(np.concatenate(eval_obs_vehicle),
                                                                   np.concatenate(eval_rnn_states_vehicle),
                                                                   np.concatenate(eval_masks_vehicle),
                                                                    eval_graph_vehicle,
                                                                    available_actions=available_actions_vehicle.squeeze(0),
                                                                   deterministic=True)
            eval_action_emv, rnn_states_emv = self.trainer_ve.policy.act(np.concatenate(eval_obs_emv),
                                                                   np.concatenate(eval_rnn_states_emv),
                                                                   np.concatenate(eval_masks_emv),
                                                                    eval_graph_emv.astype(np.float32),
                                                                    available_actions=available_actions_emv.squeeze(0),
                                                                   deterministic=True)

            eval_actions_tl = np.array(np.split(_t2n(eval_action_tl), self.n_eval_rollout_threads))
            rnn_states_tl = np.array(np.split(_t2n(rnn_states_tl), self.n_eval_rollout_threads))
            eval_actions_vehicle = np.array(np.split(_t2n(eval_action_vehicle), self.n_eval_rollout_threads))
            rnn_states_vehicle = np.array(np.split(_t2n(rnn_states_vehicle), self.n_eval_rollout_threads))
            eval_actions_emv = np.array(np.split(_t2n(eval_action_emv), self.n_eval_rollout_threads))
            rnn_states_emv = np.array(np.split(_t2n(rnn_states_emv), self.n_eval_rollout_threads))
            eval_actions_ve = np.concatenate((eval_actions_vehicle, eval_actions_emv), axis=1)
            # Obser reward and next obs
            # eval_obs_tl, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            obs_tl, eval_rewards_tl, eval_dones_tl, eval_return_ve, eval_infos_tl, eval_obs_ve, eval_rewards_ve, eval_dones_ve, eval_infos_ve, eval_graph, eval_obs_change_ve, eval_new_ves, eval_new_emv, eval_ves, eval_emv = self.eval_envs.step(eval_actions_tl.astype(np.int64)[:, :, 0], eval_actions_ve.astype(np.int64)[:, :, 0])
            eval_obs_vehicle, eval_obs_emv, eval_rewards_vehicle, eval_rewards_emv, eval_dones_vehicle, eval_dones_emv, eval_return_vehicle, eval_return_emv, eval_obs_change_vehicle, eval_obs_change_emv = self.separate(
                eval_obs_ve, eval_rewards_ve, eval_dones_ve, eval_return_ve, eval_obs_change_ve)
            available_actions_vehicle, available_actions_emv = self.get_ava_actions_ve_eval()
            available_actions_ve = np.concatenate((available_actions_vehicle, available_actions_emv), axis=1)
            self.ava = self.eval_envs.get_unava_phase_index()
            available_actions_tl = self.get_ava_actions(self.ava)

            with torch.no_grad():
                graph_tl, graph_ve = self.get_graph_state(eval_graph)
                graph_tl = _t2n(graph_tl['signal_light'].to(torch.float))
                graph_vehicle = _t2n(graph_ve['vehicle'].to(torch.float))
                graph_emv = _t2n(graph_ve['emergency'].to(torch.float))
                graph_ve = np.concatenate((graph_vehicle,graph_emv), axis=0)
            if (eval_step+1) % 15 == 0:
                self.ava = self.eval_envs.get_unava_phase_index()
                available_actions_tl = self.get_ava_actions(self.ava)
                eval_obs_tl = obs_tl
                eval_graph_tl = graph_tl
                eval_episode_rewards_tl.append(eval_rewards_tl)
            if eval_step % 15 == 0:
                eval_rnn_states_tl = rnn_states_tl

            eval_masks_tl = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks_tl[eval_dones_tl == True] = np.zeros(((eval_dones_tl == True).sum(), 1), dtype=np.float32)
            eval_rnn_states_tl[eval_dones_tl == True] = np.zeros(
                ((eval_dones_tl == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_rnn_states_vehicle[eval_dones_vehicle == True] = np.zeros(
                ((eval_dones_vehicle == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks_vehicle = np.ones((self.n_eval_rollout_threads, self.num_vehicle_agents, 1), dtype=np.float32)
            eval_masks_vehicle[eval_dones_vehicle == True] = np.zeros(((eval_dones_vehicle == True).sum(), 1),
                                                                      dtype=np.float32)
            #
            eval_rnn_states_emv[eval_dones_emv == True] = np.zeros(
                ((eval_dones_emv == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks_emv = np.ones((self.n_eval_rollout_threads, self.num_emv_agents, 1), dtype=np.float32)
            eval_masks_emv[eval_dones_emv == True] = np.zeros(((eval_dones_emv == True).sum(), 1), dtype=np.float32)

            for i, r in enumerate(eval_obs_change_vehicle[0]):
                if r[0] == True:
                    eval_episode_rewards_vehicle[:, i, :] += eval_rewards_vehicle[:, i, :]
                    eval_graph_vehicle[i, :] = graph_vehicle[i, :]
                if eval_return_vehicle[0][i] == True:
                    eval_rnn_states_vehicle[:, i, :] = rnn_states_vehicle[:, i, :]

            for i, r in enumerate(eval_obs_change_emv[0]):
                if r[0] == True:
                    eval_episode_rewards_emv[:, i, :] += eval_rewards_emv[:, i, :]
                    eval_graph_emv[i, :] = graph_emv[i, :]
                if eval_return_emv[0][i] == True:
                    eval_rnn_states_emv[:, i, :] = rnn_states_emv[:, i, :]
            if eval_dones_ve.all() == True:
                break
        eval_episode_rewards_vehicle = eval_episode_rewards_vehicle[:,:self.num_vehicle_agents,:]
        eval_episode_rewards_emv = eval_episode_rewards_emv[:,self.num_emv_agents:,:]
        eval_episode_rewards_tl = np.array(eval_episode_rewards_tl)
        eval_env_infos = {}
        eval_env_infos['eval_average_episode_rewards_tl'] = np.sum(np.array(eval_episode_rewards_tl), axis=0)
        eval_average_episode_rewards_tl = np.mean(eval_env_infos['eval_average_episode_rewards_tl'])

        # eval_episode_rewards_vehicle = np.array(eval_episode_rewards_vehicle)
        eval_env_infos_vehicle = {}
        eval_env_infos_vehicle['eval_average_episode_rewards_vehicle'] = np.nansum(np.array(eval_episode_rewards_vehicle), axis=0)
        eval_average_episode_rewards_vehicle = np.mean(eval_env_infos_vehicle['eval_average_episode_rewards_vehicle'])

        # eval_episode_rewards_emv = np.array(eval_episode_rewards_emv)
        eval_env_infos_emv = {}
        eval_env_infos_emv['eval_average_episode_rewards_emv'] = np.nansum(np.array(eval_episode_rewards_emv), axis=0)
        eval_average_episode_rewards_emv = np.mean(eval_env_infos_emv['eval_average_episode_rewards_emv'])

        print("eval average episode tl rewards of agent: " + str(eval_average_episode_rewards_tl))
        print("eval average episode vehicle rewards of agent: " + str(eval_average_episode_rewards_vehicle))
        print("eval average episode emv rewards of agent: " + str(eval_average_episode_rewards_emv))
        self.log_eval_tl(eval_env_infos, total_num_steps)
        self.log_eval_ve(eval_env_infos_vehicle, total_num_steps)
        self.log_eval_emv(eval_env_infos_emv, total_num_steps)
        self.log_eval_info_tl(eval_infos_tl)
        self.log_eval_info_ve(eval_infos_ve)

    @torch.no_grad()
    def render(self):
        print("render")
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render('rgb_array')[0][0]
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(np.concatenate(obs),
                                                    np.concatenate(rnn_states),
                                                    np.concatenate(masks),
                                                    deterministic=True)
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i]+1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render('human')

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
