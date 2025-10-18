import sys

import wandb
import os
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from CoEMV_graph.onpolicy.utils.shared_buffer import SharedReplayBuffer
from CoEMV_graph.onpolicy.utils.separated_buffer import SeparatedReplayBuffer
import logging
import os
import time


def create_logger(logger_file_path, logger_name):
    if not os.path.exists(logger_file_path):
        os.makedirs(logger_file_path)

    log_name = '{}.log'.format(time.strftime('%Y-%m-%d-%H-%M'))
    final_log_file = os.path.join(logger_file_path, log_name)

    # 使用独立的 logger 名称，避免使用 root logger
    logger = logging.getLogger(logger_name)  # 设定日志对象
    logger.setLevel(logging.INFO)  # 设定日志等级

    # 设置输出格式
    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s ")

    # 设置文件输出和控制台输出
    file_handler = logging.FileHandler(final_log_file)  # 文件输出
    console_handler = logging.StreamHandler()  # 控制台输出

    # 设置文件和控制台的输出格式
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加 handler 到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):
    """
    Base class for training recurrent policies.
    :param config: (dict) Config dictionary containing parameters for training.
    """
    def __init__(self, config):

        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']
        self.num_vehicle_agents = config['num_vehicle_agents']
        self.num_emv_agents = config['num_emv_agents']

        if config.__contains__("render_envs"):
            self.render_envs = config['render_envs']       

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.n_render_rollout_threads = self.all_args.n_render_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N
        self.step_tls = 0
        self.step_ve = np.zeros((1, self.num_vehicle_agents))
        self.step_emv = np.zeros((1, self.num_emv_agents))
        self.num_ve_agents = self.all_args.num_ve_agents

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval
        self.tl_state = np.ones((self.n_rollout_threads, 1), dtype=bool)
        self.ve_state = np.zeros((self.n_rollout_threads, 1), dtype=bool)
        self.rnn_state_tl = torch.zeros(self.num_agents, 1, self.hidden_size)
        self.rnn_state_ve = torch.zeros(self.num_vehicle_agents, 1, self.hidden_size)
        self.rnn_state_emv = torch.zeros(self.num_emv_agents, 1, self.hidden_size)
        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_wandb:
            self.save_dir = str(wandb.run.dir)
            self.run_dir = str(wandb.run.dir)
        else:
            self.run_dir = config["run_dir"]
            self.log_dir = str(self.run_dir / 'logs')
            if not os.path.exists(self.log_dir):
                os.makedirs(self.log_dir)
            self.writter = SummaryWriter(self.log_dir)
            self.save_dir = str(self.run_dir / 'models-{}'.format(time.strftime('%Y-%m-%d-%H-%M')))
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)




        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            from CoEMV_graph.onpolicy.algorithms.mat.mat_trainer import MATTrainer as TrainAlgo
            from CoEMV_graph.onpolicy.algorithms.mat.algorithm.transformer_policy import TransformerPolicy as Policy
        else:
            from CoEMV_graph.onpolicy.algorithms.r_mappo.r_mappo import R_MAPPO as TrainAlgo
            from CoEMV_graph.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy as Policy
        # if self.all_args.use_ours:
        #     from CoEMV_origin.onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy_Trans as Policy



        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]
        vehicle_share_observation_space = self.envs.vehicle_share_observation_space[0] if self.use_centralized_V else self.envs.vehicle_observation_space[0]
        emv_share_observation_space = self.envs.emv_share_observation_space[0] if self.use_centralized_V else self.envs.vehicle_observation_space[0]
        # policy network

        self.policy = Policy(self.all_args, self.envs.observation_space[0], share_observation_space, self.envs.action_space[0], device=self.device, type = 'signal')
        self.policy_ve = Policy(self.all_args, self.envs.vehicle_observation_space[0], vehicle_share_observation_space,
                             self.envs.vehicle_action_space[0], device=self.device, type = 'vehicle')
        self.policy_emv = Policy(self.all_args, self.envs.vehicle_observation_space[0], emv_share_observation_space,
                             self.envs.vehicle_action_space[0], device=self.device, type = 'emergency')

        if self.model_dir is not None:
            self.restore(self.model_dir)
        self.logger_tl = create_logger(str(self.run_dir / 'train_tl'), 'logger_tl')
        self.logger_ve = create_logger(str(self.run_dir / 'train_ve'), 'logger_ve')
        self.logger_emv = create_logger(str(self.run_dir / 'train_emv'), 'logger_emv')
        self.logger_tl_eval = create_logger(str(self.run_dir / 'eval_tl'), 'logger_tl_eval')
        self.logger_ve_eval = create_logger(str(self.run_dir / 'eval_ve'), 'logger_ve_eval')
        self.logger_emv_eval = create_logger(str(self.run_dir / 'eval_emv'), 'logger_emv_eval')
        # else:
        #     self.logger_tl = create_logger(str(self.run_dir / 'train_tl'), 'logger_tl')
        #     self.logger_ve = create_logger(str(self.run_dir / 'train_ve'), 'logger_ve')
        #     self.logger_emv = create_logger(str(self.run_dir / 'train_emv'), 'logger_emv')


        # algorithm
        self.trainer = TrainAlgo(self.all_args, self.policy, self.num_agents, device = self.device)
        self.trainer_ve = TrainAlgo(self.all_args, self.policy_ve,self.num_vehicle_agents, device=self.device)
        self.trainer_emv = TrainAlgo(self.all_args, self.policy_ve, self.num_emv_agents,device=self.device)
        
        # buffer
        self.buffer = SharedReplayBuffer(self.all_args,
                                        self.num_agents,
                                        self.envs.observation_space[0],
                                        share_observation_space,
                                        self.envs.action_space[0],
                                         self.num_agents)

        self.ve_buffer = SharedReplayBuffer(self.all_args,
                                            self.num_vehicle_agents,
                                            self.envs.vehicle_observation_space[0],
                                            vehicle_share_observation_space,
                                            self.envs.vehicle_action_space[0],
                                            self.num_vehicle_agents)
        #
        self.emv_buffer = SharedReplayBuffer(self.all_args,
                                             self.num_emv_agents,
                                             self.envs.vehicle_observation_space[0],
                                             vehicle_share_observation_space,
                                             self.envs.vehicle_action_space[0],
                                             self.num_emv_agents)

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError
    
    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        self.trainer_ve.prep_rollout()
        # self.trainer_emv.prep_rollout()
        # print(self.ve_buffer.step_ve[1111]-1)
        # print(self.ve_buffer.share_obs[self.ve_buffer.step_ve[1111]-1,0 , 0,:])
        # print(self.ve_buffer.share_obs[self.ve_buffer.step_ve[1]-1, 0, 1, :])
        # print(np.array([self.ve_buffer.share_obs[self.ve_buffer.step_ve[0]-1,0 , 0,:],self.ve_buffer.share_obs[self.ve_buffer.step_ve[1], 0, 1, :]]).shape)
        ve_share_obs = np.array([self.ve_buffer.share_obs[self.ve_buffer.step_ve[i]-1,0 , i,:] for i in range(self.num_ve_agents)])
        ve_rnn_states_critic = np.array([self.ve_buffer.rnn_states_critic[self.ve_buffer.step_ve[i]-1,0,i,:] for i in range(self.num_ve_agents)])
        ve_masks = np.array([self.ve_buffer.masks[self.ve_buffer.step_ve[i]-1, 0, i, :] for i in range(self.num_ve_agents)])
        ve_graph = np.array([_t2n(self.ve_buffer.graph_obs[self.ve_buffer.step_ve[i]-1, 0, i, :]) for i in range(self.num_ve_agents)])

        # emv_share_obs = np.array(
        #     [self.emv_buffer.share_obs[self.emv_buffer.step_ve[i]-1, 0, i, :] for i in range(self.num_emv_agents)])
        # emv_rnn_states_critic = np.array(
        #     [self.emv_buffer.rnn_states_critic[self.emv_buffer.step_ve[i]-1, 0, i, :] for i in range(self.num_emv_agents)])
        # emv_masks = np.array([self.ve_buffer.masks[self.emv_buffer.step_ve[i]-1, 0, i, :] for i in range(self.num_emv_agents)])
        # emv_graph = np.array(
        #     [_t2n(self.ve_buffer.graph_obs[self.emv_buffer.step_ve[i] - 1, 0, i, :]) for i in range(self.num_emv_agents)])

        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                        np.concatenate(self.buffer.obs[-1]),
                                                        np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                        np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.graph_obs[-1]))
        else:
            # next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
            #                                                    np.concatenate(self.buffer.obs[-1]),
            #                                                    np.concatenate(self.buffer.rnn_states[-1]),
            #                                                    np.concatenate(self.buffer.rnn_states_critic[-1]),
            #                                                    np.concatenate(self.buffer.masks[-1]),
            #                                                    np.concatenate(self.buffer.available_actions[-1])
            #                                                    )
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[self.buffer.step]),
                                                        np.concatenate(self.buffer.rnn_states_critic[self.buffer.step]),
                                                        np.concatenate(self.buffer.masks[self.buffer.step]),
                                                         np.concatenate(_t2n(self.buffer.graph_obs[self.buffer.step]))
                                                         )
        ve_next_values = self.trainer_ve.policy.get_values(ve_share_obs, ve_rnn_states_critic, ve_masks, ve_graph)
        # emv_next_values = self.trainer_emv.policy.get_values(emv_share_obs, emv_rnn_states_critic, emv_masks, emv_graph)
        ve_next_values = np.array(np.split(_t2n(ve_next_values), self.n_rollout_threads))
        # emv_next_values = np.array(np.split(_t2n(emv_next_values), self.n_rollout_threads))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))

        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)
        self.ve_buffer.compute_returns_ve(ve_next_values, self.trainer_ve.value_normalizer)
        # self.emv_buffer.compute_returns_ve(emv_next_values, self.trainer_emv.value_normalizer)


    def train(self):
        """Train policies with data in buffer. """
        self.trainer.prep_training()
        self.trainer_ve.prep_training()
        # self.trainer_emv.prep_training()

        train_infos = self.trainer.train(self.buffer)
        train_infos_ve = self.trainer_ve.train_ve(self.ve_buffer, self.num_ve_agents)
        # train_infos_emv = self.trainer_emv.train_ve(self.emv_buffer, self.num_emv_agents)

        self.buffer.after_update()
        # self.ve_buffer.after_update_ve()
        # self.emv_buffer.after_update_ve()
        return train_infos, train_infos_ve

    def save(self, episode):
        """Save policy's actor and critic networks."""
        policy_actor = self.trainer.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_tl_{}.pt".format(episode))
        policy_critic = self.trainer.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_tl_{}.pt".format(episode))
        if self.trainer._use_valuenorm:
            policy_vnorm = self.trainer.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_tl_{}.pt".format(episode))

        policy_actor = self.trainer_ve.policy.actor
        torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_ve_{}.pt".format(episode))
        policy_critic = self.trainer_ve.policy.critic
        torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_ve_{}.pt".format(episode))
        if self.trainer_ve._use_valuenorm:
            policy_vnorm = self.trainer_ve.value_normalizer
            torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_ve_{}.pt".format(episode))

        # policy_actor = self.trainer_emv.policy.actor
        # torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_emv_{}.pt".format(episode))
        # policy_critic = self.trainer_emv.policy.critic
        # torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_emv_{}.pt".format(episode))
        # if self.trainer_emv._use_valuenorm:
        #     policy_vnorm = self.trainer_emv.value_normalizer
        #     torch.save(policy_vnorm.state_dict(), str(self.save_dir) + "/vnorm_emv_{}.pt".format(episode))

    def restore(self, model_dir):
        """Restore policy's networks from a saved model."""
        if self.algorithm_name == "mat" or self.algorithm_name == "mat_dec":
            self.policy.restore(model_dir)
        else:
            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_tl_100.pt')
            self.policy.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_tl_100.pt')
                self.policy.critic.load_state_dict(policy_critic_state_dict)
                # if self.all_args.use_valuenorm:
                #     policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm_tl_5.pt')
                #     self.trainer.value_normalizer.load_state_dict(policy_vnorm_state_dict)

            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_ve_100.pt')
            self.policy_ve.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_ve_100.pt')
                self.policy_ve.critic.load_state_dict(policy_critic_state_dict)
                # if self.all_args.use_valuenorm:
                #     policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm_ve_5.pt')
                #     self.trainer_ve.value_normalizer.load_state_dict(policy_vnorm_state_dict)

            policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_emv_100.pt')
            self.policy_emv.actor.load_state_dict(policy_actor_state_dict)
            if not self.all_args.use_render:
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_emv_100.pt')
                self.policy_emv.critic.load_state_dict(policy_critic_state_dict)
                # if self.all_args.use_valuenorm:
                #     policy_vnorm_state_dict = torch.load(str(self.model_dir) + '/vnorm_tl_5.pt')
                #     self.trainer_emv.value_normalizer.load_state_dict(policy_vnorm_state_dict)

    def compute_reward(self, graph_ve, graph_emv, reward):
        reward = torch.nan_to_num(reward, nan=0.0)
        cos_sim = F.cosine_similarity(graph_ve.unsqueeze(1), graph_emv.unsqueeze(0), dim=2)
        max_indices = torch.argmax(cos_sim, dim=1, keepdim=True)
        # 乘以 reward 并累加 (600, 1)
        max_values = cos_sim.gather(1, max_indices)
        selected_weights = reward[max_indices.squeeze()]
        result = max_values * selected_weights
        # weighted_reward = cos_sim * reward.T  # reward 进行转置 (5,1) -> (1,5)

        # 取最大值 (600, 1)
        # reward_ve = weighted_reward.max(dim=1, keepdim=True)[0]

        return _t2n(result.unsqueeze(0))


    def log_train_tl(self, train_infos, total_num_steps):
        """
        Log training info.
        :param train_infos: (dict) information about training update.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
                self.logger_tl.info(f'epoch {total_num_steps / 3600}, key {k}, value {v}')

    def log_train_vehicle(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if self.use_wandb:
                wandb.log({k: np.mean(v)}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
                self.logger_ve.info(f'epoch {total_num_steps / 3600}, key {k}, value {v}')



    def log_train_emv(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
           if self.use_wandb:
                wandb.log({k: np.mean(v)}, step=total_num_steps)
           else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)
                self.logger_emv.info(f'epoch {total_num_steps / 3600}, key {k}, value {v}')

    def log_env_tl(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_tl.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_env_vehicle(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_ve.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_env_emv(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v)>0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_emv.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_eval_tl(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_tl_eval.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_eval_ve(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_ve_eval.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_eval_emv(self, env_infos, total_num_steps):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.logger_emv_eval.info(f'epoch {total_num_steps / 3600}, key {k}, value {np.mean(v)}')

    def log_eval_info_tl(self,eval_infos):
        env_info_tl = {}
        for k, v in eval_infos.items():
            for s, a in v.items():
                if s not in env_info_tl:
                    env_info_tl[s] = []
                env_info_tl[s].append(a)
        for k, v in env_info_tl.items():
            if len(v) > 0:
                self.logger_tl_eval.info(f'key {k}, value {np.mean(v)}')

    def log_eval_info_ve(self, eval_infos):
        env_infos_ve = {}
        env_infos_emv = {}
        for agent_id in range(self.num_vehicle_agents + self.num_emv_agents):
            if agent_id < self.num_vehicle_agents:
                for k, v in eval_infos[str(agent_id)].items():
                    if k not in env_infos_ve:
                        env_infos_ve[k] = []
                    env_infos_ve[k].append(v)
            else:
                for k, v in eval_infos["emergency_" + str(agent_id - self.num_vehicle_agents)].items():
                    if k not in env_infos_emv:
                        env_infos_emv[k] = []
                    env_infos_emv[k].append(v)

        for k, v in env_infos_ve.items():
            if len(v) > 0:
                self.logger_emv_eval.info(f'key {k}, value {np.mean(v)}')
        for k, v in env_infos_emv.items():
            if len(v) > 0:
                self.logger_emv_eval.info(f'key {k}, value {np.mean(v)}')