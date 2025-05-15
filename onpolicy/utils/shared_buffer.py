import torch
import numpy as np
import torch.nn.functional as F
from onpolicy.utils.util import get_shape_from_obs_space, get_shape_from_act_space
from torch_geometric.data import HeteroData
import copy


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])


def _cast(x):
    return x.transpose(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _cast_t(x):
    return x.permute(1, 2, 0, 3).reshape(-1, *x.shape[3:])

def _cast_n(x):
    num_agents = x.shape[2]
    return x.transpose(2, 1, 0, 3).reshape(num_agents, -1, *x.shape[3:]).transpose(1,0,2)
def _shuffle_agent_grid(x, y):
    rows = np.indices((x, y))[0]
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])
    cols = np.stack([np.arange(y) for _ in range(x)])
    return rows, cols

class SharedReplayBuffer(object):
    """
    Buffer to store training data.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param num_agents: (int) number of agents in the env.
    :param obs_space: (gym.Space) observation space of agents.
    :param cent_obs_space: (gym.Space) centralized observation space of agents.
    :param act_space: (gym.Space) action space for agents.
    """

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, share_num_agents):
        self.args = args
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits
        self.algo = args.algorithm_name
        self.num_agents = num_agents
        self.share_num_agents = share_num_agents

        self.obs_shape = get_shape_from_obs_space(obs_space)
        self.share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(self.obs_shape[-1]) == list:
            self.obs_shape = self.obs_shape[:1]

        if type(self.share_obs_shape[-1]) == list:
            self.share_obs_shape = self.share_obs_shape[:1]

        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.share_num_agents, *self.share_obs_shape),
                                  dtype=np.float32)

        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *self.obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *self.obs_shape),
                            dtype=np.float32)

        self.rnn_states = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, self.recurrent_N, self.hidden_size),
            dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)
        # self.rnn_states_gnn = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, self.hidden_size), dtype=torch.float32)

        self.value_preds = np.zeros(
            (self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.act_space = act_space
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, act_space.n),
                                             dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, act_shape), dtype=np.float32)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.zeros_like(self.masks)

        self.step = 0
        self.step_ve = np.zeros(self.num_agents, dtype=int)
        # self.graph = np.empty((self.episode_length + 1, self.n_rollout_threads, num_agents), dtype=HeteroData)
        self.graph_obs = torch.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 64),
                            dtype=torch.float32)
        self.trans_masks = np.ones((self.episode_length, self.n_rollout_threads, num_agents, num_agents),
                                   dtype=np.float32)

    def reset_tl(self):
        self.step = 0
        self.step_ve = np.zeros(self.num_agents, dtype=int)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)


    def reset(self):
        self.step = 0
        self.step_ve = np.zeros(self.num_agents, dtype=int)
        self.active_masks = np.zeros_like(self.masks)
        self.rewards = np.zeros(
            (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *self.share_obs_shape),
        #                           dtype=np.float32)
        #
        # self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, self.num_agents, *self.obs_shape), dtype=np.float32)
        #
        # self.rnn_states = np.zeros(
        #     (self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
        #     dtype=np.float32)
        # self.rnn_states_critic = np.zeros_like(self.rnn_states)
        #
        # self.value_preds = np.zeros(
        #     (self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # self.returns = np.zeros_like(self.value_preds)
        # self.advantages = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #
        # if self.act_space.__class__.__name__ == 'Discrete':
        #     self.available_actions = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, self.act_space.n),
        #                                      dtype=np.float32)
        # else:
        #     self.available_actions = None
        #
        # act_shape = get_shape_from_act_space(self.act_space)
        #
        # self.actions = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        # self.action_log_probs = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, self.num_agents, act_shape), dtype=np.float32)
        # self.rewards = np.zeros(
        #     (self.episode_length, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        #
        # self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # self.bad_masks = np.ones_like(self.masks)
        # self.active_masks = np.zeros_like(self.masks)

    def insert_0(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
                 value_preds, rewards, masks, graph_obs,bad_masks=None, active_masks=None, available_actions=None):
        self.step = (self.step + 1) % self.episode_length
        self.rewards[self.step-1] = rewards.copy()
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.next_obs[self.step-1] = obs.copy()

        self.masks[self.step] = masks.copy()
        # for i in range(self.num_agents):
        #     self.graph[self.step][0][i] = copy.deepcopy(graph)
        self.graph_obs[self.step] = graph_obs
        # self.rnn_states_gnn[self.step] = rnn_state_gnn
        if bad_masks is not None:
                self.bad_masks[self.step] = bad_masks.copy()
        if active_masks is not None:
                self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
                self.available_actions[self.step] = available_actions.copy()



    def insert_1(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) action space for agents.
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rnn_states[self.step+1] = rnn_states_actor.copy()
        self.rnn_states_critic[self.step+1] = rnn_states_critic.copy()



    def insert_ve_0(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, return_ve, graph_obs,bad_masks=None, active_masks=None, available_actions=None):
        # print(obs[:,0], return_ve[0][0], "insert ve 0-1")
        for i, r in enumerate(return_ve[0]):
            if r[0] == True:
                self.step_ve[i] = (self.step_ve[i] + 1) % self.episode_length
                self.share_obs[self.step_ve[i]-1][0][i] = share_obs[:, i].copy()
                self.obs[self.step_ve[i]-1][0][i] = obs[:, i].copy()
                self.graph_obs[self.step_ve[i] - 1][0][i] = graph_obs[i]
                # self.rnn_states_gnn[self.step_ve[i] - 1][0][i]=rnn_state_gnn[:,i]

                # self.graph[self.step_ve[i]-1][0][i] = copy.deepcopy(graph)
                if bad_masks is not None:
                    self.bad_masks[self.step_ve[i]-1][0][i] = bad_masks[:, i].copy()
                if active_masks is not None:
                    self.active_masks[self.step_ve[i]-1][0][i] = active_masks[:, i].copy()
                if available_actions is not None:
                    self.available_actions[self.step_ve[i]-1][0][i] = available_actions[:, i].copy()
                   #return change road
                if self.step_ve[i] > 1:
                    self.next_obs[self.step_ve[i] - 2][0][i] = obs[:, i].copy()
                    self.rewards[self.step_ve[i] - 2][0][i] = rewards[:,i].copy()
                    self.masks[self.step_ve[i]-1][0][i] = masks[:,i].copy()

        # print(self.obs[self.step_ve[0],:,0,:])
        # print(self.step_ve, "insert_ve_0, step")

    def insert_ve_1(self, share_obs, obs, rnn_states_actor, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, do_action, bad_masks=None, active_masks=None, available_actions=None):

        for i, r in enumerate(do_action[0]):
            if r[0] == True:
                self.actions[self.step_ve[i]-1,:,i,:] = actions[:,i].copy()
                self.action_log_probs[self.step_ve[i]-1,:,i,:] = action_log_probs[:,i].copy()
                self.value_preds[self.step_ve[i]-1,:,i,:] = value_preds[:,i].copy()
                self.rnn_states[self.step_ve[i]][0][i] = rnn_states_actor[:, i].copy()
                self.rnn_states_critic[self.step_ve[i]][0][i] = rnn_states_critic[:, i].copy()




    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        """
        Insert data into the buffer. This insert function is used specifically for Hanabi, which is turn based.
        :param share_obs: (argparse.Namespace) arguments containing relevant model, policy, and env information.
        :param obs: (np.ndarray) local agent observations.
        :param rnn_states_actor: (np.ndarray) RNN states for actor network.
        :param rnn_states_critic: (np.ndarray) RNN states for critic network.
        :param actions:(np.ndarray) actions taken by agents.
        :param action_log_probs:(np.ndarray) log probs of actions taken by agents
        :param value_preds: (np.ndarray) value function prediction at each step.
        :param rewards: (np.ndarray) reward collected at each step.
        :param masks: (np.ndarray) denotes whether the environment has terminated or not.
        :param bad_masks: (np.ndarray) denotes indicate whether whether true terminal state or due to episode limit
        :param active_masks: (np.ndarray) denotes whether an agent is active or dead in the env.
        :param available_actions: (np.ndarray) actions available to each agent. If None, all actions are available.
        """
        self.share_obs[self.step] = share_obs.copy()
        self.obs[self.step] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        """Copy last timestep data to first index. Called after update to model."""
        self.share_obs[0] = self.share_obs[self.step].copy()
        self.obs[0] = self.obs[self.step].copy()
        self.rnn_states[0] = self.rnn_states[self.step].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[self.step].copy()
        self.masks[0] = self.masks[self.step].copy()
        self.bad_masks[0] = self.bad_masks[self.step].copy()
        self.active_masks[0] = self.active_masks[self.step].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[self.step].copy()

    def after_update_ve(self):
        """Copy last timestep data to first index. Called after update to model."""
        for i in range(self.num_agents):
            self.share_obs[0,:,i,:] = self.share_obs[self.step_ve[i],:,i,:].copy()
            self.obs[0,:,i,:] = self.obs[self.step_ve[i],:,i,:].copy()
            self.rnn_states[0,:,i,:] = self.rnn_states[self.step_ve[i],:,i,:].copy()
            self.rnn_states_critic[0,:,i,:] = self.rnn_states_critic[self.step_ve[i],:,i,:].copy()
            self.masks[0,:,i,:] = self.masks[self.step_ve[i],:,i,:].copy()
            self.bad_masks[0,:,i,:] = self.bad_masks[self.step_ve[i],:,i,:].copy()
            self.active_masks[0,:,i,:] = self.active_masks[self.step_ve[i],:,i,:].copy()
            if self.available_actions is not None:
                self.available_actions[0,:,i,:] = self.available_actions[self.step_ve[i],:,i,:].copy()


    def chooseafter_update(self):
        """Copy last timestep data to first index. This method is used for Hanabi."""
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:   #1
                self.value_preds[self.step] = next_value #1
                gae = 0 #1
                for step in reversed(range(self.step)): #1
                    if self._use_popart or self._use_valuenorm: #1
                        if self.algo == "mat" or self.algo == "mat_dec":
                            value_t = value_normalizer.denormalize(self.value_preds[step])
                            value_t_next = value_normalizer.denormalize(self.value_preds[step + 1])
                            rewards_t = self.rewards[step]

                            # mean_v_t = np.mean(value_t, axis=-2, keepdims=True)
                            # mean_v_t_next = np.mean(value_t_next, axis=-2, keepdims=True)
                            # delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            delta = rewards_t + self.gamma * self.masks[step + 1] * value_t_next - value_t
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self.returns[step] = gae + value_t
                        else:   #1
                            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                                self.value_preds[step + 1]) * self.masks[step + 1] \
                                    - value_normalizer.denormalize(self.value_preds[step])
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        if self.algo == "mat" or self.algo == "mat_dec":
                            rewards_t = self.rewards[step]
                            mean_v_t = np.mean(self.value_preds[step], axis=-2, keepdims=True)
                            mean_v_t_next = np.mean(self.value_preds[step + 1], axis=-2, keepdims=True)
                            delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                            # delta = rewards_t + self.gamma * self.value_preds[step + 1] * \
                            #         self.masks[step + 1] - self.value_preds[step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.advantages[step] = gae
                            self.returns[step] = gae + self.value_preds[step]

                        else:
                            delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                    self.masks[step + 1] - self.value_preds[step]
                            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                            self.returns[step] = gae + self.value_preds[step]

            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def compute_returns_ve(self, next_value, value_normalizer=None):
        """
        Compute returns either as discounted sum of rewards, or using GAE.
        :param next_value: (np.ndarray) value predictions for the step after the last episode step.
        :param value_normalizer: (PopArt) If not None, PopArt value normalizer instance.
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        # step + 1
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                            self.value_preds[step + 1]) * self.masks[step + 1] \
                                - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * gae * self.masks[step + 1]
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - \
                                self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(
                            self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[
                            step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:   #1
                for i in range(self.num_agents):
                    if self.step_ve[i] > 0:
                        self.value_preds[self.step_ve[i]-1,:,i,:] = next_value[:,i,:] #1
                        gae = 0 #1
                        for step in reversed(range(self.step_ve[i]-1)): #1
                            if self._use_popart or self._use_valuenorm: #1
                                if self.algo == "mat" or self.algo == "mat_dec":
                                    value_t = value_normalizer.denormalize(self.value_preds[step])
                                    value_t_next = value_normalizer.denormalize(self.value_preds[step + 1])
                                    rewards_t = self.rewards[step]

                                    # mean_v_t = np.mean(value_t, axis=-2, keepdims=True)
                                    # mean_v_t_next = np.mean(value_t_next, axis=-2, keepdims=True)
                                    # delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                                    delta = rewards_t + self.gamma * self.masks[step + 1] * value_t_next - value_t
                                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                                    self.advantages[step] = gae
                                    self.returns[step] = gae + value_t
                                else:   #1
                                    delta = self.rewards[step,:,i,:] + self.gamma * value_normalizer.denormalize(
                                        self.value_preds[step + 1,:,i,:]) * self.masks[step + 1,:,i,:] - value_normalizer.denormalize(self.value_preds[step,:,i,:])
                                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1,:,i,:] * gae
                                    self.returns[step,:,i,:] = gae + value_normalizer.denormalize(self.value_preds[step,:,i,:])
                            else:
                                if self.algo == "mat" or self.algo == "mat_dec":
                                    rewards_t = self.rewards[step]
                                    mean_v_t = np.mean(self.value_preds[step], axis=-2, keepdims=True)
                                    mean_v_t_next = np.mean(self.value_preds[step + 1], axis=-2, keepdims=True)
                                    delta = rewards_t + self.gamma * self.masks[step + 1] * mean_v_t_next - mean_v_t

                                    # delta = rewards_t + self.gamma * self.value_preds[step + 1] * \
                                    #         self.masks[step + 1] - self.value_preds[step]
                                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                                    self.advantages[step] = gae
                                    self.returns[step] = gae + self.value_preds[step]

                                else:
                                    delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                                            self.masks[step + 1] - self.value_preds[step]
                                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                                    self.returns[step] = gae + self.value_preds[step]

            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]



    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grid(batch_size, num_agents)

        # keep (num_agent, dim)
        # share_obs = self.share_obs[:-1].reshape(-1, num_agents, *self.share_obs.shape[3:])
        # obs = self.obs[:-1].reshape(-1, num_agents, *self.obs.shape[3:])
        # rnn_states = self.rnn_states[:-1].reshape(-1, num_agents, *self.rnn_states.shape[3:])
        # rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, num_agents, *self.rnn_states_critic.shape[3:])
        # actions = self.actions.reshape(-1, num_agents, self.actions.shape[-1])
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]

        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states = rnn_states[rows, cols]
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_critic = rnn_states_critic[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # [L,T,N,Dim]-->[L*T,N,Dim]-->[index,N,Dim]-->[index*N, Dim]
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            rnn_states_batch = rnn_states[indices].reshape(-1, *rnn_states.shape[2:])
            rnn_states_critic_batch = rnn_states_critic[indices].reshape(-1, *rnn_states_critic.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        """
        Yield training data for MLP policies.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param mini_batch_size: (int) number of samples in each minibatch.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * episode_length * num_agents

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) * number of agents ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, num_agents,
                          n_rollout_threads * episode_length * num_agents,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[3:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[3:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N M Dim]-->[T N M Dim]-->[T*N*M,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        """
        Yield training data for non-chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        """
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        batch_size = n_rollout_threads * num_agents
        assert n_rollout_threads * num_agents >= num_mini_batch, (
            "PPO requires the number of processes ({})* number of agents ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_agents, num_mini_batch))
        num_envs_per_batch = batch_size // num_mini_batch
        perm = torch.randperm(batch_size).numpy()

        share_obs = self.share_obs.reshape(-1, batch_size, *self.share_obs.shape[3:])
        obs = self.obs.reshape(-1, batch_size, *self.obs.shape[3:])
        rnn_states = self.rnn_states.reshape(-1, batch_size, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic.reshape(-1, batch_size, *self.rnn_states_critic.shape[3:])
        actions = self.actions.reshape(-1, batch_size, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions.reshape(-1, batch_size, self.available_actions.shape[-1])
        value_preds = self.value_preds.reshape(-1, batch_size, 1)
        returns = self.returns.reshape(-1, batch_size, 1)
        masks = self.masks.reshape(-1, batch_size, 1)
        active_masks = self.active_masks.reshape(-1, batch_size, 1)
        action_log_probs = self.action_log_probs.reshape(-1, batch_size, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, batch_size, 1)

        for start_ind in range(0, batch_size, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(share_obs[:-1, ind])
                obs_batch.append(obs[:-1, ind])
                rnn_states_batch.append(rnn_states[0:1, ind])
                rnn_states_critic_batch.append(rnn_states_critic[0:1, ind])
                actions_batch.append(actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[:-1, ind])
                value_preds_batch.append(value_preds[:-1, ind])
                return_batch.append(returns[:-1, ind])
                masks_batch.append(masks[:-1, ind])
                active_masks_batch.append(active_masks[:-1, ind])
                old_action_log_probs_batch.append(action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, dim) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into. 2/8
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN. 10
        """
        _, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        episode_length = self.step
        batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            next_obs = self.next_obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.next_obs.shape[3:])
        else:
            share_obs = _cast(self.share_obs[:self.step])
            obs = _cast(self.obs[:self.step])
            next_obs = _cast(self.next_obs[:self.step])
        graph = _cast_t(self.graph_obs[:self.step])
        actions = _cast(self.actions[:self.step])
        action_log_probs = _cast(self.action_log_probs[:self.step])
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:self.step])
        returns = _cast(self.returns[:self.step])
        masks = _cast(self.masks[:self.step])
        active_masks = _cast(self.active_masks[:self.step])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:self.step].transpose(1, 2, 0, 3, 4).reshape(-1, *self.rnn_states.shape[3:])
        rnn_states_critic = self.rnn_states_critic[:self.step].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                         *self.rnn_states_critic.shape[
                                                                                          3:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:self.step])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            next_obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            # graph_batch = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                next_obs_batch.append(next_obs[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                # graph_batch.append(graph[ind:ind + data_chunk_length])
            graph_batch = torch.stack([graph[index * data_chunk_length:index * data_chunk_length + data_chunk_length] for index in indices], dim=0)
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)           
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            graph_batch = graph_batch.permute(1, 0, 2)
            next_obs_batch = np.stack(next_obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            graph_batch = _flatten(L, N, graph_batch)
            actions_batch = _flatten(L, N, actions_batch)
            next_obs_batch = _flatten(L, N,next_obs_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, next_obs_batch, graph_batch, available_actions_batch

    def recurrent_generator_ve(self, advantages, num_mini_batch, data_chunk_length):
        """
        Yield training data for chunked RNN training.
        :param advantages: (np.ndarray) advantage estimates.
        :param num_mini_batch: (int) number of minibatches to split the batch into.
        :param data_chunk_length: (int) length of sequence chunks with which to train RNN.
        """
        _, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # episode_length = self.step-2
        batch_size = n_rollout_threads * (np.sum(self.step_ve[(self.step_ve != 1) & (self.step_ve != 0)]-1))
        # batch_size = n_rollout_threads * episode_length * num_agents
        data_chunks = batch_size // data_chunk_length  # [C=r*T*M/L]
        mini_batch_size = data_chunks // num_mini_batch

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 4:
            share_obs = self.share_obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.share_obs.shape[3:])
            obs = self.obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.obs.shape[3:])
            next_obs = self.next_obs[:self.step].transpose(1, 2, 0, 3, 4, 5).reshape(-1, *self.next_obs.shape[3:])
        else:
            # share_obs = _cast(self.share_obs[:self.step-2])
            # print(self.share_obs[:self.step_ve[1],:,0,:].shape)
            share_obs_list, obs_list, next_obs_list = [], [], []
            actions_list, action_log_probs_list, advantages_list = [], [], []
            value_preds_list, returns_list, masks_list = [], [], []
            active_masks_list, rnn_states_list, rnn_states_critic_list = [], [], []
            available_actions_list = []
            graph_list = None

            for i in range(num_agents):
                step = self.step_ve[i] - 1  # 当前代理的有效步数
                if step > 0:  # 只有当步数大于0时才处理数据
                    # 对每个变量进行切片、重塑并拼接
                    share_obs_list.append(_cast(self.share_obs[:step, :, i, :][:, :, np.newaxis, :]))
                    obs_list.append(_cast(self.obs[:step, :, i, :][:, :, np.newaxis, :]))
                    graph_obs = _cast_t(self.graph_obs[:step, :, i, :].unsqueeze(2))  # 确保维度匹配

                    if graph_list is None:
                        graph_list = graph_obs  # **首次赋值**
                    else:
                        graph_list = torch.cat([graph_list, graph_obs], dim=0)
                    next_obs_list.append(_cast(self.next_obs[:step, :, i, :][:, :, np.newaxis, :]))

                    actions_list.append(_cast(self.actions[:step, :, i, :][:, :, np.newaxis, :]))
                    action_log_probs_list.append(_cast(self.action_log_probs[:step, :, i, :][:, :, np.newaxis, :]))
                    advantages_list.append(_cast(advantages[:step, :, i, :][:, :, np.newaxis, :]))
                    value_preds_list.append(_cast(self.value_preds[:step, :, i, :][:, :, np.newaxis, :]))
                    returns_list.append(_cast(self.returns[:step, :, i, :][:, :, np.newaxis, :]))
                    masks_list.append(_cast(self.masks[:step, :, i, :][:, :, np.newaxis, :]))
                    active_masks_list.append(_cast(self.active_masks[:step, :, i, :][:, :, np.newaxis, :]))

                    # 对 RNN 状态做特殊处理，进行转置和重塑
                    rnn_states_list.append(
                        self.rnn_states[:step, :, i, :][:, :, np.newaxis, :].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                                              *self.rnn_states.shape[
                                                                                                               3:]))
                    rnn_states_critic_list.append(
                        self.rnn_states_critic[:step, :, i, :][:, :, np.newaxis, :].transpose(1, 2, 0, 3, 4).reshape(-1,
                                                                                                                     *self.rnn_states_critic.shape[
                                                                                                                      3:]))

                    # 如果 available_actions 不为 None，则进行拼接
                    if self.available_actions is not None:
                        available_actions_list.append(
                            _cast(self.available_actions[:step, :, i, :][:, :, np.newaxis, :]))

            # 将所有列表按轴连接起来
            share_obs = np.concatenate(share_obs_list, axis=0)
            # graph = torch.cat(graph_list, dim=0)
            obs = np.concatenate(obs_list, axis=0)
            next_obs = np.concatenate(next_obs_list, axis=0)

            actions = np.concatenate(actions_list, axis=0)
            action_log_probs = np.concatenate(action_log_probs_list, axis=0)
            advantages = np.concatenate(advantages_list, axis=0)
            value_preds = np.concatenate(value_preds_list, axis=0)
            returns = np.concatenate(returns_list, axis=0)
            masks = np.concatenate(masks_list, axis=0)
            active_masks = np.concatenate(active_masks_list, axis=0)
            rnn_states = np.concatenate(rnn_states_list, axis=0)
            rnn_states_critic = np.concatenate(rnn_states_critic_list, axis=0)

            if self.available_actions is not None:
                available_actions = np.concatenate(available_actions_list, axis=0)

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            # graph_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            next_obs_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:

                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N M Dim]-->[N,M,T,Dim]-->[N*M*T,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind + data_chunk_length])
                obs_batch.append(obs[ind:ind + data_chunk_length])
                # graph_batch.append(graph[ind:ind + data_chunk_length])
                actions_batch.append(actions[ind:ind + data_chunk_length])
                next_obs_batch.append(next_obs[ind:ind + data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind + data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind + data_chunk_length])
                return_batch.append(returns[ind:ind + data_chunk_length])
                masks_batch.append(masks[ind:ind + data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind + data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind + data_chunk_length])
                adv_targ.append(advantages[ind:ind + data_chunk_length])
                # size [T+1 N M Dim]-->[T N M Dim]-->[N M T Dim]-->[N*M*T,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
            graph_batch = torch.stack([graph_list[index * data_chunk_length:index * data_chunk_length+data_chunk_length] for index in indices], dim=0)
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)
            # graph_batch = np.stack(graph_batch, axis=1)
            graph_batch = graph_batch.permute(1, 0, 2)
            actions_batch = np.stack(actions_batch, axis=1)
            next_obs_batch = np.stack(next_obs_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[3:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[3:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            graph_batch = _flatten(L, N, graph_batch)
            actions_batch = _flatten(L, N, actions_batch)
            next_obs_batch = _flatten(L, N, next_obs_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch,\
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                  adv_targ, next_obs_batch, graph_batch, available_actions_batch

