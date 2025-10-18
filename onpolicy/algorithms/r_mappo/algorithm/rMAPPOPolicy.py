import torch
from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic, MINE, RNN, R_Actor_Trans, R_Critic_Trans, \
    R_Critic_Trans_all, R_Critic_all,ICMModel, HeteroGCN
from onpolicy.utils.util import update_linear_schedule
from onpolicy.algorithms.utils.rnn import RNNLayer
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from onpolicy.algorithms.utils.util import init, check

class R_MAPPOPolicy:
    """
    MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (gym.Space) observation space.
    :param cent_obs_space: (gym.Space) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (gym.Space) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu"), type = 'vehicle'):
        self.args = args
        self.device = device
        self.lr = args.lr
        self.lr_gnn = args.lr_gnn
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.eta = 0.01

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        # if type == 'vehicle':
        self.mine_emv = MINE(args.hidden_size, args.hidden_size, device=self.device)
        # self.mine = MINE(args.hidden_size, args.hidden_size,device = self.device)
        self.mine = MINE(args.hidden_size, (args.num_emv_agents + args.num_emv_agents) * (args.hidden_size), device=self.device)
        self.rnn = RNN(self.obs_space, args.hidden_size, args.recurrent_N, args.use_orthogonal,device = self.device)
        # self.rnn = RNNLayer(args.hidden_size, args.hidden_size, args.recurrent_N, args.use_orthogonal)
        self.gnn = HeteroGCN(self.obs_space,args.hidden_size, device = self.device)
        # self.icm = ICMModel(args, self.obs_space, 64, self.act_space, self.device)
        self.actor_para = list(self.actor.base.parameters())+list(self.actor.act.parameters())


        self.actor_optimizer = torch.optim.Adam(self.actor_para,
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        # if type == 'vehicle':
        #     self.gnn_optimizer = torch.optim.Adam(
        #         list(self.gnn.parameters()) + list(self.mine.parameters()) + list(self.mine_emv.parameters()) + list(self.rnn.parameters()),
        #         lr=self.lr_gnn, eps=args.opti_eps_gnn,
        #         weight_decay=args.weight_decay_gnn)
        # else:
        self.gnn_optimizer = torch.optim.Adam(list(self.gnn.parameters())+list(self.mine.parameters()) + list(self.mine_emv.parameters()) +list(self.rnn.parameters()),
                                                lr=self.lr_gnn, eps=args.opti_eps_gnn,
                                                weight_decay=args.weight_decay_gnn)
        # self.rnn_optimizer = torch.optim.Adam(self.rnn.parameters(), lr=self.lr, eps=self.opti_eps, weight_decay=self.weight_decay)
        # self.icm_optimizer = torch.optim.Adam(self.icm.parameters(),
        #                                       lr=self.lr,
        #                                       eps=self.opti_eps,
        #                                       weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        # update_linear_schedule(self.icm_optimizer, episode, episodes, self.lr)

    def lr_decay_gnn(self, episode, episodes):
        update_linear_schedule(self.gnn_optimizer, episode, episodes, self.lr_gnn)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, graph, available_actions=None,
                    deterministic=False, trans_masks=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        if np.random.uniform() > self.args.epsilon:
            deterministic = True
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 graph,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks, graph)
        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic

    def get_values(self, cent_obs, rnn_states_critic, masks, graph):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks, graph)
        return values

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, graph,
                         available_actions=None, active_masks=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     graph,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks, graph)
        return values, action_log_probs, dist_entropy

    def act(self, obs, rnn_states_actor, masks, graph, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, graph, available_actions, deterministic)
        return actions, rnn_states_actor

    # def compute_intrinsic_reward(self, state, next_state, action):
    #     state = torch.FloatTensor(state).to(self.device)
    #     next_state = torch.FloatTensor(next_state).to(self.device)
    #     action = torch.LongTensor(action).to(self.device)
    #
    #     action_onehot = torch.FloatTensor(
    #         len(action), self.act_space.n).to(
    #         self.device)
    #     action_onehot.zero_()
    #     action_onehot.scatter_(1, action.view(len(action), -1), 1)
    #
    #     real_next_state_feature, pred_next_state_feature, pred_action = self.icm(
    #         [state, next_state, action_onehot])
    #     intrinsic_reward = self.eta * F.mse_loss(real_next_state_feature, pred_next_state_feature,
    #                                              reduction='none').mean(-1)
    #     return intrinsic_reward.data.cpu().numpy()

    def get_state(self, graph):
        return self.gnn(graph)

    def get_dict(self, dict):
        return self.gnn.get_dict(dict)

    def get_hidden_states(self, x_dict, h=None):
        return self.rnn(x_dict, h)

