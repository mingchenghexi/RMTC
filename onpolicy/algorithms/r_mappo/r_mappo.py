import sys

import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_gard_norm, huber_loss, mse_loss
from onpolicy.utils.valuenorm import ValueNorm
from onpolicy.algorithms.utils.util import check
import torch.nn.functional as F
import torch.optim as optim

class R_MAPPO():
    """
    Trainer class for MAPPO to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.reverse_scale = 1

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.ce = nn.CrossEntropyLoss()
        self.forward_mse = nn.MSELoss()

        assert (self._use_popart and self._use_valuenorm) == False, ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
        if self._use_popart or self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update(self, sample, update_actor=True):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic update.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        if len(sample) == 14:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, next_obs_batch,graph, available_actions_batch = sample

        else:
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
            value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
            adv_targ, next_obs_batch, graph,available_actions_batch, _ = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        obs_batch = check(obs_batch).to(**self.tpdv)
        next_obs_batch = check(next_obs_batch).to(**self.tpdv)
        graph = check(graph).to(**self.tpdv)
        # Reshape to do in a single forward pass for all steps

        # action = torch.LongTensor(actions_batch).to(self.device)
        #
        # action_onehot = torch.FloatTensor(
        #     len(action), self.policy.act_space.n).to(
        #     self.device)
        # action_onehot.zero_()
        # action_onehot.scatter_(1, action.view(len(action), -1), 1)
        #
        # real_next_state_feature, pred_next_state_feature, pred_action = self.policy.icm(
        #     [obs_batch, next_obs_batch, action_onehot])
        # inverse_loss = self.ce(pred_action, action_onehot)
        #
        # forward_loss = self.forward_mse(pred_next_state_feature, real_next_state_feature.detach())
        # self.train_feature(graph, obs_batch)

        # self.policy.icm_optimizer.step()
        # graph = self.policy.get_dict(graph)
        # loss = nn.functional.mse_loss(obs_batch, graph)
        # self.policy.gnn_optimizer.zero_grad()
        # loss.requires_grad_(True)
        # loss.backward()
        # self.policy.gnn_optimizer.step()
        #
        # for name, param in self.policy.actor.gnn.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name} has gradient: {param.grad}")
        #     else:
        #         print(f"Parameter {name} does not have a gradient")

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch, 
                                                                              rnn_states_batch, 
                                                                              rnn_states_critic_batch, 
                                                                              actions_batch, 
                                                                              masks_batch,
                                                                              graph.data,
                                                                              available_actions_batch,
                                                                              active_masks_batch)

        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()
        # for name, param in self.policy.actor.gnn.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name} has gradient: {param.grad}")
        #     else:
        #         print(f"Parameter {name} does not have a gradient")
        # critic update
        if len(values.shape) == 3: # reshape
            T_, B_ = values.shape[0], values.shape[1]
            values = values.reshape(T_*B_, -1)
            value_preds_batch = value_preds_batch.reshape(T_*B_, -1)
            return_batch = return_batch.reshape(T_*B_, -1)
            active_masks_batch = active_masks_batch.reshape(T_*B_, -1)
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        self.policy.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())

        self.policy.critic_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights


    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:buffer.step] - self.value_normalizer.denormalize(buffer.value_preds[:buffer.step])
        else:
            advantages = buffer.returns[:buffer.step] - buffer.value_preds[:buffer.step]
        advantages_copy = advantages.copy()#12,1,16,1
        advantages_copy[buffer.active_masks[:buffer.step] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:  #1
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def train_gnn(self, graph, obs, obs_ve, obs_emv, graph_tl_last):
        self.prep_training_gnn()
        emv_state = graph['emergency']
        tl_state = graph['signal_light']
        ve_state = graph['vehicle']
        num = tl_state.shape[0]
        num_ve = ve_state.shape[0]
        loss_total = 0.0
        tau = 0.1
        for node_type in ['signal_light', 'vehicle','emergency']:  # 仅对 signal_light 和 vehicle 进行对比学习
            h1 = F.normalize(graph_tl_last[node_type], dim=-1)  # 归一化特征
            h2 = F.normalize(graph[node_type], dim=-1)

            # 计算正样本相似度 (对角线)
            sim_matrix = torch.matmul(h1, h2.T) / tau  # 计算相似度矩阵 (N, N)
            pos_sim = torch.diag(sim_matrix)  # 取对角线作为正样本

            # 计算 InfoNCE Loss
            loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
            loss_total += loss.mean()
        obs_emv_tl = obs_emv.squeeze(1).view(1,-1).expand(num, -1)
        emv_obs = emv_state.view(1,-1).expand(num, -1)
        relate = torch.cat([emv_obs, obs_emv_tl], dim=1)
        joint = self.policy.mine(tl_state, relate).mean()


        # 负样本 (A, B_shuffled)
        B_shuffled = relate[torch.randperm(relate.shape[0])]
        mine_output = self.policy.mine(tl_state, B_shuffled)
        mine_output = torch.clamp(mine_output, max=10.0)
        marginal = torch.exp(mine_output).mean()

        # 互信息下界估计
        mi_loss = -torch.log2(torch.exp(torch.tensor(1.0)))*(joint - torch.log(marginal + 1e-10))
        print("loss",mi_loss)

        obs_emv_ve = obs_emv.squeeze(1).view(1, -1).expand(num_ve, -1)
        emv_obs_ve = emv_state.view(1, -1).expand(num_ve, -1)
        relate_ve = torch.cat([emv_obs_ve, obs_emv_ve], dim=1)
        joint_ve = self.policy.mine(ve_state, relate_ve).mean()

        # 负样本 (A, B_shuffled)
        B_shuffled_ve = relate_ve[torch.randperm(relate_ve.shape[0])]
        mine_output_ve = self.policy.mine(ve_state, B_shuffled_ve)
        mine_output_ve = torch.clamp(mine_output_ve, max=10.0)
        marginal_ve = torch.exp(mine_output_ve).mean()

        # 互信息下界估计
        mi_loss_ve = -torch.log2(torch.exp(torch.tensor(1.0))) * (joint_ve - torch.log(marginal_ve + 1e-10))
        print("loss", mi_loss)

        relate_emv = obs_emv.squeeze(1)
        joint_emv = self.policy.mine_emv(emv_state, relate_emv).mean()

        # 负样本 (A, B_shuffled)
        B_shuffled_emv = relate_emv[torch.randperm(relate_emv.shape[0])]
        mine_output_emv = self.policy.mine_emv(emv_state, B_shuffled_emv)
        mine_output_emv = torch.clamp(mine_output_emv, max=10.0)
        marginal_emv = torch.exp(mine_output_emv).mean()

        # 互信息下界估计
        mi_loss_emv = -torch.log2(torch.exp(torch.tensor(1.0))) * (joint_emv - torch.log(marginal_emv + 1e-10))
        print("loss", mi_loss_emv)

        loss = mi_loss + mi_loss_ve + mi_loss_emv + loss_total
        self.policy.gnn_optimizer.zero_grad()
        loss.backward()
        self.policy.gnn_optimizer.step()
        #
        # for name, param in self.policy.gnn.named_parameters():
        #     if param.grad is not None:
        #         print(f"Parameter {name} has gradient: {param.grad}")
        #     else:
        #         print(f"Parameter {name} does not have a gradient")

    def train_gnn_emv(self,  graph, obs, obs_emv, graph_ve_last):
        self.prep_training_gnn()

        emv_state = graph['emergency']
        ve_state = graph['vehicle']
        num = ve_state.shape[0]
        loss_total = 0.0
        tau = 0.1
        for node_type in ['signal_light', 'vehicle', 'emergency']:  # 仅对 signal_light 和 vehicle 进行对比学习
            h1 = F.normalize(graph_ve_last[node_type], dim=-1)  # 归一化特征
            h2 = F.normalize(graph[node_type], dim=-1)

            # 计算正样本相似度 (对角线)
            sim_matrix = torch.matmul(h1, h2.T) / tau  # 计算相似度矩阵 (N, N)
            pos_sim = torch.diag(sim_matrix)  # 取对角线作为正样本

            # 计算 InfoNCE Loss
            loss = -torch.log(torch.exp(pos_sim) / torch.sum(torch.exp(sim_matrix), dim=1))
            loss_total += loss.mean()

        obs = obs.squeeze(1).view(1, -1).expand(num, -1)
        emv_obs = emv_state.view(1, -1).expand(num, -1)
        relate = torch.cat([emv_obs, obs.squeeze(1)], dim=1)
        joint = self.policy.mine(ve_state, relate).mean()

        # 负样本 (A, B_shuffled)
        B_shuffled = relate[torch.randperm(relate.shape[0])]
        mine_output = self.policy.mine(ve_state, B_shuffled)
        mine_output = torch.clamp(mine_output, max=10.0)
        marginal = torch.exp(mine_output).mean()

        # 互信息下界估计
        print("marginal", marginal)
        print("joint - torch.log(marginal+1e-10)", joint - torch.log(torch.clamp(marginal, min=1e-8)))
        mi_loss = -torch.log2(torch.exp(torch.tensor(1.0))) * (joint - torch.log(marginal + 1e-10))
        print("loss", mi_loss)

        # emv_obs = emv_state.view(1,-1).expand(num, -1)
        relate_emv = obs_emv.squeeze(1)
        joint_emv = self.policy.mine_emv(emv_state, relate_emv).mean()

        # 负样本 (A, B_shuffled)
        B_shuffled_emv = relate_emv[torch.randperm(relate_emv.shape[0])]
        mine_output_emv = self.policy.mine_emv(emv_state, B_shuffled_emv)
        mine_output_emv = torch.clamp(mine_output_emv, max=10.0)
        marginal_emv = torch.exp(mine_output_emv).mean()

        # 互信息下界估计
        print("marginal", marginal_emv)
        print("joint - torch.log(marginal+1e-10)", joint_emv - torch.log(torch.clamp(marginal_emv, min=1e-8)))
        mi_loss_emv = -torch.log2(torch.exp(torch.tensor(1.0)))*(joint - torch.log(marginal + 1e-10))
        print("loss",mi_loss_emv)
        loss =mi_loss_emv + mi_loss + loss_total
        self.policy.gnn_optimizer.zero_grad()
        loss.backward()
        self.policy.gnn_optimizer.step()

    def prep_training_gnn(self):
        self.policy.gnn.train()
        self.policy.rnn.train()

    def prep_eval_gnn(self):
        self.policy.gnn.eval()
        self.policy.rnn.eval()

    def train_ve(self, buffer, num_agents, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]

        advantages_copy = advantages.copy()
        active_masks = buffer.active_masks.copy()
        for i in range(num_agents):
            active_masks[buffer.step_ve[i]-1,:,i] = 0
        advantages_copy[active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        # advantages[buffer.active_masks[:-1] == 0.0] = np.nan
        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator_ve(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppo_update(sample, update_actor)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        # self.policy.icm.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        # self.policy.icm.eval()

    def eval_gnn(self):
        self.policy.actor.eval()
        self.policy.rnn.eval()




