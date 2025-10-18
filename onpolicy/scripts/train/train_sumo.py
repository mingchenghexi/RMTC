#!/usr/bin/env python
import sys
import os
import wandb
wandb.login(key="1951d3299b50ccbe91fb1dbb0c4e767fc1710c27")
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import sys

import sys
import cProfile
#
sys.path.insert(0,'/home/PhD/ChenM/experiment/CoEMV/')
#
# # sys.path.append('/home/data/cm/experiment/CoEMV_origin')
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#
import os
os.environ["WANDB_API_KEY"] = "1951d3299b50ccbe91fb1dbb0c4e767fc1710c27"
os.environ["WANDB_MODE"] = "offline"
# print ("----------------------------------------")
# print (sys.path)
# print ("----------------------------------------")

# import utils.gen_utils as utils
import time



from onpolicy.config import get_config
# from onpolicy.envs.mpe.MPE_env import MPEEnv
from onpolicy.envs.sumo_files_marl.config import config as config_env
from onpolicy.envs.sumo_files_marl.SUMO_env import SUMOEnv

from onpolicy.envs.env_wrappers import SubprocVecEnv, DummyVecEnv

"""Train script for MPEs."""



def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args, rank)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "SUMO":
                env = SUMOEnv(all_args,rank)
            else:
                print("Can not support the " +
                      all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--num_agents", type=int, default=0, help="the number of the agents.")
    parser.add_argument('--scenario_name', type=str, default='sumo_test', help="Which scenario to run on")

    all_args = parser.parse_known_args(args)[0]

    return all_args


def main(args):
    # profiler = cProfile.Profile()
    # profiler.enable()  # 开始性能分析
    # # 你的 main 函数逻辑...
    # profiler.disable()  # 停止性能分析
    # profiler.print_stats()  # 打印分析结果

    parser = get_config()
    all_args = parse_args(args, parser)

    all_args.episode_length = config_env['episode']['rollout_length']  # 获取回合长度
    all_args.num_actions = config_env['environment']['num_actions']  # 获取可用动作数量

    # 根据算法名称设置相关参数
    if all_args.algorithm_name == "rmappo":
        print("u are choosing to use rmappo, we set use_recurrent_policy to be True")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("u are choosing to use mappo, we set use_recurrent_policy & use_naive_recurrent_policy to be False")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("u are choosing to use ippo, we set use_centralized_V to be False")
        all_args.use_centralized_V = True
        # all_args.use_recurrent_policy = False
    else:
        raise NotImplementedError

    # 检查策略共享的合法性
    assert (all_args.share_policy == True and all_args.scenario_name == 'simple_speaker_listener') == False, (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py.")

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:1")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
    #     str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))
    setproctitle.setproctitle(str(time.strftime('%Y-%m-%d-%H-%M')))

    # run dir
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results_sumo") / all_args.env_name / all_args.experiment_name / all_args.algorithm_name

    if not run_dir.exists():
        os.makedirs(str(run_dir))

    ##### save args
    argsDict = all_args.__dict__  # 获取参数字典
    with open(str(run_dir) + '/args.txt', 'w') as f:
        f.writelines('------------------ start save arguments------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         notes=socket.gethostname(),
                         name=str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed) + "_" + str(time.strftime('%Y-%m-%d-%H-%M')),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    else:
        curr_run = 'seed_' + str(all_args.seed) + '_'
        if not run_dir.exists():
            curr_run += 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run += 'run1'
            else:
                curr_run += 'run%i' % (max(exst_run_nums) + 1)

        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    all_args.num_agents = len(envs.action_space)

    #####
    if type(all_args.use_ours) == type('str'):
        all_args.use_ours = eval(all_args.use_ours)
    if type(all_args.use_trans_critic) == type('str'):
        all_args.use_trans_critic = eval(all_args.use_trans_critic)
    if type(all_args.use_kl) == type('str'):
        all_args.use_kl = eval(all_args.use_kl)
    if type(all_args.use_3cons) == type('str'):
        all_args.use_3cons = eval(all_args.use_3cons)
    if type(all_args.use_trans_hidden) == type('str'):
        all_args.use_trans_hidden = eval(all_args.use_trans_hidden)
    if type(all_args.part_mask) == type('str'):
        all_args.part_mask = eval(all_args.part_mask)
    if type(all_args.epsilon_decay) == type('str'):
        all_args.epsilon_decay = eval(all_args.epsilon_decay)
    if type(all_args.use_sym_loss) == type('str'):
        all_args.use_sym_loss = eval(all_args.use_sym_loss)

    ####
    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
        "num_vehicle_agents": all_args.num_vehicle_agents,
        "num_emv_agents": all_args.num_emv_agents
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared_policy.sumo_runner import SUMORunner as Runner
    else:
        from onpolicy.runner.shared.sumo_runner import SUMORunner as Runner
    runner = Runner(config)
    # runner.run()
    for i in range(1, 10):
        runner.eval(441)
    # if not all_args.use_test:
    #     runner.run()
    # else:
    #     runner.test()
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if not all_args.use_eval:
        if all_args.use_wandb:
            run.finish()
        else:
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":

    for i in range(5, 10):
        # for i in range(1, 2):
        # for i in range(2, 3):
        # for i in range(3, 4):
        # for i in range(4, 5):
        args = ['--env_name', 'SUMO', '--algorithm_name', 'rmappo', '--num_agents', '0', '--seed', '1',
                '--experiment_name', 'test', #9
                '--n_training_threads', '8', '--n_rollout_threads', '2', '--num_mini_batch', '1', '--num_env_steps',
                '9000000', #17
                '--ppo_epoch', '10', '--gain', '0.01', '--lr', '5e-4', '--critic_lr', '5e-5', '--use_wandb', 'False',
                '--use_ReLU']
        #
        # args[7] = '6'
        args[7] = str(i)
        threads = ['2', '32', '64', '128']
        # num_mini_batch = ['2', '32', '16', '16']
        # num_mini_batch = ['4', '32', '16', '16']
        num_mini_batch = ['8', '32', '16', '16']
        index_ = 0
        index_ = 0
        # args[13] = threads[index_]
        args[13] = '1'  # n_rollout_threads
        args[11] = threads[index_]  # n_training_threads
        args[15] = num_mini_batch[index_]  # num_mini_batch

        args[3] = 'ippo'

        index = 0

        if index == 0:
            cong_flag = 'SUMO-best-finetune-mask-1-grid4x4/config3/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15] = '4'
            index_port_add = 0

        elif index == 1:
            cong_flag = 'SUMO-best-finetune-mask-1-fenglin/config4-2/'  ####
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15] = '4'
            index_port_add = 3

        elif index == 2:
            cong_flag = 'SUMO-best-finetune-mask-1-nanshan/config7-5-1-1/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15], args[23], args[25] = '4', '5e-4', '5e-5'
            index_port_add = 4


        elif index == 4:
            cong_flag = 'SUMO-best-finetune-mask-1-arterial4x4/config3/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15], args[23], args[25] = '4', '5e-4', '5e-5'
            index_port_add = 6

        elif index == 5:
            cong_flag = 'SUMO-best-finetune-mask-1-ingolstadt21/config8/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15] = '4'
            index_port_add = 13  #

        elif index == 6:
            cong_flag = 'SUMO-best-finetune-mask-1-cologne8/config3/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15], args[23], args[25] = '4', '5e-4', '5e-5'  # minibatch, lr, critic-lr, seed
            index_port_add = 14

        elif index == 7:
            cong_flag = 'SUMO-best-finetune-mask-1-grid5x5/config4/'
            args.extend(
                ['--use_ours', 'True', '--use_trans_critic', 'False', '--use_kl', 'True', '--use_K', '1', '--use_3cons',
                 'False'])
            args.extend(['--use_trans_hidden', 'True'])  ###
            args[15] = '4'
            index_port_add = 19  ###

        cong_lst = ['grid4x4-flatten-ours-', 'fenglin-flatten-ours-', 'nanshan-flatten-ours-',
                    'arterial4x4-flatten-ours-', 'ingolstadt21-flatten-ours-', 'cologne8-flatten-ours-',
                    'grid5x5-flatten-ours-']

        sumocfg_files_lst = [
            "sumo_files_marl/scenarios/resco_envs/cologne8_test/cologne8.sumocfg",
            # "sumo_files_marl/scenarios/Jinan/jinan.sumocfg",
            # "sumo_files_marl/scenarios/resco_envs/arterial4x4_trip/arterial4x4.sumocfg",
            
            # "sumo_files_marl/scenarios/resco_envs/congestion/grid_300/grid4x4.sumocfg",
            # 'sumo_files_marl/scenarios/resco_envs/congestion/grid_200/grid4x4.sumocfg',
            "sumo_files_marl/scenarios/large_grid2_trip/exp_1.sumocfg",
            "sumo_files_marl/scenarios/sumo_fenglin_base_road_trip/base.sumocfg",
            "sumo_files_marl/scenarios/nanshan/osm.sumocfg",
            'sumo_files_marl/scenarios/resco_envs/arterial4x4_trip/arterial4x4.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/ingolstadt21/ingolstadt21.sumocfg',
            'sumo_files_marl/scenarios/resco_envs/cologne8/cologne8.sumocfg',
            'sumo_files_marl/scenarios/large_grid2/exp_0.sumocfg'
        ]
        state_key = config_env['environment']['state_key']
        # print(sumocfg_files_lst[index])
        args.extend(['--state_key', state_key, '--port_start', '14444', '--sumocfg_files', sumocfg_files_lst[index]])
        ################################################################

        args[9] = cong_flag + cong_lst[3] + args[3] + '_thr_' + args[13]

        ####
        # epsilon greedy
        args.extend(['--epsilon_decay', 'True'])


        main(args)
        sys.exit()
