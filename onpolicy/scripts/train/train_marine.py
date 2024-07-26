#!/usr/bin/env python
from email.policy import default
import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import random
from onpolicy.config import get_config
from onpolicy.envs.marine.MarineMultiEnv import Marine

from onpolicy.envs.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from onpolicy.utils.logging import get_logger
from os.path import dirname, abspath


"""Train script for Marine environment."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Marine":
                print(all_args)
                env = Marine(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "Marine":
                # print(all_args)
                env = Marine(all_args)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    """add hetnet specific parameters"""
    parser.add_argument("--n_types", type=int, default=2,
                        help="number of different heterogeneous types")
    parser.add_argument("--num_P", type=int, default=3,
                        help="number of routng agents for Marine")
    parser.add_argument("--num_A", type=int, default=2,
                        help="number of logistics agents for Marine")
    parser.add_argument("--dim", type=int, default=10,
                        help="size of grid dimension")
    parser.add_argument("--episode_limit", type=int, default=100,
                        help="max steps in a single episode before timing out")
    parser.add_argument("--num_X", type=int, default=0,
                        help="number of X agents for 3-class problems")

    parser.add_argument("--with_state", type=bool, default=True,
                        help="whether or not to add state in the graph composition")
    parser.add_argument("--with_two_state", type=bool, default=True,
                        help="include two state nodes in the graph, ignored if --with_state is False")

    parser.add_argument("--use_tune", type=bool, default=False,
                        help="")

    parser.add_argument('--nenemies', type=int, default=0, help="**No enemies in ocean env**")
    parser.add_argument('--vision', type=int, default=1,
                     help="Vision of agents") # routing
    parser.add_argument('--mode', default='cooperative', type=str,
                        help='cooperative|competitive|mixed (default: cooperative)')
    parser.add_argument('--intensity_levels', default=4, type=int,
                     help='the number of refueling efficiency levels (default: 4)')
    parser.add_argument('--A_vision', type=int, default=1,
                     help="Vision of A agents. If -1, defaults to blind") # logistics
    parser.add_argument('--tensor_obs', action="store_true", default=False,
                     help="Do you want a tensorized observation?")
    parser.add_argument('--limited_refuel', action="store_true", default=False,
                        help="To limit the ability to refuel")
    parser.add_argument('--num_dest', type=int, default=1,
                        help="Number of destination")
    parser.add_argument('--logis_init_position_over_the_space', action='store_true', default=False, help='either set the init position of logis all over the plane or the bottom half circle')
    parser.add_argument('--schedule_rewards', action='store_true', default=False, help='if we want to add schedule rewards to the total rewards (matters only if env = OceanRealMulti; if env == OceanRealMultiScheduling, assumed always True)')

    all_args = parser.parse_known_args(args)[0]

    return all_args



def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)
    # all_args.use_wandb = False
    print(all_args.use_wandb)
    if all_args.algorithm_name == "rmappo" or all_args.algorithm_name == "hetgat_mappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), "check recurrent policy!"
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError


    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    scenario_name = '%dx%d'%(all_args.dim, all_args.dim)
    if all_args.num_dest > 1:
        scenario_name = scenario_name + '_num_dest_%d'%all_args.num_dest
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name / scenario_name
    run_dir = Path(run_dir)


    if not run_dir.exists():
        os.makedirs(str(run_dir))

    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.env_name,
                         dir=str(run_dir),
                         job_type="training",
                         #  resume=True,
                         reinit=True)
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))

    print('Run dir: ', run_dir)

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)
    random.seed(all_args.seed)
    deterministic = False
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # env
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_A + all_args.num_P

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    # run experiments
    if all_args.share_policy:
        from onpolicy.runner.shared.marine_runner import MarineRunner as Runner
    else:
        raise NotImplementedError

    runner = Runner(config)
    runner.run()


    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    
    main(sys.argv[1:])





















