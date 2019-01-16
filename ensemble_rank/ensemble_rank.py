from baselines.common import explained_variance, zipsame, dataset
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common import colorize
from collections import deque
from baselines.common import set_global_seeds
from baselines.common.mpi_adam import MpiAdam
from baselines.common.cg import cg
from baselines.common.input import observation_placeholder
from baselines.common.policies import build_policy
from contextlib import contextmanager
from baselines.ensemble_rl.trpo import trpo
from pprint import pprint
import pdb

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# 返回一个迭代器，每过一个horizon返回一次数据
def traj_segment_generator(models, env, horizon, stochastic, num_reward = 1):
    # horizon 就是 timesteps_per_batch
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
    ac_space = env.action_space
    new = True
    rew = 0.0
    ob = env.reset()

    cur_ep_ret = 0
    cur_ep_len = 0
    ep_rets = []
    ep_lens = []

    # Initialize history arrays
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros((horizon,num_reward), 'float32')
    vpreds = np.zeros((horizon,num_reward), 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred, _, _ = ensemble_step(ob, models, ac_space, num_reward,stochastic=stochastic)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = ensemble_step(ob, models, ac_space, num_reward, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def ensemble_step(ob,models, ac_space, num_reward, stochastic=True):
    mr_pi = []
    vpred = np.zeros((num_reward))
    for i in range(len(models)):
        pi, v, state, neglogp = models[i].pi.dis_step(ob, stochastic = stochastic)
        #ac, v, state, neglogp = models[i].pi.step(ob, stochastic=stochastic)
        if isinstance(mr_pi,np.ndarray):
            mr_pi = np.vstack((mr_pi,pi))
        else:
            mr_pi = pi
        vpred[i] = v
    mr_pi = mr_pi.T
    exp_mr_pi = np.exp(mr_pi)
    mr_pi = exp_mr_pi / np.sum(exp_mr_pi,axis = 0)
    mr_pi = mr_pi.T
    ac  = morl_ensemble(mr_pi,ensemble_type = 3)
    return ac, vpred, state, neglogp


def morl_ensemble(policy_actions, ensemble_type = 3, weights = None):
    '''  Ensemble multiple policies to select one action
    @Params:
        policy_actions : numpy.array, shape = (n_policies, n_actions)
          every row is a distribution on actions, policy for one reward
        ensemble_type : int
          1 : linear ensemble
              a = argmax_a \sum_i \frac{1}{n_policies} policy_actions_{i,a}
          2 : vote by majority
              (1) for every i-th policy:
                    a_i = argmax_a policy_actions_{i}
              (2) select majority policies from {a_i}
          3 : vote by rank
              (1) transform policy_actions_{i, a} to its rank in policy_actions_{i}
              (2) p_{i, a} = \frac{n_actions - rank(a)}{n_policies - 1}
              (3) linear ensemble p_{i, a}
        weights : numpy.array shape = (n_policies,)
          the weights of every policy, default 1 / n_policies
    @Return:
        action_index : int, range of [0, n_actions - 1]
    '''
    n_policies, n_actions = policy_actions.shape

    for i in range(n_policies):
        assert abs(np.sum(policy_actions[i]) - 1.0) < 1e-6, "The {}-th policy_actions is not a distribution, row_sum must be 1.0".format(i)

    if not isinstance(weights, np.ndarray):
        weights = np.ones(n_policies) / n_policies

    if ensemble_type == 1:
        action_index = np.argmax(np.dot(policy_actions.transpose(), weights))
    elif ensemble_type == 2:
        votes = np.zeros(n_actions)
        max_index = np.argmax(policy_actions, axis = 1)

        for i in range(n_policies):
            votes[max_index[i]] += weights[i]

        # all actions have the same votes
        if len(np.unique(votes)) == 1:
            action_index = np.random.randint(0, n_actions)
        else:
            action_index = np.argmax(votes)
    elif ensemble_type == 3:
        rank = np.argsort(np.argsort(policy_actions, axis = 1), axis = 1)
        rank = rank / (n_policies - 1)
        action_index = np.argmax(np.dot(rank.transpose(), weights))

    return action_index

def add_vtarg_and_adv(seg, gamma, lam, num_reward = 1):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"],axis=0)
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty((T,num_reward), 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]

def learn(*,
        network,
        env,
        total_timesteps,
        timesteps_per_batch=1024, # what to train on
        max_kl=0.001,
        cg_iters=10,
        gamma=0.99,
        lam=1.0, # advantage estimation
        seed=None,
        ent_coef=0.0,
        cg_damping=1e-2,
        vf_stepsize=3e-4,
        vf_iters =3,
        max_episodes=0, max_iters=0,  # time constraint
        callback=None,
        load_path=None,
        num_reward=1,
        **network_kwargs
        ):
    '''
    learn a policy function with TRPO algorithm

    Parameters:
    ----------

    network                 neural network to learn. Can be either string ('mlp', 'cnn', 'lstm', 'lnlstm' for basic types)
                            or function that takes input placeholder and returns tuple (output, None) for feedforward nets
                            or (output, (state_placeholder, state_output, mask_placeholder)) for recurrent nets

    env                     environment (one of the gym environments or wrapped via baselines.common.vec_env.VecEnv-type class

    timesteps_per_batch     timesteps per gradient estimation batch

    max_kl                  max KL divergence between old policy and new policy ( KL(pi_old || pi) )

    ent_coef                coefficient of policy entropy term in the optimization objective

    cg_iters                number of iterations of conjugate gradient algorithm

    cg_damping              conjugate gradient damping

    vf_stepsize             learning rate for adam optimizer used to optimie value function loss

    vf_iters                number of iterations of value function optimization iterations per each policy optimization step

    total_timesteps           max number of timesteps

    max_episodes            max number of episodes

    max_iters               maximum number of policy optimization iterations

    callback                function to be called with (locals(), globals()) each policy optimization step

    load_path               str, path to load the model from (default: None, i.e. no model is loaded)

    **network_kwargs        keyword arguments to the policy / network builder. See baselines.common/policies.py/build_policy and arguments to a particular type of network

    Returns:
    -------

    learnt model

    '''

    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        rank = MPI.COMM_WORLD.Get_rank()
    else:
        nworkers = 1
        rank = 0
    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))
    
    set_global_seeds(seed)
    # 创建policy
    policy = build_policy(env, network, value_network='copy', num_reward = num_reward, **network_kwargs)
    

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    
    trpo_model = []
    for i in range(num_reward):
        trpo_model.append(trpo(timed, policy, ob_space, ac_space, max_kl=max_kl, cg_iters=cg_iters,
        ent_coef=ent_coef,cg_damping=cg_damping,vf_stepsize=vf_stepsize,vf_iters =vf_iters,load_path=load_path,num_reward=num_reward,index = i))
    # 这是一个生成数据的迭代器
    seg_gen = traj_segment_generator(trpo_model, env, timesteps_per_batch, stochastic=True, num_reward = num_reward)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()

    # 双端队列
    lenbuffer = deque(maxlen=40) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=40) # rolling buffer for episode rewards

    if sum([max_iters>0, total_timesteps>0, max_episodes>0])==0:
        # noththing to be done
        return pi

    assert sum([max_iters>0, total_timesteps>0, max_episodes>0]) < 2, \
        'out of max_iters, total_timesteps, and max_episodes only one should be specified'

    while True:
        if callback: callback(locals(), globals())
        if total_timesteps and timesteps_so_far >= total_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        logger.log("********** Iteration %i ************"%iters_so_far)

        with timed("sampling"):
            seg = seg_gen.__next__()

        # 计算累积回报
        add_vtarg_and_adv(seg, gamma, lam , num_reward = num_reward)

        vpredbefore = seg["vpred"]
        tdlamret = seg["tdlamret"]
        for i in range(num_reward):
            trpo_model[i].train(seg["ob"], seg["ac"], seg["adv"][:,i], seg["tdlamret"][:,i])

        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))

        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        if MPI is not None:
            listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        else:
            listoflrpairs = [lrlocal]

        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if rank==0:
            logger.dump_tabular()
        #pdb.set_trace()
    return trpo_model

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]


