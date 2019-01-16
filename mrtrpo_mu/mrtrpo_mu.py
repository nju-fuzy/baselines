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
from baselines.common.mr_policy import build_policy
from contextlib import contextmanager
from baselines.mrtrpo_mu.optim import get_coefficient
from pprint import pprint
import pdb
import os

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

# 返回一个迭代器，每过一个horizon返回一次数据
def traj_segment_generator(pi, env, horizon, stochastic, num_reward = 1):
    # horizon 就是 timesteps_per_batch
    # Initialize state variables
    t = 0
    ac = env.action_space.sample()
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
        ac, vpred, _, _ = pi.step(ob, stochastic=stochastic)
        #print(vpred)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens}
            _, vpred, _, _ = pi.step(ob, stochastic=stochastic)
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        obs[i] = ob
        vpreds[i] = vpred
        #print("==========================")
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        ob, rew, new, _ = env.step(ac)
        rews[i] = rew
        #print('rew')
        #print(rews)
        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            ob = env.reset()
        t += 1

def add_vtarg_and_adv(seg, gamma, lam, num_reward = 1):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"],axis=0)
    #print(seg["vpred"])
    #print(vpred.shape)
    #pdb.set_trace()
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty((T,num_reward), 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    #pdb.set_trace()
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

    cpus_per_worker = 1
    U.get_session(config=tf.ConfigProto(
            allow_soft_placement=True,
            inter_op_parallelism_threads=cpus_per_worker,
            intra_op_parallelism_threads=cpus_per_worker
    ))
    
    set_global_seeds(seed)
    # 创建policy
    policy = build_policy(env, network, value_network='copy', num_reward = num_reward, **network_kwargs)
    
    process_dir = logger.get_dir()
    save_dir = process_dir.split('Data')[-2] + 'log/seed' + process_dir[-1] +'/'
    os.makedirs(save_dir, exist_ok = True)
    coe_save = []
    impro_save = []
    grad_save = []
    adj_save = []
    coe = np.ones((num_reward))/num_reward

    np.set_printoptions(precision=3)
    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    
    #################################################################
    # ob ac ret atarg 都是 placeholder
    # ret atarg 此处应该是向量形式
    ob = observation_placeholder(ob_space)

    # 创建pi和oldpi
    with tf.variable_scope("pi"):
        pi = policy(observ_placeholder=ob)
    with tf.variable_scope("oldpi"):
        oldpi = policy(observ_placeholder=ob)
    
    # 每个reward都可以算一个atarg
    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None,num_reward]) # Empirical return

    ac = pi.pdtype.sample_placeholder([None])
    
    #此处的KL div和entropy与reward无关
    ##################################
    kloldnew = oldpi.pd.kl(pi.pd)
    ent = pi.pd.entropy()
    meankl = tf.reduce_mean(kloldnew)
    meanent = tf.reduce_mean(ent)
    # entbonus 是entropy loss
    entbonus = ent_coef * meanent
    #################################
    
    ###########################################################
    # vferr 用来更新 v 网络
    vferr = tf.reduce_mean(tf.square(pi.vf - ret))
    ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) 
    # advantage * pnew / pold
    surrgain = tf.reduce_mean(ratio * atarg)

    # optimgain 用来更新 policy 网络, 应该每个reward有一个
    optimgain = surrgain + entbonus
    losses = [optimgain, meankl, entbonus, surrgain, meanent]
    loss_names = ["optimgain", "meankl", "entloss", "surrgain", "entropy"]
   
    ###########################################################
    dist = meankl
    
    # 定义要优化的变量和 V 网络 adam 优化器
    all_var_list = get_trainable_variables("pi")
    # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
    # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
    var_list = get_pi_trainable_variables("pi")
    vf_var_list = get_vf_trainable_variables("pi")

    vfadam = MpiAdam(vf_var_list)

    # 把变量展开成一个向量的类
    get_flat = U.GetFlat(var_list)

    # 这个类可以把一个向量分片赋值给var_list里的变量
    set_from_flat = U.SetFromFlat(var_list)
    # kl散度的梯度
    klgrads = tf.gradients(dist, var_list)
    
    ####################################################################
    # 拉直的向量
    flat_tangent = tf.placeholder(dtype=tf.float32, shape=[None], name="flat_tan")

    # 把拉直的向量重新分成很多向量
    shapes = [var.get_shape().as_list() for var in var_list]
    start = 0
    tangents = []
    for shape in shapes:
        sz = U.intprod(shape)
        tangents.append(tf.reshape(flat_tangent[start:start+sz], shape))
        start += sz
    ####################################################################
    
    ####################################################################
    # 把kl散度梯度与变量乘积相加
    gvp = tf.add_n([tf.reduce_sum(g*tangent) for (g, tangent) in zipsame(klgrads, tangents)]) #pylint: disable=E1111
    # 把gvp的梯度展成向量
    fvp = U.flatgrad(gvp, var_list)
    ####################################################################

    # 用学习后的策略更新old策略
    assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(get_variables("oldpi"), get_variables("pi"))])
    
    
    # 计算loss
    compute_losses = U.function([ob, ac, atarg], losses)
    # 计算loss和梯度
    compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
    # 计算fvp
    compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
    # 计算值网络的梯度
    compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))

    @contextmanager
    def timed(msg):
        if rank == 0:
            print(colorize(msg, color='magenta'))
            tstart = time.time()
            yield
            print(colorize("done in %.3f seconds"%(time.time() - tstart), color='magenta'))
        else:
            yield

    def allmean(x):
        assert isinstance(x, np.ndarray)
        if MPI is not None:
            out = np.empty_like(x)
            MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
            out /= nworkers
        else:
            out = np.copy(x)

        return out
    
    # 初始化variable
    U.initialize()
    if load_path is not None:
        pi.load(load_path)
    
    # 得到初始化的参数向量
    th_init = get_flat()
    if MPI is not None:
        MPI.COMM_WORLD.Bcast(th_init, root=0)
    
    # 把向量the_init的值分片赋值给var_list
    set_from_flat(th_init)

    #同步
    vfadam.sync()
    print("Init param sum", th_init.sum(), flush=True)

    # Prepare for rollouts
    # ----------------------------------------

    # 这是一个生成数据的迭代器
    seg_gen = traj_segment_generator(pi, env, timesteps_per_batch, stochastic=True, num_reward = num_reward)

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
###########$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ToDo
        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))

        # ob, ac, atarg, tdlamret 的类型都是ndarray
        #ob, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        _, ac, atarg, tdlamret = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"]
        #print(seg['ob'].shape,type(seg['ob']))
        #print(seg['ac'],type(seg['ac']))
        #print(seg['adv'],type(seg['adv']))
        #print(seg["tdlamret"].shape,type(seg['tdlamret']))
        vpredbefore = seg["vpred"] # predicted value function before udpate

        # 标准化
        #print("============================== atarg =========================================================")
        #print(atarg)
        atarg = (atarg - np.mean(atarg,axis = 0)) / np.std(atarg,axis=0) # standardized advantage function estimate
        #atarg = (atarg) / np.max(np.abs(atarg),axis=0)
        #print('======================================= standardized atarg ====================================')
        #print(atarg)
        if hasattr(pi, "ret_rms"): pi.ret_rms.update(tdlamret)
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy
        
        ## set old parameter values to new parameter values
        assign_old_eq_new()

        G = None
        S = None
        mr_lossbefore = np.zeros((num_reward,len(loss_names)))
        grad_norm = np.zeros((num_reward+1))
        for i in range(num_reward):
            args = seg["ob"], seg["ac"], atarg[:,i]
            #print(atarg[:,i])
            # 算是args的一个sample，每隔5个取出一个
            fvpargs = [arr[::5] for arr in args]
            
            # 这个函数计算fisher matrix 与向量 p 的 乘积
            def fisher_vector_product(p):
                return allmean(compute_fvp(p, *fvpargs)) + cg_damping * p

            with timed("computegrad of " + str(i+1) +".th reward"):
                *lossbefore, g = compute_lossandgrad(*args)
            lossbefore = allmean(np.array(lossbefore))
            mr_lossbefore[i] = lossbefore
            g = allmean(g)
            #print("***************************************************************")
            #print(g)
            if isinstance(G,np.ndarray):
                G = np.vstack((G,g))
            else:
                G = g
            
            # g是目标函数的梯度
            # 利用共轭梯度获得更新方向
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with timed("cg of " + str(i+1) +".th reward"):
            	    # stepdir 是更新方向
                    stepdir = cg(fisher_vector_product, g, cg_iters=cg_iters, verbose=rank==0)
                    shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                    lm = np.sqrt(shs / max_kl)
                    # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                    fullstep = stepdir / lm
                    grad_norm[i] = np.linalg.norm(fullstep)
                assert np.isfinite(stepdir).all()
                if isinstance(S,np.ndarray):
                    S = np.vstack((S,stepdir))
                else:
                    S = stepdir
        #print('======================================= G ====================================')
        #print(G)
        #print('======================================= S ====================================')
        #print(S)
        new_coe = get_coefficient( G, S)
        #coe = 0.99 * coe + 0.01 * new_coe
        coe = new_coe
        coe_save.append(coe)
        #根据梯度的夹角调整参数
        GG = np.dot(S, S.T)
        D = np.sqrt(np.diag(1/np.diag(GG)))
        GG = np.dot(np.dot(D,GG),D)
        #print('======================================= inner product ====================================')
        #print(GG)
        adj = np.sum(GG) / (num_reward ** 2)
        #print('======================================= adj ====================================')
        #print(adj)
        adj_save.append(adj)
        adj_max_kl = adj * max_kl
        #################################################################
        grad_norm = grad_norm * np.sqrt(adj)
        stepdir = np.dot(coe, S)
        g = np.dot(coe, G)
        lossbefore = np.dot(coe,mr_lossbefore)
        #################################################################
        
        shs = .5*stepdir.dot(fisher_vector_product(stepdir))
        lm = np.sqrt(shs / adj_max_kl)
        # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
        fullstep = stepdir / lm
        grad_norm[num_reward] = np.linalg.norm(fullstep)
        grad_save.append(grad_norm)
        expectedimprove = g.dot(fullstep)
        surrbefore = lossbefore[0]
        stepsize = 1.0
        thbefore = get_flat()

        def compute_mr_losses():
            mr_losses = np.zeros((num_reward,len(loss_names)))
            for i in range(num_reward):
                args = seg["ob"], seg["ac"], atarg[:,i]
                one_reward_loss = allmean(np.array(compute_losses(*args)))
                mr_losses[i] = one_reward_loss
            mr_loss = np.dot(coe,mr_losses)
            return mr_loss,mr_losses

        # 做10次搜索
        for _ in range(10):
            thnew = thbefore + fullstep * stepsize
            set_from_flat(thnew)
            mr_loss_new,mr_losses_new = compute_mr_losses()
            mr_impro = mr_losses_new - mr_lossbefore
            meanlosses = surr, kl, *_ = allmean(np.array(mr_loss_new))
            improve = surr - surrbefore
            logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
            if not np.isfinite(meanlosses).all():
                logger.log("Got non-finite value of losses -- bad!")
            elif kl > adj_max_kl * 1.5:
                logger.log("violated KL constraint. shrinking step.")
            elif improve < 0:
                logger.log("surrogate didn't improve. shrinking step.")
            else:
                logger.log("Stepsize OK!")
                impro_save.append(np.hstack((mr_impro[:,0],improve)))
                break
            stepsize *= .5
        else:
            logger.log("couldn't compute a good step")
            set_from_flat(thbefore)
        if nworkers > 1 and iters_so_far % 20 == 0:
            paramsums = MPI.COMM_WORLD.allgather((thnew.sum(), vfadam.getflat().sum())) # list of tuples
            assert all(np.allclose(ps, paramsums[0]) for ps in paramsums[1:])

        for (lossname, lossval) in zip(loss_names, meanlosses):
            logger.record_tabular(lossname, lossval)

        with timed("vf"):
            #print('======================================= tdlamret ====================================')
            #print(seg["tdlamret"])
            for _ in range(vf_iters):
                for (mbob, mbret) in dataset.iterbatches((seg["ob"], seg["tdlamret"]),
                include_final_partial_batch=False, batch_size=64):
                    #with tf.Session() as sess:
                    #    sess.run(tf.global_variables_initializer())
                    #    aaa = sess.run(pi.vf,feed_dict={ob:mbob,ret:mbret})
                    #    print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
                    #    print(aaa.shape)
                    #    print(mbret.shape)
                    g = allmean(compute_vflossandgrad(mbob, mbret))
                    vfadam.update(g, vf_stepsize)
            #print(mbob,mbret)
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
    np.save(save_dir + 'coe.npy',coe_save)
    np.save(save_dir + 'grad.npy',grad_save)
    np.save(save_dir + 'improve.npy',impro_save)
    np.save(save_dir + 'adj.npy',adj_save)
    return pi

def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]

def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]


