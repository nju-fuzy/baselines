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
from contextlib import contextmanager
from pprint import pprint
import pdb

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


class trpo(object):
    def __init__(self, timed, policy, ob_space, ac_space, max_kl=0.001, cg_iters=10,
        ent_coef=0.0,cg_damping=1e-2,vf_stepsize=3e-4,vf_iters =3,load_path=None, num_reward=1, index=1):
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
        
        #################################################################
        # ob ac ret atarg 都是 placeholder
        # ret atarg 此处应该是向量形式
        ob = observation_placeholder(ob_space)

        # 创建pi和oldpi
        with tf.variable_scope(str(index)+"pi"):
            pi = policy(observ_placeholder=ob)
        with tf.variable_scope(str(index)+"oldpi"):
            oldpi = policy(observ_placeholder=ob)
        
        # 每个reward都可以算一个atarg
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

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
        all_var_list = get_trainable_variables(str(index)+"pi")
        # var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("pol")]
        # vf_var_list = [v for v in all_var_list if v.name.split("/")[1].startswith("vf")]
        var_list = get_pi_trainable_variables(str(index)+"pi")
        vf_var_list = get_vf_trainable_variables(str(index)+"pi")

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
            for (oldv, newv) in zipsame(get_variables(str(index)+"oldpi"), get_variables(str(index)+"pi"))])
        
        
        # 计算loss
        compute_losses = U.function([ob, ac, atarg], losses)
        # 计算loss和梯度
        compute_lossandgrad = U.function([ob, ac, atarg], losses + [U.flatgrad(optimgain, var_list)])
        # 计算fvp
        compute_fvp = U.function([flat_tangent, ob, ac, atarg], fvp)
        # 计算值网络的梯度
        compute_vflossandgrad = U.function([ob, ret], U.flatgrad(vferr, vf_var_list))
        

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


        self.MPI = MPI
        self.pi = pi
        self.oldpi = oldpi

        self.compute_losses = compute_losses
        self.compute_lossandgrad = compute_lossandgrad
        self.compute_fvp = compute_fvp
        self.compute_vflossandgrad = compute_vflossandgrad

        self.assign_old_eq_new = assign_old_eq_new
        self.get_flat = get_flat
        self.set_from_flat = set_from_flat
        self.vfadam = vfadam
        # params
        self.max_kl = max_kl
        self.cg_iters = cg_iters
        self.ent_coef = ent_coef
        self.cg_damping = cg_damping
        self.vf_stepsize = vf_stepsize
        self.vf_iters = vf_iters
        
        self.rank = rank
        self.index = index
        self.timed = timed

    def train(self,ob, ac, atarg, tdlamret):

        # 标准化
        atarg = (atarg - np.mean(atarg,axis = 0)) / np.std(atarg,axis=0) # standardized advantage function estimate
        if hasattr(self.pi, "ret_rms"): self.pi.ret_rms.update(tdlamret)
        if hasattr(self.pi, "ob_rms"): self.pi.ob_rms.update(ob) # update running mean/std for policy
        
        ## set old parameter values to new parameter values
        self.assign_old_eq_new()

        args = ob, ac, atarg
        # 算是args的一个sample，每隔5个取出一个
        fvpargs = [arr[::5] for arr in args]
            
            # 这个函数计算fisher matrix 与向量 p 的 乘积
        def fisher_vector_product(p):
            return allmean(self.compute_fvp(p, *fvpargs)) + self.cg_damping * p

        with self.timed("computegrad of " + str(self.index+1) +".th reward"):
            *lossbefore, g = self.compute_lossandgrad(*args)
        lossbefore = allmean(np.array(lossbefore))
        g = allmean(g)
            
        # g是目标函数的梯度
        # 利用共轭梯度获得更新方向
        try:
            if np.allclose(g, 0):
                logger.log("Got zero gradient. not updating")
            else:
                with self.timed("cg of " + str(self.index+1) +".th reward"):
            	    # stepdir 是更新方向
                    stepdir = cg(fisher_vector_product, g, cg_iters=self.cg_iters, verbose=self.rank==0)
                assert np.isfinite(stepdir).all()
            
                shs = .5*stepdir.dot(fisher_vector_product(stepdir))
                lm = np.sqrt(shs / self.max_kl)
                # logger.log("lagrange multiplier:", lm, "gnorm:", np.linalg.norm(g))
                fullstep = stepdir / lm
                expectedimprove = g.dot(fullstep)
                surrbefore = lossbefore[0]
                stepsize = 1.0
                thbefore = self.get_flat()

                # 做10次搜索
                for _ in range(10):
                    thnew = thbefore + fullstep * stepsize
                    self.set_from_flat(thnew)
                    meanlosses = surr, kl, *_ = allmean(np.array(self.compute_losses(*args)))
                    improve = surr - surrbefore
                    logger.log("Expected: %.3f Actual: %.3f"%(expectedimprove, improve))
                    if not np.isfinite(meanlosses).all():
                        logger.log("Got non-finite value of losses -- bad!")
                    elif kl > self.max_kl * 1.5:
                        logger.log("violated KL constraint. shrinking step.")
                    elif improve < 0:
                        logger.log("surrogate didn't improve. shrinking step.")
                    else:
                        logger.log("Stepsize OK!")
                        break
                    stepsize *= .5
                else:
                    logger.log("couldn't compute a good step")
                    self.set_from_flat(thbefore)
            with self.timed("vf"):
                for _ in range(self.vf_iters):
                    for (mbob, mbret) in dataset.iterbatches((ob , tdlamret),
                    include_final_partial_batch=False, batch_size=64):
                        #with tf.Session() as sess:
                        #    sess.run(tf.global_variables_initializer())
                        #    print(sess.run(vferr,feed_dict={ob:mbob,ret:mbret}))
                        g = allmean(self.compute_vflossandgrad(mbob, mbret))
                        self.vfadam.update(g, self.vf_stepsize)
        except:
            print("can't learn")



def get_variables(scope):
    return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

def get_trainable_variables(scope):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

def get_vf_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'vf' in v.name[len(scope):].split('/')]

def get_pi_trainable_variables(scope):
    return [v for v in get_trainable_variables(scope) if 'pi' in v.name[len(scope):].split('/')]


def allmean(x):
    assert isinstance(x, np.ndarray)
    if MPI is not None:
        nworkers = MPI.COMM_WORLD.Get_size()
        out = np.empty_like(x)
        MPI.COMM_WORLD.Allreduce(x, out, op=MPI.SUM)
        out /= nworkers
    else:
        out = np.copy(x)

    return out


