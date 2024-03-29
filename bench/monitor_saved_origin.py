__all__ = ['Monitor', 'get_monitor_files', 'load_results']

import gym
from gym.core import Wrapper
import time
from glob import glob
import csv
import os
import os.path as osp
import json
import numpy as np
import shutil

class Monitor(Wrapper):
    EXT = "monitor.csv"
    f = None

    def __init__(self, env, filename, allow_early_resets=False, reset_keywords=(), info_keywords=(), num_reward = 1, reward_type = 1):
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        '''
        self.myresults_writer = MyResultWriter(
        filename,
        env,
        reset_keywords,
        info_keywords,
        num_reward
        )
        '''
        # orgin version
        self.results_writer = ResultsWriter(
            filename,
            header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
            extra_keys=reset_keywords + info_keywords
        )
        
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets
        self.rewards = None
        self.needs_reset = True
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_times = []
        self.total_steps = 0
        self.current_reset_info = {} # extra info about the current episode, that was passed in during reset()
        self.num_reward = num_reward
        self.reward_type = reward_type

    def reset(self, **kwargs):
        self.reset_state()
        for k in self.reset_keywords:
            v = kwargs.get(k)
            if v is None:
                raise ValueError('Expected you to pass kwarg %s into reset'%k)
            self.current_reset_info[k] = v
        #print(self.current_reset_info)
        return self.env.reset(**kwargs)

    def reset_state(self):
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError("Tried to reset an environment before done. If you want to allow early resets, wrap your env with Monitor(env, path, allow_early_resets=True)")
        self.rewards = []
        self.needs_reset = False


    def step(self, action):
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        ob, rew, done, info = self.env.step(action)
        if self.num_reward == 1:
            self.update(ob, rew, done, info)
            return (ob, rew, done, info)
        else:
            self.update(ob, rew[0], done, info)
        if self.reward_type == 0:
            return (ob, rew, done, info)
        else:
            return (ob, rew[self.reward_type - 1], done, info)

    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            # epnew 成为向量
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            ''' 
            epinfo = {"r": eprew, "l": eplen, "t": round(time.time() - self.tstart, 6)}
            '''
            epinfo = {"r": round(eprew, 6), "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            
            self.results_writer.write_row(epinfo)
            '''
            self.myresults_writer.write_row(epinfo)
            '''
            if isinstance(info, dict):
                info['episode'] = epinfo
            #print("epinfo",epinfo)
        self.total_steps += 1

    def close(self):
        if self.f is not None:
            self.f.close()

    def get_total_steps(self):
        return self.total_steps

    def get_episode_rewards(self):
        return self.episode_rewards

    def get_episode_lengths(self):
        return self.episode_lengths

    def get_episode_times(self):
        return self.episode_times

class LoadMonitorResultsError(Exception):
    pass


class ResultsWriter(object):
    def __init__(self, filename=None, header='', extra_keys=()):
        self.extra_keys = extra_keys
        if filename is None:
            self.f = None
            self.logger = None
        else:
            if not filename.endswith(Monitor.EXT):
                if osp.isdir(filename):
                    filename = osp.join(filename, Monitor.EXT)
                else:
                    filename = filename + "." + Monitor.EXT
            #if os.path.exists(filename):
                #print('remove','!'*50)
                #os.remove(filename)
            if not os.path.exists(filename):
                os.mknod(filename)
            self.f = open(filename, "wt")
            if isinstance(header, dict):
                header = '# {} \n'.format(json.dumps(header))
            self.f.write(header)
            self.logger = csv.DictWriter(self.f, fieldnames=('r', 'l', 't')+tuple(extra_keys))
            self.logger.writeheader()
            self.f.flush()
    def write_row(self, epinfo):
        if self.logger:
            #print('file',self.f)
            #print('epinfo',epinfo)
            self.logger.writerow(epinfo)
            self.f.flush()

class MyResultWriter(object):
    def __init__(self,filename,env,reset_keywords,info_keywords,num_reward):
        #print('filename',filename)         # /home/lamda3/logs/freeway161400/SEED0/0.7
        self.results_writer=[]
        self.num_reward = num_reward
        part1, part2 =filename.split('SEED')[-2],filename.split('SEED')[-1]
        for i in range(num_reward):
            new_dir=part1+'r'+str(i)+'-'+part2.split('/')[0]+'/'
            if not  os.path.exists(new_dir):
                os.mkdir(new_dir)
            #print('new_dir',new_dir)       # /home/lamda3/logs/freeway161400/r1-0/ 
            #print('files:',os.listdir(new_dir))
            
            new_filename=part1+'r'+str(i)+'-'+part2
            #print('new_file',new_filename) # /home/lamda3/logs/freeway161400/r1-0/0.7
    
            self.results_writer.append(ResultsWriter(
                new_filename,
                header={"t_start": time.time(), 'env_id' : env.spec and env.spec.id},
                extra_keys=reset_keywords + info_keywords)
            )
    def write_row(self,epinfo):
        for i in range(self.num_reward):
            newepinfo={}
            for key in epinfo.keys():
                if key != 'r':
                    newepinfo[key]=epinfo[key]
                else:
                    newepinfo['r']=epinfo['r'][i]
            self.results_writer[i].write_row(newepinfo)

def get_monitor_files(dir):
    return glob(osp.join(dir, "*" + Monitor.EXT))

def load_results(dir):
    import pandas
    monitor_files = (
        glob(osp.join(dir, "*monitor.json")) +
        glob(osp.join(dir, "*monitor.csv"))) # get both csv and (old) json files
    if not monitor_files:
        raise LoadMonitorResultsError("no monitor files of the form *%s found in %s" % (Monitor.EXT, dir))
    dfs = []
    headers = []
    for fname in monitor_files:
        with open(fname, 'rt') as fh:
            if fname.endswith('csv'):
                firstline = fh.readline()
                if not firstline:
                    continue
                assert firstline[0] == '#'
                header = json.loads(firstline[1:])
                df = pandas.read_csv(fh, index_col=None)
                headers.append(header)
            elif fname.endswith('json'): # Deprecated json format
                episodes = []
                lines = fh.readlines()
                header = json.loads(lines[0])
                headers.append(header)
                for line in lines[1:]:
                    episode = json.loads(line)
                    episodes.append(episode)
                df = pandas.DataFrame(episodes)
            else:
                assert 0, 'unreachable'
            df['t'] += header['t_start']
        dfs.append(df)
    df = pandas.concat(dfs)
    df.sort_values('t', inplace=True)
    df.reset_index(inplace=True)
    df['t'] -= min(header['t_start'] for header in headers)
    df.headers = headers # HACK to preserve backwards compatibility
    return df

def test_monitor():
    env = gym.make("CartPole-v1")
    env.seed(0)
    mon_file = "/tmp/baselines-test-%s.monitor.csv" % uuid.uuid4()
    menv = Monitor(env, mon_file)
    menv.reset()
    for _ in range(1000):
        _, _, done, _ = menv.step(0)
        if done:
            menv.reset()

    f = open(mon_file, 'rt')

    firstline = f.readline()
    assert firstline.startswith('#')
    metadata = json.loads(firstline[1:])
    assert metadata['env_id'] == "CartPole-v1"
    assert set(metadata.keys()) == {'env_id', 'gym_version', 't_start'},  "Incorrect keys in monitor metadata"

    last_logline = pandas.read_csv(f, index_col=None)
    assert set(last_logline.keys()) == {'l', 't', 'r'}, "Incorrect keys in monitor logline"
    f.close()
    os.remove(mon_file)
