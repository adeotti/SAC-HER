import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import torch.nn.functional as F

class custom(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
    
    def reset(self,**kwargs):
        obs,info = super().reset(**kwargs)
        self.env.unwrapped.data.qpos[0] = .3 
        self.env.unwrapped.data.qpos[1] = .5 
        self.env.unwrapped.data.qpos[17] = .4
        return obs,info

    def step(self,action):
        return super().step(action)

def process_obs(obs:dict):
    observation = obs.get("observation")[:9]
    achieved_goal = obs.get("achieved_goal")
    desired_goal = obs.get("desired_goal")
    return torch.from_numpy(np.append(observation,(achieved_goal,desired_goal))).to(dtype=torch.float32)

def make_env():
    x = gym.make("FetchPickAndPlace-v3",max_episode_steps=100,render_mode="human")
    x = custom(x)
    return x

class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(15,128)
        self.l2 = nn.Linear(128,128)
        self.mean = nn.Linear(128,4)
        self.std = nn.Linear(128,4)
        self.optim = Adam(self.parameters(),lr=3e-4)

    def forward(self,obs):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        mean = self.mean(x)
        std = self.std(x).clamp(-20,2).exp()
        dist = Normal(mean,std)
        pretanh = dist.rsample()
        action = F.tanh(pretanh)
        log = dist.log_prob(pretanh)
        log -= torch.log(1-action.pow(2) + 1e-6)
        log = log.sum(-1,True)
        return action,log,mean
    
policyy = policy()
policyy.load_state_dict(torch.load("./model.pth"))
env = make_env()
state = process_obs(env.reset()[0])
for n in range(1000):
    action = policyy(state)[0]
    st,re,done,trunc,info = env.step(action.detach().numpy())
    if trunc:
        state = process_obs(env.reset()[0])
    env.render()