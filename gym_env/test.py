import gymnasium as gym
import gymnasium_robotics,torch,random
gym.register_envs(gymnasium_robotics)
import numpy as np
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import torch.nn.functional as F
from gymnasium.wrappers import Autoreset
from gymnasium.spaces import Box,Dict

class custom(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.observation_space = Dict(
            {
            "observation" : Box(-np.inf,np.inf,(9,),np.float64),
            "achieved_goal" : Box(-np.inf,np.inf,(3,),np.float64),
            "desired_goal" : Box(-np.inf,np.inf,(3,),np.float64)
            }
        )
    
    def reset(self,**kwargs):
        obs,info = super().reset(**kwargs)
        target = random.choice([True,True,False])
        self.env.unwrapped.unwrapped.target_in_the_air = target
        obs["observation"] = obs["observation"][:9]
        self.env.unwrapped.data.qpos[0] = .3  
        self.env.unwrapped.data.qpos[1] = .5  
        self.env.unwrapped.data.qpos[17] = .4
        return obs,info

    def step(self,action):
        state,reward,done,trunc,info = super().step(action)
        state["observation"] = state["observation"][:9]
        return state,reward,done,trunc,info

def process_obs(obs:dict):
    observation = obs.get("observation")
    achieved_goal = obs.get("achieved_goal")
    desired_goal = obs.get("desired_goal")
    return torch.from_numpy(np.append(observation,(achieved_goal,desired_goal))).to(dtype=torch.float32)

def make_env():
    x = gym.make("FetchPickAndPlace-v3",max_episode_steps=50,render_mode="human")
    x = custom(x)
    x = Autoreset(x)
    return x
 
class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(15,64)
        self.l2 = nn.Linear(64,64)
        self.mean = nn.Linear(64,4)
        self.std = nn.Linear(64,4)
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
policyy.load_state_dict(torch.load("./model-1600000.pth",map_location="cpu"))
env = make_env()
state = process_obs(env.reset()[0])
for n in range(1000):
    action = policyy(state)[0]
    st,re,done,trunc,info = env.step(action.detach().numpy()) # env.action_space.sample()
    env.render()