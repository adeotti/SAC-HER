import torch
import robosuite as suite
import numpy as np
import gymnasium as gym

from main import Actor,env_configs
from robosuite.wrappers.gym_wrapper import GymWrapper
from stable_baselines3.common.running_mean_std import RunningMeanStd

env = suite.make(
    env_name="Stack", 
    robots="Panda",  
    gripper_types="JacoThreeFingerDexterousGripper",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    horizon = 500,
    control_freq = 20
)
env = GymWrapper(env,list(env.observation_spec()))

def normalize(obs,obs_rms:RunningMeanStd): # norm obs
    mean = torch.from_numpy(obs_rms.mean)
    std = torch.from_numpy(obs_rms.var).sqrt() 
    output = (obs - mean) / (std + 1e-8)
    return output.clamp(-5,5).to(dtype=torch.float32)

obs = env.reset()[0]
policy = Actor()

checkpoint = torch.load("./34.pth",map_location="cpu") 
policy.load_state_dict(checkpoint["actor state"])

obs_rms = RunningMeanStd(shape=(162,))
obs_rms.mean = checkpoint[""]
obs_rms.var = checkpoint[""]
obs_rms.count = checkpoint[""]

with torch.no_grad():
    for i in range(100000):
        _,_,action = policy(normalize(torch.from_numpy(obs),obs_rms))
        obs,reward,done,info,trunc = env.step(action.numpy())
        env.render()
        if trunc or done:
            obs = env.reset()[0]

