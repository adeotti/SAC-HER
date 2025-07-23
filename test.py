import robosuite as suite
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers import Autoreset
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import warnings,logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(214,512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512,512)
        self.lmean = nn.Linear(512,9)

    def forward(self,obs:Tensor):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        eval_action = F.tanh(self.lmean(x))
        return eval_action

    def to(self,device="cpu"):
        self.to(device)

env_configs = {
    "robots":["Panda"],
    "gripper_types":["JacoThreeFingerDexterousGripper"],
    "has_renderer":True,
    "use_camera_obs":False,
    "has_offscreen_renderer":False,
    "reward_shaping":True,
    "horizon":500, 
}

def make_env():
    x = suite.make(env_name ="PickPlace" ,**env_configs)
    x = GymWrapper(x,keys=list(x.observation_spec()))
    x.metadata = {"render_mode":[]}
    return x

actor = Actor()
env = make_env()
state,info = env.reset()
for n in range(10_000):
    st = torch.from_numpy(state).to(torch.float32) 
    action = actor(st) 
    state,reward,done,info,trunc = env.step(action.detach().tolist())
    if done:
        state,info = env.reset()
     
    env.render()