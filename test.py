import robosuite as suite
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers import Autoreset
import torch
from torch import Tensor
import torch.nn as nn

shared_net = nn.Sequential(
    nn.LazyLinear(512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU(),
    nn.Linear(512,512),
    nn.ReLU()
)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared_network = shared_net
        self.lmean = nn.Linear(512,9)
        self.lstd = nn.Linear(512,9)

    def forward(self,obs:Tensor):
        x = shared_net(obs)
        mean = self.lmean(x)
        eval_action = torch.tanh(mean)
        return eval_action

    def to(self,device="cpu"):
        self.to(device)


env_configs = {
    "robots":["Panda"],
    "gripper_types":["JacoThreeFingerDexterousGripper"],
    "has_renderer":True,
    "use_camera_obs":False,
    "has_offscreen_renderer":False,
    "horizon":50000, 
}

def make_env():
        x = suite.make(env_name ="PickPlace" ,**env_configs)
        x = GymWrapper(x,keys=list(x.observation_spec()))
        x.metadata = {"render_mode":[]}
        x = Autoreset(x)
        return x

actor = Actor()
env = make_env()
state,info = env.reset()
for n in range(10_000):
    state = torch.from_numpy(state).to(torch.float32) 
    action = actor(state) 
    state,reward,done,info,trunc = env.step(action.detach().tolist())
    env.render()