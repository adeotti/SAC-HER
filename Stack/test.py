import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers import GymWrapper
from gymnasium.wrappers import Autoreset
import torch
from torch.distributions import Normal
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import warnings,logging,sys
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(74,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.lmean = nn.Linear(256,9)
        self.lstd = nn.Linear(256,9)
        #self.optim = torch.optim.Adam(self.parameters(),hypers.lr)

    def forward(self,obs:Tensor):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.lmean(x)
        std = self.lstd(x).clamp(-20,2).exp()
        dist = Normal(mean,std) 
        pre_tanh = dist.rsample()
        action = F.tanh(pre_tanh)

        log = dist.log_prob(pre_tanh).sum(-1,True)
        log -= torch.log(1-action.pow(2) + 1e-9).sum(-1,True)  

        eval_action = F.tanh(mean)
        return action,log,eval_action

    def to(self,device="cpu"):
        self.to(device)

cont_config = load_composite_controller_config(robot="Panda")
env_configs = {
    "robots":["Panda"],
    "controller_configs": cont_config,
    "gripper_types":["JacoThreeFingerDexterousGripper"],
    "has_renderer":True,
    "use_camera_obs":False,
    "has_offscreen_renderer":False,
    "reward_shaping":True,               
    "horizon":500,                        
    "control_freq":20,
    "reward_scale":2.0
}
def make_env():
    x = suite.make(env_name ="Stack" ,**env_configs)
    x = GymWrapper(x,keys=list(x.active_observables))
    x.metadata = {"render_mode":[]}
    x = Autoreset(x)
    return x

actor = Actor()
chk = torch.load("./model_60.pth",map_location="cpu")
actor.load_state_dict(chk["actor state"],strict=True)
env = make_env()

state,info = env.reset()
for n in range(10_000):
    st = torch.from_numpy(state).to(torch.float32) 
    _,_,action = actor(st) 
    state,reward,done,info,trunc = env.step(action.detach().tolist())
    if done:
        state,info = env.reset()
    env.render()