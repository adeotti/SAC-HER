import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv
try:
    from gymnasium.wrappers import AutoResetWrapper
except ImportError:
    from gymnasium.wrappers import Autoreset
import torch,sys
from dataclasses import dataclass
from IPython.display import clear_output

from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from copy import deepcopy
from tqdm import tqdm
from itertools import chain
from torch.utils.tensorboard import SummaryWriter

import warnings,logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
clear_output()

@dataclass(frozen=False)
class Hypers:
    ROBOT = "Panda"
    env_name = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_dim = 149       # observation space, dim -1  
    action_dim = 9    # action space for a single env
    batchsize = 256
    lr = 3e-4
    gamma = .99
    tau = .005
    warmup = int(5e4)
    max_steps = int(5e6)
    num_envs = 8
    horizon = 500
    gripper = []
    
    
cont_config = load_composite_controller_config(robot=hypers.ROBOT)
env_configs = {
    "robots":[hypers.ROBOT],
    "controller_configs": cont_config,
    "gripper_types":hypers.gripper,
    "has_renderer":False,
    "use_camera_obs":False,
    "has_offscreen_renderer":False,
    "reward_shaping":True,             # Dense rewards env version 
    "horizon":hypers.horizon,          # Max steps before reset or trunc = True
    "control_freq":20,
    "reward_scale":1.0

def vec_env():
    def make_env():
        x = suite.make(
            env_name = "Stack"
            gripper_types = "JacoThreeFingerDexterousGripper",
            horizon = hypers.horizon,
            **env_configs
        )
        x = GymWrapper(x,list(x.observation_spec()))
        x.metadata = {"render_mode":[]}
        try:
            x = Autoreset(x)
        except NameError:
            x = AutoResetWrapper(x)
        return x
    return SyncVectorEnv([make_env for _ in range(hypers.num_envs)])



def weight_init(l):
    if isinstance(l,nn.Linear):
        torch.nn.init.orthogonal_(l.weight)
        torch.nn.init.constant_(l.bias,0.0)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.lmean = nn.Linear(256,hypers.action_dim)
        self.lstd = nn.Linear(256,hypers.action_dim)
        self.apply(weight_init)
        self.optim = Adam(self.parameters(),hypers.lr)

    def forward(self,obs:Tensor):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.lmean(x)
        std = self.lstd(x).clamp(-20,2).exp()
        dist = Normal(mean,std) 
        pre_tanh = dist.rsample()
        action = F.tanh(pre_tanh)
        log = dist.log_prob(pre_tanh)
        log -= torch.log(1-action.pow(2) + 1e-8) # change of variable correction 
        log = log.sum(-1,True)  
        return action,log,mean
    

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim + hypers.action_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.output = nn.Linear(256,1)
        self.apply(weight_init)

    def forward(self,obs:Tensor,action:Tensor):
        cat = torch.cat((obs,action),dim=-1)
        x = F.relu(self.l1(cat))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.output(x)
        return x

class buffer: 
    def _init_storage(self,data_path=None,capacity=hypers.max_steps):
        obs_dim = (hypers.num_envs,hypers.obs_dim)     
        act_dim = (hypers.num_envs,hypers.action_dim) 
        if data_path is not None:
            self.data = torch.load(data_path,weights_only=False)
            self.stor_curr_states = self.data["curr_states"]  
            self.stor_nx_states = self.data["nx_states"]  
            self.stor_rewards = self.data["rewards"]  
            self.stor_dones = self.data["dones"]  
            self.stor_actions = self.data["actions"]  
            self.pointer = int(self.data["pointer"])
        else:
            self.stor_curr_states = torch.empty((capacity,*obs_dim),dtype=torch.float32)
            self.stor_nx_states = torch.empty((capacity,*obs_dim),dtype=torch.float32)
            self.stor_rewards = torch.empty((capacity,hypers.num_envs,),dtype=torch.float32)
            self.stor_dones = torch.empty((capacity,hypers.num_envs,),dtype=torch.bool)
            self.stor_actions = torch.empty((capacity,*act_dim),dtype=torch.float32)
            self.pointer = 0
    
    def __init__(self,env,policy):
        self._init_storage(data_path=None)
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()[0]
        self.epi_reward = torch.empty(hypers.num_envs)
        self.reward = torch.empty(hypers.num_envs)
        self.to_tensor = lambda x : torch.from_numpy(np.array(x)).to(hypers.device,dtype=torch.float32)
        self.steps = 0
        self.obs_rms = RunningMeanStd(shape=(hypers.obs_dim,))
        self.norm_obs = None
        self.log_reward = None
    
    def store(self,curr_state,nx_state,reward,done,action):
        self.stor_curr_states[self.pointer] = curr_state
        self.stor_nx_states[self.pointer] = nx_state
        self.stor_rewards[self.pointer] = reward
        self.stor_dones[self.pointer] = done
        self.stor_actions[self.pointer] = action

    def normalize(self,obs,obs_rms:RunningMeanStd): # Welford's algorithm
        running_mean = torch.from_numpy(obs_rms.mean).to(hypers.device)
        running_std = torch.from_numpy(obs_rms.var).sqrt().to(hypers.device)
        output = (torch.from_numpy(obs).to(hypers.device) - running_mean ) / (running_std + 1e-8)
        return output.clamp(-5,5).to(device=hypers.device,dtype=torch.float32) 

    @torch.no_grad()
    def step(self):
        self.steps+=1
        self.obs_rms.update(self.obs) # tracking values for running stats
        if self.pointer<hypers.warmup:
            action = self.env.action_space.sample()
        else:
            self.norm_obs = self.normalize(self.obs,self.obs_rms)
            action,_,_ = self.policy(self.norm_obs)
            action = action.squeeze()
          
        nx_state,reward,done,_,_ = self.env.step(action.tolist())
        
        for n in range(hypers.num_envs):
            self.reward[n] += reward[n]
            if done[n]:
                self.epi_reward[n] = self.reward[n]
                self.reward[n] = 0

        saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)

        self.store(
            self.to_tensor(self.obs),
            self.to_tensor(nx_state),
            self.to_tensor(reward),
            self.to_tensor(done),
            saved_action
        )
        self.obs = nx_state
        self.pointer+=1  
        self.log_reward = reward
  
    def sample(self,batch):
        idx = torch.randint(0,self.pointer,(batch,))
        return (
            self.stor_curr_states[idx].float().flatten(0,1).to(device=hypers.device),
            self.stor_nx_states[idx].float().flatten(0,1).to(device=hypers.device),
            self.stor_rewards[idx].unsqueeze(-1).flatten(0,1).to(device=hypers.device),
            self.stor_dones[idx].float().unsqueeze(-1).flatten(0,1).to(device=hypers.device),
            self.stor_actions[idx].float().flatten(0,1).to(device=hypers.device)
        )
       
    def save(self):
        data = {
            "curr_states":self.stor_curr_states.half(),
            "nx_states":self.stor_nx_states.half(),
            "rewards":self.stor_rewards.half(),
            "dones":self.stor_dones.bool(),
            "actions":self.stor_actions.half(),
            "pointer":self.pointer
        }
        torch.save(data,"./data.pth") 
    
    def utils(self):
        return self.norm_obs.mean(),self.norm_obs.std(),self.log_reward.mean()



class main:
    def __init__(self):
        self.actor = Actor().to(hypers.device)

        self.q1 = Critic().to(hypers.device)
        self.q2 = Critic().to(hypers.device)

        self.q1_target = deepcopy(self.q1).to(hypers.device)
        self.q2_target = deepcopy(self.q2).to(hypers.device)

        self.actor.compile()
        self.q1.compile()
        self.q2.compile()

        self.critic_optim = Adam(chain(self.q1.parameters(),self.q2.parameters()),lr=hypers.lr)

        self.entropy_target = -hypers.action_dim
        self.log_alpha = torch.tensor(0.0,requires_grad=True,device=hypers.device)  
        self.alpha_optim = Adam([self.log_alpha],lr=hypers.lr)
        
        self.env = vec_env()
        self.buffer = buffer(self.env,self.actor)
        self.writter = SummaryWriter("./")
    
    def save(self,step):
        check = {
            "actor state":self.actor.state_dict(),
            "actor optim" : self.actor.optim.state_dict(),
            "q1 state":self.q1.state_dict(),
            "q1 target":self.q1_target.state_dict(),
            "q2 state":self.q2.state_dict(),
            "q2 target":self.q2_target.state_dict(),
            "critic optim":self.critic_optim.state_dict(),
            "alpha optim":self.alpha_optim.state_dict(),
            "log_alpha":self.log_alpha
        }
        torch.save(check,f"./{step}.pth")
    
    def load(self,model_path = None,strict=True):
        if model_path is not None:
            check = torch.load(model_path,map_location=hypers.device)
            self.actor.load_state_dict(check["actor state"],strict)
            self.actor.optim.load_state_dict(check["actor optim"])
            self.q1.load_state_dict(check["q1 state"],strict)
            self.q1_target.load_state_dict(check["q1 target"],strict)
            self.q2.load_state_dict(check["q2 state"],strict)
            self.q2_target.load_state_dict(check["q2 target"],strict)
            self.critic_optim.load_state_dict(check["critic optim"])
            self.log_alpha.data.copy_(check["log_alpha"].data)
            self.alpha_optim.load_state_dict(check["alpha optim"])
    
    def normalize(self,obs,obs_rms:RunningMeanStd): # Welford's algorithm with no update
        running_mean = torch.from_numpy(obs_rms.mean).to(hypers.device)
        running_std = torch.from_numpy(obs_rms.var).sqrt().to(hypers.device)
        output = (obs - running_mean ) / (running_std + 1e-8)
        return output.clamp(-5,5).to(device=hypers.device,dtype=torch.float32) 
        
    def train(self,start=False):
        if start:
            self.load() 
            n = 0 
            #alpha = self.log_alpha.exp() for entropy autotune 
            alpha = torch.tensor([0.2],dtype=torch.float32,device=hypers.device)
            
            for traj in tqdm(range(hypers.max_steps-1),total=hypers.max_steps-1):
                if not self.buffer.pointer == hypers.max_steps:
                    self.buffer.step()

                if self.buffer.pointer > hypers.warmup:
                    states,nx_states,reward,dones,actions = self.buffer.sample(hypers.batchsize) 
                    states = self.normalize(states,self.buffer.obs_rms)
                    nx_states = self.normalize(nx_states,self.buffer.obs_rms)
        
                    with torch.no_grad():
                        nx_actions,log_nx_actions,_ = self.actor(nx_states)
                        min_q_target = torch.min(
                                self.q1_target(nx_states,nx_actions),self.q2_target(nx_states,nx_actions)
                        ) 
                        q_target = reward + hypers.gamma * (1-dones) * (min_q_target - alpha * log_nx_actions)
                        # reward(st|at) + gamma * Q(st,at) - alpha*log policy(at|st))

                    critic_loss = F.mse_loss(self.q1(states,actions),q_target) 
                    critic_loss += F.mse_loss(self.q2(states,actions),q_target)

                    self.critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(chain(self.q1.parameters(),self.q2.parameters()),1.0)
                    self.critic_optim.step()
                    
                    new_action,log_pi,_ = self.actor(states)
                    with torch.no_grad():
                        min_q = torch.min(self.q1(states,new_action),self.q2(states,new_action))
                    policy_loss = ((alpha * log_pi) -  min_q).mean() # alpla * log policy(at|st) - Q(st,at)
                    self.actor.optim.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)
                    self.actor.optim.step()

                    """ # Entropy auto tune
                    alpha_loss = -(self.log_alpha*(log_pi+self.entropy_target).detach()).mean()
                    self.alpha_optim.zero_grad(set_to_none=True)
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    alpha = self.log_alpha.exp()
                    self.writter.add_scalar("Main/entropy loss",alpha_loss,traj)
                    """
                   
                    for q1_pars,q1_target_pars in zip(self.q1.parameters(),self.q1_target.parameters()):
                        q1_target_pars.data.mul_(1.0 - hypers.tau).add_(q1_pars.data,alpha=hypers.tau)
                    
                    for q2_pars,q2_target_pars in zip(self.q2.parameters(),self.q2_target.parameters()):
                        q2_target_pars.data.mul_(1.0 - hypers.tau).add_(q2_pars.data,alpha=hypers.tau)
                        
                    if traj != 0 and traj%int(5e3) == 0 :
                        n+=1
                        self.save(n)
                        self.buffer.save() 
                        
                    if self.buffer.pointer == hypers.max_steps:
                        self.buffer.save()
                    
                    coll_obs_mean,coll_obs_std,coll_reward = self.buffer.utils()
                    self.writter.add_scalar("Norm/Collection obs mean",coll_obs_mean,traj)
                    self.writter.add_scalar("Norm/Collection obs std",coll_obs_std,traj)
                    self.writter.add_scalar("Main/Collection rewards",coll_reward,traj)
                    self.writter.add_scalar("Main/episodes rewards",self.buffer.epi_reward.mean(),traj)
                    self.writter.add_scalar("Norm/training state mean",states.mean(),traj)
                    self.writter.add_scalar("Norm/training state std",states.std(),traj)
                    self.writter.add_scalar("Norm/training nx state mean",nx_states.mean(),traj)
                    self.writter.add_scalar("Norm/training nx state std",nx_states.std(),traj)
                    
                    self.writter.add_scalar("Main/loss Policy",policy_loss,traj)
                    self.writter.add_scalar("Main/alpha value",alpha,traj)
                    self.writter.add_scalar("Main/critic Loss",critic_loss,traj)
                    self.writter.add_scalar("Main/action variance",actions.var(),traj)
                    self.writter.add_scalar("Main/policy loss action variance",new_action.var(),traj)
                     
if __name__ == "__main__":
    #main().train(False)
    #Test().run(True)
