import robosuite as suite
from robosuite import load_composite_controller_config
from robosuite.wrappers.gym_wrapper import GymWrapper
from gymnasium.vector import SyncVectorEnv,AutoresetMode
try:
    from gymnasium.wrappers import AutoResetWrapper
except ImportError:
    from gymnasium.wrappers import Autoreset

import torch,sys
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam

import numpy as np
import mlflow
from stable_baselines3.common.running_mean_std import RunningMeanStd
from copy import deepcopy
from tqdm import tqdm
from itertools import chain
from dataclasses import dataclass
import multiprocessing as mp


@dataclass(frozen=False)
class Hypers:
    ROBOT = "Panda"
    env_name = None
    device = torch.device("cuda:0")
    obs_dim = 162       # observation space, dim -1  
    action_dim = 9      # action space for a single env
    batchsize = 1024
    lr = 3e-4
    gamma = .99
    tau = .005
    warmup = 5000
    max_steps = int(5e6)
    num_envs = 10
    horizon = 500
    buffer_size = int(1e5)

hypers = Hypers()
    
cont_config = load_composite_controller_config(robot=hypers.ROBOT)
env_configs = {
    "robots":"Panda",
    "controller_configs": cont_config,
    "gripper_types":"JacoThreeFingerDexterousGripper",
    "has_renderer":False,
    "use_camera_obs":False,
    "has_offscreen_renderer":False,
    "reward_shaping":True,             # Dense rewards env version 
    "horizon":hypers.horizon,          # Max steps before reset or trunc = True
    "control_freq":20,
    "reward_scale":10.0
    }


def vec_env():
    def make_env():
        x = suite.make(env_name = "Stack",**env_configs)
        x = GymWrapper(x,list(x.observation_spec()))
        x.metadata = {"render_mode":[]}
        try:
            x = Autoreset(x)
        except NameError:
            x = AutoResetWrapper(x)
        return x
    return SyncVectorEnv([make_env for _ in range(hypers.num_envs)],
            autoreset_mode=AutoresetMode.SAME_STEP
    )


def weight_init(l):
    if isinstance(l,nn.Linear):
        nn.init.orthogonal_(l.weight)
        nn.init.constant_(l.bias,0.0)


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim,512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512,512)
        self.l_mean = nn.Linear(512,hypers.action_dim)
        self.l_std = nn.Linear(512,hypers.action_dim)
        self.apply(weight_init)
        self.optim = Adam(self.parameters(),hypers.lr)

    def forward(self,obs):
        x = F.silu(self.l1(obs))
        x = F.silu(self.l2(x))
        x = F.silu(self.l3(x))
        
        mean = self.l_mean(x)
        std = self.l_std(x).clamp(-2,2).exp()
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
        self.l1 = nn.Linear(hypers.obs_dim + hypers.action_dim,512)
        self.l2 = nn.Linear(512,512)
        self.l3 = nn.Linear(512,512)
        self.output = nn.Linear(512,1)

        self.ln1 = nn.LayerNorm(512)
        self.ln2 = nn.LayerNorm(512)
        self.ln3 = nn.LayerNorm(512)
        self.dropout = nn.Dropout(0.01)
        self.apply(weight_init) 

    def forward(self,obs,action): # TODO update forward
        cat = torch.cat((obs,action),dim=-1)
        x = F.silu(self.ln1(self.dropout(self.l1(cat))))
        x = F.silu(self.ln2(self.dropout(self.l2(x))))
        x = F.silu(self.ln3(self.dropout(self.l3(x))))
        x = self.output(x)
        return x


def normalize(obs,obs_rms:RunningMeanStd): # Welford's algorithm with no update
    running_mean = torch.from_numpy(obs_rms.mean)
    running_std = torch.from_numpy(obs_rms.var).sqrt()
    output = (obs - running_mean ) / (running_std + 1e-8)
    return output.clamp(-5,5) 


def create_storage():
    obs_dim = (hypers.num_envs,hypers.obs_dim)     
    act_dim = (hypers.num_envs,hypers.action_dim)
    return (
        torch.empty((hypers.horizon,*obs_dim),dtype=torch.half),
        torch.empty((hypers.horizon,*obs_dim),dtype=torch.half),
        torch.empty((hypers.horizon,hypers.num_envs,),dtype=torch.half),
        torch.empty((hypers.horizon,hypers.num_envs,),dtype=torch.bool),
        torch.empty((hypers.horizon,*act_dim),dtype=torch.half) 
    )


@torch.no_grad()
def step(queue,env,policy,obs_rms):
    stor_curr_states,stor_nx_states,stor_rewards,stor_terminated,stor_actions = create_storage()     
    pointer = 0
    reward__ = torch.zeros(hypers.num_envs)
    obs = torch.from_numpy(env.reset()[0])

    while True:
        obs_rms.update(obs.numpy() if torch.is_tensor(obs) else obs) # tracking values for running stats

        if pointer < hypers.warmup:
            action = env.action_space.sample()
        else:
            norm_obs = normalize(obs,obs_rms)
            action,_,_ = policy(obs)
            action = action.squeeze()
         
        nx_state,reward,done,terminated,info = env.step(action.tolist())
        
        reward__ += reward
        if np.all(done):
            last_obs = list(info.get("final_obs")) 
            buffer_nx_state = torch.from_numpy(np.stack(last_obs))
            reward__ *= 0
        else:
            buffer_nx_state = nx_state

        saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)
        
        stor_curr_states[pointer].copy_(torch.as_tensor(obs))
        stor_nx_states[pointer].copy_(torch.as_tensor(buffer_nx_state))
        stor_rewards[pointer].copy_(torch.from_numpy(reward))
        stor_terminated[pointer].copy_(torch.from_numpy(terminated))
        stor_actions[pointer].copy_(saved_action)

        obs = nx_state
        pointer+=1     
     
        if pointer == hypers.horizon:
            data = (stor_curr_states,stor_nx_states,stor_rewards,stor_terminated,stor_actions)
            queue.put(data)
            pointer = 0
            reward__.mean() # TODO log mean()
            stor_curr_states,stor_nx_states,stor_rewards,stor_terminated,stor_actions = create_storage()
            reward_ = torch.zeros(hypers.num_envs)


def sample(ep_queue,gpu_stream):
    while True:
        ep_curr_state,ep_nx_state,ep_rewards,ep_terminated,ep_actions = ep_queue.get()
        
        print(ep_curr_state.shape,ep_nx_state.shape,ep_rewards.shape,ep_terminated.shape)
        sys.exit()
    
        idx = torch.randint(0,self.pointer,(batch,))
        return (
            self.stor_curr_states[idx].float().flatten(0,1),
            self.stor_nx_states[idx].float().flatten(0,1),
            self.stor_rewards[idx].unsqueeze(-1).flatten(0,1),
            self.stor_terminated[idx].float().unsqueeze(-1).flatten(0,1),
            self.stor_actions[idx].float().flatten(0,1)
        )
   

class main:
    def __init__(self,storage_path):
        self.actor = Actor().to(hypers.device)
        self.q1 = Critic().to(hypers.device)
        self.q2 = Critic().to(hypers.device)

        self.q1_target = deepcopy(self.q1).to(hypers.device)
        self.q2_target = deepcopy(self.q2).to(hypers.device)

        self.actor.compile()
        self.q1.compile()
        self.q2.compile()

        self.critic_optim = Adam(chain(self.q1.parameters(),self.q2.parameters()),lr=hypers.lr,fused=True)

        self.entropy_target = -hypers.action_dim
        self.log_alpha = torch.tensor(1.0,requires_grad=True,device=hypers.device)  
        self.alpha_optim = Adam([self.log_alpha],lr=1e-6)
        
        self.storage_path = storage_path
        self.n = 0 # tracking number for model data saving      
            

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
            "log_alpha":self.log_alpha,

            "obs_rms_mean":self.buffer.obs_rms.mean,
            "obs_rms_var":self.buffer.obs_rms.var,
            "obs_rms_count":self.buffer.obs_rms.count
        }
        torch.save(check,f"{self.storage_path}{step}")


    def train(self,start=False):
        if start:
            actor_cpu = Actor()
            actor_cpu.share_memory()

            ep_queue = mp.Queue(maxsize=50) # epside queue
            for n in range(5):
                obs_rms = RunningMeanStd(shape=(hypers.obs_dim,))
                env = vec_env()
                step_thread = mp.Process(target=step,args=(ep_queue,env,actor_cpu,obs_rms,),daemon=True)
                step_thread.start()

            if ep_queue.full():
                batch_queue = mp.Queue(maxsize=10)
                batch_process = mp.Process(target=sample,args=(ep_queue,batch_queue,),daemon=True)
                batch_process.start()

                sys.exit()
                alpha = self.log_alpha.exp() 
                
                for traj in tqdm(range(hypers.max_steps + 1),total=hypers.max_steps + 1):
                    states,nx_states,reward,terminated,actions = self.buffer.sample(hypers.batchsize) 
                    states = self.normalize(states,self.buffer.obs_rms)
                    nx_states = self.normalize(nx_states,self.buffer.obs_rms)

                    with torch.no_grad():
                        nx_actions,log_nx_actions,_ = self.actor(nx_states)
                        min_q_target = torch.min(
                            self.q1_target(nx_states,nx_actions),self.q2_target(nx_states,nx_actions)
                        )
                        q_target = reward + hypers.gamma * (1-terminated) * (min_q_target - alpha * log_nx_actions)
                        # target = reward(st|at) + gamma * Q(st,at) - alpha * log policy(at|st))

                    q1_pred = self.q1(states,actions) 
                    q2_pred = self.q2(states,actions) 
                    critic_loss = F.mse_loss(q1_pred,q_target) 
                    critic_loss += F.mse_loss(q2_pred,q_target)

                    self.critic_optim.zero_grad(set_to_none=True)
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(chain(self.q1.parameters(),self.q2.parameters()),1.0)
                    self.critic_optim.step()

                    for q1_pars,q1_target_pars in zip(self.q1.parameters(),self.q1_target.parameters()):
                        q1_target_pars.data.mul_(1.0 - hypers.tau).add_(q1_pars.data,alpha=hypers.tau)
                
                    for q2_pars,q2_target_pars in zip(self.q2.parameters(),self.q2_target.parameters()):
                        q2_target_pars.data.mul_(1.0 - hypers.tau).add_(q2_pars.data,alpha=hypers.tau)
                    
                    for p in self.q1.parameters() : p.requires_grad = False
                    for p in self.q2.parameters() : p.requires_grad = False

                    new_action,log_pi,_ = self.actor(states)
                    min_q = torch.min(self.q1(states,new_action),self.q2(states,new_action))
                    policy_loss = ((alpha.detach() * log_pi) -  min_q).mean() # alpla * log policy(at|st) - Q(st,at)
                    
                    self.actor.optim.zero_grad(set_to_none=True)
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(),1.0)
                    self.actor.optim.step()

                    for p in self.q1.parameters(): p.requires_grad = True
                    for p in self.q2.parameters(): p.requires_grad = True

                    # Entropy auto tune
                    alpha_loss = -(self.log_alpha*(log_pi+self.entropy_target).detach()).mean()
                    self.alpha_optim.zero_grad(set_to_none=True)
                    alpha_loss.backward() 
                    self.alpha_optim.step()
                    alpha = self.log_alpha.exp()

                    if traj > 0 and traj % int(5e3) == 0 :
                        self.n+=1
                        self.save(self.n)

                    if traj > 0 and traj % int(1e3) == 0 :
                        coll_obs_mean,coll_obs_std,coll_reward = self.buffer.utils()

                        mlflow.log_metrics(
                            {
                                "Main/collection rewards" : coll_reward,
                                "Main/episodes rewards" : self.buffer.epi_reward.mean().item(),

                                "Main/entropy loss" : alpha_loss.item(),
                                "Main/alpha value" : alpha.item(),

                                "Norm/collection obs mean" : coll_obs_mean.item(),
                                "Norm/collection obs std" : coll_obs_std.item(), 
                                "Norm/training state mean" : states.mean().item(),
                                "Norm/training state std" : states.std().item(),
                                "Norm/training nx state mean" : nx_states.mean().item(),
                                "Norm/training nx state std" : nx_states.std().item(),

                                "policy/log action" : (alpha * log_pi).mean().item(),
                                "policy/pred min Q target" : min_q.mean().item(),
                                "policy/policy loss action variance" : new_action.var().item(),
                                "policy/loss Policy" : policy_loss.item(),
                                "policy/action variance" : actions.var().item(),

                                "critic/log action" : (alpha * log_nx_actions).mean().item(),
                                "critic/pred min Q target" : min_q_target.mean().item(),
                                "critic/critic Loss" : critic_loss.item()
                            },
                            step = traj
                        )
                        

if __name__ == "__main__":
    import warnings,logging
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    main(storage_path="./").train(True)
    
