import gymnasium as gym
import gymnasium_robotics,torch,sys
gym.register_envs(gymnasium_robotics)
from gymnasium.vector import SyncVectorEnv 
import numpy as np
from dataclasses import dataclass

from torch import Tensor
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import torch.nn.functional as F
from torch import linalg as LA

from collections import deque
import random
from tqdm import tqdm
import threading,queue,itertools,copy
from torch.utils.tensorboard import SummaryWriter


class custom(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
      
    def reset(self,**kwargs):
        obs,info = super().reset(**kwargs)
        self.env.unwrapped.data.qpos[0] = .3  # robot base x pos
        self.env.unwrapped.data.qpos[1] = .5  # robot base y pos
        self.env.unwrapped.data.qpos[17] = .4 # block's z pos
        return obs,info

    def step(self,action):
        state,reward,done,trunc,info = super().step(action)
        return state,reward,done,trunc,info

def vec_env():
    def make_env():
        x = gym.make("FetchPickAndPlace-v3",max_episode_steps=100)
        x = custom(x)
        return x
    return SyncVectorEnv(
        [make_env for _ in range(hypers.num_envs)],autoreset_mode=gym.vector.AutoresetMode.DISABLED
    )

def test_env(render_mode="human"):
    x = (gym.make("FetchPickAndPlace-v3",max_episode_steps=100) if render_mode is None 
        else gym.make("FetchPickAndPlace-v3",max_episode_steps=100,render_mode = render_mode)
    )
    x = custom(x)
    return x

@dataclass()
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = int(1e7)+1 
    lr = 3e-4
    action_dim = 4
    obs_dim = 31
    warmup = int(1e5)
    gamma = 0.99
    tau = 5e-3
    batch_size = 256
    num_envs = 8
    horizon = 50

hypers = Hypers()


class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,256)
        self.mean = nn.Linear(256,hypers.action_dim)
        self.std = nn.Linear(256,hypers.action_dim)
        self.apply(weight_init)
        self.optim = Adam(self.parameters(),lr=hypers.lr)

    def forward(self,obs:Tensor):
        x = F.relu(self.l1(obs))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        mean = self.mean(x)
        std = self.std(x).clamp(-20,2).exp()
        dist = Normal(mean,std)
        pretanh = dist.rsample()
        action = F.tanh(pretanh)
        log = dist.log_prob(pretanh)
        log -= torch.log(1-action.pow(2) + 1e-6) 
        log = log.sum(-1,True)
        return action,log,mean
        
class q_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim+hypers.action_dim,256)
        self.l2 = nn.Linear(256,256)
        self.l3 = nn.Linear(256,1)
        self.apply(weight_init)
    
    def forward(self,obs:Tensor,action:Tensor):
        x = torch.cat((obs,action),dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x 


def process_obs(obs:dict): 
    observation = obs["observation"]     # (n env,25)
    achieved_goal = obs["achieved_goal"] # (n env,3)
    desired_goal = obs["desired_goal"]   # (n env,3)
    output = torch.from_numpy(np.concatenate([observation,achieved_goal,desired_goal],axis=-1))
    assert output.shape == torch.Size([hypers.num_envs,hypers.obs_dim]) or output.shape == torch.Size([hypers.obs_dim])
    return output.to(device=hypers.device,dtype=torch.float32) 

def process_her_states(observation,achieved_goal,desired_goal):
    output = torch.from_numpy(np.concatenate([observation,achieved_goal,desired_goal],axis=-1))
    assert output.shape == torch.Size([hypers.num_envs,hypers.obs_dim]) 
    return output.to(device=hypers.device,dtype=torch.float32) 

def her_reward(goal_a,goal_b):
    goal_a = torch.from_numpy(goal_a)
    goal_b = torch.from_numpy(goal_b)
    distance_threshold = 0.05
    output = LA.norm(goal_a - goal_b,dim=-1)
    return -(output > distance_threshold).to(device=hypers.device,dtype=torch.float32)

class buffer: 
    def _init_storage(self,path=None,capacity=hypers.max_steps): 
        if path is not None:
            self.data = torch.load(path,map_location=hypers.device,weights_only=False)
            self.curr_state = self.data["curr_states"]
            self.nx_states = self.data["nx_states"]
            self.stor_rewards = self.data["rewards"]
            self.stor_truncs = self.data["truncs"]
            self.stor_actions = self.data["actions"]
            self.pointer = self.data["pointer"]
        else:
            act_dim = (hypers.num_envs,hypers.action_dim) # action shape
            self.curr_state = [] # current states storage
            self.nx_states = []  # next states storage
            self.stor_rewards = torch.empty((capacity,hypers.num_envs,),dtype=torch.float16,device=hypers.device) 
            self.stor_truncs = torch.empty((capacity,hypers.num_envs,),dtype=torch.bool,device=hypers.device)
            self.stor_actions = torch.empty((capacity,*act_dim),dtype=torch.float16,device=hypers.device)
            self.pointer = 0

    def __init__(self,writter,env,policy):
        self.writter = writter
        self._init_storage(path=None)
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()[0]
        self.epi_reward = deque(maxlen=hypers.num_envs)
        self.reward = torch.empty(hypers.num_envs,dtype=torch.float16)
        self.her_storage = deque(maxlen=hypers.horizon)

    def store(self,reward,trunc,action):  
        self.stor_rewards[self.pointer] = reward
        self.stor_truncs[self.pointer] = trunc
        self.stor_actions[self.pointer]= action

    @torch.no_grad()
    def add(self):
        if len(self)<hypers.warmup:
            action = self.env.action_space.sample()
        else:
            action,_,_ = self.policy(process_obs(self.obs))
            action = action.squeeze()
        
        nx_state,reward,done,trunc,info = self.env.step(action.tolist())
        self.writter.add_scalar(
            "Main/Truncsss",torch.from_numpy(trunc).to(torch.float).mean(),self.pointer,new_style=True
        )
    
        for i in range(hypers.num_envs): # manual reset done environment and episodic reward tracking
            self.reward[i]+=reward[i]
            if trunc[i]:
                reset = self.env.envs[i].reset()[0]
                self.obs["achieved_goal"][i] = reset["achieved_goal"]
                self.obs["desired_goal"][i] = reset["desired_goal"]
                self.obs["observation"][i] = reset["observation"]
                self.env._autoreset_envs[i] = np.False_  
                self.epi_reward.append(self.reward[i])
                self.reward[i] = 0
 
        saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)

        self.curr_state.append(self.obs)
        self.nx_states.append(nx_state)

        self.store(
            torch.from_numpy(reward).to(device=hypers.device),
            torch.from_numpy(trunc).to(device=hypers.device),
            saved_action.to(device=hypers.device)
        )
        self.obs = nx_state
        self.pointer+=1
    
    def save(self):
        data = {
            "curr_states":self.curr_state,
            "nx_states":self.nx_states,
            "rewards":self.stor_rewards.half(),
            "truncs":self.stor_truncs.bool(),
            "actions":self.stor_actions.half(),
            "pointer":self.pointer
        }
        torch.save(data,"./data.pth")
    
    def util(self): 
        return torch.as_tensor([self.epi_reward]).mean()
    
    def __len__(self):
        return len(self.curr_state)

def her_sample(batch_size,k, curr_states,nx_states,rewards,truncs,actions,writter): # target ratio 4:1, strategy : future
    num_episodes = len(curr_states)//hypers.horizon
    all_curr,all_nx,all_rewards,all_truncs,all_actions = [],[],[],[],[]

    for _ in range(hypers.horizon):
        epi_idx = np.random.randint(num_episodes)
        epi_start_idx = epi_idx*hypers.horizon
        batch = curr_states[epi_start_idx:epi_start_idx+hypers.horizon]
        nx_batch = nx_states[epi_start_idx:epi_start_idx+hypers.horizon]
        idx = random.randint(0,len(batch)-2)

        all_curr.append(process_obs(batch[idx]))
        all_nx.append(process_obs(nx_batch[idx]))
        all_rewards.append(rewards[epi_start_idx+idx])
        all_truncs.append(truncs[epi_start_idx+idx])
        all_actions.append(actions[epi_start_idx+idx])

        for _ in range(k):
            future_idx = random.randint(idx+1,len(batch)-1)
            curr = batch[idx] # t
            nx = nx_batch[idx] # t+1
            future = batch[future_idx] # t'
            h_rewards = her_reward(nx["achieved_goal"],future["achieved_goal"])
            writter.add_scalar("Main/HER reward",h_rewards.mean(),new_style=True)
            curr_her_transition = process_her_states(curr["observation"],curr["achieved_goal"],future["achieved_goal"])
            nx_her_transition = process_her_states(nx["observation"],nx["achieved_goal"],future["achieved_goal"])
           
            all_curr.append(curr_her_transition)
            all_nx.append(nx_her_transition)
            all_rewards.append(h_rewards)
            all_truncs.append(truncs[epi_start_idx+idx])  
            all_actions.append(actions[epi_start_idx+idx])  
        
    assert (len(all_curr)==len(all_nx)==len(all_rewards)==len(all_truncs)==len(all_actions)==(hypers.horizon*k)+hypers.horizon) 

    s_c = torch.stack(all_curr) 
    s_nx = torch.stack(all_nx)
    s_r = torch.stack(all_rewards)
    s_tr = torch.stack(all_truncs)
    s_a = torch.stack(all_actions)

    sample_idx = torch.randperm(s_c.size(0))[:batch_size]
    
    return (
        s_c[sample_idx].float(),
        s_nx[sample_idx].float(),
        s_r[sample_idx].unsqueeze(-1).float(),
        s_tr[sample_idx].unsqueeze(-1).float(),
        s_a[sample_idx].float(),
    )

def her_worker(queue,buffer:buffer,writter): 
    while True:
        if len(buffer)>hypers.horizon:
            states,nx_state,reward,trunc,action = her_sample(
                hypers.batch_size,
                4, # ration of HER transitions
                buffer.curr_state,
                buffer.nx_states,
                buffer.stor_rewards,
                buffer.stor_truncs,
                buffer.stor_actions,
                writter
            )
            queue.put((states,nx_state,reward,trunc,action))

class main:
    def __init__(self):
        self.policy = policy().to(device=hypers.device)
        self.q1 = q_network().to(device=hypers.device)
        self.q2 = q_network().to(device=hypers.device)
        self.q1_target = copy.deepcopy(self.q1).to(device=hypers.device)
        self.q2_target = copy.deepcopy(self.q2).to(device=hypers.device)
        self.q_optim = Adam(itertools.chain(self.q1.parameters(),self.q2.parameters()),lr=hypers.lr)
        self.writter = SummaryWriter("./")

        self.env = vec_env()
        self.buffer = buffer(self.writter,self.env,self.policy)

        self.queue = queue.Queue(maxsize=hypers.warmup//hypers.horizon)
        self.thread = threading.Thread(target=her_worker,args=(self.queue,self.buffer,self.writter),daemon=True)
        self.thread.start()

        self.entropy_target = -hypers.action_dim
        self.log_alpha = torch.tensor([0.0],requires_grad=True,device=hypers.device)
        self.alpha_optim = Adam([self.log_alpha],lr=1e-5)

        self.test_env = test_env(render_mode=None)
        self.test_step = 0
    
    def save(self,step):
        check_point = {
            "policy_state":self.policy.state_dict(),
            "policy_optim":self.policy.optim.state_dict(),
            "q1_state":self.q1.state_dict(),
            "q2_state":self.q2.state_dict(),
            "q1_target":self.q1_target.state_dict(),
            "q2_target":self.q2_target.state_dict(),
            "q_optim":self.q_optim.state_dict(),
            "alpha_optim":self.alpha_optim.state_dict(),
            "log_alpha":self.log_alpha
        }
        torch.save(check_point,f"./model-{step}.pth")
    
    def load(self,path=None,strict=True):
        if path is not None:
            check_point = torch.load(path,map_location=hypers.device)
            self.policy.load_state_dict(check_point["policy_state"],strict)
            self.policy.optim.load_state_dict(check_point["policy_optim"])
            self.q1.load_state_dict(check_point["q1_state"],strict)
            self.q2.load_state_dict(check_point["q2_state"],strict)
            self.q1_target.load_state_dict(check_point["q1_target"],strict)
            self.q2_target.load_state_dict(check_point["q2_target"],strict)
            self.log_alpha.data.copy_(check_point["log_alpha"].data) 
            self.alpha_optim.load_state_dict(check_point["alpha_optim"])

    def train(self,start=False):
        if start:
            self.load()
            t = 0
            alpha = self.log_alpha.exp()
            for n in tqdm(range(hypers.max_steps-1),total=hypers.max_steps-1):
                self.buffer.add()

                if (n+1) % hypers.horizon == 0:
                    self.writter.add_scalar("Main/epi reward",self.buffer.util(),n,new_style=True)

                if len(self.buffer) >= hypers.warmup:
                    states,nx_state,reward,trunc,action = self.queue.get()
                  
                    with torch.no_grad():
                        target_action,log_target_action,_ = self.policy(nx_state)
                        q1_target = self.q1_target(nx_state,target_action)
                        q2_target = self.q2_target(nx_state,target_action)
                        q_target = reward + (1-trunc) * hypers.gamma * (torch.min(q1_target,q2_target) - alpha * log_target_action)
                    q1 = self.q1(states,action) 
                    q2 = self.q2(states,action)
                    q_loss = F.mse_loss(q1,q_target) + F.mse_loss(q2,q_target)
                    self.q_optim.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(itertools.chain(self.q1.parameters(),self.q2.parameters()),1.0)
                    self.q_optim.step()
                    
                    p_action,log_p_action,_ = self.policy(states)
                    min_q = torch.min(self.q1(states,p_action),self.q2(states,p_action))
                    policy_loss = ((alpha * log_p_action) - min_q).mean()
                    self.policy.optim.zero_grad()
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(),1.0)
                    self.policy.optim.step()

                    alpha_loss = -(self.log_alpha*(log_p_action + self.entropy_target).detach()).mean()
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()
                    alpha = self.log_alpha.exp()

                    for q1_params,q1_target_parms in zip(self.q1.parameters(),self.q1_target.parameters()):
                        q1_target_parms.data.mul_(1.0-hypers.tau).add_(q1_params.data,alpha=hypers.tau)
                    for q2_params,q2_target_params in zip(self.q2.parameters(),self.q2_target.parameters()):
                        q2_target_params.data.mul_(1.0-hypers.tau).add_(q2_params.data,alpha=hypers.tau)
                    
                    self.writter.add_scalar("Main/alpha exp",alpha,n,new_style=True)
                    self.writter.add_scalar("Main/alpha loss",alpha_loss,n,new_style=True)
                    self.writter.add_scalar("Main/action variance",action.var(),n,new_style=True)
                    self.writter.add_scalar("Main/policy loss action variance",p_action.var(),n,new_style=True)
                    self.writter.add_scalar("Main/policy loss",policy_loss,n,new_style=True)
                    self.writter.add_scalar("Main/critic loss",q_loss,n,new_style=True)
                    self.writter.flush()
            
                    if (n+1) % int(10e3) == 0:
                        t+=1
                        self.save(t)
                        self.buffer.save()
                     
            self.save("final")

main().train(True)
