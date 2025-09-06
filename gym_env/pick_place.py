import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import Autoreset
from gymnasium.spaces import Box,Dict
import numpy as np
from dataclasses import dataclass
import torch,random,sys,threading,queue,itertools,copy
 
import torch.nn as nn
from torch.distributions import Normal
from torch.optim import Adam
import torch.nn.functional as F
from torch import linalg as LA

from collections import deque
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        target = random.choice([True,False])
        self.env.unwrapped.unwrapped.target_in_the_air = target
        obs["observation"] = obs["observation"][:9]
        self.env.unwrapped.data.qpos[0] = .3  # robot base x pos
        self.env.unwrapped.data.qpos[1] = .5  # robot base y pos
        # self.env.unwrapped.data.qpos[15]    # block's x pos
        # self.env.unwrapped.data.qpos[16]    # block's y pos
        self.env.unwrapped.data.qpos[17] = .4
        return obs,info

    def step(self,action):
        state,reward,done,trunc,info = super().step(action)
        state["observation"] = state["observation"][:9]
        return state,reward,done,trunc,info

def vec_env():
    def make_env():
        x = gym.make("FetchPickAndPlace-v3",max_episode_steps=50)
        x = custom(x)
        x = Autoreset(x)
        return x
    return AsyncVectorEnv([make_env for _ in range(hypers.num_envs)])

def test_env():
    x = gym.make("FetchPickAndPlace-v3",max_episode_steps=50)
    x = custom(x)
    return x

@dataclass()
class Hypers:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_steps = int(2e6)+1 
    lr = 3e-4
    action_dim = 4
    obs_dim = 15
    warmup = 2_000  
    gamma = 0.99
    tau = 5e-3
    batch_size = 256
    num_envs = 5
    horizon = 50

hypers = Hypers()

def weight_init(l):
    if isinstance(l,nn.Linear):
        torch.nn.init.orthogonal_(l.weight)
        l.bias.data.fill_(0.0)


class policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim,64)
        self.l2 = nn.Linear(64,64)
        self.mean = nn.Linear(64,hypers.action_dim)
        self.std = nn.Linear(64,hypers.action_dim)
        self.optim = Adam(self.parameters(),lr=hypers.lr)
        self.apply(weight_init)

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
        
class q_network(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(hypers.obs_dim+hypers.action_dim,64)
        self.l2 = nn.Linear(64,64)
        self.l3 = nn.Linear(64,1)
        self.optim = Adam(self.parameters(),lr=hypers.lr)
        self.apply(weight_init)
    
    def forward(self,obs,action):
        x = torch.cat((obs,action),dim=-1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return x 
    

def process_obs(obs:dict):
    observation = obs["observation"]     # (n env,9)
    achieved_goal = obs["achieved_goal"] # (n env,3)
    desired_goal = obs["desired_goal"]   # (n env,3)
    output = torch.from_numpy(
        np.concatenate([observation,achieved_goal,desired_goal],axis=-1) 
    )
    assert output.shape == torch.Size([hypers.num_envs,15]) or output.shape == torch.Size([15]),f"-->> {output.shape}"
    return output.to(device=hypers.device,dtype=torch.float32)  

def process_her_states(observation,achieved_goal,desired_goal):
    output = torch.from_numpy(np.concatenate([observation,achieved_goal,desired_goal],axis=-1))
    assert output.shape == torch.Size([hypers.num_envs,15]) 
    return output.to(device=hypers.device)

def her_reward(goal_a,goal_b):
    goal_a = torch.from_numpy(goal_a)
    goal_b = torch.from_numpy(goal_b)
    distance_threshold = 0.05
    output = LA.norm(goal_a - goal_b,dim=-1)
    return -(output > distance_threshold).to(device=hypers.device,dtype=torch.float32)


class hindsight_buffer: 
    def _init_storage(self,capacity=hypers.max_steps): 
        act_dim = (hypers.num_envs,hypers.action_dim) # action shape
        self.stor_rewards = torch.zeros((capacity,hypers.num_envs,),dtype=torch.float16,device=hypers.device) 
        self.stor_truncs = torch.zeros((capacity,hypers.num_envs,),dtype=torch.bool,device=hypers.device)
        self.stor_actions = torch.zeros((capacity,*act_dim),dtype=torch.float16,device=hypers.device)
        self.pointer = 0

    def __init__(self,env,policy):
        self.curr_state = [] # current states storage
        self.nx_states = []  # next states storage
        self._init_storage()
        self.env = env
        self.policy = policy
        self.obs = self.env.reset()[0]
        self.epi_reward = deque(maxlen=hypers.num_envs)
        self.reward = np.zeros(hypers.num_envs,np.float16)
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

        for i in range(hypers.num_envs):
            self.reward[i]+=reward[i]
            if trunc[i]:
                self.epi_reward.append(self.reward[i])
                self.reward[i] = 0
 
        saved_action = (torch.from_numpy(np.array(action)) if isinstance(action,np.ndarray) else action)

        self.curr_state.append(self.obs)
        self.nx_states.append(nx_state)

        self.store(
            torch.from_numpy(reward).to(device=hypers.device),
            torch.tensor(trunc).to(device=hypers.device),
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
        return torch.tensor([self.epi_reward]).mean()
    
    def __len__(self):
        return len(self.curr_state)
    

def her_sample(batch_size,k, curr_states,nx_states,rewards,truncs,actions,writter):  # target ratio 4:1, strategy : future
    num_episodes = len(curr_states)//50
    epi_idx = np.random.randint(num_episodes)
    epi_start_idx = epi_idx*50
    batch = curr_states[epi_start_idx:epi_start_idx+50]
    nx_batch = nx_states[epi_start_idx:epi_start_idx+50]

    _her_curr = []
    _her_nx = []
    _her_rewards = []
    _her_truncs = []
    _her_actions = []

    for _ in range(hypers.horizon):
        idx = random.randint(0,len(batch)-2)
        for _ in range(k):
            future_idx = random.randint(idx+1,len(batch)-1)
            curr = batch[idx] # t
            nx = batch[idx+1] # t+1
            future = batch[future_idx] # t'  
            her_rewards = her_reward(curr["achieved_goal"],future["achieved_goal"])
            writter.add_scalar("Main/alpha exp",her_rewards.mean(),new_style=True)
            curr_her_transition = process_her_states(curr["observation"],curr["achieved_goal"],future["achieved_goal"])
            nx_her_transition = process_her_states(nx["observation"],nx["achieved_goal"],future["achieved_goal"])
           
            _her_curr.append(curr_her_transition)
            _her_nx.append(nx_her_transition)
            _her_rewards.append(her_rewards)
            _her_truncs.append(truncs[idx])  
            _her_actions.append(actions[idx])  
        
    assert (len(_her_actions)==len(_her_curr)==len(_her_nx)==len(_her_rewards)==len(_her_truncs)==50*k)

    c = torch.stack([process_obs(n) for n in batch])     # normal transitions
    nx = torch.stack([process_obs(m) for m in nx_batch])
    r = rewards[epi_start_idx:epi_start_idx+hypers.horizon] 
    tr = truncs[epi_start_idx:epi_start_idx+hypers.horizon] 
    a = actions[epi_start_idx:epi_start_idx+hypers.horizon] 

    s_c = torch.cat([c,torch.stack(_her_curr)])  # normal transitions + HER transitons
    s_nx = torch.cat([nx,torch.stack(_her_nx)])
    s_r = torch.cat([r,torch.stack(_her_rewards)])
    s_tr = torch.cat([tr,torch.stack(_her_truncs)])
    s_a = torch.cat([a,torch.stack(_her_actions)])

    sample_idx = torch.randperm(s_c.size(0))[:batch_size]
    
    return (
        s_c[sample_idx].float(),
        s_nx[sample_idx].float(),
        s_r[sample_idx].unsqueeze(-1).float(),
        s_tr[sample_idx].unsqueeze(-1).float(),
        s_a[sample_idx].float(),
    )

def her_worker(queue,buffer:hindsight_buffer,writter):  # multithread worker
    while True:
        if len(buffer)>50:
            states,nx_state,reward,trunc,action = her_sample(
                hypers.batch_size,
                4,
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
        self.buffer = hindsight_buffer(self.env,self.policy)

        self.queue = queue.Queue(maxsize=40)
        self.thread = threading.Thread(target=her_worker,args=(self.queue,self.buffer,self.writter),daemon=True)
        self.thread.start()

        self.entropy_target = -hypers.action_dim
        self.log_alpha = torch.tensor([0.0],requires_grad=True,device=hypers.device)
        self.alpha_optim = Adam([self.log_alpha],lr=1e-5)

        self.test_env = test_env()
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
            self.alpha_optim.load_state_dict(check_point["alpha_optim"])
            self.log_alpha = check_point["log_alpha"]

    @torch.no_grad()
    def test(self):
        state = self.test_env.reset()[0]
        re_test = 0
        for _ in range(500):
            _,_,action_test = self.policy(process_obs(state))
            obs,reward_test,_,trunc,_ = self.test_env.step(action_test.tolist())
            state = obs
            re_test+=reward_test
            if trunc:
                self.test_step+=1
                self.writter.add_scalar("Main/reward test",re_test,self.test_step,new_style=True)
                re_test = 0
                state = self.test_env.reset()[0]

    def train(self,start=False):
        if start:
            self.load()
            t = 0
            alpha = self.log_alpha.exp()
            for n in tqdm(range(hypers.max_steps-1),total=hypers.max_steps-1):
                self.buffer.add()

                if (n+1) % 50 == 0:
                    self.writter.add_scalar("Main/epi reward",self.buffer.util(),n,new_style=True)

                if len(self.buffer) >= hypers.warmup: 
                    states,nx_state,reward,trunc,action = self.queue.get()
            
                    with torch.no_grad():
                        target_action,log_target_action,_ = self.policy(states)
                        q1_target = self.q1_target(nx_state,target_action)
                        q2_target = self.q2_target(nx_state,target_action)
                        q_target = reward + (1-trunc) * hypers.gamma * (torch.min(q1_target,q2_target) - alpha * log_target_action)
                    q1 = self.q1(states,action) 
                    q2 = self.q2(states,action)
                    q_loss = F.mse_loss(q1,q_target) + F.mse_loss(q2,q_target)
                    self.q_optim.zero_grad()
                    q_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.q1.parameters(),1.0)
                    torch.nn.utils.clip_grad_norm_(self.q2.parameters(),1.0)
                    self.q_optim.step()

                    p_action,log_p_action,_ = self.policy(states)
                    min_q = torch.min(self.q1(states,p_action),self.q2(states,p_action))
                    policy_loss = ((alpha*log_p_action) - min_q).mean()
                    self.policy.optim.zero_grad()
                    policy_loss.backward()
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
             
                    if (n+1) % int(1e5) == 0:
                        t+=1
                        self.save(t)
                        self.buffer.save()
                        self.test() # test every 100k gradient steps for just 10 episodes
                     
            torch.save(self.policy.state_dict(),f"./model-final.pth")


if __name__ == "__main__":
    main().train(True)