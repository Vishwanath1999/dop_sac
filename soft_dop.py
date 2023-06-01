import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)#,dtype=np.uint8
        self.new_state_memory = np.zeros((self.mem_size, *input_shape),dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.noise_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros((self.mem_size,))
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, state_, done,noise):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.noise_memory[index] = noise

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        noise = self.noise_memory[batch]

        return states, actions, rewards, states_, dones, noise

class ActorNetwork(nn.Module):
    def __init__(self,alpha,input_dims,n_actions,fc1_dims,fc2_dims,name,\
        chkpt_dir='tmp/dop',action_scale=10,noise=0.1):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.action_scale = action_scale
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.chkpt_file = os.path.join(chkpt_dir,name)
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(np.prod(input_dims),self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims,self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self,state):
        x = self.fc1(state)
        x = F.elu(x)
        x = self.fc2(x)
        x = F.elu(x)
        mu = self.mu(x)
        sigma = T.clamp(self.sigma(x),-20,2)
        sigma = T.exp(sigma)
        return mu,sigma

    def sample_normal(self,state,reparameterize=False,deterministic=False,actor_update = False):
        mu,sigma = self.forward(state)
        probabilities = T.distributions.Normal(mu, sigma)
        if deterministic==True:
            return mu
        else:
            if reparameterize == True:
                actions = probabilities.rsample()
            else:
                actions = probabilities.sample()
            
            action = T.tanh(actions)
            log_probs = probabilities.log_prob(actions)
            log_probs -= T.log(1-action.pow(2)) + self.reparam_noise
            action *= self.action_scale

            if actor_update == True:
                return actions
            else:
                return action,log_probs
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

    
class MixerNetwork(nn.Module):
    def __init__(self,beta,input_dims,n_agents,n_actions,fc1_dims,name,chkpt_dir='./tmp/dop'):
        super(MixerNetwork, self).__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.n_agents = n_agents
        self.fc1_dims = fc1_dims
        self.chkpt_file = os.path.join(chkpt_dir,name)
        self.n_actions = n_actions

        self.grus = nn.ModuleList([nn.GRU(self.input_dims[1],self.fc1_dims) for _ in range(self.n_agents)])
        self.fc1s = nn.ModuleList([nn.Linear(self.fc1_dims+2*self.n_actions, self.fc1_dims) for _ in range(self.n_agents)])
        self.qs = nn.ModuleList([nn.Linear(self.fc1_dims, 1) for _ in range(self.n_agents)])

        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.gru_mixer = nn.GRU(self.input_dims[1],self.fc1_dims)
        self.weight_mixer = nn.Linear(self.fc1_dims,self.n_agents)
        self.bias_mixer = nn.Linear(self.fc1_dims,1)

        self.optimizer = optim.Adam(self.parameters(), lr=self.beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def forward_mixer(self,state):
        # state=T.flatten(state,start_dim=1)
        weight_bias,_ = self.gru_mixer(state)
        norm_weight_mixer = T.softmax(self.weight_mixer(weight_bias[:,-1,:]),dim=1)
        bias_mixer = self.bias_mixer(weight_bias[:,-1,:])
        return norm_weight_mixer, bias_mixer
    
    def forward(self,state,actions,noise,index=None):
        
        if index == None:
            q_values_ind = T.zeros(actions.shape).to(self.device)
            for idx in range(self.n_agents):
                gru_state,_ = self.grus[idx](state)
                q_value  = T.cat([gru_state[:,-1,:],actions[:,idx][:,None],noise[:,idx][:,None]],dim=1)
                q_value = F.elu(self.fc1s[idx](q_value))
                q_value = self.qs[idx](q_value)
                q_values_ind[:,idx] = q_value[:,0]  
            weight_mixer, bias_mixer = self.forward_mixer(state)
            mixed_q = (weight_mixer*q_values_ind).sum(axis=1,keepdims=True)
            q_tot =  mixed_q + bias_mixer
            return q_tot
        else:
            q_values_ind = None            
            gru_state,_ = self.grus[index](state)
            q_value  = T.cat([gru_state[:,-1,:],actions,noise[:,index][:,None]],dim=1)
            q_value = F.elu(self.fc1s[index](q_value))
            q_value = self.qs[index](q_value)
            q_values_ind = q_value[:,0]
            return q_values_ind


    def save_checkpoint(self):
        T.save(self.state_dict(), self.chkpt_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.chkpt_file))

class dop_agent():
    def __init__(self,input_dims,n_agents,n_actions=1,alpha=5e-4,beta=5e-4,\
        fc1_dims=64,fc2_dims=64,tau=5e-3,env_id=None,action_scale=10,gamma=0.99,max_size=int(1e6),\
        batch_size=1024):
        
        self.n_agents = n_agents
        
        self.actors = [ActorNetwork(alpha=alpha,input_dims=input_dims,n_actions=n_actions,fc1_dims=fc1_dims,\
            fc2_dims=fc2_dims,name=env_id+'_actor_'+str(idx),action_scale=action_scale) for idx in range(self.n_agents)]   
        self.target_actors = [ActorNetwork(alpha=alpha,input_dims=input_dims,n_actions=n_actions,fc1_dims=fc1_dims,\
            fc2_dims=fc2_dims,name=env_id+'_target_actor_'+str(idx),action_scale=action_scale) for idx in range(self.n_agents)]

        self.mixer_1 = MixerNetwork(beta=beta, input_dims=input_dims,n_agents=self.n_agents,\
            n_actions=n_actions,fc1_dims=fc1_dims,name=env_id+'_mixer_1')
        self.target_mixer_1 = MixerNetwork(beta=beta, input_dims=input_dims,n_agents=self.n_agents,\
            n_actions=n_actions,fc1_dims=fc1_dims,name=env_id+'_target_mixer_1')
        self.mixer_2 = MixerNetwork(beta=beta, input_dims=input_dims,n_agents=self.n_agents,\
            n_actions=n_actions,fc1_dims=fc1_dims,name=env_id+'_mixer_2')
        self.target_mixer_2 = MixerNetwork(beta=beta, input_dims=input_dims,n_agents=self.n_agents,\
            n_actions=n_actions,fc1_dims=fc1_dims,name=env_id+'_target_mixer_2')

        self.tau = tau
        self.action_scale = action_scale
        self.update_network_parameters(tau=1) 
        self.memory = ReplayBuffer(max_size,input_dims,n_agents)
        self.batch_size = batch_size
        self.gamma = gamma
        self.time_step=0
        self.learn_step_cntr = 0
        self.action_scale = action_scale
        self.ent_coef = 1e-2
        
    
    def choose_action(self,state,deterministic=False):
        actions = np.zeros((self.n_agents,))
        state = T.tensor(state, dtype=T.float).to(self.mixer_1.device)
        for idx,actor in enumerate(self.actors):
            actor.eval()
            acts,_ = actor.sample_normal(state,deterministic=deterministic)
            actions[idx] = acts.cpu().detach().numpy()[0]
            actor.train()   
        return actions
    
    def mem_ready(self):
        flag = False
        if self.memory.mem_cntr > self.batch_size:
            flag = True
        return flag
    
    def remember(self,state,action,reward,state_,done,noise):
        self.memory.store_transition(state,action,reward,state_,done,noise)

    def learn(self):
        if self.mem_ready() == False:
            return

        device = self.actors[0].device
        state, action, reward, state_, done,noise = \
                                self.memory.sample_buffer(self.batch_size)
                                
        states = T.tensor(state,dtype=T.float).to(device)
        actions = T.tensor(action,dtype=T.float).to(device)
        rewards = T.tensor(reward,dtype=T.float).to(device)
        dones = T.tensor(done).to(device)
        next_states = T.tensor(state_,dtype=T.float).to(device)
        noise = T.tensor(noise,dtype=T.float).to(device)

        with T.no_grad():
            target_actions = T.zeros(self.batch_size,self.n_agents).to(device)
            log_probs_ = T.zeros(self.batch_size,self.n_agents).to(device)
            for idx in range(self.n_agents):
                ta,lp_ = self.target_actors[idx].sample_normal(T.flatten(next_states,start_dim=1))
                target_actions[:,idx] = ta[:,0]
                log_probs_[:,idx] = lp_[:,0]
            
            log_probs_ = log_probs_.sum(1)
            q_tot_1_ = self.target_mixer_1.forward(next_states,target_actions,noise).view(-1)
            q_tot_2_ = self.target_mixer_2.forward(next_states,target_actions,noise).view(-1)          
            q_tot_ = T.min(q_tot_1_,q_tot_2_)
            target = rewards + (1-dones.int())*self.gamma*(q_tot_-self.ent_coef*log_probs_)

        q_tot_1 = self.mixer_1.forward(states,actions,noise).view(-1)
        q_tot_2 = self.mixer_2.forward(states,actions,noise).view(-1)
        self.mixer_1.optimizer.zero_grad()
        self.mixer_2.optimizer.zero_grad()
        mixer_1_loss = 0.5*F.mse_loss(target,q_tot_1)
        mixer_2_loss = 0.5*F.mse_loss(target,q_tot_2)
        mixer_loss = mixer_1_loss + mixer_2_loss
        ml = mixer_loss.item()
        mixer_loss.backward(retain_graph=False)
        T.nn.utils.clip_grad_norm_(self.mixer_1.parameters(), 0.5)
        T.nn.utils.clip_grad_norm_(self.mixer_2.parameters(), 0.5)
        self.mixer_1.optimizer.step()
        self.mixer_2.optimizer.step()

        
        weights_,_ = self.mixer_1.forward_mixer(states)
        actions,lps = T.zeros(self.batch_size,1).requires_grad_(True),T.zeros(self.batch_size,1).requires_grad_(True)
        al = T.zeros(len(self.actors))
        for idx, actor in enumerate(self.actors):   #actor.sample_normal(T.flatten(states,start_dim=1),reparameterize=True,actor_update=True)         
            actor.optimizer.zero_grad()
            actions.data,lps.data = actor.sample_normal(T.flatten(states,start_dim=1),reparameterize=True)
            q1 = self.mixer_1.forward(states,actions,noise,idx)
            q2 = self.mixer_2.forward(states,actions,noise,idx)
            q_val = T.min(q1,q2).view(-1)
            actor_loss = T.mean(self.ent_coef*lps - weights_[:,idx]*q_val)
            al[idx] = actor_loss.data
            T.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
            actor_loss.backward(retain_graph=True)
            actor.optimizer.step()
        
        self.update_network_parameters()
        return ml,al.mean()


    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = []
        target_actor_params = []
        for actor,target_actor in zip(self.actors,self.target_actors):
            actor_params.append(dict(actor.named_parameters()))
            target_actor_params.append(dict(target_actor.named_parameters()))

        mixer_1_params = self.mixer_1.named_parameters()        
        target_mixer_1_params = self.target_mixer_1.named_parameters()
        mixer_2_params = self.mixer_2.named_parameters()        
        target_mixer_2_params = self.target_mixer_2.named_parameters()

        mixer_1 = dict(mixer_1_params)
        target_mixer_1 = dict(target_mixer_1_params)
        mixer_2 = dict(mixer_2_params)
        target_mixer_2 = dict(target_mixer_2_params)

        for name in mixer_1:
            mixer_1[name] = tau*mixer_1[name].clone() + \
                    (1-tau)*target_mixer_1[name].clone()
        
        for name in mixer_2:
            mixer_2[name] = tau*mixer_2[name].clone() + \
                    (1-tau)*target_mixer_2[name].clone()

        for actor,target_actor in zip(actor_params,target_actor_params):
            for name in actor:
                actor[name] = tau*actor[name].clone() + \
                        (1-tau)*target_actor[name].clone()

        self.target_mixer_1.load_state_dict(mixer_1)
        self.target_mixer_2.load_state_dict(mixer_2)
        
        for actor,target_actor in zip(actor_params,self.target_actors):
            target_actor.load_state_dict(actor)

    def save_models(self):
        for actor,target_actor in zip(self.actors,self.target_actors):
            actor.save_checkpoint()
            target_actor.save_checkpoint()
        self.mixer_1.save_checkpoint()
        self.target_mixer_1.save_checkpoint()
        self.mixer_2.save_checkpoint()
        self.target_mixer_2.save_checkpoint()

    def load_models(self):
        for actor,target_actor in zip(self.actors,self.target_actors):
            actor.load_checkpoint()
            target_actor.load_checkpoint()
        self.mixer_1.load_checkpoint()
        self.target_mixer_1.load_checkpoint()
        self.mixer_2.load_checkpoint()
        self.target_mixer_2.load_checkpoint()

