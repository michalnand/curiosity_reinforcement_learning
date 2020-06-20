import numpy
import torch


from .CuriosityModule import *

class Buffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, logprobs, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.logprobs.append(logprobs)
        self.rewards.append(reward)
        self.dones.append(done)

    
    def clear(self):
        self.states         = []
        self.actions        = []
        self.logprobs       = []
        self.rewards        = []
        self.dones          = []

    def size(self):
        return len(self.states)

    def sample(self, batch_size):
        states         = []
        actions        = []
        logprobs       = []
        rewards        = []
        dones          = []

        for i in range(batch_size):
            idx = numpy.random.randint(self.size())

            states.append(self.states[idx])
            actions.append(self.actions[idx])
            logprobs.append(self.logprobs[idx])
            rewards.append(self.rewards[idx])
            dones.append(self.dones[idx])

        return states, actions, logprobs, rewards, dones



class AgentPPOCuriosity():
    def __init__(self, env, model_ppo, model_curiosity, config):
        self.env = env
 
        self.gamma          = config.gamma
        self.eps_clip       = config.eps_clip
        self.entropy_beta   = config.entropy_beta


        self.buffer_size        = config.buffer_size
        self.batch_size         = config.batch_size 
        self.training_epochs    = config.training_epochs

        self.curiosity_scale = config.curiosity_scale

        
        self.state_shape = self.env.observation_space.shape
        self.actions_count     = self.env.action_space.n

        self.state  = self.env.reset()

        self.buffer         = Buffer()


        self.model_ppo          = model_ppo.Model(self.state_shape, self.actions_count)
        self.model_ppo_old      = model_ppo.Model(self.state_shape, self.actions_count)
        self.optimizer_ppo  = torch.optim.Adam(self.model_ppo.parameters(), lr=config.learning_rate)
        
        self.curiosity_module = CuriosityModule(model_curiosity, self.state_shape, self.actions_count, config.curiosity_learning_rate, config.curiosity_buffer_size)

        self.enable_training()

        self.iterations = 0


    def enable_training(self):
        self.enabled_training = True

    def disable_training(self):
        self.enabled_training = False

  

    def main(self):
        self.iterations+= 1

        state_t = torch.from_numpy(self.state).float().to(self.model_ppo.device)

        action_idx, action, logprobs = self._get_action(state_t.unsqueeze(0))


        self.state, reward, done, _ = self.env.step(action_idx)
        
        if self.enabled_training: 

            self.buffer.add(state_t, action, logprobs, reward, done)

            if self.buffer.size() >= self.buffer_size:
                self._train()
                self.buffer.clear()

        if self.enabled_training:
            self.curiosity_module.add(self.state, action_idx)
            self.curiosity_module.train()
            
        if done:
            self.state = self.env.reset()

        return reward, done
                

           
    def save(self, save_path):
        self.model_ppo.save(save_path)
        self.curiosity_module.save(save_path)

    def load(self, save_path):
        self.model_ppo.load(save_path)
        self.model_ppo_old.load(save_path)
        self.curiosity_module.load(save_path)
    

    def _train(self):   
        
        #compute discounted rewards 
        rewards = self._calc_rewards(self.buffer.rewards, self.buffer.dones)

        #normalise rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        self.buffer.rewards = rewards
 

        n_batch = (self.training_epochs*self.buffer_size)//self.batch_size
        for batch in range(n_batch):
            
            states_s, actions_s, logprobs_s, rewards_s, dones_s = self.buffer.sample(self.batch_size)

            #create tensors 
            states_t    = torch.stack(states_s).to(self.model_ppo.device).detach()
            actions_t   = torch.stack(actions_s).to(self.model_ppo.device).detach()
            logprobs_t  = torch.stack(logprobs_s).to(self.model_ppo.device).detach()
            rewards_t   = torch.tensor(rewards_s).to(self.model_ppo.device).detach()
            
            '''

            states_t    = torch.stack(self.buffer.states).to(self.model.device).detach()
            actions_t   = torch.stack(self.buffer.actions).to(self.model.device).detach()
            logprobs_t  = torch.stack(self.buffer.logprobs).to(self.model.device).detach()
            rewards_t   = torch.tensor(self.buffer.rewards).to(self.model.device).detach()
            '''

            states_next_t = torch.cat((states_t[1:], states_t[-1].unsqueeze(0)), dim=0)
            curiosity = self.curiosity_module.eval(states_t, actions_t, states_next_t)


            rewards_t = rewards_t + self.curiosity_scale*curiosity

            #evaluate policy and value:
            logprobs, state_values, dist_entropy = self._evaluate(states_t, actions_t)

                
            #compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(logprobs - logprobs_t.detach())
            
            #compute loss
            advantage = rewards_t - state_values.detach()
            
            loss1 = ratio*advantage
            loss2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)*advantage
                
            loss = -torch.min(loss1, loss2)
            loss+= ((rewards_t - state_values)**2) 
            loss+= -self.entropy_beta*dist_entropy

            loss = loss.mean()

            #take gradient step
            self.optimizer_ppo.zero_grad()
            loss.backward()
            self.optimizer_ppo.step()

        
        #TODO
        #copy new weights into old policy:
        self.model_ppo_old.load_state_dict(self.model_ppo.state_dict())




    def _get_action(self, state):
        policy, _ = self.model_ppo_old.forward(state)

        policy = policy.squeeze(0)

        dist   = torch.distributions.Categorical(policy)
        action = dist.sample()
        
        return action.item(), action, dist.log_prob(action)
        

    def _evaluate(self, state, action):

        policy, value = self.model_ppo.forward(state)


        dist   = torch.distributions.Categorical(policy)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, torch.squeeze(value), dist_entropy
       


    def _calc_rewards(self, rewards, done):
        size    = len(rewards)
        result  = numpy.zeros((size, ))

        q = 0.0
        for n in reversed(range(size)):
            if done[n]:
                gamma = 0.0
            else:
                gamma = self.gamma

            q = rewards[n] + gamma*q
            result[n] = q

        return result
   