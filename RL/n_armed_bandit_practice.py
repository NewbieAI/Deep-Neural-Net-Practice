# Reinforcement Learning Practice

# n-armed bandit problem

# testbed generation
import numpy as np
import matplotlib.pyplot as mp
# each row is the different set of Q*(a)
# n --> the number of actions that can be taken
# m --> the number of separate test runs
n = 10
m = 2000
Q_values = np.random.randn(m,n)
total_reward = np.zeros((m,n))
observations = np.zeros((m,n))


# when e = 0, it's just simple greedy strategy
def epsilon_greedy(total_reward,observations,e):
    if not (e>=0 and e<=1):
        raise ValueError('Epsilon must be within [0,1]')
    else:
        # using average reward as an estimate of Q_values
        # elementwise division, 0/0 is defined as 0
        Q_observed = total_reward/np.maximum(observations,1)
        # if more than one maximum is observed, then the tie is
        # broken randomly
        is_max = np.equal(Q_observed,Q_observed.max(axis=1,keepdims=True))
        action = np.empty(m)
        for row_index in range(m):
            if np.random.random()>e or is_max.all(): # takes the greedy action
                choices = is_max[row_index,:].nonzero()
                
            else: # takes the non greedy action
                choices = (~is_max)[row_index,:].nonzero()
            if len(choices[0])==1:
                action[row_index] = choices[0][0]
            else:
                action[row_index] = np.random.choice(choices[0])
        return action

def softmax_selection(total_reward,observations,t):
    if not t>0:
        raise ValueError('Temperature must be with [0, Inf)')
    else:
        # also uses average reward as an estimate of Q_values
        Q_observed = total_reward/np.maximum(observations,1)
        temp = np.exp((Q_observed-Q_observed.max(axis=1,keepdims=True))/t)
        selection_prob = temp/temp.sum(axis=1,keepdims=True)
        action = np.empty(m)
        for row_index in range(m):
            action[row_index] = np.random.choice(n,p=selection_prob[row_index,:])
        return action
        
    
def reward(action):
    if all([a in range(n) for a in action]):
        r = Q_values[range(m),action.astype(int)]+np.random.randn(m)
        return r
    else:
        raise ValueError('invalid action')
    
def use(strategy,trials,parameter):
    total_reward = np.zeros((m,n))
    observations = np.zeros((m,n))
    avg_reward = []
    percent_optimal = []
    for i in range(trials):
        action = strategy(total_reward,observations,parameter)
        reward_received = reward(action)
        avg_reward.append(reward_received.mean())
        percent_optimal.append((action==Q_values.argmax(axis=1)).mean())
        total_reward[range(m),action.astype(int)]+=reward_received
        observations[range(m),action.astype(int)]+=1
    return [avg_reward,percent_optimal]



def run():
    [r1,p1]=use(epsilon_greedy,1000,0)
    [r2,p2]=use(softmax_selection,1000,0.1)
    [r3,p3]=use(softmax_selection,1000,0.01)
    mp.subplot(211)
    mp.plot(r1,'r')
    mp.plot(r2,'g')
    mp.plot(r3,'b')
    mp.legend(['temperature = 0','temperature = 0.1','temperature = 1'])
    mp.title('Epsilon-Greedy Strategy')
    mp.ylabel('Average Reward')
    mp.subplot(212)
    mp.plot(p1,'r')
    mp.plot(p2,'g')
    mp.plot(p3,'b')
    mp.legend(['temperature = 0','temperature = 0.1','temperature = 1'])
    mp.ylabel('% Optimal Action')
    mp.xlabel('Plays')
    mp.show()


run()
