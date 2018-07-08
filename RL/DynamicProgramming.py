# File created to practice Dynamic Programming method

# Jack manages two locations for a nationwide car rental company. Each day,
# some number of customers arrive at each location to rent cars. If Jack
# has a car available, he rents it out and is credited $10 by the national
# company. If he is out of cars at that location, then the business is lost.
# Cars become available for renting the day after they are returned. To help
# ensure that cars are available where they are needed, Jack can move them
# between the two locations overnight, at a cost of $2 per car moved. We
# assume that the number of cars requested and returned at each location
# are Poisson random variables. Expected number of rental request is 3 for
# location 1 and 4 for location 2, and expected number of returned cars is
# 3 for location 1 and 2 for location 2.To simplify the problem slightly,
# we assume that there can be no more than 20 cars at each location (any
# additional cars are returned to the nationwide company, and thus disappear
# from the problem) and a maximum of five cars can be moved from one location
# to the other in one night. We take the discount rate to be gamma and formulate
# this as a continuing finite MDP, where the time steps are days, the state
# is the number of cars at each location at the end of the day, and the actions
# are the net numbers of cars moved between the two locations overnight.

import numpy as np
from scipy.stats import skellam
import matplotlib.pyplot as plt
import math


discount = 0.9

pi_0 = np.zeros((21,21))



##def expected_rentals(a,b):
##    # a--> the number of cars available, b--> the expected number of rental
##    # requests
##    poisson = np.vectorize(lambda n,l: l**n*math.exp(-l)/math.factorial(n))
##    s = poisson(np.arange(a),b)
##    return np.inner(s,np.arange(a))+a*(1-s.sum())
##
##
##Rewards = np.empty((21,21))
### expected amount of money to be made given the state
##for s1 in range(Rewards.shape[0]):
##    for s2 in range(Rewards.shape[1]):
##        Rewards[s1,s2] = 10*(expected_rentals(s1+1,3)
##                             + expected_rentals(s2+1,4))
##
##Transitions = np.empty((21,21,21,21))
##for s1loc1 in range(21):
##    for s1loc2 in range(21):
##        for s2loc1 in range(21):
##            for s2loc2 in range(21):
##                if s2loc1 == 0:
##                    tloc1 = skellam.cdf(s2loc1-s1loc1,3,3)
##                elif s2loc1==20:
##                    tloc1 = skellam.sf(s2loc1-s1loc1,3,3) + skellam.pmf(s2loc1-s1loc1,3,3)
##                else:
##                    tloc1 = skellam.pmf(s2loc1-s1loc1,3,3)
##                if s2loc2 == 0:
##                    tloc2 = skellam.cdf(s2loc2-s1loc2,2,4)
##                elif s2loc2 == 20:
##                    tloc2 = skellam.sf(s2loc2-s1loc2,2,4) + skellam.pmf(s2loc2-s1loc2,2,4)
##                else:
##                    tloc2 = skellam.pmf(s2loc2-s1loc2,2,4)
##                Transitions[s1loc1,
##                            s1loc2,
##                            s2loc1,
##                            s2loc2] = tloc1*tloc2
##np.save('rewards.npy',Rewards)
##np.save('transitions.npy',Transitions)
Rewards = np.load('rewards.npy')
transitions = np.load('transitions.npy')

######################################################

def policy_eval(pi, V_0, theta): 
    V_s = V_0
    v = V_s.copy()
    for i in range(V_s.shape[0]):
        for j in range(V_s.shape[1]):
            ip = i - pi[i,j].astype(int)
            jp = j + pi[i,j].astype(int)
            V_s[i,j] = Rewards[i,j]+discount*(transitions[ip,jp,:,:]*V_s).sum()
    delta = np.abs(v-V_s).max()
    while (delta>theta):
        #print(delta)
        v = V_s.copy()
        for i in range(V_s.shape[0]):
            for j in range(V_s.shape[1]):
                ip = i - pi[i,j].astype(int)
                jp = j + pi[i,j].astype(int)
                V_s[i,j] = Rewards[i,j]+discount*(transitions[ip,jp,:,:]*V_s).sum()
        delta = np.abs(v-V_s).max()
    return V_s

def policy_select(V):
    new_pi = np.empty((21,21))
    for i in range(new_pi.shape[0]):
        for j in range(new_pi.shape[1]):
            action_range = np.array(range(max(-5,i-20,-j),min(5,i,20-j)+1)) # the range of legal actions
            #print(max(-5,i-20,-j),min(5,i,j-20)+1,action_range.shape)
            action_values = np.empty(action_range.shape)
            for option in range(len(action_range)):
                ip = i-action_range[option]
                jp = j+action_range[option]
                action_cost = -2*abs(action_range[option])
                action_values[option] = action_cost + Rewards[ip,jp] + discount*(transitions[ip,jp,:,:]*V).sum()
            new_pi[i,j] = action_range[action_values.argmax()]
    return new_pi

    
def main():
    V_0 = np.zeros((21,21))
    pi = pi_0.copy()
    V = policy_eval(pi,V_0,0.01)
    pi_improved = policy_select(V)
    policy_stable = (pi_improved==pi).all()
    while not policy_stable:
        print(policy_stable)
        V = policy_eval(pi_improved,V,0.01)
        pi = pi_improved.copy()
        pi_improved = policy_select(V)
        policy_stable = (pi_improved==pi).all()

    print(policy_stable)
    plt.pcolormesh(pi_improved)
    plt.show()

main()
    



    



