import csv

import numpy as np
import pandas as pd

from scipy.stats import t
from scipy.special import softmax

import cma

from tqdm import tqdm

def create_splits():
    '''
    Create the candidate & safety data split
    '''

    with open("./data.csv") as f, open("./candidate.csv", "w") as c, open("./safety.csv", "w") as s:
        num_episodes = int(next(f))
        print(num_episodes)

        num_candidate = int(0.8 * num_episodes)

        c.write(str(num_candidate) + "\n")
        s.write(str(num_episodes - num_candidate) + "\n")

        for ep in tqdm(range(num_episodes)):
            # Whether to write to candidate file or safety file
            out_file = c if ep < num_candidate else s
 
            ep_len = int(next(f))

            out_file.write(str(ep_len) + "\n")

            for tr in range(ep_len):
                out_file.write(next(f))

def get_behavior_policy():
    # Read the candidate to create behaviour policy.
    policy = np.zeros((18, 4))
    policy.fill(0.25)

    return policy

def compute_pdis(theta, states, actions, rewards):
    # Normalize theta
    theta = theta - theta.max()
    theta = softmax(theta, axis=1)

    pi_ratio = theta[states, actions] / behavior[states, actions]
    pi_ratio = pi_ratio.cumprod(axis=1)

    # Get average PDIS of the batch
    pdis = (gamma * pi_ratio * rewards).sum(axis=1)

    return pdis

def j_pi(theta):
    '''
    Compute J(pi) estimate through Per Decision Importance Sampling (PDIS)
    Input - Pi (S x A) of logits
    '''
    # Reshape theta vector into tabular policy
    theta = theta.reshape(18, 4)

    # Normalize theta
    # theta = theta - theta.max()
    # theta = softmax(theta, axis=1)

    # Get a batch of data
    batch_size = 100000
    
    # Sample from dataset
    sample = np.random.choice(np.arange(len(states)), size=batch_size)
    state_b, action_b, rewards_b = states[sample], actions[sample], rewards[sample]

    # pi_ratio = theta[state_b, action_b] / behavior[state_b, action_b]
    # pi_ratio = pi_ratio.cumprod(axis=1)

    # # Get average PDIS of the batch
    # pdis = (gamma * pi_ratio * rewards_b).sum(axis=1).mean() 

    pdis = compute_pdis(theta, state_b, action_b, rewards_b)
    pdis = pdis.mean()

    # Need to maximize pdis
    return -pdis

def t_test(theta):
    '''
    Check whether the pdis values using t-tests
    '''
    confidence = 0.99
    states_s, actions_s, rewards_s = [np.load(f"./safety/{name}.npy") for name in ("states", "actions", "rewards")]

    # Compute the pdis

    std = np.std(pdis)
    t_statistic = t.ppf(confidence, test_size)


def policy_to_file(tabular_policy, idx):
    '''
    Input - Tabular Policy S x A
    Write to file
    '''
    tabular_policy = tabular_policy.ravel()

    with open(f"./policies/policy{idx}.txt", "w") as f:
        for e in tabular_policy:
            f.write(str(e) + "\n")

def print_episode_quantiles():
    ep_lengths = []
    with open("./candidate.csv") as c:
        eps = int(next(c))
        for ep in tqdm(range(eps)):
            ep_len = int(next(c))
            ep_lengths.append(ep_len)

            for _ in range(ep_len):
                next(c)
    df = pd.Series(ep_lengths)
    print(df.quantile([.7,.8,.9,.99]))

def create_numpy_array():
    with open("./safety.csv") as c:
        num_eps = int(next(c))

        states = np.zeros((num_eps, 100), dtype=np.int)
        actions = np.zeros((num_eps, 100), dtype=np.int)
        rewards = np.zeros((num_eps, 100), dtype=np.float)

        for ep in tqdm(range(num_eps)):
            ep_len = int(next(c))
            
            for t in range(ep_len):
                if t >= 100:
                    next(c)
                    continue
                s_t, a_t, r_t, _ = next(c).split(",")
                
                states[ep, t] = int(s_t)
                actions[ep, t] = int(a_t)
                rewards[ep, t] = float(r_t)
    
    np.save("./safety/states.npy", states)
    np.save("./safety/actions.npy", actions)
    np.save("./safety/rewards.npy", rewards)

def get_gamma(g = 0.99):
    gamma = np.zeros(100)
    gamma.fill(g)

    gamma = gamma ** np.arange(0, 100)

    return gamma

if __name__ == "__main__":
    print("Reading states, action, rewards")
    states, actions, rewards = [np.load(f"./data/{name}.npy") for name in ("states", "actions", "rewards")]

    print("Creating behavior policy, gamma vector")
    behavior = get_behavior_policy()
    gamma = get_gamma(0.95)

    # The new policy should be at least as good as this
    BOUND = 1.5

    # Initial Theta
    theta = np.zeros((18 * 4))
    print("Running CMA-ES")
    theta_c, es = cma.fmin2(j_pi, theta, 0.4, options={'maxiter': 100})

    # Write theta to file
    print("Writing theta to file")
    policy_to_file(theta_c, 4)

    # Validate this policy
    # t_test(theta_c)
  