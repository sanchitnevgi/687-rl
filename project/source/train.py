import csv
import os
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

def j_pi(theta, states, actions, rewards):
    '''
    Compute J(pi) estimate through Per Decision Importance Sampling (PDIS)
    Input - Pi (S x A) of logits
    '''
    # Reshape theta vector into tabular policy
    theta = theta.reshape(18, 4)

    # Get a batch of data
    batch_size = 10000
    
    # Sample from dataset
    sample = np.random.choice(np.arange(len(states)), size=batch_size)
    state_b, action_b, rewards_b = states[sample], actions[sample], rewards[sample]

    pdis = compute_pdis(theta, state_b, action_b, rewards_b)
    pdis = pdis.mean()

    # Need to maximize pdis
    return -pdis

def policy_to_file(tabular_policy, idx):
    '''
    Input - Tabular Policy S x A
    Write to file
    '''
    tabular_policy = tabular_policy.ravel()

    with open(f"../policy{idx}.txt", "w") as f:
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

def create_numpy_array(split="candidate"):

    with open(f"./{split}.csv") as c:
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

    os.mkdir(split)
    np.save(f"./{split}/states.npy", states)
    np.save(f"./{split}/actions.npy", actions)
    np.save(f"./{split}/rewards.npy", rewards)

def get_gamma(g = 0.99):
    gamma = np.zeros(100)
    gamma.fill(g)

    gamma = gamma ** np.arange(0, 100)

    return gamma

def train():
    print("Reading states, action, rewards")
    states, actions, rewards = [np.load(f"./candidate/{name}.npy") for name in ("states", "actions", "rewards")]

    # Initial Theta
    theta = np.zeros((18 * 4))
    print("Running CMA-ES")
    theta_c, _ = cma.fmin2(j_pi, theta, 0.4, args=(states, actions, rewards,), options={'maxiter': 100})

    theta_c = theta_c.reshape((18, 4))

    return theta_c

def evaluate(theta_c):
    confidence = 0.99
    states_s, actions_s, rewards_s = [np.load(f"./safety/{name}.npy") for name in ("states", "actions", "rewards")]

    pdis = compute_pdis(theta_c, states_s, actions_s, rewards_s)
    test_size = len(pdis)

    t_statistic = t.ppf(confidence, test_size - 1)
    std = np.std(pdis)

    policy_return = pdis.mean() - std * t_statistic / np.sqrt(test_size)

    return policy_return

if __name__ == "__main__":
    print("Creating candidate & Safety split")
    create_splits()

    print("Creating numpy arrays")
    create_numpy_array("candidate")
    create_numpy_array("safety")

    print("Creating behavior policy, gamma vector")
    behavior = get_behavior_policy()
    gamma = get_gamma(0.95)
    
    # The new policy should be at least as good as this
    BOUND = 1.5

    # Generate n "safe" policies
    n_policies = 100
    policy_i = 0

    while policy_i < n_policies:
        # Generate theta
        theta_c = train()

        # Validate this policy
        _return = evaluate(theta_c)

        print("Policy Return", _return)

        if _return >= BOUND:
            # Write theta to file
            print("Writing theta to file")
            policy_to_file(theta_c, policy_i + 1)
            policy_i += 1
    
