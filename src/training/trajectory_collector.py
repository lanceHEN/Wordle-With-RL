import torch
from torch.distributions import Categorical
import random

def generate_trajectory(env, guess_list, actor_critic, word_encodings, device="cpu", gamma=1):
    """
    Simulates one episode of Wordle using the current policy.

    At each step:
    - Encodes the observation.
    - Computes logits for all words, masking invalid ones.
    - Samples an action after taking softmax over the logits, logs probability, and records reward and value.
    
    Returns:
        A dictionary with everything needed for PPO training - observations, actions, log probs, returns, advantages, and indices of valid words.
    """
    obs = env.reset()
    done = False

    observations = []
    actions = []
    log_probs = []
    rewards = []
    values = []
    valid_indices_all = []

    with torch.no_grad(): # not training during collection
        while not done:
            
        
            valid_indices = obs["valid_indices"]
            scores, value = actor_critic([obs], word_encodings)
            dist = Categorical(logits=scores)
            # Choose an action
            action_idx = dist.sample()
            log_prob = dist.log_prob(action_idx) # get log probability - taking the log provides numerical stability

            # Get actual word from index
            action_word = guess_list[action_idx.item()]

            # Step the environment
            next_obs, reward, done = env.step(action_word)

            value = value.squeeze() # get predicted value for state

            # Store trajectory
            observations.append(obs)
            actions.append(action_idx.item())
            log_probs.append(log_prob.squeeze(0))
            rewards.append(torch.tensor(reward, dtype=torch.float, device=device))
            values.append(value)
            valid_indices_all.append(valid_indices)

            obs = next_obs
            #print(obs["valid_indices"])

        # final value - always 0 because of terminal state
        last_value = 0

        # Advantage and return calculations
        values.append(torch.tensor(last_value, device=device))
        advantages, returns = compute_advantages(rewards, values, gamma=gamma, device=device)


        return {
            "observations": observations,
            "actions": actions,
            "log_probs": log_probs,
            "returns": returns,
            "advantages": advantages,
            "valid_indices": valid_indices_all
        }

def compute_advantages(rewards, values, gamma=1, device="cpu"):
    """
    Computes simple advantages and returns.
    Advantages are simply the difference of returns and predicted values.
    Returns are the discounted sum of current and future rewards (if gamma=1, there's no discount)
    
    Args:
        rewards: list of individual rewards [r_0, r_1, ..., r_T-1]
        values: list of value estimates [v_0, v_1, ..., v_T] (note: T+1 entries)
        gamma: discount factor
    
    Returns:
        advantages: A_t = G_t - V(s_t)
    """
    returns = []
    G = values[-1].item()  # start with bootstrapped value 0 
    for r in reversed(rewards): # work backwards from end, accumulating actual rewards
        G = r.item() + gamma * G
        returns.insert(0, G)

    # Convert to tensors
    returns = torch.tensor(returns, dtype=torch.float, device=device)
    values = torch.stack(values[:-1])
    advantages = returns - values

    return advantages, returns

def generate_batched_trajectories(
    batched_env,
    guess_list,
    answer_list,
    word_encodings,
    actor_critic,
    gamma=1.0,
    device="cpu",
    fifo_queue=None,  # FIFO queue for hard solutions
    fifo_percentage=0.0
):
    """
    Simulates each episode of Wordle in the given batched_env simultaneously using the current policy. Includes a FIFO queue from which to take no more than fifo_percentage*batch_size
    challenging words, to boost performance on those words through concentrated training.

    At each step:
    - Encodes the observations.
    - Computes logits for all words, for each episode, masking invalid ones.
    - Samples actions after taking softmax over the scores, logs probabilities, and records rewards and values.
    
    Returns:
        A dictionary with everything needed for PPO training - observations, actions, log probs, returns, advantages, and indices of valid words.
        For maintaining the FIFO queue, also includes the answers (secret words) to each game in the environments, whether they were won, and the number of guesses taken.
    """
    batch_size = batched_env.batch_size

    # FIFO implementation, for the given percentage, portion of the batch will be trained on previously missed words
    num_hard_words = int(batch_size * fifo_percentage)
    num_hard_words = min(num_hard_words, len(fifo_queue))
    # Grab the oldest words (FIFO order)
    hard_words_sample = []
    for _ in range(num_hard_words):
        hard_words_sample.append(fifo_queue.popleft())
    available_words = list(set(answer_list) - set(hard_words_sample))
    rest = random.sample(available_words, batch_size - num_hard_words)
    starting_words = hard_words_sample + rest

    obs_list = batched_env.reset(starting_words) #FIFO starting words, else randomized

    all_obs = [[] for _ in range(batch_size)]
    all_actions = [[] for _ in range(batch_size)]
    all_log_probs = [[] for _ in range(batch_size)]
    all_rewards = [[] for _ in range(batch_size)]
    all_values = [[] for _ in range(batch_size)]

    with torch.no_grad():
        while not batched_env.all_done():
            # Vectorized observation encoding
            logits, values = actor_critic(obs_list, word_encodings)

            dist = Categorical(logits=logits)
            actions = dist.sample()
            #actions = torch.argmax(logits, dim=-1)
            log_probs = dist.log_prob(actions)
            words = [guess_list[a.item()] for a in actions]

            next_obs_list, rewards, dones = batched_env.step(words)

            for i in range(batch_size):
                if not dones[i]: #if false for done
                    all_obs[i].append(obs_list[i])
                    all_actions[i].append(actions[i].item())
                    all_log_probs[i].append(log_probs[i])
                    all_rewards[i].append(torch.tensor(rewards[i], dtype=torch.float))
                    all_values[i].append(values[i])
                    #envAnswers[i].append(obs_list[i]["answer"]) 
                    #successBools[i].append(obs_list[i]["success"]) 

            obs_list = next_obs_list

    # Compute returns and advantages
    trajectories = {"observations": [], "actions": [], "log_probs": [], "returns": [], "advantages": [], "env_answers": [], "success_bools": [], "num_guesses": []}

    for i in range(batch_size):
        rewards = all_rewards[i]
        values = all_values[i]
        
        # Skip if no steps were taken
        if len(values) == 0:
            continue
        
        values.append(torch.tensor(0.0, device=device))  # final value = 0
        returns, advantages = compute_advantages(rewards, values, gamma, device=device)
        trajectories["observations"].extend(all_obs[i])
        trajectories["actions"].extend(all_actions[i])
        trajectories["log_probs"].extend(all_log_probs[i])
        trajectories["returns"].extend(returns)
        trajectories["advantages"].extend(advantages)

        # Append the answers, success flag, and guesses for this trajectory
        env = batched_env.envs[i]
        trajectories["env_answers"].append(env.game.word)
        trajectories["success_bools"].append(env.game.is_won)
        trajectories["num_guesses"].append(len(rewards))  # number of guesses = number of rewards - if the game is already over, the guesses don't increase

    return trajectories
