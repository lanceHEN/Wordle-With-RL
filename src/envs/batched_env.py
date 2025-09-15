from envs.wordle_env import WordleEnv

# This class allows for a batch of wordle environments together, using the same reset and step methods as in WordleEnv.
# This was done to enable quicker trajectory collection and training time.
class BatchedWordleEnv:
    
    # Given an environment class (e.g. WordleEnv), guess list, answer list, batch size, win reward, and lose reward,
    # produces a BatchedWordleEnv, a batched set of environments each with the given guess list, answer list, batch size, env class, win reward,
    # and lose reward, with the given number of batches.
    def __init__(self, guess_list, answer_list, batch_size, env_class=WordleEnv, win_reward=20, lose_reward=-10):
        self.envs = [env_class(guess_list, answer_list, win_reward=win_reward, lose_reward=lose_reward) for _ in range(batch_size)]
        self.batch_size = batch_size

        # store current states
        self.current_obs = [env.reset() for env in self.envs]
        self.dones = [False] * batch_size

    # Resets across each env in the batch, and returns the observations for each.
    # If starting words specified, starts each env with those starting words.
    def reset(self, starting_words=None):
        if starting_words:
            self.current_obs = [
                env.reset(starting_words[i]) if i < len(starting_words) else env.reset()
                for i, env in enumerate(self.envs)
            ]
        else:
            self.current_obs = [env.reset() for env in self.envs]
        self.dones = [False] * self.batch_size
        return self.current_obs

    # For each action in the list, applies that on the corresponding environment, returning the corresponding observations, rewards, and dones each as lists.
    def step(self, actions):
        """
        Args:
            actions: list of str (words) to play for each env
        Returns:
            next_obs_list, reward_list, done_list
        """
        next_obs = []
        rewards = []
        new_dones = []

        for i, (env, action, done) in enumerate(zip(self.envs, actions, self.dones)):
            if done:
                next_obs.append(self.current_obs[i])
                rewards.append(0.0)
                new_dones.append(True)
            else:
                obs, reward, done = env.step(action)
                self.current_obs[i] = obs
                next_obs.append(obs)
                rewards.append(reward)
                new_dones.append(done)

        self.dones = new_dones
        return next_obs, rewards, new_dones

    # Determines if all the envs are done (are all games over?).
    def all_done(self):
        return all(self.dones)
