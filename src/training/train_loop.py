import torch
from tqdm import trange
from training.trajectory_collector import generate_batched_trajectories
from training.ppo_trainer import ppo_update
import random
from eval.evaluate_policy import evaluate_policy_on_all_answers
from envs.batched_env import BatchedWordleEnv
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque



# Main training loop for PPO applied to Wordle.
# Trains the model for the specified number of epochs, at each epoch collecting trajectories from the batched environments, and applying PPO updates.
# Also periodically (every eval_and_save_per epochs) prints out and saves to tensorboard the success rate, and average number of guesses, and also saves the model.
# This also includes a FIFO queue, used for concentrating the model on challenging words - words which take fifo_threshold or more guesses.
# No more than fifo_percentage of the total number of guesses is set aside for words in the FIFO queue.
def training_loop(
    batched_env, # the batched wordle environments, as a BatchedWordleEnv
    actor_critic, # the wrapped model architecture, as a WordleActorCritic
    optimizer, # optimizer
    guess_list, # list of all words that can be used as guesses
    answer_list, # list of all words that can be used as answers
    word_encodings, # encodings for every word in guess_list, as a [len(guess_list), 130] torch tensor
    save_dir, # directory to save the model parameters
    log_dir, # directory to save tensorboard logs
    num_epochs=300, # number of overall epochs
    start_epoch=0, # epoch to start (useful for resuming training)
    ppo_epochs=4, # how many ppo epochs to run on a given set of observations
    eval_and_save_per=20, # how often, in # epochs, to evaluate model performance on all words in answer_list and save the state
    minibatch_size=256, # how many steps to consider at once in the PPO update
    gamma=1.0, # discount factor 
    clip_epsilon=0.2, # how tightly to clip the probability ratio for PPO
    value_loss_coef=0.5, # coefficient for value loss
    device=torch.device("cpu"), # device
    fifo_queue=deque(maxlen=20), # size of the FIFO queue to store challenging words
    fifo_threshold=5, # minimum number of required guesses to add a word to the FIFO queue
    fifo_percentage=0.2, # percentage of words in the environment that should be devoted to FIFO words
):
    
    os.makedirs(save_dir, exist_ok=True) # make sure checkpoint dir exists
    writer = SummaryWriter(log_dir)
    
    for epoch in trange(start_epoch, num_epochs, desc="Training"):
        # Collect one full batch of trajectories from the batched environment
        traj = generate_batched_trajectories(
            batched_env,
            guess_list,
            answer_list,
            word_encodings,
            actor_critic,
            gamma=gamma,
            device=device,
            fifo_queue=fifo_queue,
            fifo_percentage=fifo_percentage
        )

        # Add hard solutions to FIFO queue
        # Meaning: words that took at least fifo_threshold guesses, whether they were found or not
        for answer, guesses in zip(traj["env_answers"], traj["num_guesses"]):
            if guesses >= fifo_threshold: # at least fifo_threshold guesses
                fifo_queue.append(answer)
                
        #print("FIFO queue:", fifo_queue) # helpful to print to monitor progress on reducing guesses

        # Normalize advantages for more stable PPO updates
        advantages_tensor = torch.tensor(traj["advantages"], dtype=torch.float32)
        mean, std = advantages_tensor.mean(), advantages_tensor.std()
        traj["advantages"] = [(a - mean) / (std + 1e-8) for a in traj["advantages"]]

        # Create dataset for PPO
        dataset = list(zip(
            traj["observations"],
            traj["actions"],
            traj["log_probs"],
            traj["returns"],
            traj["advantages"]
        ))

        dataset_size = len(dataset)
        indices = list(range(dataset_size))

        # PPO updates
        for ppo_epoch in range(ppo_epochs):
            random.shuffle(indices)

            for start_idx in range(0, dataset_size, minibatch_size):
                batch_indices = indices[start_idx:start_idx + minibatch_size]

                obs_batch = [dataset[i][0] for i in batch_indices]
                act_batch = [dataset[i][1] for i in batch_indices]
                logp_batch = [dataset[i][2] for i in batch_indices]
                ret_batch = [dataset[i][3] for i in batch_indices]
                adv_batch = [dataset[i][4] for i in batch_indices]

                ppo_update(
                    actor_critic,
                    optimizer,
                    obs_batch,
                    act_batch,
                    adv_batch,
                    ret_batch,
                    logp_batch,
                    word_encodings,
                    clip_epsilon=clip_epsilon,
                    value_loss_coef=value_loss_coef,
                    device=device,
                    writer=writer,
                    global_step=epoch * ppo_epochs + ppo_epoch
                )
        
        if epoch % eval_and_save_per == 0: # evaluate and save state
            print("Evaluating policy on all answers...")
            win_rate, avg_guesses = evaluate_policy_on_all_answers(
            env_class=BatchedWordleEnv,
            guess_list=guess_list,
            answer_list=answer_list,
            word_encodings=word_encodings,
            actor_critic=actor_critic)
            writer.add_scalar("Eval/win_rate", win_rate, epoch)
            writer.add_scalar("Eval/avg_guesses", avg_guesses, epoch)
            
            print(f"Saving model state for epoch {epoch}...")
            
            # save model state
            checkpoint = {
                "model": actor_critic.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pth"))
            print("Model state saved!")
            
    # close writer
    writer.close()
