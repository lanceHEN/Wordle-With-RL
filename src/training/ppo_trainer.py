import torch
import torch.nn.functional as F

def ppo_update(
    actor_critic, # WordleActorCritic model
    optimizer, # optimizer
    observations, # list of observations
    actions, # list of actions (indices for the guess list)
    advantages, # difference between actual and expected returns
    returns, # actual returns
    old_log_probs, # log probabilities for the actions at the time of the episodes
    word_encodings, # word encodings
    clip_epsilon=0.2, # clip epsilon for PPO - how tightly to clamp the new vs old prob ratios
    value_loss_coef=0.5, # coefficient for value loss
    device="cpu", # device
    writer=None, # writer for logging PPO loss
    global_step=None, # step in the overarching model training
):
    """
    Performs a single PPO update step over the given trajectory, with the given observations, rewards, etc to batch.
    For positive advantage (scoring higher than expected), we want to encourage a higher probability of that action.
    For negative advantage (scoring lower than expected), we want to encourage a lower probability of that action.
    """
    # Clear gradients explicitly
    optimizer.zero_grad(set_to_none=True)
    
    # Turn lists into tensors
    actions = torch.as_tensor(actions, device=device)
    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.tensor(returns, dtype=torch.float32, device=device)
    old_log_probs = torch.stack(old_log_probs).to(device)

    # Forward pass
    logits, values = actor_critic(observations, word_encodings)
    #print("[ppo_update] after policy_head: query-related logits shape:", logits.shape)

    log_probs = F.log_softmax(logits, dim=-1)
    taken_log_probs = log_probs[torch.arange(len(actions), device=device), actions]

    # PPO loss
    ratios = torch.exp(taken_log_probs - old_log_probs.detach()) # new vs old
    #print(torch.max(logits, dim=1))
    #print(taken_log_probs)
    #print(old_log_probs)
    clipped_ratios = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon)
    policy_loss = -torch.min(ratios * advantages, clipped_ratios * advantages).mean()

    value_loss = F.mse_loss(values.view(-1), returns.view(-1))
    
    total_loss = policy_loss + value_loss_coef*value_loss
    #print(total_loss)
    
    if writer: # log the losses to tensorboard, if writer given
        writer.add_scalar("Loss/policy", policy_loss.item(), global_step)
        writer.add_scalar("Loss/value", value_loss.item(), global_step)

    # Backward + optimize
    #torch.autograd.set_detect_anomaly(True) # for debugging
    #print("[ppo_update] about to call backward()")
    total_loss.backward()
    
    optimizer.step()
