import torch
from models.wordle_actor_critic import WordleActorCritic

# A simple user-friendly model wrapper around a WordleActorCritic, (optionally) taking in WordleActorCritic,
# word list and word encodings and producing a guess word whenever given a particular
# observation. This is useful for actually playing the game in an interpretable manner (e.g. interacting with WordleView)
class ModelWrapper:
    def __init__(self, word_list, word_matrix, model=None, device='cpu'):
        if model is None:
            self.model = WordleActorCritic().to(device)
        else:
            self.model = model.to(device)
        self.model.eval()
        self.word_list = word_list
        self.word_matrix = word_matrix.to(device)
        self.device = device

    def get_guess(self, obs):
        # Prepare observation in batch form
        obs_batch = [obs]

        with torch.no_grad():
            logits, _ = self.model(obs_batch, self.word_matrix)

        best_idx = torch.argmax(logits[0]).item()
        return self.word_list[best_idx]
