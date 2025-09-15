import torch
import torch.nn as nn
from models.letter_encoder import LetterEncoder

# Batching implementation of the Observation Encoder. Given batched observations (as a list), produces a batch of per-observation numerical representations
# friendly for inputs to a neural network
class ObservationEncoder(nn.Module):
    
    # Initializes an ObservationEncoder with a LetterEncoder and guess list size. The LetterEncoder can optionally be left unspecified,
    # and created with default params on construction.
    def __init__(self, letter_encoder=None, num_guess_words=14855):
        super().__init__()
        self.num_guess_words = num_guess_words

        # letter embeddings (learnable)
        self.letter_encoder = letter_encoder if letter_encoder is not None else LetterEncoder()

        # feedback will be one-hot, size 3: ("gray", "yellow", "green")
        self.feedback_dim = 3

        # final per-cell embedding size = letter + feedback dim
        self.embed_dim = self.letter_encoder.letter_embed_dim + self.feedback_dim

    # Given batched observations (as a list), produces a batch of per-observation numerical representations friendly for inputs to a neural network.
    # In particular, produces:
    # 1. Grid tensor: a [B x 6 x 5 x letter_embed_dim + 3] tensor, storing letter (learnable embeddings from the given LetterEncoder)
    # and feedback (one hot) data for every position in the game (filled or unfilled), for each in the batch. Rows for unfilled guesses have
    # all elements set to 0, to maintain constant dimensions.
    # 2. Meta tensor: a [B x 2] tensor storing the current turn and number of candidate words remaining (divided by total size of guess list) for each
    # game in the batch.
    def forward(self, obs_batch):
        device = next(self.parameters()).device
        batch_size = len(obs_batch)
    
        letter_tensor = torch.zeros((batch_size, 6, 5), dtype=torch.long, device=device)
        feedback_tensor = torch.zeros((batch_size, 6, 5, self.feedback_dim), dtype=torch.float32, device=device)
        meta_vector = torch.zeros((batch_size, 2), dtype=torch.float32, device=device) # [B, 2]
    
        fb_map = {"gray": 0, "yellow": 1, "green": 2}
    
        for b, obs in enumerate(obs_batch):
            for t, (word, feedback) in enumerate(obs["feedback"]):
                for i, (char, fb) in enumerate(zip(word, feedback)):
                    letter_tensor[b, t, i] = ord(char) - ord('a')
                    feedback_tensor[b, t, i, fb_map[fb]] = 1.0
            meta_vector[b, 0] = obs["turn_number"]
            meta_vector[b, 1] = len(obs["valid_indices"]) / self.num_guess_words

        letter_embs = self.letter_encoder(letter_tensor)  # [B, 6, 5, embed_dim]
        grid_tensor = torch.cat([letter_embs, feedback_tensor], dim=-1)  # [B, 6, 5, embed_dim + 3]
        return grid_tensor, meta_vector
