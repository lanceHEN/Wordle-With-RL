import torch
import torch.nn as nn
from models.observation_encoder import ObservationEncoder
from models.shared_encoder import SharedEncoder
from models.letter_encoder import LetterEncoder

# Wrapper class that combines the ObservationEncoder and SharedEncoder modules into one, allowing a latent vector to be produced
# when given an observation.
# This can then be combined with either a PolicyHead or ValueHead.
# For convenience, the ObservationEncoder and SharedEncoder do not have to be given on construction, they can be made on construction
# if set to None
class ObservationSharedWrapper(nn.Module):
    
    def __init__(self, observation_encoder=None, shared_encoder=None):
        super().__init__()
        if observation_encoder is None:
            self.observation_encoder = ObservationEncoder(LetterEncoder())
        else:
            self.observation_encoder = observation_encoder
        
        if shared_encoder is None:
            self.shared_encoder = SharedEncoder()
        else:
            self.shared_encoder = shared_encoder
        
    # given batched observations, produces batched latent vector representations to be fed either into policy head or value head
    # in other words, this combines the forward pass for ObservationEncoder and SharedEncoder.
    def forward(self, batched_obs):
        # batched_obs: [B x 6 x 5 x letter_embed_dim + 3]
        encoded_grid, meta_tensor = self.observation_encoder(batched_obs)
        h = self.shared_encoder(encoded_grid, meta_tensor)
        return h # [B, output_dim]
