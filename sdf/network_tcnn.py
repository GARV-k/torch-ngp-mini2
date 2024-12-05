import torch
import torch.nn as nn
import torch.nn.functional as F

import tinycudann as tcnn
from hash_imports.encodings import HashEncoding


class SDFNetwork(nn.Module):
    def __init__(self,
                 tcnn_network,
                 encoding="hashgrid",
                 num_layers=3,
                 skips=[],
                 hidden_dim=64,
                 clip_sdf=None,
                 hashmap_size = 12,
                 ):
        super().__init__()


        self.num_layers = num_layers
        self.skips = skips
        self.hidden_dim = hidden_dim
        self.clip_sdf = clip_sdf
        
        assert self.skips == [], 'TCNN does not support concatenating inside, please use skips=[].'

        # self.encoder = tcnn.Encoding(
        #     n_input_dims=3,
        #     encoding_config={
        #         "otype": "HashGrid",
        #         "n_levels": 16,
        #         "n_features_per_level": 2,
        #         "log2_hashmap_size": 19,
        #         "base_resolution": 16,
        #         "per_level_scale": 1.3819,
        #     },
        # )
        
        self.encoder = HashEncoding(
            num_levels=16,
            min_res= 16,
            # max_res= 512,
            log2_hashmap_size=hashmap_size,
            features_per_level=2,
            hash_init_scale=0.001,
            implementation= "torch",
            # interpolation: Optional[Literal["Nearest", "Linear", "Smoothstep"]] = "Linear",
            interpolation="Linear",
        )

        # self.backbone = tcnn.Network(
        #     n_input_dims=32,
        #     n_output_dims=1,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "None",
        #         "n_neurons": hidden_dim,
        #         "n_hidden_layers": num_layers - 1,
        #     },
        # )
        self.backbone = tcnn_network   

    
    def forward(self, x,mode='normal',W=None,N=None,M=None,B1=None,B2=None):   
        # x: [B, 3]

        x = (x + 1) / 2 # to [0, 1]
        # if N != None:
        x = self.encoder(x,mode,W,N,M,B1,B2)
        
        # else:
        #     x = self.encoder(x)
        h = self.backbone(x)

        if self.clip_sdf is not None:
            h = h.clamp(-self.clip_sdf, self.clip_sdf)

        return h
