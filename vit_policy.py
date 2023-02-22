from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import gym
import torch 
import math
from torch import nn 
import torch.nn.functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from small_env_img import * 
from vit import * 



#register_small_env('small-v0','tree',hold_out=10)







class TransformerFeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box,attn_dim=64,num_heads=8,num_layers=3,conv=0,dropout=0.):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(TransformerFeatureExtractor, self).__init__(observation_space, features_dim=163)
        self.conv=conv 
        self.attn_dim=attn_dim 
        self.num_heads=num_heads
        self.num_layers=num_layers 
        self.dropout=dropout
        if conv>0:
            self.visual_cnn = nn.Sequential(
                nn.Conv2d(3, conv, kernel_size=2, stride=1, padding=0),
                nn.ReLU()
            )
            self.transformer=ViT(6,1,attn_dim,num_layers,num_heads,channels=conv,dropout=dropout,emb_dropout=dropout) 
        else:
            self.transformer=ViT(7,1,attn_dim,num_layers,num_heads,channels=3,dropout=dropout,emb_dropout=dropout)
        # Update the features dim manually
        self._features_dim = self.attn_dim  

    def forward(self, observations) -> torch.Tensor: 
        #observations=torch.from_numpy(observations).float()
        if self.conv>0:
            conv_out=self.visual_cnn(observations)
            output=self.transformer(conv_out)
        else:
            output=self.transformer(observations)
        return output 
        
    

if __name__=='__main__':
    
    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    extractor=TransformerFeatureExtractor(env.observation_space,conv=10)

    obs=env.reset()
    print(obs.shape)
    env.step([0]) 
    obs=env.step([2])[0]
    
    print(extractor(obs).shape)
    print(obs.shape)












        