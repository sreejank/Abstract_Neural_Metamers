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




class CoRelNet2FeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CoRelNet2FeatureExtractor, self).__init__(observation_space,features_dim=163)
        self.n_seq=49
        self.mlp_decoder=nn.Sequential(nn.Flatten(),nn.Linear(2401, 256), nn.ReLU(),nn.Linear(256, 256))

        self.rearrange=Rearrange('b c h w -> b (h w) c')

        self._features_dim = 256

    def forward(self, observations) -> torch.Tensor: 
        #observations=torch.from_numpy(observations).float()
        b,c,h,w=observations.shape 
        x=self.rearrange(observations)
        R=F.softmax(torch.matmul(x,x.transpose(2,1)),dim=2)
        assert R.shape==(b,h*w,h*w )
        output=self.mlp_decoder(R)
        return output 
        


class CoRelNetFeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box,num_layers=2,num_heads=7,dropout=0.):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(CoRelNetFeatureExtractor, self).__init__(observation_space,features_dim=163)
        self.attn_dim=49 
        self.n_seq=49
        self.num_heads=num_heads 
        self.num_layers=num_layers
        self.transformer=SimpleTransformer(self.attn_dim,self.n_seq,self.num_layers,self.num_heads,dropout=dropout,emb_dropout=dropout)

        self.rearrange=Rearrange('b c h w -> b (h w) c')

        self._features_dim = self.attn_dim  

    def forward(self, observations) -> torch.Tensor: 
        #observations=torch.from_numpy(observations).float()
        b,c,h,w=observations.shape 
        x=self.rearrange(observations)
        R=F.softmax(torch.matmul(x,x.transpose(2,1)),dim=2)
        assert R.shape==(b,h*w,h*w )
        output=self.transformer(R)
        return output 
        
    

if __name__=='__main__':
    
    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    extractor=CoRelNet2FeatureExtractor(env.observation_space)

    obs=env.reset()
    print(obs.shape)
    env.step([0]) 
    obs=env.step([2])[0]
    
    print(extractor(obs).shape)
    print(obs.shape)












        