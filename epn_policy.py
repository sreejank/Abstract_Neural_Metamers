from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
import gym
import torch 
import math
from torch import nn 
import torch.nn.functional as F
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from episodic_env import * 
from neko_fixed_torch_transformer import neko_MultiheadAttention

#register_small_env('small-v0','tree',hold_out=10)

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=200):
        super(PositionalEncoding,self).__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        div_term=torch.exp(torch.arange(0,d_model,2).float()*(-math.log(10000.0)/d_model))
        pe[:,0::2]=torch.sin(position*div_term)
        if d_model%2==0:
            pe[:,1::2]=torch.cos(position*div_term)
        else:
            pe[:,1::2]=torch.cos(position*div_term[:-1])
        pe=pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self,x):
        return x+self.pe[:,:x.size(1),:]







class EPNFeatureExtractor(BaseFeaturesExtractor):
    

    def __init__(self, observation_space: gym.spaces.Box,attn_dim=64,memsize=40,num_heads=8):
        # TODO we do not know features-dim here before going over all the items, so put something there. This is dirty!
        super(EPNFeatureExtractor, self).__init__(observation_space, features_dim=163)

        self.visual_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(), 
            nn.Flatten()
        )
        self.visual_linear=nn.Sequential(nn.Linear(400, 49), nn.ReLU())

        self.attn_dim=attn_dim
        self.memsize=memsize 
        self.num_heads=num_heads 

        self.attn_proj=nn.Linear(196,attn_dim)
        self.layer_norm=nn.LayerNorm((self.memsize,self.attn_dim))
        self.pos_enc=PositionalEncoding(self.attn_dim)
        #self.attention=nn.MultiheadAttention(self.attn_dim,self.num_heads,batch_first=True)
        self.attention=neko_MultiheadAttention(self.attn_dim,self.num_heads,batch_first=True)


        self.shared_mlp = nn.Sequential(
            nn.Linear(self.attn_dim, self.attn_dim),
            nn.ELU(),
            nn.Linear(self.attn_dim, self.attn_dim),
            nn.ELU(),
        ) 

        # Update the features dim manually
        self._features_dim = 49 + 50 + self.attn_dim

    def forward(self, observations) -> torch.Tensor: 
        curr_board=(observations['state'])[:,:49]
        visual_input=torch.reshape(curr_board,(-1,1,7,7)).float()
        prev_outputs=(observations['state'][:,49:])
        visual_output=self.visual_linear(self.visual_cnn(visual_input))

        memory=(observations['memory']).float()
        memory_mask=torch.gt((1.0-(observations['memory_mask'])),0).reshape((-1,memory.shape[1]))
        #print(memory.shape,memory_mask.shape) 
        pre_proj_attn=torch.cat([memory,curr_board.unsqueeze(1).repeat(1,memory.shape[1],1)],dim=2)
        #print(pre_proj_attn.shape) 

        encoded_input=self.pos_enc(self.attn_proj(pre_proj_attn))
        attn_input=self.layer_norm(encoded_input)
        attn_out,attn_weights=self.attention(attn_input,attn_input,attn_input,key_padding_mask=memory_mask)
        attn_out=F.elu(attn_out+encoded_input)

        attn_out=self.shared_mlp(attn_out) 
        attn_out=torch.max(attn_out,dim=1).values

        combined_output=torch.cat((visual_output,prev_outputs,attn_out),dim=1)
        
        #print(combined_output)
        #print('-----')
        #print(attn_out)

        return combined_output 
    

if __name__=='__main__':
    
    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    extractor=EPNFeatureExtractor(env.observation_space)

    obs=env.reset()
    #env.step([0]) 
    #obs=env.step([2])[0]
    
    print(extractor(obs))











        