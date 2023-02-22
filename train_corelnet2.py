from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import MaskablePPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecEnv
import torch.nn as nn 
import gym 
import torch 
from small_env_img import * 
import sys 
from corelnet_policy import * 
from stable_baselines3.common.sb2_compat.rmsprop_tf_like import RMSpropTFLike

rules=sys.argv[1]
register_small_env('small-v0',rules,hold_out=10,pretrain=0)
register_small_env('test-v0',rules,hold_out=-1,pretrain=0)

hyperparams_dict=pickle.load(open('data/hyperparams_corelnet2softmax_ppo_'+rules+'.pkl','rb'))
n_steps=hyperparams_dict['n_steps']
gamma=hyperparams_dict['gamma']
learning_rate=hyperparams_dict['learning_rate']
lr_schedule=hyperparams_dict['lr_schedule']
ent_coef=hyperparams_dict['ent_coef']
vf_coef=hyperparams_dict['vf_coef']

conv=0.0
clip_range_vf=None
batch_size=n_steps 
gae_lambda=1
n_epochs=1
clip_range=0.2
normalize_advantage=False 
max_grad_norm = 0.5


def linear_schedule(initial_value):
    """
    Linear learning rate schedule.
    :param initial_value: (float or str)
    :return: (function)
    """
    if isinstance(initial_value, str):
        initial_value = float(initial_value)

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0
        :param progress_remaining: (float)
        :return: (float)
        """
        return progress_remaining * initial_value

    return func
def make_small_env(n_envs=1,env_name='small-v0'):
    return make_vec_env(env_name,n_envs=n_envs,vec_env_cls=SubprocVecEnv)
env_fn=make_small_env


if lr_schedule=='linear':
    learning_rate=linear_schedule(learning_rate)

pk=dict(
        optimizer_class=RMSpropTFLike,
        optimizer_kwargs=dict(eps=1e-5),
        features_extractor_class=CoRelNet2FeatureExtractor,
        features_extractor_kwargs=dict()
)


action_converter=[]
for i in range(7):
        for j in range(7):
                action_converter.append((i,j)) 

if __name__=='__main__':
    params={
        'policy':'CnnPolicy',
        "n_steps": n_steps,
        "batch_size": batch_size,
        "gamma": gamma,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "clip_range": clip_range,
        'clip_range_vf':clip_range_vf,
        'normalize_advantage':normalize_advantage,
        "n_epochs": n_epochs,
        "gae_lambda": gae_lambda, 
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        # "sde_sample_freq": sde_sample_freq,
        'env':env_fn(n_envs=1,env_name='small-v0'),
        "policy_kwargs": pk,
        'verbose':1
    }
    env=params['env']
    num_episodes=8000000
    model=MaskablePPO(**params)
    print("Start")
    model.learn(num_episodes,log_interval=1)    
    print("Saving") 
    model.save("/scratch/gpfs/sreejank/corelnet2softmax_models/ppo_"+rules+"_metalearning_rep8.zip")
    print('loading')
    #model=model.load("/scratch/gpfs/sreejank/epn_models/ppo_"+rules+"_metalearning_rep.zip")

    obs=env.reset() 
    state=None
    done = [False for _ in range(env.num_envs)]

    reward_buffer=[]
    evals=[]
    num_evals=25
    tot_rewards=[]
    print("Begin")

    env.close()
    env=make_vec_env('test-v0',n_envs=1,vec_env_cls=SubprocVecEnv)
    obs=env.reset() 
    state=None
    done = [False for _ in range(env.num_envs)]
    episode_start=np.asarray([True for _ in range(env.num_envs)])
    num_evals=25
    print('Begin')
    raw_performance=np.zeros((15,25))
    mean_performance=[]
    raw_choices_total=[]
    test_boards=np.load('data/'+rules+"_sample.npy")  
    for i in range(15):
        reward_buffer=[]
        evals=[]
        tot_rewards=[]
        raw_choices_buffer=[]
        tot_raw_choices=[]
        while len(tot_rewards)<num_evals: 
            # We need to pass the previous state and a mask for recurrent policies
            # to reset lstm state when a new episode begin
            
            action, state = model.predict(obs, state=state, episode_start=episode_start,action_masks=env.env_method('action_masks'))
            if episode_start[0]:
                episode_start[0]=False 
            
            obs, reward , done, _ = env.step(action)
            reward_buffer.append(reward[0])
            raw_choices_buffer.append(action_converter[action[0]])

            if done[0]: 
                state=None 
                reward_array=np.asarray(reward_buffer)
                raw_choices_array=raw_choices_buffer[:]
                reward_buffer=[]
                raw_choices_buffer=[]
                episode_start[0]=True 


                #print(np.sum(reward_array<0))
                if reward[0]==10: 
                    print("Finished")
                    raw_performance[i,len(tot_rewards)]=np.sum(reward_array==-1)
                    tot_rewards.append(np.sum(reward_array==-1))
                    tot_raw_choices.append(raw_choices_array)
                else:
                    print("Didnt finish")
                    raw_performance[i,len(tot_rewards)]=49-test_boards[len(tot_rewards)].sum()
                    tot_rewards.append(49-test_boards[len(tot_rewards)].sum())
                    tot_raw_choices.append(raw_choices_array) 

        mean_performance.append(np.mean(tot_rewards))
        raw_choices_total.append(tot_raw_choices)  
    tot_rewards=np.asarray(mean_performance) 
    #np.save('data/raw_choices_ppo_heldout2_'+version+'_agent_'+str(rules)+'.npy',np.asarray(raw_choices_total))   
    #np.save('data/raw_performance_ppo_heldout2_'+version+'_agent_'+str(rules)+".npy",raw_performance)   
    np.save('data/raw_choices_corelnet2softmax_agent8_'+str(rules)+'.npy',np.asarray(raw_choices_total))  
    np.save('data/raw_performance_corelnet2softmax_agent8_'+str(rules)+".npy",raw_performance) 


