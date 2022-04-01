"""
Meta-RL agent training code for second set of experiments (4x4 gsp boards). 
data/gsp_4x4_sample.npy and data/gsp_4x4_sample_starts.npy (add _null after rule name for metamer version) contains the held-out test set. This test set was also used for the human experiments. 
data/performancedata.csv contains the performance of the agent (as well as humans) on these held-out test sets. 
"""
import optuna
from stable_baselines.a2c import A2C
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common import set_global_seeds, make_vec_env
from small_env_4x4 import *
#from trace_env import * 
import sys
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.tf_layers import conv,conv_to_fc
from stn import spatial_transformer_network as transformer 

rules=sys.argv[2] 
register_small_env('small-v0',rules,hold_out=10,pretrain=0)  
#register_small_env('test-v0',rules,hold_out=-1,pretrain=0) 
register_small_env('test-v0',rules,hold_out=-1,pretrain=0)   
  
#register_trace_env('trace-v0',4,['chain'],20,use_precomputed=True)
#register_battleship_env('ship-v0',4,['chain'],20)
hyperparam_dict=pickle.load(open('data/hyperparams_'+rules+".pkl",'rb'))
""" 
gamma=0.9
n_steps=8
lr_schedule='linear'
lr=0.0023483181861598565 
ent_coef=0.0006747109316677081
vf_coef=0.00635090082912515
num_layers=3
#n_lstm=15 
n_lstm=120 
"""
gamma=hyperparam_dict['gamma']
n_steps=hyperparam_dict['n_steps']
lr_schedule=hyperparam_dict['lr_schedule']
lr=hyperparam_dict['lr']
ent_coef=hyperparam_dict['ent_coef']
vf_coef=hyperparam_dict['vf_coef']
num_layers=2 
#n_lstm=15 
#n_lstm=hyperparam_dict['n_lstm'] 
n_lstm=120


def mlp_skip(input_tensor, **kwargs): 
    """
    

    :param input_tensor: (TensorFlow Tensor) Observation input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    net_arch=[784 for _ in range(num_layers)]
    visual_output=tf.slice(input_tensor,[0,0],[-1,784],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,784],[-1,785],'prev_outputs')
    activ = tf.tanh
    for i, layer_size in enumerate(net_arch):
        visual_output = activ(linear(visual_output, 'pi_fc' + str(i), n_hidden=layer_size,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1) 
    #total_output=visual_output
    return total_output 

def cnn_skip_small(input_tensor,**kwargs):
    visual_input=tf.slice(input_tensor,[0,0],[-1,16],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,16],[-1,17],'prev_outputs')
    visual_input=tf.reshape(visual_input,(-1,4,4,1))
    activ=tf.nn.relu

    layer_1 = activ(conv(visual_input, 'c1', n_filters=16, filter_size=2, stride=1, init_scale=np.sqrt(2), **kwargs)) 
    #print(layer_1.shape,visual_input.shape)
    #layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #layer_3=conv_to_fc(layer_2)
    layer_2=conv_to_fc(layer_1)
    visual_output=activ(linear(layer_2,'fc1',n_hidden=16,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1)  
    return total_output 


def mlp_skip_small(input_tensor, **kwargs): 
    """
    

    :param input_tensor: (TensorFlow Tensor) Observation input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    net_arch=[16 for _ in range(num_layers)]
    visual_output=tf.slice(input_tensor,[0,0],[-1,16],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,16],[-1,17],'prev_outputs')
    activ = tf.tanh
    for i, layer_size in enumerate(net_arch):
        visual_output = activ(linear(visual_output, 'pi_fc' + str(i), n_hidden=layer_size,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1) 
    #total_output=visual_output
    return total_output 



def make_env(env_id, rank, seed=0,board=None):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :param board: (numpy array) pre-determined board for env. 
    """
    if board is not None:
        def _init(): 
            env = gym.make(env_id)
            env.seed(seed + rank)
            env.reset_task(board)
            return env 
    else:
        def _init():
            env = gym.make(env_id)
            env.seed(seed + rank)
            return env
    set_global_seeds(seed)
    return _init



action_converter=[]
for i in range(4):
        for j in range(4):
                action_converter.append((i,j)) 




if __name__=='__main__':
    num_episodes=int(sys.argv[1]) 

    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    

 

    policy_kwargs={'cnn_extractor':cnn_skip_small,'n_lstm':n_lstm}  
    model=A2C(policy='CnnLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
        n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
    policy_kwargs={'net_arch':['lstm'],'n_lstm':n_lstm} 
    model=A2C(policy='MlpLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
        n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
    
    model.learn(num_episodes)   
    print("Saving") 
    model.save("models/4x4_"+rules+"_convolutional_metalearning.zip")   
    

     
