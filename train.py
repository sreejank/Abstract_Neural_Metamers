"""
Meta-RL agent training code for first set of experiments (7x7 boards). 
data/<rules>_sample.npy and data/<rules>_sample_starts.npy (add _null after rule name for metamer version) contains the held-out test set. This test set was also used for the human experiments. 
data/performancedata.csv contains the performance of the agent (as well as humans) on these held-out test sets. 
"""
import optuna
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.a2c import A2C
from stable_baselines.common import set_global_seeds, make_vec_env
from small_env import *
import sys
from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines.common.tf_layers import conv,conv_to_fc
from stn import spatial_transformer_network as transformer 
import pickle 
rules=sys.argv[2] 
yes_cnn=int(sys.argv[3]) 
yes_lstm=int(sys.argv[4]) 

register_small_env('small-v0',rules,hold_out=10,pretrain=0)  
#register_small_env('test-v0',rules,hold_out=-1,pretrain=0,max_episode_steps=120)  
register_small_env('test-v0',rules,hold_out=-1,pretrain=0,max_episode_steps=120)       
  
#register_trace_env('trace-v0',4,['chain'],20,use_precomputed=True)
#register_battleship_env('ship-v0',4,['chain'],20)

hyperparam_dict=pickle.load(open('data/hyperparams_'+rules+"_1_1.pkl",'rb'))

"""
gamma=0.9
n_steps=8
lr_schedule='linear'
lr=0.0023483181861598565
ent_coef=0.0006747109316677081
vf_coef=0.00635090082912515
num_layers=2
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



#num_conv_layers=hyperparam_dict['num_conv_layers']
#num_filters=hyperparam_dict['num_filters']
#num_fc_layers=hyperparam_dict['num_fc_layers']

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
    visual_input=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    visual_input=tf.reshape(visual_input,(-1,7,7,1))
    activ=tf.nn.relu

    layer_1 = activ(conv(visual_input, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)) 
    #print(layer_1.shape,visual_input.shape)
    #layer_2 = activ(conv(layer_1, 'c2', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #layer_3=conv_to_fc(layer_2)
    layer_2=conv_to_fc(layer_1)
    visual_output=activ(linear(layer_2,'fc1',n_hidden=49,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1)  
    return total_output 

"""
def cnn_skip_small(input_tensor,**kwargs):
    visual_input=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    visual_input=tf.reshape(visual_input,(-1,7,7,1))
    activ=tf.nn.relu
    conv_output=visual_input
    for i in range(num_conv_layers):
        conv_output = activ(conv(conv_output, 'c'+str(i), n_filters=num_filters, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))  

    flattened=conv_to_fc(conv_output)
    fc_output=flattened
    for i in range(num_fc_layers):
        fc_output=activ(linear(fc_output,'fc'+str(i),n_hidden=49,init_scale=np.sqrt(2)))
    visual_output=fc_output
    total_output=tf.concat([visual_output,prev_output],1)  
    return total_output
"""
def mlp_skip_small(input_tensor, **kwargs): 
    """
    

    :param input_tensor: (TensorFlow Tensor) Observation input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    net_arch=[49 for _ in range(num_layers)]
    visual_output=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    activ = tf.tanh
    for i, layer_size in enumerate(net_arch):
        visual_output = activ(linear(visual_output, 'pi_fc' + str(i), n_hidden=layer_size,init_scale=np.sqrt(2)))
    total_output=tf.concat([visual_output,prev_output],1) 
    #total_output=visual_output
    return total_output 

def stn_skip_small(input_tensor,return_fixation=False,**kwargs): 
    visual_input=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    activ=tf.nn.relu



    layer_1 = activ(conv(tf.reshape(visual_input,(-1,7,7,1)), 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)) 
    layer_2=conv_to_fc(layer_1)
    layer_3=activ(linear(layer_2,'fc0',n_hidden=24,init_scale=np.sqrt(2))) 
    
    initial = np.array([[1., 1e-6, 1e-6], [1e-6, 1., 1e-6]]) 
    initial = initial.astype('float32').flatten()

    W_fc1=tf.Variable(tf.constant(1e-6,dtype=tf.float32,shape=[24,6]),name='W_fc1',trainable=True)
    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1',trainable=True)
    h_fc1 = tf.matmul(layer_3, W_fc1) + b_fc1
    transformed_input=transformer(tf.reshape(visual_input,(-1,7,7,1)),h_fc1,(5,5))
    transformed_input_named=tf.identity(tf.reshape(transformed_input,(-1,25)),name='stn_output') 

    #layer_1 = activ(conv(transformed_input, 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #layer_2=conv_to_fc(layer_1)
    visual_output=activ(linear(transformed_input_named,'fc2',n_hidden=49,init_scale=np.sqrt(2))) 
    total_output=tf.concat([visual_output,prev_output],1)  
    if return_fixation:
        return transformed_input,total_output 
    else:
        return total_output 



def gaussian_mask(params, R, C):
    """Define a mask of size RxC given by one 1-D Gaussian per row.
    u, s and d must be 1-dimensional vectors"""
    u, s, d = (params[..., i] for i in range(3))

    for i in (u, s, d):
        assert len(u.get_shape()) == 1, i

    batch_size = tf.to_int32(tf.shape(u)[0])

    R = tf.range(tf.to_int32(R))
    C = tf.range(tf.to_int32(C))
    R = tf.to_float(R)[tf.newaxis, tf.newaxis, :]
    C = tf.to_float(C)[tf.newaxis, :, tf.newaxis]
    C = tf.tile(C, (batch_size, 1, 1))

    u, d = u[:, tf.newaxis, tf.newaxis], d[:, tf.newaxis, tf.newaxis]
    s = s[:, tf.newaxis, tf.newaxis]

    ur = u + (R - 0.) * d
    sr = tf.ones_like(ur) * s

    mask = C - ur
    mask = tf.exp(-.5 * (mask / sr) ** 2)

    mask /= tf.reduce_sum(mask, 1, keep_dims=True) + 1e-8
    return mask


def gaussian_glimpse(inpt, attention_params, glimpse_size):
    """Extracts an attention glimpse
    :param inpt: tensor of shape == (batch_size, img_height, img_width)
    :param attention_params: tensor of shape = (batch_size, 6) as
        [uy, sy, dy, ux, sx, dx] with u - mean, s - std, d - stride"
    :param glimpse_size: 2-tuple of ints as (height, width),
        size of the extracted glimpse
    :return: tensor
    """

    ap = attention_params
    shape = inpt.get_shape()
    rank = len(shape)

    assert rank in (3, 4), "Input must be 3 or 4 dimensional tensor"

    inpt_H, inpt_W = shape[1:3]
    if rank == 3:
        inpt = inpt[..., tf.newaxis]
        rank += 1
    
    Fy = gaussian_mask(ap[..., 0:3], glimpse_size[0], inpt_H)
    Fx = gaussian_mask(ap[..., 3:6], glimpse_size[1], inpt_W)

    gs = []
    for channel in tf.unstack(inpt, axis=rank - 1):
        g = tf.matmul(tf.matmul(Fy, channel, adjoint_a=True), Fx)
        gs.append(g)
    g = tf.stack(gs, axis=rank - 1)

    
    return g



def gaussian_skip_small(input_tensor,**kwargs):
    visual_input=tf.slice(input_tensor,[0,0],[-1,49],name='input_img') 
    prev_output=tf.slice(input_tensor,[0,49],[-1,50],'prev_outputs')
    activ=tf.nn.relu



    layer_1 = activ(conv(tf.reshape(visual_input,(-1,7,7,1)), 'c1', n_filters=16, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs)) 
    layer_2=conv_to_fc(layer_1)
    layer_3=activ(linear(layer_2,'fc0',n_hidden=24,init_scale=np.sqrt(2))) 
    layer_4=activ(linear(layer_3,'fc01',n_hidden=12,init_scale=np.sqrt(2))) 
    
    #Before: 1,.5,1 1,.5,1
    initial = np.array([[1., 0.5, 1.], [1., 0.5, 1.]]) 
    initial = initial.astype('float32').flatten()

    W_fc1=tf.Variable(tf.constant(1e-6,dtype=tf.float32,shape=[12,6]),name='W_fc1',trainable=True) 
    b_fc1 = tf.Variable(initial_value=initial, name='b_fc1',trainable=True)
    gaussian_params = tf.matmul(layer_4, W_fc1) + b_fc1

    transformed_input=gaussian_glimpse(tf.reshape(visual_input,(-1,7,7)),gaussian_params,(4,4))
    transformed_input_named=tf.identity(tf.reshape(transformed_input,(-1,16)),name='gaussian_output') 

    visual_output=activ(linear(transformed_input_named,'fc2',n_hidden=49,init_scale=np.sqrt(2))) 
    total_output=tf.concat([visual_output,prev_output],1)  
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
for i in range(7):
        for j in range(7):
                action_converter.append((i,j)) 

if __name__=='__main__':
    num_episodes=int(sys.argv[1])

    env=make_vec_env('small-v0',n_envs=1,vec_env_cls=SubprocVecEnv) 
    

 
    if yes_cnn:
        if yes_lstm:
            policy_kwargs={'cnn_extractor':cnn_skip_small,'n_lstm':n_lstm}  
            model=A2C(policy='CnnLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
                n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
        else:
            policy_kwargs={'cnn_extractor':cnn_skip_small}
            model=A2C(policy='CnnPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
                n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
    else:
        if yes_lstm:
            policy_kwargs={'cnn_extractor':mlp_skip_small,'n_lstm':n_lstm}  
            model=A2C(policy='CnnLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
                n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
        else:
            policy_kwargs={'cnn_extractor':mlp_skip_small}
            model=A2C(policy='CnnPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
                n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
    policy_kwargs={'net_arch':['lstm'],'n_lstm':n_lstm} 
    model=A2C(policy='MlpLstmPolicy',policy_kwargs=policy_kwargs,tensorboard_log='test_log',env=env,gamma=gamma,
        n_steps=n_steps,lr_schedule=lr_schedule,learning_rate=lr,ent_coef=ent_coef,vf_coef=vf_coef,verbose=True)
    
    model.save("data/7x7_"+rules+"_"+str(yes_cnn)+"_"+str(yes_lstm)+"_2e6_metalearning.zip")    



     
