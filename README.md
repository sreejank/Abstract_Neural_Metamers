# Abstract_Neural_Metamers
Code and Data for paper "Disentangling Abstraction from Statistical Pattern Matching in Human and Machine Learning" (under review). 

This is the initial version that contains for generating abstract task distributions, building metamer task distributions, training the RL agents, and behavioral data (see notes in each file for more information). The next version will contain more extensive documentation on use and cleaner more user friendly code. Code for a paper with preliminary findings that uses a lot of the same methods is available here: https://github.com/sreejank/Compositional_MetaRL. 

Current files:

abstract_task_distributions.py: Generating abstract task distributions

metamer_distribution_generator.py: Generating metamer task distributions

small_env.py: Grid training enviornment for RL agent (first set of experiments that use the eight chosen abstractions)

episodic_env.py: Same grid training environment, but keeps track of a memory buffer (n prev timesteps) for the Episodic Planning Network agent 

train.py: Training code for Recurrent Meta-RL agent in first set of experiments (goes with small_env.py)

train_epn.py: Training code for EPN agent in (goes with episodic_env.py)

train_corelnet2.py: Training code for CorelNet agent. 

train_transformer.py: Training code for Transformer Agent. 

epn_policy.py: Implementation of EPN agent

vit_policy.py: Implementation of Transformer agent

corelnet_policy.py: Implementation of CorelNet agent. 

vit.py: Helper class for Transformer agent (implementation of self-attention)

data/hyperparams_<rules>.pkl: Hyperparameters for a particular task distribution. Add "_null" to use the metamer distribution. 
  
data/<rules>_sample.npy | data/<rules>_sample_starts.npy: Held-out test set for each task distribution (and the start tile location). Used in human experiments as well. 

data/gsp_4x4_full.npy | data/gsp_4x4_full_probs.npy: Boards produced from GSP experiments and their counts (how much they occured in the chain). 
  
data/performancedata.csv: Behavioral data of human and agent performance across all conditions (all abstractions, metamer/abstract conditions)


