# Abstract_Neural_Metamers
Code and Data for paper "Disentangling Abstraction from Statistical Pattern Matching in Human and Machine Learning" (under review). 

This is the initial version that contains for generating abstract task distributions, building metamer task distributions, training the RL agents, and behavioral data (see notes in each file for more information). The next version will contain more extensive documentation on use and cleaner more user friendly code. Code for a paper with preliminary findings that uses a lot of the same methods is available here: https://github.com/sreejank/structured_metarl 

Current files:

abstract_task_distributions.py: Generating abstract task distributions

metamer_distribution_generator.py: Generating metamer task distributions

small_env.py: Grid training enviornment for RL agent (first set of experiments that use the eight chosen abstractions)

small_env_4x4.py: Grid training enviornment for RL agent (second set of experiments that use the data-driven Gibbs Sampling with People abstract task distribution)

train.py: Training code for RL agent in first set of experiments (goes with small_env.py)

train_4x4.py: Training code for RL agent in second set of experiments (goes with small_env_4x4.py)

data/hyperparams_<rules>.pkl: Hyperparameters for a particular task distribution. Add "_null" to use the metamer distribution. 
  
data/<rules>_sample.npy | data/<rules>_sample_starts.npy: Held-out test set for each task distribution (and the start tile location). Used in human experiments as well. 
  
data/performancedata.csv: Behavioral data of human and agent performance across all conditions (all abstractions, metamer/abstract conditions)
  
