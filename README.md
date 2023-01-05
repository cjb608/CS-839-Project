# Overview
This is the codebase for Cameron's CS839 final project. The code uses the PPO reinforcement learning 
model from Stable Baselines to train an agent how to draw simple objects that can be constructed with only 
two straight lines. These objects consist of the following: '-', '|', '\', '/', '^', '<', '>', and '+'. 

# Necessary Python Packages
The following python packages are needed to run this code:
1. gym
2. numpy
3. math
4. cv2
5. matplotlib
6. argparse
7. stable_baselines3

# Running Code
## Training
In order to train this model, run the command `python train.py` along with the following arguments as needed.
When an argument is not provided, the default value will be used.

`-env <environment>` : Specify what environment to use for training. i.e., '-env baseline' will train the model using 
the baseline environment. The available environments are baseline or modified. Default='baseline'.
The modified environment corresponds to the reward function modification discussed in the Final Report. 

`-lr <learning rate>` : The learning rate to be used for training. Default=0.001.

`-n_steps <n steps>` : The number of steps to run per training update. Default=512.

`-batch_size <batch size>` : The minibatch size used when training. Default=64.

`-gamma <gamma>` : The discount factor used in training. Default=0.99.

`-tensorboard_log <file path>` : The file path for where to save the tensorboard log files. Default='./ppo_draw/'.

`-time_steps <time steps>` : The number of time steps to train the agent for. Default=1000000.

`-save_model_path <file path>` : The file path and name of the file for where to save the trained model. Default='model'

## Testing
In order to test a trained model, run the command python test.py along with the following arguments as needed.
When an argument is not provided, the default value will be used.

`-env <environment>` : Specify what environment was used to train the saved model and should be used for testing. 
i.e., '-env baseline' will test the trained model using the baseline environment. The available environments are 
baseline or modified. Default='baseline'. The modified environment corresponds to the reward function modification 
discussed in the Final Report.

`-load_model_path <file path>` : The file path and name of the file for where the trained model to be test is saved.
Default='model'.

# Recreating Experiments 
## Empirical Analysis
In order to recreate the empirical analysis that compares the baseline implementation to the modified reward function
implementation run the following commands to train each model.

`python train.py -env baseline -lr 0.001 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 2000000 
-save_model_path baseline_model`

`python train.py -env modified -lr 0.001 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 2000000 
-save_model_path modified_model`

## Sensitivity Analysis
In order to recreate the sensitivity analysis that studies the impact of the learning rate on the PPO algorithm run the 
following commands to train a model using each learning rate.

`python test.py -env baseline -lr 0.0001 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 1000000 
-save_model_path lr_0001_model`

`python test.py -env baseline -lr 0.0005 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 1000000 
-save_model_path lr_0005_model`

`python test.py -env baseline -lr 0.00075 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 1000000 
-save_model_path lr_00075_model`

`python test.py -env baseline -lr 0.0009 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 1000000 
-save_model_path lr_0009_model`

`python test.py -env baseline -lr 0.001 -n_steps 512 -batch_size 64 -gamma 0.99 -time_steps 1000000 
-save_model_path lr_001_model`
