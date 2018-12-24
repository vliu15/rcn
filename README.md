# Recurrent Control Nets as Central Pattern Generators
In our [previous repository](https://github.com/vliu15/CPG-RL), we explored various models that could improve upon the baseline Multilayer Perceptron policy baseline and found that Recurrent Neural Networks work well. In this repository, we provide our code for our Recurrent Control Net, which beats Multilayer Perceptron and Structured Control Net baselines.

## Usage
We provide scripts to design and train models. Upon exit, the weights of that training session are automatically saved. Below are run commands to render the environment with pretrained weights as well as log episodic reward during training.

To setup, see the [CPG-RL](https://github.com/vliu15/CPG-RL) repository for installing dependencies for MuJoCo.
```
# clone repository
git clone https://github.com/vliu15/RNN-CPG
cd RNN-CPG
```

### Training
Weights from training are automatically saved as `.pkl` files in `weights`.
```
# train on environments
python3 run.py --model rcn \
    --env HalfCheetah-v2 \
    --num_timesteps 2000000
```

### Logging
In data collecting mode, the logs are written to `.csv` files in `data`.
```
# log training for plotting
python3 run.py --model rcn \
    --env HalfCheetah-v2 \
    --num_timesteps 2000000 \
    --collect_data
```

### Rendering
```
# render with pretrained weights
python3 run.py --model rcn \
    --env HalfCheetah-v2 \
    --num_timesteps 2000000 \
    --render \
    --weights_file /path/to/weights_file
```

### Plotting
Plots are written to `.png` files in `plots`.
```
# log episodic reward vs timestep, env is mandatory
python3 plot.py --env HalfCheetah-v2 \
    --avg_window 100
```

## Environment
We continue to use MuJoCo v2 environments with OpenAI Gym as our means of testing our models. We find that MuJoCo provides a variety of different locomotive tasks that force a model to learn movements along different axes requiring different amounts of complexities.

## Optimizer
We use Evolutionary Strategies as our optimization algorithm, as training through disturbance by random Gaussian noise has been proven to be very effective. We do not use another top alternative, OpenAI's Proximal Policy Optimization, because of its inflexibility to Recurrent Neural Networks (and extremely poor documentation).

## Recurrent Control Net (RCN)
Building off the idea of linear and nonlinear control modules outlined in [this](https://arxiv.org/abs/1802.08311) paper, the Recurrent Control Net uses a vanilla RNN as the nonlinear module and a simple linear mapping as the linear module. The RCN has:
- Nonlinear module:
  - 3 weight kernels
  - 2 bias vectors, should the `n_use_bias` flags be set to `True`
  - Layer activations after each kernel mapping
- Linear module:
  - 1 weight kernel
  - 1 bias vector, should the `l_use_bias` flag be set to `True`
  
We find that not using biases yields best results with Evolutionary Strategies as the training algorithm.

## Citation
To cite this repository in publications:
```
@misc{RCN,
    author={Liu, Vincent},
    contributors={Adeniji, Ademi and Lee, Nate and Zhao, Jason},
    title={Recurrent Control Nets as Central Pattern Generators},
    year={2018},
    publisher={GitHub},
    journal={GitHub repository},
    howpublished={\url{https://github.com/vliu15/RecurrentControlNets}},
}
```
