# Recurrent Neural Networks as Central Pattern Generators
In our [previous repository](https://github.com/vliu15/CPG-RL), we explored various models that could improve upon the baseline Multilayer Perceptron policy baseline. We found that Recurrent Neural Networks work pretty well and intend to explore the recurrent architecture in depth in this repository. Additionally, the addition of gates and enhanced memory (i.e. Gated Recurrent Units, Long Short-Term Memories) decreased performance in these tasks. As a result, we only focus on vanilla Recurrent Neural Networks.

## Usage
See the [CPG-RL](https://github.com/vliu15/CPG-RL) repository for installing dependencies for MuJoCo.
```
# clone repository
git clone https://github.com/vliu15/RNN-CPG
cd RNN-CPG

# train on environments
python3 run.py --model rnn \
    --env Swimmer-v2 \
    --num_timesteps 2000000
```
See `utils/cli_parser.py` for all command line arguments for training.

## Environment
We continue to use MuJoCo v2 environments with OpenAI Gym as our means of testing our models. We find that MuJoCo provides a variety of different locomotive tasks that force a model to learn movements along different axes requiring different amounts of complexities.

## Optimizer
We use Evolutionary Strategies as our optimization algorithm, as training through disturbance by random Gaussian noise has been proven to be very effective. We do not use another top alternative, OpenAI's Proximal Policy Optimization, because of its inflexibility to Recurrent Neural Networks (and extremely poor documentation).

## Models
This repository is dedicated to exploring the efficacy of Recurrent Neural Networks. We have the following models, the hyperparameters to which are in `config.py`.

### Vanilla Recurrent Neural Network (RNN)
The most basic Recurrent Neural Network, this model is our baseline (with hidden size 32). The RNN has:
- 3 weight kernels
- 2 bias vectors, should the `use_bias` flag be set to `True`
- Layer activations after each kernel mapping

### Recurrent Control Net (RCN)
Building off the idea of linear and nonlinear control modules outlined in [this](https://arxiv.org/abs/1802.08311) paper, the Recurrent Control Net uses a vanilla RNN as the nonlinear module and a simple linear mapping as the linear module. The RCN has:
- Nonlinear module:
  - 3 weight kernels
  - 2 bias vectors, should the `n_use_bias` flags be set to `True`
  - Layer activations after each kernel mapping
- Linear module:
  - 1 weight kernel
  - 1 bias vector, should the `l_use_bias` flag be set to `True`

### Time-Delay Neural Network (TDNN)
Inspired (and taken) from the Time-Delay Neural Networks used in [Deep Speech 2](https://arxiv.org/abs/1512.02595) for speech recognition, the TDNN performs a 1-D convolution along the time axis across past observations. This aims to learn patterns in past observations (at different levels of granularity). The TDNN has:
- Convolutional kernels (# set in `config.py`): operate without padding and change the number of channels
- Corresponding bias vectors, should the `use_bias` flag be set to `True`
- Layer activations after each convolution

### Time-Delay Control Net (TDCN)
Building off the idea of linear and nonlinear control modules outlined in [this](https://arxiv.org/abs/1802.08311) paper, the Time-Delay Control Net uses a TDNN as the nonlinear module and a simple linear mapping as the linear module. The TDCN has:
- Nonlinear module:
  - Convolutional kernels (# set in `config.py`): operate without padding and change the number of channels
  - Corresponding bias vectors, should the `n_use_bias` flag be set to `True`
  - Layer activations after each convolution
- Linear module:
  - 1 weight kernel
  - 1 bias vector, should the `l_use_bias` flag be set to `True`
