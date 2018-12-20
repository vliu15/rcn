# Recurrent Neural Networks as Central Pattern Generators
In our [previous repository](https://github.com/vliu15/CPG-RL), we explored various models that could improve upon the baseline Multilayer Perceptron policy baseline. We found that Recurrent Neural Networks work pretty well and intend to explore the recurrent architecture in depth in this repository. Additionally, the addition of gates and enhanced memory (i.e. Gated Recurrent Units, Long Short-Term Memories) decreased performance in these tasks. As a result, we only focus on vanilla Recurrent Neural Networks.

## Environment
We continue to use MuJoCo environments with OpenAI Gym as our means of testing our models. We find that MuJoCo provides a variety of different locomotive tasks that force a model to learn movements along different axes requiring different amounts of complexities.

## Optimizer
We use Evolutionary Strategies as our optimization algorithm, as training through disturbance by random Gaussian noise has been proven to be very effective. We do not use another top alternative, Proximal Policy Optimization, because of its inflexibility to Recurrent Neural Networks.

## Models
This repository is dedicated to exploring the efficacy of Recurrent Neural Networks. We have the following models, the hyperparameters to which are in `config.py`.

### Vanilla Recurrent Neural Network (RNN)
The most basic Recurrent Neural Network, this model is our baseline (with hidden size 32). The RNN has:
- 3 weight kernels
- 2 bias vectors, should the `use_bias` flag be set to `True`

### Recurrent Control Net (RCN)
Building off the idea of linear and nonlinear control modules outlined in [this](https://arxiv.org/abs/1802.08311) paper, the Recurrent Control Net uses a vanilla RNN as the nonlinear module and a simple linear mapping as the linear module. The RCN has:
- 4 weight kernels
- 3 bias vectors (2 nonlinear control, 1 linear control), set the `use_bias` flags be set to `True`
