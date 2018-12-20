## Evolutionary Strategy

This folder contains the models and run-scripts for training our models with Evolutionary Strategies in MuJoCo.

Models:
- `frs.py`: contains `LocomotorNet`
- `scn.py`: contains `StructuredControlNet`, `DeepStructuredControlNet`
- `mlp.py`: contains `MultilayerPerceptron`, `ParallelMultilayerPerceptron`
- `rnn.py`: contains `RecurrentNeuralNetwork`, `GatedRecurrentUnit`, `LongShortTermMemory`

Auxiliaries:
- `evostra.py`: Evolutionary Strategies algorithm, taken from [evostra](https://github.com/alirezamika/evostra)
- `config`: contains all model parameters

### Training
To train in MuJoCo, run the following script from the home directory:
```
cd ../
python3 -m es.run \
    --env Walker2d-v2 \         # mujoco environment
    --num_timesteps 1000000 \   # timesteps to train for
    --model scn \               # model to use as policy
    --print_steps 10            # interval to print during training
```
