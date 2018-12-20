import gym
import random
import pickle as pickle
import numpy as np
import csv
import sys
import os

from rnn.base import RecurrentNeuralNetwork
from es.evostra import EvolutionStrategy
from config import map_str_model


class EvolutionStrategyWrapper(object):

    def __init__(self, args):
        """Initialize the ES-env structure."""
        # wrapper parameters
        self.AGENT_HISTORY_LENGTH = args.agent_history_len
        self.POPULATION_SIZE = args.population_size
        self.EPS_AVG = args.eps_avg
        self.SIGMA = args.sigma
        self.LEARNING_RATE = args.learning_rate
        self.INITIAL_EXPLORATION_PROB = args.initial_exploration_prob
        self.FINAL_EXPLORATION = args.final_exploration
        self.EXPLORATION_DEC_STEPS = args.exploration_dec_steps
        self.NUM_THREADS = args.num_threads
        self.PRINT_STEPS = args.print_steps
        self.MAX_TIMESTEPS = args.num_timesteps

        # to record timestep info
        self.timestep_rewards = []
        self.timesteps_recorded = 0

        # for self.exit() to use to save
        self.env_name = args.env
        self.model_name = args.model

        # es parameters
        self.env = gym.make(args.env)
        Model = map_str_model.get(args.model, RecurrentNeuralNetwork)
        print("Running {}...".format(Model.__name__))
        self.model = Model(self.env.observation_space.shape[0], self.env.action_space.shape[0])
        self.es = EvolutionStrategy(self.model.get_weights(), self.get_reward, self.POPULATION_SIZE,
                                    self.SIGMA, self.LEARNING_RATE, num_threads=self.NUM_THREADS)
        self.exploration = self.INITIAL_EXPLORATION_PROB

        if args.collect_data:
            path = os.path.join('data', 'env_{}_model_{}.csv'.format(args.env, args.model))
            if not os.path.exists('data'):
                os.makedirs('data')
            elif os.path.exists(path):
                raise Exception("File already exists.")
            self.CSV_SAVE_PATH = path
        else:
            self.CSV_SAVE_PATH = ''

    def get_predicted_action(self, sequence, current_timestep):
        # check if model is time dependent
        try:
            prediction = self.model.predict(np.array(sequence), current_timestep)
        except:
            prediction = self.model.predict(np.array(sequence))
        return prediction

    def load(self, filename='weights.pkl'):
        with open(filename, 'rb') as fp:
            self.model.set_weights(pickle.load(fp))
        self.es.weights = self.model.get_weights()

    def save(self, filename='weights.pkl'):
        with open(filename, 'wb') as fp:
            pickle.dump(self.es.get_weights(), fp)

    def train(self, iterations):
        self.timestep_rewards = []
        self.es.run(iterations, print_step=self.PRINT_STEPS)

    def render(self, episodes=100):
        """Play the agent for episodes."""
        self.model.set_weights(self.es.weights)

        for episode in range(episodes):
            total_reward = 0
            # get initial observation
            observation = self.env.reset()
            # fill the observation sequence AGENT_HISTORY_LENGTH times
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            while not done:
                self.env.render()
                action = self.get_predicted_action(sequence, 0)
                # get results of action
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                # shift observation sequence to include new one
                sequence = sequence[1:]
                sequence.append(observation)

            print("total reward: ", total_reward)

    def get_reward(self, weights):
        total_reward = 0.0
        self.model.set_weights(weights)

        for episode in range(self.EPS_AVG):
            observation = self.env.reset()
            sequence = [observation]*self.AGENT_HISTORY_LENGTH
            done = False
            current_timestep = 0
            while not done:
                # explore with small probability
                self.exploration = max(self.FINAL_EXPLORATION, self.exploration -
                                       self.INITIAL_EXPLORATION_PROB/self.EXPLORATION_DEC_STEPS)
                if random.random() < self.exploration:
                    action = self.env.action_space.sample()
                    # for models that need to always update internal state
                    if hasattr(self.model, 'every_t'):
                        _ = self.get_predicted_action(sequence, current_timestep)
                else:
                    action = self.get_predicted_action(sequence, current_timestep)
                observation, reward, done, _ = self.env.step(action)
                # track timesteps
                current_timestep += 1
                total_reward += reward
                sequence = sequence[1:]
                sequence.append(observation)

            self.timesteps_recorded += current_timestep
            self.timestep_rewards.append((self.timesteps_recorded, total_reward))

        final_reward = total_reward/self.EPS_AVG

        # indicate data collection
        if self.CSV_SAVE_PATH != '':
            if len(self.timestep_rewards) % 5000 < 20:
                with open(self.CSV_SAVE_PATH, 'a') as out:
                    csv_out = csv.writer(out)
                    csv_out.writerows(self.timestep_rewards)
                self.timestep_rewards = []

        # check for early stopping by timesteps
        if self.timesteps_recorded > int(self.MAX_TIMESTEPS*1.05):
            self.exit()

        return final_reward

    def exit(self):
        """Forces exit after saving weights."""
        print("Reached max timesteps for training.")
        print("Saving weights...")
        self.save(os.path.join('weights',
            'weights_{}_model_{}.pkl'.format(self.env_name, self.model_name)))
        sys.exit(0)