import argparse


def arg_parser():
    """Create an argparse.ArgumentParser for es/run.py."""
    parser = argparse.ArgumentParser()

    # adjust for trials as desired
    parser.add_argument('--env', help='environment ID', nargs='?', type=str, default='Swimmer-v2')
    parser.add_argument('--num_iters', help='number of iterations to train for', nargs='?', type=int, default=1000000)
    parser.add_argument('--num_timesteps', help='number of timesteps to train for', nargs='?', type=int, default=2000000)
    parser.add_argument('--model', help='model to run', nargs='?', type=str, default='scn')

    # adjust for training stats
    parser.add_argument('--collect_data', action='store_true', default=False, dest='collect_data', help='allow data collection')
    parser.add_argument('--render', action='store_true', default=False, dest='render', help='render trained weights in environment')
    parser.add_argument('--weights_file', nargs='?', default='',help='file to load weights')

    # set parameters
    parser.add_argument('--print_steps', help='training print interval', nargs='?', type=int, default=1)
    parser.add_argument('--seed', help='RNG seed', nargs='?', type=int, default=None)
    parser.add_argument('--agent_history_len', nargs='?', type=int, default=1)
    parser.add_argument('--population_size', nargs='?', type=int, default=20)
    parser.add_argument('--eps_avg', nargs='?', type=int, default=1)
    parser.add_argument('--sigma', nargs='?', type=float, default=0.18)
    parser.add_argument('--learning_rate', nargs='?', type=float, default=0.02)
    parser.add_argument('--initial_exploration_prob', nargs='?', type=float, default=1.0)
    parser.add_argument('--final_exploration', nargs='?', type=float, default=0.0)
    parser.add_argument('--exploration_dec_steps', nargs='?', type=int, default=1000000)
    parser.add_argument('--num_threads', nargs='?', type=int, default=1)

    return parser
