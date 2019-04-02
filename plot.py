from matplotlib import interactive
import matplotlib.pyplot as plt
import sys, os, argparse, csv
import numpy as np

interactive(True)

ENVS = ['Hopper-v2', 'Swimmer-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Humanoid-v2']

def moving_average(a, n=100):
    """
    Computes moving average within a window of size n.
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def average(env, model, files):
    """
    Writes averages of trainings to the file 'env_{}_model_{}_avg.csv'.format(env, model).
    Note: .csv files may not align with timestep logging
    """
    print("Averaging {} files for {} with {}...".format(len(files), env, MODELS[ABBRVS.index(model)]))

    # logs maps timestep to reward
    logs = {}
    counts = {}
    print("Compiling averages...")
    for f in files:
        with open(os.path.join(IN_DIR, f), 'r') as c:
            csv_reader = csv.reader(c, delimiter=',')
            for row in csv_reader:
                x = float(row[0])
                y = float(row[1])
                # compute new reward y at timestep x
                total = counts.get(x, 0.0) * logs.get(x, 0.0) + y
                # update logs and counts
                counts[x] = counts.get(x, 0.0) + 1.0
                logs[x] = total / counts[x]
    print("Writing averages...")
    with open(os.path.join(IN_DIR, 'env_{}_model_{}_avg.csv'.format(env, model)), 'w') as c:
        csv_writer = csv.writer(c, delimiter=',')
        keys = list(logs.keys())
        keys.sort()
        for x in keys:
            csv_writer.writerow((x, logs[x]))

def plot(args):
    """
    Plot averages per model for specified environment.
    Must have one .csv file per model for specified environment.
    Generates .csv average every time for plotting.
    File format is 'env_{}_model_{}_avg.csv'.format(env, abbrv).
    """
    env = args.env[0]
    max_timestep = args.max_timestep
    timescale = args.timescale
    avg_window = args.avg_window
    overwrite = args.overwrite

    # obtain .csv files corresponding to env
    f_ = list(os.listdir(IN_DIR))
    f = [i for i in f_ if '_{}_'.format(env) in i]
    
    # initialize plot
    print('Generating plot for environment {}...'.format(env))
    plt.rcParams.update({'font.size': 13})
    plt.figure()
    plt.xlim(0, max_timestep)
    plt.xticks([x for x in range(0, max_timestep+1, timescale)],
        [str(x/1000000) + 'M' for x in range(0, max_timestep+1, timescale)])
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Reward')
    plt.title('{}'.format(env))

    # iterate through each model for plotting
    for m, a, l in zip(MODELS, ABBRVS, LABELS):
        # obtain .csv files corresponding to model
        files = [i for i in f if a in i]
        if len(files) == 0:
            raise Exception("No .csv file to plot for environment {}, model {}".format(env, m))

        # calculate average of all (env, model) files if necessary
        avg_file = os.path.join(IN_DIR, 'env_{}_model_{}_avg.csv'.format(env, a))
        if not os.path.exists(avg_file) or overwrite:
            # filter out existing avg file
            files_ = [i for i in files if 'avg' not in i]
            average(env, a, files_)

        with open(avg_file, 'r') as c:
            csv_reader = csv.reader(c, delimiter=',')
            x, y = [], []
            for row in csv_reader:
                x.append(float(row[0]))
                y.append(float(row[1]))
            x = np.array(x)
            y = np.array(y)
            print('Lines read: {}'.format(x.shape[0]))
            if avg_window > 1:
                y_moveavg = moving_average(y, avg_window)
                y_moveavg = np.append(y_moveavg, [y_moveavg[-1]] * (avg_window - 1))
                plt.plot(x, y_moveavg, label=l)
            else:
                plt.plot(x, y, label=l)

    # save compiled plot
    print('Saving...')
    plt.legend()
    plt.savefig((os.path.join(OUT_DIR, '{}.jpg'.format(env))))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    # specify env
    parser.add_argument('--env', nargs=1, type=str, help='environment to plot')
    parser.add_argument('--mode', nargs='?', type=str, default='baselines', help='models to plot')
    # directories
    parser.add_argument('--in_dir', nargs='?', type=str, default=os.path.join(os.getcwd(), 'data'), help='data directory')
    parser.add_argument('--out_dir', nargs='?', type=str, default=os.path.join(os.getcwd(), 'plots'), help='graph directory')
    # plot parameters
    parser.add_argument('--avg_window', nargs='?', type=int, default=100, help='moving average window')
    parser.add_argument('--max_timestep', nargs='?', type=int, default=10000000, help='max timestep to plot to')
    parser.add_argument('--timescale', nargs='?', type=int, default=2000000, help='timestep scale')
    parser.add_argument('--overwrite', action='store_true', default=False, dest='overwrite', help='overwrite existing avg')

    args = parser.parse_args()

    global IN_DIR
    global OUT_DIR
    # input directory must exist
    IN_DIR = args.in_dir
    assert(os.path.exists(IN_DIR))
    # output directory will be made if it doesnt exist
    OUT_DIR = args.out_dir
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    global MODELS
    global ABBRVS
    global LABELS
    # for plotting recurrent architectures against each other
    if args.mode == 'rnns':
        MODELS = ['RecurrentNeuralNetwork', 'GatedRecurrentUnit', 'LongShortTermMemory']
        ABBRVS = ['rnn', 'gru', 'lstm']
        LABELS = ['RNN-32', 'GRU-32', 'LSTM-32']
    # for plotting RCNs for bias vector testing
    elif args.mode == 'rcn-biases':
        MODELS = ['RecurrentControlNet,WithBias', 'RecurrentControlNet,NoBias']
        ABBRVS = ['rcn_bias', 'rcn_nobias']
        LABELS = ['RCN-32 with bias', 'RCN-32 with no bias']
    # for plotting baselines against each other
    else:
        MODELS = ['MultilayerPerceptron', 'StructuredControlNet', 'RecurrentNeuralNetwork', 'RecurrentControlNet']
        ABBRVS = ['mlp', 'scn', 'rnn', 'rcn']
        LABELS = ['MLP-64', 'SCN-16', 'RNN-32', 'RCN-32']

    plot(args)

if __name__ == '__main__':
    main()
