from matplotlib import interactive
import matplotlib.pyplot as plt
import sys, os, argparse, csv
import numpy as np

interactive(True)

ENVS = ['Hopper-v2', 'Swimmer-v2', 'Walker2d-v2', 'HalfCheetah-v2', 'Humanoid-v2', 'Reacher-v2']
MODELS = ['MultilayerPerceptron', 'StructuredControlNet', 'RecurrentNeuralNetwork', 'RecurrentControlNet']
ABBRVS = ['mlp', 'scn', 'rnn', 'rcn']

# input directory must exist
IN_DIR = os.path.join(os.getcwd(), 'data')
assert(os.path.exists(IN_DIR))
# output directory will be made if it doesnt exist
OUT_DIR = os.path.join(os.getcwd(), 'plots')
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

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
    f = [i for i in f_ if env in i]
    
    # initialize plot
    print('Generating plot for environment {}...'.format(env))
    plt.figure()
    plt.xlim(0, max_timestep)
    plt.xticks([x for x in range(0, max_timestep+1, timescale)],
        [str(x/1000000) + 'M' for x in range(0, max_timestep+1, timescale)])
    plt.xlabel('Timesteps')
    plt.ylabel('Episodic Reward')
    plt.title('Episodic Reward, {}'.format(env))

    # iterate through each model for plotting
    for m, a in zip(MODELS, ABBRVS):
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
                plt.plot(x, y_moveavg, label=a)
            else:
                plt.plot(x, y, label=a)

    # save compiled plot
    print('Saving...')
    plt.legend()
    plt.savefig((os.path.join(OUT_DIR, '{}.png'.format(env))))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    # specify mode and envs
    # required_args = parser.add_argument_group('required arguments')
    parser.add_argument('--env', nargs=1, type=str, help='environment to plot')
    # plot parameters
    parser.add_argument('--avg_window', nargs='?', type=int, default=1000, help='moving average window')
    parser.add_argument('--max_timestep', nargs='?', type=int, default=10000000, help='max timestep to plot to')
    parser.add_argument('--timescale', nargs='?', type=int, default=2000000, help='timestep scale')
    parser.add_argument('--overwrite', action='store_true', default=False, dest='overwrite', help='overwrite existing avg')

    args = parser.parse_args()

    plot(args)

if __name__ == '__main__':
    main()
