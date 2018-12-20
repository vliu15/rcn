import os

from utils.cli_parser import arg_parser
from es.wrapper import EvolutionStrategyWrapper

def main():
    args = arg_parser().parse_args()
    agent = EvolutionStrategyWrapper(args)

    # load existing weights if specified
    if args.weights_file != '':
        agent.load(args.weights_file)

    # render environment if specified
    if args.render:
        agent.render()
    else:
        if not os.path.exists('weights'):
            os.makedirs('weights')
        path = os.path.join('weights', 'weights_{}_model_{}.pkl'.format(args.env, args.model))
        # run until end of iterations
        try:
            agent.train(args.num_iters)
        # catch early exit from training
        except KeyboardInterrupt:
            print("Saving weights...")
            agent.save(path)
            return
        print("Saving weights...")
        agent.save(path)


if __name__ == '__main__':
    main()
 