import epph.config as config
import epph.utils as utils
from epph.nap.preprocessor import Preprocessor
import epph.nap.tester as test
import epph.nap.trainer as train
import epph.exp.explainer as exp


def run_experiment(args):

    # For reproducible results
    if args.seed:
        utils.set_seed(args)

    # Init measurements file
    measures = utils.measures
    # if args.mode == 0 or args.mode == 2:
    #    utils.clear_measurement_file(args)

    # Init preprocessor and event log
    preprocessor = Preprocessor()
    event_log = preprocessor.get_event_log(args)

    # Split validation
    train_indices, test_indices = preprocessor.get_indices_split_validation(args, event_log)

    if args.mode == 0:

        best_model_id = train.train(args, preprocessor, event_log, train_indices, measures)
        predicted_distributions, ground_truths = test.test(args, preprocessor, event_log, test_indices, best_model_id,
                                                           measures)

        measures = utils.calculate_measures(args, measures, predicted_distributions, ground_truths,
                                            best_model_id=best_model_id)
        utils.print_measures(args, measures)
        utils.write_measures(args, measures)

    elif args.mode == 1:

        trace = preprocessor.get_random_case(args, event_log, args.rand_lower_bound, args.rand_upper_bound)
        exp.calc_and_plot_relevance_scores_instance(event_log, trace, args, preprocessor, train_indices)



if __name__ == '__main__':

    args = config.load()

    if args.run_experiments:

        experiments = utils.load_experiments(args)
        for experiment_id, experiment_config in experiments.items():
            # Configure experiment
            args = utils.set_experiment_config(args, experiment_config)
            # Run experiment
            print("Run experiment %s ..." % experiment_id)
            run_experiment(args)

    else:
        # single execution of code with initial configuration
        run_experiment(args)
