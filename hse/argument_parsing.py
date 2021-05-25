import argparse


def non_empty_string(s):
    if not s:
        raise ValueError("Must not be empty string")
    return s


def get_parser():
    parser = argparse.ArgumentParser(
        description='Predict speedup of pytorch model from communication and computation benchmarks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Optional parameters
    parser.add_argument('-s', '--system', help="Current system name. (Only lisa supported)",
                        default='lisa', type=non_empty_string)
    parser.add_argument('-mp', '--max-processes', help="Maximum number of MPI processes to consider",
                        default=32, type=int)
    parser.add_argument('-mb', '--max-batch-size', help="Maximum number of batches to consider",
                        default=400, type=int)
    parser.add_argument('-it', '--iterations', help="Number of iterations to estimate forward time per input batch",
                        default=100, type=int)
    parser.add_argument('-gc', '--garbage-collect', help="Enable garbage collection between forward passes",
                        action='store_true')
    parser.add_argument('-save', '--save-result', help="Save plot of result as file (hse-result.png)",
                        action='store_true')
    parser.add_argument('-v', '--verbose', help="Increase output verbosity",
                        action='store_true')

    # Conditional Parameters
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('-il', '--input-layer', help="Name of first layer (should have in_features property)",
                             type=non_empty_string)
    input_group.add_argument('-is', '--input-size', help="Shape of input layer for a single data point",
                             type=int, nargs='+')

    # Required Parameters
    parser.add_argument('module', help="The python module to import the model from",
                        type=non_empty_string)
    parser.add_argument('model', help="The name of the model to import from module (subclass of torch.nn.Module)",
                        type=non_empty_string)

    return parser


def get_args():
    return get_parser().parse_args()
