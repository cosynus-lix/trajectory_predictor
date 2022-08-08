from inspect import signature
import itertools

def multiparameter_generator(callable, args_dict):
    """
    Produces a generator of all possible combinations of the given parameters.

    :param callable: callable to be called with the given parameters
    :param args_dict: dictionary of parameters to list of possible values
    """
    params = signature(callable).parameters.keys()

    # Params should be valid
    assert len(params) == len(args_dict), "Number of parameters and given parameters must match"
    for param in params:
        if param not in args_dict:
            raise ValueError(f'Parameter {param} is not in parameters dictionary')
        elif len(args_dict[param]) == 0:
            raise ValueError(f'Parameter {param} should have at least one element')

    # Generate all possible combinations of the given parameters
    combinations = [args_dict[param] for param in params]
    for combination in itertools.product(*combinations):
        yield callable(*combination)
