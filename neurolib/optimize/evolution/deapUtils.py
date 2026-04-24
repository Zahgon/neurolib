import random
import copy
import numpy as np

def randomParameters(paramInterval):
    """
    Generate a sequence of random parameters from a ParamsInterval using a uniform distribution.
    Format: [mean_par1, mean_par2, ...]
    """
    pass

def randomParametersAdaptive(paramInterval):
    """
    Generate a sequence of random parameters from a ParamsInterval using a uniform distribution.
    Format: [mean_par1, mean_par2, ..., sigma_par1, sigma_par2, ...]
    The second half of the parameter list is set of adaptive mutation std deviation parameters.
    """
    pass

def mutateUntilValid(pop, paramInterval, toolbox, MUTATE_P={}, maxTries=100):
    """Checks the validity of new individuals' parameter. If they are invalid 
    (for example if they are out of the predefined paramter space bounds), 
    mutate the individual, until valid.

    :param pop: population to mutate
    :param paramInterval: parameter interval (from parameterSpace.named_tuple)
    :param toolbox: deap toolbox
    :param maxTries: how many mutations to try until valid
    """
    pass

def checkParamValidity(individual, paramInterval):
    """
    Check if an individual is within the specified bounds
    Return True if it is correct, False otherwise
    """
    pass

def selRank(pop, k, s=1.5):
    """
    Select k individuals from a population using rank selection. (Eiben&Smith, p.81)
    Individuals are selected according to the fitness rank.
    To support multiobjective fitness functions, the weighted sum of fitness is used.

    :param pop: population
    :type pop: list
    :param k: number of individuals to select
    :type k: int
    :param s: selection probability parameter
    :type s: float

    :return: population of selected individuals
    :rtype: list
    """
    pass

def selBest_multiObj(pop, k):
    """
    Select the best k individuals.

    This function accept multiobjective function by summing the fitness all of objectives.
    """
    pass

def cxNormDraw_adapt(ind1, ind2, sigma_scale=2.0):
    """The new attributes of the two individuals are set according to a normal distribution whose mean is
    the mean between both individual's attributes and the standard deviation being the distance between the two attributes.
    
    Similar to mutation parameter described in Ono et al 2003 but with only 2 parents (and not 3).

    Info: The individuals are composed of the gene values first and then the mutation rates.
    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param sigma_scale: Scaling of sigma (distance of parents / sigma_scale)
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    pass

def cxUniform_adapt(ind1, ind2, indpb):
    """The new attributes of the two individuals are set according to a normal distribution whose mean is
    the mean between both individual's attributes and the standard deviation being the distance between the two attributes.
    
    Info: The individuals are composed of the gene values first and then the mutation rates.
    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.

    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    pass

def cxUniform_normDraw_adapt(ind1, ind2, indpb):
    """Executes a uniform crossover that modify in place the two
    :term:`sequence` individuals.
    The attributes of the 2 individuals are set according to a normal distribution whose mean is
    the mean between both individual attributes and the standard deviation the distance between the 2 attributes.
    The individuals are composed of the gene values first and then the mutation rates.
    Warning: a check should be done afterward on the parameter to be sure they are not out of bound.
    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Independent probabily for each attribute to be exchanged.
    :returns: A tuple of two individuals.
    This function uses the :func:`~random.random` function from the python base
    :mod:`random` module.
    """
    pass

def gaussianAdaptiveMutation_nStepSizes(individual, gamma_gl=None, gamma=None):
    """
    Perform an uncorrelated adaptive mutation with n step sizes on the individual

    Warning: the mutations is in place, i.e. it modifies the given individual
    Parameters:
        :param individual: Inidivual to mutate. This should a sequence of length 2 * n_params 
        the last n_params elements being the individual adaptation rates)
        :param gamma_gl: Global adaptive mutation param ( should be proportional to 1/sqrt(2 n_params ) )
        :param gamma: Adaptive mutation parameters ( should be proportional to 1/sqrt(2 sqrt(n_params) ) )

    :returns: the individual

    """
    pass