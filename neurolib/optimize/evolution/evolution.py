import datetime
import logging
import multiprocessing
import os
from functools import partial
import deap
import numpy as np
import pandas as pd
import pypet as pp
from deap import base, creator, tools
from ...utils import paths as paths
from ...utils import pypetUtils as pu
from ...utils.collections import BACKWARD_REPLACE, unwrap_star_dotdict
from ...utils.parameterSpace import ParameterSpace
from . import deapUtils as du
from . import evolutionaryUtils as eu

class Evolution:
    """Evolutionary parameter optimization. This class helps you to optimize any function or model using an evolutionary algorithm.
    It uses the package `deap` and supports its builtin mating and selection functions as well as custom ones.
    """

    def __init__(self, evalFunction, parameterSpace, weightList=None, model=None, filename='evolution.hdf', ncores=None, POP_INIT_SIZE=100, POP_SIZE=20, NGEN=10, algorithm='adaptive', matingOperator=None, MATE_P=None, mutationOperator=None, MUTATE_P=None, selectionOperator=None, SELECT_P=None, parentSelectionOperator=None, PARENT_SELECT_P=None, individualGenerator=None, IND_GENERATOR_P=None):
        """Initialize evolutionary optimization.
        :param evalFunction: Evaluation function of a run that provides a fitness vector and simulation outputs
        :type evalFunction: function
        :param parameterSpace: Parameter space to run evolution in.
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`
        :param weightList: List of floats that defines the dimensionality of the fitness vector returned from evalFunction and the weights of each component for multiobjective optimization (positive = maximize, negative = minimize). If not given, then a single positive weight will be used, defaults to None
        :type weightList: list[float], optional
        :param model: Model to simulate, defaults to None
        :type model: `neurolib.models.model.Model`, optional

        :param filename: HDF file to store all results in, defaults to "evolution.hdf"
        :type filename: str, optional
        :param ncores: Number of cores to simulate on (max cores default), defaults to None
        :type ncores: int, optional

        :param POP_INIT_SIZE: Size of first population to initialize evolution with (random, uniformly distributed), defaults to 100
        :type POP_INIT_SIZE: int, optional
        :param POP_SIZE: Size of the population during evolution, defaults to 20
        :type POP_SIZE: int, optional
        :param NGEN: Numbers of generations to evaluate, defaults to 10
        :type NGEN: int, optional

        :param matingOperator: Custom mating operator, defaults to deap.tools.cxBlend
        :type matingOperator: deap operator, optional
        :param MATE_P: Mating operator keyword arguments (for the default crossover operator cxBlend, this defaults `alpha` = 0.5)
        :type MATE_P: dict, optional

        :param mutationOperator: Custom mutation operator, defaults to du.gaussianAdaptiveMutation_nStepSizes
        :type mutationOperator: deap operator, optional
        :param MUTATE_P: Mutation operator keyword arguments
        :type MUTATE_P: dict, optional

        :param selectionOperator: Custom selection operator, defaults to du.selBest_multiObj
        :type selectionOperator: deap operator, optional
        :param SELECT_P: Selection operator keyword arguments
        :type SELECT_P: dict, optional

        :param parentSelectionOperator: Operator for parent selection, defaults to du.selRank
        :param PARENT_SELECT_P: Parent selection operator keyword arguments (for the default operator selRank, this defaults to `s` = 1.5 in Eiben&Smith p.81)
        :type PARENT_SELECT_P: dict, optional

        :param individualGenerator: Function to generate initial individuals, defaults to du.randomParametersAdaptive
        """
        if weightList is None:
            logging.info('weightList not set, assuming single fitness value to be maximized.')
            weightList = [1.0]
        trajectoryName = 'results' + datetime.datetime.now().strftime('-%Y-%m-%d-%HH-%MM-%SS')
        logging.info(f'Trajectory Name: {trajectoryName}')
        self.HDF_FILE = os.path.join(paths.HDF_DIR, filename)
        trajectoryFileName = self.HDF_FILE
        logging.info('Storing data to: {}'.format(trajectoryFileName))
        logging.info('Trajectory Name: {}'.format(trajectoryName))
        if ncores is None:
            ncores = multiprocessing.cpu_count()
        logging.info('Number of cores: {}'.format(ncores))
        env = pp.Environment(trajectory=trajectoryName, filename=trajectoryFileName, use_pool=False, multiproc=True, ncores=ncores, complevel=9, log_config=paths.PYPET_LOGGING_CONFIG)
        traj = env.traj
        assert trajectoryName == traj.v_name, f'Pypet trajectory has a different name than trajectoryName {trajectoryName}'
        self.model = model
        self.evalFunction = evalFunction
        self.weightList = weightList
        self.NGEN = NGEN
        assert POP_SIZE % 2 == 0, 'Please chose an even number for POP_SIZE!'
        self.POP_SIZE = POP_SIZE
        assert POP_INIT_SIZE % 2 == 0, 'Please chose an even number for POP_INIT_SIZE!'
        self.POP_INIT_SIZE = POP_INIT_SIZE
        self.ncores = ncores
        self.comments = 'no comments'
        self.traj = env.traj
        self.env = env
        self.trajectoryName = trajectoryName
        self.trajectoryFileName = trajectoryFileName
        self._initialPopulationSimulated = False
        self.verbose = False
        self.verbose_plotting = True
        self.plotColor = 'C0'
        self.parameterSpace = parameterSpace
        self.ParametersInterval = self.parameterSpace.named_tuple_constructor
        self.paramInterval = self.parameterSpace.named_tuple
        self.toolbox = deap.base.Toolbox()
        if algorithm == 'adaptive':
            logging.info(f'Evolution: Using algorithm: {algorithm}')
            self.matingOperator = tools.cxBlend
            self.MATE_P = {'alpha': 0.5} or MATE_P
            self.mutationOperator = du.gaussianAdaptiveMutation_nStepSizes
            self.selectionOperator = du.selBest_multiObj
            self.parentSelectionOperator = du.selRank
            self.PARENT_SELECT_P = {'s': 1.5} or PARENT_SELECT_P
            self.individualGenerator = du.randomParametersAdaptive
        elif algorithm == 'nsga2':
            logging.info(f'Evolution: Using algorithm: {algorithm}')
            self.matingOperator = tools.cxSimulatedBinaryBounded
            self.MATE_P = {'low': self.parameterSpace.lowerBound, 'up': self.parameterSpace.upperBound, 'eta': 20.0} or MATE_P
            self.mutationOperator = tools.mutPolynomialBounded
            self.MUTATE_P = {'low': self.parameterSpace.lowerBound, 'up': self.parameterSpace.upperBound, 'eta': 20.0, 'indpb': 1.0 / len(self.weightList)} or MUTATE_P
            self.selectionOperator = tools.selNSGA2
            self.parentSelectionOperator = tools.selTournamentDCD
            self.individualGenerator = du.randomParameters
        else:
            raise ValueError("Evolution: algorithm must be one of the following: ['adaptive', 'nsga2']")
        self.matingOperator = self.matingOperator if hasattr(self, 'matingOperator') else matingOperator
        self.mutationOperator = self.mutationOperator if hasattr(self, 'mutationOperator') else mutationOperator
        self.selectionOperator = self.selectionOperator if hasattr(self, 'selectionOperator') else selectionOperator
        self.parentSelectionOperator = self.parentSelectionOperator if hasattr(self, 'parentSelectionOperator') else parentSelectionOperator
        self.individualGenerator = self.individualGenerator if hasattr(self, 'individualGenerator') else individualGenerator
        self.MATE_P = self.MATE_P if hasattr(self, 'MATE_P') else {}
        self.PARENT_SELECT_P = self.PARENT_SELECT_P if hasattr(self, 'PARENT_SELECT_P') else {}
        self.MUTATE_P = self.MUTATE_P if hasattr(self, 'MUTATE_P') else {}
        self.SELECT_P = self.SELECT_P if hasattr(self, 'SELECT_P') else {}
        self._initDEAP(self.toolbox, self.env, self.paramInterval, self.evalFunction, weightList=self.weightList, matingOperator=self.matingOperator, mutationOperator=self.mutationOperator, selectionOperator=self.selectionOperator, parentSelectionOperator=self.parentSelectionOperator, individualGenerator=self.individualGenerator)
        self._initPypetTrajectory(self.traj, self.paramInterval, self.POP_SIZE, self.NGEN, self.model)
        self.history = {}
        self.evaluationCounter = 0
        self.last_id = 0

    def run(self, verbose=False, verbose_plotting=True):
        """Run the evolution or continue previous evolution. If evolution was not initialized first
        using `runInitial()`, this will be done.

        :param verbose: Print and plot state of evolution during run, defaults to False
        :type verbose: bool, optional
        """
        pass

    def getIndividualFromTraj(self, traj):
        """Get individual from pypet trajectory

        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Individual (`DEAP` type)
        :rtype: `deap.creator.Individual`
        """
        pass

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this individual
        :params traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual

        :returns model: Model with the parameters of this individual.

        :param traj: Pypet trajectory with individual (traj.individual) or directly a deap.Individual
        :type traj: `pypet.trajectory.Trajectory`
        :return: Model with the parameters of this individual.
        :rtype: `neurolib.models.model.Model`
        """
        pass

    def getIndividualFromHistory(self, id):
        """Searches the entire evolution history for an individual with a specific id and returns it.

        :param id: Individual id
        :type id: int
        :return: Individual (`DEAP` type)
        :rtype: `deap.creator.Individual`
        """
        pass

    def individualToDict(self, individual):
        """Convert an individual to a parameter dictionary.

        :param individual: Individual (`DEAP` type)
        :type individual: `deap.creator.Individual`
        :return: Parameter dictionary of this individual
        :rtype: dict
        """
        pass

    def _initPypetTrajectory(self, traj, paramInterval, POP_SIZE, NGEN, model):
        """Initializes pypet trajectory and store all simulation parameters for later analysis.

        :param traj: Pypet trajectory (must be already initialized!)
        :type traj: `pypet.trajectory.Trajectory`
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param POP_SIZE: Population size
        :type POP_SIZE: int
        :param MATE_P: Crossover parameter
        :type MATE_P: float
        :param NGEN: Number of generations
        :type NGEN: int
        :param model: Model to store the default parameters of
        :type model: `neurolib.models.model.Model`
        """
        pass

    def _initDEAP(self, toolbox, pypetEnvironment, paramInterval, evalFunction, weightList, matingOperator, mutationOperator, selectionOperator, parentSelectionOperator, individualGenerator):
        """Initializes DEAP and registers all methods to the deap.toolbox

        :param toolbox: Deap toolbox
        :type toolbox: deap.base.Toolbox
        :param pypetEnvironment: Pypet environment (must be initialized first!)
        :type pypetEnvironment: [type]
        :param paramInterval: Parameter space, from ParameterSpace class
        :type paramInterval: parameterSpace.named_tuple
        :param evalFunction: Evaluation function
        :type evalFunction: function
        :param weightList: List of weiths for multiobjective optimization
        :type weightList: list[float]
        :param matingOperator: Mating function (crossover)
        :type matingOperator: function
        :param selectionOperator: Parent selection function
        :type selectionOperator: function
        :param individualGenerator: Function that generates individuals
        """
        pass

    def _evalPopulationUsingPypet(self, traj, toolbox, pop, gIdx):
        """Evaluate the fitness of the popoulation of the current generation using pypet
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :param toolbox: `deap` toolbox
        :type toolbox: deap.base.Toolbox
        :param pop: Population
        :type pop: list
        :param gIdx: Index of the current generation
        :type gIdx: int
        :return: Evaluated population with fitnesses
        :rtype: list
        """
        pass

    def getValidPopulation(self, pop=None):
        """Returns a list of the valid population.

        :params pop: Population to check, defaults to self.pop
        :type pop: deap population
        :return: List of valid population
        :rtype: list
        """
        pass

    def getInvalidPopulation(self, pop=None):
        """Returns a list of the invalid population.

        :params pop: Population to check, defaults to self.pop
        :type pop: deap population
        :return: List of invalid population
        :rtype: list
        """
        pass

    def _tagPopulation(self, pop):
        """Take a fresh population and add id's and attributes such as parameters that we can use later

        :param pop: Fresh population
        :type pop: list
        :return: Population with tags
        :rtype: list
        """
        pass

    def runInitial(self):
        """Run the first round of evolution with the initial population of size `POP_INIT_SIZE`
        and select the best `POP_SIZE` for the following evolution. This needs to be run before `runEvolution()`
        """
        pass

    def runEvolution(self):
        """Run the evolutionary optimization process for `NGEN` generations."""
        pass

    def _buildEvolutionTree(self):
        """Builds a genealogy tree that is networkx compatible.

        Plot the tree using:

            import matplotlib.pyplot as plt
            import networkx as nx
            from networkx.drawing.nx_pydot import graphviz_layout

            G = nx.DiGraph(evolution.tree)
            G = G.reverse()     # Make the graph top-down
            pos = graphviz_layout(G, prog='dot')
            plt.figure(figsize=(8, 8))
            nx.draw(G, pos, node_size=50, alpha=0.5, node_color=list(evolution.genx.values()), with_labels=False)
            plt.show()
        """
        pass

    def info(self, plot=True, bestN=5, info=True, reverse=False):
        """Print and plot information about the evolution and the current population

        :param plot: plot a plot using `matplotlib`, defaults to True
        :type plot: bool, optional
        :param bestN: Print summary of `bestN` best individuals, defaults to 5
        :type bestN: int, optional
        :param info: Print information about the evolution environment
        :type info: bool, optional
        """
        pass

    def plotProgress(self, reverse=False):
        """Plots progress of fitnesses of current evolution run"""
        pass

    def saveEvolution(self, fname=None):
        """Save evolution to file using dill.

        :param fname: Filename, defaults to a path in ./data/
        :type fname: str, optional
        """
        pass

    def loadEvolution(self, fname):
        """Load evolution from previously saved simulations.

        Example usage:
        ```
        evaluateSimulation = lambda x: x # the function can be omitted, that's why we define a lambda here
        pars = ParameterSpace(['a', 'b'], # should be same as previously saved evolution
                      [[0.0, 4.0], [0.0, 5.0]])
        evolution = Evolution(evaluateSimulation, pars, weightList = [1.0])
        evolution = evolution.loadEvolution("data/evolution-results-2020-05-15-00H-24M-48S.dill")
        ```

        :param fname: Filename, defaults to a path in ./data/
        :type fname: str
        :return: Evolution
        :rtype: self
        """
        pass

    def _outputToDf(self, pop, df):
        """Loads outputs dictionary from evolution from the .outputs attribute
        and writes data into a dataframe.

        :param pop: Population of which to get outputs from.
        :type pop: list
        :param df: Dataframe to which outputs are written
        :type df: pandas.core.frame.DataFrame
        :return: Dataframe with outputs
        :rtype: pandas.core.frame.DataFrame
        """
        pass

    def _dropDuplicatesFromDf(self, df):
        """Drops duplicates from dfEvolution dataframe.
        Tries vanilla drop_duplicates, which fails if the Dataframe contains
        data objects like numpy.arrays. Tries to drop via key "id" if it fails.

        :param df: Input dataframe with duplicates to drop
        :type df: pandas.core.frame.DataFrame
        :return: Dataframe without duplicates
        :rtype: pandas.core.frame.DataFrame
        """
        pass

    def dfPop(self, outputs=False):
        """Returns a `pandas` DataFrame of the current generation's population parameters.
        This object can be further used to easily analyse the population.
        :return: Pandas DataFrame with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        pass

    def dfEvolution(self, outputs=False):
        """Returns a `pandas` DataFrame with the individuals of the the whole evolution.
        This method can be usef after loading an evolution from disk using loadEvolution()

        :return: Pandas DataFrame with all individuals and their parameters
        :rtype: `pandas.core.frame.DataFrame`
        """
        pass

    def loadResults(self, filename=None, trajectoryName=None):
        """Load results from a hdf file of a previous evolution and store the
        pypet trajectory in `self.traj`

        :param filename: hdf filename of the previous run, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory in the hdf file to load. If not given, the last one will be loaded, defaults to None
        :type trajectoryName: str, optional
        """
        pass

    def getScores(self):
        """Returns the scores of the current valid population"""
        pass

    def getScoresDuringEvolution(self, traj=None, drop_first=True, reverse=False):
        """Get the scores of each generation's population.

        :param traj: Pypet trajectory. If not given, the current trajectory is used, defaults to None
        :type traj: `pypet.trajectory.Trajectory`, optional
        :param drop_first: Drop the first (initial) generation. This can be usefull because it can have a different size (`POP_INIT_SIZE`) than the succeeding populations (`POP_SIZE`) which can make data handling tricky, defaults to True
        :type drop_first: bool, optional
        :param reverse: Reverse the order of each generation. This is a necessary workaraound because loading from the an hdf file returns the generations in a reversed order compared to loading each generation from the pypet trajectory in memory, defaults to False
        :type reverse: bool, optional
        :return: Tuple of list of all generations and an array of the scores of all individuals
        :rtype: tuple[list, numpy.ndarray]
        """
        pass