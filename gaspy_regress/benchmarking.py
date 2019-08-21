'''
This module is meant to be used to assess regression models for their
fitness-for-use in our active discovery workflows.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'


import gc
import sys
import math
import random
import warnings
from copy import deepcopy
import pickle
from bisect import bisect_right
import numpy as np
from scipy.stats import norm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import ticker
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
import torch
import gpytorch
from gaspy_regress import fingerprinters

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


# Used to put commas into figures' axes' labels
FORMATTER = ticker.FuncFormatter(lambda x, p: format(int(x), ','))


class ActiveDiscoverer:
    '''
    This is a parent class for simulating active discovery routines. The
    child classes are meant to be used to judge the efficacy of training
    ando/or acquisition routines. It does so by "simulating" what the routine
    would have done given a particular sampling space.

    Required methods:
        _update_regret      This method should take the output of the
                            `choose_next_batch` method and then use it to
                            calculate the new cumulative regret. It should then
                            append it to the `regret_history` attribute.
        _train              This method should take the output of the
                            `choose_next_batch` method; calculate the current
                            model's residuals on that batch and extend them
                            onto the `residuals` attribute; use the training
                            batch [re]train the surrogate model; and finally
                            extend the `self.training_set` attribute with the
                            batch that it is passed.
        _choose_next_batch  This method should choose `self.batch_size` samples
                            from the `self.sampling_space` attribute, then
                            put them into a list and assign it to the
                            `self.training_batch` attribute. It should also
                            remove anything it selected from the
                            `self.sampling_space` attribute.
        plot_parity         This method should return an instance of a
                            `matplotlib.pyplot.figure` object with the parity
                            plotted.
    '''
    def __init__(self, optimal_value, training_set, sampling_space,
                 init_train=True, batch_size=200):
        '''
        Perform the initial training for the active discoverer, and then
        initialize all of the other attributes.

        Args:
            optimal_value   An object that represents the optimal value of
                            whatever it is you're optimizing. Should be used in
                            the `update_regret` method.
            training_set    A sequence that can be used to train/initialize the
                            surrogate model of the active discoverer.
            sampling_space  A sequence containing the rest of the possible
                            sampling space.
            batch_size      An integer indicating how many elements in the
                            sampling space you want to choose
            init_train      Boolean indicating whether or not you want to do
                            your first training during this initialization.
                            You should keep it `True` unless you're doing a
                            warm start (manually).
        '''
        # Attributes we use to judge the discovery
        self.regret_history = [0.]
        self.residuals = []

        # Attributes we need to hallucinate the discovery
        self.optimal_value = optimal_value
        self.next_batch_number = 0
        self.training_set = []
        self.training_batch = list(deepcopy(training_set))
        self.sampling_space = list(deepcopy(sampling_space))
        self.batch_size = batch_size
        if init_train:
            self._train()

        # Attributes used in the `__assert_correct_hallucination` method
        self.__previous_training_set_len = len(self.training_set)
        self.__previous_sampling_space_len = len(self.sampling_space)
        self.__previous_regret_history_len = len(self.regret_history)
        self.__previous_residuals_len = len(self.residuals)

    def simulate_discovery(self, starting_batch_number=0):
        '''
        Perform the discovery simulation until all of the sampling space has
        been consumed.
        '''
        n_batches = math.ceil(len(self.sampling_space) / self.batch_size)
        for i in tqdm(range(0, n_batches), desc='Hallucinating discovery...'):
            self._hallucinate_next_batch()

    def _hallucinate_next_batch(self):
        '''
        Choose the next batch of data to get, add them to the `samples`
        attribute, and then re-train the surrogate model with the new samples.
        '''
        # Perform one iteration of active discovery
        self._choose_next_batch()
        self._train()
        self._update_regret()

        # Make sure it was done correctly
        self.__assert_correct_hallucination()

    def __assert_correct_hallucination(self):
        '''
        There are quite a few things that the user needs to do correctly to
        make a child class for this. This method will verify they're done
        correctly and then let the user know if it's not.
        '''
        # Make sure that the the sampling space is being reduced
        try:
            assert len(self.sampling_space) < self.__previous_sampling_space_len
            self.__previous_sampling_space_len = len(self.sampling_space)
        except AssertionError as error:
            message = ('\nWhen creating the `_choose_next_batch` method for '
                       'a child-class of `ActiveDiscoverer`, you need to '
                       'remove the chosen batch from the `sampling_space` '
                       'attribute.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the training set is being increased
        try:
            assert len(self.training_set) > self.__previous_training_set_len
            self.__previous_training_set_len = len(self.training_set)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveDiscoverer`, you need to extend '
                       'the `training_set` attribute with the new training '
                       'batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the residuals are being recorded
        try:
            assert len(self.residuals) > self.__previous_residuals_len
            self.__previous_residuals_len = len(self.residuals)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveDiscoverer`, you need to extend '
                       'the `residuals` attribute with the model\'s residuals '
                       'of the new batch (before retraining).')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure we are calculating the regret at each step
        try:
            assert len(self.regret_history) > self.__previous_regret_history_len
            self.__previous_regret_history_len = len(self.regret_history)
        except AssertionError as error:
            message = ('\nWhen creating the `_update_regret` method for a '
                       'child-class of `ActiveDiscoverer`, you need to append '
                       'the `regret_history` attribute with the cumulative '
                       'regret given the new batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

    def _pop_next_batch(self):
        '''
        Optional helper function that you can use to choose the next batch from
        `self.sampling_space`, remove it from the attribute, place the new
        batch onto the `self.training_batch` attribute, increment the
        `self.next_batch_number`.

        This method will only work if you have already sorted the
        `self.sampling_space` such that the highest priority samples are
        earlier in the index.
        '''
        samples = []
        for _ in range(self.batch_size):
            try:
                sample = self.sampling_space.pop(0)
                samples.append(sample)
            except IndexError:
                break
        self.training_batch = samples
        self.next_batch_number += 1

    def plot_performance(self, window=20, metric='mean'):
        '''
        Light wrapper for plotting the regret and residuals over the course of
        the discovery.

        Arg:
            window  How many residuals to average at each point in the learning
                    curve
            metric  String indicating which metric you want to plot in the
                    learning curve.  Corresponds exactly to the methods of the
                    `pandas.DataFrame.rolling` class, e.g., 'mean', 'median',
                    'min', 'max', 'std', 'sum', etc.
        Returns:
            regret_fig  The matplotlib figure object for the regret plot
            resid_fig   The matplotlib figure object for the residual plot
        '''
        regret_fig = self.plot_regret()
        learning_fig = self.plot_learning_curve(window)
        parity_fig = self.plot_parity()
        return regret_fig, learning_fig, parity_fig

    def plot_regret(self):
        '''
        Plot the regret vs. discovery batch

        Returns:
            fig     The matplotlib figure object for the regret plot
        '''
        # Plot
        fig = plt.figure()
        sampling_sizes = [i*self.batch_size for i, _ in enumerate(self.regret_history)]
        ax = sns.scatterplot(sampling_sizes, self.regret_history)

        # Format
        _ = ax.set_xlabel('Number of discovery queries')  # noqa: F841
        _ = ax.set_ylabel('Cumulative regret [eV]')  # noqa: F841
        _ = fig.set_size_inches(15, 5)  # noqa: F841
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)
        _ = ax.get_yaxis().set_major_formatter(FORMATTER)
        return fig

    def plot_learning_curve(self, window=20, metric='mean'):
        '''
        Plot the rolling average of the residuals over time

        Arg:
            window  How many residuals to average at each point
            metric  String indicating which metric you want to plot.
                    Corresponds exactly to the methods of the
                    `pandas.DataFrame.rolling` class, e.g., 'mean', 'median',
                    'min', 'max', 'std', 'sum', etc.
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        # Format the data
        df = pd.DataFrame(self.residuals, columns=['Residuals [eV]'])
        rolling_residuals = getattr(df, 'Residuals [eV]').rolling(window=window)
        rolled_values = getattr(rolling_residuals, metric)().values
        query_numbers = list(range(len(rolled_values)))

        # Create and format the figure
        fig = plt.figure()
        ax = sns.lineplot(query_numbers, rolled_values)
        _ = ax.set_xlabel('Number of discovery queries')
        _ = ax.set_ylabel('Rolling %s of residuals (window = %i) [eV]' % (metric, window))
        _ = ax.set_xlim([query_numbers[0], query_numbers[-1]])
        _ = fig.set_size_inches(15, 5)  # noqa: F841
        _ = ax.get_xaxis().set_major_formatter(FORMATTER)

        # Add a dashed line at zero residuals
        plt.plot([0, query_numbers[-1]], [0, 0], '--k')

        return fig


class AdsorptionDiscoverer(ActiveDiscoverer):
    '''
    Here we extend the `ActiveDiscoverer` class while making the following
    assumptions:  1) we are trying to optimize the adsorption energy and 2) our
    inputs are a list of dictionaries with the 'energy' key.
    '''
    def _update_regret(self):
        '''
        Calculates the cumulative regret of the discovery thus far. Assumes
        that your sampling space is a list of dictionaries, and that the key
        you want to optimize is 'energy'.
        '''
        # Find the current regret
        regret = self.regret_history[-1]

        # Add the new regret
        for doc in self.training_batch:
            energy = doc['energy']
            difference = energy - self.optimal_value
            regret += abs(difference)
        self.regret_history.append(regret)

    def plot_parity(self):
        '''
        Make the parity plot, where the residuals were the intermediate
        residuals we got along the way.

        Returns:
            fig     The matplotlib figure object for the parity plot
        '''
        # Pull the data that we have residuals for
        sampled_data = self.training_set[-len(self.residuals):]

        # Get both the DFT energies and the predicted energies
        energies_dft = np.array([doc['energy'] for doc in sampled_data])
        energies_pred = energies_dft + np.array(self.residuals)

        # Plot and format
        fig = plt.figure()
        energy_range = [min(energies_dft.min(), energies_pred.min()),
                        max(energies_dft.max(), energies_pred.max())]
        jgrid = sns.jointplot(energies_dft, energies_pred,
                              extent=energy_range*2,
                              kind='hex', bins='log')
        ax = jgrid.ax_joint
        _ = ax.set_xlabel('DFT-calculated adsorption energy [eV]')  # noqa: F841
        _ = ax.set_ylabel('ML-predicted adsorption energy [eV]')  # noqa: F841
        _ = fig.set_size_inches(10, 10)  # noqa: F841

        # Add the parity line
        _ = ax.plot(energy_range, energy_range, '--')  # noqa: F841
        return fig


class RandomAdsorptionDiscoverer(AdsorptionDiscoverer):
    '''
    This discoverer simply chooses new samples randomly. This is intended to be
    used as a baseline for active discovery.
    '''
    def _train(self):
        '''
        There's no real training here. We just do the bare minimum:  make up
        some residuals and extend the training set.
        '''
        try:
            # We'll just arbitrarily set the "model's" guess to the current average
            # of the training set
            energies = [doc['energy'] for doc in self.training_set]
            energy_guess = sum(energies) / max(len(energies), 1)
            residuals = [energy_guess - doc['energy'] for doc in self.training_batch]
            self.residuals.extend(residuals)

            # Mandatory extension of the training set to include this next batch
            self.training_set.extend(self.training_batch)

        # If this is the first batch, then don't bother recording residuals
        except TypeError:
            pass

    def _choose_next_batch(self):
        '''
        This method will choose a subset of the sampling space randomly and
        assign it to the `training_batch` attribute.. It will also remove
        anything we chose from the `self.sampling_space` attribute.
        '''
        random.shuffle(self.sampling_space)
        self._pop_next_batch()


class OmniscientAdsorptionDiscoverer(AdsorptionDiscoverer):
    '''
    This discoverer has perfect knowledge of all data points and chooses the
    best ones perfectly. No method can beat this, and as such it provides a
    ceiling of performance.
    '''
    def _train(self):
        '''
        There's no real training here. We just do the bare minimum:  make up
        some residuals and extend the training set.
        '''
        try:
            # The model is omnipotent, so the residuals will be zero.
            residuals = [0.] * len(self.training_batch)
            self.residuals.extend(residuals)
            self.training_set.extend(self.training_batch)

        # If this is the first batch, then don't bother recording residuals
        except TypeError:
            pass

    def _choose_next_batch(self):
        '''
        This method will choose the portion of the sampling space whose
        energies are nearest to the target assign it to the `training_batch`
        attribute. It will also remove anything we chose from the
        `self.sampling_space` attribute.
        '''
        self.sampling_space.sort(key=lambda doc: abs(doc['energy'] - self.optimal_value))
        self._pop_next_batch()


class TPOTGaussianAdsorptionDiscoverer(AdsorptionDiscoverer):
    '''
    This discoverer uses a Gaussian selection method with a TPOT model to select
    new sampling points.

    ...sorry for the awful code. This is a hack-job and I know it.
    '''
    # The width of the Gaussian selection curve
    stdev = 0.1

    def _train(self):
        '''
        Calculate the residuals of the current training batch, then retrain on
        everything
        '''
        # Instantiate the preprocessor and TPOT if we haven't done so already
        if not hasattr(self, 'preprocessor'):
            self._train_preprocessor()
        if not hasattr(self, 'tpot'):
            self.tpot = TPOTRegressor(generations=2,
                                      population_size=32,
                                      offspring_size=32,
                                      verbosity=2,
                                      scoring='neg_median_absolute_error',
                                      n_jobs=16,
                                      warm_start=True)
            features = self.preprocessor.transform(self.training_batch)
            energies = [doc['energy'] for doc in self.training_batch]
            self.tpot.fit(features, energies)

        # Calculate and save the residuals of this next batch
        features = self.preprocessor.transform(self.training_batch)
        tpot_predictions = self.tpot.predict(features)
        dft_energies = np.array([doc['energy'] for doc in self.training_batch])
        residuals = tpot_predictions - dft_energies
        self.residuals.extend(list(residuals))

        # Retrain
        self.training_set.extend(self.training_batch)
        self.__train_tpot()

    def _train_preprocessor(self):
        '''
        Trains the preprocessing pipeline and assigns it to the `preprocessor`
        attribute.
        '''
        # Open the cached preprocessor
        try:
            cache_name = 'caches/preprocessor.pkl'
            with open(cache_name, 'rb') as file_handle:
                self.preprocessor = pickle.load(file_handle)

        # If there is no cache, then remake it
        except FileNotFoundError:
            inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
            outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
            fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                                outer_fingerprinter)
            scaler = StandardScaler()
            pca = PCA()
            preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                               ('scaler', scaler),
                                               ('pca', pca)])
            preprocessing_pipeline.fit(self.training_batch)
            self.preprocessor = preprocessing_pipeline

            # Cache it for next time
            with open(cache_name, 'wb') as file_handle:
                pickle.dump(preprocessing_pipeline, file_handle)

    def __train_tpot(self):
        '''
        Train TPOT using the `training_set` attached to the class
        '''
        # Cache the current point for (manual) warm-starts, because there's a
        # solid chance that TPOT might cause a segmentation fault.
        cache_name = 'caches/%.3i_discovery_cache.pkl' % self.next_batch_number
        with open(cache_name, 'wb') as file_handle:
            cache = {'training_set': self.training_set,
                     'sampling_space': self.sampling_space,
                     'residuals': self.residuals,
                     'regret_history': self.regret_history,
                     'next_batch_number': self.next_batch_number,
                     'training_batch': self.training_batch}
            pickle.dump(cache, file_handle)

        # Instantiate the preprocessor and TPOT if we haven't done so already
        if not hasattr(self, 'preprocessor'):
            self._train_preprocessor()
        if not hasattr(self, 'tpot'):
            self.tpot = TPOTRegressor(generations=2,
                                      population_size=32,
                                      offspring_size=32,
                                      verbosity=2,
                                      scoring='neg_median_absolute_error',
                                      n_jobs=16,
                                      warm_start=True)

        # [Re-]train
        features = self.preprocessor.transform(self.training_set)
        energies = [doc['energy'] for doc in self.training_set]
        self.tpot.fit(features, energies)
        self.next_batch_number += 1

        # Try to address some memory issues by collecting garbage
        _ = gc.collect()  # noqa: F841

    def _choose_next_batch(self):
        '''
        Choose the next batch "randomly", where the probability of selecting
        sites are weighted using a combination of a Gaussian distribution and
        TPOT's prediction of their distance from the optimal energy. Snippets
        were stolen from the GASpy_feedback module.
        '''
        # Use the energies to calculate probabilities of selecting each site
        features = self.preprocessor.transform(self.sampling_space)
        energies = self.tpot.predict(features)
        gaussian_distribution = norm(loc=self.optimal_value, scale=self.stdev)
        probability_densities = [gaussian_distribution.pdf(energy) for energy in energies]

        # Perform a weighted shuffling of the sampling space such that sites
        # with better energies are more likely to be early in the list
        self.sampling_space = self.weighted_shuffle(self.sampling_space,
                                                    probability_densities)

        self._pop_next_batch

    @staticmethod
    def weighted_shuffle(sequence, weights):
        '''
        This function will shuffle a sequence using weights to increase the chances
        of putting higher-weighted elements earlier in the list. Credit goes to
        Nicky Van Foreest, whose function I based this off of.

        Args:
            sequence    A sequence of elements that you want shuffled
            weights     A sequence that is the same length as the `sequence` that
                        contains the corresponding probability weights for
                        selecting/choosing each element in `sequence`
        Returns:
            shuffled_list   A list whose elements are identical to those in the
                            `sequence` argument, but randomly shuffled such that
                            the elements with higher weights are more likely to
                            be in the front/start of the list.
        '''
        shuffled_list = np.empty_like(sequence)

        # Pack the elements in the sequences and their respective weights
        pairings = list(zip(sequence, weights))
        for i in range(len(pairings)):

            # Randomly choose one of the elements, and get the corresponding index
            cumulative_weights = np.cumsum([weight for _, weight in pairings])
            rand = random.random() * cumulative_weights[-1]
            j = bisect_right(cumulative_weights, rand)

            # Pop the element out so we don't re-select
            try:
                shuffled_list[i], _ = pairings.pop(j)

            # Hack a quick fix to some errors I don't feel like solving
            except IndexError:
                try:
                    shuffled_list[i], _ = pairings.pop(-1)
                except IndexError:
                    break

        return shuffled_list.tolist()


class BayesianOptimizer(AdsorptionDiscoverer):
    '''
    This "discoverer" is actually just a Bayesian optimizer for trying to find
    adsorption energies.
    '''

    def _train(self):
        '''
        Train the GP hyperparameters
        '''
        # Instantiate the preprocessor and GP if we haven't done so already
        if not hasattr(self, 'preprocessor'):
            self._train_preprocessor()

        # Calculate and save the residuals of this next batch
        try:
            ml_energies, _ = self.__make_predictions(self.training_batch)
            dft_energies = np.array([doc['energy'] for doc in self.training_batch])
            residuals = ml_energies - dft_energies
            self.residuals.extend(list(residuals))
        # If this is the very first training batch, then we don't need to save
        # the residuals
        except AttributeError:
            pass

        # Mandatory extension of the training set to include this next batch
        self.training_set.extend(self.training_batch)
        # Re-train on the whole training set
        self.__init_GP()
        _ = self.__train_GP()

    def _train_preprocessor(self):
        '''
        Trains the preprocessing pipeline and assigns it to the `preprocessor`
        attribute.
        '''
        # Open the cached preprocessor
        try:
            cache_name = 'caches/preprocessor.pkl'
            with open(cache_name, 'rb') as file_handle:
                self.preprocessor = pickle.load(file_handle)

        # If there is no cache, then remake it
        except FileNotFoundError:
            inner_fingerprinter = fingerprinters.InnerShellFingerprinter()
            outer_fingerprinter = fingerprinters.OuterShellFingerprinter()
            fingerprinter = fingerprinters.StackedFingerprinter(inner_fingerprinter,
                                                                outer_fingerprinter)
            scaler = StandardScaler()
            pca = PCA()
            preprocessing_pipeline = Pipeline([('fingerprinter', fingerprinter),
                                               ('scaler', scaler),
                                               ('pca', pca)])
            preprocessing_pipeline.fit(self.training_batch)
            self.preprocessor = preprocessing_pipeline

            # Cache it for next time
            with open(cache_name, 'wb') as file_handle:
                pickle.dump(preprocessing_pipeline, file_handle)

    def __init_GP(self):
        '''
        Initialize the exact GP model and assign the appropriate class attributes

        Returns:
            train_x     A `torch.Tensor` of the featurization of the current
                        training set
            train_y     A `torch.Tensor` of the output of the current training set
        '''
        # Grab the initial training data from the current (probably first)
        # training batch
        train_x = torch.Tensor(self.preprocessor.transform(self.training_set))
        train_y = torch.Tensor([doc['energy'] for doc in self.training_set])

        # Initialize the GP
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.GP = ExactGPModel(train_x, train_y, self.likelihood)

        # Optimize the GP hyperparameters
        self.GP.train()
        self.likelihood.train()

        # Set the optimizer that will tune parameters during training:  ADAM
        self.optimizer = torch.optim.Adam([{'params': self.GP.parameters()}], lr=0.1)

        # Set the "loss" function:  marginal log likelihood
        self.loss = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.GP)

        return train_x, train_y

    def __make_predictions(self, docs):
        '''
        Use the GP to make predictions on the current training batch

        Args:
            docs    A list of dictionaries that correspond to the sites you
                    want to make predictions on
        Returns:
            means   A numpy array giving the GP's mean predictions for the
                    `docs` you gave this method.
            stdevs  A numpy array giving the GP's standard deviation/standard
                    error predictions for the `docs` you gave this method.
        '''
        # Get into evaluation (predictive posterior) mode
        self.GP.eval()
        self.likelihood.eval()

        # Make the predictions
        features = torch.Tensor(self.preprocessor.transform(docs))
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            predictions = self.GP(features)

        # Format and return the predictions
        means = predictions.mean.cpu().detach().numpy()
        stdevs = predictions.stddev.cpu().detach().numpy()
        return means, stdevs

    def __train_GP(self):
        '''
        Re-trains the GP on all of the training data
        '''
        # Re-initialize the GP
        train_x, train_y = self.__init_GP()

        # If the loss increases too many times in a row, we will stop the
        # tuning short. Here we initialize some things to keep track of this.
        current_loss = float('inf')
        loss_streak = 0

        # Do at most 50 iterations of training
        for i in tqdm(range(50), desc='GP tuning'):

            # Zero backprop gradients
            self.optimizer.zero_grad()
            # Get output from model
            output = self.GP(train_x)
            # Calc loss and backprop derivatives
            loss = -self.loss(output, train_y)
            loss.backward()
            self.optimizer.step()

            # Stop training if the loss increases twice in a row
            new_loss = loss.item()
            if new_loss > current_loss:
                loss_streak += 1
                if loss_streak >= 2:
                    break
            else:
                current_loss = new_loss
                loss_streak = 0

    def _choose_next_batch(self):
        self.__sort_sampling_space_by_EI()
        self._pop_next_batch()

    def __sort_sampling_space_by_EI(self):
        '''
        Brute-force calculate the expected improvement for each of the sites in
        the sampling space.

        An explanation of the formulas used here can be found on
        http://krasserm.github.io/2018/03/21/bayesian-optimization/

        Here are the equations we used:
            EI = E[max(f(x) - f(x+), 0)]
               = (mu(x) - f(x+) - xi) * Phi(Z) + sigma(x)*phi(Z) if sigma(x) > 0
                                                                    sigma(x) == 0
            Z = (mu(x) - f(x+) - xi) / sigma(x)     if sigma(x) > 0
                0                                      sigma(x) == 0

        EI = expected improvement
        mu(x) = GP's estimate of the mean value at x
        sigma(x) = GP's standard error/standard deviation estimate at x
        f(x+) = best objective value observed so far
        xi = exploration/exploitation balance factor (higher value promotes exploration)
        Phi(Z) = cumulative distribution function of normal distribution at Z
        phi(Z) = probability distribution function of normal distribution at Z
        Z = test statistic at x
        '''
        # Initialize by getting all the GP predictions, the best energy so far,
        # and setting an exploration/exploitation value.
        means, stdevs = self.__make_predictions(self.sampling_space)
        f_best = min(abs(doc['energy'] - self.optimal_value)
                     for doc in self.training_set)
        xi = 0.01

        # Calculate EI for every single point we may sample
        for doc, mu, sigma in zip(self.sampling_space, means, stdevs):

            # Calculate the test statistic
            if sigma > 0:
                Z = (mu - f_best - xi) / sigma
            elif sigma == 0:
                Z = 0.
            else:
                raise RuntimeError('Got a negative standard error from the GP')

            # Calculate EI
            Phi = norm.cdf(Z)
            phi = norm.pdf(Z)
            if sigma > 0:
                EI = (mu - f_best - xi)*Phi + sigma*phi
            elif sigma == 0:
                EI = 0.

            # Save the EI results directly to the sampling space, then sort our
            # sampling space by it. High values of EI will show up first in the
            # list.
            doc['EI'] = EI
        self.sampling_space.sort(key=lambda doc: doc['EI'], reverse=True)


class ExactGPModel(gpytorch.models.ExactGP):
    '''
    We will use the simplest form of GP model with exact inference. This is
    taken from one of GPyTorch's tutorials.
    '''
    def __init__(self, train_x, train_y, likelihood):
        '''
        Args:
            train_x     A numpy array with your training features
            train_y     A numpy array with your training labels
            likelihood  An instance of one of the `gpytorch.likelihoods.*` classes
        '''
        # Convert the training data into tensors, which GPyTorch needs to run
        train_x = torch.Tensor(train_x)
        train_y = torch.Tensor(train_y)

        # Initialize the model
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def benchmark_adsorption_regret(discoverers):
    '''
    This function will take the regret curves of trained `AdsorptionDiscoverer`
    instances and the compare them to the floor and ceilings of performance for
    you.

    Arg:
        discoverers     A dictionary whose keys are the string names that you
                        want to label each discoverer with and whose values are
                        the trained instances of an `AdsorptionDiscoverer`
                        where the `simulate_discovery` methods for each are
                        already executed.
    Returns:
        regret_fig  The matplotlib figure object for the regret plot
        axes        A dictionary whose keys correspond to the names of the
                    discovery classes and whose values are the matplotlib axis
                    objects.
    '''
    # Initialize
    regret_fig = plt.figure()
    axes = {}

    # Fetch the data used by the first discoverer to "train" the floor/ceiling
    # models---i.e., get the baseline regret histories.
    example_discoverer_name = list(discoverers.keys())[0]
    example_discoverer = list(discoverers.values())[0]
    optimal_value = example_discoverer.optimal_value
    sampling_size = example_discoverer.batch_size * len(example_discoverer.regret_history)
    training_docs = deepcopy(example_discoverer.training_set[:sampling_size])
    sampling_docs = deepcopy(example_discoverer.training_set[-sampling_size:])

    # Plot the worst-case scenario
    random_discoverer = RandomAdsorptionDiscoverer(optimal_value,
                                                   training_docs,
                                                   sampling_docs)
    random_discoverer.simulate_discovery()
    sampling_sizes = [i*random_discoverer.batch_size
                      for i, _ in enumerate(random_discoverer.regret_history)]
    random_label = 'random selection (worst case)'
    axes[random_label] = plt.plot(sampling_sizes,
                                  random_discoverer.regret_history,
                                  '--r',
                                  label=random_label)

    # Plot the regret histories
    for name, discoverer in discoverers.items():
        sampling_sizes = [i*discoverer.batch_size
                          for i, _ in enumerate(discoverer.regret_history)]
        ax = sns.scatterplot(sampling_sizes, discoverer.regret_history, label=name)
        axes[name] = ax

    # Plot the best-case scenario
    omniscient_discoverer = OmniscientAdsorptionDiscoverer(optimal_value,
                                                           training_docs,
                                                           sampling_docs)
    omniscient_discoverer.simulate_discovery()
    omniscient_label = 'omniscient selection (ideal)'
    sampling_sizes = [i*omniscient_discoverer.batch_size
                      for i, _ in enumerate(omniscient_discoverer.regret_history)]
    axes[omniscient_label] = plt.plot(sampling_sizes,
                                      omniscient_discoverer.regret_history,
                                      '--b',
                                      label=omniscient_label)

    # Sort the legend correctly
    legend_info = {label: handle for handle, label in zip(*ax.get_legend_handles_labels())}
    labels = [random_label]
    labels.extend(list(discoverers.keys()))
    labels.append(omniscient_label)
    handles = [legend_info[label] for label in labels]
    ax.legend(handles, labels)

    # Formatting
    example_ax = axes[example_discoverer_name]
    # Labels axes
    _ = example_ax.set_xlabel('Number of discovery queries')  # noqa: F841
    _ = example_ax.set_ylabel('Cumulative regret [eV]')  # noqa: F841
    # Add commas to axes ticks
    _ = example_ax.get_xaxis().set_major_formatter(FORMATTER)
    _ = example_ax.get_yaxis().set_major_formatter(FORMATTER)
    # Set bounds/limits
    _ = example_ax.set_xlim([0, sampling_sizes[-1]])  # noqa: F841
    _ = example_ax.set_ylim([0, random_discoverer.regret_history[-1]])  # noqa: F841
    # Set figure size
    _ = regret_fig.set_size_inches(15, 5)  # noqa: F841
    return regret_fig
