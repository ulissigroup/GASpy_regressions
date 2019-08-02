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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tpot import TPOTRegressor
from gaspy_regress import fingerprinters

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


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

    def plot_performance(self, n_violins=20):
        '''
        Light wrapper for plotting the regret an residuals over the course of
        the discovery.

        Arg:
            n_violins   The number of violins you want plotted in the residuals
                        figure
        Returns:
            regret_fig  The matplotlib figure object for the regret plot
            resid_fig   The matplotlib figure object for the residual plot
        '''
        regret_fig = self.plot_regret()
        learning_fig = self.plot_learning_curve(n_violins)
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
        return fig

    def plot_learning_curve(self, n_violins=20):
        '''
        Plot the residuals over time

        Arg:
            n_violins   The number of violins you want plotted
        Returns:
            fig     The matplotlib figure object for the learning curve
        '''
        # Figure out what violin number to label each residual with
        chunked_residuals = [self.residuals[i::n_violins] for i in range(n_violins)]
        violin_numbers = []
        for violin_number, chunk in enumerate(chunked_residuals):
            violin_numbers.extend([violin_number for _ in chunk])

        # Create and format the figure
        fig = plt.figure()
        ax = sns.violinplot(violin_numbers, self.residuals, cut=0, scale='width')
        _ = ax.set_xlabel('Optimization portion (~time)')  # noqa:  F841
        _ = ax.set_ylabel('Residuals [eV]')  # noqa:  F841
        _ = fig.set_size_inches(15, 5)  # noqa: F841

        # Add a line at perfect residuals
        _ = ax.plot([-1, max(violin_numbers)+1], [0, 0], '--')  # noqa: F841
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
        _ = ax.set_ylabel('TPOT-predicted adsorption energy [eV]')  # noqa: F841
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
        samples = []

        # Simultaneously choose `self.batch_size` samples while removing them
        # from the sampling space
        for _ in range(self.batch_size):
            try:
                sample = self.sampling_space.pop()
                samples.append(sample)
            except IndexError:
                break
        self.training_batch = samples
        self.next_batch_number += 1


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
        self.sampling_space.sort(key=lambda doc: abs(doc['energy'] - self.optimal_value),
                                 reverse=True)
        samples = []

        # Simultaneously choose `self.batch_size` samples while removing them
        # from the sampling space
        for _ in range(self.batch_size):
            try:
                sample = self.sampling_space.pop()
                samples.append(sample)
            except IndexError:
                break
        self.training_batch = samples
        self.next_batch_number += 1


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
        self._train_tpot()

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

    def _train_tpot(self):
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
        self.sampling_space = weighted_shuffle(self.sampling_space, probability_densities)

        # Simultaneously choose `self.batch_size` samples while removing them
        # from the sampling space
        samples = []
        for _ in range(self.batch_size):
            try:
                sample = self.sampling_space.pop(0)
                samples.append(sample)
            except IndexError:
                break
        self.training_batch = samples


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
