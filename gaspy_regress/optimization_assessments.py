'''
This module is meant to be used to assess regression models for their
fitness-for-use in our active optimization workflows.
'''

__author__ = 'Kevin Tran'
__email__ = 'ktran@andrew.cmu.edu'

import sys
import math
import random
import warnings
from copy import deepcopy
import seaborn as sns
import matplotlib.pyplot as plt

# The tqdm autonotebook is still experimental, and it warns us. We don't care,
# and would rather not hear about the warning everytime.
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tqdm.autonotebook import tqdm


class ActiveOptimizer():
    '''
    This is a parent class for simulating active optimization routines. The
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
                            return them in a list. It should also remove
                            anything it selected from the `self.sampling_space`
                            attribute.
    '''
    def __init__(self, optimal_value, training_set, sampling_space, batch_size=200):
        '''
        Perform the initial training for the active optimizer, and then
        initialize all of the other attributes.

        Args:
            optimal_value   An object that represents the optimal value of
                            whatever it is you're optimizing. Should be used in
                            the `update_regret` method.
            training_set    A sequence that can be used to train/initialize the
                            surrogate model of the active optimizer.
            sampling_space  A sequence containing the rest of the possible
                            sampling space.
            batch_size      An integer indicating how many elements in the
                            sampling space you want to choose
        '''
        # Attributes we use to judge the optimization
        self.regret_history = [0.]
        self.residuals = []

        # Attributes we need to hallucinate the optimization
        self.optimal_value = optimal_value
        self.training_set = []
        self._train(list(deepcopy(training_set)))
        self.sampling_space = list(deepcopy(sampling_space))
        self.batch_size = batch_size

        # Attributes used in the `__assert_correct_hallucination` method
        self.__previous_training_set_len = len(self.training_set)
        self.__previous_sampling_space_len = len(self.sampling_space)
        self.__previous_regret_history_len = len(self.regret_history)
        self.__previous_residuals_len = len(self.residuals)

    def simulate_optimization(self):
        '''
        Perform the optimization simulation until all of the sampling space has
        been consumed.
        '''
        n_batches = math.ceil(len(self.sampling_space) / self.batch_size)
        for i in tqdm(range(n_batches), desc='Hallucinating optimization...'):
            self._hallucinate_next_batch()

    def _hallucinate_next_batch(self):
        '''
        Choose the next batch of data to get, add them to the `samples`
        attribute, and then re-train the surrogate model with the new samples.
        '''
        # Perform one iteration of active optimization
        batch = self._choose_next_batch()
        self._train(batch)
        self._update_regret(batch)

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
                       'a child-class of `ActiveOptimizer`, you need to '
                       'remove the chosen batch from the `sampling_space` '
                       'attribute.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the training set is being increased
        try:
            assert len(self.training_set) > self.__previous_training_set_len
            self.__previous_training_set_len = len(self.training_set)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveOptimizer`, you need to extend '
                       'the `training_set` attribute with the new training '
                       'batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure that the residuals are being recorded
        try:
            assert len(self.residuals) > self.__previous_residuals_len
            self.__previous_residuals_len = len(self.residuals)
        except AssertionError as error:
            message = ('\nWhen creating the `_train` method for a '
                       'child-class of `ActiveOptimizer`, you need to extend '
                       'the `residuals` attribute with the model\'s residuals '
                       'of the new batch (before retraining).')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

        # Make sure we are calculating the regret at each step
        try:
            assert len(self.regret_history) == self.__previous_regret_history_len + 1
            self.__previous_regret_history_len = len(self.regret_history)
        except AssertionError as error:
            message = ('\nWhen creating the `_update_regret` method for a '
                       'child-class of `ActiveOptimizer`, you need to append '
                       'the `regret_history` attribute with the cumulative '
                       'regret given the new batch.')
            raise type(error)(str(error) + message).with_traceback(sys.exc_info()[2])

    def plot_performance(self, n_violins=20):
        '''
        Light wrapper for plotting the regret an residuals over the course of
        the optimization.

        Arg:
            n_violins   The number of violins you want plotted in the residuals
                        figure
        Returns:
            regret_fig  The matplotlib figure object for the regret plot
            resid_fig   The matplotlib figure object for the residual plot
        '''
        regret_fig = self.plot_regret()
        resid_fig = self.plot_residuals(n_violins)
        return regret_fig, resid_fig

    def plot_regret(self):
        '''
        Plot the regret vs. optimization batch

        Returns:
            fig     The matplotlib figure object for the regret plot
        '''
        # Plot
        fig = plt.figure()
        sampling_sizes = [i*self.batch_size for i, _ in enumerate(self.regret_history)]
        ax = sns.scatterplot(sampling_sizes, self.regret_history)

        # Format
        _ = ax.set_xlabel('Number of optimization queries')  # noqa: F841
        _ = ax.set_ylabel('Cumulative regret [eV]')  # noqa: F841
        fig.set_size_inches(15, 5)
        return fig

    def plot_residuals(self, n_violins=20):
        '''
        Plot the residuals over time

        Arg:
            n_violins   The number of violins you want plotted
        Returns:
            fig     The matplotlib figure object for the residual plot
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
        fig.set_size_inches(15, 5)
        return fig


class AdsorptionOptimizer(ActiveOptimizer):
    '''
    Here we extend the `ActiveOptimizer` class while making the following
    assumptions:  1) we are trying to optimize the adsorption energy and 2) our
    inputs are a list of dictionaries with the 'energy' key.
    '''
    def _update_regret(self, batch):
        '''
        Calculates the cumulative regret of the optimization thus far. Assumes
        that your sampling space is a list of dictionaries, and that the key
        you want to optimize is 'energy'.

        Arg:
            batch   The output of whatever the `choose_next_batch` method
                    returns
        '''
        # Find the current regret
        regret = self.regret_history[-1]

        # Add the new regret
        for doc in batch:
            energy = doc['energy']
            difference = energy - self.optimal_value
            regret += abs(difference)
        self.regret_history.append(regret)


class RandomAdsorptionOptimizer(AdsorptionOptimizer):
    '''
    This optimizer simply chooses new samples randomly. This is intended to be
    used as a baseline for active optimization.
    '''
    def _train(self, training_batch):
        '''
        There's no real training here. We just do the bare minimum:  make up
        some residuals and extend the training set.
        '''
        # We'll just arbitrarily set the "model's" guess to the current average
        # of the training set
        energies = [doc['energy'] for doc in self.training_set]
        energy_guess = sum(energies) / max(len(energies), 1)
        residuals = [energy_guess - doc['energy'] for doc in training_batch]
        self.residuals.extend(residuals)

        self.training_set.extend(training_batch)

    def _choose_next_batch(self):
        '''
        This method will choose a subset of the sampling space randomly. It
        will also remove anything we chose from the `self.sampling_space`
        attribute.

        Returns:
            samples     A list of length `self.batch_size` whose elements are
                        random elements from the `self.sample_space` argument
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
        return samples
