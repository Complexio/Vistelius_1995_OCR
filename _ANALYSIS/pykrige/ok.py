from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

__doc__ = """
PyKrige
=======

Code by Benjamin S. Murphy and the PyKrige Developers
bscott.murphy@gmail.com

Summary
-------
Contains class OrdinaryKriging, which provides easy access to
2D Ordinary Kriging.

References
----------
.. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
    Hydrogeology, (Cambridge University Press, 1997) 272 p.

Copyright (c) 2015-2018, PyKrige Developers
"""

import numpy as np
import scipy.linalg
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from . import variogram_models
from . import core
from .core import _adjust_for_anisotropy, _initialize_variogram_model, \
    _make_variogram_parameter_list, _find_statistics
import warnings

from joblib import Parallel, delayed
import multiprocessing


class OrdinaryKriging:
    """Convenience class for easy access to 2D Ordinary Kriging.

    Parameters
    ----------
    x : array_like
        X-coordinates of data points.
    y : array_like
        Y-coordinates of data points.
    z : array-like
        Values at data points.
    variogram_model : str, optional
        Specifies which variogram model to use; may be one of the following:
        linear, power, gaussian, spherical, exponential, hole-effect.
        Default is linear variogram model. To utilize a custom variogram model,
        specify 'custom'; you must also provide variogram_parameters and
        variogram_function. Note that the hole-effect model is only technically
        correct for one-dimensional problems.
    variogram_parameters : list or dict, optional
        Parameters that define the specified variogram model. If not provided,
        parameters will be automatically calculated using a "soft" L1 norm
        minimization scheme. For variogram model parameters provided in a dict,
        the required dict keys vary according to the specified variogram
        model: ::
            linear - {'slope': slope, 'nugget': nugget}
            power - {'scale': scale, 'exponent': exponent, 'nugget': nugget}
            gaussian - {'sill': s, 'range': r, 'nugget': n}
                        OR
                       {'psill': p, 'range': r, 'nugget':n}
            spherical - {'sill': s, 'range': r, 'nugget': n}
                         OR
                        {'psill': p, 'range': r, 'nugget':n}
            exponential - {'sill': s, 'range': r, 'nugget': n}
                           OR
                          {'psill': p, 'range': r, 'nugget':n}
            hole-effect - {'sill': s, 'range': r, 'nugget': n}
                           OR
                          {'psill': p, 'range': r, 'nugget':n}
        Note that either the full sill or the partial sill
        (psill = sill - nugget) can be specified in the dict.
        For variogram model parameters provided in a list, the entries
        must be as follows: ::
            linear - [slope, nugget]
            power - [scale, exponent, nugget]
            gaussian - [sill, range, nugget]
            spherical - [sill, range, nugget]
            exponential - [sill, range, nugget]
            hole-effect - [sill, range, nugget]
        Note that the full sill (NOT the partial sill) must be specified
        in the list format.
        For a custom variogram model, the parameters are required, as custom
        variogram models will not automatically be fit to the data.
        Furthermore, the parameters must be specified in list format, in the
        order in which they are used in the callable function (see
        variogram_function for more information). The code does not check
        that the provided list contains the appropriate number of parameters
        for the custom variogram model, so an incorrect parameter list in
        such a case will probably trigger an esoteric exception someplace
        deep in the code.
        NOTE that, while the list format expects the full sill, the code
        itself works internally with the partial sill.
    variogram_function : callable, optional
        A callable function that must be provided if variogram_model is
        specified as 'custom'. The function must take only two arguments:
        first, a list of parameters for the variogram model; second, the
        distances at which to calculate the variogram model. The list
        provided in variogram_parameters will be passed to the function
        as the first argument.
    nlags : int, optional
        Number of averaging bins for the semivariogram. Default is 6.
    weight : bool, optional
        Flag that specifies if semivariance at smaller lags should be weighted
        more heavily when automatically calculating variogram model.
        The routine is currently hard-coded such that the weights are
        calculated from a logistic function, so weights at small lags are ~1
        and weights at the longest lags are ~0; the center of the logistic
        weighting is hard-coded to be at 70% of the distance from the shortest
        lag to the largest lag. Setting this parameter to True indicates that
        weights will be applied. Default is False. (Kitanidis suggests that the
        values at smaller lags are more important in fitting a variogram model,
        so the option is provided to enable such weighting.)
    anisotropy_scaling : float, optional
        Scalar stretching value to take into account anisotropy.
        Default is 1 (effectively no stretching).
        Scaling is applied in the y-direction in the rotated data frame
        (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
        is not 0). This parameter has no effect if coordinate_types is
        set to 'geographic'.
    anisotropy_angle : float, optional
        CCW angle (in degrees) by which to rotate coordinate system in
        order to take into account anisotropy. Default is 0 (no rotation).
        Note that the coordinate system is rotated. This parameter has
        no effect if coordinate_types is set to 'geographic'.
    verbose : bool, optional
        Enables program text output to monitor kriging process.
        Default is False (off).
    enable_plotting : bool, optional
        Enables plotting to display variogram. Default is False (off).
    enable_statistics : bool, optional
        Default is False
    coordinates_type : str, optional
        One of 'euclidean' or 'geographic'. Determines if the x and y
        coordinates are interpreted as on a plane ('euclidean') or as
        coordinates on a sphere ('geographic'). In case of geographic
        coordinates, x is interpreted as longitude and y as latitude
        coordinates, both given in degree. Longitudes are expected in
        [0, 360] and latitudes in [-90, 90]. Default is 'euclidean'.

    References
    ----------
    .. [1] P.K. Kitanidis, Introduction to Geostatistcs: Applications in
        Hydrogeology, (Cambridge University Press, 1997) 272 p.
    """

    eps = 1.e-10   # Cutoff for comparison to zero
    variogram_dict = {'linear': variogram_models.linear_variogram_model,
                      'power': variogram_models.power_variogram_model,
                      'gaussian': variogram_models.gaussian_variogram_model,
                      'spherical': variogram_models.spherical_variogram_model,
                      'exponential': variogram_models.exponential_variogram_model,
                      'hole-effect': variogram_models.hole_effect_variogram_model}

    def __init__(self, x, y, z, variogram_model='linear',
                 variogram_parameters=None, variogram_function=None, nlags=6,
                 weight=False, anisotropy_scaling=1.0, anisotropy_angle=0.0,
                 verbose=False, enable_plotting=False, enable_statistics=False,
                 coordinates_type='euclidean', saveplot=False, saveloc=None,
                 principal_component=None, pluton=None, show_nlag_pairs=False,
                 show_range_determine_guide=False, range_estimate=None):

        # Code assumes 1D input arrays of floats. Ensures that any extraneous
        # dimensions don't get in the way. Copies are created to avoid any
        # problems with referencing the original passed arguments.
        # Also, values are forced to be float... in the future, might be worth
        # developing complex-number kriging (useful for vector field kriging)
        self.X_ORIG = \
            np.atleast_1d(np.squeeze(np.array(x, copy=True, dtype=np.float64)))
        self.Y_ORIG = \
            np.atleast_1d(np.squeeze(np.array(y, copy=True, dtype=np.float64)))
        self.Z = \
            np.atleast_1d(np.squeeze(np.array(z, copy=True, dtype=np.float64)))

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        self.variance = np.var(z, ddof=1)
        self.saveplot = saveplot
        self.saveloc = saveloc
        self.component = principal_component
        self.pluton = pluton
        self.nlags = nlags
        self.show_nlag_pairs = show_nlag_pairs
        self.show_range_determine_guide = show_range_determine_guide
        self.range_estimate = range_estimate

        # adjust for anisotropy... only implemented for euclidean (rectangular)
        # coordinates, as anisotropy is ambiguous for geographic coordinates...
        if coordinates_type == 'euclidean':
            self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG))/2.0
            self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG))/2.0
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            if self.verbose:
                print("Adjusting data for anisotropy...")
            self.X_ADJUSTED, self.Y_ADJUSTED = \
                _adjust_for_anisotropy(np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                                       [self.XCENTER, self.YCENTER],
                                       [self.anisotropy_scaling],
                                       [self.anisotropy_angle]).T
        elif coordinates_type == 'geographic':
            # Leave everything as is in geographic case.
            # May be open to discussion?
            if anisotropy_scaling != 1.0:
                warnings.warn("Anisotropy is not compatible with geographic "
                              "coordinates. Ignoring user set anisotropy.",
                              UserWarning)
            self.XCENTER = 0.0
            self.YCENTER = 0.0
            self.anisotropy_scaling = 1.0
            self.anisotropy_angle = 0.0
            self.X_ADJUSTED = self.X_ORIG
            self.Y_ADJUSTED = self.Y_ORIG
        else:
            raise ValueError("Only 'euclidean' and 'geographic' are valid "
                             "values for coordinates-keyword.")
        self.coordinates_type = coordinates_type

        # set up variogram model and parameters...
        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and \
                        self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' "
                             "is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for "
                                 "custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]

        if self.verbose:
            print("Initializing variogram model...")

        vp_temp = _make_variogram_parameter_list(self.variogram_model,
                                                 variogram_parameters)

        # for item in _initialize_variogram_model(np.vstack((self.X_ADJUSTED,
        #                                        self.Y_ADJUSTED)).T,
        #                             self.Z, self.variogram_model, vp_temp,
        #                             self.variogram_function, nlags,
        #                             weight, self.coordinates_type):
        #     print(item)
        self.lags, self.semivariance, self.variogram_model_parameters, self.n_pairs = \
            _initialize_variogram_model(np.vstack((self.X_ADJUSTED,
                                                   self.Y_ADJUSTED)).T,
                                        self.Z, self.variogram_model, vp_temp,
                                        self.variogram_function, nlags,
                                        weight, self.coordinates_type)

        if self.verbose:
            print("Coordinates type: '%s'" % self.coordinates_type, '\n')
            if self.variogram_model == 'linear':
                print("Using '%s' Variogram Model" % 'linear')
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], '\n')
            elif self.variogram_model == 'power':
                print("Using '%s' Variogram Model" % 'power')
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
            elif self.variogram_model == 'custom':
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print("Full Sill:", self.variogram_model_parameters[0] +
                      self.variogram_model_parameters[2])
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        if enable_statistics:
            self.delta, self.sigma, self.epsilon = \
                _find_statistics(np.vstack((self.X_ADJUSTED,
                                            self.Y_ADJUSTED)).T,
                                 self.Z, self.variogram_function,
                                 self.variogram_model_parameters,
                                 self.coordinates_type)
            self.Q1 = core.calcQ1(self.epsilon)
            self.Q2 = core.calcQ2(self.epsilon)
            self.cR = core.calc_cR(self.Q2, self.sigma)
            if self.verbose:
                print("Q1 =", self.Q1)
                print("Q2 =", self.Q2)
                print("cR =", self.cR, '\n')
        else:
            self.delta, self.sigma, self.epsilon, self.Q1, self.Q2, self.cR = [None]*6

    def update_variogram_model(self, variogram_model,
                               variogram_parameters=None,
                               variogram_function=None, nlags=6, weight=False,
                               anisotropy_scaling=1., anisotropy_angle=0.):
        """Allows user to update variogram type and/or
        variogram model parameters.

        Parameters
        __________
        variogram_model (string): May be any of the variogram models listed
            above. May also be 'custom', in which case variogram_parameters
            and variogram_function must be specified.
        variogram_parameters (list or dict, optional): List or dict of
            variogram model parameters, as explained above. If not provided,
            a best fit model will be calculated as described above.
        variogram_function (callable, optional): A callable function that must
            be provided if variogram_model is specified as 'custom'.
            See above for more information.
        nlags (int, optional): Number of averaging bins for the semivariogram.
            Default is 6.
        weight (boolean, optional): Flag that specifies if semivariance at
            smaller lags should be weighted more heavily when automatically
            calculating the variogram model. See above for more information.
            True indicates that weights will be applied. Default is False.
        anisotropy_scaling (float, optional): Scalar stretching value to
            take into account anisotropy. Default is 1 (effectively no
            stretching). Scaling is applied in the y-direction.
        anisotropy_angle (float, optional): CCW angle (in degrees) by
            which to rotate coordinate system in order to take into
            account anisotropy. Default is 0 (no rotation).
        """

        if anisotropy_scaling != self.anisotropy_scaling or \
           anisotropy_angle != self.anisotropy_angle:
            if self.coordinates_type == 'euclidean':
                if self.verbose:
                    print("Adjusting data for anisotropy...")
                self.anisotropy_scaling = anisotropy_scaling
                self.anisotropy_angle = anisotropy_angle
                self.X_ADJUSTED, self.Y_ADJUSTED = \
                    _adjust_for_anisotropy(np.vstack((self.X_ORIG, self.Y_ORIG)).T,
                                           [self.XCENTER, self.YCENTER],
                                           [self.anisotropy_scaling],
                                           [self.anisotropy_angle]).T
            elif self.coordinates_type == 'geographic':
                if anisotropy_scaling != 1.0:
                    warnings.warn("""Anisotropy is not compatible with
                                     geographic coordinates. Ignoring user set
                                     anisotropy.""", UserWarning)
                self.anisotropy_scaling = 1.0
                self.anisotropy_angle = 0.0
                self.X_ADJUSTED = self.X_ORIG
                self.Y_ADJUSTED = self.Y_ORIG

        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]
        if self.verbose:
            print("Updating variogram mode...")

        # See note above about the 'use_psill' kwarg...
        vp_temp = _make_variogram_parameter_list(self.variogram_model,
                                                 variogram_parameters)
        self.lags, self.semivariance, self.variogram_model_parameters = \
            _initialize_variogram_model(np.vstack((self.X_ADJUSTED,
                                                   self.Y_ADJUSTED)).T,
                                        self.Z, self.variogram_model, vp_temp,
                                        self.variogram_function, nlags,
                                        weight, self.coordinates_type)

        if self.verbose:
            print("Coordinates type: '%s'" % self.coordinates_type, '\n')
            if self.variogram_model == 'linear':
                print("Using '%s' Variogram Model" % 'linear')
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], '\n')
            elif self.variogram_model == 'power':
                print("Using '%s' Variogram Model" % 'power')
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
            elif self.variogram_model == 'custom':
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Partial Sill:", self.variogram_model_parameters[0])
                print("Full Sill:", self.variogram_model_parameters[0] +
                      self.variogram_model_parameters[2])
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = \
            _find_statistics(np.vstack((self.X_ADJUSTED, self.Y_ADJUSTED)).T,
                             self.Z, self.variogram_function,
                             self.variogram_model_parameters,
                             self.coordinates_type)
        self.Q1 = core.calcQ1(self.epsilon)
        self.Q2 = core.calcQ2(self.epsilon)
        self.cR = core.calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, '\n')

    def display_variogram_model(self):
        """Displays variogram model with the actual binned data."""
        fig = plt.figure()
        ax = fig.add_subplot(111)

        # Added to plot the nugget as well ------------------------------------
        lags_plot = np.insert(self.lags, 0, 0)
        # semivariance_plot = np.insert(self.semivariance, 0,
        #                               self.variogram_model_parameters[2])

        ax.plot(self.lags, self.semivariance, 'r*',
                label="experimental variogram")

        ax.plot(lags_plot,
                self.variogram_function(self.variogram_model_parameters,
                                        lags_plot),
                'k-',
                label=f"{self.variogram_model} model variogram")
        # -----------------------------------------------------------------------

        # ax.plot(self.lags, self.semivariance, 'r*', label="experimental variogram")
        # ax.plot(self.lags,
        #         self.variogram_function(self.variogram_model_parameters,
        #                                 self.lags), 'k-', label=f"{self.variogram_model} model variogram")
        ax.hlines(self.variance, 0, self.lags[-1], 'g')
        # plt.text(0,
        #          self.variance * 1.05,
        #          f"Variance: {self.variance:.3f}",
        #          color='g')
        text = f"Partial sill: {self.variogram_model_parameters[0]:.3f}\nRange: {self.variogram_model_parameters[1]:.3f}\nNugget: {self.variogram_model_parameters[2]:.3f}\nnlags: {self.nlags}"

        if self.show_range_determine_guide:
            ax.hlines(0.95 * self.variance, 0, self.lags[-1], 'b')

        if self.show_nlag_pairs:
            for lag, semi, n in zip(self.lags,
                                    self.semivariance,
                                    self.n_pairs):
                plt.text(lag*1.01, semi, f"{n:4.0f}", color='grey', fontsize=8)

        if self.range_estimate:
            ax.vlines(self.range_estimate, 0, self.variance, 'm')

        # plt.text(0.05, 0.75, text, transform=ax.transAxes)
        plt.xlabel("Lag")
        plt.ylabel(r"$\gamma_{h}$")
        plt.title(f"{self.pluton}_{self.component}")
        plt.legend(loc='lower center',
                   fontsize=8,
                   bbox_to_anchor=(0.5, -0.275),
                   ncol=2)
        # plt.ylim((0, self.semivariance[-1] * 1.05))
        if self.saveplot:
            plt.savefig(f"{self.saveloc}{self.pluton}_variogram_{self.variogram_model}_{self.component}.pdf", bbox_inches = "tight")
        plt.show()

    def get_variogram_points(self):
        """Returns both the lags and the variogram function evaluated at each
        of them.
        The evaluation of the variogram function and the lags are produced
        internally. This method is convenient when the user wants to access to
        the lags and the resulting variogram (according to the model provided)
        for further analysis.
        Returns
        -------
        (tuple) tuple containing:
            lags (array) - the lags at which the variogram was evaluated
            variogram (array) - the variogram function evaluated at the lags
        """
        return self.lags, self.variogram_function(
                                self.variogram_model_parameters,
                                self.lags)

    def switch_verbose(self):
        """Allows user to switch code talk-back on/off. Takes no arguments."""
        self.verbose = not self.verbose

    def switch_plotting(self):
        """Allows user to switch plot display on/off. Takes no arguments."""
        self.enable_plotting = not self.enable_plotting

    def get_epsilon_residuals(self):
        """Returns the epsilon residuals for the variogram fit."""
        return self.epsilon

    def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        plt.show()

    def get_statistics(self):
        """Returns the Q1, Q2, and cR statistics for the variogram fit
        (in that order). No arguments.
        """
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        """Prints out the Q1, Q2, and cR statistics for the variogram fit.
        NOTE that ideally Q1 is close to zero, Q2 is close to 1,
        and cR is as small as possible.
        """
        print("Q1 =", self.Q1)
        print("Q2 =", self.Q2)
        print("cR =", self.cR)

    def _get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""

        xy = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                             self.Y_ADJUSTED[:, np.newaxis]), axis=1)
        d = cdist(xy, xy, 'euclidean')
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - self.variogram_function(self.variogram_model_parameters,
                                              d)
        np.fill_diagonal(a, 0.)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

    def _exec_vector(self, a, bd, mask):
        """Solves the kriging system as a vectorized operation. This method
        can take a lot of memory for large grids and/or large datasets."""

        npt = bd.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zero_index = None
        zero_value = False

        a_inv = scipy.linalg.inv(a)

        if np.any(np.absolute(bd) <= self.eps):
            zero_value = True
            zero_index = np.where(np.absolute(bd) <= self.eps)

        b = np.zeros((npt, n+1, 1))
        b[:, :n, 0] = \
            - self.variogram_function(self.variogram_model_parameters, bd)
        if zero_value:
            b[zero_index[0], zero_index[1], 0] = 0.0
        b[:, n, 0] = 1.0

        if (~mask).any():
            mask_b = np.repeat(mask[:, np.newaxis, np.newaxis], n+1, axis=1)
            b = np.ma.array(b, mask=mask_b)

        x = np.dot(a_inv, b.reshape((npt, n+1)).T).reshape((1, n+1, npt)).T
        zvalues = np.sum(x[:, :n, 0] * self.Z, axis=1)
        sigmasq = np.sum(x[:, :, 0] * -b[:, :, 0], axis=1)

        return zvalues, sigmasq

    def _exec_loop(self, a, bd_all, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        verboses = True

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        if verboses:
            print("npt", npt)
            print("n", n)
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        a_inv = scipy.linalg.inv(a)

        for j in np.nonzero(~mask)[0]:   # Note that this is the same thing as range(npt) if mask is not defined,
            bd = bd_all[j]               # otherwise it takes the non-masked elements.
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            b = np.zeros((n+1, 1))
            b[:n, 0] = \
                - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0
            x = np.dot(a_inv, b)

            if verboses:
                print("a_inv", a_inv)
                print("x", x)
                print("x[:n, 0]", x[:n, 0])
                print("b", b)
                print("self.Z", self.Z)
                print("b[:, 0]", b[:, 0])
                print("b.shape", b.shape)

            zvalues[j] = np.sum(x[:n, 0] * self.Z)
            # same as doing np.dot(x[:n, 0], self.Z)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])
            # same as doing np.dot(x[:, 0], -b[:, 0])

        return zvalues, sigmasq

    def _exec_loop_moving_window(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""
        import scipy.linalg.lapack

        verboses = False
        implement_nugget = False

        npt = bd_all.shape[0]
        if verboses:
            # print(self.variogram_model_parameters[1])
            print("npt", npt)
            print("bd_idx.shape", bd_idx.shape)
            print("a_all", a_all)
            print("np.array([a_all.shape[0] - 1])",
                  np.array([a_all.shape[0] - 1]))
        # Number of relevant initial points for one grid point
        n = bd_idx.shape[1]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        # Note that this is the same thing as range(npt) if mask is not
        # defined, otherwise it takes the non-masked elements.
        for i in np.nonzero(~mask)[0]:

            # Indices of relevant initial points
            b_selector = bd_idx[i]
            # print(b_selector[~np.isnan(b_selector)])
            b_selector = b_selector[~np.isnan(b_selector)]
            b_selector = b_selector.astype('int')

            # Lag distances of relevant initial points
            bd = bd_all[i]
            bd = bd[~np.isnan(bd)]

            # print(np.array([a_all.shape[0] - 1]))

            a_selector = np.concatenate((b_selector,
                                         np.array([a_all.shape[0] - 1])))
            if verboses:
                print("a_selector", a_selector)
            a = a_all[a_selector[:, None], a_selector]

            # Set the diagonal of the kriging matrix equal to the nugget
            if implement_nugget:
                a[np.diag_indices(a.shape[0])] = \
                    self.variogram_model_parameters[-1]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            n = b_selector[~np.isnan(b_selector)].shape[0]
            if verboses:
                print("a", a)
                print()
                print("n", n)
            # Semivariances according to chosen variogram model
            b = np.zeros((n+1, 1))
            b[:n, 0] = \
                - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            if verboses:
                print("b", b)
                print("b_selector.shape", b_selector.shape)
                print("self.Z[b_selector]", self.Z[b_selector])
                print("b[:, 0]", b[:, 0])
                print("b.shape", b.shape)

            # kriging weights
            """If there are too few (only zero or one) points found within the search radius, the linalg.solve will throw 'Matrix is singular' error since it the system of equations cannot be solved. In this case fill the kriging werights matrix (x) with nan values to force the solution to be zero for both the estimations as the estimation variances."""
            try:
                x = scipy.linalg.solve(a, b)
            except np.linalg.LinAlgError as err:
                if 'Matrix is singular' in str(err):
                    x = np.empty((n+1, 1)) * np.nan
                    # print(n)
                    # print(x.shape)
                else:
                    raise err

            if verboses:
                print("x", x)
                print("x[:n, 0]", x[:n, 0])
            # kriging estimation values
            zvalues[i] = x[:n, 0].dot(self.Z[b_selector])
            # kriging estimation variance
            sigmasq[i] = - x[:, 0].dot(b[:, 0])

        return zvalues, sigmasq

    def _exec_loop_moving_window_parallel(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""
        import scipy.linalg.lapack

        verboses = True

        npt = bd_all.shape[0]
        if verboses:
            print("npt", npt)
            print("bd_idx.shape", bd_idx.shape)
        n = bd_idx.shape[1]
        zvalues = []
        sigmasq = []

        def loop_moving_window_parallel_calculations(self, i, npt, n, zvalues,
                                                     sigmasq, a_all, bd_all,
                                                     mask, bd_idx):
            b_selector = bd_idx[i]
            # print(b_selector[~np.isnan(b_selector)])
            b_selector = b_selector[~np.isnan(b_selector)]
            b_selector = b_selector.astype('int')

            bd = bd_all[i]
            bd = bd[~np.isnan(bd)]

            # print(np.array([a_all.shape[0] - 1]))
            a_selector = np.concatenate((b_selector,
                                         np.array([a_all.shape[0] - 1])))
            if verboses:
                print("a_selector", a_selector)
            a = a_all[a_selector[:, None], a_selector]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            n = b_selector[~np.isnan(b_selector)].shape[0]
            if verboses:
                print("n", n)
            b = np.zeros((n+1, 1))
            b[:n, 0] = \
                - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0
            # print(a)
            # print(b)

            x = scipy.linalg.solve(a, b)
            if verboses:
                print(x)
                print("b_selector.shape", b_selector.shape)
                print("b.shape", b.shape)
            zvalues.append(x[:n, 0].dot(self.Z[b_selector]))
            sigmasq.append(- x[:, 0].dot(b[:, 0]))

            return np.array(zvalues), np.array(sigmasq)

        parallel_results = Parallel(n_jobs=-1)(delayed(
            loop_moving_window_parallel_calculations)(self,
                                                      i,
                                                      npt,
                                                      n,
                                                      zvalues,
                                                      sigmasq,
                                                      a_all,
                                                      bd_all,
                                                      mask,
                                                      bd_idx)
            for i in np.nonzero(~mask)[0])
        # Note that this is the same thing as range(npt) if mask is not
        # defined, otherwise it takes the non-masked elements.

        print(np.array(parallel_results)[:, 0].shape)

        return np.array(parallel_results)[:, 0], \
            np.array(parallel_results)[:, 1]

    def execute(self, style, xpoints, ypoints, mask=None, backend='vectorized',
                n_closest_points=None, search_radius=None):
        """Calculates a kriged grid and the associated variance.

        This is now the method that performs the main kriging calculation.
        Note that currently measurements (i.e., z values) are considered
        'exact'. This means that, when a specified coordinate for interpolation
        is exactly the same as one of the data points, the variogram evaluated
        at the point is forced to be zero. Also, the diagonal of the kriging
        matrix is also always forced to be zero. In forcing the variogram
        evaluated at data points to be zero, we are effectively saying that
        there is no variance at that point (no uncertainty,
        so the value is 'exact').

        In the future, the code may include an extra 'exact_values' boolean
        flag that can be adjusted to specify whether to treat the measurements
        as 'exact'. Setting the flag to false would indicate that the
        variogram should not be forced to be zero at zero distance
        (i.e., when evaluated at data points). Instead, the uncertainty in
        the point will be equal to the nugget. This would mean that the
        diagonal of the kriging matrix would be set to
        the nugget instead of to zero.

        Parameters
        ----------
        style : str
            Specifies how to treat input kriging points. Specifying 'grid'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid. Specifying 'points' treats
            xpoints and ypoints as two arrays that provide coordinate pairs
            at which to solve the kriging system. Specifying 'masked'
            treats xpoints and ypoints as two arrays of x and y coordinates
            that define a rectangular grid and uses mask to only evaluate
            specific points in the grid.
        xpoints : array_like, shape (N,) or (N, 1)
            If style is specific as 'grid' or 'masked',
            x-coordinates of MxN grid. If style is specified as 'points',
            x-coordinates of specific points at which to solve
            kriging system.
        ypoints : array_like, shape (M,) or (M, 1)
            If style is specified as 'grid' or 'masked',
            y-coordinates of MxN grid. If style is specified as 'points',
            y-coordinates of specific points at which to solve kriging
            system. Note that in this case, xpoints and ypoints must have
            the same dimensions (i.e., M = N).
        mask : bool, array_like, shape (M, N), optional
            Specifies the points in the rectangular grid defined
            by xpoints and ypoints that are to be excluded in the
            kriging calculations. Must be provided if style is specified
            as 'masked'. False indicates that the point should not be
            masked, so the kriging system will be solved at the point.
            True indicates that the point should be masked, so the kriging
            system should will not be solved at the point.
        backend : str, optional
            Specifies which approach to use in kriging.
            Specifying 'vectorized' will solve the entire kriging problem
            at once in a vectorized operation. This approach is faster but
            also can consume a significant amount of memory for large grids
            and/or large datasets. Specifying 'loop' will loop through each
            point at which the kriging system is to be solved.
            This approach is slower but also less memory-intensive.
            Specifying 'C' will utilize a loop in Cython.
            Default is 'vectorized'.
        n_closest_points : int, optional
            For kriging with a moving window, specifies the number of
            nearby points to use in the calculation. This can speed up the
            calculation for large datasets, but should be used
            with caution. As Kitanidis notes, kriging with a moving window
            can produce unexpected oddities if the variogram model
            is not carefully chosen.

        Returns
        -------
        zvalues : ndarray, shape (M, N) or (N, 1)
            Z-values of specified grid or at the specified set of points.
            If style was specified as 'masked', zvalues will
            be a numpy masked array.
        sigmasq : ndarray, shape (M, N) or (N, 1)
            Variance at specified grid points or at the specified
            set of points. If style was specified as 'masked', sigmasq
            will be a numpy masked array.
        """

        verboses = False

        # if self.verbose:
        #     print("Executing Ordinary Kriging...\n")

        if style != 'grid' and style != 'masked' and style != 'points':
            raise ValueError("style argument must be 'grid', "
                             "'points', or 'masked'")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        a = self._get_kriging_matrix(n)

        if style in ['grid', 'masked']:
            if style == 'masked':
                if mask is None:
                    raise IOError("Must specify boolean masking array "
                                  "when style is 'masked'.")
                if mask.shape[0] != ny or mask.shape[1] != nx:
                    if mask.shape[0] == nx and mask.shape[1] == ny:
                        mask = mask.T
                    else:
                        raise ValueError("Mask dimensions do not match "
                                         "specified grid dimensions.")
                mask = mask.flatten()
            npt = ny*nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()

        elif style == 'points':
            if xpts.size != ypts.size:
                raise ValueError("xpoints and ypoints must have "
                                 "same dimensions when treated as "
                                 "listing discrete points.")
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', "
                             "'points', or 'masked'")

        if self.coordinates_type == 'euclidean':
            xpts, ypts = _adjust_for_anisotropy(np.vstack((xpts, ypts)).T,
                                                [self.XCENTER, self.YCENTER],
                                                [self.anisotropy_scaling],
                                                [self.anisotropy_angle]).T
            xy_data = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                                      self.Y_ADJUSTED[:, np.newaxis]), axis=1)
            xy_points = np.concatenate((xpts[:, np.newaxis],
                                        ypts[:, np.newaxis]), axis=1)
        elif self.coordinates_type == 'geographic':
            # Quick version: Only difference between euclidean and spherical
            # space regarding kriging is the distance metric.
            # Since the relationship between three dimensional euclidean
            # distance and great circle distance on the sphere is monotonous,
            # use the existing (euclidean) infrastructure for nearest neighbour
            # search and distance calculation in euclidean three space and
            # convert distances to great circle distances afterwards.
            lon_d = self.X_ADJUSTED[:, np.newaxis] * np.pi / 180.0
            lat_d = self.Y_ADJUSTED[:, np.newaxis] * np.pi / 180.0
            xy_data = np.concatenate((np.cos(lon_d) * np.cos(lat_d),
                                      np.sin(lon_d) * np.cos(lat_d),
                                      np.sin(lat_d)), axis=1)
            lon_p = xpts[:, np.newaxis] * np.pi / 180.0
            lat_p = ypts[:, np.newaxis] * np.pi / 180.0
            xy_points = np.concatenate((np.cos(lon_p) * np.cos(lat_p),
                                        np.sin(lon_p) * np.cos(lat_p),
                                        np.sin(lat_p)), axis=1)

        if style != 'masked':
            mask = np.zeros(npt, dtype='bool')

        c_pars = None
        if backend == 'C':
            try:
                from .lib.cok import _c_exec_loop, _c_exec_loop_moving_window
            except ImportError:
                print('Warning: failed to load Cython extensions.\n'
                      '   See https://github.com/bsmurphy/PyKrige/issues/8 \n'
                      '   Falling back to a pure python backend...')
                backend = 'loop'
            except:
                raise RuntimeError("Unknown error in trying to "
                                   "load Cython extension.")

            c_pars = {key: getattr(self, key) for key in ['Z', 'eps', 'variogram_model_parameters', 'variogram_function']}

        if n_closest_points is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(xy_data)
            # print("OK")

            if search_radius is not None:
                if verboses:
                    print("OK")
                bd, bd_idx = tree.query(xy_points, k=tree.n,
                                        distance_upper_bound=search_radius,
                                        eps=0.0)
                # print(xy_points)
                if verboses:
                    print(type(bd), bd.shape, type(bd_idx), bd_idx.shape)
                # print([list(item) for item in mask_nd(bd, np.invert(np.isinf(bd)))])
                # print([list(item) for item in mask_nd(bd_idx, bd_idx != tree.n)])

                # bd = np.ma.masked_where(np.isinf(bd), bd)
                bd = np.array([list(item) for item in
                               mask_nd(bd, np.invert(np.isinf(bd)))])
                # print(bd)

                # Determine n_closest_points being used for kriging estimate
                # within search_radius
                nvalues = get_n_used(bd)
                if verboses:
                    print("nvalues", nvalues)

                # Determine size of largest subset
                max_size = get_largest_subset(bd, verboses=verboses)
                get_smallest_subset(bd, verboses=verboses)

                # Fill matrix with np.inf to have rectangular dimensions
                filled_bd_matrix = fill_matrix(bd, max_size)
                """ TO DO: other name for bd here since it is redefined and can cause issues if we want to but the calculation of nvalues later in the code"""
                bd = filled_bd_matrix.copy()

                # bd = np.ma.masked_where(np.isinf(bd), bd)
                # bd_idx = np.ma.masked_equal(bd_idx, tree.n)

                bd_idx = np.array([list(item) for item in
                                   mask_nd(bd_idx, bd_idx != tree.n)])

                # print(bd_idx)
                # Fill matrix with np.inf to have rectangular dimensions
                filled_bd_idx_matrix = fill_matrix(bd_idx, max_size)
                bd_idx = filled_bd_idx_matrix.copy()

                # bd_idx = np.ma.masked_equal(bd_idx, tree.n)
                if verboses:
                    print("bd_idx", bd_idx)

            else:
                # print("normal")
                bd, bd_idx = tree.query(xy_points, k=n_closest_points, eps=0.0)
            # print(bd[0], bd_idx[0])
            # print(bd, bd_idx)

            if self.coordinates_type == 'geographic':
                # Convert euclidean distances to great circle distances:
                bd = core.euclid3_to_great_circle(bd)

            if backend == 'loop':
                if verboses:
                    print("bd", bd)
                # print(type(bd), bd.shape)
                zvalues, sigmasq = \
                    self._exec_loop_moving_window(a, bd, mask, bd_idx)
            elif backend == 'C':
                zvalues, sigmasq = \
                    _c_exec_loop_moving_window(a, bd, mask.astype('int8'),
                                               bd_idx,
                                               self.X_ADJUSTED.shape[0],
                                               c_pars)
            else:
                raise ValueError('Specified backend {} for a moving window '
                                 'is not supported.'.format(backend))
        else:
            bd = cdist(xy_points,  xy_data, 'euclidean')
            if self.coordinates_type == 'geographic':
                # Convert euclidean distances to great circle distances:
                bd = core.euclid3_to_great_circle(bd)

            if backend == 'vectorized':
                zvalues, sigmasq = self._exec_vector(a, bd, mask)
            elif backend == 'loop':
                zvalues, sigmasq = self._exec_loop(a, bd, mask)
            elif backend == 'C':
                zvalues, sigmasq = _c_exec_loop(a, bd, mask.astype('int8'),
                                                self.X_ADJUSTED.shape[0],
                                                c_pars)
            else:
                raise ValueError('Specified backend {} is not supported for '
                                 '2D ordinary kriging.'.format(backend))

        if style == 'masked':
            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ['masked', 'grid']:
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq, nvalues

    def determine_mask_array(self, xpoints,
                             ypoints,
                             search_radius):
        from scipy.spatial import cKDTree

        """Determine the area of the contour plot that should be masked
        based on a threshold which resides in the minimum closest points a
        grid node should have within a certain search_radius"""

        xy_data = np.concatenate((self.X_ADJUSTED[:, np.newaxis],
                                  self.Y_ADJUSTED[:, np.newaxis]),
                                 axis=1)

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))

        grid_x, grid_y = np.meshgrid(xpts, ypts)
        xpts = grid_x.flatten()
        ypts = grid_y.flatten()

        xy_points = np.concatenate((xpts[:, np.newaxis],
                                    ypts[:, np.newaxis]), axis=1)
        print(xy_data.shape)
        print(xy_points.shape)
        tree = cKDTree(xy_data)

        bd, bd_idx = tree.query(xy_points, k=tree.n,
                                distance_upper_bound=search_radius,
                                eps=0.0)

        bd = np.array([list(item) for item in
                      mask_nd(bd, np.invert(np.isinf(bd)))])

        # Determine n_closest_points being used for kriging estimate
        # within search_radius
        nvalues = get_n_used(bd)

        print(xpoints.shape[0], ypoints.shape[0])
        nvalues = nvalues.reshape(ypoints.shape[0], xpoints.shape[0])

        return nvalues


def mask_nd(x, m):
    '''
    https://stackoverflow.com/questions/53918392/mask-2d-array-preserving-shape
    Mask a 2D array and preserve the
    dimension on the resulting array
    ----------
    x: np.array
        2D array on which to apply a mask
    m: np.array
        2D boolean mask
    Returns
    -------
    List of arrays. Each array contains the
    elements from the rows in x once masked.
    If no elements in a row are selected the
    corresponding array will be empty
    '''
    take = m.sum(axis=1)
    return np.split(x[m], np.cumsum(take)[:-1])


def get_n_used(bd):
    nvalues = [len(item) for item in bd]
    return np.array(nvalues)


def get_largest_subset(bd, verboses=True):
    # Determine size of largest subset
    max_size = 0
    for item in bd:
        if len(item) > max_size:
            max_size = len(item)
    if verboses:
        print("max_size = ", max_size)

    return max_size


def get_smallest_subset(bd, verboses=True):
    # Determine size of largest subset
    min_size = 0
    for item in bd:
        if len(item) < min_size:
            min_size = len(item)
    if verboses:
        print("min_size = ", min_size)

    return min_size


def fill_matrix(bd, max_size):
    # Fill matrix with np.inf to have rectangular dimensions
    fill_matrix = []

    for i, item in enumerate(bd):
        fill_matrix.append(item)
        if len(item) < max_size:
            times = max_size - len(item)
            fill_matrix[i].extend(np.NaN for _ in range(times))

    fill_matrix = np.array(fill_matrix)
    # print(fill_matrix.shape)

    return fill_matrix
