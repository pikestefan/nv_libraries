# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:28:45 17

@author: Lucio Stefan, lucio.stefan@gmail.com
"""

import numpy as np
from scipy import linalg as la
import scipy.constants as const
import scipy.signal as sci_sig
import scipy.optimize as sci_opt
import re
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
from math import pi, sin, cos, sqrt
import scipy.fftpack as fft
from lmfit import Model, Parameters

mu0 = const.mu_0
mu02pi = mu0 / (2 * pi)
g_fact, _, _ = const.physical_constants["electron g factor"]
g_fact = abs(g_fact)
bohr_mag, _, _ = const.physical_constants["Bohr magneton"]
gamma, _, _ = const.physical_constants["electron gyromag. ratio over 2 pi"]
gamma *= 1e6  # Is given in MHz/T, convert in Hz/T


Sz = np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
Sx = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / sqrt(2)
Sy = np.array([[0, 1, 0], [-1, 0, 1], [0, -1, 0]]) / (1j * sqrt(2))

Sx_sq = Sx.dot(Sx)
Sy_sq = Sy.dot(Sy)
Sz_sq = Sz.dot(Sz)


def polar_field(Bnorm, theta, phi):
    """
    Gives a cartesian vector using polar coordinates. Only one of the three
    inputs can be a ndarray with shape (N1,...Nm)
    Inputs:
        -Bnorm: required norm of the magnetic field
        -theta: the azimuthal angle
        -phi: the equatorial angle
    Outputs:
        -ndarray with shape (N1, ..., Nm, 3). The ndarray has the three B field
        cartesian components in the last dimension
    """

    if type(Bnorm) is np.ndarray:
        the_shape = Bnorm.shape
    elif type(theta) is np.ndarray:
        the_shape = theta.shape
    elif type(phi) is np.ndarray:
        the_shape = phi.shape
    return np.stack(
        (
            Bnorm * np.sin(theta) * np.cos(phi),
            Bnorm * np.sin(theta) * np.sin(phi),
            Bnorm * np.cos(theta),
        ),
        axis=len(the_shape),
    )


def strayB_on_NV_components(Bfields, nv_theta, nv_phi):
    """
    Projects the field on the NV axis, calculating the parallel and orthogonal components
    """
    input_shape = Bfields.shape
    if input_shape == (3,):
        Bfields = np.array([Bfields])
        bfield_num = 1
        input_shape = (1,)
    elif len(input_shape) == 2:
        bfield_num = input_shape[0]
    elif len(input_shape) > 2:
        bfield_num = 1
        for ii in range(0, len(input_shape) - 1):
            bfield_num = bfield_num * input_shape[ii]
        Bfields = np.reshape(Bfields, (bfield_num, 3))
    Nvx, Nvy, Nvz = [
        np.sin(nv_theta) * np.cos(nv_phi),
        np.sin(nv_theta) * np.sin(nv_phi),
        np.cos(nv_theta),
    ]

    Bparallel = Bfields[:, 0] * Nvx + Bfields[:, 1] * Nvy + Bfields[:, 2] * Nvz

    Bort_norm = np.sqrt(
        np.square(np.linalg.norm(Bfields, axis=1)) - np.square(Bparallel)
    )

    return Bparallel.reshape(input_shape[:-1]), Bort_norm.reshape(input_shape[:-1])


def vectorised_eigenvalue_solver(
    Bfields, nv_theta=0, nv_phi=0, Dsplit=2.87e9, Esplit=0, return_eistates=False
):
    """
    Solves a zero-field+Zeeman hamiltonian for a B field nd_array.
    Inputs:
        - Bfields: the ndarray of magnetic fields. Shape needs to be (N1,...,Nm,3).
        - nv_theta: the nv azimuthal angle
        - nv_phi: the nv equatorial angle
        - Dsplit: the zero-field splitting
        - Esplit: the strain splitting
        - return_eistates: boolean. If True, the function returns both the
                           eigenstates and the eigenvalues. If false, it only
                           return the eigenvalues
    Ouputs:
        -If return_eistates is True:
            (eigenstates, (lower energy transition freq., higher energy transition freq.) )
        -If return_eistates is False:
            (lower energy transition freq., higher energy transition freq.)
        The eigenstates is a (3,3) matrix, where the rows are the coefficients in the 0,-1,1
        basis. The rows are associated to the eigenfrequencies in ascending order.
    """

    input_shape = Bfields.shape
    if input_shape == (3,):
        Bfields = np.array([Bfields])
        bfield_num = 1
        input_shape = (1,)
    elif len(input_shape) == 2:
        bfield_num = input_shape[0]
    elif len(input_shape) > 2:
        bfield_num = 1
        for ii in range(0, len(input_shape) - 1):
            bfield_num = bfield_num * input_shape[ii]
        Bfields = np.reshape(Bfields, (bfield_num, 3))
    NV_axis = np.array(
        [
            np.sin(nv_theta) * np.cos(nv_phi),
            np.sin(nv_theta) * np.sin(nv_phi),
            np.cos(nv_theta),
        ]
    )

    Bparallel_norm = np.abs(np.dot(Bfields, NV_axis))
    diff = np.square(np.linalg.norm(Bfields, axis=1)) - np.square(Bparallel_norm)
    if np.any(diff < 0):
        negative_diffs = diff[diff < 0]
        diff[diff < 0] = np.abs(negative_diffs)
        string = "Diff values are {}, they have been made positive.".format(
            negative_diffs
        )
        warnings.warn(string, Warning)
    Bort_norm = np.sqrt(diff)
    zero_f_ham = Dsplit * Sz_sq + Esplit * (Sx_sq - Sy_sq)

    tot_hamiltonian = zero_f_ham + gamma * (
        np.einsum("i,jk->ijk", Bort_norm, Sx)
        + np.einsum("i,jk->ijk", Bparallel_norm, Sz)
    )

    if return_eistates:
        eistates, eifreqs = np.linalg.eigh(tot_hamiltonian)
        freq_0, freq_1, freq_2 = eifreqs
        eistates = eistates.transpose([0, 2, 1]).reshape(input_shape[:-1] + (3,))

        freq_minus = (eifreqs[:, 1] - eifreqs[:, 0]).reshape(input_shape[:-1])
        freq_plus = (eifreqs[:, 2] - eifreqs[:, 0]).reshape(input_shape[:-1])
        return eistates, (freq_minus, freq_plus)
    else:
        eifreqs = np.linalg.eigvalsh(tot_hamiltonian)
        freq_minus = (eifreqs[:, 1] - eifreqs[:, 0]).reshape(input_shape[:-1])
        freq_plus = (eifreqs[:, 2] - eifreqs[:, 0]).reshape(input_shape[:-1])
        return freq_minus, freq_plus


def angle_dot_prod(arr1, arr2):
    """
    Returns the angle between two vectors
    """
    return (
        np.arccos(np.dot(arr1, arr2) / (np.linalg.norm(arr1) * np.linalg.norm(arr2)))
        * 180
        / np.pi
    )


def custom_colbar(mappable=None, side="right", size="5%", pad=0.05, *args, **kwargs):
    """
    Creates a colorbar that doesn't suffer from rescaling issues.
    Mappable is the plot handle. *args and **kwargs are passed to the colorbar
    """
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size, pad)
    return fig.colorbar(mappable, cax=cax, *args, **kwargs)


def isoB_simulator(
    freq1_ndarray=None,
    freq2_ndarray=None,
    query_freqs=None,
    linewidth=(2e6, 2e6),
    contrast=(0.2, 0.2),
    plrate=1.0,
):

    if freq1_ndarray.shape == freq2_ndarray.shape:
        resonances_shape = freq1_ndarray.shape
        shape_prod = 1
        for shape_el in resonances_shape:
            shape_prod *= shape_el
    else:
        raise (Exception("Input resonances shapes must match"))
    f1 = np.reshape(freq1_ndarray, shape_prod)
    f2 = np.reshape(freq2_ndarray, shape_prod)

    def nv_counts(nu_query, nu0_1, nu0_2, delta, c, plrate):
        return plrate * (
            1
            - c[0]
            * delta[0] ** 2
            / (np.square(nu_query[:, None] - nu0_1[None, :]) + delta[0] ** 2)
            - c[1]
            * delta[1] ** 2
            / (np.square(nu_query[:, None] - nu0_2[None, :]) + delta[1] ** 2)
        )

    output_counts = nv_counts(query_freqs, f1, f2, linewidth, contrast, plrate)
    return output_counts.reshape(query_freqs.shape + resonances_shape)


def cartesian2spherical(cartesian_matrix):
    """
    Shape of the array needs to be (M,N,3)
    """
    norm = np.linalg.norm(cartesian_matrix, axis=2)
    masknorm = norm == 0

    theta = np.divide(cartesian_matrix[:, :, 2], norm)
    theta = np.arccos(theta)
    phi = np.arctan2(cartesian_matrix[:, :, 1], cartesian_matrix[:, :, 0])
    phi[phi < 0] = phi[phi < 0] + 2 * pi

    norm[masknorm] = np.nan
    phi[masknorm] = np.nan
    theta[masknorm] = np.nan

    return np.stack([norm, theta, phi], axis=2)


def nv_eigenfrequencies_analytical(
    Bnorm=0,
    Btheta=0,
    Bphi=0,
    Dsplit=2.87e9,
    Esplit=0,
    angles_are_deg=True,
    freqs_are_GHz=False,
    fields_are_mT=False,
    return_trans_freq=False,
):
    """
    Finds the analytical solution to the NV Hamiltonian secular equation, for arbitrary magnetic fields.
    
    Returns a tuple:
        if return_trans_freq is False (lower transition frequency, higher transition frequency)
        if return_trans_freq is True (lowest eigenfrequency, mid eigenfrequency, highest eigefrequency)
    """
    gamma_ratio = g_fact * bohr_mag / const.Planck

    if angles_are_deg:
        Btheta = Btheta * np.pi / 180.0
        Bphi = Bphi * np.pi / 180.0
    if freqs_are_GHz:
        if Dsplit == 2.87e9:
            Dsplit = 2.87
        gamma_ratio = gamma_ratio * 1e-9
    if fields_are_mT:
        gamma_ratio = gamma_ratio * 1e-3
    B = gamma_ratio * Bnorm

    secular_const = (
        -Dsplit * (np.square(B) + 4 * np.square(Esplit)) / 6
        + 2 * np.power(Dsplit, 3) / 27
        - np.square(B)
        * (
            Dsplit * np.cos(2 * Btheta)
            + 2 * Esplit * np.cos(2 * Bphi) * np.square(np.sin(Btheta))
        )
        / 2
    )
    secular_const = secular_const.astype(complex)

    secular_coeff = -(np.square(Dsplit) / 3 + np.square(B) + np.square(Esplit))
    secular_coeff = secular_coeff.astype(complex)

    xiconst = -(1 - np.sqrt(3) * 1j) / 2

    root = np.sqrt(np.square(secular_const) / 4 + np.power(secular_coeff, 3) / 27)

    left_block = np.power(-secular_const / 2 + root, 1 / 3)
    right_block = np.power(-secular_const / 2 - root, 1 / 3)

    eival1 = np.real(left_block + right_block)
    eival2 = np.real(xiconst * left_block + xiconst ** 2 * right_block)
    eival3 = np.real(xiconst ** 2 * left_block + xiconst ** 4 * right_block)

    if return_trans_freq:
        return eival1 - eival2, eival3 - eival2
    else:
        return eival2, eival1, eival3


def single_lorentz(f, f0, width, contrast):
    lorentz_line = contrast * (width / 2) ** 2 / ((f - f0) ** 2 + (width / 2) ** 2)
    lorentz_line = lorentz_line
    return lorentz_line


def baseline(f, intensity):
    return intensity * np.ones(f.shape)


class odmr_fitter(object):
    """
    Class to initialize an odmr fitter based on lmfit Models. If the guesses
    for the peak frequencies are not set calling set_frequency_guess before 
    calling the fit method, an  automatic dip search based on scipy.signal.find_peaks
    is started. Frequency, contrast and linewidth guesses can be provided when
    the class in initialized, or later by calling the methods set_frequency_guess, 
    set_contrast_guess and set_linewidth_guess. 
    To fit the data, call the .fit method.
    Optional kwargs are:
        - smoothed data: an array of smoothed data, to help improve peak the autofinding                 
        - gtol of scipy.optimize.curve_fit
        - max_nfev
    """

    def __init__(
        self,
        dip_number,
        freq_guess=None,
        contrast_guess=None,
        linewidth_guess=None,
        **kwargs
    ):
        self.dip_number = dip_number
        self.freq_guess = freq_guess
        self.contrast_guess = contrast_guess
        self.linewidth_guess = linewidth_guess
        self.smoothed_data = kwargs.pop("smoothed_data", None)
        self.max_nfev = kwargs.pop("max_nfev", 10000)
        self.index_parameter = kwargs.pop("index_parameter", None)
        self.bounds = kwargs.pop("bounds", None)
        self.kwargs = kwargs

        self.parameters = Parameters()

        self._model = None
        self._prefix = "l{:d}_"

        self._make_model()
        if self.freq_guess is not None:
            self.set_frequency_guess(self.freq_guess)
        if self.contrast_guess:
            self.set_contrast_guess(self.contrast_guess)
        if self.linewidth_guess:
            self.set_linewidth_guess(self.linewidth_guess)
        if self.bounds is not None:
            self.set_bounds(self.bounds)

    def _make_model(self):
        for ii in range(self.dip_number):
            prefix = self._dipprefix(ii)
            if ii == 0:
                self._model = Model(single_lorentz, prefix=prefix)
            else:
                self._model += Model(single_lorentz, prefix=prefix)
        self._model = Model(baseline) - Model(baseline, prefix="_") * self._model

        self.parameters = self._model.make_params()
        self.parameters["intensity"].min = 0
        self.parameters["_intensity"].expr = "intensity"
        for ii in range(self.dip_number):
            prefix = self._dipprefix(ii)
            self.parameters[prefix + "f0"].min = 0
            self.parameters[prefix + "f0"].value = 2.87e9
            self.parameters[prefix + "width"].min = 0
            self.parameters[prefix + "width"].value = 0.5e6
            self.parameters[prefix + "contrast"].min = 0
            self.parameters[prefix + "contrast"].max = 1
            self.parameters[prefix + "contrast"].value = 0.1

    def set_frequency_guess(self, freq_guess, dip_number=None):
        if isinstance(freq_guess, (list, np.ndarray)) and (
            len(freq_guess) != self.dip_number
        ):
            raise ValueError(
                "The length of guesses needs to be equal to the number of dips."
            )
        if dip_number is None:
            self.freq_guess = freq_guess
            for ii, guess in enumerate(self.freq_guess):
                prefix = self._dipprefix(ii)
                self.parameters[prefix + "f0"].value = guess
        else:
            self.freq_guess[dip_number] = freq_guess
            self.parameters[self._dipprefix(dip_number) + "f0"].value = freq_guess

    def set_contrast_guess(self, contrast_guess, dip_number=None):
        if isinstance(contrast_guess, (list, np.ndarray)) and (
            len(contrast_guess) != self.dip_number
        ):
            raise ValueError(
                "The length of guesses needs to be equal to the number of dips."
            )
        if dip_number is None:
            self.contrast_guess = contrast_guess
            for ii, guess in enumerate(self.contrast_guess):
                prefix = self._dipprefix(ii)
                self.parameters[prefix + "contrast"].value = guess
        else:
            self.contrast_guess[dip_number] = contrast_guess
            self.parameters[
                self._dipprefix(dip_number) + "contrast"
            ].value = contrast_guess

    def set_linewidth_guess(self, linewidth_guess, dip_number=None):
        if isinstance(linewidth_guess, (list, np.ndarray)) and (
            len(linewidth_guess) != self.dip_number
        ):
            raise ValueError(
                "The length of guesses needs to be equal to the number of dips."
            )
        if dip_number is None:
            self.linewidth_guess = linewidth_guess
            for ii, guess in enumerate(self.linewidth_guess):
                prefix = self._dipprefix(ii)
                self.parameters[prefix + "width"].value = guess
        else:
            self.linewidth_guess[dip_number] = linewidth_guess
            self.parameters[
                self._dipprefix(dip_number) + "width"
            ].value = linewidth_guess

    def set_bounds(self, bound_dictionary):
        self.bounds = bound_dictionary
        for parameter, subdict in bound_dictionary.items():
            for dip_num, bounds in subdict.items():
                par_name = self._dipprefix(dip_num) + parameter
                self.parameters[par_name].min = bounds[0]
                self.parameters[par_name].max = bounds[1]

    def _dipprefix(self, dip_index):
        return self._prefix.format(dip_index)

    def peak_autofinder(self, freqs, odmr_data):
        if self.smoothed_data is not None:
            data_4_guess = self.smoothed_data
        else:
            data_4_guess = odmr_data
        height = self.kwargs.get("height", None)
        threshold = self.kwargs.get("threshold", None)
        distance = self.kwargs.get("distance", None)
        prominence = self.kwargs.get("prominence", None)
        width = self.kwargs.get("width", None)
        wlen = self.kwargs.get("wlen", None)
        rel_height = self.kwargs.get("rel_height", 0.5)

        found_pks, _ = sci_sig.find_peaks(
            1 - data_4_guess / data_4_guess.mean(),
            height=height,
            threshold=threshold,
            distance=distance,
            prominence=prominence,
            width=width,
            wlen=wlen,
            rel_height=rel_height,
        )
        if len(found_pks) < self.dip_number:
            raise (Exception("Found dips are less than the input dip number."))
        max_amps = data_4_guess[found_pks]
        found_pks = found_pks[max_amps.argsort()]

        self.set_frequency_guess(freqs[found_pks[0 : self.dip_number]])

    def fit(self, frequency, data):
        """
        Method that fits the provided data to the current multipeak model.
        Parameters
        ----------
        frequency : np.ndarray
            The array of the frequency scan.
        data : np.ndarray
            The omdr data

        Returns
        -------
        fit_results : ModelResult
            The fit results from lmfit.fit().
        model: Model
            The multiLorentz lmfit.Model.
        """
        self.parameters["intensity"].value = data.mean()
        if self.freq_guess is None:
            self.peak_autofinder(frequency, data)
        fit_results = self._model.fit(
            data, params=self.parameters, f=frequency, max_nfev=self.max_nfev
        )
        return fit_results, self._model


def Bx_right_edge(xdata, topo, dist_nv, x_edge, Is):
    return (
        mu02pi
        * Is
        * (dist_nv + topo)
        / (np.square(xdata - x_edge) + np.square(dist_nv + topo))
    )


def Bz_edge(xdata, topo, dist_nv, x_edge, Is):
    return (
        mu02pi
        * Is
        * (xdata - x_edge)
        / (np.square(xdata - x_edge) + np.square(dist_nv + topo))
    )


def NV_on_edge(
    x_axis,
    topography_slice,
    x_edge,
    dist_nv,
    Esplit,
    NVtheta,
    NVphi,
    Bbias,
    Is,
    edge_side="right",
):
    if edge_side == "right":
        sign_coeff = 1
    else:
        sign_coeff = -1
    Bx_pr = (
        sign_coeff
        * Bx_right_edge(
            xdata=x_axis, topo=topography_slice, dist_nv=dist_nv, x_edge=x_edge, Is=Is
        )
        * np.sin(NVtheta)
        * np.cos(NVphi)
    )
    Bz_pr = Bz_edge(
        xdata=x_axis, topo=topography_slice, dist_nv=dist_nv, x_edge=x_edge, Is=Is
    ) * np.cos(NVtheta)
    projection = Bx_pr + Bz_pr

    return 2 * np.sqrt(np.square(Esplit) + np.square(gamma * (projection + Bbias)))


def mag_edge_fit(
    x_axis,
    data,
    topography_slice,
    Esplit,
    NVtheta=None,
    NVphi=None,
    Bbias=0,
    edge_side="right",
    guesses=None,
):
    """
    Function that is used to fit the Zeeman **splitting** of a scan over the
    edge of a uniformly magnetized strip. Uses NV_on_edge as fit model.
    
    Parameters
    ----------
    x_axis : np.ndarray
        The axis with the coordinates of the line scan.
    data : np.ndarray
        The Zeeman splitting.
    topography_slice : np.ndarray
        The topography of the sample.
    Esplit : float
        The NV strain splitting.
    NVtheta : float, optional
        The polar angle of the NV. If not given, it is used as a free parameter
        in the fit.
    NVphi : float, optional
        The azimuthal angle of the NV. If not given, it is used as a free parameter
        in the fit.
    Bbias : float, optional
        The Bias field along the NV axis. The default is 0. If set as None,
        it is used as a free parameter of the fit.
    edge_side : str, optional
        The edge of the strip, which is either left or right. The default is 'right'.
    guesses : dict, optional
        A dictionary containing the guesses for the parameters, which are found
        in the NV_on_edge function. The default is None.

    Returns
    -------
    fit_results : ModelResult
        The output of the fit.

    """
    independent_vars = ["x_axis", "edge_side", "topography_slice", "Esplit"]

    fit_kwargs = {}
    if NVtheta is not None:
        independent_vars.append("NVtheta")
        fit_kwargs["NVtheta"] = NVtheta
    if NVphi is not None:
        independent_vars.append("NVphi")
        fit_kwargs["NVphi"] = NVphi
    if Bbias is not None:
        independent_vars.append("Bbias")
        fit_kwargs["Bbias"] = Bbias
    model = Model(NV_on_edge, independent_vars=independent_vars)
    parameters = model.make_params()
    parameters["Is"].min = 0
    parameters["x_edge"].min = x_axis[0]
    parameters["x_edge"].max = x_axis[-1]
    parameters["dist_nv"].min = 0
    if "NVtheta" in parameters:
        parameters["NVtheta"].min = -pi
        parameters["NVtheta"].max = pi
    if "NVphi" in parameters:
        parameters["NVphi"].min = 0
        parameters["NVphi"].max = 2 * pi
    if "Bbias" in parameters:
        parameters["Bbias"].min = 0
    for parameter, guess_value in guesses.items():
        parameters[parameter].value = guess_value
    fit_results = model.fit(
        data,
        params=parameters,
        x_axis=x_axis,
        topography_slice=topography_slice,
        Esplit=Esplit,
        edge_side=edge_side,
        **fit_kwargs
    )
    return fit_results


def check_neighbour_pixel(input_matrix, index_tuple, neighbours=1, empty_value=np.nan):
    """
    Takes a (M,N) matrix and check the n nearest neighbours at the index tuple.
    Returns their average value.
    Inputs:
        - the (M,N) numpy array
        - index_tuple: index tuple
        - neighbours: how many nearest neighbours to check
        - empty value: for incomplete scans. Tells the function which value
                       corresponds to an empty pixel
    Returns:
        the average of the n nearest neighbours of index_tuple.
    """
    size_x, size_y = input_matrix.shape
    ii, jj = index_tuple

    xlowbound, xhighbound = (
        ii - neighbours if (ii - neighbours >= 0) else 0,
        ii + neighbours + 1 if (ii + neighbours + 1 < size_x) else size_x,
    )
    ylowbound, yhighbound = (
        jj - neighbours if (jj - neighbours >= 0) else 0,
        jj + neighbours + 1 if (jj + neighbours + 1 < size_y) else size_y,
    )

    sub_mat = np.array(input_matrix[xlowbound:xhighbound, ylowbound:yhighbound])
    if ~np.isnan(empty_value):
        sub_mat[sub_mat == empty_value] = np.nan
    return np.nanmean(sub_mat)


def get_pixel_neighbourhood(input_matrix, index_tuple, neighbours=1):
    """
    Takes a (M,N) matrix and check the n nearest neighbours at the index tuple.
    Returns their average value.
    Inputs:
        - the (M,N) numpy array
        - index_tuple: index tuple
        - neighbours: how many nearest neighbours to check
    Returns:
        the average of the n nearest neighbours of index_tuple.
    """
    size_x, size_y = input_matrix.shape
    ii, jj = index_tuple

    xlowbound, xhighbound = (
        ii - neighbours if (ii - neighbours >= 0) else 0,
        ii + neighbours + 1 if (ii + neighbours + 1 < size_x) else size_x,
    )
    ylowbound, yhighbound = (
        jj - neighbours if (jj - neighbours >= 0) else 0,
        jj + neighbours + 1 if (jj + neighbours + 1 < size_y) else size_y,
    )

    sub_mat = np.array(input_matrix[xlowbound:xhighbound, ylowbound:yhighbound])

    return sub_mat


def select_region(input_matrix, index_tuple, neighbours=(1, 1)):
    """
    Selects a region of a matrix around a given index tuple.
    -Inputs: 
        -input_matrix, (M,N) shaped np.array
        -index_tuple
        -neighbours: number of neighbour pixels around the central pixel
    -Returns: selected (n,n) shaped matrix
    """
    size_x, size_y = input_matrix.shape
    x_neighbours, y_neighbours = neighbours
    ii, jj = index_tuple

    xlowbound, xhighbound = (
        ii - x_neighbours if (ii - x_neighbours >= 0) else 0,
        ii + x_neighbours + 1 if (ii + x_neighbours + 1 < size_x) else size_x,
    )
    ylowbound, yhighbound = (
        jj - y_neighbours if (jj - y_neighbours >= 0) else 0,
        jj + y_neighbours + 1 if (jj + y_neighbours + 1 < size_y) else size_y,
    )

    sub_mat = np.array(input_matrix[xlowbound:xhighbound, ylowbound:yhighbound])

    return sub_mat


def angle_dependent_splitting(
    Bnorm, Btheta=0, Bphi=0, NVtheta=0, NVphi=0, Esplit=2.5e6
):

    B_nv = Bnorm * (
        np.cos(Btheta) * np.cos(NVtheta)
        + np.sin(Btheta) * np.sin(NVtheta) * np.cos(Bphi - NVphi)
    )

    return 2 * np.sqrt(np.square((gamma * B_nv)) + np.square(Esplit))


def fit_nv_angle(
    phi_sweep_and_split=None,
    theta_sweep_and_split=None,
    Bnorm=None,
    fixed_theta=0,
    fixed_phi=0,
    Esplit=0,
    x0=[0, 0],
    *args,
    **kwargs
):
    """
    Fit the NV azimuthal and polar angles. Does a combined fit of an
    azimuthal angle sweep and an polar angle sweep. Sorry for the terrible list names!
    Inputs:
        -phi_sweep_and_split: the ESR ***splitting***, fixed theta and sweeping phi
        -theta_sweep_and_split: the ESR ***splitting***, fixed phi and sweep theta
        - Bnorm: the magnetic field norm. Default is None. If None, it will be
                 used as a free parameter in the fit
        -fixed_theta: the fixed polar angle when you sweep the equatorial angle
        -fixed_phi  the fixed azimuthal angle when you sweep the polar angle
        -Esplit: the strain splitting (ideally coming from the fit of a zero-field ESR!)
        -x0: the initial guess for minimisation. Shape needs to be (3,) when Bnorm is None, (2,)
            otherwise
        *args, **kwargs are passed to leastsq (from scipy.optimize)
    """
    phi_rows, _ = phi_sweep_and_split.shape
    theta_rows, _ = theta_sweep_and_split.shape
    full_data = np.vstack((phi_sweep_and_split, theta_sweep_and_split))

    if Bnorm is None:

        def error_func(params, xdata, ydata):
            func = lambda pars, x: np.hstack(
                (
                    angle_dependent_splitting(
                        pars[2],
                        fixed_theta,
                        x[0:phi_rows],
                        NVtheta=pars[0],
                        NVphi=pars[1],
                        Esplit=Esplit,
                    ),
                    angle_dependent_splitting(
                        pars[2],
                        x[phi_rows:],
                        fixed_phi,
                        NVtheta=pars[0],
                        NVphi=pars[1],
                        Esplit=Esplit,
                    ),
                )
            )
            return func(params, xdata) - ydata

    else:

        def error_func(params, xdata, ydata):
            func = lambda pars, x: np.hstack(
                (
                    angle_dependent_splitting(
                        Bnorm,
                        fixed_theta,
                        x[0:phi_rows],
                        NVtheta=pars[0],
                        NVphi=pars[1],
                        Esplit=Esplit,
                    ),
                    angle_dependent_splitting(
                        Bnorm,
                        x[phi_rows:],
                        fixed_phi,
                        NVtheta=pars[0],
                        NVphi=pars[1],
                        Esplit=Esplit,
                    ),
                )
            )
            return func(params, xdata) - ydata

    fit_pars, _ = sci_opt.leastsq(
        error_func, x0=x0, args=(full_data[:, 0], full_data[:, 1]), *args, **kwargs
    )

    fitted_data = (
        error_func(fit_pars, full_data[:, 0], full_data[:, 1]) + full_data[:, 1]
    )
    return fit_pars, fitted_data[0:phi_rows], fitted_data[phi_rows:]


def mag_components_from_resonances(
    nu_minus_mat=None, nu_plus_mat=None, Dsplit=2.87e9, Esplit=0
):
    """
    Calculates the parallel and orthogonal components on the NV axis, given the
    two NV resonances. All parameters are in SI units (i.e. Hz and T).
    Inputs:
        -nu_minus_mat: First resonance ndarray (doesn't matter if it is the low frequency one)
        -nu_plus_mat: Second resonance ndarray (doesn't matter if it is the high frequency one)
        -Dsplit: The zero-field splitting (Default is 2.87e9 Hz)
        -Esplit: Strain splitting (Default is 0Hz)
    Returns:
        - (Bparallel, Borthogonal) tuple. Units are Tesla.
    """

    # Convert everything in GHz to reduce numerical errors
    nu_minus_mat = nu_minus_mat / 1e9
    nu_plus_mat = nu_plus_mat / 1e9
    Dsplit = Dsplit / 1e9
    Esplit = Esplit / 1e9
    gamma = 28.02495164  # GHz/T

    nu_mp_sum = np.square(nu_minus_mat) + np.square(nu_plus_mat)
    nu_mp_prod = nu_minus_mat * nu_plus_mat

    # The solution of the cubic equation, as presented in Balasubramanian paper
    Bnorm_sq = ((nu_mp_sum - nu_mp_prod - Dsplit ** 2) - Esplit ** 2) / 3
    Delta = (
        7 * Dsplit ** 3
        + 2
        * (nu_minus_mat + nu_plus_mat)
        * (2 * (nu_mp_sum) - 5 * nu_mp_prod - 9 * Esplit ** 2)
        - 3 * Dsplit * (nu_mp_sum - nu_mp_prod + 9 * Esplit ** 2)
    ) / 27
    # The above is almost the Delta, but without the 1/B^2 coefficient

    # cos(x) = sqrt( (1+cos(2x))/2 ), sin(x) = sqrt( (1-cos(2x))/2 )
    # Bpar = np.sqrt( (Dsplit * Bnorm_sq + Delta) / (2 * Dsplit) ) / gamma
    # Bort = np.sqrt( (Dsplit * Bnorm_sq - Delta) / (2 * Dsplit) ) / gamma
    Bpar = (Dsplit * Bnorm_sq + Delta) / (2 * Dsplit) / gamma
    Bort = (Dsplit * Bnorm_sq - Delta) / (2 * Dsplit) / gamma
    return Bpar, Bort


def get_matrix_neighbourhoods(matrix, neighbourhood=1, **kwargs):
    """
    Returns a 3D matrix containing the neighbourhood of each matrix element
    
    Parameters
    ----------
    matrix : (M,N) 2D array
        The 2D matrix.
    neighbourhood : int, optional
        The neighbourhood of the pixel
        
    Optional Parameters
    -------------------
    **kwargs are passed to numpy.pad function

    Returns
    -------
    arr_view:  numpy.ndarray
        The matrix containing the neighbourhoods, has (M * N, neighbourhood, neighbourhood)
        shape
    
    """

    mode = kwargs.pop("mode", "constant")
    constant_values = kwargs.pop("constant_values", 0)

    padded = np.pad(
        matrix, neighbourhood, mode=mode, constant_values=constant_values, **kwargs
    )

    sub_shape = (neighbourhood * 2 + 1,) * 2
    view_shape = tuple(np.subtract(padded.shape, sub_shape) + 1) + sub_shape
    arr_view = np.lib.stride_tricks.as_strided(padded, view_shape, padded.strides * 2)

    arr_view = arr_view.reshape((-1,) + sub_shape)

    return arr_view


def Bfield_fromBnv(Bnv_matrix=None, nv_theta=54.7 * np.pi / 180, nv_phi=0, delta=1e-20):
    """
    Reconstructs the three components of the magnetic field assuming a planar
    distribution of magnetisation, starting from the magnetic field measured
    along the NV axis.
    
    Parameters
    ----------
    Bnv_matrix : (M,N) 2D array
        The 2D matrix containing the magnetic field along the NV axis.
    nv_theta : float, optional
        The NV azimuthal angle (in radiants). The default is 54.7*np.pi /180.
    nv_phi : float, optional
        The NV equtorial angle (in radiants). The default is 0.
    delta : float, optional
        Small number used to replace the k-vector norms when they are zero.
        The default is 1e-20.

    Returns
    -------
    Br_reconstructed : (M,N,3) numpy.ndarray
        The matrix containing the reconstructed magnetic field components,
        ordered as x,y,z along the last axis.

    """

    nx, ny, nz = sin(nv_theta) * cos(nv_phi), sin(nv_theta) * sin(nv_phi), cos(nv_theta)

    # Generate the k-vectors from the Bnv shape (units of k don't matter)
    kx = fft.fftfreq(Bnv_matrix.shape[1], 1)
    ky = fft.fftfreq(Bnv_matrix.shape[0], 1)
    """
    fftfreq returns 1/wavelength, not a momentum, but we don't care about
    coefficients, since the equations contain only ratios k(x,y)/norm(K).
    The following line is thus disabled
    
    kx, ky = kx*2*np.pi, ky*2*np.pi
    """
    Kx, Ky = np.meshgrid(kx, ky)

    normK = np.linalg.norm(np.stack((Kx, Ky), axis=2), axis=2)
    normK[normK == 0] = delta  # To avoid division by zero

    Kx_normalised = Kx / normK
    Ky_normalised = Ky / normK

    Bnv_kspace = fft.fft2(Bnv_matrix)

    # Assume the curl(B) = 0 in the region where the NV has measured the field.
    # Then the stray fields are related by the following relations, in the k-space
    # lying in the (x,y) plane:
    Bz_kspace = Bnv_kspace / (-1j * nx * Kx_normalised - 1j * ny * Ky_normalised + nz)
    By_kspace = -1j * Ky_normalised * Bz_kspace
    Bx_kspace = -1j * Kx_normalised * Bz_kspace

    B_kspace_stack = np.stack((Bx_kspace, By_kspace, Bz_kspace), axis=2)
    Br_reconstructed = fft.ifft2(B_kspace_stack, axes=(0, 1)).real
    return Br_reconstructed


def Bfield_fromM_fourier(
    mag_3dmatrix, x_array, y_array, z=0, thickness=1e-9, delta=1e-100
):
    """
    Parameters
    ----------
    mag_3dmatrix : np.ndarray
        Matrix with shape (M, N, 3). The last three dimensions contain the three
        components of the magnetisation.
    x_array : np.array
        1d array with length N. The x axis of the 2d image.
    y_array : np.array
        1d array with length M. The y axis of the 2d image.
    z : scalar or np.array, optional
        The distance from the surface at which the field is probed. It can be 
        a scalar or a list of values. Default is 0.
    thickness: float, optional
        The thickness of the magnetic layer. Default is 1e-9.
    delta : float, optional
        The small number used to avoid division by zero. The default is 1e-100.

    Returns
    -------
    bfield : np.ndarray
        The matrix with the stray fields components. If z is an array, it has
        shape (len(z), M, N, 3),  (M, N, 3) otherwise.
    """

    mag_fourier = fft.fft2(mag_3dmatrix, axes=(0, 1))

    if isinstance(z, list):
        z = np.array(z)
    elif isinstance(z, (int, float, complex)):
        z = np.array([z])
    kx = 2 * np.pi * fft.fftfreq(len(x_array), d=abs(x_array[1] - x_array[0]))
    ky = 2 * np.pi * fft.fftfreq(len(y_array), d=abs(y_array[1] - y_array[0]))

    Kx, Ky = np.meshgrid(kx, ky)

    K = np.sqrt(np.square(Kx) + np.square(Ky))
    K[K == 0] = delta

    dip_tensor = np.zeros((len(z),) + Kx.shape + (3, 3), dtype=complex)

    dip_tensor[..., 0, 0] = -np.square(Kx / K)
    dip_tensor[..., 0, 1] = -Kx * Ky / np.square(K)
    dip_tensor[..., 0, 2] = -1j * Kx / K
    dip_tensor[..., 1, 0] = dip_tensor[..., 0, 1]
    dip_tensor[..., 1, 1] = -np.square(Ky / K)
    dip_tensor[..., 1, 2] = -1j * Ky / K
    dip_tensor[..., 2, 0] = dip_tensor[..., 0, 2]
    dip_tensor[..., 2, 1] = dip_tensor[..., 1, 2]
    dip_tensor[..., 2, 2] = 1

    # Add dimensions for broadcasting
    K = K[..., None, None]  #
    z = z[:, None, None, None, None]
    dip_tensor *= (mu0 / 2) * (np.exp(-K * z) - np.exp(-K * (z + thickness)))

    bfield_kspace = np.einsum("lijmk,ijk->lijm", dip_tensor, mag_fourier)

    bfield = fft.ifft2(bfield_kspace, axes=(1, 2)).real

    if len(z) == 1:
        bfield = bfield[0]
    return bfield


def Bfield_fromM_realspace(
    Mx=None,
    My=None,
    Mz=None,
    X_array=None,
    Y_array=None,
    mat_thick=10,
    stray_height=100,
):
    # input lengths are in nm

    # Point spread function
    def alpha_xy(x=0, y=0, d=stray_height * 10 ** 9, t=mat_thick * 10 ** 9):
        term1 = 1 / ((d ** 2 + x ** 2 + y ** 2) ** 0.5)
        term2 = 1 / (((d + t) ** 2 + x ** 2 + y ** 2) ** 0.5)
        return (1 / (2 * np.pi)) * (term1 - term2)

    def alpha_z(x=0, y=0, d=stray_height * 10 ** 9, t=mat_thick * 10 ** 9):
        term1 = d + t + ((d + t) ** 2 + x ** 2 + y ** 2) ** 0.5
        term2 = d + (d ** 2 + x ** 2 + y ** 2) ** 0.5
        return (1 / (2 * np.pi)) * (term1 / term2)

    deltaX = np.mean(np.diff(X_array)) * 10 ** 9
    deltaY = np.mean(np.diff(Y_array)) * 10 ** 9

    # x y matrix for kernel (needs to be from -2L to 2L, L:length of scan)
    rx = np.linspace(
        -2 * (X_array.size - 1) * deltaX,
        2 * (X_array.size - 1) * deltaX,
        2 * (X_array.size - 1) + 1,
    )
    ry = np.linspace(
        -2 * (Y_array.size - 1) * deltaY,
        2 * (Y_array.size - 1) * deltaY,
        2 * (Y_array.size - 1) + 1,
    )
    rx_mat, ry_mat = np.meshgrid(rx, ry)
    # kernel aka PSF
    kerXY = alpha_xy(rx_mat, ry_mat)
    kerZ = alpha_z(rx_mat, ry_mat)

    dxMx = np.gradient(Mx, deltaX, axis=1)
    dxdxMx = np.gradient(dxMx, deltaX, axis=1)
    dydxMx = np.gradient(dxMx, deltaY, axis=0)

    dyMy = np.gradient(My, deltaY, axis=0)
    dydyMy = np.gradient(dyMy, deltaY, axis=0)
    dydxMy = np.gradient(dyMy, deltaX, axis=1)

    dxMz = np.gradient(Mz, deltaX, axis=1)
    dyMz = np.gradient(Mz, deltaY, axis=0)
    ddMz = np.gradient(dxMz, deltaX, axis=1) + np.gradient(dyMz, deltaY, axis=0)
    print(type(kerZ))
    print(np.shape(kerZ))

    Bx = (
        sci_sig.convolve2d(kerZ, dxdxMx)
        - sci_sig.convolve2d(kerZ, dydxMy)
        + sci_sig.convolve2d(kerXY, dxMz)
    )
    By = (
        sci_sig.convolve2d(kerZ, dydxMx)
        - sci_sig.convolve2d(kerZ, dydyMy)
        + sci_sig.convolve2d(kerXY, dyMz)
    )
    Bz = (
        sci_sig.convolve2d(kerXY, dxMx)
        + sci_sig.convolve2d(kerXY, dyMy)
        + sci_sig.convolve2d(kerXY, ddMz)
    )
    B = np.array([Bx, By, Bz])
    return B


# def multi_lorentz( f, params, dip_number = 2):
#     np_params = np.array(params)
#     f0 = np_params[0:dip_number]
#     widths = np_params[dip_number: 2 * dip_number]
#     amps = np_params[2 * dip_number: 3 * dip_number]
#     intensity = np_params[-1]

#     lorentz_line =  ( amps[:, None] *
#                      np.square(widths[:,None]) /
#                      ( np.square(f - f0[:,None]) + np.square(widths[:,None]) ) )
#     lorentz_line = intensity *(1 - np.sum(lorentz_line, axis = 0) )
#     return  lorentz_line

# def odmr_fit( freqs, odmr_data, dip_number = 2, freq_guess = None, amp_guess = None,
#              linewid_guess = None, bounds = None, **kwargs ):
#     """
#     Fits an arbitrary number of odmr dips. If freq_guess is not specified it will do an automatic dip search based on
#     scipy.signal.find_peaks.
#     Order of the data is:
#         [freq_1,...,freq_N, width_1,...,width_N ,amp_1,...,amp_N, max_intensity]
#     Optional kwargs are:
#         - smoothed data: an array of smoothed data, to help improve peak finding
#         - index_parameter: value which is printed in the error message when the fit fails
#                             (useful when the fit is in a loop)
#         - all the keyword arguments of scipy.signal.find_peaks
#         - gtol of scipy.optimize.curve_fit
#         - max_nfev of gtol of scipy.optimize.curve_fit

#     Returns:
#         - optimal parameters
#         - covariance matrix
#         - an array of the fitted data
#         - fail fit flag
#         - the guessed frequencies


#     If the fitting fails, it returns a NaN
#     """
#     smoothed_data = kwargs.get( 'smoothed_data', None )
#     maxfev =  kwargs.get( 'maxfev', 200 )
#     gtol = kwargs.get( 'gtol', 1e-8 )
#     index_parameter = kwargs.get( 'index_parameter', None )
#     show_guess = kwargs.get( 'show_guess', False)

#     fail_fit_flag = 0
#     if smoothed_data is not None:
#         data_4_guess = smoothed_data
#     else:
#         data_4_guess = odmr_data

#     if freq_guess is None:
#         height = kwargs.get( 'height', None )
#         threshold = kwargs.get( 'threshold', None )
#         distance = kwargs.get( 'distance', None )
#         prominence = kwargs.get( 'prominence', None )
#         width = kwargs.get( 'width', None )
#         wlen = kwargs.get( 'wlen', None )
#         rel_height = kwargs.get( 'rel_height', 0.5 )

#         found_pks, _ = sci_sig.find_peaks( 1 - data_4_guess / data_4_guess.mean(),
#                                           height = height, threshold = threshold, distance = distance, prominence = prominence,
#                                           width = width, wlen = wlen, rel_height = rel_height)
#         if len( found_pks ) < dip_number:
#             print(found_pks)
#             raise( Exception( "Found dips are less than the input dip number." ) )
#         max_amps = data_4_guess[found_pks]
#         #sorted_amps = max_amps.sort()
#         #print(max_amps.sort())
#         found_pks = found_pks[max_amps.argsort()]
#         freq_guess = freqs[ found_pks[0:dip_number] ]

#     if amp_guess is None:
#         amp_guess = 0.1 * np.ones( (dip_number,)  )
#     if linewid_guess is None:
#         linewid_guess = 1e6 * np.ones( (dip_number,)  )

#     fit_func = lambda x, *pars: multi_lorentz(x, pars, dip_number = dip_number)

#     init_guesses = np.concatenate( (freq_guess, linewid_guess, amp_guess, [odmr_data.mean()]) )

#     if bounds is None:
#         bounds = ( 0, np.inf * np.ones( (len(init_guesses, )) ) )
#     try:
#         opt_pars, cov_mat = sci_opt.curve_fit( fit_func, freqs, odmr_data, p0 = init_guesses,
#                                                bounds = bounds, maxfev = maxfev,
#                                                gtol = gtol)
#         fitted_data = multi_lorentz( freqs, opt_pars, dip_number = dip_number )
#     except:
#         opt_pars = np.repeat(np.nan, len(init_guesses))
#         cov_mat = np.full((len(init_guesses), len(init_guesses)), np.nan)
#         fitted_data = np.repeat(np.nan, len(freqs))
#         fail_fit_flag = 1
#         print("Failed: fit did not converge. Position is: {}".format(index_parameter) )

#     return opt_pars, cov_mat, fit_func, fail_fit_flag, freq_guess

# def _edge_field(x_axis = None, topography = None, x_edge = 0, dist_nv = 1, Is = 1):
#     """
#     Stray field of a strip with PMA.

#     Parameters
#     ----------
#     x_axis: np.array
#         The array containing the x coordinates
#     topography: np.array, optional
#         The array containing the topography of the strip. Default is an array
#         of zeros. It must have the same shape of x_axis.
#     x_edge: float
#         The position of the edge. Default is 0.
#     dist_nv: float
#         The distance between sample surface and probing height. If topography
#         is non-zero, dist_nv is a constant added to the topography. Default is 0.
#     Is: float
#         The saturation magnetisation of the strip

#     Returns
#     -------
#     Bfield_array: np.ndarray
#         2D array, with shape (len(x_array), 2). The first column contains the
#         x-component of the stray field, the second the z-component
#     """
#     mu02pi = const.mu_0 / (np.pi * 2)

#     bx = -mu02pi * Is * (dist_nv + topography) / ( np.square(x_axis - x_edge) + np.square(dist_nv + topography) )

#     bz = mu02pi * Is * (x_axis - x_edge) / ( np.square(x_axis - x_edge) + np.square(dist_nv + topography) )

#     return np.vstack( (bx, bz) ).T


# def fit_esr_edge_signal( x_axis = None, topography_slice = None, ESR_slice = None,
#                          Dsplit = None, Esplit = None, NVtheta = None, NVphi = None, Bbias = 0,
#                          *args, **kwargs ):
#     """
#     Fits the ESR ***splitting*** to an edge with  PMA.
#     Fit inital guesses must be ordered as:
#         -[nv2sample distance, position of edge (of the magnetic signal), surface saturation magnetisation]
#     To fit a left-side edge or a right-side edge, just change the sat. magnetisation sign guess!
#     Inputs:
#         -x_axis: the x coordinates
#         -topography_slice: the topography data
#         -ESR_slice: the ESR data
#         -Dsplit: the zero field splitting
#         -Esplit: the strain splitting
#         -NVtheta: the NV azimuthal angle
#         -NVphi: the NV equatorial angle
#         -Bbias: fixed parameter (i.e. not fed into the fit), to specify an
#                 on-axis bias field
#         *args and **kwargs are passed to curve_fit (from scipy.optimize)
#     Outputs:
#         tuple: (fit output parameters, covariance matrix, array containing the fitting line)
#         the output parameters are ordered as:
#             [nv2sample distance, position of edge (of the magnetic signal), surface saturation magnetisation]
#     """

#     mu02pi = const.mu_0 / (np.pi * 2)

#     def Bx_left_edge(xdata, topo, dist_nv, x_edge, Is):
#         return -mu02pi * Is * (dist_nv + topo) / ( np.square(xdata - x_edge) + np.square(dist_nv + topo) )
#     def Bz_left_edge(xdata, topo, dist_nv, x_edge, Is):
#         return mu02pi * Is * (xdata - x_edge) / ( np.square(xdata - x_edge) + np.square(dist_nv + topo) )

#     def edge_signal( x_axis, dist_nv, x_edge, Is ):
#         Bx_pr = ( Bx_left_edge(xdata  = x_axis, topo = topography_slice, dist_nv = dist_nv, x_edge = x_edge, Is = Is)  *
#                  np.sin(NVtheta) * np.cos( NVphi ) )
#         Bz_pr = ( Bz_left_edge(xdata  = x_axis, topo = topography_slice, dist_nv = dist_nv, x_edge = x_edge, Is = Is) *
#                  np.cos(NVtheta) )
#         projection = Bx_pr + Bz_pr
#         return 2 * np.sqrt( np.square(Esplit) +  np.square(gamma * (projection + Bbias )) )

#     opt_pars, cov_mat = sci_opt.curve_fit(edge_signal, x_axis, ESR_slice, *args, **kwargs)
#     fitted_data = edge_signal( x_axis, opt_pars[0], opt_pars[1], opt_pars[2] )
#     return opt_pars, cov_mat, fitted_data

# def fit_esr_strip_signal( x_axis = None, topography_slice = None, ESR_slice = None,
#                          Dsplit = None, Esplit = None, NVtheta = None, NVphi = None, Bbias = 0,
#                          *args, **kwargs ):
#     """
#     Fits the ESR ***splitting*** to a strip with  PMA.
#     Fit inital guesses must be ordered as:
#         -[nv2sample distance, position of left edge (of the magnetic signal),
#         position of right edge (of the magnetic signal), surface saturation magnetisation]
#     Inputs:
#         -x_axis: the x coordinates
#         -topography_slice: the topography data
#         -ESR_slice: the ESR data
#         -Dsplit: the zero field splitting
#         -Esplit: the strain splitting
#         -NVtheta: the NV azimuthal angle
#         -NVphi: the NV equatorial angle
#         -Bbias: fixed parameter (i.e. not fed into the fit), to specify an
#                 on-axis bias field
#         *args and **kwargs are passed to curve_fit (from scipy.optimize)
#     Outputs:
#         tuple: (fit output parameters, covariance matrix, array containing the fitting line)
#         the output parameters are ordered as:
#             [nv2sample distance, position of edge (of the magnetic signal),
#             position of right edge (of the magnetic signal), surface saturation magnetisation]
#     """

#     mu02pi = const.mu_0 / (np.pi * 2)

#     def Bx_edge(xdata, topo, dist_nv, x_edge, Is):
#         return -mu02pi * Is * (dist_nv + topo) / ( np.square(xdata - x_edge) + np.square(dist_nv + topo) )
#     def Bz_edge(xdata, topo, dist_nv, x_edge, Is):
#         return mu02pi * Is * (xdata - x_edge) / ( np.square(xdata - x_edge) + np.square(dist_nv + topo) )

#     def Bx_strip(xdata, topo, dist_nv, x_edge_l, x_edge_r, Is):
#         return Bx_edge(xdata, topo, dist_nv, x_edge_l, Is) - Bx_edge(xdata, topo, dist_nv, x_edge_r, Is)
#     def Bz_strip(xdata, topo, dist_nv, x_edge_l, x_edge_r, Is):
#         return Bz_edge(xdata, topo, dist_nv, x_edge_l, Is) - Bz_edge(xdata, topo, dist_nv, x_edge_r, Is)


#     def strip_signal( x_axis, dist_nv, x_edge_l, x_edge_r, Is ):
#         Bx_pr = ( Bx_strip(xdata  = x_axis, topo = topography_slice, dist_nv = dist_nv,
#                            x_edge_l = x_edge_l, x_edge_r = x_edge_r, Is = Is)  *
#                  np.sin(NVtheta) * np.cos( NVphi ) )
#         Bz_pr = ( Bz_strip(xdata  = x_axis, topo = topography_slice, dist_nv = dist_nv,
#                            x_edge_l = x_edge_l, x_edge_r = x_edge_r, Is = Is) *
#                  np.cos(NVtheta) )
#         projection = Bx_pr + Bz_pr
#         return 2 * np.sqrt( np.square(Esplit) +  np.square(gamma * (projection + Bbias )) )

#     opt_pars, cov_mat = sci_opt.curve_fit(strip_signal, x_axis, ESR_slice, *args, **kwargs)
#     fitted_data = strip_signal( x_axis, opt_pars[0], opt_pars[1], opt_pars[2], opt_pars[3] )
#     return opt_pars, cov_mat, fitted_data

if __name__ == "__main__":
    # Code testing section!
    A = np.reshape(np.arange(0, 25), (5, 5))
    B = get_matrix_neighbourhoods(A, 2, constant_values=-1)
