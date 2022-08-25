#!/usr/bin/env python
# coding: utf-8

"""
Cross Wavelet Analysis Functions
| Option | Description |
| ------ | ----------- |
| title           | wavelets.py |
| authors         | Caitriona Douglas, Guillaume Dumas |
| date            | 2022-08-25 |
"""
# In[1]:
import numpy as np
import mne
import matplotlib.pyplot as plt
import sys
if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")
from typing import Union
import math
import matplotlib


def COIError():
    """
    Called by the xwt, phase, and wtc plotting functions to display an Error
    message if all data falls within COI.
    Indicates that time span of data is too short for analysis.

    Arguments:
        None

    Returns:
        str: COI ERROR message
    """
    return 'ERROR: INVALID WT. All results are within COI.'


def xwt(sig1: mne.Epochs, sig2: mne.Epochs, sfreq: Union[int, float],
        freqs: Union[int, np.ndarray]) -> np.ndarray:
    """
    Perfroms a cross wavelet transform on two signals.

    Arguments:

        sig1 : mne.Epochs
            Signal (eg. EEG data) of first participant.

        sig2 : mne.Epochs
            Signal (eg. EEG data) of second participant.

        sfreq: int | float
            Sampling frequency of the data in Hz.

        freqs: int | float
            Range of frequencies of interest in Hz.

    Note:
        This function relies on MNE's mne.time_frequency.morlet
        and mne.time_frequency.tfr.cwt functions.

    Returns:
        cross_sigs, wtc:
       -cross_sigs: the crosswavelet transform results
       -wtc: wavelet transform coherence calculated according to
        Maraun & Kurths (2004)
    """

    # Set the mother wavelet
    Ws = mne.time_frequency.tfr.morlet(sfreq, freqs, n_cycles=6.0, sigma=None,
                                       zero_mean=True)

    # Set parameters for the output
    n_freqs = len(freqs)
    n_epochs, n_chans, n_samples = sig1.get_data().shape
    cross_sigs = np.zeros(
        (n_chans, n_epochs, n_freqs, n_samples),
        dtype=complex) * np.nan

    # perform a continuous wavelet transform on all epochs of each signal
    for ind, ch_label in enumerate(sig1.ch_names):

        # Check the channels are the same between participants
        assert sig2.ch_names[ind] == ch_label

        # Extract the channel's data for both participants and apply cwt
        cur_sig1 = np.squeeze(sig1.get_data(mne.pick_channels(sig1.ch_names,
                                                              [ch_label])))
        out1 = mne.time_frequency.tfr.cwt(cur_sig1, Ws, use_fft=True,
                                          mode='same', decim=1)

        cur_sig2 = np.squeeze(sig2.get_data(mne.pick_channels(sig2.ch_names,
                                                              [ch_label])))
        out2 = mne.time_frequency.tfr.cwt(cur_sig2, Ws, use_fft=True,
                                          mode='same', decim=1)

        # Perfrom the cross wavelet transform
        tfr_cwt1 = out1.mean(0)
        tfr_cwt2 = out2.mean(0)
        wps1 = tfr_cwt1 * tfr_cwt1.conj()
        wps2 = tfr_cwt2 * tfr_cwt2.conj()
        cross_sigs = (out1 * out2.conj()).mean(0)
        coh = (cross_sigs) / (np.sqrt(wps1*wps2))
        abs_coh = np.abs(coh)
        wtc = (abs_coh - np.min(abs_coh)) / (np.max(abs_coh) - np.min(abs_coh))
        return (cross_sigs, wtc)


def plot_xwt(analysis: str, xwt_result: np.ndarray, sfreq: Union[int, float],
             freqs: Union[int, float, np.ndarray],
             time: int, figsize: tuple, tmin: int,
             x_units: Union[int, float]):
    """
    Plots the results of the Cross wavelet analysis.

    Arguments:
        xwt_result: np.ndarray
            Results of the crosswavelet transform (xwt).

        freqs: int | float | np.ndarray
            Frequency range of interest in Hz.

        time: int
            Time of sample duration in seconds.

        figsize: tuple
            Figure size (default is (30, 8)).

        xmin: int
            Minimum x-value (default is 0).

        x_units: int | float
            distance between xticks on x-axis (time) (default is 100)

    Note:
        This function is not meant to be called indepedently,
        but is meant to be called when using plot_xwt_crosspower
        or plot_xwt_phase_angle.

    Returns:
    Figure of xwt results.
    """

    dt = 1/sfreq
    xmax = time/dt
    xmin = tmin * dt
    tick_units = xmax/x_units
    unit_conv = time/x_units
    xticks = np.arange(xmin, xmax, tick_units)
    x_labels = np.arange(tmin, time, unit_conv)
    xmark = []
    for xlabel in x_labels:
        if xlabel != '':
            mark = str(round(xlabel, 3))
            xmark.append(mark)
        else:
            xmark = xlabel

    data = xwt_result

    coi = []
    for f in freqs:
        dt_f = 1/f
        f_coi_init = math.sqrt(2*2*dt_f)
        f_coi = f_coi_init/dt
        coi.append(f_coi)

    coi_check = []
    for item in coi:
        if item >= (time/dt):
            coi_check.append(False)
        else:
            coi_check.append(True)

    if False in coi_check:
        print(COIError())
    else:
        print('Time window appropriate for wavelet transform')

    coi_index = np.arange(0, len(freqs))

    rev_coi = []
    for f in freqs:
        dt_f = 1/f
        f_coi = math.sqrt(2*2*dt_f)
        sub_max = (time-f_coi)/dt
        rev_coi.append(sub_max)

    fig = plt.figure(figsize=figsize)
    plt.subplot(122)

    if analysis == 'phase':
        my_cm = matplotlib.cm.get_cmap('hsv')
        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='nearest')

    elif analysis == 'power':
        my_cm = matplotlib.cm.get_cmap('viridis')
        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='lanczos')

    elif analysis == 'wct':
        my_cm = matplotlib.cm.get_cmap('plasma')
        plt.imshow(data, aspect='auto', cmap=my_cm, interpolation='lanczos')

    else:
        ValueError('Metric type not supported')

    plt.gca().invert_yaxis()
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (s)')
    ylabels = np.linspace((freqs[0]), (freqs[-1]), len(freqs[0:]))
    ymark = []
    for ylabel in ylabels:
        if ylabel != '':
            mark = str(round(ylabel, 3))
            ymark.append(mark)
        else:
            ymark = ylabel

    plt.gca().set_yticks(ticks=np.arange(0, len(ylabels)), labels=ymark)
    plt.gca().set_xticks(ticks=xticks, labels=xmark)
    plt.xlim(tmin, xmax)
    plt.ylim(0, int(len(ylabels[0:-1])))
    plt.plot(coi, coi_index, 'w')
    plt.plot(rev_coi, coi_index, 'w')

    plt.fill_between(coi, coi_index, hatch='X', fc='w', alpha=0.5)
    plt.fill_between(rev_coi, coi_index, hatch='X', fc='w', alpha=0.5)

    plt.axvspan(xmin, min(coi), hatch='X', fc='w', alpha=0.5)
    plt.axvspan(xmax, max(rev_coi), hatch='X', fc='w', alpha=0.5)

    return fig


def plot_xwt_crosspower(sig1: mne.Epochs, sig2: mne.Epochs,
                        sfreq: Union[int, float],
                        freqs: Union[int, float, np.ndarray],
                        time: int, figsize: tuple = (30, 8),
                        tmin: int = 0, x_units: Union[int, float] = 100):
    """
    Plots cross wavelet crosspower results.

    Arugments:
        sig1: mne.Epochs
            Neural data signal of participant 1.

        sig2: mne.Epochs
            Neural data signal of participant 2.

        sfreq: int | float
            Sampling frequency of the data in Hz.

        freqs: int | float | np.ndarray
            Frequency range of interest in Hz.

        time: int
            Time of sample duration in seconds.

        figsize: tuple
            Figure size (default is (30, 8)).

        xmin: int
            Minimum x-value (default is 0).

        x_units: int | float
            Distance between xticks on x-axis (default is 100).

    Returns:
        Figure of xwt crosspower results
    """

    result, _ = xwt(sig1, sig2, sfreq, freqs)
    data = np.abs((result[:, :]))
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    analysis = 'power'
    fig = plot_xwt(analysis, normed_data, sfreq, freqs, time, figsize, tmin,
                   x_units)
    plt.title('Cross Wavelet Transform (Crosspower)')
    plt.colorbar(label='crosspower')

    return fig


def plot_xwt_phase_angle(sig1: mne.Epochs, sig2: mne.Epochs,
                         sfreq: Union[int, float],
                         freqs: Union[int, float, np.ndarray], time: int,
                         figsize: tuple = (30, 8), tmin: int = 0,
                         x_units: Union[int, float] = 100):
    """
    Plots crosswavelet phase angle results.

    Arugments:

        sig1: mne.Epochs
            Neural data signal of participant 1.

        sig2: mne.Epochs
            Neural data signal of participant 2.

        sfreq: int | float
            Sampling frequency of the data in Hz.

        freqs: int | float | np.ndarray
            Frequency range of interest in Hz.

        time: int
            Time of sample duration in seconds.

        figsize: tuple
            Figure size (default is (30, 8).

        tmin: int
            Minimum x-value (default is 0).

        x_units: int | float
            Distance between xticks on x-axis (default is 100).

    Returns:
        Figure of xwt phase angle results
    """
    result, _ = xwt(sig1, sig2, sfreq, freqs)
    data = np.angle(result[:, :], 0)
    analysis = 'phase'
    fig = plot_xwt(analysis, data, sfreq, freqs, time, figsize, tmin, x_units)
    plt.title('Cross Wavelet Transform (Phase Angle)')
    plt.colorbar(label='phase difference')

    return fig


def plot_wct(sig1: mne.Epochs, sig2: mne.Epochs, sfreq: Union[int, float],
             freqs: Union[int, float, np.ndarray], time: int,
             figsize: tuple = (30, 8), tmin: int = 0,
             x_units: Union[int, float] = 100):
    """
    Plots wavelet coherence analyses results.

    Arugments:
        sig1: mne.Epochs
            Neural data signal of participant 1.

        sig2: mne.Epochs
            Neural data signal of participant 2.

        sfreq: int | float
            Sampling frequency of the data in Hz.

        freqs: int | float | np.ndarray
            Frequency range of interest in Hz.

        time: int
            Time of sample duration in seconds.

        figsize: tuple
            Figure size (default is (30, 8)).

        xmin: int
            Minimum x-value (default is 0).

        x_units: int | float
            Distance between xticks on x-axis (default is 100).

    Returns:
        Figure of wct results.
    """

    _, result = xwt(sig1, sig2, sfreq, freqs)
    analysis = 'wct'

    fig = plot_xwt(analysis, result, sfreq, freqs, time, figsize, tmin,
                   x_units)
    plt.title('Wavelet Coherence plot')
    plt.colorbar(label='Coherence')

    return fig
