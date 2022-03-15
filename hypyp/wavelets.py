"""
Cross wavelet transform and plots
| Option | Description |
| ------ | ----------- |
| title           | wavelets.py |
| authors         | Caitriona Douglas, Guillaume Dumas |
| date            | 2021-03-03 |
"""


import matplotlib

import matplotlib.pyplot as plt

import mne

import numpy as np

import sys

from typing import Union

if not sys.warnoptions:
    import warnings
    warnings.filterwarnings("ignore")


def xwt(sig1: mne.Epochs, sig2: mne.Epochs, sfreq: int,
        freqs: Union[int, np.ndarray]) -> np.ndarray:
    """
    Perfroms a cross wavelet transform on two signals.

    Arguments:
        sig1: signal (eg. EEG data) of first participant
        sig2: signal (eg. EEG data) of second participant
        sfreq: sampling frequency of the data
        freqs: range of frequencies of interest in Hz

    Note:
        This function relies on MNE's mne.time_frequency.morlet
        and mne.time_frequency.tfr.cwt functions.

    Returns:
        (cross_sigs): the crosswavelet transform results
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
        tfr_cwt1 = mne.time_frequency.tfr.cwt(cur_sig1, Ws, use_fft=True,
                                              mode='same', decim=1)

        cur_sig2 = np.squeeze(sig2.get_data(mne.pick_channels(sig2.ch_names,
                                                              [ch_label])))
        tfr_cwt2 = mne.time_frequency.tfr.cwt(cur_sig2, Ws, use_fft=True,
                                              mode='same', decim=1)

        # Perfrom the cross wavelet transform
        cross_sigs[ind, :, :, :] = (tfr_cwt1 * tfr_cwt2.conj())
    return (cross_sigs)


def half_wave(sfreq: int, freqs: Union[int, np.ndarray], n_cycles: int = 6.0,
              sigma: float = None, zero_mean: bool = True) -> tuple:
    """
    Calculates the minimum and maximum half-wavelength of each wavelet, so
    that the area outside of the cone of influence is excluded
    due to the edge effects created by zero padding.

    Arguments:
        sfreq: sampling frequency
        freqs: ranges of frequencies of interest in Hz
        n_cycles: number of cycles of the morelet wavelet (default is 6.0)
        sigma: Controls the width of the wavelet (i.e. temporal resolution.)
        If sigma is None the temporal resolution is adapted with the frequency
        like for all wavelet transform.
        The higher the frequency the shorter is the wavelet.
        If sigma is fixed the temporal resolution is fixed
        like for the short time Fourier transform
        and the number of oscillations
        increases with the frequency (default is None).
        zero_mean: If True, the wavelet has a mean of zero (default is True)

    Note:
        This function relies on MNE's mne.time_frequency.morlet function.

     Returns:
        max_half_wave, min_half_wave: The maximum and minimum half wave lengths
        that specify the bounds of the cone of influence
    """
    Wave = mne.time_frequency.tfr.morlet(sfreq, freqs, n_cycles,
                                         sigma, zero_mean)

    min_half_wave = []
    max_half_wave = []

    for number in Wave:
        min_half_wave.append(len(number) / 2)

    for wl in min_half_wave:
        max_wl = sfreq - wl
        max_half_wave.append(max_wl)

    return min_half_wave, max_half_wave


def plot_xwt(xwt_result: np.ndarray, sfreq: int, freqs: Union[int, np.ndarray],
             time: int, time_conv: int, figsize: tuple, xmin: int,
             x_units: int, y_units: int):
    """
    Plot the results of the Cross wavelet analysis.

    Arguments:
        xwt_result: results of the crosswavelet transform (xwt)
        sfreq: sampling frequency
        freqs: frequency range of interest in Hz
        time: time of sample duration in seconds
        time_conv: time conversion (default is 1000 so time is converted to ms)
        figsize: figure size (default is (30, 8))
        xmin: minimum x-value (default is 0)
        x_units: distance between xticks on x-axis (time) (default is 100)
        y_units: distance between yticks on y-axis (default is 5)

    Note:
        This function is not meant to be called indepedently,
        but is meant to be called when using plot_xwt_crosspower
        or plot_xwt_phase_angle.

    Returns:
    Figure of xwt results
    """
    dt = (time / sfreq) * time_conv

    min_half_wave, max_half_wave = half_wave(sfreq, freqs)

    xmax = len(xwt_result[0, :])
    xticks = np.arange(xmin, xmax, x_units)

    data = xwt_result
    my_cm = matplotlib.cm.get_cmap()
    normed_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    mapped_data = my_cm(normed_data)

    fig = plt.figure(figsize=figsize)
    plt.subplot(122)
    plt.imshow(mapped_data, aspect='auto', interpolation='lanczos')
    plt.gca().invert_yaxis()
    plt.plot(min_half_wave, np.flip(freqs), color='c')
    plt.plot(max_half_wave, np.flip(freqs), color='c')
    plt.ylabel('Frequencies (Hz)')
    plt.xlabel('Time (ms)')
    plt.gca().set_yticks(ticks=range(0, len(freqs), y_units))
    plt.gca().set_yticklabels(labels=freqs[range(0, len(freqs),
                                           y_units)].astype(int))
    plt.gca().set_xticks(ticks=xticks)
    plt.gca().set_xticklabels(labels=xticks * dt)
    plt.xlim(xmin, xmax)
    plt.ylim(xmin, int(len(freqs) - 1))
    plt.cm.get_cmap(my_cm)

    plt.fill_between(min_half_wave, list(range(len(min_half_wave), xmin, -1)),
                     np.max(list(range(xmin, len(min_half_wave)))), color='c')
    plt.fill_between(max_half_wave, list(range(len(max_half_wave), xmin, -1)),
                     np.max(list(range(xmin, len(min_half_wave)))), color='c')
    plt.fill_betweenx
    plt.axvspan(xmin, min(min_half_wave), color='c')
    plt.axvspan(max(max_half_wave), xmax, color='c')

    return fig


def plot_xwt_crosspower(sig1: mne.Epochs, sig2: mne.Epochs, sfreq: int,
                        freqs: Union[int, np.ndarray], time: int,
                        time_conv: int = 1000, figsize: tuple = (30, 8),
                        xmin: int = 0, x_units: int = 100, y_units: int = 5):
    """
    Plots cross wavelet crosspower results.

    Arugments:
        sig1: neural data signal of participant 1
        sig2: neural data signal of participant 2
        sfreq: sampling frequency of the data in Hz
        freqs: frequency range of interest in Hz
        time: time of sample duration in seconds
        time_conv: time conversion (default is 1000 so time is converted to ms)
        figsize: figure size (default is (30, 8))
        xmin: minimum x-value (default is 0)
        x_units: distance between xticks on x-axis (default is 100)
        y_units: distance between yticks on y-axis (default is 5)

    Returns:
        Figure of xwt crosspower results
    """

    result = xwt(sig1, sig2, sfreq, freqs)
    data = np.abs(np.squeeze(np.mean(result[0, :, :, :], 0)))
    fig = plot_xwt(data, sfreq, freqs, time, time_conv, figsize, xmin,
                   x_units, y_units)
    plt.title('Cross Wavelet Transform (Crosspower)')
    plt.colorbar(label='crosspower')

    return fig


def plot_xwt_phase_angle(sig1: mne.Epochs, sig2: mne.Epochs, sfreq: int,
                         freqs: Union[int, np.ndarray], time: int,
                         time_conv: int = 1000, figsize: tuple = (30, 8),
                         xmin: int = 0, x_units: int = 100, y_units: int = 5):
    """
    Plots crosswavelet phase angle results.

    Arugments:
        sig1: neural data signal of participant 1
        sig2: neural data signal of participant 2
        sfreq: sampling frequency of the data in Hz
        freqs: frequency range of interest in Hz
        time: time of sample duration in seconds
        time_conv: time conversion (default is 1000 so time is converted to ms)
        figsize: figure size (default is (30, 8)
        xmin: minimum x-value (default is 0)
        x_units: distance between xticks on x-axis (default is 100)
        y_units: distance between ticks on y-axis (default is 5)

    Returns:
        Figure of xwt phase angle results
    """
    result = xwt(sig1, sig2, sfreq, freqs)
    data = np.angle(np.squeeze(np.mean(result[0, :, :, :], 0)))
    fig = plot_xwt(data, sfreq, freqs, time, time_conv, figsize, xmin,
                   x_units, y_units)
    plt.title('Cross Wavelet Transform (Phase Angle)')
    plt.colorbar(label='phase difference')

    return fig
