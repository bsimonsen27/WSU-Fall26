# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 21:05:00 2026

@author: brend
"""

import numpy as np
import matplotlib.pyplot as plt

def main():
    matched_filter()
    return

def matched_filter():
    fs = 10e6
    tau = 10e-6
    A = 1
    PRI = 200e-6
    PRF = 1/PRI
    D = tau/PRI
    
    t = np.arange(0,1*PRI,1/fs)
    t_pulse = np.arange(0, PRI, 1/fs)
    
    # transmitted rectangular pulse
    tx = A * np.zeros(len(t))
    tx[(t%PRI) < tau] = A
    pulse = A*np.ones(int(tau*fs))
    
    # received signal
    # r(t) = a*s(t-tau_d) + n(t)
    distance = 18e3 #meters
    c = 3e8
    tau_d = 2*distance / c
    alpha = 0.2
    rx = np.zeros(len(t))
    rx[(t>= tau_d) & (t<tau_d+tau)] = alpha*A
    
    # plot transmitted pulse
    plt.figure(figsize=(10,4))
    plt.plot(t*1e6,tx, label="Tx")
    # plt.plot(t_pulse*1e6, pulse, label="Tx")
    plt.plot(t*1e6, rx, label="Rx")
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("Signals")
    plt.legend()
    plt.grid()
    plt.show()
    
    # add noise
    noise_std = 0.15
    n_t = np.random.normal(0, noise_std, len(t))
    rx_noisy = rx + n_t
    
    plt.figure()
    plt.plot(t*1e6, rx_noisy, label="Rx noisey")
    plt.plot(t*1e6, rx, label="Rx")
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("Rx clean and noisy")
    plt.legend()
    plt.grid()
    plt.show()
    
    # let's work on the matched filter
    # h[n] = s*[N-1-n]
    h = np.conj(pulse[::-1])
    # convolve noisy signal with matched filter
    mf_output = np.convolve(rx_noisy, h, mode='full')
    # normalize output
    mf_output_norm = mf_output / np.sum(np.abs(pulse)**2)
    # new time vector for convolved output
    t_mf = t[0] + (np.arange(len(mf_output)) - (len(pulse)-1))/fs
    
    plt.figure(figsize=(10,4))
    plt.plot(t*1e6, rx_noisy, label="Rx noisey")
    plt.plot(t*1e6, rx, linestyle="--", label="Rx clean")
    plt.plot(t_mf*1e6, mf_output_norm, color="red", label="Rx matched filter")
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("Rx Comparison Normalized")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
    
    plt.figure(figsize=(10,4))
    plt.plot(t*1e6, rx_noisy, label="Rx noisey")
    plt.plot(t*1e6, rx, linestyle="--", label="Rx clean")
    plt.plot(t_mf*1e6, mf_output, color="red", label="Rx matched filter")
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("Rx Comparison")
    plt.legend(loc="upper right")
    plt.grid()
    plt.show()
    
    # plot in dB
    mf_dB = 20 * np.log10(np.abs(mf_output)/np.max(mf_output)) + 1e-12
    
    plt.figure(figsize=(10,4))
    plt.plot(t_mf*1e6, mf_dB)
    plt.xlabel("Time(us)")
    plt.ylabel("Amplitude (dB)")
    plt.ylim(-40,0)
    plt.title("Matched Filter output (dB)")
    plt.grid()
    plt.show()
    
    # targets range calculated from mf_output
    range_axis = c * t_mf / 2
    range_idx = np.argmax(mf_output)
    range_val = range_axis[range_idx]
    print(f"The target is at {range_val}km")
    
    plt.figure(figsize=(10,4))
    plt.plot(range_axis/1000, mf_output)
    plt.axvline(x=range_val/1000, color="red", linestyle="--")
    plt.xlabel("Range (km)")
    plt.title("Range Detection")
    plt.grid()
    plt.show()
    
    

if __name__=="__main__":
    main()