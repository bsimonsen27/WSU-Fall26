# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 09:59:53 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p3()
    return

def p1():
    # constants
    n = 0.6
    f = 10e9
    c_0 = 300e6
    lam = c_0/f
    G_db = np.linspace(25,35,100)
    # convert from db to linear
    G = 10**(G_db/10)
    
    D = np.sqrt(G/n) * lam/np.pi
    
    #plotting
    plt.plot(G_db, D)
    plt.xlabel("Gain (dB)")
    plt.ylabel("Diameter (m)")
    plt.title("X-Band Radar Diameter v Gain")
    plt.grid()
    plt.show()
    
def p2():
    # constants
    f = 1e9
    c_0 = 299.792458e6
    lam = c_0/f
    v = 100
    beta = v/c_0
    
    # doppler frequencies
    delta_f = 2*(v/lam)
    f_real = f*((1+beta)/(1-beta) - 1)
    err = np.abs(delta_f - f_real)
    
    print("delta_f=",delta_f)
    print("f_real=",f_real)
    print("err=",err)
    
    # part b
    v2 = 686
    beta2 = v2/c_0
    delta_f = 2*(v2/lam)
    f_real = f*((1+beta2)/(1-beta2) - 1)
    err = np.abs(delta_f - f_real)
    
    print("\nb)")
    print("delta_f=",delta_f)
    print("f_real=",f_real)
    print("err=",err)
    
def p3():
    # constants
    v = 200          
    h = 50 / 1.60934 
    
    # horizontal position of aircraft
    x = np.linspace(-200, 200, 1000)
    
    # radial velocity
    vr = -v * x / np.sqrt(x**2 + h**2)
    
    # plot
    plt.plot(x, vr)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.xlabel("Horizontal position from radar (miles)")
    plt.ylabel("Radial velocity (mph)")
    plt.title("Radial Velocity of Aircraft")
    plt.grid()
    
    plt.show()
    
def p4():
    # constants
    N = 20              # number of pulses
    PRI = 1e-3          # 1 ms
    tau = 10e-6         # 10 us pulse width
    
    PRF = 1 / PRI
    CPI = N * PRI
    fs = 1e6            # 1 MHz sampling rate
    dt = 1 / fs
    
    # Create time vector for the CPI
    t = np.arange(0, CPI, dt)
    
    # pulse train
    signal = np.zeros(len(t))
    
    for n in range(N):
        start = n * PRI
        end = start + tau
        signal[(t >= start) & (t < end)] = 1
    
    # FFT
    S = np.fft.fftshift(np.fft.fft(signal))
    freq = np.fft.fftshift(np.fft.fftfreq(len(signal), dt))
    
    # normalize
    S_mag = np.abs(S)
    S_mag = S_mag / np.max(S_mag)
    
    #khz
    freq_khz = freq / 1000
    
    # plot
    plt.figure(figsize=(10, 5))
    plt.plot(freq_khz, S_mag)
    
    plt.xlim(-0.052, 0.052)
    plt.xlabel("Frequency (kHz)")
    plt.ylabel("Normalized Magnitude")
    plt.title("Spectrum Pulse Train")
    plt.grid()
    
    plt.show()


if __name__ == "__main__":
    main()