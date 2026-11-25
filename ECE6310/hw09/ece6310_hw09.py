# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 11:30:58 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    # p4()
    # p5()
    return

def p1():
    # parameters
    M = 12
    lam = 1
    delta_x = lam / 2
    k = 2*np.pi/lam
    N = M//2
    # window = np.hamming(M)
    window = np.blackman(M)
    ramp = np.linspace(-1,1,M)
    window_d = window * ramp
    # weights = window/ np.max(window)
    # weights_d = window_d/ np.max(window_d)
    
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    AF_sum = np.zeros_like(theta)
    AF_diff = np.zeros_like(theta)
    AF = np.zeros_like(theta, dtype=complex)
    
    for n in range(N):
        AF += np.exp(-1j*2*np.pi/lam*n*delta_x*np.sin(theta))
    AF *= 1/N
    
    AF_db = np.abs(AF) / np.max(np.abs(AF))
    AF_db = 20*np.log10(AF_db)
    
    for m in range(M // 2):
        phase = (k / 2) * (2*m + 1) * delta_x * np.sin(theta)

        AF_sum += window[m] * np.cos(phase)
        AF_diff += window_d[m] * np.sin(phase)
        
    AF_sum = np.abs(AF_sum) / np.max(np.abs(AF_sum))
    AF_diff = np.abs(AF_diff) / np.max(np.abs(AF_diff))
    #dB conversion
    AF_sum = 20*np.log10(AF_sum)
    AF_diff = 20*np.log10(AF_diff)

    plt.figure(figsize=(10, 5))
    plt.plot(np.rad2deg(theta), AF_sum, label="Sum Pattern")
    plt.plot(np.rad2deg(theta), AF_diff, label="Difference Pattern", linestyle='--')
    # plt.plot(np.degrees(theta), AF_db, label="Test")
    plt.ylim(-30,0)
    plt.xlim(-80,80)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Array Factor (dB)")
    plt.title("Monopulse Sum and Difference Patterns")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    AF_sum_linear = 10**(AF_sum / 20)
    AF_diff_linear = 10**(AF_diff / 20)
    
    plt.figure(figsize=(8,8))
    ax = plt.subplot(111, projection='polar')
    ax.plot(theta, AF_sum, label="Sum Pattern")
    ax.plot(theta, AF_diff,label="Diff Pattern", linestyle='--')
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(-30, 0)
    ax.set_title("Monopulse-Sum and DIfference Polar")
    ax.legend(loc="upper right")
    ax.grid(True)
    plt.show()

def p4():
    # barker code
    B_11 = np.array([1,1,1,-1,-1,-1,1,-1,-1,1,-1])
    # autocorrelation of B_11
    R_xx = np.correlate(B_11, B_11, mode='full')
    # there are 2N-1 values in R_xx
    lags = np.arange(-len(B_11)+1, len(B_11))
    #plot R_xx
    plt.figure(figsize=(10,5))
    plt.stem(lags,R_xx)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title('Barker-11 Autocorrelation')
    plt.grid()
    plt.show()
    
    # weiner khincine theorom
    ms = np.size(R_xx)
    m = np.zeros(ms)
    Wdb = np.zeros(ms)
    PSD = np.fft.fftshift(np.fft.fft(R_xx)).real
    Ws = PSD
    Wmg = np.abs(Ws)
    Wmx = np.max(Wmg)
    Wref = np.log10(Wmx)
    N = len(R_xx)
    f = np.fft.fftshift(np.fft.fftfreq(N,d=1))
    freq = np.fft.fftshift(np.fft.fftfreq(len(R_xx)))
    
    # db conversion
    mm = int((ms-1)/2)
    for ii in range(0,ms):
        m[ii] = -mm + ii
        Wnm = (1/Wmx)*Wmg[ii]
        Wdb[ii] = np.log10(Wnm)+Wref
    
    #plot PSD
    plt.stem(m,np.abs(Wdb))
    plt.title('PSD of Barker-11')
    plt.xlabel('m')
    plt.ylabel('Sw_db[m]')
    plt.grid(True)
    plt.show()
      
    
    plt.plot(f,PSD)
    plt.xlabel('Normalized Frequencies')
    plt.ylabel('Power')
    plt.title('Barker-11 PSD')
    plt.grid()
    plt.show()
    
    
    #plt.plot(freq, PSD_max)
    plt.stem(PSD)
    plt.ylabel("PSD")
    plt.xlabel("Frequencey (Hz)")
    plt.title("PSD of 11-segment Barker Code")
    plt.grid(True)
    plt.show()
    
    return

def p5():
    tau = 100e-6    #seconds
    PRF = 1e3       # hertz
    B = 8e3         # hertz
    f_s = 50e3      # hertz
    PRI = 1/PRF
    d = tau/PRI     # duty cycle of 10%
    A = 1
    # create frequencies
    k = np.arange(-8,9)
    f_original = k*PRF
    C = A*d*np.sinc(k*d)
    # replicas from sampling
    m = np.arange(-3,4)
    freq=[]
    amplitude = []
    for i in m:
        freq.extend(f_original + i*f_s)
        amplitude.extend(C)
        
    freq = np.array(freq)
    amplitude = np.array(amplitude)
    
    # Sort frequencies
    idx = np.argsort(freq)
    freq = freq[idx]
    amplitude = amplitude[idx]
    
    # Plot
    plt.figure(figsize=(13, 5))
    plt.stem(freq/1e3, amplitude)
    plt.xlabel('Frequency (kHz)')
    plt.ylabel('Normalized Amplitude')
    plt.title('Spectrum of Impulse-Sampled Rectangular Pulse Train')
    plt.grid()
    plt.xlim(-10, 70)
    plt.show()
    
    # set up time domain rect pulse
    #d = 0.4
    n_harmonics = 8
    w = 2 * np.pi * PRF
    t = np.linspace(0, 4*PRI, 10000)
    x_t = np.ones_like(t)
    for n in range(1, n_harmonics + 1):
        x_t += 2 * (np.sin(n*np.pi*d)/(n*np.pi*d) * 
                    np.cos(n*w*t))
    x_t = A*d*x_t
    
    # plotting
    plt.plot(t*1e3, x_t)
    plt.xlabel('Time (ms)')
    plt.ylabel('Amplitude')
    plt.title('Rectangular Pulse Train')
    plt.grid()
    plt.show()
    
    
    return

if __name__=="__main__":
    main()