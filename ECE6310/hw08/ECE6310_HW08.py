# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 17:14:11 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    p2()

def p1():
    # number of array elements
    M = 10
    # antenna parameters
    theta = np.linspace(-np.pi,np.pi, 10000)
    lam = 1
    delta_x = lam / 2
    k = 2 * np.pi / lam
    
    gamma = k*delta_x*np.sin(theta)
    AF = np.ones_like(theta)
    AF_n = np.ones_like(theta)
    
    for m in range(1,M+1):
        a_m = 0.5 *(1 + np.cos((m/M) * (np.pi/2)))
        AF += 2 * a_m * np.cos(m*gamma)
        AF_n += 2 * np.cos(m*gamma)
    # normalize the array factor
    AF = np.abs(AF)
    AF_norm = AF/np.max(AF)
    AF_n = np.abs(AF_n)
    AF_n_norm = AF_n/np.max(AF_n)
    # decibels conversion
    AF_dB = 20 * np.log10(np.maximum(AF_norm, 1e-16))
    AF_n_dB = 20 * np.log10(np.maximum(AF_n_norm, 1e-16))
    
    # instance of just M = 4
    AF4 = np.ones_like(theta)
    M2 = 4
    for m in range(1,M2+1):
        a_m = 0.5 *(1 + np.cos((m/M2) * (np.pi/2)))
        AF4 += 2 * a_m * np.cos(m*gamma)
    # normalize the array factor
    AF4 = np.abs(AF4)
    AF_norm4 = AF4/np.max(AF4)
    # decibels conversion
    AF_dB4 = 20 * np.log10(np.maximum(AF_norm4, 1e-16))
    
    # M3
    AF_3 = np.ones_like(theta)
    M3 = 14
    for m in range(1,M3+1):
        a_m = 0.5 *(1 + np.cos((m/M3) * (np.pi/2)))
        AF_3 += 2 * a_m * np.cos(m*gamma)
    # normalize the array factor
    AF_3 = np.abs(AF_3)
    AF_norm3 = AF_3/np.max(AF_3)
    # decibels conversion
    AF_dB3 = 20 * np.log10(np.maximum(AF_norm3, 1e-16))
    
    #####################################################
    # implementind blackman window
    N = 2*M + 1
    # generate Blackman window
    window = np.blackman(N)
    AF_blackman = np.full_like(theta, window[M], dtype=float)

    for m in range(1, M + 1):
        AF_blackman += (
            2 * window[M + m] * np.cos(m * gamma)
        )
    #normalize
    AF_blackman = np.abs(AF_blackman)
    AF_blackman = AF_blackman/ np.max(AF_blackman)
    # convert to dB
    AF_blackman_dB = 20*np.log10(np.maximum(AF_blackman,1e-16))
    
    #####################################################
    # implementind Hamming window
    N = 2*M + 1
    # generate Blackman window
    window = np.hamming(N)
    # np.full_like similar to np.zeros_like. First element is w[M]
    AF_hamming = np.full_like(theta, window[M], dtype=float)

    for m in range(1, M + 1):
        AF_hamming += (
            2 * window[M + m] * np.cos(m * gamma)
        )
    #normalize
    AF_hamming = np.abs(AF_hamming)
    AF_hamming = AF_hamming/ np.max(AF_hamming)
    # convert to dB
    AF_hamming_dB = 20*np.log10(np.maximum(AF_hamming,1e-16))
    
    # plotting
    plt.figure(figsize=(7,5))
    #plt.plot(np.rad2deg(theta), AF_dB4, label=f'M={M2}')
    plt.plot(np.rad2deg(theta), AF_n_dB, label=f'M={M} no a[m]')
    plt.plot(np.rad2deg(theta), AF_dB, label=f'M={M}')
    plt.plot(np.rad2deg(theta), AF_blackman_dB, label='Blackman')
    plt.plot(np.rad2deg(theta), AF_hamming_dB, label='Hamming')
    #plt.plot(np.rad2deg(theta), AF_dB3, label=f'M={M3}')
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Noramlized AF (dB)")
    plt.title(f"Array Factor Sidelobe Reduction M={M}")
    plt.grid(True)
    plt.legend()
    plt.ylim(-70, 5)
    plt.xlim(-90,90)
    plt.show()
    
    # polar plotting
    AF_polar = AF_dB + 40
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    tho = np.pi/2
    ax.set_theta_direction(-1)
    ax.set_theta_offset(tho)
    ax.set_rlim(0,40)
    ax.set_rticks([0, 10, 20, 30, 40])
    ax.set_yticklabels(["-40 dB", "-30 dB", "-20 dB", "-10 dB", "0 dB"])
    ax.plot(theta, AF_polar)
    #ax.set_ylim(0,1)
    ax.set_title(f"Array Factor Sidelobe Reduction M={M} (Polar)")
    ax.grid(True)
    plt.show()
    

def p2():
    # parameters
    M = 12
    lam = 1
    delta_x = lam / 2
    k = 2*np.pi/lam
    N = M//2 + 1
    window = np.hamming(N)
    weights = window/ np.max(window)
    
    theta = np.linspace(-np.pi/2, np.pi/2, 10000)
    AF_sum = np.zeros_like(theta)
    AF_diff = np.zeros_like(theta)
    
    for m in range(M // 2 + 1):
        phase = (k / 2) * (2*m + 1) * delta_x * np.sin(theta)

        AF_sum += weights[m] * np.cos(phase)
        AF_diff += weights[m] * np.sin(phase)
        
    AF_sum = np.abs(AF_sum) / np.max(np.abs(AF_sum))
    AF_diff = np.abs(AF_diff) / np.max(np.abs(AF_diff))
    #dB conversion
    AF_sum = 20*np.log10(np.maximum(AF_sum,1e-16))
    AF_diff = 20*np.log10(np.maximum(AF_diff,1e-16))

    plt.figure(figsize=(10, 5))
    plt.plot(np.rad2deg(theta), AF_sum, label="Sum Pattern")
    plt.plot(np.rad2deg(theta), AF_diff, label="Difference Pattern")
    plt.ylim(-60,0)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Array Factor (dB)")
    plt.title("Monopulse Sum and Difference Patterns")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    
    return

if __name__ == "__main__":
    main()