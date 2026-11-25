# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 12:01:27 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

c0 = 300e6

def main():
    p1()
    p2()
    p6()
    pass

def p1():
    r = np.linspace(10e3,100e3,100)     # km
    delta_t = r / c0
    delta_t = delta_t * 10e3        # ms
    
    # plot
    plt.figure(figsize=(12,6))
    plt.plot(r, delta_t)
    plt.title("Range vs. Delay Time")
    plt.xlabel("Range (km)")
    plt.ylabel("Delay Time (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def p2():
    f = np.linspace(200, 2e3, 100)  # Hz
    r = c0 / (2*f)                  # meters
    r *= 10e-3      # km
    f *= 10e-3       # kHz
    
    # plot
    plt.figure(figsize=(12,6))
    plt.plot(r, f)
    plt.title("PRF vs Range")
    plt.ylabel("Range (km)")
    plt.xlabel("PRF (kHz)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # plot
    plt.figure(figsize=(12,6))
    plt.loglog(r, f)
    plt.title("PRF vs Range (Logarithmic)")
    plt.ylabel("Range (km)")
    plt.xlabel("PRF (kHz)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()
    
def p6():
    f_lin = np.linspace(100e3, 100e9, 1000)
    f = np.logspace(5, 11, 1000)
    c = 300e6
    lam = c / f
    lam_lin = c / f_lin
    
    plt.figure(figsize=(12,6))
    plt.plot(f_lin,lam_lin)
    plt.title("Frequency vs Wavelength")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Wavelength (m)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12,6))
    plt.loglog(f,lam)
    plt.title("Frequency vs Wavelength")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Wavelength (m)")
    plt.grid(True, which="both")
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()