# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 09:33:23 2026

@author: brend
"""
import numpy as np
import matplotlib.pyplot as plt

def main():
    p1()
    p2()
    p3()
    hw07p2()
    hw07p3()
    return

def p1():
    
    theta = np.linspace(-np.pi, np.pi, 1000)
    AF = 2 * np.cos(np.pi / 4 * (np.cos(theta) - 1))
    # normalize
    AF_norm = np.abs(AF) / np.max(np.abs(AF))
    
    # plot
    fix, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(theta, AF_norm)
    ax.set_ylim(0,1)
    ax.set_title("Endfire-Cardiod Pattern")
    ax.grid(True)
    plt.show()
    # fix, ax = plt.subplots(subplot_kw={"projection": "polar"})
    # ax.plot(theta, AF)
    # ax.set_ylim(0,2)
    # ax.set_title("Endfire-Cardiod Pattern")
    # ax.grid(True)
    # plt.show()

def p2():
    #parameter
    delta_x = 1/4
    theta = np.linspace(-np.pi, np.pi, 1000)
    k = 2*np.pi
    
    delta = delta_x * np.sin(theta)
    t = k * delta / 2
    
    M_list = [3,5,11,15]
    
    # conversion to degrees
    theta_deg = np.rad2deg(theta)
    
    # create plot
    for M in M_list:
        AF = np.sinc(M * t / np.pi) / np.sinc(t / np.pi)
        # get magnitude
        AF_mag = np.abs(AF)
        # to dB
        AF_db = 20 * np.log10(np.maximum(AF_mag, 1e-6))
        # plot
        fig, ax = plt.subplots()
        ax.plot(theta_deg, AF_db)
        ax.set_title(f"Broadside Pattern where M={M}")
        ax.set_xlabel("dB")
        ax.set_ylabel("Theta (degrees)")
        ax.set_xlim(-180,180)
        ax.set_ylim(-20,0)
        ax.grid(True)
    
    plt.tight_layout()
    plt.show

def p3():
    theta = np.linspace(-np.pi, np.pi, 1000)
    k = 2*np.pi 
    M = 11
    delta_x = 1/4
    theta_s = np.deg2rad(30)
    delta = delta_x * (np.sin(theta) - np.sin(theta_s))
    t = k * delta / 2
    # normalize AF
    AF = np.sinc(M * t/np.pi) / np.sinc(t / np.pi)
    # get mag and dB
    AF_mag = np.abs(AF)
    AF_dB = 20 * np.log10(np.maximum(AF_mag, 1e-6))
    # convert to degrees
    theta_deg = np.rad2deg(theta)
    
    # plot
    plt.plot(theta_deg, AF_dB)
    plt.title("Scan at 30 degrees")
    plt.xlabel("dB")
    plt.ylabel("Theta (degrees)")
    plt.grid(True)
    plt.xlim(-180,180)
    plt.ylim(-30,0)
    
    plt.show()
    
def hw07():
    theta = np.linspace(-np.pi, np.pi, 1000)
    theta_deg = np.rad2deg(theta)

    lam = 1
    delta_x = 0.35 * lam
    k = 2 * np.pi

    M = np.arange(5, 26)

    # Array length
    L_eff = M * delta_x

    beamwidth = np.zeros_like(M, dtype=float)

    # Half-power magnitude
    half_power = 1 / np.sqrt(2)

    # Calculate array factor for each M
    for i, m in enumerate(M):

        t = k * delta_x * np.sin(theta) / 2

        AF = np.sinc(m * t / np.pi) / np.sinc(t / np.pi)
        AF_mag = np.abs(AF)

        # Find peak of main beam
        peak_idx = np.argmax(AF_mag)

        # Find the index corresponding to broadside (theta = 0)
        peak_idx = np.argmin(np.abs(theta))
        
        # Find half-power crossing to the left of broadside
        left_indices = np.where(
            AF_mag[:peak_idx] <= half_power
        )[0]
        
        left_idx = left_indices[-1]
        
        # Find half-power crossing to the right of broadside
        right_indices = np.where(
            AF_mag[peak_idx:] <= half_power
        )[0]
        
        right_idx = peak_idx + right_indices[0]
        
        # Interpolate the crossing angles
        theta_left = np.interp(
            half_power,
            AF_mag[left_idx:left_idx + 2][::-1],
            theta_deg[left_idx:left_idx + 2][::-1]
        )
        
        theta_right = np.interp(
            half_power,
            AF_mag[right_idx - 1:right_idx + 1][::-1],
            theta_deg[right_idx - 1:right_idx + 1][::-1]
        )
        
        # Calculate full half-power beamwidth
        beamwidth[i] = theta_right - theta_left
        
    # Plot results
    plt.figure()
    plt.plot(L_eff, beamwidth, 'o-')
    plt.xlabel(r'Effective Array Length ($L_{\mathrm{eff}}/\lambda$)')
    plt.ylabel('Half-Power Beamwidth (degrees)')
    plt.title('Array Length vs. Half-Power Beamwidth')
    plt.grid(True)
    plt.show()
    
def hw07p2():
    # question 2
    M = 11
    # from 0.25*lambda to lambda. Step size of 0.25
    delta_x = [0.25, 0.5, 0.75, 1.0]
    # parameters
    theta = np.linspace(-np.pi, np.pi, 1000)
    theta_deg = np.rad2deg(theta)

    k = 2 * np.pi
    
    fig, ax = plt.subplots(figsize=(12,5))
    
    # itereate through delta_x values
    for x in delta_x:
        # calculate array factor
        t = k*x*np.sin(theta) / 2
        AF = np.sinc(M * t / np.pi) / np.sinc(t/np.pi)
        #magnitude
        AF_mag = np.abs(AF)
        AF_dB = 20*np.log10(np.maximum(AF_mag, 1e-6))
        
        # plot 
        ax.plot(theta_deg, AF_dB, label=fr'$\Delta x={x}\lambda$')
    
    ax.set_xlabel('Angle (degrees)')
    ax.set_ylabel('Array Factor (dB)')
    ax.set_title('Element Spacing vs Array Pattern')
    ax.set_xlim(-90, 90)
    ax.set_ylim(-40, 0)
    ax.grid(True)
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    
def hw07p3():

    # Parameter
    n_B = 16
    max_range = 68 #km
    R1 = 16 #km
    targets = [12,36,48]
    pulse_pos1 = np.arange(0,max_range+1, R1)
    
    # add target 1 blips
    t1_echo = pulse_pos1 + targets[0]
    t1_echo = t1_echo[t1_echo <= max_range]
    
    # add target 2 blips
    t2_echo = pulse_pos1 + targets[1]
    t2_echo = t2_echo[t2_echo <= max_range]
    
    # add target 3 blips
    t3_echo = pulse_pos1 + targets[2]
    t3_echo = t3_echo[t3_echo <= max_range]
    
    #plotting
    plt.figure(figsize=(15,5))
    plt.stem(pulse_pos1, np.ones(len(pulse_pos1)),
             basefmt=" ", label="Radar Pulse")
    #t1 blips
    plt.stem(
        t1_echo,
        np.full(len(t1_echo), 0.6),
        linefmt="r-",
        markerfmt="rv",
        basefmt=" ",
        label="Target 1"
    )
    #t2 blips
    markerline, stemlines, baseline = plt.stem(
        t2_echo,
        np.full(len(t2_echo), 0.6),
        markerfmt="v",
        basefmt=" ",
        label="Target 2"
    )
    plt.setp(stemlines, color="darkorange")
    plt.setp(markerline, color="darkorange")
    #t3 blips
    markerline3, stemlines3, baseline3 = plt.stem(
        t3_echo,
        np.full(len(t3_echo), 0.6),
        markerfmt="v",
        basefmt=" ",
        label="Target 3"
    )
    plt.setp(stemlines3, color="green")
    plt.setp(markerline3, color="green")
    # ticks for bins
    plt.xticks(np.arange(0, max_range + 1, 1))
    
    # Label each 1 km bin from 0 to 15, repeating every 16 km
    bin_ticks = np.arange(0, max_range + 1, 1)
    bin_labels = [str(int(x % n_B)) for x in bin_ticks]
    
    plt.xticks(bin_ticks, bin_labels)
    plt.xlim(-1, max_range)
    plt.ylim(0, 1.3)
    plt.xlabel("Range position bins")
    plt.title("Radar Pulses and Returns for PRF1")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(True, axis="x", alpha=0.3)
    plt.show()
    
    # PRF2
    R2 = 17 #km
    pulse_pos2 = np.arange(0,max_range+1, R2)
    
    # add target 1 blips
    t1_echo = pulse_pos2 + targets[0]
    t1_echo = t1_echo[t1_echo <= max_range]
    
    # add target 2 blips
    t2_echo = pulse_pos2 + targets[1]
    t2_echo = t2_echo[t2_echo <= max_range]
    
    # add target 3 blips
    t3_echo = pulse_pos2 + targets[2]
    t3_echo = t3_echo[t3_echo <= max_range]
    
    #plotting
    plt.figure(figsize=(15,5))
    plt.stem(pulse_pos2, np.ones(len(pulse_pos2)),
             basefmt=" ", label="Radar Pulse")
    #t1 blips
    plt.stem(
        t1_echo,
        np.full(len(t1_echo), 0.6),
        linefmt="r-",
        markerfmt="rv",
        basefmt=" ",
        label="Target 1"
    )
    #t2 blips
    markerline, stemlines, baseline = plt.stem(
        t2_echo,
        np.full(len(t2_echo), 0.6),
        markerfmt="v",
        basefmt=" ",
        label="Target 2"
    )
    plt.setp(stemlines, color="darkorange")
    plt.setp(markerline, color="darkorange")
    #t3 blips
    markerline3, stemlines3, baseline3 = plt.stem(
        t3_echo,
        np.full(len(t3_echo), 0.6),
        markerfmt="v",
        basefmt=" ",
        label="Target 3"
    )
    plt.setp(stemlines3, color="green")
    plt.setp(markerline3, color="green")
    # ticks for bins
    bin_width = R2 / n_B
    bin_ticks = np.arange(0, max_range + bin_width, bin_width)
    bin_ticks = bin_ticks[bin_ticks <= max_range]
    
    bin_labels = [str(i % n_B) for i in range(len(bin_ticks))]
    
    plt.xticks(bin_ticks, bin_labels)
    plt.xlim(-1, max_range)
    plt.ylim(0, 1.3)
    plt.xlabel("Range position bins")
    plt.title("Radar Pulses and Returns for PRF2")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.grid(True, axis="x", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()