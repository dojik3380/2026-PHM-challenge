import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import hilbert

def compare_spectrum(healthy_spec, degraded_spec, fs=25600, nperseg=1024, save_path=None):
    freqs = np.fft.rfftfreq(nperseg, 1/fs)
    if len(freqs) > len(healthy_spec):
        freqs = freqs[:len(healthy_spec)]
    
    plt.figure(figsize=(14, 10))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, healthy_spec, label='Healthy Spectrum', color='blue', alpha=0.8)
    plt.plot(freqs, degraded_spec, label='Degraded Spectrum', color='red', alpha=0.6)
    plt.title("Healthy vs Degraded Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (log1p)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    diff = degraded_spec - healthy_spec
    plt.plot(freqs, diff, label='Difference (Degraded - Healthy)', color='purple', alpha=0.8)
    plt.title("Spectrum Difference (Harmonic Growth Visualization)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Difference")
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

def plot_fault_harmonics(ax, fault_frequencies, max_harmonics=3):
    """
    fault_frequencies: dict e.g. {'BPFO': 105.2, 'BPFI': 142.5}
    """
    colors = ['r', 'g', 'b', 'orange']
    for idx, (fault_name, base_freq) in enumerate(fault_frequencies.items()):
        c = colors[idx % len(colors)]
        for h in range(1, max_harmonics + 1):
            freq = base_freq * h
            alpha = max(0.3, 1.0 - (h - 1) * 0.2)
            label = f"{fault_name} ({h}x)" if h == 1 else None
            ax.axvline(x=freq, color=c, linestyle=':', alpha=alpha, label=label)
            
def plot_degradation_spectrum_evolution(healthy_spec, mid_spec, failure_spec, freqs, fault_frequencies=None, save_path=None):
    plt.figure(figsize=(15, 12))
    
    # 1. Evolution Overlay
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(freqs, healthy_spec, label='Healthy (0-20%)', color='#2ca02c', alpha=0.7)
    ax1.plot(freqs, mid_spec, label='Mid Degradation (40-60%)', color='#ff7f0e', alpha=0.7)
    ax1.plot(freqs, failure_spec, label='Failure (80-100%)', color='#d62728', alpha=0.8)
    
    if fault_frequencies:
        plot_fault_harmonics(ax1, fault_frequencies, max_harmonics=3)
        
    ax1.set_title("Degradation Spectrum Evolution (Healthy -> Mid -> Failure)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Difference Evolution (Failure - Healthy, Mid - Healthy)
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(freqs, mid_spec - healthy_spec, label='Mid Growth', color='#ff7f0e', alpha=0.6)
    ax2.plot(freqs, failure_spec - healthy_spec, label='Failure Growth', color='#d62728', alpha=0.8)
    ax2.axhline(0, color='black', linestyle='--')
    
    if fault_frequencies:
        plot_fault_harmonics(ax2, fault_frequencies, max_harmonics=3)
        
    ax2.set_title("Energy Growth vs Healthy State")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Difference")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

def plot_spectrogram(spectrogram, freqs, times, rpm_trajectory=None, base_fault_freqs=None, save_path=None):
    plt.figure(figsize=(14, 6))
    # spectrogram shape: (len(freqs), len(times)) expected
    if spectrogram.shape[0] != len(freqs):
        spectrogram = spectrogram.T
        
    plt.pcolormesh(times, freqs, spectrogram, shading='gouraud', cmap='viridis')
    plt.title("Time-Frequency Degradation Heatmap (Spectrogram)")
    plt.ylabel("Frequency [Hz]")
    plt.xlabel("Time [Index]")
    plt.colorbar(label='Magnitude')
    
    # RPM trajectory 기반 Moving Harmonics 오버레이
    if rpm_trajectory is not None and base_fault_freqs is not None:
        colors = ['r', 'g', 'orange', 'm']
        for idx, (fault_name, base_freq) in enumerate(base_fault_freqs.items()):
            c = colors[idx % len(colors)]
            # time 단위가 index라면 rpm_trajectory와 길이가 같음
            fault_freq_curve = base_freq * (rpm_trajectory / 1000.0)
            
            # 1x, 2x, 3x harmonics
            for h in range(1, 4):
                alpha = max(0.4, 1.0 - (h - 1) * 0.3)
                label = f"{fault_name} ({h}x)" if h == 1 else None
                plt.plot(times, fault_freq_curve * h, color=c, linestyle='--', alpha=alpha, label=label, linewidth=1.5)
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

def plot_envelope_spectrum(raw_signal, fs, fault_frequencies=None, save_path=None):
    analytic_signal = hilbert(raw_signal)
    amplitude_envelope = np.abs(analytic_signal)
    
    # Remove DC offset from envelope
    amplitude_envelope = amplitude_envelope - np.mean(amplitude_envelope)
    
    env_fft = np.fft.rfft(amplitude_envelope)
    freqs = np.fft.rfftfreq(len(amplitude_envelope), d=1/fs)
    magnitude = np.abs(env_fft)
    
    plt.figure(figsize=(14, 6))
    ax = plt.gca()
    ax.plot(freqs, magnitude, color='black', alpha=0.7)
    
    if fault_frequencies:
        plot_fault_harmonics(ax, fault_frequencies, max_harmonics=3)
        
    plt.title("Envelope Spectrum (Impulsive Modulation Analysis)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Envelope Magnitude")
    plt.xlim(0, 500) # Usually impulsive faults are low frequency in envelope
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close('all')

