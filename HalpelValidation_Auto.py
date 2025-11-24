from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks

# (The hampel_filter helper function remains the same)
def hampel_filter(data, window_size=80, n_sigmas=6.0):
    """Applies a Hampel filter to a 1D array to remove outliers."""
    n = len(data)
    new_data = data.copy()
    k = 1.4286

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)
        window = data[start:end]

        local_median = np.median(window)
        mad = np.median(np.abs(window - local_median))
        S = k * mad
        threshold = n_sigmas * S

        if np.abs(data[i] - local_median) > threshold:
            new_data[i] = local_median
            
    return new_data

def validate_hampel_filter_per_channel(raw_off, raw_on, plots_dir):
    """
    Validates the Hampel filter on each EEG channel individually.
    """
    base_filename = raw_off.filenames[0].stem
    print(f"--- Running per-channel validation for {base_filename} ---")
    
    sfreq = raw_off.info['sfreq']
    eeg_picks = mne.pick_types(raw_off.info, eeg=True)
    eeg_ch_names = [raw_off.ch_names[i] for i in eeg_picks]

    # 1. Get data for all EEG channels
    eeg_clean, times = raw_off.get_data(picks='eeg', return_times=True)
    eeg_on = raw_on.get_data(picks='eeg')

    print("\n--- Validation Results ---")
    # 2. Loop through each EEG channel to perform validation
    for i, ch_name in enumerate(eeg_ch_names):
        # Find artifact parameters from the "ON" data for this channel
        n_fft = len(eeg_on[i])
        spectrum_on = np.fft.fft(eeg_on[i])
        freqs = np.fft.fftfreq(n_fft, 1/sfreq)
        power = np.abs(spectrum_on)**2
        mean_power = np.mean(power[freqs > 1])
        peaks, _ = find_peaks(power, height=mean_power * 50)
        artifact_freqs = freqs[peaks]
        artifact_amps = np.abs(spectrum_on[peaks]) / n_fft

        # Synthesize artifact and create contaminated signal for this channel
        artifact_synth = np.zeros_like(eeg_clean[i])
        for freq, amp in zip(artifact_freqs, artifact_amps):
            if freq == 0: continue
            random_phase = np.random.uniform(-np.pi/2, np.pi/2)
            artifact_synth += amp * np.sin(2 * np.pi * freq * times + random_phase)
        eeg_contaminated = eeg_clean[i] + artifact_synth
        
        # Filter the contaminated signal for this channel
        spectrum_contaminated = np.fft.fft(eeg_contaminated)
        real_cleaned = hampel_filter(np.real(spectrum_contaminated))
        imag_cleaned = hampel_filter(np.imag(spectrum_contaminated))
        eeg_filtered = np.real(np.fft.ifft(real_cleaned + 1j * imag_cleaned))
        
        # Calculate and print metrics for this channel
        r_squared = np.corrcoef(eeg_clean[i], eeg_filtered)[i, i]**2
        mse = np.mean((eeg_clean[i] - eeg_filtered[i])**2)
        print(f"Channel {ch_name}: r^2 = {r_squared:.4f}, MSE = {mse:.6f}")

        # Plot the results for this channel
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(times, eeg_clean[i] * 1e6, color='blue', label='Original (Ground Truth)')
        ax.plot(times, eeg_contaminated * 1e6, color='gray', label='Contaminated', alpha=0.5)
        ax.plot(times, eeg_filtered * 1e6, color='red', label='Filtered')
        ax.set_xlim(times[0], times[0] + 1)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV)')
        ax.set_title(f'Hampel Filter Validation for Channel {ch_name}')
        ax.legend()
        ax.grid(True, linestyle=':')
        fig.tight_layout()
        plot_path = plots_dir / f"{base_filename}_validation_plot_{ch_name}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    print(f"\n-> Saved individual validation plots to: {plots_dir}")


if __name__ == "__main__":
    base_dir = Path(".")
    in_dir = base_dir / "Data" / "raw data" / "XU" 
    plots_dir = base_dir / "Data" / "validation_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    file_off = "XUAWAKEPRE_deidentified.EDF"
    file_on = "XUAWAKE7.EDF"
    
    path_off = in_dir / file_off
    path_on = in_dir / file_on
    
    if path_off.exists() and path_on.exists():
        raw_off = mne.io.read_raw_edf(path_off, preload=True, verbose='WARNING')
        raw_on = mne.io.read_raw_edf(path_on, preload=True, verbose='WARNING')
        validate_hampel_filter_per_channel(raw_off, raw_on, plots_dir)
    else:
        print(f"Error: Could not find one or both files.")

    print("\n✅ Validation script complete.")