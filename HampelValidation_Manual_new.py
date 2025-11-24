# Save this as a new file, e.g., validate_hampel_with_psd.py
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal # For Welch's method to calculate PSD

def hampel_filter(data, window_size=80, n_sigmas=6.0):
    """
    Applies a Hampel filter to the real and imaginary parts of a complex FFT spectrum.
    This identifies and replaces outliers in the frequency domain.
    """
    real_part = np.real(data)
    imag_part = np.imag(data)
    n = len(data)
    new_real = real_part.copy()
    new_imag = imag_part.copy()
    k = 1.4826 

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2 + 1)
        
        # --- Process Real Part ---
        window_real = real_part[start:end]
        median_real = np.median(window_real)
        mad_real = np.median(np.abs(window_real - median_real))
        threshold_real = n_sigmas * k * mad_real
        if mad_real > 1e-9 and np.abs(real_part[i] - median_real) > threshold_real:
            new_real[i] = median_real

        # --- Process Imaginary Part ---
        window_imag = imag_part[start:end]
        median_imag = np.median(window_imag)
        mad_imag = np.median(np.abs(window_imag - median_imag))
        threshold_imag = n_sigmas * k * mad_imag
        if mad_imag > 1e-9 and np.abs(imag_part[i] - median_imag) > threshold_imag:
            new_imag[i] = median_imag
            
    return new_real + 1j * new_imag

# --- 1. Configuration ---
file_on = "XUAWAKE7.EDF"
file_off = "XUAWAKEPRE_deidentified.EDF"
channel_to_process = 'Fp2'
dbs_stim_freq = 130.0  # Hz
num_harmonics = 5

base_dir = Path(".")
in_dir = base_dir / "Data" / "raw data" / "XU"
plots_dir = base_dir / "Data" / "validation_sun_method_plots"
plots_dir.mkdir(parents=True, exist_ok=True)

# --- 2. Load and Pre-process Data ---
path_on = in_dir / file_on
path_off = in_dir / file_off
if not (path_on.exists() and path_off.exists()):
    raise FileNotFoundError("Please ensure both ON and OFF EDF files are in the correct directory.")

print("Loading EEG data...")
raw_on = mne.io.read_raw_edf(path_on, preload=True, verbose='WARNING')
raw_off = mne.io.read_raw_edf(path_off, preload=True, verbose='WARNING')

# --- NEW: Apply bandpass (0.1-70 Hz) and notch (60 Hz) filters ---
print("Applying initial bandpass and notch filters...")
raw_off.filter(l_freq=0.1, h_freq=70, fir_design='firwin', verbose=False)
raw_off.notch_filter(freqs=60, fir_design='firwin', verbose=False)

raw_on.filter(l_freq=0.1, h_freq=70, fir_design='firwin', verbose=False)
raw_on.notch_filter(freqs=60, fir_design='firwin', verbose=False)

# Ensure signals are the same length after potential filtering artifacts
min_samples = min(len(raw_on), len(raw_off))
raw_on.crop(tmax=(min_samples - 1) / raw_on.info['sfreq'])
raw_off.crop(tmax=(min_samples - 1) / raw_off.info['sfreq'])

sfreq = raw_on.info['sfreq']

# Get the pre-processed data arrays
eeg_on_preprocessed, times = raw_on.get_data(picks=[channel_to_process], return_times=True)
eeg_clean_preprocessed, _ = raw_off.get_data(picks=[channel_to_process], return_times=True)
eeg_on_preprocessed = eeg_on_preprocessed.flatten()
eeg_clean_preprocessed = eeg_clean_preprocessed.flatten()

# --- 3. Extract Realistic Artifact from PRE-PROCESSED 'ON' Data ---
print(f"\nExtracting artifact based on {dbs_stim_freq} Hz fundamental frequency...")
n_fft = len(eeg_on_preprocessed)
freqs_fft = np.fft.fftfreq(n_fft, 1/sfreq)
spectrum_on_preprocessed = np.fft.fft(eeg_on_preprocessed)
spectrum_artifact = np.zeros_like(spectrum_on_preprocessed, dtype=complex)

for i in range(1, num_harmonics + 1):
    harmonic_freq = dbs_stim_freq * i
    if harmonic_freq >= sfreq / 2:
        break
    idx = np.argmin(np.abs(freqs_fft - harmonic_freq))
    spectrum_artifact[idx] = spectrum_on_preprocessed[idx]
    spectrum_artifact[-idx] = spectrum_on_preprocessed[-idx]

artifact_extracted = np.real(np.fft.ifft(spectrum_artifact))

# --- 4. Create Contaminated Signal and Apply Hampel Filter ---
print("\nCreating contaminated signal and applying Hampel filter...")
eeg_contaminated = eeg_clean_preprocessed + artifact_extracted

spectrum_contaminated_fft = np.fft.fft(eeg_contaminated)
spectrum_filtered_fft = hampel_filter(spectrum_contaminated_fft)
eeg_filtered_contaminated = np.real(np.fft.ifft(spectrum_filtered_fft))

# --- 5. Calculate PSDs for Comparison ---
# Welch's method provides a smoother, more reliable PSD estimate
nperseg = int(sfreq * 2)  # Use 2-second windows for Welch calculation

# Calculate PSD for the three signals
freqs_welch, psd_clean = signal.welch(eeg_clean_preprocessed, fs=sfreq, nperseg=nperseg)
_, psd_contaminated = signal.welch(eeg_contaminated, fs=sfreq, nperseg=nperseg)
_, psd_filtered = signal.welch(eeg_filtered_contaminated, fs=sfreq, nperseg=nperseg)

# --- 6. Plot PSD Comparison ---
print("Generating PSD comparison plot...")
fig, ax = plt.subplots(figsize=(12, 7))

# Plotting on a logarithmic scale (semilogy) is standard for PSDs
ax.semilogy(freqs_welch, psd_clean, 'b-', label='Clean EEG (Ground Truth)', linewidth=2)
ax.semilogy(freqs_welch, psd_contaminated, 'g-', label='Contaminated (Clean + Artifact)', alpha=0.8)
ax.semilogy(freqs_welch, psd_filtered, 'r-', label='Recovered (After Hampel)', linewidth=1.5, linestyle='--')

ax.set_title(f'PSD Comparison for Hampel Filter Validation ({channel_to_process})')
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Power Spectral Density (V²/Hz)')
ax.set_xlim(0, 70)  # Limit x-axis to our band of interest
ax.grid(True, which='both', linestyle=':', linewidth=0.5)
ax.legend(loc='upper right')
fig.tight_layout()

plot_path = plots_dir / f"psd_validation_plot_{channel_to_process}.png"
fig.savefig(plot_path, dpi=150)
print(f"\nSaved PSD comparison plot to: {plot_path}")



# # Save this as a new file, e.g., validate_with_extracted_artifact.py
# from pathlib import Path
# import mne
# import matplotlib.pyplot as plt
# import numpy as np

# def hampel_filter(data, window_size=80, n_sigmas=6.0):
#     """
#     Applies a Hampel filter to the real and imaginary parts of a complex FFT spectrum.
#     This identifies and replaces outliers in the frequency domain.
#     """
#     real_part = np.real(data)
#     imag_part = np.imag(data)
#     n = len(data)

#     new_real = real_part.copy()
#     new_imag = imag_part.copy()
    
#     # The scaling factor 1.4826 converts the MAD to an estimate of the standard deviation
#     # for normally distributed data.
#     k = 1.4826 

#     for i in range(n):
#         start = max(0, i - window_size // 2)
#         end = min(n, i + window_size // 2 + 1)
        
#         # --- Process Real Part ---
#         window_real = real_part[start:end]
#         median_real = np.median(window_real)
#         # Median Absolute Deviation (MAD) for the real part
#         mad_real = np.median(np.abs(window_real - median_real))
#         # Threshold is based on n sigmas from the median
#         threshold_real = n_sigmas * k * mad_real
#         # If the point is an outlier, replace it with the local median
#         if mad_real > 1e-9 and np.abs(real_part[i] - median_real) > threshold_real:
#             new_real[i] = median_real

#         # --- Process Imaginary Part ---
#         window_imag = imag_part[start:end]
#         median_imag = np.median(window_imag)
#         # Median Absolute Deviation (MAD) for the imaginary part
#         mad_imag = np.median(np.abs(window_imag - median_imag))
#         # Threshold is based on n sigmas from the median
#         threshold_imag = n_sigmas * k * mad_imag
#         # If the point is an outlier, replace it with the local median
#         if mad_imag > 1e-9 and np.abs(imag_part[i] - median_imag) > threshold_imag:
#             new_imag[i] = median_imag
            
#     # Recombine the filtered real and imaginary parts
#     return new_real + 1j * new_imag

# # --- 1. Configuration ---
# # --- Files and Channel ---
# file_on = "XUAWAKE7.EDF"
# file_off = "XUAWAKEPRE_deidentified.EDF"
# channel_to_process = 'Fp2'

# # --- DBS Settings ---
# # This is crucial for identifying which frequencies to extract
# dbs_stim_freq = 130.0  # Hz
# # How many harmonics to extract (e.g., 130Hz, 260Hz, 390Hz, etc.)
# # Higher numbers capture more of the artifact's complex shape.
# num_harmonics = 5

# # --- Directories ---
# base_dir = Path(".")
# in_dir = base_dir / "Data" / "raw data" / "XU"
# plots_dir = base_dir / "Data" / "validation_sun_method_plots"
# plots_dir.mkdir(parents=True, exist_ok=True)

# # --- 2. Load and Prepare Data ---
# path_on = in_dir / file_on
# path_off = in_dir / file_off

# if not (path_on.exists() and path_off.exists()):
#     raise FileNotFoundError("Please ensure both ON and OFF EDF files are in the correct directory.")

# print("Loading and preparing EEG data...")
# raw_on = mne.io.read_raw_edf(path_on, preload=True, verbose='WARNING')
# raw_off = mne.io.read_raw_edf(path_off, preload=True, verbose='WARNING')

# # For a valid comparison, the signals must be the same length.
# min_samples = min(len(raw_on), len(raw_off))
# raw_on.crop(tmax=(min_samples - 1) / raw_on.info['sfreq'])
# raw_off.crop(tmax=(min_samples - 1) / raw_off.info['sfreq'])

# sfreq = raw_on.info['sfreq']
# eeg_on, times = raw_on.get_data(picks=[channel_to_process], return_times=True)
# eeg_clean, _ = raw_off.get_data(picks=[channel_to_process], return_times=True)
# eeg_on = eeg_on.flatten()
# eeg_clean = eeg_clean.flatten()

# # --- 3. Extract Realistic Artifact from 'ON' Data (Sun et al. Method) ---
# print(f"\nExtracting artifact based on {dbs_stim_freq} Hz fundamental frequency...")
# n_fft = len(eeg_on)
# freqs = np.fft.fftfreq(n_fft, 1/sfreq)

# # Get the full spectrum of the noisy 'ON' signal
# spectrum_on = np.fft.fft(eeg_on)

# # Create a blank spectrum to hold only the artifact components
# spectrum_artifact = np.zeros_like(spectrum_on, dtype=complex)

# # Identify the frequency bins for each harmonic and copy the data
# print("Identifying and copying harmonic components:")
# for i in range(1, num_harmonics + 1):
#     harmonic_freq = dbs_stim_freq * i
#     if harmonic_freq >= sfreq / 2:
#         print(f"  - Stopping at {harmonic_freq:.1f} Hz (Nyquist limit).")
#         break
    
#     # Find the index of the frequency bin closest to the target harmonic
#     idx = np.argmin(np.abs(freqs - harmonic_freq))
    
#     # Copy the complex value (real and imaginary parts) from the ON spectrum
#     # This preserves both the magnitude and phase of the artifact's component
#     spectrum_artifact[idx] = spectrum_on[idx]
#     spectrum_artifact[-idx] = spectrum_on[-idx]  # Copy the symmetric negative frequency

#     print(f"  - Harmonic {i} ({harmonic_freq} Hz): Copied data from bin {freqs[idx]:.2f} Hz.")

# # Convert the artifact's spectrum back into a time-domain signal
# artifact_extracted = np.real(np.fft.ifft(spectrum_artifact))

# # --- 4. Create Test Signals and Run Validation ---
# print("\nCreating contaminated signal and running filter...")
# # The realistic simulated 'ON' signal
# eeg_contaminated = eeg_clean + artifact_extracted

# # Apply Hampel filter to the contaminated signal
# spectrum_contaminated_fft = np.fft.fft(eeg_contaminated)
# spectrum_filtered_fft = hampel_filter(spectrum_contaminated_fft)
# eeg_filtered_contaminated = np.real(np.fft.ifft(spectrum_filtered_fft))

# # Apply Hampel filter to the clean signal (for the safety check)
# spectrum_clean_fft = np.fft.fft(eeg_clean)
# spectrum_filtered_clean_fft = hampel_filter(spectrum_clean_fft)
# eeg_filtered_clean = np.real(np.fft.ifft(spectrum_filtered_clean_fft))

# # --- 5. Calculate and Report Metrics ---
# power_clean = np.mean(eeg_clean**2)
# power_artifact = np.mean(artifact_extracted**2)
# snr = power_clean / power_artifact

# # a. OFF vs ON (Baseline - how much the artifact corrupts the signal)
# r2_off_on = np.corrcoef(eeg_clean, eeg_contaminated)[0, 1]**2
# mse_off_on = np.mean((eeg_clean - eeg_contaminated)**2)

# # b. OFF vs OFF_HF (Safety - how much the filter distorts a clean signal)
# r2_off_offhf = np.corrcoef(eeg_clean, eeg_filtered_clean)[0, 1]**2
# mse_off_offhf = np.mean((eeg_clean - eeg_filtered_clean)**2)

# # c. OFF vs ON_HF (Performance - how well the filter recovers the clean signal)
# r2_off_onhf = np.corrcoef(eeg_clean, eeg_filtered_contaminated)[0, 1]**2
# mse_off_onhf = np.mean((eeg_clean - eeg_filtered_contaminated)**2)

# print("\n--- Validation Results ---")
# print(f"Signal-to-Noise Ratio (EEG Power / Artifact Power): {snr:.4f}")
# print(f"Artifact power is {1/snr:.1f}x greater than EEG power.")
# print("-" * 28)
# print(f"1. OFF-ON (Baseline):      r^2 = {r2_off_on:<7.4f} | MSE% = {mse_off_on / power_clean * 100:.2f}%")
# print(f"2. OFF-OFFHF (Safety Check): r^2 = {r2_off_offhf:<7.4f} | MSE% = {mse_off_offhf / power_clean * 100:.2f}%")
# print(f"3. OFF-ONHF (Performance):   r^2 = {r2_off_onhf:<7.4f} | MSE% = {mse_off_onhf / power_clean * 100:.2f}%")

# # --- 6. Plot for Visual Inspection ---
# fig, ax = plt.subplots(figsize=(15, 7))
# plot_start_time = 20.0
# plot_duration = 0.4

# time_mask = (times >= plot_start_time) & (times < plot_start_time + plot_duration)

# ax.plot(times[time_mask], eeg_clean[time_mask] * 1e6, 'b-', label='Original EEG (OFF)', linewidth=1.5)
# ax.plot(times[time_mask], eeg_contaminated[time_mask] * 1e6, 'g-', label='EEG + Extracted Artifact (Simulated ON)', alpha=0.6)
# ax.plot(times[time_mask], eeg_filtered_contaminated[time_mask] * 1e6, 'r-', label='Filtered Signal (Recovered)', linewidth=1.5)

# ax.set_xlim(plot_start_time, plot_start_time + plot_duration)
# ax.set_xlabel('Time (s)')
# ax.set_ylabel('Amplitude (µV)')
# ax.set_title(f'Filter Validation using Empirically Extracted Artifact ({channel_to_process})')
# ax.legend(loc='upper right')
# ax.grid(True, linestyle=':')
# fig.tight_layout()

# plot_path = plots_dir / f"validation_plot_{channel_to_process}.png"
# fig.savefig(plot_path, dpi=150)
# print(f"\nSaved validation plot to: {plot_path}")