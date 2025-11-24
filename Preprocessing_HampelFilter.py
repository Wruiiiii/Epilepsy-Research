import os
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, cheby2, sosfiltfilt

RUN_MODE = 'INSPECT' 


COMPONENTS_TO_EXCLUDE = {
    'XUSLEEP7': [],
    'XUSLEEP60': [],
    'XUSLEEPT100': [],
    'XUAWAKE7': [],
    'XUAWAKE60': [],
    'XUAWAKET100': []
}

def hampel_filter(data, window_size, n_sigmas=3.0):
    n = len(data)
    new_data = data.copy()
    k = 1.4286  # Scale factor for MAD to estimate standard deviation

    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)
        window = data[start:end]

        local_median = np.median(window)
        mad = np.median(np.abs(window - local_median))
        threshold = n_sigmas * k * mad

        if np.abs(data[i] - local_median) > threshold: #|xj - x*| > tS
            new_data[i] = local_median
            
    return new_data

def preprocessing_full_pipeline(raw, plots_dir, out_dir, base_filename):
    print(f"\n--- Processing: {base_filename} ---")

    ## --- Step 1: Load Data ---

    ## --- Step 2: Apply Hampel Filter for DBS Artifact ---
    print("Applying pre-filters (High-pass 1Hz, Low-pass 100Hz)...")
    raw_hampel = raw.copy()
    raw_filtered = raw.copy()

    raw_filtered.filter(
        l_freq=1.0, h_freq=None, method='iir',
        iir_params=dict(order=4, ftype='butter', output='sos')
    )
    raw_filtered.filter(
        l_freq=None, h_freq=100.0, method='iir',
        iir_params=dict(order=8, ftype='cheby2', rs=40, output='sos')
    )
    
    # Extract data and process channel-by-channel
    eeg_data = raw_filtered.get_data(picks='eeg')
    cleaned_eeg_data = np.zeros_like(eeg_data)
    window_size = 80 # N = 80
    n_sigmas = 6.0   # t = 6

    print(f"Applying Hampel filter with N={window_size}, t={n_sigmas}...")
    for i, channel_data in enumerate(eeg_data):
        # 3. FFT
        spectrum = np.fft.fft(channel_data)
        
        # 4. Hampel Identification on real and imaginary parts
        real_cleaned = hampel_filter(np.real(spectrum), window_size, n_sigmas)
        imag_cleaned = hampel_filter(np.imag(spectrum), window_size, n_sigmas)
        
        # 5. Reconstruct the spectrum
        cleaned_spectrum = real_cleaned + 1j * imag_cleaned
        
        # 6. Inverse FFT
        cleaned_signal = np.fft.ifft(cleaned_spectrum)
        cleaned_eeg_data[i, :] = np.real(cleaned_signal)

    # Create a new Raw object with the cleaned data
    info = mne.pick_info(raw.info, mne.pick_types(raw.info, eeg=True))
    raw_cleaned = mne.io.RawArray(cleaned_eeg_data, info)
    

    # --- Step 3: Run ICA ---
    print("Step 3: Running ICA...")
    raw_for_ica = raw_hampel.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw_for_ica, picks='eeg')

    ## --- Step 4: INSPECT Mode ---
    if RUN_MODE == 'INSPECT':
        print("Step 4 (INSPECT): Generating ICA diagnostic plots...")
        ica.plot_components(show=False).savefig(plots_dir / f"{base_filename}_ICA_components_topo.png")
        eog_epochs = mne.preprocessing.create_eog_epochs(raw_hampel, ch_name=['LOC1', 'LOC2'], verbose=False)
        if len(eog_epochs) > 0:
            ica.plot_sources(eog_epochs, show=False).savefig(plots_dir / f"{base_filename}_ICA_EOG_sources.png")
        plt.close('all')
        print("-> Plots generated. Please inspect them and update COMPONENTS_TO_EXCLUDE.")

    ## --- Step 5: CLEAN Mode ---
    elif RUN_MODE == 'CLEAN':
        print("Step 4 (CLEAN): Applying ICA and saving results...")
        components_to_exclude = COMPONENTS_TO_EXCLUDE.get(base_filename)
        if components_to_exclude is None:
            print(f"-> No component list found for {base_filename} in the script. Skipping file.")
            return

        raw_final_cleaned = raw_hampel.copy()
        ica.exclude = components_to_exclude
        ica.apply(raw_final_cleaned)
        return raw_final_cleaned


def comparison_plots(raw_path, cleaned_path, plots_dir, base_filename):
    raw_before = mne.io.read_raw_edf(raw_path, preload=True, verbose='WARNING')
    raw_after = mne.io.read_raw_fif(cleaned_path, preload=True, verbose='WARNING')

    #nyquist_freq = raw.info['sfreq'] / 2
    # --- Create "Before" Plot ---
    fig_before, ax_before = plt.subplots(figsize=(12, 6))
    raw_before.compute_psd(fmax=70).plot(axes=ax_before, show=False, color='black')
    ax_before.set_ylim(-70, 70)
    ax_before.set_title(f'PSD Before Hampel Filter for {base_filename}')
    plot_path_before = plots_dir / f"{base_filename}_psd_before_hampel.png"
    fig_before.savefig(plot_path_before, dpi=150)
    plt.close(fig_before)
    print(f"Saved 'Before' plot to: {plot_path_before}")

    # --- Create "After" Plot ---
    fig_after, ax_after = plt.subplots(figsize=(12, 6))
    raw_after.compute_psd(fmax=70).plot(axes=ax_after, show=False, color='red')
    ax_after.set_ylim(-70, 70)
    ax_after.set_title(f'PSD After Hampel Filter for {base_filename}')
    plot_path_after = plots_dir / f"{base_filename}_psd_after_hampel.png"
    fig_after.savefig(plot_path_after, dpi=150)
    plt.close(fig_after)
    print(f"Saved 'After' plot to: {plot_path_after}")
    
    # --- 2. Time-Domain Butterfly Plot Comparison ---
    scaling = 200e-6 # 200 µV

    # Plot and save the "Before" figure
    fig_before_td = raw_before.plot(duration=5, scalings=scaling, show=False)
    fig_before_td.suptitle(f"Time-Domain Before Hampel Filter - {base_filename}", y=1.0)
    fig_before_td.savefig(plots_dir / f"{base_filename}_time_domain_butterfly_BEFORE.png", dpi=150)
    plt.close(fig_before_td)
    
    # Plot and save the "After" figure
    fig_after_td = raw_after.plot(duration=5, scalings=scaling, show=False)
    fig_after_td.suptitle(f"Time-Domain After Hampel Filter - {base_filename}", y=1.0)
    fig_after_td.savefig(plots_dir / f"{base_filename}_time_domain_butterfly_AFTER.png", dpi=150)
    plt.close(fig_after_td)
    print(f"Saved separate 'Before' and 'After' time-domain butterfly plots.")

    # --- 3. NEW: Single Channel Overlay Plot ---
    # The paper used C3, let's pick a channel that exists in your data
    ch_to_plot = 'A2' # Change this to any channel you want to inspect
    if ch_to_plot in raw_before.ch_names:
        ch_idx = raw_before.ch_names.index(ch_to_plot)
        
        # Get 2 seconds of data
        start_sec, stop_sec = 5, 7
        start_samp = int(start_sec * raw_before.info['sfreq'])
        stop_samp = int(stop_sec * raw_before.info['sfreq'])
        
        data_before, times = raw_before[ch_idx, start_samp:stop_samp]
        data_after, _ = raw_after[ch_idx, start_samp:stop_samp]
        
        fig_overlay, ax_overlay = plt.subplots(figsize=(12, 6))
        ax_overlay.plot(times, data_before.T * 1e6, color='black', label='Before Filter')
        ax_overlay.plot(times, data_after.T * 1e6, color='red', label='After Filter', alpha=0.8)
        ax_overlay.set_xlabel('Time (s)')
        ax_overlay.set_ylabel('Amplitude (µV)')
        ax_overlay.set_title(f"Single Channel Comparison ({ch_to_plot}) for {base_filename}")
        ax_overlay.legend()
        ax_overlay.grid(True, linestyle=':')
        fig_overlay.tight_layout()
        fig_overlay.savefig(plots_dir / f"{base_filename}_time_domain_overlay_{ch_to_plot}.png", dpi=150)
        plt.close(fig_overlay)
        print(f"Saved single channel overlay plot for {ch_to_plot}.")



# Main execution block
if __name__ == "__main__":
    RUN_PROCESSING = True
    
    base_dir = Path(".")
    in_dir = base_dir / "Data" / "raw data" / "XU"
    out_dir = base_dir / "Data" / "processed data" / "XU_Processed_Hampel"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    file_names = ["XUSLEEP7", "XUSLEEP60", "XUSLEEPT100","XUAWAKE7", "XUAWAKE60", "XUAWAKET100"]

    for name in file_names:
        edf_path = in_dir / f"{name}.EDF"
        # We create the base_filename from the path here
        base_filename = edf_path.stem
        
        if not edf_path.exists():
            print(f"Warning: missing {edf_path}, skipping.")
            continue
        
        if RUN_PROCESSING:
            print(f"\n--- Running full processing for: {base_filename} ---")
            raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
            raw.set_channel_types({
                'EKGL': 'ecg', 'EKGR': 'ecg', 'LOC1': 'eog', 'LOC2': 'eog',
                'EMG1': 'emg', 'EMG2': 'emg'
            }, verbose=False)
            misc_candidates = ['T1','T2','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','DC1','DC2','DC3','DC4','OSAT','PR']
            misc_map = {ch: 'misc' for ch in misc_candidates if ch in raw.ch_names}
            if misc_map:
                raw.set_channel_types(misc_map, verbose=False)
            raw.set_montage("standard_1020", on_missing="warn", verbose=False)
            
            cleaned_raw = preprocessing_full_pipeline(raw, plots_dir, out_dir, base_filename)
            
            out_fif = out_dir / f"{base_filename}_hampel_cleaned-raw.fif"
            cleaned_raw.save(out_fif, overwrite=True)
            print(f"Saved cleaned data to: {out_fif}")

        else:
            cleaned_fif_path = out_dir / f"{base_filename}_hampel_cleaned-raw.fif"
            if not cleaned_fif_path.exists():
                print(f"Warning: missing cleaned file {cleaned_fif_path}, skipping plot.")
                continue
            # Also pass the base_filename to the plotting function
            comparison_plots(edf_path, cleaned_fif_path, plots_dir, base_filename)

    print("\n✅ All processing complete.")