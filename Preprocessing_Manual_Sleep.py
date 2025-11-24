
import os
from pathlib import Path
import mne
from mne.preprocessing import ICA
import matplotlib.pyplot as plt
import numpy as np


# ====== USER OPTIONS ======

COMPONENTS_TO_EXCLUDE = {
    'XUSLEEP7': [0, 3, 5, 10, 13],
    'XUSLEEP60': [8, 11],
    'XUSLEEPT100': [8]
}

FREQ_BANDS = {             
    "Delta (0.5-4)": (0.5, 4),
    "Theta (4-7)": (4, 7),
    "Alpha (8-12)": (8, 12),
    "Beta (13-30)": (13, 30),
}

BAD_CHANNELS = {
    'XUSLEEP7': ['EKGL', 'EKGR'],
    'XUSLEEP60': ['EKGL', 'EKGR'],
    'XUSLEEPT100': ['EKGL', 'EKGR']
}
# ==========================

def preprocess_eeg_file(edf_path: Path, plots_dir: Path):
    base_filename = edf_path.stem
    print(f"\n--- Processing: {base_filename} ---")

    # 1. Load, set channel types, montage, and filter
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    # ... (channel type and montage setup code as before) ...
    raw.set_channel_types({
        'EKGL': 'ecg', 'EKGR': 'ecg', 'LOC1': 'eog', 'LOC2': 'eog',
        'EMG1': 'emg', 'EMG2': 'emg'
    }, verbose=False)
    misc_candidates = ['T1','T2','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','DC1','DC2','DC3','DC4','OSAT','PR']
    misc_map = {ch: 'misc' for ch in misc_candidates if ch in raw.ch_names}
    if misc_map:
        raw.set_channel_types(misc_map, verbose=False)
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)


    if base_filename == 'XUSLEEP7':
        raw_before_notch = raw.copy()
        psd = raw.compute_psd(method='welch', n_fft=4096, fmax=80, verbose=False)
        data, freqs = psd.get_data(return_freqs=True)
        
        freq_range = (freqs >= 10) & (freqs <= 80)
        avg_power = np.mean(data, axis=0)
        
        if np.any(freq_range):
            peak_index_in_slice = np.argmax(avg_power[freq_range])
            peak_freq = freqs[freq_range][peak_index_in_slice]
            notch_freqs = np.arange(peak_freq, raw.info['sfreq']/2, peak_freq)
            print(f"Automatically detected noise around {peak_freq:.2f} Hz. Applying a robust notch filter.")
            raw.notch_filter(
                freqs=notch_freqs,
                notch_widths=3,  # Filter a 3 Hz wide band around each frequency
                verbose=False
            )
        else:
            print("No significant power-line noise found in the 10-80 Hz range.")

        # --- Plot "Before" ---
        psd_before = raw_before_notch.compute_psd(fmax=65, verbose=False)
        fig_before, ax_before = plt.subplots(figsize=(12, 6))
        psd_before.plot(axes=ax_before, color='black', show=False)
        ax_before.set_title(f'PSD Before Notch Filter for {base_filename}')
        plot_path_before = plots_dir / f"{base_filename}_psd_before_notch.png"
        fig_before.savefig(plot_path_before, dpi=150)
        plt.close(fig_before)
        print(f"Saved 'Before' plot to: {plot_path_before}")
        
        # --- Plot "After" ---
        psd_after = raw.compute_psd(fmax=65, verbose=False)
        fig_after, ax_after = plt.subplots(figsize=(12, 6))
        psd_after.plot(axes=ax_after, color='blue', show=False)
        ax_after.set_title(f'PSD After Notch Filter for {base_filename}')
        plot_path_after = plots_dir / f"{base_filename}_psd_after_notch.png"
        fig_after.savefig(plot_path_after, dpi=150)
        plt.close(fig_after)
        print(f"Saved 'After' plot to: {plot_path_after}")

    # Band-pass filter
    raw.filter(1., 40., phase="zero-double", verbose=False)




    # 2. ✅ Select ONLY EEG channels for ICA
    raw_eeg_only = raw.copy().pick('eeg')

    # 3. Fit ICA on the EEG-only data
    print("Fitting ICA on EEG channels only...")
    #ica = ICA(n_components=0.99, random_state=97, max_iter="auto")
    ica = ICA(n_components=22, random_state=97,max_iter=800)

    ica.fit(raw_eeg_only)
    #--- plot and choose bad components
    # 4. ✅ Plot all component topographies for manual inspection
    print("Generating component topography plot for inspection...")
    fig = ica.plot_components(inst=raw_eeg_only, show=False)
   # The function can return a list of figures, so we loop through it
    for i, fig in enumerate(fig):
        plot_path = plots_dir / f"{base_filename}_ICA_components_part{i+1}.png"
        fig.savefig(plot_path, dpi=150)
        print(f"Saved component plots to: {plot_path}")
    # plt.close('all') # Close all open figures

    # plot ica properties for a few specific components
    components_to_investigate = range(22)
    print(f"Generating detailed property plots for components: {components_to_investigate}...")
    
    # Loop through the list and save a plot for each
    for comp_idx in components_to_investigate:
        fig_prop = ica.plot_properties(raw, picks=comp_idx, show=False)
        prop_plot_path = plots_dir / f"{base_filename}_ICA{comp_idx:03d}_properties.png"
        fig_prop[0].savefig(prop_plot_path, dpi=150) # plot_properties returns a list of figures
        plt.close(fig_prop[0])
    print("Finished generating property plots.")
    # ---
    # Manually apply the exclusion list
    if base_filename in COMPONENTS_TO_EXCLUDE:
        ica.exclude = COMPONENTS_TO_EXCLUDE[base_filename]
        print(f"Manually excluding components for {base_filename}: {ica.exclude}")
    else:
        ica.exclude = []

     # 5. Create "Before" and "After" datasets
    # "Before" is re-referenced but NOT ICA cleaned
    raw_before_ica = raw.copy().set_eeg_reference("average", projection=False, verbose=False)

    # "After" is the re-referenced data WITH ICA cleaning applied
    raw_after_ica = ica.apply(raw_before_ica.copy())
    
    # 6. Generate and save the final PSD topography comparison plot
    print("Generating and saving final comparison topography plot...")
    psd_before = raw_before_ica.compute_psd(verbose=False)
    psd_after = raw_after_ica.compute_psd(verbose=False)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for i, (band_name, (fmin, fmax)) in enumerate(FREQ_BANDS.items()):
        psd_before.plot_topomap(bands={band_name: (fmin, fmax)}, ch_type="eeg", axes=axes[0, i], show=False)
        axes[0, i].set_title(band_name)
        psd_after.plot_topomap(bands={band_name: (fmin, fmax)}, ch_type="eeg", axes=axes[1, i], show=False)
        axes[1, i].set_title("")

    axes[0, 0].set_ylabel("Before ICA", fontsize=14, rotation=90, labelpad=10)
    axes[1, 0].set_ylabel("After ICA", fontsize=14, rotation=90, labelpad=10)
    
    fig.suptitle(f"Power Topomap Comparison for {base_filename}", fontsize=20)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    plot_path = plots_dir / f"{base_filename}_topo_comparison.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved combined comparison plot to: {plot_path}")

    # 7. Generate and save separate PSD curve plots with new frequency range
    print("Generating and saving separate PSD curve plots...")
    
    # Compute PSDs with a 70 Hz maximum frequency
    psd_before_curve = raw_before_ica.compute_psd(fmax=70, verbose=False)
    psd_after_curve = raw_after_ica.compute_psd(fmax=70, verbose=False)

    # --- Plot "Before" ---
    fig_before, ax_before = plt.subplots(figsize=(10, 6))
    psd_before_curve.plot(axes=ax_before, show=False, color='black')
    ax_before.set_title(f"PSD Before ICA for {base_filename}")
    plot_path_before = plots_dir / f"{base_filename}_psd_before_ICA.png"
    fig_before.savefig(plot_path_before, dpi=150)
    plt.close(fig_before)
    print(f"Saved 'Before' PSD plot to: {plot_path_before}")

    # --- Plot "After" ---
    fig_after, ax_after = plt.subplots(figsize=(10, 6))
    psd_after_curve.plot(axes=ax_after, show=False, color='red')
    ax_after.set_title(f"PSD After ICA for {base_filename}")
    plot_path_after = plots_dir / f"{base_filename}_psd_after_ICA.png"
    fig_after.savefig(plot_path_after, dpi=150)
    plt.close(fig_after)
    print(f"Saved 'After' PSD plot to: {plot_path_after}")

    raw_after_ica.plot(duration=20, n_channels=30, scalings='auto')


    # Plot Cleaned Data for Bad Channel Inspection
    print("Generating plot for manual bad channel inspection...")
    fig = raw_after_ica.plot(duration=10, n_channels=30, scalings='auto', show=False)
    plot_path = plots_dir / f"{base_filename}_for_bad_channel_inspection.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved inspection plot to: {plot_path}")

    # Visualize Before and After Marking Bad Channels
    print("Generating before/after plots for bad channel removal...")
    # --- Plot "Before" ---
    fig_before = raw_after_ica.plot(duration=20, n_channels=30, scalings='auto', show=False)
    fig_before.suptitle('Before Marking Bad Channels', y=0.98)
    plot_path_before = plots_dir / f"{base_filename}_bad_channels_before.png"
    fig_before.savefig(plot_path_before, dpi=150)
    plt.close(fig_before)

    # --- Mark Bad Channels ---
    if base_filename in BAD_CHANNELS and BAD_CHANNELS[base_filename]:
        raw_after_ica.info['bads'] = BAD_CHANNELS[base_filename]
        print(f"Marking bad channels: {raw_after_ica.info['bads']}")
    
    # --- Plot "After" ---
    fig_after = raw_after_ica.plot(duration=20, n_channels=30, scalings='auto', show=False)
    fig_after.suptitle('After Marking Bad Channels', y=0.98)
    plot_path_after = plots_dir / f"{base_filename}_bad_channels_after.png"
    fig_after.savefig(plot_path_after, dpi=150)
    plt.close(fig_after)
    print("Saved before/after bad channel plots.")
    
    # 5. Rejecting Epochs based on Amplitude
    print("Segmenting data into epochs and rejecting bad epochs...")
    epochs = mne.make_fixed_length_epochs(raw_after_ica, duration=2, overlap=1, preload=True)
    reject_criteria = dict(eeg=150e-6) # 150 µV
    epochs.drop_bad(reject=reject_criteria)
    print(f"Epochs remaining after rejection: {len(epochs)}")
    
    # 6. ✅ Generate and save the final PSD topography plot
    print("Generating final PSD topography plot...")
    fig = epochs.compute_psd().plot_topomap(
        bands=FREQ_BANDS,
        ch_type='eeg',
        show=False
    )
    fig.suptitle(f'Final PSD Topographies for {base_filename}', fontsize=16)
    output_plot_path = plots_dir / f"{base_filename}_final_psd_topo.png"
    fig.savefig(output_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved final PSD topography plot to: {output_plot_path}")


    return epochs


# Main execution block
if __name__ == "__main__":
    base_dir = Path(".")
    in_dir = base_dir / "Data" / "raw data" / "XU"
    out_dir = base_dir / "Data" / "processed data" / "XU_Processed_Manual_Sleep"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    file_names = [ "XUSLEEP7", "XUSLEEP60", "XUSLEEPT100"]
    # file_names = ["XUSLEEP7"]

    for name in file_names:
        edf = in_dir / f"{name}.EDF"
        if not edf.exists():
            print(f"Warning: missing {edf}, skipping.")
            continue
        cleaned = preprocess_eeg_file(edf, plots_dir)
        out_fif = out_dir / f"{name}_preprocessed-raw.fif"
        cleaned.save(out_fif, overwrite=True)
        print(f"Saved: {out_fif}")

        cleaned_epochs = preprocess_eeg_file(edf, plots_dir)
        out_fif = out_dir / f"{name}_preprocessed-epo.fif"
        cleaned_epochs.save(out_fif, overwrite=True)

