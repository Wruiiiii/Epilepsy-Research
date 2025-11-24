import os
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, cheby2, sosfiltfilt

CONFIG = {
    'run_hampel': True,
    'run_ica_inspect': True,
    'run_ica_plots': True,
    'run_ica_clean': False,
    'run_plots': True,
    'run_hampel_ParaTest':False
}

COMPONENTS_TO_EXCLUDE = {
    'XUSLEEP7': [],
    'XUSLEEP60': [],
    'XUSLEEPT100': [],
    'XUAWAKE7': [],
    'XUAWAKE60': [],
    'XUAWAKET100': [],
    'XUAWAKEPRE_deidentified': [],
    'XUSLEEP_deidentified': []
}
# ===================================================================

def hampel_filter(data, window_size, n_sigmas=6.0):
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

def run_hampel_ParaTest(raw_path, plots_dir, window_size, n_sigmas):
    print(f"--- Running Stage 1: Hampel Filtering for {raw_path.name} ---")
    raw = mne.io.read_raw_edf(raw_path, preload=True, verbose='WARNING')
    raw.set_channel_types({
        'EKGL': 'ecg', 'EKGR': 'ecg', 'LOC1': 'eog', 'LOC2': 'eog',
        'EMG1': 'emg', 'EMG2': 'emg'
    }, verbose=False)
    misc_candidates = ['T1','T2','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','DC1','DC2','DC3','DC4','OSAT','PR']
    misc_map = {ch: 'misc' for ch in misc_candidates if ch in raw.ch_names}
    if misc_map:
        raw.set_channel_types(misc_map, verbose=False)
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)

    raw_hampel = raw.copy()
    raw_to_filter = raw.copy()
    
    raw_to_filter.filter(
        l_freq=1.0, h_freq=None, method='iir',
        iir_params=dict(order=4, ftype='butter', output='sos')
    )
    raw_to_filter.filter(
        l_freq=None, h_freq=100.0, method='iir',
        iir_params=dict(order=8, ftype='cheby2', rs=40, output='sos')
    )
    
    eeg_data = raw_to_filter.get_data(picks='eeg')
    cleaned_eeg_data = np.zeros_like(eeg_data)
    
    for i, channel_data in enumerate(eeg_data):
        spectrum = np.fft.fft(channel_data)
        real_cleaned = hampel_filter(np.real(spectrum), window_size=window_size, n_sigmas=n_sigmas)
        imag_cleaned = hampel_filter(np.imag(spectrum), window_size=window_size, n_sigmas=n_sigmas)
        cleaned_eeg_data[i, :] = np.real(np.fft.ifft(real_cleaned + 1j * imag_cleaned))
    
    raw_hampel._data[mne.pick_types(raw.info, eeg=True), :] = cleaned_eeg_data

    # Create and save a comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    raw.compute_psd(fmax=70).plot(axes=ax1, show=False, color='black')
    ax1.set_title('Original PSD'); ax1.set_ylim(-70, 70)
    raw_hampel.compute_psd(fmax=70).plot(axes=ax2, show=False, color='blue')
    ax2.set_title(f'Filtered PSD (N={window_size}, t={n_sigmas})'); ax2.set_ylim(-70, 70)

    plot_filename = f"{raw_path.stem}_hampel_test_N{window_size}_t{int(n_sigmas)}.png"
    fig.savefig(plots_dir / plot_filename)
    plt.close(fig)
    print(f"-> Saved test plot: {plot_filename}")


def run_hampel_filter(raw_path, out_path):
    print(f"--- Running Stage 1: Hampel Filtering for {raw_path.name} ---")
    raw = mne.io.read_raw_edf(raw_path, preload=True, verbose='WARNING')
    raw.set_channel_types({
        'EKGL': 'ecg', 'EKGR': 'ecg', 'LOC1': 'eog', 'LOC2': 'eog',
        'EMG1': 'emg', 'EMG2': 'emg'
    }, verbose=False)
    misc_candidates = ['T1','T2','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','DC1','DC2','DC3','DC4','OSAT','PR']
    misc_map = {ch: 'misc' for ch in misc_candidates if ch in raw.ch_names}
    if misc_map:
        raw.set_channel_types(misc_map, verbose=False)
    raw.set_montage("standard_1020", on_missing="warn", verbose=False)

    raw_hampel = raw.copy()
    raw_to_filter = raw.copy()
    
    raw_to_filter.filter(
        l_freq=1.0, h_freq=None, method='iir',
        iir_params=dict(order=4, ftype='butter', output='sos')
    )
    raw_to_filter.filter(
        l_freq=None, h_freq=100.0, method='iir',
        iir_params=dict(order=8, ftype='cheby2', rs=40, output='sos')
    )
    
    eeg_data = raw_to_filter.get_data(picks='eeg')
    cleaned_eeg_data = np.zeros_like(eeg_data)
    
    for i, channel_data in enumerate(eeg_data):
        spectrum = np.fft.fft(channel_data)
        real_cleaned = hampel_filter(np.real(spectrum), window_size=80, n_sigmas=6.0)
        imag_cleaned = hampel_filter(np.imag(spectrum), window_size=80, n_sigmas=6.0)
        cleaned_signal = np.real(np.fft.ifft(real_cleaned + 1j * imag_cleaned))
        cleaned_eeg_data[i, :] = cleaned_signal
        
    raw_hampel._data[mne.pick_types(raw.info, eeg=True), :] = cleaned_eeg_data
    
    raw_hampel.save(out_path, overwrite=True)
    print(f"-> Saved Hampel-filtered data to: {out_path}")

    


def run_ica_inspect(hampel_path, plots_dir):
    print(f"--- Running Stage 2: ICA Inspection for {hampel_path.name} ---")
    base_filename = hampel_path.name.replace("-stg1-hampel.fif", "")
    raw_hampel = mne.io.read_raw_fif(hampel_path, preload=True)
    
    raw_for_ica = raw_hampel.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)

    ica.fit(raw_for_ica, picks='eeg')
    
    #plot a selected channel_EOG
    ica.plot_components(show=False).savefig(plots_dir / f"{base_filename}_ICA_components_topo.png")
    eog_epochs = mne.preprocessing.create_eog_epochs(raw_hampel, ch_name=['LOC1', 'LOC2'], verbose=False)
    if len(eog_epochs) > 0:
        ica.plot_sources(eog_epochs, show=False).savefig(plots_dir / f"{base_filename}_ICA_EOG_sources.png")
    plt.close('all')

    #plot a selected channel_ECG
    ecg_epochs = mne.preprocessing.create_ecg_epochs(raw_hampel, ch_name='EKGL', verbose=False)
    if len(ecg_epochs) > 0:
        ica.plot_sources(ecg_epochs, show=False).savefig(plots_dir / f"{base_filename}_ICA_ECG_sources.png")
    plt.close('all')
    print("-> Inspection plots generated. Please review and update COMPONENTS_TO_EXCLUDE.")


def run_ica_clean(hampel_path, out_path, components_to_exclude):
    """Loads Hampel-filtered data, applies ICA cleaning, and saves final result."""
    print(f"--- Running Stage 3: ICA Cleaning for {hampel_path.name} ---")
    raw_hampel = mne.io.read_raw_fif(hampel_path, preload=True)
    
    raw_for_ica = raw_hampel.copy().filter(l_freq=1.0, h_freq=None)
    ica = mne.preprocessing.ICA(n_components=15, max_iter='auto', random_state=97)
    ica.fit(raw_for_ica, picks='eeg')
    
    raw_final_cleaned = raw_hampel.copy()
    ica.exclude = components_to_exclude
    ica.apply(raw_final_cleaned)
    
    raw_final_cleaned.save(out_path, overwrite=True)
    print(f"-> Saved final cleaned data to: {out_path}")


def run_plots(raw_path, hampel_path, plots_dir):
    print(f"--- Running Hampel Comparison Plots for {raw_path.name} ---")
    base_filename = raw_path.stem
    
    raw_before = mne.io.read_raw_edf(raw_path, preload=True, verbose='WARNING')
    raw_after = mne.io.read_raw_fif(hampel_path, preload=True, verbose='WARNING')
    
    # PSD Plots
    fig_psd, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    raw_before.compute_psd(fmax=70).plot(axes=ax1, show=False, color='black')
    ax1.set_title('PSD Before Hampel Filter'); ax1.set_ylim(-70, 70)
    raw_after.compute_psd(fmax=70).plot(axes=ax2, show=False, color='red')
    ax2.set_title('PSD After Hampel Filter'); ax2.set_ylim(-70, 70)
    fig_psd.suptitle(f"Hampel Filter PSD Comparison for {base_filename}")
    fig_psd.tight_layout(); fig_psd.savefig(plots_dir / f"{base_filename}_hampel_psd_comparison.png")
    plt.close(fig_psd)

    # Time-Domain Butterfly Plots
    scaling = 200e-6
    fig_before_td = raw_before.plot(duration=5, scalings=scaling, show=False)
    fig_before_td.suptitle(f"Time-Domain Before - {base_filename}", y=1.0)
    fig_before_td.savefig(plots_dir / f"{base_filename}_hampel_td_before.png")
    fig_after_td = raw_after.plot(duration=5, scalings=scaling, show=False)
    fig_after_td.suptitle(f"Time-Domain After - {base_filename}", y=1.0)
    fig_after_td.savefig(plots_dir / f"{base_filename}_hampel_td_after.png")
    plt.close('all')

    print(f"-> Comparison plots saved.")

def run_ica_plots(data_path, plots_dir):
    print(f"--- Running ICA Plots for {data_path.name} ---")
    base_filename = data_path.stem.replace("-stg1-hampel", "")
    raw = mne.io.read_raw_fif(data_path, preload=True, verbose='WARNING')
    # raw.set_channel_types({
    #     'EKGL': 'ecg', 'EKGR': 'ecg', 'LOC1': 'eog', 'LOC2': 'eog',
    #     'EMG1': 'emg', 'EMG2': 'emg'
    # }, verbose=False)
    # misc_candidates = ['T1','T2','X9','X10','X11','X12','X13','X14','X15','X16','X17','X18','DC1','DC2','DC3','DC4','OSAT','PR']
    # misc_map = {ch: 'misc' for ch in misc_candidates if ch in raw.ch_names}

    # EOG Raw Artifact Visualization
    eog_epochs_raw = mne.preprocessing.create_eog_epochs(raw, ch_name=['LOC1', 'LOC2'], baseline=(-0.5, -0.2), verbose=False)
    if len(eog_epochs_raw) > 0:
        avg_eog_evoked = eog_epochs_raw.average()
        fig_joint_eog = avg_eog_evoked.plot_joint(title="Average EOG Artifact", show=False)
        fig_joint_eog.savefig(plots_dir / f"{base_filename}_EOG_raw_joint_plot.png")
        plt.close(fig_joint_eog)
    else:
        print("-> No raw EOG events found for diagnostic plot.")

    # ECG Raw Artifact Visualization
    ecg_epochs_raw = mne.preprocessing.create_ecg_epochs(raw, ch_name='EKGL', baseline=(-0.5, -0.2), verbose=False)
    if len(ecg_epochs_raw) > 0:
        avg_ecg_evoked = ecg_epochs_raw.average()
        fig_joint_ecg = avg_ecg_evoked.plot_joint(title="Average ECG Artifact", show=False)
        fig_joint_ecg.savefig(plots_dir / f"{base_filename}_ECG_raw_joint_plot.png")
        plt.close(fig_joint_ecg)
    else:
        print("-> No raw ECG events found for diagnostic plot.")
    print("-> Diagnostic plots saved.")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    base_dir = Path(".")
    in_dir = base_dir / "Data" / "raw data" / "XU"
    out_dir = base_dir / "Data" / "processed_data" / "XU_Preprocessing_Full"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # file_names = ["XUSLEEP7", "XUSLEEP60", "XUSLEEPT100", "XUAWAKE7", "XUAWAKE60", "XUAWAKET100", "XUAWAKEPRE_deidentified", "XUSLEEPP_deidentified"]
    file_names = ["XUSLEEP7", "XUAWAKE7", "XUAWAKEPRE_deidentified", "XUSLEEP_deidentified"]


    for name in file_names:
        raw_path = in_dir / f"{name}.EDF"
        hampel_path = out_dir / f"{name}-stg1-hampel.fif"
        final_path = out_dir / f"{name}-stg2-final.fif"
        
        if not raw_path.exists():
            print(f"Warning: missing raw file {raw_path}, skipping.")
            continue
        
        if CONFIG['run_hampel']:
            run_hampel_filter(raw_path, hampel_path)

        if CONFIG['run_hampel_ParaTest']:
            # Define parameter sets to test here
            params_to_test = [
                {'window_size': 80, 'n_sigmas': 6.0}, # The original from the paper
                {'window_size': 80, 'n_sigmas': 8.0}, # Less aggressive
                {'window_size': 80, 'n_sigmas': 10.0} # Even less aggressive
            ]
            for params in params_to_test:
                run_hampel_ParaTest(raw_path, plots_dir, **params)
            
        if CONFIG['run_ica_inspect']:
            if not hampel_path.exists():
                print(f"Warning: missing Hampel file {hampel_path}, skipping Stage 2.")
                continue
            run_ica_inspect(hampel_path, plots_dir)
            
        if CONFIG['run_ica_plots']:
            run_ica_plots(raw_path, plots_dir)
            
        if CONFIG['run_ica_clean']:
            if not hampel_path.exists():
                print(f"Warning: missing Hampel file {hampel_path}, skipping Stage 3.")
                continue
            components = COMPONENTS_TO_EXCLUDE.get(name)
            if components is not None:
                run_ica_clean(hampel_path, final_path, components)
            else:
                print(f"Warning: No component list for {name} in COMPONENTS_TO_EXCLUDE. Skipping Stage 3.")

        if CONFIG['run_plots']:
            if not hampel_path.exists():
                print(f"Warning: missing Hampel file {hampel_path}, skipping plots.")
                continue
            run_plots(raw_path, hampel_path, plots_dir)

    print("\n Pipeline complete.")