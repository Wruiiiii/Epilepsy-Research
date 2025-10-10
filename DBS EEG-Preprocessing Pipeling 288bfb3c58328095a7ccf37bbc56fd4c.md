# DBS EEG-Preprocessing Pipeling

This repository contains a Python-based pipeline for preprocessing scalp EEG data, particularly for subjects with Deep Brain Stimulation (DBS). The pipeline is designed to clean raw `.EDF` files by removing noise and artifacts, preparing the data for subsequent analysis.

## Directory Structure

```python

├── Preprocessing_Full.py
└── Data/
    ├── raw data/
    │   └── XU/
    │       ├── XUSLEEP7.EDF
    │       └── ... (other raw .EDF files)
    └── processed_data/
        └── XU_Preprocessing_Full/
            ├── plots/
            └── ... (output files will be saved here)
```

## Pipeline Workflow

The entire pipeline is controlled by the **`CONFIG`** dictionary at the top of the script. You can run the pipeline in stages by setting the boolean flags (`True`/`False`).

### **Step 1: Configuration**

First, configure the main script to control which steps are executed.

- **`CONFIG` Dictionary**: This is your main control panel.
    - **`run_hampel`**: Set to `True` to perform the initial Hampel filtering.
    - **`run_ica_inspect`**: Set to `True` to generate ICA component plots for manual inspection. **This is a crucial first step for ICA cleaning.**
    - **`run_ica_clean`**: Set to `True` **after** you have identified the bad components and updated the `COMPONENTS_TO_EXCLUDE` dictionary.
    - **`run_plots`**: Set to `True` to generate comparison plots (PSD and time-domain) showing the effect of the Hampel filter.
- **`COMPONENTS_TO_EXCLUDE` Dictionary**: This is where you will manually input the ICA components you want to remove for each file.

### **Step 2: Hampel Filtering**

This step applies a Hampel filter to remove sharp artifacts from the frequency spectrum of the EEG signals.

1. **What it does**: The `run_hampel_filter` function loads the raw data, applies a band-pass filter, transforms the signal to the frequency domain using FFT, removes outliers from the spectrum using the Hampel filter, and transforms it back to the time domain.
2. **How to run**: In the `CONFIG` dictionary, set **`run_hampel = True`**.
3. **Output**: A new file named `[filename]-stg1-hampel.fif` is saved in the `processed_data/XU_Preprocessing_Full/` folder. This file is the input for the next stage.
4. **Parameter Testing**: Your script also includes a `run_hampel_ParaTest` function. To test different Hampel filter parameters (`window_size`, `n_sigmas`), set **`run_hampel_ParaTest = True`**. This will generate PSD plots for each parameter set without saving `.fif` files, helping you choose the best settings.

### **Step 3: ICA Inspection (Stage 2a - Manual Inspection)**

This is the most important **manual step** in the pipeline. The goal is to identify which Independent Components (ICs) correspond to artifacts like eye blinks or heartbeats.

1. **What it does**: The `run_ica_inspect` function takes the Hampel-filtered data, fits an ICA model, and generates several diagnostic plots.
2. **How to run**:
    - Make sure you have a `[filename]-stg1-hampel.fif` file from the previous step.
    - In the `CONFIG` dictionary, set **`run_ica_inspect = True`**.
3. **Output**: Several plots will be saved in the `plots/` directory for each file:
    - `[filename]_ICA_components_topo.png`: Shows the scalp topography of each component. Look for **frontal patterns** typical of eye blinks.
    - `[filename]_ICA_EOG_sources.png`: Shows the time course of components overlaid with EOG events (blinks). Components that show **strong, repeated patterns time-locked to the blinks** are EOG artifacts.
    - `[filename]_ICA_ECG_sources.png`: Shows the time course of components overlaid with ECG events (heartbeats). Components showing a **rhythmic pattern matching the QRS complex** are ECG artifacts.

### **Step 4: Update Components & Run ICA Cleaning (Stage 2b)**

After inspecting the plots from Step 3, you need to tell the script which components to remove.

1. **Action**: Open the script and go to the **`COMPONENTS_TO_EXCLUDE`** dictionary. For each file, add the numbers of the artifactual components to the list. For example, if you identified components 0 and 3 as artifacts for `XUSLEEP7`:Python
    
    `COMPONENTS_TO_EXCLUDE = {
        'XUSLEEP7': [0, 3], 
        'XUSLEEP60': [], 
        # ... and so on for other files
    }`
    
2. **How to run**:
    - Update the **`COMPONENTS_TO_EXCLUDE`** dictionary with your findings.
    - In the `CONFIG` dictionary, set **`run_ica_clean = True`**.
    - It's good practice to set `run_ica_inspect = False` now to avoid re-generating the plots.
3. **Output**: The final, cleaned data is saved as `[filename]-stg2-final.fif` in the `processed_data/XU_Preprocessing_Full/` folder.