# EEG Preprocessing and Validation Scripts

This directory contains scripts for preprocessing EEG data with DBS (Deep Brain Stimulation) artifacts and validating the Hampel filter performance.

## Overview

The preprocessing pipeline consists of three main components:

1. **Preprocessing_Full.py** - Main preprocessing pipeline that applies Hampel filtering and ICA cleaning
2. **HampelValidation_Manual_new.py** - Manual validation method using known DBS stimulation frequencies
3. **HalpelValidation_Auto.py** - Automatic validation method that detects artifact frequencies automatically

**Note**: The scripts `Preprocessing_Manual_Sleep.py` and `Preprocessing_Manual_Awake.py` are exploration scripts used to develop and test the preprocessing process; they are not part of the main production pipeline.

---

## 1. Preprocessing_Full.py

### Description
Main preprocessing pipeline for EEG data with DBS artifacts. This script performs:
- **Stage 1**: Hampel filtering in the frequency domain to remove DBS artifacts
- **Stage 2**: ICA inspection to identify artifact components
- **Stage 3**: ICA cleaning to remove identified components
- **Visualization**: Generates comparison plots (PSD, time-domain, ICA components)

### Configuration

Before running, configure the script by editing the `CONFIG` dictionary at the top:

```python
CONFIG = {
    'run_hampel': True,          # Apply Hampel filter
    'run_ica_inspect': True,      # Generate ICA inspection plots
    'run_ica_plots': True,        # Generate diagnostic ICA plots
    'run_ica_clean': False,       # Apply ICA cleaning (requires component list)
    'run_plots': True,            # Generate comparison plots
    'run_hampel_ParaTest': False  # Test different Hampel parameters
}
```

### Component Exclusion

If using ICA cleaning, specify which components to exclude for each file in `COMPONENTS_TO_EXCLUDE`:

```python
COMPONENTS_TO_EXCLUDE = {
    'XUSLEEP7': [0, 1, 2],  # Example: exclude components 0, 1, 2
    'XUAWAKE7': [1, 3],
    # ... add entries for each file
}
```

### File Selection

Modify the `file_names` list in the main section to process specific files:

```python
file_names = ["XUSLEEP7", "XUAWAKE7", "XUAWAKEPRE_deidentified", "XUSLEEP_deidentified"]
```

### Usage

```bash
cd Script
python Preprocessing_Full.py
```

### Input/Output

- **Input**: EDF files from `Data/raw data/XU/`
- **Output**: 
  - Processed FIF files in `Data/processed_data/XU_Preprocessing_Full/`
  - Plots in `Data/processed_data/XU_Preprocessing_Full/plots/`

### Processing Stages

1. **Hampel Filtering** (`run_hampel=True`):
   - Applies bandpass (1-100 Hz) and Chebyshev filters
   - Performs Hampel filtering in frequency domain (window_size=80, n_sigmas=6.0)
   - Saves output as `{filename}-stg1-hampel.fif`

2. **ICA Inspection** (`run_ica_inspect=True`):
   - Fits ICA model (20 components)
   - Generates component topography plots
   - Generates EOG/ECG source plots for artifact identification
   - **Action Required**: Review plots and update `COMPONENTS_TO_EXCLUDE`

3. **ICA Cleaning** (`run_ica_clean=True`):
   - Applies ICA with specified components excluded
   - Saves final cleaned data as `{filename}-stg2-final.fif`

4. **Parameter Testing** (`run_hampel_ParaTest=True`):
   - Tests different Hampel filter parameters
   - Generates comparison plots for parameter selection

### Dependencies

- `mne` - MNE-Python for EEG processing
- `numpy` - Numerical computations
- `matplotlib` - Plotting
- `scipy` - Signal processing

---

## 2. HampelValidation_Manual_new.py

### Description
Validates the Hampel filter using a manual method that extracts DBS artifacts from DBS-ON data based on known stimulation frequencies. This method:
- Extracts artifacts at the DBS fundamental frequency and its harmonics
- Creates contaminated signals by adding artifacts to clean (DBS-OFF) data
- Applies Hampel filter and evaluates performance using PSD comparison

### Configuration

Edit the configuration section at the top of the script:

```python
file_on = "XUAWAKE7.EDF"              # DBS-ON file (with artifacts)
file_off = "XUAWAKEPRE_deidentified.EDF"  # DBS-OFF file (clean reference)
channel_to_process = 'Fp2'             # Channel to validate
dbs_stim_freq = 130.0                  # DBS stimulation frequency (Hz)
num_harmonics = 5                      # Number of harmonics to extract
```

### Usage

```bash
cd Script
python HampelValidation_Manual_new.py
```

### How It Works

1. **Loads Data**: Reads both DBS-ON and DBS-OFF EDF files
2. **Preprocessing**: Applies bandpass (0.1-70 Hz) and notch (60 Hz) filters
3. **Artifact Extraction**: 
   - Extracts frequency components at DBS frequency and harmonics (e.g., 130 Hz, 260 Hz, 390 Hz, ...)
   - Converts back to time domain to get realistic artifact signal
4. **Contamination**: Adds extracted artifact to clean DBS-OFF signal
5. **Filtering**: Applies Hampel filter to contaminated signal
6. **Validation**: Compares PSDs of clean, contaminated, and filtered signals

### Output

- **Plot**: PSD comparison plot saved to `Data/validation_sun_method_plots/psd_validation_plot_{channel}.png`
- Shows three curves:
  - Blue: Clean EEG (ground truth)
  - Green: Contaminated signal (clean + artifact)
  - Red: Recovered signal (after Hampel filter)

### Interpretation

- **Good performance**: Red curve should closely match blue curve
- **Poor performance**: Red curve may still show peaks at artifact frequencies
- The plot helps assess whether the Hampel filter effectively removes DBS artifacts

### Dependencies

- `mne` - EEG data loading
- `numpy` - FFT and signal processing
- `matplotlib` - Plotting
- `scipy.signal` - Welch's method for PSD calculation

---

## 3. HalpelValidation_Auto.py

### Description
Automatic validation script that detects artifact frequencies automatically from DBS-ON data and validates the Hampel filter per channel. This method:
- Automatically detects artifact peaks in the frequency spectrum
- Synthesizes artifacts based on detected frequencies
- Validates filter performance for each EEG channel individually

### Configuration

Edit the file paths in the main section:

```python
file_off = "XUAWAKEPRE_deidentified.EDF"  # DBS-OFF file (clean reference)
file_on = "XUAWAKE7.EDF"                  # DBS-ON file (with artifacts)
```

### Usage

```bash
cd Script
python HalpelValidation_Auto.py
```

### How It Works

1. **Loads Data**: Reads both DBS-ON and DBS-OFF files
2. **Peak Detection**: For each channel:
   - Computes FFT of DBS-ON data
   - Finds frequency peaks above threshold (50x mean power)
   - Identifies artifact frequencies automatically
3. **Artifact Synthesis**: 
   - Creates synthetic artifacts at detected frequencies
   - Adds random phases for realism
4. **Filtering**: Applies Hampel filter to contaminated signal
5. **Validation**: Calculates metrics (r², MSE) for each channel
6. **Visualization**: Generates validation plots per channel

### Output

- **Console**: Validation metrics (r², MSE) printed for each channel
- **Plots**: Individual validation plots saved to `Data/validation_plots/{filename}_validation_plot_{channel}.png`
- Each plot shows:
  - Blue: Original clean signal (ground truth)
  - Gray: Contaminated signal
  - Red: Filtered signal

### Metrics

- **r² (R-squared)**: Correlation coefficient squared (higher is better, max = 1.0)
- **MSE (Mean Squared Error)**: Average squared difference (lower is better)

### Advantages

- No need to know DBS stimulation frequency in advance
- Validates each channel individually
- Provides quantitative metrics for filter performance

### Dependencies

- `mne` - EEG data loading
- `numpy` - FFT and signal processing
- `matplotlib` - Plotting
- `scipy.signal.find_peaks` - Peak detection

---

## Workflow Recommendation

### Typical Workflow

1. **Start with Validation**:
   - Run `HampelValidation_Manual_new.py` or `HalpelValidation_Auto.py` to assess filter performance
   - Review validation plots to ensure Hampel filter is working correctly

2. **Run Preprocessing**:
   - Configure `Preprocessing_Full.py` with `run_ica_clean=False` initially
   - Run with `run_hampel=True` and `run_ica_inspect=True`
   - Review ICA plots to identify artifact components

3. **Update Component List**:
   - Based on ICA inspection plots, update `COMPONENTS_TO_EXCLUDE` dictionary
   - Set `run_ica_clean=True` and re-run preprocessing

4. **Final Validation**:
   - Optionally re-run validation scripts on processed data to confirm improvement

### File Naming Conventions

- **Raw data**: `{filename}.EDF` in `Data/raw data/XU/`
- **Hampel filtered**: `{filename}-stg1-hampel.fif`
- **ICA cleaned**: `{filename}-stg2-final.fif`
- **Plots**: Various naming conventions in respective plot directories

---

## Common Issues and Troubleshooting

### Issue: Missing input files
**Solution**: Ensure EDF files are in `Data/raw data/XU/` directory

### Issue: ICA components not identified
**Solution**: 
- Review ICA component topography plots
- Check EOG/ECG source plots for correlation with known artifacts
- Manually inspect component time series if needed

### Issue: Poor validation results
**Solution**:
- Check that DBS-ON and DBS-OFF files are correctly specified
- Verify DBS stimulation frequency matches actual frequency
- Consider adjusting Hampel filter parameters (window_size, n_sigmas)

### Issue: Memory errors
**Solution**:
- Process files one at a time
- Reduce data length using `raw.crop()` if needed
- Ensure sufficient RAM available

---

## Additional Scripts

### Preprocessing_HampelFilter.py

**Status**: **Deprecated/Older Version** - Consider removing this file.

This script is an earlier version of the preprocessing pipeline that has been superseded by `Preprocessing_Full.py`. 

**Comparison with Preprocessing_Full.py**:

| Feature | Preprocessing_HampelFilter.py | Preprocessing_Full.py |
|---------|------------------------------|---------------------|
| Structure | Single monolithic function | Modular functions per stage |
| Configuration | RUN_MODE switch (INSPECT/CLEAN) | CONFIG dictionary with multiple flags |
| Parameter Testing | Not available | Built-in parameter testing |
| Output Directory | `XU_Processed_Hampel` | `XU_Preprocessing_Full` |
| Comparison Plots | Separate before/after plots + single channel overlay | Side-by-side comparison plots |
| ICA Components | 15 components | 20 components (inspect), 15 (clean) |
| Code Organization | Less modular | More modular and maintainable |

**Recommendation**: 
- **Delete** `Preprocessing_HampelFilter.py` if you're using `Preprocessing_Full.py` as your main pipeline
- The only unique feature in the old script is the single-channel overlay plot, which can be easily added to `Preprocessing_Full.py` if needed
- `Preprocessing_Full.py` is more comprehensive, better organized, and includes additional features like parameter testing

---

## Notes

- All scripts assume data is in EDF format
- Channel types are automatically set (ECG, EOG, EMG, misc)
- Standard 10-20 montage is applied automatically
- Plots are saved with high resolution (150 DPI)
- Scripts create output directories automatically if they don't exist

---

## Contact

For questions or issues, please refer to the project documentation or contact the research team.

