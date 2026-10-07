# Vanderbilt resting-state SEEG for seizure-onset-zone classification (derivative dataset)

## Overview
NEMAR redistribution of Pennsieve Discover dataset 492, version 1 (doi:10.26275/wtaz-dbst; "Vanderbilt BIEN Lab SOZ Resting
State SEEG Classification"; licence on the record: "Creative Commons Attribution - NonCommercial-ShareAlike", written here as
CC-BY-NC-SA-4.0). It accompanies Sundrani et al. (2025), "Deep learning on brief interictal intracranial recordings can
accurately characterize seizure onset zones", Epilepsia 66(9):3180-3192, doi:10.1111/epi.18478.

**This is a processed-data (derivative) dataset**: the signals are the authors' filtered bipolar SEEG; raw recordings are
not part of the release.

Release description (verbatim): This repository contains a de-identified codebase for training and analyzing deep neural networks that classify seizure onset zones (SOZ) vs. non-SOZ brain regions using resting-state stereoelectroencephalography (SEEG) recordings.

The citation for the associated manuscript is listed below:

Sundrani, S., Johnson, G. W., Doss, D. J., Makhoul, G. S., Hidalgo Monroy Lerma, B., Reda, A., Cavender, A. C., Liao, E., Rogers, B. P., Williams Roberson, S., Bick, S. K., Morgan, V. L., & Englot, D. J. (2025). Deep learning on brief interictal intracranial recordings can accurately characterize seizure onset zones. Epilepsia, 66(9), 3180–3192. Portico. https://doi.org/10.1111/epi.18478

## Contents
- 79 participants, one 5-minute recording each (`sub-<code>/ieeg/sub-<code>_task-rest_ieeg.vhdr`), BrainVision,
  float32, microvolts, 500-2048 Hz as released. Participant codes are the release codes (`Epat##`, `Spat##`, `pat##`;
  the release does not explain the prefixes). The paper reports 78 patients; the release has 79 files.
  Codes not in any cross-validation test fold of the authors' results: ['Epat09'].
- `_channels.tsv`: bipolar channel, region (release `bip_montage_region`), and `soz_region_label`: the region-level SOZ
  ground truth used by the authors ('true_label' in the results JSON). See channels.json.
- `_electrodes.tsv` + `_coordsystem.json`: one row per bipolar channel with its region; the release gives no coordinates (x/y/z n/a).
- `participants.tsv`: release code and the authors' test fold.
- Unreadable samples in the release: Epat09: rows 46875-46999 (93.750-94.000 s), channels 0-64 (one corrupt gzip chunk in the HDF5 file; the file matches the Pennsieve SHA-256, so the release itself carries it; the samples are NaN here and marked in that recording's _events.tsv)
- `sourcedata/pennsieve-492-v1/`: the complete release byte-identical (MAT files, code, model weights, results JSON,
  location_accuracy.csv, README.md, Pennsieve readme/manifest/changelog/banner), except macOS `.DS_Store` files and a
  compiled `.pyc` (1 files skipped: ['files/models/__pycache__/multi_scale_ori.cpython-39.pyc']); the Pennsieve listing with checksums;
  `roundtrip_float32.tsv` (max relative float32 error over all files: 5.77e-08).

## Recording and processing (from the paper)
Five minutes of resting-state (interictal, non-SPES) SEEG from patients with drug-resistant epilepsy evaluated at
Vanderbilt University Medical Center; patients awake, eyes closed, told to try not to fall asleep; recordings at least 4 h
apart from electroclinical activity. Filtering: MATLAB filtfilt Butterworth passbands 1-59, 61-119 and 121-150 Hz.
Contacts were assigned to Desikan-Killiany regions; SOZs were defined on a region basis as regions containing any contact
involved in the ictal onset of one or more seizures. Cohort (paper Table 1, n = 78): 45 female, 33 male; per-patient age and
sex are not in the release.

## Changes made for NEMAR
filt_data (float64, volts) written as float32 microvolts (BrainVision); channel names without spaces; unreadable source samples written as NaN (see Contents). Nothing else changed.

## Privacy
The release contains only signals, channel/region labels, model outputs and code. No names, dates or hospital numbers were
found in the MAT strings, JSON or text files. The code README lists the corresponding author's institutional e-mail.

## Ethics approval
Sundrani et al. 2025, Methods (Participants and Resting State SEEG): "This study was approved by the Vanderbilt
Institutional Review Board, and informed subject consent was obtained."

## Funding
Sundrani et al. 2025, Acknowledgements (verbatim in dataset_description.json).

## Licence
CC-BY-NC-SA-4.0 (Pennsieve licence field: "Creative Commons Attribution - NonCommercial-ShareAlike"). Non-commercial use only.
