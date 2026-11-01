# Industrial Pump Anomaly Detection

Unsupervised anomaly detection for an industrial water pump using an LSTM autoencoder, with an evaluation designed for a dataset that has very few fault events and a drifting "normal" behaviour.

> **Status:** model development and evaluation are complete (see `notebooks/`). A Flutter + Firebase monitoring app that serves the model is in progress (see [Roadmap](#roadmap)).

## Why this project is different from a basic autoencoder demo

An autoencoder trained on normal data is easy to build, but its results are easy to misjudge. The first version of this project reported a precision of only **0.08** (it flagged almost every window as abnormal) because of data leakage, a narrow training period and an uncalibrated threshold. This repository therefore focuses on **measuring the model honestly**:

| Problem found | What was done |
|---|---|
| Scaler and threshold used information from the test period (leakage) | Block-based split; scaler fitted on normal training rows only |
| Only a handful of fault events, so one test set is a matter of luck | 5-fold block cross-validation (every event tested once) and bootstrap confidence intervals |
| Thresholds tuned on fault labels do not transfer and cannot be used in deployment | A **label-free threshold** (99th percentile of normal validation errors), reported next to the tuned one |
| Training runs differ from each other | Ensemble of 3 seeds, mean and standard deviation over runs |
| Random split ignores time | **Forward-chaining** evaluation (train on the past, test on later periods) |
| Normal behaviour may drift | Model-free **drift diagnosis** per sensor and several drift-oriented model variants |
| Only one comparison method | Six comparison methods (Isolation Forest, PCA, One-Class SVM, dense and GRU autoencoders, a hybrid) |

## Key findings

1. **The LSTM ranks abnormal windows well** (AUC about 0.98 on the primary test set, 0.987 in cross-validation), but **simpler methods match it**: One-Class SVM, a GRU autoencoder, PCA and an LSTM + Isolation Forest hybrid are statistically indistinguishable from the LSTM ensemble on the primary test set. Only the plain Isolation Forest was clearly worse there (paired bootstrap).
2. **Threshold choice matters more than the model.** In cross-validation the F1-score varies strongly between folds (standard deviation about 0.2 to 0.3) while the AUC stays near 0.99: the ranking is reliable, the decision threshold is not. The label-free threshold performs about as well as the tuned one on the primary test set (F1 0.78 vs 0.79) and usually better in forward chaining, and it needs no fault labels.
3. **Normal behaviour of this pump drifts.** One sensor (`sensor_50`) moves by roughly 18 training standard deviations in the second half of the data. A model with a min-max scaler fitted on the past reconstructs *normal* later data about 7 to 8 times worse than normal training data, and its AUC drops in a later period (0.977 in the first forward-chaining step, 0.775 in the second).
4. **No drift remedy won clearly.** A robust scaler restored the AUC in the second step (0.982) but lowered recall at a fixed false-alarm rate in the first; first differences and an adaptive threshold failed. Over both steps the plain **Isolation Forest was the most robust** (AUC 0.990, F1 0.61), and the LSTM + Isolation Forest hybrid came second.
5. **Where the LSTM leads:** it alarms on more of the windows containing a `BROKEN` reading than the Isolation Forest (42 of 70 vs 24 of 70 across all cross-validation folds, label-free threshold), and its per-sensor reconstruction error provides an explanation of *which* sensors behave unusually. Note that the dataset contains only 7 `BROKEN` readings.

## Dataset

`sensor.csv`: 220,320 readings from 51 sensors with a `machine_status` column (NORMAL 205,836; RECOVERING 14,477; BROKEN 7). About 6.6% of the readings are abnormal and they are clustered in a few periods. The data is a public benchmark and **not** from a live deployment. Source: *add link here*. The file is not included in this repository (it is large).

## Approach

1. Clean the data, label `NORMAL` as 0 and everything else as abnormal (1).
2. Cut the series into windows of 30 consecutive readings (stride 3). A window is abnormal if any of its rows is abnormal.
3. Train an LSTM autoencoder (LSTM 128 -> 64 -> RepeatVector -> LSTM 64 -> 128 -> Dense) on **normal windows only**.
4. Score every window with its reconstruction error (mean absolute error). A window above the threshold raises an alarm.
5. Threshold used for deployment: the 99th percentile of the errors of normal validation windows (**label-free**).

## Evaluation design

- **Block split:** the data is cut into blocks of 3,000 rows and every block belongs to exactly one of train, validation or test, so overlapping windows of one event never appear in two sets.
- **Block cross-validation (5 folds):** every block, and therefore every fault event, is tested once.
- **Forward chaining:** training on the blocks before a cutoff; validation and test on the following periods.
- **Metrics:** AUC, average precision (AP), recall at a fixed 1% false-alarm rate, precision, recall, F1 (tuned and label-free threshold), plus event-level detection rate and false-alarm episodes. Precision and F1 depend strongly on the share of abnormal windows in a test set, which is why threshold-free metrics are reported as well.

## Results

### Primary test set (18 test blocks, 6 of them with abnormal data)

| Method | AUC | AP | F1 (tuned) | F1 (label-free) |
|---|---|---|---|---|
| LSTM ensemble (3 seeds) | 0.982 | 0.726 | 0.789 | 0.783 |
| LSTM, mean of 3 runs | 0.982 | 0.726 | 0.728 | 0.749 |
| One-Class SVM | 0.983 | 0.733 | 0.760 | 0.799 |
| Dense autoencoder | 0.982 | 0.804 | 0.693 | 0.714 |
| GRU autoencoder | 0.977 | 0.663 | 0.758 | 0.762 |
| LSTM + Isolation Forest (hybrid) | 0.983 | 0.726 | 0.720 | 0.767 |
| PCA reconstruction error | 0.976 | 0.714 | 0.694 | 0.575 |
| Isolation Forest | 0.942 | 0.491 | 0.547 | 0.554 |
| First version of this project | - | - | 0.16 | - |

With only six abnormal test blocks the 95% bootstrap intervals are wide (for example LSTM ensemble AUC [0.954, 0.997], label-free F1 [0.456, 0.911]). Only the Isolation Forest differs significantly from the LSTM ensemble in all three metrics; the dense autoencoder is lower in label-free F1 only.

### 5-fold block cross-validation (mean over folds)

| Method | AUC | AP | F1 (tuned) | F1 (label-free) | Events detected | False-alarm episodes |
|---|---|---|---|---|---|---|
| LSTM autoencoder | 0.987 | 0.769 | 0.723 | 0.646 | 12 of 13 | 52 |
| Isolation Forest | 0.989 | 0.758 | 0.687 | 0.683 | 11 of 13 | 65 |

Event and false-alarm counts use the label-free threshold. The F1 differences are far smaller than the spread between folds.

### Forward chaining (train on the past, test on a later period; mean of 2 steps)

| Variant | AUC | AP | F1 (tuned) | F1 (label-free) |
|---|---|---|---|---|
| LSTM (min-max scaler) | 0.876 | 0.663 | 0.348 | 0.424 |
| LSTM + robust scaler | 0.962 | 0.532 | 0.465 | 0.457 |
| LSTM + stable sensors only | 0.907 | 0.704 | 0.229 | 0.462 |
| LSTM + first differences | 0.491 | 0.189 | 0.006 | 0.155 |
| LSTM (adaptive threshold) | 0.730 | 0.227 | 0.006 | 0.068 |
| LSTM + Isolation Forest | 0.988 | 0.728 | 0.534 | 0.506 |
| Isolation Forest | 0.990 | 0.734 | 0.613 | 0.669 |

Only two forward-chaining steps contain abnormal data (about 23% and 0.6% abnormal windows, 3 and 2 events), so these numbers are indications, not proof.

### Other measurements

- **Event level (primary test set):** 6 of 6 fault events detected by the LSTM ensemble (label-free threshold), with 21 false-alarm episodes (Isolation Forest: 5 of 6, 87 episodes).
- **Training data size:** using 25%, 50% or 100% of the training windows did not give a clear improvement.
- **Speed (Colab T4, indicative):** about 19,000 windows per second and about 76 ms for a single window with the LSTM; about 52,000 windows per second for the Isolation Forest.

## Limitations

- Only **7 `BROKEN` readings**: genuine breakdowns cannot be evaluated separately from recovery periods; the system detects abnormal states in general.
- **One machine, one public dataset.** Generalisation to other machines or live data is untested.
- **Few events.** Cross-validation contains about a dozen fault events and forward chaining even fewer, so differences between methods should not be over-interpreted.
- A label-free threshold still assumes that a period known to be healthy is available for calibration.
- Drift is diagnosed and monitored, not solved: a deployed system needs periodic retraining or recalibration.
- Event boundaries are cut by the block split, and a window counts as abnormal if any of its rows is abnormal.

## Repository structure

```
notebooks/model_development.ipynb   full experiment with outputs (run on a GPU, about 1.5 hours on a Colab T4)
model_artifacts/                    lstm_autoencoder.keras, scaler.joblib, config.json (not committed if large)
bridge/                             planned: script that scores data and writes to Firestore
flutter_app/                        planned: monitoring app
docs/                               diagrams and screenshots
data/                               sensor.csv (not committed)
```

## How to reproduce

1. Open the notebook in Google Colab, choose **Runtime -> Change runtime type -> GPU**.
2. Put `sensor.csv` into `MyDrive/ML_Project/` (or change `DATA_PATH` in the configuration cell).
3. **Runtime -> Run all.** All settings (window length, seeds, folds, cutoffs, variants) are in the first code cell. The saved artifacts are written to `MyDrive/ML_Project/artifacts/`.

## Roadmap

A Flutter + Firebase application is planned around the saved model:

```
Python bridge (scores windows with the model) -> Firestore -> Flutter dashboard
```

- live status and reconstruction error versus threshold,
- alert history with the most unusual sensors per alert,
- a **drift monitor** based on the reference statistics stored in `config.json` (per-sensor mean and standard deviation of normal data), because the evaluation showed that sensor drift is the main risk for this model.

The model is kept replaceable: the evaluation suggests that an Isolation Forest or an LSTM + Isolation Forest hybrid may be more robust to drift than the LSTM alone, while the LSTM provides the per-sensor explanation.
