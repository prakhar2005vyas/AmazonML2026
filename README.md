# Amazon ML Challenge 2026: Business Entity Resolution

A machine learning pipeline for large-scale multilingual business entity resolution across heterogeneous data sources, built for the **Amazon ML Challenge 2026** by **Prakhar Vyas** and [@atharvaajmera](https://github.com/atharvaajmera).

---

## Problem Statement

The challenge requires resolving noisy business entity records across three distinct sources ($S_1$, $S_2$, $S_3$) spanning three geographic jurisdictions (**United States**, **India**, and **France**), evaluated under the **Macro $F_{0.5}$** metric (which places double the weight on precision over recall). The dataset features diverse structural challenges: pervasive spelling errors, variable abbreviations, native Indic scripts (Hindi, Marathi, Bengali, Tamil, etc.), glued alphanumeric tokens, missing address fields, and significant cross-country domain shifts.

---

## 🚀 Live Demo

Try the real trained model (0.977 portal score) on a live interactive demo: **[https://amazonml2026-live-demo-in2twke8scuyr6lgbrwe6g.streamlit.app/](https://amazonml2026-live-demo-in2twke8scuyr6lgbrwe6g.streamlit.app/)**

Runs the actual normalization, blocking, feature extraction, and LightGBM inference pipeline end-to-end in your browser. Uses a synthetic benchmark database in compliance with competition data redistribution rules (see [Honest Scope & Limitations](#honest-scope--limitations) section below) -- the model and pipeline code are 100% real.

---

## Architecture Overview

The system is structured as a high-throughput, multi-stage retrieval and re-ranking architecture designed to process millions of entity pairs efficiently:

```
[Raw Entities (S1, S2, S3)]
            |
            v
[1. Text Normalization & Transliteration]
  - Unicode/homoglyph folding, glued token unglue
  - Native Indic script -> Latin transliteration (learned dictionary + ITRANS fallback)
  - Regional cleaning (French street types & filler suffix stripping)
            |
            v
[2. Multi-View Blocking] (TF-IDF + sparse_dot_topn)
  - State-partitioned sparse character n-gram matching
  - Wide country-level fallback for records lacking address/state metadata
  - Pair recall: 0.9816 overall (US: 0.9864, India: 0.9743)
            |
            v
[3. Pairwise Feature Engineering]
  - RapidFuzz token set, ratio, Jaro-Winkler, Levenshtein
  - SoftTFIDF on name and address tokens, Monge-Elkan
  - Metaphone phonetic keys, character 3-shingles, raw-vs-lite ratio differentials
  - Competitive context & rival features (margin to runner-up candidate)
            |
            v
[4. Two-Stage LightGBM Classifier]
  - Stage 1 base pairwise ranker (16.13 MB artifact tracked via Git LFS)
  - Stage 2 out-of-fold context re-ranking absorbing candidate group dynamics
            |
            v
[5. Country-Stratified Threshold Tuning & Assignment]
  - US: tau = 0.60 (Val Macro F0.5: 0.9772)
  - India: tau = 0.70 (Val Macro F0.5: 0.9674)
  - France: tau = 0.70 (extrapolated conservative threshold)
            |
            v
[Submission Output: matching_results.tsv & candidate_pairs.tsv]
```

---

## Leaderboard Progression

Iterative improvements were benchmarked locally on a hard validation split (hash bucket 9 of $S_1$, 180,971 records) and confirmed on the official competition portal leaderboard:

| # | Stage / Experiment | Core Modifications | Hard Val $F_{0.5}$ | Portal Public $F_{0.5}$ | Status |
|:---:|:---|:---|:---:|:---:|:---:|
| **00** | Single-stage baseline | Initial single LightGBM model, basic character blocking | 0.9719 | **0.961** | Baseline |
| **01** | `base_s2x` | 2-stage LightGBM architecture with competitive context features (`FEATURES_X`) | 0.9849 ($\tau=0.75$) | **0.976** | Major leap (+0.015) |
| **02** | `stage1_x2` | Added `FEATURES_X2`: SoftTFIDF name/addr, Monge-Elkan, homoglyph fold, Metaphone keys, address 3-shingles | 0.9852 ($\tau=0.70$) | **0.976** | Verified stability |
| **03** | `stage2_norm` | Full normalization rebuild: glued digit/word splits, raw-lite columns, native transliteration folding (`fold_itrans`), script features | **0.9854** ($\tau=0.70$) | **0.977** | **Final Best Model** |

---

## Individual Contributions (Prakhar Vyas)

Responsible for the full modeling and algorithmic pipeline post-initial repository setup:

- **Text Normalization Engine (`normalize.py`, `build_translit.py`)**: Built the preprocessing pipeline including regex alphanumeric separation (`unglue`), raw-vs-lite column representations, Hindi-style schwa deletion and anusvara folding for out-of-vocabulary transliterations, and zero-shot French legal/address token normalizers.
- **High-Recall Blocking Engine (`blocking.py`, `sweep_nostate_k.py`)**: Formulated the state-blocked TF-IDF retrieval system powered by `sparse_dot_topn`. Discovered that empty-address records caused 61% of all missed validation pairs, and designed the wider country-level fallback search (`ER_K_NOSTATE=50`) recovering 94% of previously unblocked records. Overall blocking recall reached **0.9816** (US: **0.9864**, India: **0.9743**).
- **Feature Engineering (`features.py`, `features2.py`)**: Implemented high-performance string metrics using RapidFuzz, phonetic Metaphone keys, character n-gram similarities, token-order invariant SoftTFIDF, and competitive rival features (scoring the gap between top candidate matches per source record).
- **Two-Stage LightGBM Re-ranking (`model.py`, `stage2.py`, `oof.py`)**: Designed the two-stage classification hierarchy. Trained the Stage 1 GBDT classifier and orchestrated out-of-fold predictions to train Stage 2 models that capture group-level competitive dynamics without data leakage.
- **France Distribution-Shift Diagnosis & Treatment**: Identified an acute domain gap on test data where French records exhibited 11.0% uncertain owner pairs ($0.1 \le q \le 0.9$) compared to 3.7% in India and 3.8% in the US. Engineered targeted rule-based French preprocessing (stripping filler suffixes like *"Et Fils"*, *"& Associés"*, unifying *"3 BIS"* vs *"3B"*, handling abbreviated street types) without overfitting training data.
- **Country-Stratified Threshold Tuning & Tooling (`eval_by_country.py`, `analyze_errors.py`, `pipeline_phase5_6.py`)**: Developed evaluation tooling to stream validation features and optimize decision boundaries by geographic region, achieving US $\tau=0.60$ (Macro $F_{0.5}$ 0.9772) and India $\tau=0.70$ (Macro $F_{0.5}$ 0.9674).

---

## Key Metrics & Experimental Verification

- **Blocking Recall**:
  - Overall Pair Recall: **0.9816**
  - United States: **0.9864**
  - India: **0.9743**
- **Decision Threshold Optimization**:
  - US entities: $\tau = 0.60 \implies \text{Macro } F_{0.5} = 0.9772$
  - India entities: $\tau = 0.70 \implies \text{Macro } F_{0.5} = 0.9674$
- **Model Artifact**:
  - `business_entity_resolution/src/models/classifier.txt` (16.13 MB, trained LightGBM model, tracked via **Git LFS**).
- **Official Portal Result**: **0.977 Macro $F_{0.5}$** (Ranked on public competition leaderboard).

---

## Honest Scope & Limitations Note

- **France Segment Ground Truth**: The training split contained zero ground-truth records for France (training ground truth was available only for US and India). The decision threshold for France was extrapolated conservatively ($\tau = 0.70$) based on similarity distributions and evaluated solely via portal submissions rather than local cross-validation.
- **Large Dataset & Artifact Storage**: Raw competition datasets (~2.4 GB) and full intermediate candidate pair files (`output/**/*.tsv`, ranging from ~686 MB to ~954 MB per stage, alongside ~15 GB of scratch parquet caches) are excluded via `.gitignore` to maintain repository hygiene. All submissions were validated using `student_resource/utils/validate_submission.py`.

---

## Repository Structure

> **Interactive Demo Source**: The standalone Streamlit live demo application and synthetic benchmark database are available in the dedicated companion repository: **[prakhar2005vyas/AmazonML2026-live-demo](https://github.com/prakhar2005vyas/AmazonML2026-live-demo)**.

```
.
├── .gitattributes                                  # Git LFS tracking for classifier.txt
├── .gitignore                                      # Excludes raw data, intermediate parquet, and large TSVs
├── plan.md                                         # Comprehensive improvement plan & error analysis
├── README.md                                       # Case study & architecture documentation
├── business_entity_resolution/
│   ├── EXPERIMENTS.md                              # Detailed log of experimental stages and portal scores
│   ├── requirements.txt                            # Python dependencies (polars, lightgbm, rapidfuzz, etc.)
│   ├── run_rebuild.sh                              # Full pipeline execution script
│   ├── run_testonly.sh                             # Fast evaluation and inference script
│   └── src/
│       ├── analyze_errors.py                       # Error analysis and slice inspection
│       ├── blocking.py                             # TF-IDF candidate generation via sparse_dot_topn
│       ├── build_translit.py                       # Native-to-Latin transliteration dictionary builder
│       ├── config.py                               # Path configuration and environment variables
│       ├── convert.py                              # TSV to parquet converter
│       ├── decide.py                               # Thresholding and match resolution
│       ├── eval_by_country.py                      # Country-stratified threshold sweep & validation tooling
│       ├── features.py                             # Core pairwise string similarity features
│       ├── features2.py                            # Advanced features (SoftTFIDF, Metaphone, Monge-Elkan)
│       ├── metrics.py                              # Macro F0.5 evaluation metric implementation
│       ├── model.py                                # LightGBM training and threshold calibration
│       ├── normalize.py                            # Multilingual text normalization & transliteration
│       ├── oof.py                                  # Out-of-fold feature generation for stage-2 training
│       ├── pipeline_phase5_6.py                    # Pairwise features, classifier training, and eval runner
│       ├── predict.py                              # Final submission candidate and matching generator
│       ├── stage2.py                               # Stage 2 re-ranking pipeline
│       ├── sweep_nostate_k.py                      # K-parameter sweep for state-less candidate blocking
│       └── models/
│           └── classifier.txt                      # Trained LightGBM model artifact (16.13 MB via Git LFS)
└── eda/                                            # Exploratory data analysis scripts and findings
```

---

## Reproduction & Pipeline Execution

```bash
# 1. Install dependencies
cd business_entity_resolution
pip install -r requirements.txt

# 2. Run the pipeline stages from src/
cd src
python convert.py              # Convert TSVs to Parquet
python build_translit.py       # Build native-to-Latin script map
python normalize.py            # Clean and normalize names/addresses
python blocking.py train       # Generate candidate pairs
python blocking.py test
python features.py train       # Compute pairwise feature matrices
python features.py test
python model.py                # Train LightGBM model & evaluate
python predict.py              # Generate output submission files
```
