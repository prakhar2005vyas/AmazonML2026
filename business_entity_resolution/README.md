# Business Entity Resolution — pipeline

Python 3.11. `pip install -r requirements.txt`. Paths are set in `src/config.py`, and each can be overridden with an environment variable:
- `ER_DATA`: the dataset folder containing `train/` and `test/`
- `ER_WORK`: where intermediate files go (about 15 GB)
- `ER_OUTPUT`: where the two output files are written

Run from `src/` in this order:

| Step | Command | What it does |
|---|---|---|
| 1 | `python convert.py` | TSV to Parquet; long-format ground-truth pairs |
| 2 | `python build_translit.py` | Learns native-script to Latin token map from train pairs |
| 3 | `python normalize.py` | Normalizes names and addresses for all sources and splits |
| 4 | `python blocking.py train`, then `python blocking.py test` | Top-K TF-IDF candidates per (country, state) block |
| 5 | `python features.py train`, then `python features.py test` | Pair features |
| 6 | `python model.py` | Trains LightGBM and tunes the decision threshold on held-out S1 entities |
| 7 | `python predict.py` | Writes `output/matching_results.tsv` and `output/candidate_pairs.tsv` |

Uses only the provided data. There are no external lookups and no pretrained models.
