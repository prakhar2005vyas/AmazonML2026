# Experiment log — plan v2 stages

Each stage from `plan.md` lands behind its own flag or tag, so earlier submissions stay reproducible.
After each stage:

1. Run validation. Record the hard-split stage-2 CV F0.5 and the E1 slice table from `analyze_errors.py`.
2. If validation improves, score test and write `output/<stage>/matching_results.tsv` and `candidate_pairs.tsv`.
3. Check both files with `student_resource/utils/validate_submission.py`.
4. Submit `matching_results.tsv` on the portal and record the public leaderboard score here.
5. Decide. **Keep** if validation and portal both improve. **Revert** if the portal drops. If validation and portal disagree, look into it before the next stage.

Validation is the hard split everywhere: hash bucket 9 of S1, with `hard_keep`, 180,971 S1 records.
Portal scores come from the public leaderboard subset.

## Results

Each iteration writes `output/NN_<stage>/NN_<stage>_matching_results.tsv` (and `..._candidate_pairs.tsv`).
The portal shows three decimals, so it can't see a change smaller than about 0.001.

| # | Stage | Change | Val F0.5 (thr) | Portal F0.5 | Decision |
|---|---|---|---|---|---|
| — | single-stage (25 Sep) | stage 1 only | 0.9719 | 0.961 | superseded |
| 01 | base_s2x | Stage 2 with FEATURES_X | 0.9849 (0.75) | **0.976** | reference |
| 02 | stage1_x2 | + FEATURES_X2: SoftTFIDF name/addr, Monge-Elkan, homoglyph fold, Metaphone keys, address 3-shingles | 0.9852 (0.70) | **0.976** | keep: tie on portal, +0.0003 on validation |
| 03 | stage2_norm | Normalization rebuild + raw-lite and script features + translit folding | 0.9854 (0.70) | **0.977** | **keep: new reference** (+0.001 on portal) |
| 04 | stage3_france | 03 + France pack (test-side only) | n/a (no French rows in validation) | _pending_ | _pending_ |
| 2 | Normalization rebuild: glued alphanumerics split, raw-lite columns, French name tokens, better translit alignment, raw-vs-normalized + script features | new `ER_WORK`, tag `hard` | | | |
| 3 | Blocking: wider no-address search (`v3`), train simulation of address-without-state (`sim`) | tags `v3`, `sim` | | | |
| 4 | Native-script tail: translit folding / IndicXlit, phonetic blocking view, French street types | | | | |

## Commands

All commands run from `business_entity_resolution/src` with `C:\Users\prash\.venvs\mlc26\Scripts\python.exe`.

Stage 1:

```
python stage2.py val hard --x2  > %ER_WORK%\log_s2_x2.txt
python analyze_errors.py hard 0.75 val_hard_oofx2.parquet > %ER_WORK%\log_err_x2.txt
python stage2.py test hard --x2 --out stage1_x2 > %ER_WORK%\log_s2_test_x2.txt
cd ..\..\6ab10eb3b23ba_student_resource\student_resource
python utils\validate_submission.py --matching ..\..\output\stage1_x2\matching_results.tsv --candidate ..\..\output\stage1_x2\candidate_pairs.tsv --test-dir dataset\test
```

Baseline E1 slices, from the current model's out-of-fold validation scores:

```
python analyze_errors.py hard 0.75 val_hard_oofx.parquet > %ER_WORK%\log_err_x_e1.txt
```

## Notes per stage

### base
- Test after prune: 8,371,142 pairs, 5,747,078 predicted, threshold 0.75.
- Submission file: `output/s2x_val9849/matching_results.tsv`, the same file as `output/matching_results.tsv`.

### stage 1: stage-2 extras only (no stage-1 rebuild)
- Code: `features2.py` group `x2` (`FEATURES_X2`). `stage2.py` now caches each extra feature group per row set
  in `WORK/{val,test}..._s2_{x,x2}.parquet` and takes `--out DIR`.
- Validation: stage-2 CV **0.98517** at thr 0.70, vs 0.98491 at thr 0.75 for the baseline. That's +0.0003,
  below the plan's +0.0005 keep bar. The gain holds at every threshold from 0.30 to 0.90.
- No `x2` feature is in the top 20 by gain. The top features are still `p`, `logit` and the rival/context features.
- Feature cost: 243 s for 3.36M validation rows, about 36 µs per pair.
- The code used for the test run is frozen in `WORK/src_stage1`.
- Submission: `output/stage1_x2/`. Validator: PASS.
- Test: 8,371,142 pairs after prune, 5,769,721 predicted, thr 0.70.
- Compared with the base submission: 35,081 of 1,732,544 S1 rows changed; 29,484 pairs added, 6,841 dropped.

### Diagnosis after the first portal results: the validation-to-portal gap is mostly France
- Validation 0.985 vs portal 0.976.
- Predicted no-match share is about 6% for every country, matching the train truth of 5.6%. Mean matches per S1 are about the same too.
- Share of test S2/S3 owner pairs with q between 0.1 and 0.9: **France 11.0%**, India 3.7%, US 3.8%.
- Plug-in expected F0.5 from the model's own q (it overestimates validation by 0.005–0.009):

  | | Estimated | Real |
  |---|---|---|
  | Validation India | 0.9920 | 0.9827 |
  | Validation US | 0.9921 | 0.9868 |
  | Test India | 0.9907 | — |
  | Test US | 0.9912 | — |
  | **Test France** | **0.9757** | — |

  To get 0.976 overall with India and US near 0.983, France has to be about **0.93–0.94**.
- Uncertain French pairs show: filler suffixes ("Et Fils", "& Associés", "Groupe", "France") on one side only;
  "3 BIS" vs "3B"; "N°230" normalized to a junk token "ndeg"; "Ch." (chemin) unmapped.

### stage 3 (next): France pack, test-side only
- `normalize.py` with `ER_FR_PACK=1`, the default, and only for `country == "france"`:
  French filler suffixes dropped from `core`; "N°"/"Nº" fixed; house-number suffixes unified (3 bis / 3B → 3 + b);
  street types ch/che/crs/qu/fbg/res/psg. No training row changes, so there's **no retraining**.
- Runner: `bash run_testonly.sh stage3_france C:/Users/prash/mlc26_work_s2 C:/Users/prash/mlc26_work_s3`.
  It reuses stage 2's models and recomputes only test normalize, blocking, features and scoring.
- It can only be judged on the portal. Validation has no French rows.

### stage 4: wider search for state-less records (running, file 05)
- Plan B1. Tags `hard3` (train) and `v3` (test): country-wide char + word name search for S2/S3 records with no state,
  with K from `ER_K_NOSTATE`. In train those are exactly the empty-address records.
- K sweep (`sweep_nostate_k.py`, train, hard S1 subset, 276,144 true pairs whose S2/S3 record has no state):

  | K | Candidates | Pair recall | Validation bucket |
  |---|---|---|---|
  | 5 | 2.6M | 0.7790 | 0.7787 |
  | 20 | 10.9M | 0.8673 | 0.8664 |
  | **50** | **27.4M** | **0.9397** | **0.9408** |
  | 100 | 55.1M | 0.9615 | 0.9624 |

  The current `hard` pipeline, K=5 char view only, gets 0.7547 on validation. K=50 was chosen: about +5,200 validation
  true pairs reach the candidates. K=100 doubles the candidate count for +2 points and risks running out of memory on this 12 GB laptop.
- Run: `TAG=hard3 TTAG=v3 ER_K_NOSTATE=50 NORM_TRAIN=<s2> NORM_TEST=<s3> bash run_rebuild.sh stage4_nostate C:/Users/prash/mlc26_work_s4`.
  It reuses stage 2's train and stage 3's test normalization, so the France pack is included.
  If 04 loses on the portal, only the test side needs a rerun with stage 2's test normalization.

### stage 3: France pack (done, file 04)
- `run_testonly.sh stage3_france C:/Users/prash/mlc26_work_s2 C:/Users/prash/mlc26_work_s3`, about 2 h.
- Test: 8,258,716 pairs after prune, 5,796,359 predicted. Validator PASS.
- Compared with 03: France 28,650 pairs added and 7,949 dropped. India +1,310/−1,301 and US +513/−485, a side effect
  of the shared test IDF.
- Uncertain-owner share on France: 10.1% in 03 (11.0% in 02) → 10.7% in 04. Recall likely goes up, and the confidence signal is mixed.

### stage 2: normalization rebuild (done, file 03)
- Train blocking: pair recall 0.9830 with 54.0M candidates, about the same as before. The address changes didn't move blocking.
- Stage 1 alone: **0.9739**, up from 0.9720, the biggest single-model gain so far. `nl_ratio` is the 9th feature by gain.
- Stage 2 CV: **0.98539** at thr 0.70, vs 0.98517 for stage 1. Stage 2 absorbs most of the stage-1 gain.
  `al_ratio` and `al_tset` are in the stage-2 top 20.
- Test: 8,228,759 pairs after prune, 5,775,621 predicted. Validator PASS.
- One crash fixed: `features._work` returned lists of float tuples, which overflowed Windows pipes (OSError 22) once the
  feature count grew. It now returns float32 arrays, and train features resume.
- E1 slices, pairs grouped by a trait of the S2/S3 record (validation, thr 0.75):

  | Trait | True pairs | Missed | Not in candidates | Blocking recall | Miss rate |
  |---|---|---|---|---|---|
  | all | 625,419 | 22,170 | 10,835 | 0.9827 | 3.5% |
  | native-script name | 44,992 | 1,165 | 904 | 0.9799 | 2.6% |
  | **empty address** | **27,789** | **13,452** | **6,817** | **0.7547** | **48.4%** |
  | glued digit/word token | 5,415 | 54 | 26 | 0.9952 | 1.0% |
  | address but no state | 0 | — | — | — | — |

  **Empty-address records cause 61% of all missed pairs** and 63% of blocking misses. That's plan item B1, and it's the next target for India and US.
  "Address but no state" is empty in train, which confirms the shift described in plan section 1.2.
- Work folder `C:\Users\prash\mlc26_work_s2`. Runner: `bash run_rebuild.sh stage2_norm C:/Users/prash/mlc26_work_s2`.
  The runner resumes after a crash. Frozen source is in `mlc26_work_s2/src_frozen`.
- Changes: N1 digit/word split in addresses (`unglue`); N3 `name_lite` / `addr_lite` columns; N5 French name tokens
  and filler words; T1 fuzzy alignment for native-script names with a different token count; F1 raw-lite features
  (`nl_ratio`, `nl_tset`, `al_ratio`, `al_tset`); F5 `script_o`. Stage 2 uses `--x2`, since stage 1 did not hurt.
- Pulled in from stage 4, item N2: colloquial folding of the rule-based fallback (`normalize.fold_itrans`).
  It applies Hindi-style schwa deletion, turns the anusvara into n/m, and maps candra-o. It was added because of the finding below.
- Finding: the native-script name dictionary covers **96.4%** of native name token occurrences in test.
  The misses are common business words that never appear in native script in train: मोटर्स, स्टोर्स, बेकरी, ज्वेलर्स,
  जनरल, फार्मेसी, ट्रेडर्स… each about 3,600 times in test. They all go through the fallback.
- T1, fuzzy alignment for differing token counts, has almost no effect. Only 10 of 551k native-name pairs differ in
  token count, and the dictionary did not change.
- Fallback quality on the 1,347 dictionary words, whose correct Latin spelling is known:

  | Fallback | Exact matches | Mean ratio | Ratio ≥ 80 |
  |---|---|---|---|
  | ITRANS (before) | 157 | 64.1 | 387 |
  | ITRANS + folding | 222 | 67.7 | 477 |
- Submission: `output/stage2_norm/`.
