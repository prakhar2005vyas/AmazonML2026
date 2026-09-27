# Pipeline Improvement Plan — v2

Rewritten on 2026-09-26. It is based on three sources:

- A data profile of the raw train and test files.
- The error log of the current best model, `log_err_s2x.txt`.
- A verified web research pass on how practitioners clean business names and addresses for entity resolution.

It replaces the previous audit. Section 1 lists what that audit got wrong.

Scope: `normalize.py`, `build_translit.py`, `blocking.py`, `features.py`, `features2.py`, `model.py`,
`stage2.py`, `oof.py`, `decide.py`, `analyze_errors.py`, `predict.py`, `requirements.txt`, and the notebooks.

---

## 0. Where we stand

| Stage | Hard validation split | Source |
|---|---|---|
| Blocking pair recall, `hard` candidates | 0.9829 | `log_model_hard.txt` |
| Blocking pair recall, `hard2` wide candidates (all train) | 0.9879 | `log_block_hard2.txt` |
| Oracle F0.5 with a perfect matcher on `hard` candidates | 0.9943 | `log_model_hard.txt` |
| Stage 1 alone | 0.9720 | `log_s2_x.txt` |
| Stage 2 with extra features, 2-fold CV | **0.9849** | `log_s2_x.txt` |

F0.5 lost by error type, from `log_err_s2x.txt` (180,971 validation S1 records):

| Error type | S1 records | F0.5 lost |
|---|---|---|
| Some true matches missed, no false matches | 19,446 | 0.0087 |
| Every true match missed | 577 | 0.0032 |
| False matches only | 1,705 | 0.0021 |
| Singleton given a false match | 113 | 0.0006 |
| Both false and missed matches | 243 | 0.0004 |

Why the 22,890 missed true pairs were missed:

| Reason | Pairs |
|---|---|
| Never became a candidate (blocking) | 10,801 |
| Scored below the threshold | 8,493 |
| The S2/S3 record went to a different S1 | 3,596 |

**Takeaway:** missed matches cost about four times as much as false matches. Blocking causes almost half of the missed pairs.
So the remaining score is mostly in **recall**: blocking, transliteration and address tokenization.
Precision work comes after that.

---

## 1. Corrections to the previous plan

### 1.1 Postal codes: drop items #1–#3 of the old plan

The old plan called postal codes "the biggest untapped signal". The data says they barely exist.
Numbers below come from a profile of 300k rows per file.

| Country | Addresses with a 6-digit PIN | Addresses with a trailing ZIP component |
|---|---|---|
| India, S1 | 0.3% | n/a |
| India, S2/S3 | 1.1% | n/a |
| US, S1/S2/S3 | n/a | **0.00–0.01%** |
| France, S1/S2/S3 | n/a | 0% (any 5-digit number appears in only 0.4–0.5%) |

- About 10% of US addresses contain a 5-digit number, but it is the house number at the front, as in "17560 Ellis Road".
  The EDA regex `\b\d{5}\b` in `eda/analyze_text_and_noise.py` counted these as ZIPs. That is where the old
  "postal codes agree 30% on true pairs" figure came from.
- **France has no postal codes.** Blocking on the department number from the postal code cannot work.
- `lstrip("0")` in `norm_addr` is **not a bug** for this data. It only touches house and flat numbers, where
  "004" and "4" really are the same premises, as in "T-004gitanjaliplaza" vs "T-4Gitanjaliplaza".

### 1.2 France: the real risk is a train/test shift, not missing departments

- Every French S1 address ends with its region. About 64% of French S2/S3 addresses name a region or department,
  and `FR_REGIONS` already covers all three regions and their departments that appear in test.
- The other ~36% of French S2/S3 records name only a city. `infer_missing_states` learns city-to-region from S1,
  and that works: 441,274 of 723,295 test records without a state got one inferred. 59,593 French records stay without a state.
- **The actual problem:** in train, a record has no state *only when its address is empty*. Train had 344,883
  records without a state, and inference filled **0** of them. The profile confirms it: every non-empty train address
  in S2/S3 has a state. So during training, `state_other_empty = 1` always meant "no address at all".
  In test France it can mean "has an address, just no region". The model has never seen that combination,
  and `state_other_empty` is a top-10 stage-2 feature. Section 3.3, item B3, fixes this.

### 1.3 Unchanged from the old plan

- **Keep LightGBM.** The research agrees: practitioners and Kaggle solutions on the closest comparable problem
  (Foursquare Location Matching 2022) use candidate retrieval plus a GBDT over many string similarities.
  One benchmark found a learned combiner beats any single metric.
- Keep the `hard` validation regime as the number to chase.
- The `lambdarank` A/B and the small-embedding feature stay as optional later items. See section 4.

---

## 2. What the research says, and how it applies here

The research verified 11 claims from primary sources. None of them measures gains from cleaning *business* names or addresses,
so treat all of this as common practice, not proven lift. Every change below needs a hard-split ablation.

| Finding | Source | What it means here |
|---|---|---|
| Normalization raises recall but causes more false matches. Heavy cleaning slightly *lowered* F-measure (0.883 → 0.875 in one study). | Randall et al. 2013; Grannis et al. 2019 | Give the model **raw and normalized** similarities side by side. Today every feature uses normalized text only. |
| SoftTFIDF (TF-IDF cosine where tokens with Jaro-Winkler > 0.9 also match) was the best single name metric on 11 datasets. A learned combiner beat it slightly. | Cohen, Ravikumar, Fienberg 2003 | Add SoftTFIDF and Monge-Elkan features. `features2._align` already does most of the work. |
| Company name matching in practice: character 2–3-gram TF-IDF picks top-50 candidates, then overlap, weighted Jaccard, Ratcliff-Obershelp, token-sort and **Editex** (phonetic) score them. | DNB `name_matching` | Our blocking uses 3-grams only and K=3–20. We have no phonetic metric. |
| Foursquare solution: unidecode, "&" to "and", punctuation removal, per-country TF-IDF on words and `char_wb` 3-grams, character-shingle Jaccard on names (k=2,3) and addresses (k=3), number-overlap features. | oasidorshin/FoursquareLocationMatching | We already do most of this. We don't have address character-shingle Jaccard. |
| IndicXlit (MIT) romanizes Hindi, Bengali, Telugu, Kannada and Tamil offline. Exact-match accuracy on named entities is only 35–61%. | AI4Bharat/IndicXlit | Better than rule-based ITRANS for names missing from our learned dictionary. Still needs fuzzy or phonetic comparison afterwards. It is hard to install on Windows, so run it on Colab. |
| Aksharamukha is AGPL. | virtualvinodh/aksharamukha | Don't use it. `indic_transliteration` (MIT) and IndicXlit are allowed. |
| USPS Publication 28 Appendix C1 lists each street suffix with its variants and misspellings (AVEN, AVNUE → AVE; STRT, STR → ST). | pe.usps.com | Extend `ADDR_CANON` with the variants that actually occur in our data. |
| libpostal `expand_address` handles US and French street types and accents, but returns a *set* of expansions, and "st" is ambiguous between street and saint. | openvenues/libpostal | **Don't adopt.** Our table covers the forms in the data, and libpostal's model data is large. Mapping both street and saint to "st" is right for this data, since the noise turns "St" into "Saint" (error log: "17 Main Saint"). |
| cleanco (MIT) and the GLEIF legal-form list (CC0, more than 3,600 forms) are used to strip legal suffixes. | psolin/cleanco; gleif.org | Our `LEGAL` set already covers what's in the data. Use GLEIF only to check the French list, since France is not in training. |

---

## 3. File-by-file changes

Priority: **P1** = do first, **P2** = next, **P3** = only if time allows.
"Target" is the error bucket from section 0 that the change should shrink.

### 3.1 `normalize.py`

**N1 (P1): split digits from letters in address tokens.** Target: missed candidates, pairs below threshold.
About 2% of Indian addresses in every source have glued tokens:
`2Ndfloor`, `9Thfloorcollegeroad`, `4Gitanjaliplaza`, `FLAT502`, `SECTOR120`, `ROAD5TH`, `Pkt2Se34Rohinilandmarkdda`.
Today `norm_addr` pulls the digits into `nums` but keeps the whole token, so `4gitanjaliplaza` never matches `gitanjaliplaza`.

```python
GLUE = re.compile(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z]{3,})")   # keep ordinals like 2nd, 5th, 4a together
ORD  = re.compile(r"^\d+(st|nd|rd|th)$")
# in norm_addr, before token handling:
t = t if ORD.match(t) else GLUE.sub(" ", t)
```

Apply the same split to `nums` extraction so numbers stay identical. Don't split letter runs inside words
like "rohinilandmarkdda". Word segmentation is out of scope.

**N2 (P1): better fallback for native-script tokens.** Target: missed candidates.
About 23% of Indian S2 names and 12% of S3 names are fully in native script, the same in train and test.
Tokens missing from `translit.json` go through `sanscript` → ITRANS, a scholarly scheme with doubled vowels,
case markers and inherent "a" endings, not colloquial English spelling. The error log has
"फॉर्च्यून सॉफ्टवेयर प्राइवेट लिमिटेड" vs "Fortune Software Private Limited" as a blocking miss.
Run the fallback on a sample of these tokens first to see exactly how far its output is from the S1 spelling.

1. Add a colloquial folding step to the ITRANS output in `translit_token`: `aa→a`, `ii→i`, `uu→u`, `~n/~N→n`,
   drop `.h`, `.N` and `M` markers, remove a trailing inherent `a` on consonant-final words, `ph→f` at the start of a word,
   `w/v→v`, `sh/Sh→sh`. Keep it in a small table so it's easy to ablate.
2. Optional, on Colab: IndicXlit `XlitEngine(src_script_type="indic", beam_width=4)` for tokens still missing from the dictionary.
   Cache results per unique token in `WORK/xlit_cache.json`, since there are only tens of thousands of unique tokens.
   Download the model files ahead of time and pin `ai4bharat-transliteration`. It needs about 12 GB of RAM, so it will not run on the laptop.
3. Write a new column `name_script` = the original script of the name, or `"latin"`. `features.py` uses it, see F5.

**N3 (P1): write raw-lite columns.** Target: pairs below threshold, false matches.
Add `name_lite` and `addr_lite`: NFKC, unidecode, lowercase, punctuation to space, collapsed whitespace.
Keep everything else, including legal words, stop words, numbers and the original order. `features.py` uses them for
raw-vs-normalized features (F1). This is the research's main guard against over-normalization.
The extra storage is about 2 × 18M short strings.

**N4 (P2): street-type table additions, only for forms the data contains.** Target: pairs below threshold.

| Add to `ADDR_CANON` | Maps to | Evidence |
|---|---|---|
| `ch`, `che`, `chem` | `chem` (chemin) | "ch" appears about 3,000 times as a leading token in French S2 |
| `crs` | `cours` | French voie type |
| `fbg`, `faubourg` | `fbg` | French voie type |
| `qu`, `quai` | `quai` | "quai" appears about 1,800 times in French S1 |
| `res`, `residence` | `res` | about 950 times in French S1 |
| `cite` | `cite` | after unidecode |
| `aven`, `avenu`, `avn`, `avnue` | `ave` | USPS Publication 28 variants |
| `strt`, `str` | `st` | USPS Publication 28 variants |
| `drv`, `driv` | `dr` | USPS Publication 28 variants |

Before adding the USPS variants, count them in `train_s2/s3` addresses and keep only those that appear.
Keep `bis` and `ter` as house-number modifiers: `5 bis` becomes `5bis` in `nums`, so it doesn't equal a plain `5`.

**N5 (P2): French name tokens.** Target: France precision and recall. France is not in training, so this has to be engineered in.
Leading and trailing tokens in French test names: `sarl` 362k, `sas` 274k, `eurl` 109k, `sa` 88k, `sasu` 79k, `sci` 68k, `ets` 36k, `cie` 33k.
All the legal forms are in `LEGAL` already. Add:

- `NAME_CANON`: `ets` / `etablissements` → `ets`, `freres` → `freres`, `&` is already "and", `fils`.
- `LEGAL`, used as filler words when building `core`: `de`, `du`, `des`, `la`, `le`, `les`, `l`, `d`.
  This matches how `the`, `of` and `and` are handled.
- Check the rest against the GLEIF list for France: SELARL, SCOP, SCM, GIE, SNC, SCA. Add only those that occur in test names.

**N6 (P3): keep Indian compound house numbers as one token.** Target: false matches from distractors with a shifted number.
`H.No.16-11-23/37/A` currently becomes the bag `16 11 23 37`. Also write `hnum_full = "16-11-23/37/a"` so the model can check
exact equality alongside the bag. The research found nothing verified on this, so it's a data-driven experiment. Accept it only if the hard split improves.

### 3.2 `build_translit.py`

**T1 (P1): learn more names from the training pairs.** Target: missed candidates.
Name tokens are aligned only when the native and S1 names have the *same number of tokens* (`len(to) == len(ts)`).
Pairs with different legal suffixes ("प्राइवेट लिमिटेड" vs "Pvt Ltd"), or with an extra "India" or "Services", are skipped entirely.

- Drop legal words on both sides before aligning, using a native-script legal list learned from the same pairs.
- For unequal lengths, align each native token to the S1 token with the best char-3-gram overlap with its ITRANS form,
  above a minimum score. Keep the same `n >= 2` and `share >= 0.5` thresholds.
- Print dictionary coverage: the share of native name tokens in validation S2/S3 found in the dictionary.
  Track it as a metric for each version.

### 3.3 `blocking.py`

**B1 (P1): records without an address get more candidates.** Target: missed candidates, the 10,801 pairs.
Many sampled blocking misses have an empty S2/S3 address, like "Modern Brothers Company", "Maid Massage L.L.C." or "Good Enegy Pvt Ltd".
In train these are exactly the records without a state, so they only get the country-wide name search (`K_NOSTATE`/`K_NOSTATE_WIDE`).

- Add a `v3` tag: `K_NOSTATE_WIDE` 20 → 50 for records with an empty address, matching DNB's default top-50.
  Try `char_wb` `ngram_range=(2, 3)` in `topk` for this path only.
- Report pair recall and candidates per S1 for this path separately. Keep the change only if recall rises and
  total candidates stay manageable for `features.py`.
- A larger K only costs compute. Precision is still decided by the model and the threshold, so this is not the "widen K blindly" trap.

**B2 (P2): phonetic-key search for names that were in native script.** Target: missed candidates.
For S2/S3 records with `name_script != "latin"`, add a third name search on a phonetic key of `core`.
Use Metaphone or NYSIIS from a permissively licensed library. Run it inside the (country, state) block with K=3.

**B3 (P1): make train look like test France.** Target: France, and the risk from section 1.2.
Add a `sim` tag. For a hashed 30% of train S2/S3 records with a non-empty address, remove the state component
from the raw address *before* `normalize.py`. Do it in `convert.py`, or as a pre-step in `normalize.main` behind a flag.
This exercises `infer_missing_states` in train, and it gives the model rows where `state_other_empty = 1` and `a_empty = 0`.
Validate on the same hash-bucket 9 split, and report F0.5 separately for the altered records.
This is the only direct France proxy available.

**B4 (P3): clean up Polars deprecation warnings.** `explode` needs explicit `empty_as_null=True`,
and horizontal `concat` in `oof.score2` should use `how="horizontal"` with equal heights, or `"horizontal_extend"`.
These don't change results, but they will break on the next Polars upgrade.

### 3.4 `features.py`, stage 1

| ID | Pri | Feature(s) | How | Target |
|---|---|---|---|---|
| F1 | P1 | `nl_ratio`, `nl_tset`, `al_tset`, `al_ratio` | rapidfuzz on `name_lite` / `addr_lite` from N3 | below threshold, false matches |
| F2 | P2 | `n_phon_eq`, `n_phon_jw` | Metaphone/NYSIIS key of `core`: equality and Jaro-Winkler | native-script names, typos |
| F3 | P2 | `a_sh3_jacc` | Jaccard on character 3-shingles of `addr` (Foursquare) | address typos such as "Hurricanne", "WILAON" |
| F4 | P2 | `n_editex` | Editex on `compact` (DNB default metric) | phonetic spelling variants |
| F5 | P1 | `script_o` | 1 if the S2/S3 name was native script (from N2), else 0 | tells the model that name similarity is less reliable |

Keep `country` out of the features, as the README requires. France is unseen, and a country feature would put it off-distribution.

Every new stage-1 feature means rerunning `features.py` (train, val, rest, test), `model.py`, `oof.py fit/score`
and `stage2.py`. That is the expensive chain, so **group F1–F5 into one rebuild** instead of one rebuild per feature.
Use the "leave one group out" check in section 5 to judge each group.

### 3.5 `features2.py`, stage-2 extras only (cheap, about 3–8M rows)

| ID | Pri | Feature(s) | How |
|---|---|---|---|
| X1 | P1 | `nx_softtfidf`, `ax_softtfidf` | IDF-weighted cosine where a token matches if Jaro-Winkler ≥ 0.9. Reuse `_IDF`, and do it next to `_align`. |
| X2 | P2 | `nx_monge_elkan` | Mean over tokens of the best Jaro-Winkler against the other name, in both directions |
| X3 | P2 | `nx_homoglyph_eq` | `compact` equality after folding `l/I/1 → l` and `0/O → o`. Error log examples: "lmpoexpo", "Lnfrastructure", "Bio1ogics". `_deleet` only fixes mixed tokens. |

These don't touch stage 1, so they only need `oof.py score2` and `stage2.py val --x`. **Start here** for a quick first signal.

### 3.6 `model.py`, `stage2.py`, `oof.py`, `decide.py`

- **M1 (P1):** No model change. Keep `PARAMS` and `PARAMS2`.
- **M2 (P1):** Use the out-of-fold stage-2 data (`all_hard_s2x`, 8.1M rows) as the default stage-2 training set once
  the branch `feat/s2x-submission-oof` confirms it beats 0.9849 on the same holdout. New feature experiments then build on that setup.
- **M3 (P2):** After each change, run `decide.py` with its expected-F0.5 rule on the new out-of-fold scores, and compare it to the global threshold.
- **M4 (P3):** Try a `lambdarank` stage-2 objective grouped by S2/S3 record. It matches the one-owner decision rule directly.
- **M5 (P3):** Small multilingual embedding cosine as a stage-2 feature (e5-small or MiniLM, Apache/MIT).
  Compute it only on stage-2 rows. It's aimed at DBA names with no shared tokens and at native-script names. Run it on a Colab GPU.

### 3.7 `analyze_errors.py`

**E1 (P1):** Add slices so every change above has a measured target size before and after:

- S2/S3 name in native script (`name_script`).
- S2/S3 address empty.
- S2/S3 address has a glued digit-letter token.
- S2/S3 record from the B3 `sim` set, with an address but no state.

For each slice print the missed pairs, the false-match pairs and the F0.5 lost. Also print blocking recall for each slice.
Right now we only know these sizes from samples.

### 3.8 `predict.py`, `requirements.txt`, submission

- `predict.py` is still the single-stage path (`lgb.txt` + `decision.json`). The submission path is `stage2.py test`.
  Either make `predict.py` call the stage-2 path, or note in the README that it is the baseline, so the final zip reproduces the leaderboard file.
- `requirements.txt`: pin every new library added by N2, B2, F2, F4 and X1–X3 (for example `ai4bharat-transliteration`, and the phonetic library).
  Put IndicXlit in a separate `requirements-colab.txt` if it won't install on Windows.
- Update `business_entity_resolution/README.md` with the new steps and tags (`v3`, `sim`) in run order.

### 3.9 Notebooks

- **`mlc.ipynb`** is untracked on purpose because `.gitignore` excludes it for its presigned links.
  It is **not on disk right now**: the committed version is a three-cell stub, and the Stage-2 out-of-fold version exists only in the open editor tab.
  **Save it from VS Code before closing the tab.**
- Structure it as a thin Colab runner that calls `src/` scripts instead of copying their logic:
  1. Runtime check: fail fast unless Linux, `/content` exists and RAM ≥ 12 GB.
     This catches the local-Julia-kernel problem seen earlier.
  2. Install pinned `requirements.txt` (+ `requirements-colab.txt`).
  3. Fetch the code (git clone, or a zip of `src/`) and the `WORK` inputs from the presigned links.
  4. One cell per step: `normalize` → `build_translit` → `blocking <tag>` → `features` → `model` → `oof` → `stage2`, each with `ER_WORK` set.
  5. Upload `WORK/*.txt` models, `decision*.json` and the two output TSVs back.
- **`test.ipynb`** is empty and bound to a Julia kernel. Delete it, or switch it to a Python kernel if it's meant as scratch space.
- Put new EDA in `eda/` scripts, not notebooks. Fix the ZIP regex in `eda/analyze_text_and_noise.py` so it only accepts a trailing
  5-digit component, and rerun it so the EDA numbers in the write-up are correct.

---

## 4. Execution order

Each batch ends with one hard-split evaluation and an `analyze_errors.py` run on the E1 slices.

| Batch | Items | Cost | Why this order |
|---|---|---|---|
| 0 | E1 slices; save `mlc.ipynb`; fix EDA ZIP regex | Low | Measure first, so every later gain has a baseline |
| 1 | X1, X3 (stage-2 extras only) | Low: `oof.py score2` + `stage2.py val --x` | Fastest signal, no stage-1 rebuild |
| 2 | N1, N3, N5, T1, F1, F5 | High: full rebuild | Biggest recall targets. Rebuild once for all of them |
| 3 | B1 (`v3` tag), B3 (`sim` tag) | Medium: blocking + downstream | Blocking recall and the France shift. Needs batch 2's normalize output |
| 4 | N2 (translit folding / IndicXlit), B2, F2, F4, N4 | High | Native-script tail. IndicXlit only on Colab |
| 5 | M3, M4, M5, N6 | Medium | Decision rule and model tweaks after the features settle |

---

## 5. How to judge each change

- **Metric:** hard-split macro F0.5, stage-2 2-fold CV grouped by S2/S3 record, same buckets as now (validation = hash bucket 9, `hard_keep`).
- **Keep a change if** F0.5 rises by at least **+0.0005**, *and* the singleton false-match loss does not grow,
  *and* the target slice from E1 improves.
- **Leave one group out:** after a batch, retrain once without each new feature group. A group that doesn't cost
  at least 0.0003 when removed is dropped. This keeps the pipeline lean for the final zip.
- **Raw vs normalized:** check that F1's raw-lite features get real importance. If they don't, normalization isn't hurting precision, and N3 can go.
- **France:** no labels exist. Use the B3 `sim` slice as the proxy, and spot-check 50 French test matches by hand before submitting.

---

## 6. Not doing, and why

| Idea | Reason |
|---|---|
| Postal-code features or blocking | Almost no postal codes in the data (section 1.1) |
| libpostal `expand_address` | Returns sets of expansions, large model data, and our table already covers the forms in the data |
| Aksharamukha | AGPL licence |
| Geocoding, business registries, GLEIF's live match API | External lookup, banned by the challenge rules |
| `country` as a model feature | README forbids hard-coding countries; France is unseen |
| Replacing LightGBM | No evidence it's the bottleneck; blocking and features are |

---

## Sources

- Randall et al., "The effect of data cleaning on record linkage quality", BMC Med Inform Decis Mak 2013 — https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/1472-6947-13-64
- Grannis et al., address standardization and linkage, JAMIA 2019 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7787357
- Cohen, Ravikumar, Fienberg, "A Comparison of String Distance Metrics for Name-Matching Tasks", IIWeb 2003 — https://www.semanticscholar.org/paper/9578679e028777dd709881f938114aa59fbbf481
- DNB name_matching — https://name-matching.readthedocs.io/en/latest/name_matching.html
- Foursquare Location Matching solution component — https://github.com/oasidorshin/FoursquareLocationMatching
- AI4Bharat IndicXlit — https://github.com/AI4Bharat/IndicXlit
- Aksharamukha (AGPL) — https://github.com/virtualvinodh/aksharamukha
- libpostal — https://github.com/openvenues/libpostal
- USPS Publication 28, Appendix C1 — https://pe.usps.com/text/pub28/28apc_002.htm
- cleanco — https://github.com/psolin/cleanco
- GLEIF ISO 20275 legal forms — https://www.gleif.org/en/about-lei/iso-20275-entity-legal-forms-code-list
