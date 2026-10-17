# Full-spectrum Transformer study: 2021 only

User-authorized scope: predict future movement and event timing/direction from all prepared spectral details plus extensive causal feature engineering. January-September both model development and parameter choice; October-December final retrospective evaluation. No next-year dataset.

Model variants share a3-layer,d96,4-head Transformer,25%dropout and128consecutive M1 observations. Learned4bar patches preserve each observation in the input projection. Original spectral transforms retain long-history information through8192-period/65537-tap kernels.

- price:33relative geometry/volatility/time/gap features.
- raw: all2356components, each with energy/cosphase/sinphase, plus33price features. Six extra component channels are zeroed in this control.
- engineered:2356×9component features +1835cross-scale/band/dynamics/response features +33price features =23072inputs per observed minute. This is3.25×the7101raw input features, not downsampling to a few selected frequencies.

Input projections have different parameter counts because the auxiliary feature dimensions differ. Reported comparisons are practical feature-set comparisons, not strict equal-parameter causal experiments.

Prediction outputs: four endpoint mean displacements, ordered10/50/90%quantiles, and9probabilities for NONE or first UP/DOWN barrier crossing within1-15/16-30/31-60/61-120observed bars. Barrier=max($2,current past60bar sigma*sqrt60). No full future OHLC path is claimed; predicted future Close=known currentClose+predicted displacement.

`prepare_remote.py` reproduces the SAME saved2021coefficients in Colab with original December2020reference norms, verifies48timestamps across all original2356complex components against HDF5, and builds compact GPU inputs. This avoids transferring23GB. Original64bit files remain untouched. Model-only storage is float16; math uses mixed bf16/fp32 on A10080GB.

`model.py` contains model and causal batch construction. `train.py` runs a32-example memorization check, actual future-mutation input check, then development selection in days20-22ofEVERYJan-Sep month with256bar symmetric embargo. These are blocked development checks, not a rolling-origin claim. It chooses epoch counts and refits from scratch on ALLJan-Sep before test. One epoch is a chronological full pass over eligible training anchors; no shuffled/sampled epochs. `evaluate.py` uses frozen validation thresholds and next-Open entry for illustrative trading evaluation. No ex-post gap filter in the trading policy.

Target labels use future data; no target or future token enters the model. Input-only normalization is fitted on selection training rows, then on all development rows for final refits.120bar labels are purged before October. All target changes are measured from the currently closed candle, not from a visible future anchor. Absolute price levels are never model inputs. Existing2021October-December analyses are disclosed; final metrics are not fresh live or blind evidence.

Reproduction on a Colab GPU runtime:
1. Extract input/reference/golden files plus core.py and prepare_remote.py under `/content/spectral`.
2. Run `python /content/spectral/prepare_remote.py`.
3. Copy model.py/train.py to the same directory and run `python /content/spectral/train.py`.
4. Retrieve results; use local evaluate.py and report.py.

The report archive contains inputs, code, checkpoints, curves, predictions, checks and labels. The transient5GBGPU feature matrix is reproducible and is not duplicated into the archive. GPU is stopped once final artifacts are retrieved. No later training stage is automatically authorized by this experiment.
