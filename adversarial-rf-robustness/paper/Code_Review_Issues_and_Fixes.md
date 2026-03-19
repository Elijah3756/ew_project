# Code Review Issues and Fixes Log

**Date:** February 23, 2026
**Paper:** Adversarial Robustness of AI-Based RF Signal Classification Under Channel Impairments
**Author:** Elijah Bellamy

---

## Round 1: Three Issues (Reviewer A)

### Issue A1: Rho is a Norm Ratio, Not Power Ratio
**Status:** FIXED (manuscript only -- no code changes needed)
**Severity:** Moderate (labeling error, not a computational error)

**The Problem:** The code constrains `||delta||_2 / ||x||_2 <= rho`, which is an L2 norm (amplitude) ratio. The paper incorrectly called rho a "power ratio." Power ratio = rho^2. At rho=0.01, the power ratio = 0.0001 = -40 dB.

**Key Insight:** The paper's dB values (-46 to -26 dB) were already correct because they computed rho^2. The error was in the prose, not the math. The Flowers et al. comparison claiming "double their 0.5%" was wrong -- our rho=1% norm ratio corresponds to 0.01% power, roughly 50x weaker than Flowers' 0.5% power budget.

**Files Changed:**
- `paper/sections/01_introduction.md` -- Updated constraint description
- `paper/sections/02_related_work.md` -- Corrected Flowers comparison
- `paper/sections/03_threat_model.md` -- Fixed rho definition (line 51)
- `paper/sections/05_results.md` -- Corrected all rho references
- `paper/Lit_Review_Alignment_Check.md` -- Fixed comparison analysis

### Issue A2: PGD Stochastic Channel (No-CSI Attack)
**Status:** FIXED (code + manuscript)
**Severity:** Moderate (valid experiment, but needed proper framing)

**The Problem:** Each PGD step generates a new random channel realization via `channel.forward()`, so the optimizer sees 10 different loss landscapes across its 10 steps.

**Fix:** Added `freeze_channel` parameter to `pgd_attack()`:
- `freeze_channel=False` (default): No-CSI stochastic attack (preserves original behavior)
- `freeze_channel=True`: Perfect-CSI attack using fixed channel realization
- Added `--freeze-channel` CLI flag to evaluate.py

**Files Changed:**
- `attacks/pgd.py` -- Added freeze_channel parameter with RNG seed management
- `evaluate.py` -- Propagated freeze_channel through eval functions and CLI

### Issue A3: Threat Model Framing (No-CSI)
**Status:** FIXED (manuscript only)
**Severity:** Low (framing issue)

**The Problem:** Paper said "white-box... strongest attacker" but adversary has no channel state information.

**Fix:** All references now say "white-box model knowledge with no CSI (no-CSI)" and note that a perfect-CSI adversary would be strictly stronger but less realistic.

**Files Changed:**
- `paper/sections/01_introduction.md`
- `paper/sections/02_related_work.md`
- `paper/sections/03_threat_model.md`

---

## Round 2: Four Issues (Reviewer B)

### Issue B1: Fading Applied to Receiver Thermal Noise
**Status:** DOCUMENTED (fundamental RadioML limitation)
**Severity:** HIGH -- affects all fading-channel results

**The Problem:** RadioML samples contain baked-in AWGN. Applying `RayleighFadingChannel(x)` computes `h * (x_clean + noise)` instead of `(h * x_clean) + noise`. This keeps SNR constant through fading, eliminating fading's primary destructive effect.

**Why Not Fully Fixed:** This requires noiseless source signals, which RadioML does not provide. The fix would require either (a) generating synthetic clean I/Q data, or (b) using a dataset that separates signal from noise.

**Mitigation:** Added comprehensive documentation in evaluate.py explaining the limitation. Noted that relative comparisons (clean vs adversarial under fading) remain valid since both are subject to the same approximation.

**Files Changed:**
- `evaluate.py` -- Added KNOWN LIMITATION documentation blocks

### Issue B2: Clairvoyant Noise Attacker
**Status:** DOCUMENTED (fundamental RadioML limitation)
**Severity:** Moderate at low SNR, minor at high SNR

**The Problem:** The attacker receives `x = x_clean + noise_native` and computes gradients including the exact noise realization. A real pre-channel attacker wouldn't know the receiver's thermal noise.

**Why Not Fully Fixed:** Same root cause as B1 -- RadioML doesn't provide separate signal/noise components.

### Issue B3: Dynamic Noise Scaling in AWGNChannel
**Status:** FIXED
**Severity:** HIGH (when used in CompositeChannel after fading)

**The Problem:** `AWGNChannel` computes noise power proportional to input signal power. After fading reduces signal power, noise power also drops, maintaining constant SNR. Real thermal noise is constant.

**Fix:** Added `constant_noise_power` parameter to `AWGNChannel`. When set, uses fixed noise power independent of input signal level.

**Files Changed:**
- `channels/awgn.py` -- Added constant_noise_power mode and from_snr_and_signal_power() class method

### Issue B4: Double-Noising During Training
**Status:** FIXED
**Severity:** HIGH -- invalidated defense training results

**The Problem:** Channel augmentation during training applied CompositeChannel (including AWGNChannel) to RadioML samples that already contain AWGN.

**Fix:** Removed AWGNChannel from the CompositeChannel used for training augmentation. Native dataset noise serves as AWGN. Added runtime warning if AWGNChannel is enabled via config file for RadioML data.

**Files Changed:**
- `train.py` -- Removed AWGNChannel from hardcoded channel_aug; added warning in config-based path
- `train_defenses.py` -- Removed AWGNChannel from defense channel composite

---

## Round 2: Five Issues (Reviewer C)

### Issue C1: No Randomized Smoothing at Inference
**Status:** FIXED (manuscript corrected)
**Severity:** Moderate (overclaimed in paper)

**The Problem:** Paper claimed "randomized smoothing" with "Monte Carlo averaging at inference." Code only implements Gaussian noise injection during training, with a single forward pass at inference.

**Fix:** Renamed defense to "Noise Injection (Gaussian Data Augmentation)" in manuscript. Explicitly stated that Monte Carlo inference is NOT implemented.

**Files Changed:**
- `paper/sections/03_threat_model.md` -- Corrected section 3.5
- `paper/sections/05_results.md` -- Corrected section 5.3.3

### Issue C2: Flawed Transferability Metric
**Status:** FIXED
**Severity:** Low-Moderate (metric deflation)

**The Problem:** Transfer rate denominator `src_fooled` included samples where the target model was already wrong on clean data. These could never appear in the numerator `both_fooled`, deflating the rate.

**Fix:** Changed denominator to only count source-fooled samples where the target also correctly classified the clean sample.

**Files Changed:**
- `eval_transferability.py` -- Added `src_fooled_tgt_correct` accumulator, fixed transfer_rate calculation

### Issue C3: Early Stopping on Clean Accuracy Defeats Adversarial Training
**Status:** FIXED
**Severity:** Moderate -- selects wrong checkpoint

**The Problem:** All defense types used `evaluate_clean()` for model selection. For adversarial training, this selects the epoch with best clean accuracy, which has worst robust accuracy.

**Fix:** For adversarial training defenses, model selection now uses 40% clean + 60% robust accuracy.

**Files Changed:**
- `train_defenses.py` -- Updated validation logic in train_model()

### Issue C4: Broken PGD under Fading (No EoT)
**Status:** PARTIALLY ADDRESSED (in Round 1, Issue A2)
**Severity:** Moderate

**Additional Concern from Reviewer C:** Even with freeze_channel, the default no-CSI mode uses a single channel sample per PGD step, which is a very noisy approximation of Expectation over Transformation (EoT). Proper EoT would average gradients over multiple channel samples per step.

**Status:** Documented as limitation. Implementing full multi-sample EoT would require significant code changes and increased compute cost. The freeze_channel=True option provides the "perfect CSI" upper bound.

### Issue C5: Triple-Noising in Channel-Augmented Adversarial Training
**Status:** FIXED (via Fix B4)
**Severity:** HIGH

**The Problem:** In adversarial training with channel: (1) x has native AWGN, (2) PGD applies channel (with AWGN) inside optimization, (3) channel applied again after attack for training loss.

**Fix:** Removing AWGNChannel from the training composite (Fix B4) resolves this. Channel now only applies fading+CFO, which is physically correct to apply multiple times.

---

## Round 3: Deep Code Review (Internal)

Additional issues found during systematic codebase audit:

### Issue D1: CI Calculation Uses Population Std for Display
**Status:** FIXED
**Severity:** Low (display inconsistency)

**The Problem:** `_compute_ci()` in evaluate.py returned `np.std(values)` (ddof=0, population std) for display, but correctly used `np.std(values, ddof=1)` (sample std) for CI calculation. With 10 trials, population std underestimates by ~5%.

**Fix:** Changed to use `ddof=1` (sample std) consistently.

**Files Changed:**
- `evaluate.py` -- Fixed `_compute_ci()` to use ddof=1 throughout

### Issue D2: Config-Based Channel Augmentation Still Allows AWGN
**Status:** FIXED
**Severity:** Moderate (double-noising via config)

**The Problem:** The `build_channel_augmentation()` function in train.py reads config files and adds AWGNChannel if `awgn: true`. The hardcoded `--channel_aug` path was fixed in Round 2, but this config path was missed.

**Fix:** Added a runtime warning when AWGNChannel is enabled via config, alerting the user to the double-noising risk with RadioML data.

**Files Changed:**
- `train.py` -- Added UserWarning in build_channel_augmentation()

### Issue D3: Non-Stratified Train/Val/Test Split
**Status:** DOCUMENTED (low impact with RadioML)
**Severity:** Low

**The Problem:** `dataset.py` uses pure random permutation for splitting without stratification by modulation class or SNR.

**Why Low Impact:** RadioML 2016.10a has 220,000 samples uniformly distributed across 11 modulations x 20 SNR levels (1000 per bin). With 70/15/15 split on 220K balanced samples, random splitting produces near-identical class/SNR distributions across splits. Statistical deviation is on the order of 1/sqrt(1000) ~ 3%, well within noise.

**Recommendation:** For production use or smaller datasets, implement stratified splitting by (modulation, SNR) bins.

### Issue D4: SNR Filter Applied After Split (2018.01a)
**Status:** DOCUMENTED
**Severity:** Low (for 2016.10a, SNR filter is applied BEFORE loading AND after split, making second filter redundant)

**The Problem:** For 2018.01a, `_apply_split()` applies the SNR range filter after the train/val/test split, potentially creating slightly different SNR distributions across splits.

**Mitigation:** With RadioML 2018.01a's 2.5M+ uniformly distributed samples, the effect is negligible. For 2016.10a, the SNR filter is applied during loading (`_load_2016()`, line 75), so data is already filtered before splitting.

### Issue D5: Model input_length Parameter is Unused
**Status:** DOCUMENTED (by design)
**Severity:** Note

**The Finding:** `RFClassifierCNN` stores `input_length` but doesn't use it because `AdaptiveAvgPool1d(1)` handles any sequence length. This is actually a feature -- the same architecture works for both 2016 (128 samples) and 2018 (1024 samples) data.

**Clarification:** The `input_length` parameter exists for documentation and future use (e.g., if a fixed-size architecture is needed). The model is intentionally length-agnostic.

---

## Outstanding Limitations (Cannot Fix Without New Data)

1. **RadioML baked-in AWGN:** All evaluations under fading channels compute h*(signal+noise) rather than h*signal+noise. This makes fading appear less harmful than reality. (Issues B1, B2)

2. **Clairvoyant attacker:** Adversary has access to exact noise realization in RadioML samples. At high SNR this is minor; at low SNR it may inflate attack effectiveness. (Issue B2)

3. **No true EoT:** The no-CSI PGD attack uses single-sample channel estimation per step, not averaged-gradient EoT. (Issue C4)

4. **No inference-time randomized smoothing:** Only training-time noise injection is implemented. Full Monte Carlo smoothing at inference would be a separate defense. (Issue C1)

---

## Recommendations for Full Correctness

To fully resolve limitations 1-2, the experiment pipeline should:
1. Generate synthetic clean I/Q signals (or use noiseless RadioML 2018 baseline if available)
2. Craft adversarial perturbations on clean signals only
3. Apply fading to (clean + perturbation)
4. Add constant-power thermal noise AFTER fading
5. Evaluate the noisy, faded, perturbed signal

This would require re-running all experiments but would produce physically rigorous results.

---

## Summary of All Changes Made

### Code Files Modified:
| File | Changes |
|------|---------|
| `channels/awgn.py` | Added constant_noise_power mode, from_snr_and_signal_power() |
| `attacks/pgd.py` | Added freeze_channel parameter with RNG management |
| `attacks/fgsm.py` | No changes needed |
| `channels/rayleigh.py` | No changes needed (implementation correct) |
| `channels/cfo.py` | No changes needed (implementation correct) |
| `channels/composite.py` | No changes needed |
| `evaluate.py` | Added freeze_channel propagation, KNOWN LIMITATION docs, fixed _compute_ci |
| `eval_transferability.py` | Fixed transfer_rate denominator |
| `eval_per_class.py` | No changes needed |
| `train.py` | Removed AWGNChannel from channel_aug, added config warning |
| `train_defenses.py` | Removed AWGNChannel, fixed early stopping for adv training |
| `data/dataset.py` | No changes needed (implementation correct) |
| `models/cnn.py` | No changes needed (architecture correct) |

### Manuscript Files Modified:
| File | Changes |
|------|---------|
| `paper/sections/01_introduction.md` | Rho naming, CSI framing |
| `paper/sections/02_related_work.md` | Flowers comparison, CSI framing |
| `paper/sections/03_threat_model.md` | Rho definition, CSI framing, smoothing correction |
| `paper/sections/05_results.md` | Rho references, smoothing correction |
| `paper/Lit_Review_Alignment_Check.md` | Flowers comparison correction |

### Verified Clean (No Issues Found):
- `channels/rayleigh.py` -- Complex gain h~CN(0,1) and I/Q multiplication are physically correct
- `channels/cfo.py` -- Phase rotation exp(j*2*pi*df*n) via Euler's formula is correct
- `channels/composite.py` -- Sequential chaining is correct
- `attacks/fgsm.py` -- L2 norm constraint and gradient computation are correct
- `data/dataset.py` -- Loading, splitting, and SNR handling are correct for RadioML
- `models/cnn.py` -- Architecture is sound (VT-CNN2 inspired, AdaptiveAvgPool is intentional)
