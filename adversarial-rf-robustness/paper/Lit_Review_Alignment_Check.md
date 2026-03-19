# Literature Review Alignment Check
## Post-SNR Methodology Fix (No Double-Noising)

**Date:** February 23, 2026
**Purpose:** Assess how corrected experimental results change the paper's narrative and identify misalignments with the literature review framing.

---

## 1. The Narrative Shift

### What the old (buggy) results implied:
The double-noising bug inflated attack success rates by adding AWGN on top of samples that already contained AWGN at their labeled SNR. This made attacks look more devastating than they actually are, because the model was being evaluated on doubly-degraded signals.

### What the corrected results show (2016 dataset, rho = 1%):

| Condition | Clean Acc (0 dB) | Clean Acc (10 dB) | PGD ASR (0 dB) | PGD ASR (10 dB) |
|-----------|-----------------|-------------------|----------------|-----------------|
| AWGN only | 83.68% | 85.59% | 9.20% | 7.37% |
| Rayleigh+AWGN | 45.21% | 41.42% | ~43% | ~46% |
| Rayleigh+CFO+AWGN | 35.59% | 32.98% | ~46% | ~54% |

### The new story (three key findings):

1. **Under AWGN alone, adversarial attacks are surprisingly weak.** PGD at rho=1% achieves only ~7-9% ASR at moderate-to-high SNR. This is far lower than the "90% to below 10%" accuracy drops cited from Flowers et al. [9]. The power-ratio constraint and correct evaluation methodology reveal attacks are much less effective than prior work suggested.

2. **Channel impairments cause far more damage than adversarial attacks.** Rayleigh fading alone drops clean accuracy from ~84% to ~45% at 0 dB -- a 39 percentage point drop. The adversarial perturbation at rho=1% only drops accuracy by ~7 percentage points under AWGN. Channel effects dominate.

3. **The ASR paradox under fading channels.** ASR appears higher under Rayleigh (~43-54%) than under AWGN (~7-9%), but this is misleading. The fading channel's stochastic nature means the adversarial gradient computed on the clean sample is decorrelated from the actual received signal. The "higher ASR" partly reflects the channel itself disrupting classification, not the attack being more effective. The robust accuracy under fading (~40-43%) is actually close to the clean accuracy under fading (~41-45%), meaning the attack adds minimal additional damage on top of what the channel already does.

---

## 2. Specific Misalignments to Fix

### 2.1 Introduction (01_introduction.md)

**Current framing (Line 5-6):** "the robustness of these AI-driven classifiers under adversarial conditions remains poorly understood... deep neural networks are susceptible to adversarial perturbations -- small, carefully crafted modifications to inputs that cause misclassification with high confidence."

**Issue:** This sets up the expectation that attacks will be devastating. Our results show they're modest under AWGN with power-ratio constraints.

**Recommended revision:** Keep the framing about vulnerability being "poorly understood" but emphasize that the interaction with channel conditions is the key unknown, rather than implying attacks are necessarily devastating. Add nuance: "Whether this vulnerability translates to operationally significant risk when attacks must contend with realistic channel propagation is an open question that existing work has not adequately addressed."

---

### 2.2 Related Work Section 2.3 (02_related_work.md)

**Current claim (Line 17):** "Flowers et al. [9] demonstrated that FGSM, PGD, and momentum-based iterative attacks can reduce modulation classification accuracy from over 90% to below 10% with perturbation power as low as 0.5% of signal power."

**Issue:** This is the most problematic claim. A critical confusion exists here: If Flowers used 0.5% as a power-ratio constraint, then our rho=1% (norm ratio) corresponds to rho^2 = 0.01% power ratio, which is 50x WEAKER than their 0.5% power ratio, not "double." Our 7-9% ASR would be much lower than their reported 90%->10% drop. If Flowers used 0.5% as a norm-ratio constraint (same as our metric), the comparison still doesn't match our results, indicating differences in evaluation methodology, SNR evaluation approach (possibly double-noising), or different model/dataset specifics.

**Recommended revision:** Keep the citation but add important context: "However, the comparison between studies depends critically on whether Flowers et al. defined their 0.5% constraint as a norm ratio or power ratio. If 0.5% power ratio, their constraint is ~50x stronger than our rho=1% norm ratio (which equals 0.01% power ratio). If 0.5% norm ratio, then constraint magnitude is comparable but our 7-9% ASR differs substantially, likely due to differences in evaluation methodology (they may have used L-inf or different norm definitions), SNR evaluation approach (possibly the double-noising artifact), or model/dataset factors. As we demonstrate in Section 5, the choice of perturbation constraint formulation and evaluation methodology substantially affects reported attack effectiveness."

---

### 2.3 Related Work Section 2.4 (02_related_work.md)

**Current framing (Line 25-26):** "channel effects may attenuate adversarial perturbations (reducing attack effectiveness) or compound with them (amplifying classifier degradation)."

**Issue:** Our results clearly answer this question -- channel fading primarily degrades clean classification rather than amplifying the adversarial effect. The perturbation's marginal damage on top of channel degradation is small.

**Recommended revision:** This is actually well-positioned as a research question. In the results section, we should clearly answer it: "Channel impairments dominate adversarial perturbations. Rayleigh fading reduces clean accuracy by ~39 percentage points, while PGD attacks add only ~2-3 percentage points of additional degradation under fading. The channel provides 'accidental robustness' not by attenuating the perturbation, but by rendering the classification task so difficult that the adversarial gradient becomes decorrelated from the received signal."

---

### 2.4 Related Work Section 2.6 - Positioning (02_related_work.md)

**Current claim (Line 39):** "Unlike prior RF adversarial studies that evaluate attacks in isolation [9, 10] or under single channel conditions, we systematically vary both channel impairments and perturbation budgets across an SNR sweep."

**Issue:** This positioning is still valid and actually strengthened by our results. Our methodology correction (no double-noising) is itself a contribution -- we show that prior work's evaluation methodology may have inflated attack effectiveness.

**Recommended addition:** Add a point about evaluation methodology: "Critically, we identify and correct a common evaluation pitfall in RF adversarial robustness research: the double-noising of samples that already contain AWGN at their labeled SNR. This methodological correction substantially changes the quantitative picture of adversarial vulnerability."

---

### 2.5 Threat Model Section 3.3 (03_threat_model.md)

**Current framing (Line 41):** "This 'pre-channel' attack models an adversary who can manipulate the signal at or near the transmitter."

**Issue:** Our results show this pre-channel attack is quite weak under AWGN. This is actually an important finding -- the pre-channel attack model, while operationally relevant, faces the fundamental challenge that channel propagation decorrelates the adversarial perturbation from the gradient used to craft it.

**Recommended addition:** Add a note after the threat model description: "We note that this pre-channel attack model represents a conservative (for the defender) scenario in the AWGN case, where the perturbation passes through the channel alongside the signal. Under fading channels, however, the stochastic channel realization decorrelates the adversarial gradient from the received signal, potentially reducing attack effectiveness."

---

### 2.6 Results Section 5.2 (05_results.md)

**Current template:** Full of [X%] placeholders expecting high attack success rates.

**Issue:** The template's framing in Section 5.2.1 says "The gap between clean and robust accuracy is most pronounced at moderate-to-high SNR" -- this is correct for AWGN but the gap is actually quite small (~7-9% ASR). The narrative needs to frame this as a key finding rather than a weakness.

**Recommended framing for results:**

The central finding should be: **"Adversarial attacks under power-ratio constraints are substantially less effective than prior norm-based evaluations suggest, particularly when evaluation correctly accounts for native dataset SNR."**

Key results narrative:
- Under AWGN: PGD achieves modest 7-9% ASR at rho=1%, far below prior reports
- Channel impairments are the dominant factor: Rayleigh fading alone causes 39pp accuracy drop
- Fading provides "accidental robustness" through gradient decorrelation
- The attack-channel interaction reveals that for EW systems, channel hardening matters more than adversarial hardening

---

### 2.7 Discussion Section 6.1 (05_results.md)

**Current template (Line 73):** "The demonstrated vulnerability of baseline RF classifiers to low-power adversarial perturbations (attack success rates of [X%] at rho = 1%) indicates that adversarial robustness must be treated as a first-order system requirement."

**Issue:** With ASR of only 7-9% under AWGN, this claim needs significant revision. The argument shifts from "attacks are devastating, we must defend" to a more nuanced "attacks are modest under benign channels but the interaction with channel impairments creates a complex threat landscape."

**Recommended revision:** "Under AWGN-only conditions, PGD attacks at rho=1% achieve modest attack success rates (~7-9%), suggesting that power-ratio-constrained adversarial perturbations pose a limited threat when the channel is benign. However, under fading channel conditions where classifier accuracy is already degraded, the operational impact of even modest adversarial perturbations may be amplified -- not because the attack is more effective, but because the system is operating with smaller robustness margins. This finding reframes adversarial robustness from a standalone concern to an interaction effect that must be evaluated in the context of expected channel conditions."

---

## 3. What the Lit Review Gets Right (Keep These)

1. **Section 2.1** - The DL-for-AMC overview is solid and doesn't need changes.

2. **Section 2.2** - Adversarial ML foundations (Goodfellow, Madry, Carlini-Wagner) are correctly described and still relevant.

3. **Section 2.4** - The channel effects discussion correctly frames the open question that our results now answer. This is actually strengthened by our findings.

4. **Section 2.5** - EW/cognitive spectrum context is well-motivated and doesn't depend on specific attack effectiveness numbers.

5. **Section 2.6 positioning** - "Unlike prior work..." is still valid and strengthened. Our methodology contribution (correcting the double-noising issue) is an additional differentiator.

---

## 4. New Contributions to Emphasize

Based on the corrected results, the paper's contribution list should be updated:

### Original Contribution 1 (keep, strengthen):
"Physically plausible threat model" -- The power-ratio constraint is even more important now, as it's a key reason our ASR numbers differ from prior work. Emphasize: "Our power-ratio constraint produces markedly different (and more operationally realistic) results than norm-based constraints used in prior RF adversarial studies."

### Original Contribution 2 (keep, reframe):
"Channel-aware robustness evaluation" -- The finding that channel effects dominate adversarial effects is the paper's most important result. Reframe: "We demonstrate that under realistic channel impairments, propagation effects cause 4-5x more accuracy degradation than adversarial perturbations at operationally relevant power levels."

### New Contribution (add):
"Evaluation methodology correction" -- The identification and correction of the double-noising evaluation pitfall is itself a methodological contribution. "We identify a common evaluation artifact in RF adversarial robustness research where AWGN is added to samples that already contain noise at their labeled SNR, artificially inflating reported attack success rates."

### Original Contribution 3 (keep, update emphasis):
"Defense comparison" -- With low baseline ASR, the defense story shifts. Channel augmentation matters more for operational robustness (handling fading) than for adversarial hardening. Adversarial training's value may be more about improving general robustness margins than defeating specific attacks.

---

## 5. Literature Gaps to Update

### Research Gaps Matrix updates needed:

**Gap 1 (Channel-Realistic OTA Attacks):** Our results partially address this. We show channel effects dramatically reduce adversarial effectiveness. The remaining gap is OTA validation with real hardware.

**Gap 3 (Robustness-Accuracy Trade-offs):** This is now more nuanced. Since baseline ASR is only 7-9% under AWGN, the clean accuracy cost of adversarial training may not be justified for AWGN-only deployments. The trade-off analysis should focus on fading conditions where the operational margins are tighter.

**Gap 7 (EW-Specific Threat Models):** Our power-ratio constraint is a step toward EW-specific threat models. The finding that attacks are weak at operationally relevant power levels is directly relevant to EW threat assessment.

---

## 6. Action Items for Manuscript Revision

- [ ] Revise Introduction: temper "vulnerability" framing, emphasize channel interaction as key unknown
- [ ] Revise Related Work 2.3: add context to Flowers et al. claim about 90%->10% drops
- [ ] Revise Related Work 2.6: add evaluation methodology correction as positioning point
- [ ] Revise Threat Model 3.3: add note about pre-channel attack limitations under fading
- [ ] Rewrite Results 5.2: frame low ASR under AWGN as key finding, not as weakness
- [ ] Rewrite Discussion 6.1: shift from "attacks are devastating" to nuanced interaction story
- [ ] Add "evaluation methodology" as explicit contribution in Introduction
- [ ] Update Conclusion with corrected narrative
- [ ] Ensure all [X%] placeholders use corrected numbers when Colab results complete
- [ ] Add comparison table: our results vs. prior reported ASR, explaining discrepancies
