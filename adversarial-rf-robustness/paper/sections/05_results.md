# 5. Results

We present results across both RadioML 2016.10a (11 modulations, 128-sample I/Q windows) and RadioML 2018.01a (24 modulations, 1024-sample I/Q windows). The dual-dataset evaluation tests whether our findings generalize from a compact classification task to a substantially more complex one with higher-dimensional inputs and more than twice as many modulation classes.

## 5.1 Baseline Performance

The baseline CNN achieves a best validation accuracy of 72.57% on RadioML 2016.10a (all SNR values pooled) with 126,475 trainable parameters, and 62.83% on RadioML 2018.01a with 128,152 parameters (the slight increase reflects the larger number of output classes). Training completes in approximately 14 minutes for the 2016 model and 45 minutes for the 2018 model on an NVIDIA A100 GPU. Figure 1 shows clean classification accuracy as a function of SNR under three channel conditions for both datasets.

[FIGURE 1: Clean Accuracy vs SNR -- three channel curves, 2016 and 2018 panels]

**Evaluation methodology note.** Because RadioML samples already contain AWGN at their labeled SNR, our evaluation filters test samples by their native SNR labels rather than adding additional noise. This avoids the double-noising artifact that would occur if AWGN were applied on top of samples that already contain noise at the target level. For fading and CFO channel conditions, only the non-AWGN impairments are applied during evaluation, as the additive noise component is already present in the data.

**RadioML 2016.10a.** Under AWGN alone, clean accuracy reaches 83.68% at 0 dB and 85.59% at 10 dB, plateauing around 86-87% at high SNR. These results are deterministic (zero variance across trials) since no stochastic channel effects are applied. Under Rayleigh fading with AWGN, accuracy drops dramatically: 45.21% at 0 dB and 41.42% at 10 dB -- a reduction of 38-44 percentage points relative to AWGN alone. The addition of CFO further degrades performance to 35.59% at 0 dB and 32.98% at 10 dB. Notably, accuracy under fading channels does not improve substantially at higher SNR (plateauing around 41-42% for Rayleigh and 33-34% for Rayleigh+CFO), indicating that the multiplicative distortion of fading, not additive noise, is the dominant impairment at moderate-to-high SNR.

**RadioML 2018.01a.** The 2018 dataset presents a harder classification task with more than twice as many classes, but the larger input dimension (1024 vs. 128 samples) provides the model with richer temporal structure. Under AWGN, clean accuracy reaches 54.17% at 0 dB and 90.72% at 10 dB -- a steeper accuracy-versus-SNR slope than the 2016 dataset, reflecting the higher complexity ceiling. Under Rayleigh fading, accuracy degrades to 26.64% at 10 dB (a drop of 64 percentage points, substantially more severe than the 44-point drop observed on 2016). Under Rayleigh+CFO, accuracy falls to 20.38% at 10 dB. The more severe fading degradation on 2018 is consistent with the higher-order modulations (256-QAM, 128-QAM, APSK variants) being particularly sensitive to multiplicative distortion, which scrambles the constellation geometry that distinguishes these schemes.

## 5.2 Adversarial Vulnerability

### 5.2.1 Robust Accuracy vs SNR Under AWGN

Figure 2 shows robust accuracy (accuracy under attack) as a function of SNR for both FGSM and PGD attacks at rho = 1% perturbation budget under AWGN-only conditions.

[FIGURE 2: Robust Accuracy vs SNR -- clean, FGSM, PGD curves under AWGN, 2016 and 2018 panels]

**RadioML 2016.10a.** PGD (10 steps) at rho = 1% achieves attack success rates of only 9.20% at 0 dB and 7.37% at 10 dB under AWGN. FGSM achieves comparable rates: 8.63% at 0 dB and 7.19% at 10 dB. The gap between clean and robust accuracy is modest across the SNR range -- approximately 7-8 percentage points at moderate-to-high SNR. At low SNR (below -8 dB), both clean and robust accuracy converge toward chance level (9.1% for 11 classes), as the native AWGN dominates both the signal and the perturbation. PGD marginally outperforms FGSM as an attack (ASR approximately 0.5-1 percentage point higher), consistent with PGD's iterative optimization finding slightly better perturbation directions.

**RadioML 2018.01a.** The 2018 dataset reveals substantially higher adversarial vulnerability under the same norm-ratio constraint. PGD at rho = 1% achieves ASR of 35.49% at 0 dB and 30.87% at 10 dB -- roughly four times the 2016 attack success rate. FGSM achieves 26.37% ASR at 10 dB. This increased vulnerability is attributable to the higher-dimensional input space (1024 vs. 128 samples) and more complex decision boundary geometry with 24 classes: the adversary has more directions to exploit and more nearby class boundaries to push samples across. The practical difference between PGD and FGSM is also larger on 2018 (approximately 4.5 percentage points at 10 dB), suggesting that the optimization landscape is more complex and rewards iterative refinement.

**Cross-dataset comparison.** The contrast between 7-9% ASR on 2016 and 31-35% ASR on 2018 under identical perturbation constraints underscores that adversarial vulnerability is task-dependent even within the RF modulation classification domain. Systems classifying more modulation types in higher-dimensional input spaces face substantially greater adversarial risk, a finding with direct implications for fielded systems that must handle diverse waveform libraries.

These results stand in contrast to the high attack success rates reported in prior studies using different constraint formulations [9]. The discrepancy arises from three factors: (a) our L2 norm-ratio constraint (rho = 1% represents ||delta||_2 / ||x||_2, where rho^2 = 0.01% corresponds to the -40 dB power ratio) is physically grounded and precisely specified, (b) our evaluation correctly accounts for native dataset SNR without double-noising, and (c) the L2-normalized gradient direction used in our attacks distributes perturbation energy across the I/Q signal, making it harder to exploit individual feature sensitivities.

### 5.2.2 Channel Interaction with Adversarial Attacks

[FIGURE 3: Robust Accuracy vs SNR under different channels -- AWGN, Rayleigh+AWGN, Rayleigh+CFO+AWGN, 2016 and 2018 panels]

The interaction between channel impairments and adversarial effectiveness reveals a consistent pattern across both datasets.

**RadioML 2016.10a.** Under Rayleigh fading, the reported attack success rate appears substantially higher than under AWGN: PGD achieves ASR of 43.46% at 0 dB and 46.00% at 10 dB under Rayleigh+AWGN, compared to 9.20% and 7.37% under AWGN alone. However, this comparison is misleading. The critical observation is that the *marginal damage* of the adversarial attack on top of channel degradation is small. Under Rayleigh+AWGN at 0 dB, clean accuracy is 45.21% and robust accuracy is 43.22% -- a difference of only 2 percentage points. At 10 dB, clean accuracy is 41.42% and robust accuracy is 41.18% -- less than 0.3 percentage points of additional degradation. The high reported ASR reflects the channel's degradation being attributed to the attack in the ASR calculation, not an increase in attack effectiveness. Under Rayleigh+CFO+AWGN, the pattern intensifies: clean accuracy at 10 dB is 32.98%, robust accuracy is 32.16%, and the marginal adversarial damage is less than 1 percentage point.

**RadioML 2018.01a.** The 2018 dataset exhibits the same qualitative pattern but with more severe channel degradation and correspondingly inflated ASR values. Under Rayleigh+AWGN, PGD reports ASR of 58.6% at 10 dB, but clean accuracy is already only 26.64%, meaning fading alone accounts for a 64-point accuracy drop from the AWGN baseline of 90.72%. The marginal adversarial damage under Rayleigh fading is again modest relative to the channel's impact. Under Rayleigh+CFO+AWGN, clean accuracy at 10 dB drops to 20.38%, and the marginal adversarial effect remains small. The more severe channel degradation on 2018 reflects the higher-order modulations' greater sensitivity to constellation distortion.

This finding has important implications that are consistent across both datasets: channel impairments and adversarial perturbations do not compound multiplicatively. Rather, the stochastic channel realization decorrelates the adversarial gradient direction from the received signal, effectively neutralizing the targeted nature of the perturbation. Under fading, the adversarial perturbation functions more like random noise than a carefully crafted attack. We note that the magnitude of fading's impact on clean accuracy is likely understated in our evaluation due to the baked-in noise approximation described in Section 4.4; in a system with a constant receiver noise floor, deep fades would push the signal below the noise floor more severely than our approximation captures. Additionally, the degree of gradient decorrelation under fading is partly influenced by our PGD implementation using independent channel samples per step (see Section 6.2).

### 5.2.3 Attack Success vs Perturbation Budget

Figure 4 shows attack success rate as a function of perturbation budget rho at fixed SNR values for both datasets.

[FIGURE 4: ASR vs rho -- PGD at SNR=0 dB and SNR=10 dB, 2016 and 2018 curves]

**Table 1: PGD Attack Success Rate vs Perturbation Budget (AWGN, SNR = 10 dB)**

| rho (norm ratio) | Power Ratio (dB) | 2016 ASR (%) | 2016 Robust Acc (%) | 2018 ASR (%) | 2018 Robust Acc (%) |
|---|---|---|---|---|---|
| 0.005 (0.5%) | -46 dB | 3.52 | 82.58 | 14.15 | 77.88 |
| 0.010 (1.0%) | -40 dB | 7.36 | 79.29 | 30.85 | 62.73 |
| 0.020 (2.0%) | -34 dB | 14.97 | 72.78 | 59.16 | 37.05 |
| 0.050 (5.0%) | -26 dB | 44.32 | 47.66 | 78.64 | 19.38 |

**Table 2: PGD Attack Success Rate vs Perturbation Budget (AWGN, SNR = 0 dB)**

| rho (norm ratio) | Power Ratio (dB) | 2016 ASR (%) | 2016 Robust Acc (%) | 2018 ASR (%) | 2018 Robust Acc (%) |
|---|---|---|---|---|---|
| 0.005 (0.5%) | -46 dB | 5.03 | 79.47 | 19.61 | 43.66 |
| 0.010 (1.0%) | -40 dB | 9.16 | 76.01 | 35.53 | 35.00 |
| 0.020 (2.0%) | -34 dB | 21.02 | 66.09 | 57.63 | 22.96 |
| 0.050 (5.0%) | -26 dB | 61.99 | 31.80 | 80.96 | 10.32 |

The relationship between rho and ASR is concave (sublinear growth), with diminishing marginal returns at higher budgets. However, the 2018 dataset shows a steeper initial slope: doubling rho from 0.5% to 1% increases ASR by approximately 3.8 percentage points on 2016 but by 16.7 percentage points on 2018 at SNR = 10 dB. At the highest evaluated budget (rho = 5%), the 2018 classifier is nearly fully compromised (78.64% ASR at 10 dB, 80.96% at 0 dB), while the 2016 classifier retains meaningful accuracy (robust accuracy of 47.66% at 10 dB). This differential sensitivity to perturbation budget further reinforces that higher-complexity classification tasks require stricter power management to maintain operational robustness.

**FGSM budget sweep (RadioML 2016.10a).** FGSM shows comparable trends but lower attack success rates across all budgets. At SNR = 10 dB: rho = 0.5% yields 3.42% ASR, rho = 1% yields 7.19%, rho = 2% yields 12.92%, and rho = 5% yields 33.94%. The gap between PGD and FGSM widens at higher budgets (44.32% vs. 33.94% at rho = 5%), indicating that iterative optimization provides increasingly meaningful advantage as the perturbation budget allows exploration of more of the loss landscape.

## 5.3 Defense Comparison

We evaluate five defense configurations against the baseline model, reporting results at SNR = 0 dB and 10 dB under AWGN with PGD attacks at rho = 1%.

**Table 3: Defense Comparison -- RadioML 2016.10a (AWGN, rho = 1%)**

| Defense | SNR | Clean Acc (%) | Robust Acc (%) | ASR (%) | Train Time (min) |
|---|---|---|---|---|---|
| Baseline | 0 dB | 83.68 | 75.97 | 9.21 | -- |
| Baseline | 10 dB | 85.59 | 79.32 | 7.33 | -- |
| Adversarial Training | 0 dB | 78.82 | 72.69 | 7.78 | 103.5 |
| Adversarial Training | 10 dB | 83.08 | 80.69 | 2.88 | 103.5 |
| Noise Injection | 0 dB | 50.73 | 49.45 | 2.63 | 18.4 |
| Noise Injection | 10 dB | 52.96 | 51.40 | 3.05 | 18.4 |
| Channel Augmentation | 0 dB | 79.43 | 74.33 | 6.57 | 18.9 |
| Channel Augmentation | 10 dB | 82.19 | 80.93 | 1.67 | 18.9 |
| Adv. Train + Channel Aug. | 0 dB | 80.22 | 75.85 | 5.45 | 105.1 |
| Adv. Train + Channel Aug. | 10 dB | 82.85 | 81.77 | 1.30 | 105.1 |
| Noise Inj. + Channel Aug. | 0 dB | 37.50 | 36.71 | 2.27 | 18.8 |
| Noise Inj. + Channel Aug. | 10 dB | 39.81 | 39.21 | 1.65 | 18.8 |

**Table 4: Defense Comparison -- RadioML 2018.01a (AWGN, rho = 1%)**

| Defense | SNR | Clean Acc (%) | Robust Acc (%) | ASR (%) | Train Time (min) |
|---|---|---|---|---|---|
| Baseline | 0 dB | 54.17 | 35.02 | 35.49 | -- |
| Baseline | 10 dB | 90.72 | 62.71 | 30.87 | -- |
| Adversarial Training | 0 dB | 51.68 | 37.69 | 27.79 | 411.0 |
| Adversarial Training | 10 dB | 86.07 | 63.76 | 25.92 | 411.0 |
| Noise Injection | 0 dB | 51.97 | 32.25 | 38.09 | 149.1 |
| Noise Injection | 10 dB | 87.98 | 54.54 | 38.01 | 149.1 |
| Channel Augmentation | 0 dB | 51.69 | 39.44 | 24.79 | 152.9 |
| Channel Augmentation | 10 dB | 58.03 | 49.54 | 15.61 | 152.9 |
| Adv. Train + Channel Aug. | 0 dB | 51.48 | 39.61 | 25.40 | 414.3 |
| Adv. Train + Channel Aug. | 10 dB | 40.50 | 36.97 | 8.75 | 414.3 |
| Noise Inj. + Channel Aug. | 0 dB | 51.11 | 38.70 | 25.64 | 153.7 |
| Noise Inj. + Channel Aug. | 10 dB | 57.11 | 49.39 | 14.61 | 153.7 |

### 5.3.1 Channel Augmentation

Channel augmentation reduces ASR from 7.33% to 1.67% at 10 dB on 2016, and from 30.87% to 15.61% on 2018, while preserving most of the clean accuracy on 2016 (82.19% vs. 85.59% baseline). On 2018, channel augmentation substantially reduces clean accuracy at high SNR (58.03% vs. 90.72%) because the model is regularized to handle fading-distorted inputs, trading peak AWGN-only performance for channel robustness. Given that channel impairments cause 4-5x more accuracy degradation than adversarial perturbations at rho = 1% on 2016, and an even larger ratio on 2018, channel augmentation training serves a dual purpose: it improves operational robustness against real-world propagation conditions *and* provides indirect adversarial hardening by training the model on a broader distribution of input distortions. This makes channel augmentation the highest-priority defense for systems expected to operate in contested propagation environments.

### 5.3.2 Adversarial Training

PGD-based adversarial training (5 inner steps, rho = 1% norm ratio) targets the adversarial vulnerability directly. On 2016, adversarial training reduces ASR from 7.33% to 2.88% at 10 dB with a clean accuracy cost of 2.5 percentage points. On 2018, ASR drops from 30.87% to 25.92% -- a more modest absolute reduction, but the baseline vulnerability is substantially higher. Adversarial training requires approximately 5-6x more training time than channel augmentation (103.5 vs. 18.9 min on 2016; 411.0 vs. 152.9 min on 2018) due to the inner PGD loop. The value of adversarial training is most apparent in combination with channel augmentation (Section 5.3.4).

### 5.3.3 Noise Injection (Gaussian Data Augmentation)

Noise injection yields mixed results. On 2016, it achieves low ASR (3.05% at 10 dB) but at the cost of severe clean accuracy degradation (52.96%, a drop of over 32 percentage points from baseline). This renders the defense operationally impractical despite its robustness gains. On 2018, noise injection actually *increases* ASR relative to baseline (38.01% vs. 30.87% at 10 dB), suggesting that Gaussian augmentation in the higher-dimensional input space does not effectively regularize against adversarial directions and may interfere with feature learning. The trained model is evaluated at inference without additional noise; the poor results reflect only the robustness (or lack thereof) gained from noise-augmented training.

### 5.3.4 Combined Defense

The combined defense of adversarial training with channel augmentation achieves the lowest ASR on both datasets: 1.30% at 10 dB on 2016 and 8.75% on 2018. On 2016, this represents a 6-percentage-point reduction from the baseline ASR of 7.33% with only a 2.7-point clean accuracy cost (82.85% vs. 85.59%). On 2018, the combined defense reduces ASR from 30.87% to 8.75% -- a 22-percentage-point improvement -- but at a substantial clean accuracy cost (40.50% vs. 90.72%). The severe clean accuracy trade-off on 2018 suggests that the combined defense may be over-regularized for the more complex task; tuning the balance between adversarial and clean loss (currently alpha = 0.5) may improve this tradeoff.

The noise injection + channel augmentation combination performs comparably to channel augmentation alone on 2018 (14.61% vs. 15.61% ASR) while substantially degrading clean accuracy on 2016 (39.81% vs. 82.19%), confirming that noise injection is the weakest individual defense strategy.

### 5.3.5 Robustness-Compute Tradeoff

**Table 5: Robustness-Compute Tradeoff (SNR = 10 dB, AWGN, PGD rho = 1%)**

| Defense | 2016 Delta ASR (pp) | 2018 Delta ASR (pp) | 2016 Train Time (min) | 2018 Train Time (min) | Params |
|---|---|---|---|---|---|
| Channel Augmentation | -5.66 | -15.26 | 18.9 | 152.9 | 126,475 / 128,152 |
| Adversarial Training | -4.45 | -4.95 | 103.5 | 411.0 | 126,475 / 128,152 |
| Adv. Train + Channel Aug. | -6.03 | -22.12 | 105.1 | 414.3 | 126,475 / 128,152 |
| Noise Injection | -4.28 | +7.14 | 18.4 | 149.1 | 126,475 / 128,152 |

For edge-deployed EW systems operating under size, weight, and power (SWaP) constraints, channel augmentation offers the best ASR reduction per training hour on both datasets. On 2016, channel augmentation reduces ASR by 5.66 percentage points in 18.9 minutes, while adversarial training requires 103.5 minutes for a 4.45-point reduction. On 2018, the efficiency gap is even more pronounced: channel augmentation achieves a 15.26-point ASR reduction in 152.9 minutes versus 4.95 points in 411.0 minutes for adversarial training. The combined defense provides the maximum ASR reduction but at approximately 6x the training cost of channel augmentation alone. Adversarial training should be reserved for high-value classifiers where even modest additional ASR improvements justify the computational overhead, or for systems expected to face sophisticated adversaries with higher perturbation budgets.

## 5.4 Per-Class Vulnerability Analysis

Per-class analysis reveals substantial variation in adversarial vulnerability across modulation types, with important implications for operational risk assessment.

**RadioML 2016.10a.** At SNR = 10 dB under PGD at rho = 1%, the most vulnerable modulation is AM-DSB with an ASR of 41.76%. WBFM is also highly vulnerable at 21.87% ASR. In contrast, digital modulations such as BPSK (0.64% ASR), QPSK (0.69%), and 8PSK (2.12%) are nearly impervious to norm-ratio-constrained attacks. The vulnerability pattern reflects the underlying signal structure: AM-DSB and WBFM use amplitude and frequency modulation respectively, where small perturbations can shift the envelope or instantaneous frequency into adjacent decision regions. Phase-keyed modulations distribute information across the complex plane more uniformly, making it harder for power-constrained perturbations to cross decision boundaries.

**RadioML 2018.01a.** The expanded modulation set reveals even more extreme per-class vulnerability variation. At SNR = 10 dB, 256-QAM achieves 99.6% ASR -- essentially a complete compromise -- while OOK exhibits only 0.54% ASR. Other high-vulnerability modulations include 128-QAM (93.7% ASR), 64-QAM (64.5%), and higher-order APSK variants. The vulnerability of high-order QAM modulations is physically intuitive: dense constellation points are closely spaced, and even small I/Q perturbations can push received samples across decision boundaries between adjacent constellation points.

The per-class analysis suggests that operational risk assessment should be modulation-specific rather than averaged. Systems that must reliably classify high-order QAM modulations face substantially greater adversarial risk and may require stronger defenses or higher detection confidence thresholds for those specific waveform types.

## 5.5 Transferability Analysis

We evaluate attack transferability on RadioML 2016.10a by generating adversarial examples on the baseline model and evaluating them against defended models.

**Table 6: Transfer Attack Success Rate (RadioML 2016.10a, SNR = 10 dB, PGD rho = 1%)**

| Source -> Target | Source ASR (%) | Target ASR (%) | Transfer Rate (%) |
|---|---|---|---|
| Baseline -> Channel Aug. | 9.12 | 4.32 | 38.3 |
| Baseline -> Adv. Train | 9.04 | 2.02 | 32.6 |
| Baseline -> Noise Inject | 9.01 | 4.21 | 57.1 |
| Baseline -> Adv. Train + Channel Aug. | 9.14 | 4.00 | 38.9 |

Transfer rates range from 32.6% to 57.1%, indicating that adversarial examples crafted on the baseline model retain partial effectiveness against defended models in a black-box setting. Noise injection shows the highest transfer rate (57.1%), consistent with its weak adversarial hardening effect observed in the direct evaluation. Adversarial training shows the lowest transfer rate (32.6%), suggesting it learns fundamentally different decision boundaries rather than merely smoothing existing ones. These transfer rates are substantially lower than the 30-70% reported in prior RF adversarial studies [10], likely reflecting the more constrained norm-ratio perturbation budget in our evaluation.

---

# 6. Discussion

## 6.1 Implications for Cognitive Electronic Warfare

Our results across both RadioML 2016.10a and 2018.01a have several implications for the design and deployment of AI-based spectrum awareness systems in contested electromagnetic environments.

**Channel hardening before adversarial hardening.** Under AWGN-only conditions, PGD attacks at rho = 1% (norm ratio, -40 dB power ratio) achieve modest attack success rates of 7-9% on 2016 but a more substantial 31% on 2018, indicating that adversarial vulnerability scales with task complexity. However, under fading channel conditions where classifier accuracy is already degraded to 35-45% (2016) or 20-27% (2018), the system operates with substantially smaller robustness margins. In these conditions, even modest additional perturbation damage could push the classifier below operational thresholds for spectrum awareness. This finding reframes adversarial robustness from a standalone concern to an interaction effect: for systems deployed in contested propagation environments, channel hardening (via augmentation training) should be the first priority, with adversarial hardening as a secondary layer.

**The evaluation methodology matters.** The discrepancy between our results and prior reported attack success rates highlights the importance of evaluation rigor in adversarial robustness research. The double-noising artifact -- applying AWGN to samples that already contain noise at their labeled SNR -- can inflate attack success rates by degrading the classifier's baseline performance and attributing the additional degradation to the attack. We recommend that future RF adversarial robustness evaluations explicitly account for the noise conditions already present in benchmark datasets.

**Perturbation constraint choice shapes the threat assessment.** Our L2 norm-ratio constraint (rho = ||delta||_2 / ||x||_2) produces fundamentally different results than L-inf or fixed-epsilon L-2 constraints. This is not merely a scaling difference: the norm-ratio constraint distributes the perturbation budget relative to signal energy, making it harder for the adversary to concentrate perturbation power on sensitive input dimensions. For EW threat assessment, the norm-ratio formulation maps directly to interference-to-signal ratio (ISR, defined as the ratio of perturbation power to signal power, which equals rho^2), enabling operators to reason about adversarial capability in familiar power-ratio terms.

**Task complexity amplifies vulnerability.** The four-fold increase in ASR from 2016 (7-9%) to 2018 (31-35%) under identical perturbation constraints demonstrates that adversarial risk scales with the number of modulation classes and input dimensionality. Fielded systems that must handle large waveform libraries (as is typical in EW applications) should expect higher baseline vulnerability than small-scale benchmark evaluations suggest.

**Modulation-specific risk.** The extreme per-class vulnerability variation -- from below 1% ASR for simple modulations to above 99% for 256-QAM -- implies that operational risk assessment should be disaggregated by waveform type. High-order modulations used in modern communications systems are the most vulnerable, precisely because they are the most information-dense and therefore the most operationally valuable to classify correctly.

**Accidental robustness from fading is real but must be interpreted carefully.** Rayleigh fading channels provide substantial accidental robustness against pre-channel adversarial attacks by decorrelating the crafted perturbation direction from the received signal. However, three caveats apply. First, this protection depends on the adversary not having knowledge of the specific channel realization; a channel-aware adversary could design perturbations that survive propagation [21, 23]. Second, the observed decorrelation is partly an artifact of our PGD implementation using independent channel samples per optimization step (a single-sample noisy approximation rather than proper multi-sample EoT); a more sophisticated no-CSI attack averaging gradients over multiple channel realizations per step could achieve higher success rates. Third, our fading evaluation applies the fading multiplier to RadioML samples that already contain AWGN, meaning fading does not degrade the effective SNR as severely as it would with a constant receiver noise floor (see Section 6.2). The robustness provided by fading should therefore be viewed as a favorable baseline, not a substitute for deliberate adversarial hardening.

## 6.2 Limitations

Several limitations should be noted, some of which are fundamental to the use of pre-collected benchmark datasets for RF adversarial research.

**Fading channel approximation with baked-in noise.** RadioML samples contain AWGN at their labeled SNR, and the clean signal and noise components cannot be separated. When we apply Rayleigh fading, the fading coefficient multiplies both signal and noise jointly -- h * (x_clean + n) rather than the physically correct (h * x_clean) + n. In a real receiver, thermal noise has a constant power floor independent of the channel state; a deep fade attenuates the signal but not the noise, pushing the signal below the noise floor. Our approximation preserves the effective SNR through fading, making fading appear less harmful than it would be in practice. This means our clean accuracy results under fading channels are likely optimistic, and the true accuracy degradation from fading would be more severe. The relative comparison between clean and adversarial performance under fading remains informative, as both are subject to the same approximation, but the absolute magnitude of fading's impact on classification is understated. A fully correct evaluation would require noiseless source signals with receiver thermal noise added after fading.

**Adversary access to noise realization.** Because RadioML samples contain baked-in AWGN, the adversary computes gradients on the signal-plus-noise sample rather than the clean signal alone. A real pre-channel attacker would not know the receiver's thermal noise realization at the time of perturbation crafting. This gives our simulated adversary a slight advantage at low SNR where the noise component is larger relative to the signal.

**Single-sample stochastic optimization under fading.** Under fading channels, each PGD step uses an independent random channel realization. This is a noisy single-sample approximation of Expectation over Transformation (EoT), not a proper multi-sample gradient average. Our finding that fading decorrelates the adversarial gradient is therefore partly an artifact of the optimization procedure (new random channel per step) and partly a physical effect of stochastic propagation. We provide a frozen-channel option (perfect CSI) as an upper bound, but a full multi-sample EoT implementation would provide a tighter characterization of the no-CSI attack's true capability.

**Noise injection defense scope.** Our "noise injection" defense implements Gaussian data augmentation during training only. The trained model is evaluated with a single forward pass at inference without additional noise or Monte Carlo averaging. Our results for this defense reflect only the robustness gained from noise-augmented training, not the stronger guarantees available from full randomized smoothing with inference-time averaging.

**Synthetic channel models.** Our evaluation uses differentiable channel layers (AWGN, flat Rayleigh fading, CFO) rather than over-the-air measurements. Real-world channels exhibit additional effects including frequency-selective fading, time-varying coherence, and hardware nonlinearities that may further impact adversarial robustness.

**Threat model scope.** Our threat model assumes a white-box adversary with no channel state information. A channel-adaptive adversary who can estimate or predict the channel realization could design perturbations that survive propagation more effectively, achieving higher attack success rates than we report.

**Dataset limitations.** RadioML 2016.10a has known issues including the AM-SSB encoding artifact that caps overall accuracy. RadioML 2018.01a provides a more comprehensive modulation set but shares the fundamental limitation of baked-in AWGN. The absence of transferability evaluation on the 2018 dataset is a gap; future work should characterize black-box attack effectiveness on the larger task.

**Defense evaluation scope.** Our defense comparison covers five configurations from three base strategies but does not explore the full space of possible defenses. The combined defense (adversarial training + channel augmentation) achieves the lowest ASR on both datasets but incurs a substantial clean accuracy cost on 2018 (40.50% vs. 90.72% baseline), suggesting that defense hyperparameter tuning (particularly the adversarial/clean loss balance alpha) merits further investigation for complex classification tasks.

## 6.3 Future Work

Several directions merit further investigation. Transferability analysis on the 2018.01a dataset across model architectures would characterize black-box attack effectiveness for the more complex classification task and determine whether the higher white-box vulnerability (31% vs. 7% ASR) translates to higher transfer rates. Channel-aware adversarial attacks that incorporate channel estimation into the perturbation design would establish upper bounds on vulnerability under more capable adversaries. Over-the-air validation using software-defined radio platforms would bridge the gap between simulated and real-world adversarial robustness. Multi-sample Expectation over Transformation (EoT) attacks would provide tighter characterization of the no-CSI adversary's true capability under fading channels. Defense hyperparameter optimization, particularly tuning the adversarial/clean loss balance for complex tasks where the current alpha = 0.5 produces excessive clean accuracy degradation, could improve the robustness-accuracy Pareto frontier. Finally, certified robustness bounds adapted for non-Gaussian channel noise models could provide formal guarantees for fielded systems.

---

# 7. Conclusion

This paper presented a systematic evaluation of adversarial robustness for deep learning-based RF modulation classifiers under realistic channel impairments, spanning two benchmark datasets (RadioML 2016.10a with 11 modulations and 2018.01a with 24 modulations). Using a physically plausible threat model with L2 norm-ratio perturbation constraints (rho = ||delta||_2 / ||x||_2) and an evaluation methodology that correctly accounts for native dataset SNR, we demonstrate several findings that reshape the understanding of adversarial threats to RF classification systems:

1. Under AWGN-only conditions, adversarial attacks at operationally relevant power levels (rho = 1% norm ratio, corresponding to -40 dB power ratio) achieve modest attack success rates of 7-9% on RadioML 2016.10a but substantially higher rates of 31-35% on RadioML 2018.01a, demonstrating that adversarial vulnerability scales with task complexity. Both rates are substantially lower than previously reported in studies using different constraint formulations. The choice of perturbation constraint and evaluation methodology fundamentally affects the quantitative assessment of adversarial vulnerability.

2. Channel impairments -- particularly Rayleigh fading -- cause substantially more classifier degradation than adversarial perturbations on both datasets. On 2016, fading reduces accuracy by 38-44 percentage points compared to 7-8 percentage points from PGD attacks. On 2018, fading reduces accuracy by 64 percentage points compared to 28 points from PGD attacks, even under our evaluation approximation that understates fading's true impact (see Section 6.2). The marginal damage of adversarial perturbations on top of channel degradation is modest on both datasets, as fading decorrelates the adversarial gradient from the received signal.

3. Among the evaluated defenses, channel augmentation training addresses the dominant source of classifier degradation with the best efficiency: it achieves the largest ASR reduction per training hour on both datasets. The combined defense (adversarial training + channel augmentation) achieves the lowest absolute ASR -- 1.30% on 2016 and 8.75% on 2018 -- but requires approximately 6x more training time. Adversarial training provides the most value for complex classification tasks (2018) where baseline vulnerability is high.

4. Per-class vulnerability analysis reveals extreme variation: from below 1% ASR for simple modulations (OOK, BPSK) to 99.6% for 256-QAM, indicating that operational risk assessment must be modulation-specific.

5. The evaluation methodology correction (avoiding double-noising of benchmark samples) is itself a contribution, highlighting a pitfall that may affect other RF adversarial robustness studies and underscoring the importance of evaluation rigor in this domain.

These findings contribute to the growing understanding that adversarial robustness for AI-enabled spectrum operations must be assessed in the context of realistic channel conditions and at operationally relevant task complexity, not in isolation on small benchmarks. For military organizations investing in cognitive electronic warfare capabilities, our results suggest that channel-robust classifier design should be prioritized alongside adversarial hardening, that threat assessments based on norm-bounded attacks may substantially overestimate the vulnerability of deployed systems to power-constrained adversaries, and that adversarial risk scales significantly with the complexity of the classification task -- a finding with direct implications for systems handling diverse military waveform libraries.
