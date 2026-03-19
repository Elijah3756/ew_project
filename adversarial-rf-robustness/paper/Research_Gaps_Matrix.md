# Research Gaps and Opportunities Matrix
## Adversarial Robustness of AI-Based RF Signal Classification Under Channel Impairments

**For CDR 2026 EW Special Issue**
**Strategic Planning Document**

---

## Executive Summary

This document identifies seven critical research gaps at the intersection of adversarial machine learning (AML), RF signal processing, and channel impairments. Each gap represents a potential research contribution that advances both theoretical understanding and practical EW/spectrum operations. The matrix below prioritizes gaps by impact, novelty, feasibility, and alignment with Army/DoD priorities.

---

## Gap Analysis Matrix

### GAP 1: Channel-Realistic Adversarial Attacks on RF Classifiers

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Most adversarial attacks on RF classifiers are validated in digital domain only (e.g., FGSM, PGD on RadioML dataset). Over-the-air (OTA) validation is limited. Channel effects (multipath fading, phase noise, CFO) corrupt adversarial perturbations in unpredictable ways. Gap: systematically characterize how channel impairments interact with adversarial attacks. |
| **Current State** | • Reference 21: Channel-aware attacks proposed but focused on WiFi CSI, not modulation recognition<br>• Reference 17: Over-the-air RF attacks exist but detection-focused, not attack generation<br>• Reference 23: OTA attacks for modulation classifiers exist but limited scope (frequency-selective attacks not fully characterized)<br>• No unified framework for predicting OTA attack success from digital-domain perturbations |
| **Why It Matters** | • Digital adversarial examples may not survive propagation; OTA attacks are the realistic threat for military RF systems<br>• Understanding channel-attack interaction is critical for threat assessment and defense design<br>• Certified defenses must account for both adversarial perturbations AND channel effects simultaneously |
| **Research Approach** | **Phase A (Characterization):**<br>1. Empirically characterize how AWGN, Rayleigh fading, CFO, phase noise degrade/enhance adversarial examples<br>2. Develop parametric model: OTA attack success = f(digital perturbation, SNR, channel type, modulation class)<br>3. Identify "robust" adversarial perturbations that survive propagation<br><br>**Phase B (Design):**<br>4. Design channel-aware adversarial attacks (extend Reference 21 methodology from WiFi to modulation recognition)<br>5. Develop detection metrics for OTA adversarial signals (extend PAPR/Softmax tests in Reference 17)<br>6. Establish metrics for "physical realism" of RF adversarial examples (detectability, power efficiency, spectral mask compliance)<br><br>**Phase C (Validation):**<br>7. Conduct over-the-air testbed experiments using GNU Radio, USRP hardware, or RF chamber<br>8. Compare digital-only vs. OTA attack success rates across modulation classes and channel conditions |
| **Novel Contributions** | • First systematic characterization of channel-adversarial interaction for modulation classifiers<br>• Channel-realistic adversarial attack generation methodology<br>• Metrics and testbed protocols for OTA adversarial robustness evaluation |
| **Experimental Artifacts** | • Augmented RadioML dataset with channel simulation parameters (fading, CFO, phase noise)<br>• OTA testbed setup and measurement protocols<br>• Attack success prediction model with quantified uncertainties |
| **Deliverables** | • 1–2 journal papers (IEEE TIFS, IEEE TWC, or IEEE S&P)<br>• OTA testbed design documentation<br>• Open-source channel-aware attack code |
| **Feasibility** | **Medium-High:** Requires OTA testbed (USRP ~$5k, RF chamber access, or outdoor environment); data collection labor-intensive but straightforward; modeling requires channel estimation expertise. |
| **Novelty** | **High:** Intersection of digital adversarial ML and physical-layer RF propagation is underexplored for modulation classifiers |
| **DoD Alignment** | **Very High:** Directly addresses threat assessment for CEW systems; informs EW attack planning and RF security |
| **Timeline** | 12–18 months for full characterization + OTA validation |
| **Estimated Resources** | 1–2 PhD students, 1 senior researcher; hardware (~$10k), testbed setup (3–6 months) |

---

### GAP 2: Certified Robustness for RF Classifiers Under Channel Variability

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Certified robustness techniques (randomized smoothing, Reference 13) exist for image classification. However, they assume Gaussian noise models. RF channels exhibit structured impairments (Rayleigh fading, not just AWGN). Certified radius computed under Gaussian assumption may be pessimistic or overoptimistic for real channel conditions. Gap: derive certified robustness bounds that account for realistic RF channel noise models. |
| **Current State** | • Reference 13–14: Randomized smoothing with L₂, L₁, L∞ norms well-developed<br>• Reference 22: Channel augmentation + randomized smoothing proposed but lacks formal certification analysis<br>• No prior work formalizing Rayleigh/Rician fading noise models within certified robustness framework<br>• Certified defenses in RF domain do not exist |
| **Why It Matters** | • Certified robustness provides formal guarantees; crucial for critical EW applications (spectrum operations, threat classification)<br>• Current Gaussian-based certification may underestimate true robustness (if channel noise is "friendly") or overestimate (if real impairments worse)<br>• Gap between theoretical robustness and practical EW performance |
| **Research Approach** | **Phase A (Theory):**<br>1. Extend Cohen et al. (Reference 13) certification framework to non-Gaussian noise models<br>2. Characterize certified radius r under Rayleigh fading: $r_{Rayleigh}(p_A, p_B, \sigma)$ with fading parameter<br>3. Derive certification for multiplicative noise (CFO-induced frequency shifts) in addition to additive AWGN<br>4. Account for phase noise and IQ imbalance in formal certification<br><br>**Phase B (Algorithms):**<br>5. Develop efficient sampling procedures for Rayleigh/Rician fading noise (less tractable than Gaussian)<br>6. Propose adaptive certification: certificate radius varies with SNR and modulation class<br>7. Design hybrid certification: combine channel smoothing with adversarial smoothing (dual robustness layers)<br><br>**Phase C (Empirical Validation):**<br>8. Empirically validate theoretical certifications on RadioML dataset augmented with realistic channel impairments<br>9. Compare certified vs. empirical robustness under PGD/C&W attacks in various SNR/channel regimes<br>10. Quantify "certification tightness" (gap between certified bound and true robustness) |
| **Novel Contributions** | • First formal certification framework for RF classifiers under realistic channel noise<br>• Rayleigh/Rician fading noise models adapted to randomized smoothing<br>• Multiplicative noise (CFO, frequency-domain effects) characterization in certification<br>• Adaptive certification methodology for SNR-dependent systems |
| **Experimental Artifacts** | • Extended RadioML dataset with ground-truth channel parameters<br>• Formal proof of certification bounds under fading assumptions<br>• Sampling algorithms for non-Gaussian noise models<br>• Empirical validation experiments with ROC curves and certified radius estimates |
| **Deliverables** | • 1–2 theoretical papers (IEEE TIFS, IEEE IT, or IEEE TAC)<br>• Implementation code for certified RF classifiers<br>• Certification verification toolkit for evaluating other RF defenses |
| **Feasibility** | **Medium:** Requires solid mathematical background (probability theory, information theory, noise modeling); no hardware required; data collection moderate. |
| **Novelty** | **Very High:** Certified robustness adapted to RF domain with non-Gaussian noise is novel; theoretical contribution significant. |
| **DoD Alignment** | **Very High:** Formal robustness guarantees critical for military acquisition and operational deployment; supports "AI Assurance" initiatives |
| **Timeline** | 12–18 months (theoretical development + empirical validation) |
| **Estimated Resources** | 1 PhD student (theory-heavy), 1 researcher (implementation); minimal hardware |

---

### GAP 3: Robustness-Accuracy Trade-offs in RF Classifiers at Low SNR

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Adversarial training (Reference 15) is the empirical defense standard but incurs accuracy loss on clean data. RF classification at low SNR is already challenging (e.g., 86% accuracy at −2 dB per Reference 19). Adversarial training may further degrade performance. Gap: characterize robustness-accuracy trade-offs specific to RF domain and develop techniques preserving accuracy while improving robustness. |
| **Current State** | • References 15–16: General ML literature shows robustness-accuracy trade-off; not studied for RF<br>• Reference 19: RF classifiers achieve good accuracy at high SNR but struggle at low SNR<br>• Data augmentation (Reference 16) improves robustness; unclear if channel-based augmentation (AWGN, fading) helps or hurts accuracy at low SNR<br>• No SNR-adaptive defense strategies for RF |
| **Why It Matters** | • Low SNR is common in military RF scenarios (long-distance propagation, covert operations)<br>• If adversarial training reduces accuracy by 10% at low SNR, operational utility may be compromised<br>• EW systems must balance robustness against adversarial attacks vs. detection of genuine low-SNR signals<br>• Gap between robustness-focused ML and practical EW requirements |
| **Research Approach** | **Phase A (Characterization):**<br>1. Measure robustness-accuracy curves for CNN, CLDNN, Transformer RF classifiers across SNR range (−5 to +20 dB)<br>2. Compare three defense strategies: (a) adversarial training (PGD-AT), (b) channel augmentation, (c) hybrid<br>3. Quantify trade-offs: ΔAccuracy vs. ΔRobustness for each defense<br>4. Identify "sweet spot" SNR/modulation classes where trade-off is acceptable<br><br>**Phase B (Mitigation):**<br>5. Develop SNR-adaptive defenses: classifier confidence used to trigger defense (aggressive at high SNR, conservative at low SNR)<br>6. Propose ensemble methods: robust model for threat assessment, standard model for low-SNR detection; combine predictions<br>7. Design curriculum learning: train on high SNR first (where robustness easier), gradually lower SNR<br>8. Explore architectural innovations: models explicitly optimizing robustness-accuracy (multi-objective learning)<br><br>**Phase C (Optimization):**<br>9. Formulate bi-objective optimization: maximize accuracy AND robustness simultaneously<br>10. Use Pareto frontiers to guide defense selection for operational scenarios<br>11. Develop EW-specific metrics beyond accuracy and robustness (e.g., false alarm rate, detection latency) |
| **Novel Contributions** | • First systematic characterization of robustness-accuracy trade-offs in RF domain<br>• SNR-adaptive defense mechanisms<br>• Ensemble methods balancing robustness and low-SNR detection<br>• Multi-objective optimization framework for RF defense design |
| **Experimental Artifacts** | • Robustness-accuracy curves across SNR ranges for multiple architectures<br>• Trained models (PGD-AT, channel augmentation, hybrid)<br>• SNR-adaptive decision rules<br>• Ensemble prediction algorithms |
| **Deliverables** | • 1–2 papers (IEEE TIFS, IEEE VT, or IEEE TAI)<br>• Benchmark results and decision trees for defense selection<br>• Trained model zoo (adversarially trained RF classifiers for public use) |
| **Feasibility** | **Medium:** Extensive experiments on RadioML (data already available); computational cost moderate (GPU training ~week); no hardware required. |
| **Novelty** | **High:** Trade-off analysis specific to RF domain with practical EW twist (SNR adaptation). |
| **DoD Alignment** | **High:** Practical concern for military RF systems; informs defense deployment guidelines. |
| **Timeline** | 12 months (experimental characterization + algorithm development) |
| **Estimated Resources** | 1–2 PhD students (experimental), GPU compute resources (~$5k), no hardware |

---

### GAP 4: Adversarial Attacks on Transformer-Based RF Classifiers

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Transformer-based RF classifiers (Reference 4: TLDNN) are state-of-the-art, achieving superior performance on RadioML datasets. However, adversarial robustness is unexplored. Attention mechanisms enable new capabilities (global correlation modeling) but may introduce new vulnerabilities. Gap: systematically evaluate adversarial susceptibility of transformer RF classifiers and propose attention-aware defenses. |
| **Current State** | • Reference 4: TLDNN (Transformer-LSTM) achieves state-of-the-art accuracy on RadioML<br>• References 6–8: FGSM, PGD, C&W attacks developed for CNNs; unclear applicability to transformers<br>• Transformer adversarial robustness literature (vision transformers, NLP) shows mixed results: some evidence of robustness, some vulnerability to attention-exploitation attacks<br>• No prior work on adversarial robustness of transformer RF classifiers |
| **Why It Matters** | • Transformers are becoming standard RF classifiers; understanding vulnerabilities is urgent<br>• Attention mechanisms may provide unexpected robustness (self-attention smooths adversarial perturbations) or create new vulnerabilities (attention maps exploitable for targeted attacks)<br>• Attention-aware attack design could render standard defenses ineffective<br>• Gap: frontier RF classifiers lack robustness evaluation |
| **Research Approach** | **Phase A (Attack Development):**<br>1. Adapt FGSM, PGD, C&W attacks to transformer architectures (attention-aware gradient computation)<br>2. Design attention-exploitation attacks: craft perturbations targeting specific attention heads (force misalignment)<br>3. Evaluate transferability: do transformer-targeted attacks transfer to CNNs and vice versa<br>4. Compare vulnerability: transformer vs. CNN vs. CLDNN under same attack budgets<br><br>**Phase B (Analysis):**<br>5. Visualize attention mechanisms under adversarial conditions (saliency maps, attention rollout)<br>6. Identify which attention heads are vulnerable vs. robust<br>7. Analyze "attention robustness": does self-attention naturally smooth adversarial perturbations<br>8. Study gradient flow: how gradients propagate through attention layers in adversarial setting<br><br>**Phase C (Defenses):**<br>9. Propose attention-aware adversarial training: augment PGD-AT with attention-specific perturbations<br>10. Design defensive mechanisms targeting fragile attention heads (e.g., attention head pruning)<br>11. Develop certified robustness for transformers (extend randomized smoothing analysis)<br>12. Propose efficient attention mechanisms with built-in robustness |
| **Novel Contributions** | • First comprehensive adversarial robustness study of transformer RF classifiers<br>• Attention-exploitation attack methodology<br>• Attention-aware adversarial training and defense techniques<br>• Mechanistic understanding of transformer robustness properties |
| **Experimental Artifacts** | • Transformer RF classifiers (TLDNN variants) with varying attention architectures<br>• Attention-aware attack implementations (custom PyTorch code)<br>• Visualization tools for attention behavior under adversarial conditions<br>• Trained models with varying defense strategies |
| **Deliverables** | • 1–2 papers (IEEE TIFS, ICML, or NeurIPS track)<br>• Attack code and visualization tools (open-source)<br>• Benchmark results and recommendations for transformer RF deployment |
| **Feasibility** | **Medium:** Requires deep learning expertise (transformers, attention mechanisms); computational cost moderate to high (training large transformers, attack computation); data available (RadioML). |
| **Novelty** | **Very High:** First to study transformer RF classifier adversarial robustness; mechanistic insights novel. |
| **DoD Alignment** | **High:** Transformers are being deployed; robustness critical before operational use. |
| **Timeline** | 12–18 months (attack development + comprehensive evaluation + defenses) |
| **Estimated Resources** | 1–2 PhD students (deep learning), GPU compute (~$10k), no specialized hardware |

---

### GAP 5: Physical Layer Signal Processing Perspective on Adversarial Defenses

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Most adversarial RF ML literature is machine-learning-focused (References 6–17). Traditional signal processing (equalization, filtering, channel coding) is largely absent from adversarial defense discussions. Gap: bridge ML and signal processing communities; explore whether classical RX-side signal processing techniques can mitigate adversarial perturbations. |
| **Current State** | • Reference 17: PAPR detection is signal-processing-inspired but limited<br>• Reference 22: Channel augmentation leverages channel knowledge but not advanced signal processing<br>• Classical signal processing (equalization, MIMO) proven effective for channel mitigation; unclear if helpful for adversarial robustness<br>• No cross-layer (PHY + ML) defense architectures |
| **Why It Matters** | • Military RF systems already employ sophisticated signal processing (equalizers, MIMO receivers); leveraging this for adversarial defense is natural synergy<br>• Signal processing operates before ML (at PHY layer); pre-processing can reject adversarial perturbations before reaching classifier<br>• Adversarial robustness + signal processing may be more efficient than pure ML defenses (fewer parameters, lower latency)<br>• Gap: missed opportunity for integrating two complementary disciplines |
| **Research Approach** | **Phase A (Signal Processing for Adversarial Rejection):**<br>1. Study how adaptive equalization (Wiener, RLS filters) affect adversarial examples<br>2. Evaluate channel coding + adversarial robustness: does error correction code help<br>3. Assess MIMO/antenna diversity: do multiple antennas naturally mitigate adversarial perturbations<br>4. Explore waveform design: are some waveforms (OFDM, single-carrier) inherently more robust than others<br><br>**Phase B (Cross-Layer Architecture Design):**<br>5. Propose three-tier defense: (a) adaptive signal processing, (b) supervised learning (classifier), (c) post-classification validation<br>6. Design handshake: signal processing provides confidence estimate to classifier; classifier adjusts sensitivity<br>7. Develop decision rules: when to trust signal processing vs. ML<br><br>**Phase C (Implementation & Validation):**<br>8. Prototype cross-layer receiver on USRP or RF testbed<br>9. Measure joint PHY+ML robustness under adversarial attacks (synthetic and OTA)<br>10. Compare pure ML defense vs. cross-layer defense in latency, complexity, accuracy |
| **Novel Contributions** | • First systematic study of signal processing for adversarial defense in RF domain<br>• Cross-layer (PHY + ML) defense architectures<br>• Waveform selection guidelines for adversarial robustness<br>• Practical USRP/testbed implementation demonstrating feasibility |
| **Experimental Artifacts** | • Adaptive equalization + classifier pipeline<br>• MIMO receiver with adversarial robustness evaluation<br>• USRP testbed implementation<br>• Latency/accuracy/complexity comparisons |
| **Deliverables** | • 1–2 papers (IEEE TIFS, IEEE TVT, or IEEE Communications Magazine)<br>• USRP testbed design documentation<br>• Cross-layer receiver code (GNU Radio module or USRP-compatible) |
| **Feasibility** | **Medium:** Requires signal processing expertise (equalization theory, MIMO); testbed setup moderate; data collection labor-intensive. |
| **Novelty** | **High:** Interdisciplinary approach (signal processing + ML) is novel; cross-layer architecture design underexplored. |
| **DoD Alignment** | **Very High:** Military RF systems already use signal processing; integration with ML robustness is natural and practical. |
| **Timeline** | 12–18 months (theory + testbed implementation) |
| **Estimated Resources** | 1–2 PhD students, 1 signal processing expert (consulting); USRP hardware (~$5k) |

---

### GAP 6: Adversarial Robustness in Distributed Spectrum Sensing Networks

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Cognitive radio networks use distributed spectrum sensing (References 27–28) where multiple sensors report spectrum occupancy. Consensus/fusion algorithms combine reports to create spectrum map. Gap: adversarial perturbations on RF signals can mislead individual sensors; no analysis of adversarial robustness for distributed sensing under consensus constraints. |
| **Current State** | • Reference 27–28: Spectrum sensing with ML and defense against SSDF (data falsification) attacks<br>• SSDF attacks assume adversary controls sensor report (software-level); not physical-layer adversarial perturbations<br>• Byzantine-robust consensus (distributed systems literature) exists but not applied to RF sensing<br>• No game-theoretic analysis of adversarial spectrum sensing with distributed sensors |
| **Why It Matters** | • In contested spectrum environments (military scenarios), adversary controls TX (can inject adversarial signals), not sensor software<br>• Distributed sensing more robust than centralized; but requires understanding adversarial propagation in distributed settings<br>• Game-theoretic perspective: sensing network is strategic; adversary's best response to distributed defense is informative<br>• Gap: vulnerability assessment for cognitive radio networks in adversarial EW scenario |
| **Research Approach** | **Phase A (Adversarial Model & Analysis):**<br>1. Formulate adversarial spectrum sensing game: k sensors, 1 adversary with power budget<br>2. Analyze adversary's optimal strategy: which sensors to target, what signals to inject<br>3. Characterize single-sensor vulnerability: how does adversarial attack fool individual sensors<br>4. Study consensus robustness: can Byzantine consensus (majority voting, weighted averaging) reject adversarial signals<br><br>**Phase B (Defense Mechanisms):**<br>5. Adapt Byzantine-robust consensus algorithms to spectrum sensing<br>6. Propose reputation-based trust for sensors (downweight sensors that disagree with majority)<br>7. Design adaptive sensing: allocate more samples to sensors with low confidence<br>8. Develop detection mechanisms: identify compromised sensors (signal anomaly detection)<br><br>**Phase C (Game-Theoretic Optimization):**<br>9. Formulate min-max game: sensing network minimizes worst-case spectrum map error; adversary maximizes<br>10. Compute game-theoretic equilibria (Stackelberg, Nash)<br>11. Design optimal sensor placement / resource allocation for robustness<br>12. Evaluate robustness-detection trade-off (sensing network wants high detection probability AND low false alarm) |
| **Novel Contributions** | • First adversarial robustness analysis for distributed RF spectrum sensing<br>• Game-theoretic framework for adversarial spectrum sensing<br>• Byzantine-robust consensus adapted to RF domain<br>• Optimal sensor network design for adversarial resilience |
| **Experimental Artifacts** | • Distributed spectrum sensing simulation (ns-3 or custom Python)<br>• Byzantine consensus implementations<br>• Game-theoretic solver (computation of equilibria)<br>• Synthetic adversarial attack scenarios |
| **Deliverables** | • 1–2 papers (IEEE TIFS, IEEE Transactions on Network Science, or IEEE SP)<br>• Open-source distributed sensing simulator with adversarial attack module<br>• Game-theoretic analysis toolkit |
| **Feasibility** | **Medium:** Requires game theory expertise (strategic thinking, equilibrium computation); no hardware required; simulation-based; computational complexity for large networks. |
| **Novelty** | **Very High:** Game-theoretic + distributed robustness perspective on RF adversarial attacks is novel. |
| **DoD Alignment** | **Very High:** Distributed sensing is standard in military spectrum operations; adversarial resilience critical. |
| **Timeline** | 12 months (game-theoretic analysis + simulation) |
| **Estimated Resources** | 1 PhD student (game theory + RF background), compute resources minimal |

---

### GAP 7: EW-Specific Threat Models and Defense Architectures

| Aspect | Details |
|--------|---------|
| **Problem Statement** | Generic adversarial ML uses norm-bounded perturbations (ℓ₂, ℓ∞, ℓ₀). EW threat models differ: power budgets, spectral occupancy regulations, dwell time, detectability constraints. Gap: formalize EW-specific threat models; develop attacks and defenses tailored to military RF scenarios. |
| **Current State** | • References 6–12: Generic FGSM, PGD, C&W attacks use ℓ₂/ℓ∞ norms; not EW-informed<br>• Reference 11: Frequency-selective stealthy attacks begin to address detectability<br>• Reference 23: WiAdv achieves "stealth" via CSI mimicry; EW analogue undefined<br>• No EW-specific threat model standard; military RF scenarios have unique constraints (spectral mask, PAPR, modulation-in-band constraints) |
| **Why It Matters** | • EW attacks are constrained: jammer power limited, must avoid detection, must respect spectral mask (avoid adjacent-channel interference)<br>• Generic adversarial budget may not map to feasible EW jamming/spoofing attack<br>• Conversely, relaxed constraints in EW (e.g., broad spectrum) may enable attacks not possible under ℓ∞ norm<br>• Gap: credibility gap between academic adversarial examples and practical EW threat assessment |
| **Research Approach** | **Phase A (Threat Model Definition):**<br>1. Engage EW experts (military labs, defense contractors) to define EW-specific perturbation constraints<br>2. Formalize as optimization constraints: (a) power budget P, (b) spectral mask M(f), (c) PAPR limit, (d) duration/dwell time, (e) detectability threshold<br>3. Map EW constraints to abstract perturbation norms: identify which norms (ℓ₂, ℓ∞, ℓ₁, spectral) best capture EW reality<br>4. Develop threat landscape: enumerate representative EW attack scenarios (jamming, spoofing, waveform mimicry)<br><br>**Phase B (Attack Design):**<br>5. Design EW-constrained adversarial attacks: $\min_{\Delta x} ||\Delta x|| \text{ s.t. } P(\Delta x) \leq P_{max}, ||M * \Delta x||_{freq} \leq M, \text{PAPR} \leq PAPR_{max}, \ldots$<br>6. Develop heuristics for efficient attack generation (EW attacks may not have closed-form solutions like FGSM)<br>7. Evaluate attack detectability: can spectrum monitoring detect adversarial signals vs. jamming<br>8. Propose signature of adversarial EW attacks (spectral/temporal features distinguishing from natural RF)<br><br>**Phase C (Defense Architecture):**<br>9. Design receiver architecture accounting for EW threat model (not just generic adversarial)<br>10. Develop EW-aware detection: identify if incoming adversarial signal is attack vs. jamming vs. interference<br>11. Propose adaptive response: spectrum hopping, power adjustment, diversity techniques specific to EW threats<br>12. Integrate with cognitive EW (Reference 24): real-time threat classification and response<br><br>**Phase D (Validation & Standardization):**<br>13. Conduct user study with military spectrum operations personnel: validation of threat model and defense usability<br>14. Propose EW-specific adversarial robustness standard (analogous to IEEE 1886 for ML robustness)<br>15. Develop test plan for EW receiver certification |
| **Novel Contributions** | • First formal EW-specific threat model for adversarial RF attacks<br>• EW-constrained adversarial attack methodology<br>• Defense architecture tailored to military EW scenarios<br>• Proposed standard for EW adversarial robustness certification |
| **Experimental Artifacts** | • Threat model specification document (with military stakeholder review)<br>• EW-constrained attack code<br>• Defense architecture proof-of-concept<br>• Robustness evaluation metrics (EW-specific ROC curves, detection latency) |
| **Deliverables** | • 2–3 papers: (1) threat model formalization (IEEE TIFS), (2) attack/defense design (IEEE JSAC or IEEE Comm Mag), (3) user study results<br>• White paper on EW adversarial robustness standard (for military acquisition)<br>• Prototype receiver demonstrating defense against EW threats |
| **Feasibility** | **Medium:** Requires military stakeholder engagement (feasibility risk); domain expertise in EW (consulting); testbed validation desirable but not essential (simulation can substitute). |
| **Novelty** | **Very High:** EW-specific threat model is novel; military relevance exceptional. |
| **DoD Alignment** | **Critical:** Directly addresses Army/DoD operational needs; potential for acquisition impact and policy influence. |
| **Timeline** | 18–24 months (stakeholder engagement + threat model definition + implementation + validation) |
| **Estimated Resources** | 2 PhD students, 1 EW expert (consultant), potential USRP testbed (~$5k), military lab partnership (desirable) |

---

## Summary Table: Gap Prioritization

| Gap | Research Maturity | DoD Alignment | Novelty | Feasibility | Timeline | Estimated Cost | Priority Rank |
|-----|------------------|---------------|---------|-------------|----------|----------------|-----------------|
| 1. Channel-Realistic OTA Attacks | Early-stage | **Very High** | High | Medium-High | 12–18 mo | $15k | **1** |
| 2. Certified Robustness for RF | Early-stage | **Very High** | Very High | Medium | 12–18 mo | $5k | **2** |
| 3. Robustness-Accuracy Trade-offs | Early-stage | High | High | Medium | 12 mo | $10k | **3** |
| 4. Transformer Adversarial Robustness | Early-stage | High | Very High | Medium | 12–18 mo | $15k | **3** (tie) |
| 5. PHY+ML Cross-Layer Defenses | Early-stage | Very High | High | Medium | 12–18 mo | $20k | **4** |
| 6. Distributed Sensing Game Theory | Early-stage | Very High | Very High | Medium | 12 mo | $5k | **5** |
| 7. EW-Specific Threat Models | Foundational | **Critical** | Very High | Medium | 18–24 mo | $30k | **1** (with Gap 1) |

---

## Recommended Research Sequencing

### **Phase 1 (Immediate): Foundational Threat Assessment**
- **Lead:** Gaps 1 & 7 (parallel)
- **Goal:** Establish EW threat model and characterize channel-OTA adversarial interaction
- **Rationale:** Both are prerequisites for all subsequent work; provide operational context
- **Timeline:** 12–18 months
- **Team:** 3–4 PhD students, 1–2 EW experts (consulting), military partnership
- **Deliverables:** Threat model white paper, OTA testbed, initial attack/defense results

### **Phase 2 (Months 12–24): Robustness Theory & Practice**
- **Lead:** Gaps 2, 3, 4 (parallel)
- **Goal:** Develop certified robustness, understand trade-offs, evaluate transformers
- **Rationale:** Build on Phase 1 threat models; establish theoretical and empirical foundations
- **Timeline:** 12–18 months parallel
- **Team:** 3–4 PhD students (theory + experiments)
- **Deliverables:** Certified robustness framework, trade-off characterization, transformer robustness results

### **Phase 3 (Months 18–30): Systems & Integration**
- **Lead:** Gaps 5, 6 (could start earlier, but depend on prior results)
- **Goal:** Integrate across layers (PHY+ML), extend to distributed systems, refine EW defense architecture
- **Rationale:** System-level insights require component-level results; distributed/game-theoretic analysis informed by Gaps 1–4
- **Timeline:** 12 months
- **Team:** 2–3 PhD students, cross-disciplinary (signal processing + ML + game theory)
- **Deliverables:** Prototype cross-layer receiver, game-theoretic analysis, EW defense architecture

### **Phase 4 (Months 24+): Validation & Standardization**
- **Lead:** Gap 7 (stakeholder engagement) + all prior results
- **Goal:** Military user evaluation, standard development, acquisition guidance
- **Rationale:** Validation and standardization ensure real-world impact; continuous stakeholder engagement
- **Timeline:** 12+ months (ongoing)
- **Team:** 1–2 researchers (military liaison), external reviewers
- **Deliverables:** EW adversarial robustness standard, certification framework, acquisition guidance

---

## Expected Research Outputs & Impact

### **Publications**
- **Estimated total:** 12–15 peer-reviewed papers
  - 3–4 in IEEE TIFS (top-tier security)
  - 2–3 in IEEE Transactions (communications, wireless, info theory)
  - 2–3 in IEEE S&P / NDSS / CCS (security conferences)
  - 2–3 in general ML venues (ICML, NeurIPS) if transformer/theory angle strong
  - 1–2 in military/defense-focused venues (CDR, etc.)

### **Tools & Artifacts**
- **Open-source software:** Channel-aware attack generators, certified defense implementations, testbed software
- **Datasets:** Augmented RadioML with channel parameters, EW threat scenario library
- **Hardware:** USRP-based testbed design, phased deployment guides

### **Military Impact**
- **Direct:** EW threat assessment framework, defense architecture for cognitive EW systems
- **Standards:** Proposed EW adversarial robustness certification (potential DoD policy)
- **Acquisition:** Guidance for evaluating RF ML systems in procurement
- **Training:** Materials for military spectrum operations personnel on adversarial threats

### **Broader Community**
- **Interdisciplinary:** Bridge between adversarial ML, signal processing, and military EW
- **Methodological:** Adapt certified robustness and game theory to physical-layer systems
- **Dataset:** Publicly release augmented RadioML with channel/adversarial annotations

---

## Risk Mitigation

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Military stakeholder unavailable for Gap 7 | Medium | High | Engage early; propose alternative DoD lab partnerships |
| OTA testbed access limited (Gap 1, 5) | Medium | High | Develop hybrid approach: simulation + limited OTA validation; explore academic RF labs |
| Transformer robustness hard (Gap 4) | Medium | Medium | Scale back scope; focus on attack development first, defenses second |
| Distributed game-theoretic analysis intractable (Gap 6) | Low-Medium | Medium | Use approximate algorithms, restrict to small sensor networks, numerical simulation |
| Certified robustness Rayleigh fading hard (Gap 2) | Medium | Medium | Relax to restricted fading models; propose alternative certification (tighter empirical bounds) |

---

## Conclusion

The seven research gaps represent a coherent research agenda at the intersection of adversarial ML, RF signal processing, and military EW operations. Gaps 1 & 7 are highest priority (establish threat model and OTA validation). Gaps 2–6 follow logically and build complementary capabilities. The sequencing above allows phased execution with clear dependencies.

**Expected outcome:** Comprehensive threat assessment framework for adversarial robustness of RF classifiers under realistic channel impairments, with practical defense architectures ready for military evaluation and standardization.

---

**Document prepared for:** CDR 2026 EW Special Issue Research Planning
**Date:** February 2026
**Contact:** [Research coordination contact]
