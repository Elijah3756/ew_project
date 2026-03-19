# Literature Review: Adversarial Robustness of AI-Based RF Signal Classification Under Channel Impairments

**Target Venue:** Cyber Defense Review (CDR) 2026 EW Special Issue, U.S. Army
**Compilation Date:** February 2026

---

## Executive Summary

This literature review synthesizes research across five critical domains relevant to adversarial robustness in RF signal classification under realistic channel conditions. The review identifies foundational work in deep learning-based modulation recognition, adversarial attack methodologies, certified defenses, channel-aware robustness, and cognitive electronic warfare (EW) applications. A significant gap exists at the intersection of channel-realistic adversarial attacks and formal robustness certification for RF classifiers—an opportunity this research project can address.

---

## Category 1: RF/Modulation Classification with Deep Learning (Foundational)

### 1.1 Seminal Works on Deep Learning for Automatic Modulation Classification

**Reference 1:**
**Authors:** T. J. O'Shea, N. West, K. Clancy
**Year:** 2016
**Title:** Convolutional Radio Modulation Recognition Networks
**Venue:** IEEE International Symposium on Signal Processing and Information Technology (ISSPIT)
**Key Contribution:** First major application of CNNs to raw I/Q samples for automatic modulation recognition without hand-crafted features. Introduced the RadioML 2016.10a dataset (220,000 modulated samples across 11 modulation classes including BPSK, QPSK, 8PSK, FSK, PSK, QAM, and analog modulations). Average recognition rate: 75% at 10 dB SNR, establishing the baseline for deep learning-based AMC.
**Relevance:** Foundation for all modern DL-based RF signal classification; defines the RF classification problem and establishes benchmark datasets used across adversarial robustness research.

---

**Reference 2:**
**Authors:** Various (survey)
**Year:** 2023–2024
**Title:** A Survey on Deep Learning Enabled Automatic Modulation Classification Methods: Data Representations, Model Structures, and Regularization Techniques
**Venue:** Signal Processing journal / IEEE (multiple surveys published)
**Key Contribution:** Comprehensive synthesis of data representations (I/Q, spectrograms, constellation diagrams), deep architectures (CNN, RNN, LSTM, Transformer, hybrid models), and regularization for AMC. Discusses CNN and ResNet architectures, highlighting advances in complex feature extraction, generalization at low SNR.
**Relevance:** Provides taxonomy of DL architectures used in RF signal classification; essential for understanding baseline models vulnerable to adversarial perturbations.

---

### 1.2 Advanced Architectures for RF Signal Classification

**Reference 3:**
**Authors:** Emam, Shalaby, et al.
**Year:** 2021–2022
**Title:** A Comparative Study between CNN, LSTM, and CLDNN Models in The Context of Radio Modulation Classification
**Venue:** IEEE Conference Publications / Semantic Scholar
**Key Contribution:** Introduces CLDNN (Convolutional LSTM Deep Neural Network) architecture combining two CNN layers, two LSTM layers, and two fully connected layers. CLDNN achieves 4–6% relative improvement over standalone LSTM by capturing both short-term CNN features and long-term LSTM temporal dependencies. Demonstrates performance gains at low SNR.
**Relevance:** CLDNN and similar hybrid architectures are increasingly studied in adversarial robustness contexts; temporal modeling may offer robustness advantages against adversarial perturbations.

---

**Reference 4:**
**Authors:** (Transformer-based modulation recognition)
**Year:** 2023–2024
**Title:** TLDNN: Transformer-LSTM Deep Neural Network for RF Signal Classification
**Venue:** IEEE/ArXiv (state-of-the-art RF classification)
**Key Contribution:** Hybrid framework incorporating transformer self-attention mechanisms to model global signal correlations alongside LSTM temporal dependencies. Achieves state-of-the-art performance on RadioML 2016.10a and RadioML 2018.01a datasets, particularly in low-SNR scenarios. 80–90% complexity reduction vs. prior art.
**Relevance:** Transformers and attention mechanisms represent the frontier of RF classifiers; their adversarial robustness properties are largely unexplored and represent a critical research gap.

---

**Reference 5:**
**Authors:** (ResNet and Deep Residual Networks)
**Year:** 2020–2023
**Title:** Deep Learning for RF Fingerprinting: A Massive Experimental Study / Radio Modulation Classification Using Deep Residual Neural Networks
**Venue:** IEEE / ArXiv / MDPI
**Key Contribution:** ResNet18/50 architectures adapted for RF signals (via spectrograms and I/Q inputs). Achieves near 99% accuracy at high SNR (>10 dB), with ResNet outperforming VGG beyond 10 dB SNR. Identity shortcuts enable faster convergence and improved generalization. Demonstrates effectiveness for both signal classification and RF device fingerprinting.
**Relevance:** ResNets are commonly used baselines in adversarial robustness studies; their skip-connection design may provide unexpected robustness properties worth investigating.

---

## Category 2: Adversarial ML for RF/Wireless Signals

### 2.1 Foundational Adversarial Attack Methods

**Reference 6:**
**Authors:** I. J. Goodfellow, J. Shlens, C. Szegedy
**Year:** 2014
**Title:** Explaining and Harnessing Adversarial Examples
**Venue:** arXiv:1412.6572 (ICLR 2015)
**Key Contribution:** Seminal work introducing FGSM (Fast Gradient Sign Method), the foundational single-step adversarial attack. Demonstrates that adversarial vulnerability stems from linear (not nonlinear) decision boundaries. Proposes adversarial training as a first defense. Simple gradient sign method: *x' = x + ε * sign(∇_x L(x, y))*.
**Relevance:** FGSM is the baseline attack method; essential to understand as the simplest and fastest attack against RF classifiers.

---

**Reference 7:**
**Authors:** A. Madry, A. Makelov, L. Schmidt, D. Tsipras, A. Vladu
**Year:** 2017
**Title:** Towards Deep Learning Models Resistant to Adversarial Attacks
**Venue:** OpenReview (ICLR 2019)
**Key Contribution:** Introduces PGD (Projected Gradient Descent) adversarial training, the most widely adopted empirical defense. Frames adversarial robustness as a min-max robust optimization problem. PGD attack is iterative: *x_{t+1} = Π(x_t + α * sign(∇_x L(x_t, y)))* where Π projects onto ε-ball. PGD-AT demonstrates consistent robustness improvement against gradient-based attacks.
**Relevance:** PGD is the *de facto* iterative attack standard; critical for evaluating RF classifier robustness. PGD-AT is the baseline defense to compare against.

---

**Reference 8:**
**Authors:** N. Carlini, D. Wagner
**Year:** 2016
**Title:** Towards Evaluating the Robustness of Neural Networks
**Venue:** IEEE Symposium on Security and Privacy (S&P 2017)
**Key Contribution:** Formulates adversarial perturbation as constrained optimization: minimize ||Δx|| subject to f(x + Δx) outputs target class. Three variants (L₀, L₂, L∞). Employs tanh transformation and iterative optimization via BFGS. C&W attack is stronger than FGSM/PGD but computationally expensive. Widely used for robustness evaluation.
**Relevance:** C&W attack represents the optimization-based perspective; important for measuring true robustness limits in RF classifiers.

---

### 2.2 Adversarial Attacks Specifically on RF Classifiers

**Reference 9:**
**Authors:** (Multi-author collaborative work on RF adversarial attacks)
**Year:** 2018–2019
**Title:** Adversarial Attacks on Deep-Learning Based Radio Signal Classification
**Venue:** arXiv:1808.07713 / IEEE Conference
**Key Contribution:** Applies FGSM, BIM (Basic Iterative Method), MIM (Momentum Iterative Method), and PGD to deep learning-based modulation classifiers trained on RadioML data. Demonstrates that iterative methods (BIM, MIM, PGD) are substantially more effective than single-step FGSM. Shows that adversarial examples can reduce accuracy from ~90% to <10% with perturbations as small as 0.5% power relative to signal.
**Relevance:** Directly establishes vulnerability of RF classifiers; benchmark attack effectiveness for modulation recognition systems.

---

**Reference 10:**
**Authors:** (Targeted Adversarial Examples)
**Year:** 2018–2019
**Title:** Targeted Adversarial Examples Against RF Deep Classifiers
**Venue:** ACM Workshop on Wireless Security and Machine Learning
**Key Contribution:** Introduces targeted adversarial attack scenarios for RF classifiers. Demonstrates white-box and black-box attacks using surrogate models. Shows adversarial transferability: perturbations crafted on one CNN architecture transfer to other architectures with non-trivial success rates (~30–70% depending on architecture similarity).
**Relevance:** Establishes threat model for RF systems; transferability indicates fundamental adversarial vulnerabilities, not architecture-specific quirks.

---

**Reference 11:**
**Authors:** (Frequency-Selective and Physical-Layer Attacks)
**Year:** 2022–2024
**Title:** Frequency-Selective Adversarial Attack Against Deep Learning-Based Wireless Signal Classifiers / Stealthy Adversarial Attacks on Machine Learning-Based Classifiers of Wireless Signals
**Venue:** IEEE Conference / arXiv
**Key Contribution:** Proposes frequency-domain adversarial perturbations that are more physically realizable than time-domain additive noise. Demonstrates that selective frequency-band modifications can fool classifiers while appearing benign in power spectral density (PSD). Introduces "stealthy" attacks that mimic natural channel effects.
**Relevance:** Bridges gap between theoretical adversarial examples and physical-layer feasibility; critical for assessing practical EW threat scenarios.

---

**Reference 12:**
**Authors:** (Adversarial Attacks on GaN Power Amplifiers / RF Identification)
**Year:** 2022–2023
**Title:** Adversarial Attacks and Active Defense on Deep Learning Based Identification of GaN Power Amplifiers Under Physical Perturbation
**Venue:** Digital Signal Processing / ScienceDirect
**Key Contribution:** Evaluates adversarial attacks on RF device identification systems. Tests FGSM, BIM, PGD, and MIM attacks on 16 Gallium Nitride (GaN) power amplifier RF fingerprints. Demonstrates drastic accuracy degradation from ~100% to <10% with only 0.5% adversarial perturbation. Shows that RF fingerprinting (device-level identification) is highly vulnerable.
**Relevance:** Extends adversarial RF research beyond modulation classification to RF fingerprinting; indicates vulnerability is systemic across RF ML applications.

---

## Category 3: Adversarial Robustness and Defenses

### 3.1 Certified Robustness via Randomized Smoothing

**Reference 13:**
**Authors:** J. M. Cohen, E. Rosenfeld, J. Z. Kolter
**Year:** 2019
**Title:** Certified Adversarial Robustness via Randomized Smoothing
**Venue:** ICML 2019 / JMLR
**Key Contribution:** Introduces randomized smoothing: transform any classifier into a provably robust classifier by adding Gaussian noise during inference. Provides L₂ robustness certification: any input misclassified by noisy classifier is robust to perturbations with ℓ₂ norm < r (certified radius). Achieves ImageNet accuracy of 49% at radius 0.5 (ℓ₂). Scalable to large networks without architectural assumptions.
**Relevance:** Provides theoretical framework for certified RF classifier robustness; can be applied to RF domain with channel noise as natural augmentation.

---

**Reference 14:**
**Authors:** (Extensions of Randomized Smoothing)
**Year:** 2020–2023
**Title:** Certified Robustness via Randomized Smoothing over [Lp] Spaces / Input-Specific Robustness Certification for Randomized Smoothing
**Venue:** IJCAI / AAAI / CVPR Workshops
**Key Contribution:** Extends Cohen et al.'s L₂ smoothing to L₁ and L∞ norms. Proposes input-specific robustness radii (more flexible than fixed radius). Addresses computational bottleneck of sampling large numbers of noisy copies.
**Relevance:** Multiple norm variants (L₁, L₂, L∞) enable diverse perturbation threat models relevant to RF (power, spectral, amplitude constraints).

---

### 3.2 Adversarial Training and Defenses

**Reference 15:**
**Authors:** (Adversarial Training Techniques)
**Year:** 2019–2023
**Title:** Adversarial Training for Free! / I-PGD-AT: Efficient Adversarial Training via Incremental PGD Adversarial Training / On the Convergence and Robustness of Adversarial Training
**Venue:** NeurIPS / OpenReview / ICML
**Key Contribution:** PGD-AT (Madry et al.) is empirically powerful but computationally expensive. "Free" adversarial training reuses gradients from data augmentation to reduce computational cost. I-PGD-AT introduces incremental perturbation updates. Analysis of convergence and robustness trade-offs in adversarial training.
**Relevance:** Practical adversarial training approaches; essential for training robust RF classifiers at scale.

---

**Reference 16:**
**Authors:** (Data Augmentation for Adversarial Robustness)
**Year:** 2020–2024
**Title:** Data Augmentation Can Improve Robustness / Fixing Data Augmentation to Improve Adversarial Robustness / Rethinking Data Augmentation for Adversarial Robustness
**Venue:** NeurIPS / OpenReview / Signal Processing
**Key Contribution:** Data augmentation (rotation, scaling, noise injection) improves adversarial robustness when combined with adversarial training. Hardness and diversity of augmentations are key: diversity improves both accuracy and robustness; hardness boosts robustness at accuracy cost. Can approach state-of-the-art robustness when combined with model weight averaging.
**Relevance:** For RF domain, channel-based augmentations (AWGN, fading, CFO) are natural and potentially more effective than pixel-level augmentations in image domain.

---

**Reference 17:**
**Authors:** (Detection-Based Defenses)
**Year:** 2019–2020
**Title:** Adversarial Examples in RF Deep Learning: Detection of the Attack and its Physical Robustness
**Venue:** arXiv / IEEE Conference
**Key Contribution:** Proposes statistical tests to detect adversarial RF examples in physical (over-the-air) settings. Tests based on Peak-to-Average-Power-Ratio (PAPR) and Softmax output probabilities. Findings: RF adversarial examples generated in digital domain often fail when transmitted over-the-air due to channel effects (multipath, fading) that corrupt the adversarial perturbation.
**Relevance:** Critical insight: channel impairments may naturally degrade adversarial perturbations, providing accidental robustness. Opens question: can we design channels-aware adversarial attacks that survive propagation?

---

## Category 4: Channel-Aware ML and Robustness Under Impairments

### 4.1 RF Impairments and Channel Effects

**Reference 18:**
**Authors:** (RF Impairments Survey)
**Year:** 2021–2023
**Title:** RF Impairments in Wireless Transceivers: Phase Noise, CFO, and IQ Imbalance – A Survey
**Venue:** IEEE Transactions on Wireless Communications
**Key Contribution:** Comprehensive review of RF impairments: phase noise (PN), carrier frequency offset (CFO), and IQ imbalance. These impairments degrade signal quality and affect ML classifier performance. CFO causes spectral spreading; phase noise introduces amplitude/phase jitter; IQ imbalance creates spectral asymmetry. Practical RF systems must handle all three simultaneously.
**Relevance:** Establishes realistic RF channel impairments that classifiers must tolerate; critical context for channel-aware adversarial robustness.

---

**Reference 19:**
**Authors:** (AMC Under Channel Effects)
**Year:** 2019–2023
**Title:** Automatic Modulation Classification Using Multi-Domain Integrated Feature Extraction in Fading Environments / Robust Automatic Modulation Classification Technique for Fading Channels via Deep Neural Network / Automatic Modulation Classification Under AWGN and Fading Channels Using Convolutional Neural Network
**Venue:** MDPI / Springer / IEEE
**Key Contribution:** Features effective for AWGN channels differ from those effective for Rayleigh fading. Fading introduces time-varying amplitude and phase distortions. Cyclostationary features (spectral correlation function, SCF) robust to fading. Recent CNN approaches achieve 86.1% accuracy at −2 dB SNR, 96.5% at 0 dB, 99.8% at 10 dB under fading. Multi-domain integration (time, frequency, time-frequency) improves fading robustness.
**Relevance:** Establishes baseline: classifiers struggle with realistic channel impairments; question becomes whether adversarial perturbations compound or interact with channel effects.

---

**Reference 20:**
**Authors:** (Carrier Frequency Offset for RF Identification)
**Year:** 2024
**Title:** Towards Robust RF Fingerprint Identification Using Spectral Regrowth and Carrier Frequency Offset
**Venue:** arXiv
**Key Contribution:** Demonstrates that CFO can be auxiliary feature for RF fingerprinting (device identification). Classification accuracy 92.76% achieved using spectral regrowth and CFO-assisted collaborative mechanisms. Shows that CFO, while not a stable feature across time, can aid classification when calibrated.
**Relevance:** CFO as both nuisance and feature; relevant for understanding how channel effects can be compensated or exploited.

---

### 4.2 Channel Augmentation as Defense

**Reference 21:**
**Authors:** (Channel-Aware Adversarial Attacks)
**Year:** 2020–2022
**Title:** Channel-Aware Adversarial Attacks Against Deep Learning-Based Wireless Signal Classifiers / Adversarial Sample Generation Method Based on Frequency Domain Transformation and Channel Awareness
**Venue:** IEEE / arXiv / IEEE Conference (CCS, Security & Privacy)
**Key Contribution:** Conventional adversarial perturbations are additive in time domain but unrealistic for over-the-air transmission (multiplicative in frequency domain due to channel convolution). Proposes channel-aware attack: generates perturbations that account for expected channel response, making them robust to transmission. Channel estimation is critical: without it, adversarial samples degrade significantly after channel propagation.
**Relevance:** Bridges gap between digital-domain adversarial examples and physical-layer realism; informs design of truly threatening RF adversarial attacks.

---

**Reference 22:**
**Authors:** (Data Augmentation with Channel Effects for Defense)
**Venue:** IEEE / arXiv
**Year:** 2021–2023
**Title:** Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems / Channel Augmentation for Robust RF Classification
**Key Contribution:** Training augmentation with simulated channel effects (AWGN, fading, CFO, phase noise) improves both clean accuracy and adversarial robustness. Gaussian noise augmentation studied: impact of noise power on robustness evaluated. More augmentation samples increase robustness. Certified defense using randomized smoothing with channel noise as natural augmentation mechanism.
**Relevance:** Channel augmentation strategy is promising: leverages realistic RF environment as defense mechanism. Natural fit for RF domain vs. abstract data augmentation.

---

**Reference 23:**
**Authors:** (Over-the-Air Adversarial Attacks)
**Year:** 2021–2024
**Title:** Over-the-Air Adversarial Attacks on Deep Learning Based Modulation Classifier Over Wireless Channels / Over-The-Air Adversarial Attacks on Deep Learning WiFi Fingerprinting / Robust Adversarial Attacks Against DNN-Based Wireless Communication Systems
**Venue:** IEEE / arXiv / ACM CCS
**Key Contribution:** Demonstrates that generating physically realizable over-the-air adversarial attacks is fundamentally different from digital perturbations. For OTA attacks, adversarial perturbations must be transmitted through channel, subject to multiplicative channel convolution (not additive). WiAdv system achieves >70% attack success on WiFi gesture recognition while remaining "stealthy" (mimics natural CSI loss). Key challenge: maintaining attack effectiveness across channel variability and receiver configurations.
**Relevance:** Establishes threat model for practical EW applications; OTA attacks are the realistic scenario for military RF systems.

---

## Category 5: EW/Cognitive EW Context

### 5.1 Cognitive Electronic Warfare and AI/ML

**Reference 24:**
**Authors:** (Cognitive EW Overview)
**Year:** 2018–2023
**Title:** Cognitive Electronic Warfare: An Artificial Intelligence Approach / Cognitive Electronic Warfare: Radio Frequency Spectrum Meets Machine Learning
**Venue:** Artech House / Avionics Digital Edition / AFCEA Signal Magazine
**Key Contribution:** Cognitive EW leverages AI/ML to enable machine-speed spectrum awareness, adaptive jamming, and rapid signal identification. Traditional EW relies on predefined threat databases; modern threats (software-defined radios, frequency-hopping, adaptive waveforms) render static databases obsolete. CEW uses neural networks (DNNs, SNNs) to sense, characterize, and respond to RF signals in real-time (millisecond to microsecond timescales). Key capability: "take the person out of the loop" for rapid response.
**Relevance:** Establishes military context and urgency; RF classifier robustness directly enables or disables CEW capabilities.

---

**Reference 25:**
**Authors:** (Military AI/ML Spectrum Management)
**Year:** 2023–2025
**Title:** Implement AI in Electromagnetic Spectrum Operations / AI- and ML-enabled Spectrum Management Tech Goal of DoD Research / Systematic Literature Review of AI-enabled Spectrum Management in 6G and Future Networks
**Venue:** Proceedings (USNI) / Military Embedded Systems / arXiv
**Key Contribution:** DoD Spectrum Access R&D Program (SAR&DP) developing near-real-time spectrum management leveraging ML/AI. Goals: adaptive spectrum allocation, interference mitigation, enhanced security. AI-powered Adaptive Spectrum Dynamics (ASD) framework for military radios. Emphasizes need for robust, reliable ML systems in contested/congested spectrum environments.
**Relevance:** DoD explicitly prioritizes AI/ML for spectrum operations; establishes alignment with military operational needs and acquisition priorities.

---

**Reference 26:**
**Authors:** (DARPA RF Machine Learning Systems Program)
**Year:** 2020–2024
**Title:** DARPA RF Machine Learning Systems Program / Smarter AI for Electronic Warfare
**Venue:** DARPA Program Description / AFCEA / Military Aerospace
**Key Contribution:** DARPA RF-ML Systems program aims to develop RF systems that are "goal-driven and can learn from data." Focus on spectrum awareness, spectrum sensing, and adaptive countermeasures. Emphasis on robustness to environmental variability, adversarial manipulation, and dynamic threat adaptation.
**Relevance:** Federal R&D agency explicitly funding adversarial robustness research in RF domain; validates importance of this research area.

---

### 5.2 Spectrum Sensing and Cognitive Radio Networks

**Reference 27:**
**Authors:** (Spectrum Sensing with ML)
**Year:** 2020–2024
**Title:** Analysis of Spectrum Sensing Using Deep Learning Algorithms: CNNs and RNNs / Deep Learning-Based Spectrum Sensing for Cognitive Radio Applications / Deep Learning Frameworks for Cognitive Radio Networks: Review and Open Research Challenges
**Venue:** ScienceDirect / IEEE / arXiv
**Key Contribution:** Spectrum sensing (detecting primary user occupancy) is critical for cognitive radio networks. Deep learning (CNN, RNN) outperforms traditional energy detection methods, especially at low SNR. Primary threats to spectrum sensing: Primary User Emulation (PUE) attacks and Spectrum Sensing Data Falsification (SSDF) attacks where adversaries spoof or falsify sensing data. ML-based defenses can detect anomalies in sensing reports.
**Relevance:** Spectrum sensing is a key ML task in RF domain; subject to both adversarial and data-poisoning attacks relevant to EW scenarios.

---

**Reference 28:**
**Authors:** (Defense Against Spectrum Sensing Attacks)
**Year:** 2023–2024
**Title:** Defense Against Spectrum Sensing Data Falsification Attack in Cognitive Radio Networks Using Machine Learning / A Survey on Cognitive Radio Network Attack Mitigation Using Machine Learning and Blockchain
**Venue:** IEEE / Springer / Wireless Communications
**Key Contribution:** Proposes ML-based detection of SSDF attacks in cognitive radio networks. Anomaly detection using Isolation Forests, support vector machines, and neural networks. Blockchain-based mechanisms for distributed defense. Demonstrates detection accuracy >95% for common SSDF attack patterns.
**Relevance:** Establishes cyber-physical security perspective on RF ML systems; informs defense architectures for resilient EW applications.

---

### 5.3 RF Fingerprinting Robustness

**Reference 29:**
**Authors:** (Adversarial-Driven Study of RF Fingerprinting)
**Year:** 2024
**Title:** An Adversarial-Driven Experimental Study on Deep Learning for RF Fingerprinting
**Venue:** arXiv:2507.14109
**Key Contribution:** Recent large-scale study of adversarial robustness in RF fingerprinting (device identification). Identifies two key vulnerabilities: (1) CNN domain shifts can be exploited for backdoor impersonation attacks; (2) entanglement of RF fingerprints with environmental and signal-pattern features in raw IQ training. Domain shift vulnerability enables external attackers to impersonate legitimate devices. Limited prior work on DL-based RFF adversarial robustness, mostly under ideal conditions.
**Relevance:** RF fingerprinting is complementary to modulation classification; adversarial challenges carry across RF ML domains.

---

**Reference 30:**
**Authors:** (RF Fingerprinting Robustness via Feature Disentanglement)
**Year:** 2024–2025
**Title:** Cross-Receiver Generalization for RF Fingerprint Identification via Feature Disentanglement and Adversarial Training / Radio Frequency Fingerprinting via Deep Learning: Challenges and Opportunities / Learning Robust Radio Frequency Fingerprints
**Venue:** arXiv / NIST / NSF
**Key Contribution:** Proposes adversarial training and style transfer to decouple transmitter (device) features from receiver (channel) features in RF fingerprinting. Addresses cross-receiver generalization problem: RF fingerprints learned on one receiver fail on another due to channel entanglement. Augmentation with diverse realistic wireless channel conditions (real data from channels) improves robustness more than synthetic augmentation.
**Relevance:** Feature disentanglement approach is applicable to modulation classification; channel-aware training is key defense strategy.

---

---

## Category 6: Specialized Topics

### 6.1 Reconfigurable Intelligent Surfaces (RIS) in RF Sensing

**Reference 31:**
**Authors:** (RIS-Aided RF Sensing)
**Year:** 2021–2024
**Title:** A Reconfigurable Intelligent Surface with Integrated Sensing Capability / Reconfigurable Intelligent Surfaces as the Key-Enabling Technology for Smart Electromagnetic Environments / Radar Sensing with Reconfigurable Intelligent Surfaces (RIS)
**Venue:** Nature Scientific Reports / IEEE / MATLAB / ETSI
**Key Contribution:** RIS (passive phased array) can dynamically shape the radio environment. Applications: enhanced RF sensing (improved SNR by steering reflections), electronic warfare (asymmetric signals to confuse eavesdroppers), spectrum awareness (adaptive environment shaping). Potential synergy: RIS can provide environmental diversity (similar to channel augmentation) to improve classifier robustness.
**Relevance:** Emerging technology with potential to naturally provide channel augmentation; represents new frontier for robust RF sensing in contested EW environments.

---

### 6.2 Transfer Learning and Domain Adaptation

**Reference 32:**
**Authors:** (Domain Adaptation for AMC)
**Year:** 2021–2024
**Title:** Semi-Supervised-Based Automatic Modulation Classification with Domain Adaptation for Wireless IoT Spectrum Monitoring / Domain Adaptation–Based Automatic Modulation Classification / Cross-Domain Automatic Modulation Classification Using Multimodal Information and Transfer Learning
**Venue:** Frontiers in Physics / Scientific Programming / Remote Sensing (MDPI)
**Key Contribution:** Training/test domain mismatch (different SNR, channels, frequency offsets) causes accuracy degradation. Domain adaptation (DAN, adversarial domain adaptation) transfers knowledge from source domain to target without labeled target data. Semi-supervised SemiAMR network achieves cross-domain recognition. Multimodal fusion (amplitude, phase, spectrum) improves domain robustness.
**Relevance:** Domain adaptation is orthogonal to adversarial robustness but closely related; domain shift and adversarial perturbations both cause misclassification. Potential synergies in defense mechanisms.

---

### 6.3 Robustness Evaluation Frameworks

**Reference 33:**
**Authors:** (Robustness Assessment Frameworks)
**Year:** 2019–2024
**Title:** A Survey of Neural Network Robustness Assessment in Image Recognition / Robustness Evaluation for Deep Neural Networks via Mutation Decision Boundaries Analysis / SMART: A Robustness Evaluation Framework for Neural Networks / Framework for Testing Robustness of Machine Learning-Based Classifiers
**Venue:** arXiv / ScienceDirect / IEEE / PMC
**Key Contribution:** Robustness encompasses adversarial perturbations and natural corruptions. CLEVER metric estimates local Lipschitz constant (model smoothness) as robustness proxy. Mutation testing identifies unstable decision boundaries. Frameworks distinguish: adversarial robustness (deliberate attacks) vs. corruption robustness (random noise). Metrics: certified radius (smoothing), empirical robustness (worst-case attack success).
**Relevance:** Provides evaluation methodology; RF classifiers need systematic robustness assessment similar to image models, adapted for channel-realistic perturbations.

---

---

## Research Gaps and Opportunities

### **Gap 1: Channel-Realistic Adversarial Attacks on RF Classifiers**

**Status:** Limited prior work at intersection.
**Problem:** Most adversarial attacks on RF classifiers (References 9, 10, 11) are tested in digital domain without rigorous over-the-air validation. Channel-aware attacks (References 21, 23) are nascent and focus on WiFi CSI, not modulation recognition. Gap: develop systematically validated adversarial attacks that survive realistic channel propagation (multipath fading, phase noise, CFO) while remaining undetectable.

**Opportunity:** This research project can:
- Design and experimentally validate over-the-air adversarial attacks on modulation classifiers
- Characterize how channel impairments degrade or enhance adversarial perturbations
- Establish metrics for "physical realism" of RF adversarial examples

---

### **Gap 2: Certified Robustness for RF Classifiers Under Channel Variability**

**Status:** Certified robustness (Reference 13, 14) is well-developed for images; applications to RF domain are absent.
**Problem:** Randomized smoothing assumes noise model (Gaussian); RF channels exhibit structured impairments (fading, not just AWGN). Certified radius may be loose if noise model mismatches reality. Gap: adapt certified robustness techniques to RF domain with realistic channel noise models.

**Opportunity:** This research project can:
- Extend randomized smoothing to Rayleigh/Rician fading noise models
- Derive certified robustness bounds accounting for CFO, phase noise, IQ imbalance
- Validate certifications against practical over-the-air adversarial attacks

---

### **Gap 3: Robustness-Accuracy Trade-offs in RF Classifiers**

**Status:** General ML literature (References 15, 16) shows adversarial training reduces clean accuracy; systematic study for RF domain is lacking.
**Problem:** RF classification at low SNR is already challenging (86% accuracy at −2 dB). Adversarial training may further degrade performance. Gap: quantify trade-offs and develop techniques that preserve accuracy while improving robustness.

**Opportunity:** This research project can:
- Empirically characterize robustness-accuracy curves for RF modulation classifiers
- Develop hybrid defenses combining adversarial training, channel augmentation, and certified robustness
- Propose SNR-adaptive defense strategies

---

### **Gap 4: Adversarial Attacks on Transformer-Based RF Classifiers**

**Status:** Transformer-based RF classifiers (Reference 4) are state-of-the-art but adversarial robustness is unexplored.
**Problem:** Attention mechanisms may enable new adversarial vulnerabilities or robustness properties. Gap: systematically evaluate adversarial susceptibility of transformer RF classifiers.

**Opportunity:** This research project can:
- Develop targeted attacks exploiting attention mechanisms
- Evaluate whether self-attention provides natural robustness
- Propose attention-aware defenses

---

### **Gap 5: Physical Layer Perspective on Adversarial Defenses**

**Status:** Most RF adversarial literature (References 17, 21, 23) is machine learning-focused; physical layer signal processing perspective is limited.
**Problem:** Traditional signal processing (equalization, filtering, channel coding) may mitigate adversarial perturbations. Gap: bridge ML and signal processing communities to develop hybrid defenses.

**Opportunity:** This research project can:
- Design receiver-side signal processing (adaptive filters, MIMO techniques) to reject adversarial perturbations
- Evaluate interaction between adaptive equalization and adversarial robustness
- Propose cross-layer defense architectures (PHY + ML)

---

### **Gap 6: Adversarial Robustness in Distributed Spectrum Sensing**

**Status:** Spectrum sensing security (References 27, 28) addresses data falsification; adversarial perturbations on sensing are underexplored.
**Problem:** In cognitive radio networks with multiple distributed sensors, an adversary could inject adversarial RF signals to mislead the collective spectrum map. Gap: analyze adversarial robustness of distributed spectrum sensing under coordination/consensus constraints.

**Opportunity:** This research project can:
- Model adversarial attacks on distributed sensing networks
- Propose Byzantine-robust consensus mechanisms for spectrum sensing
- Evaluate game-theoretic aspects of adversarial spectrum sensing

---

### **Gap 7: EW-Specific Threat Models and Defenses**

**Status:** General ML adversarial robustness (References 6–8) uses ℓ₂/ℓ∞ norms; EW threat models are different.
**Problem:** In military EW context, threat constraints differ: power budgets, spectral occupancy regulations, dwell time, detectability. Generic adversarial budgets (e.g., ℓ₂ ≤ ε) may not map to realistic EW threats. Gap: formalize EW-specific threat models and evaluate robustness under these constraints.

**Opportunity:** This research project can:
- Define EW-relevant adversarial perturbation budgets (e.g., PAPR, spectral mask, detectability)
- Develop attacks and defenses specific to EW scenarios (jamming, spoofing, waveform mimicry)
- Validate findings against representative EW threat scenarios

---

---

## Recommendations for Research Direction

Based on this literature review, the recommended research trajectory is:

1. **Foundational Assessment (Phase 1):**
   - Systematically characterize vulnerability of modern RF classifiers (CNNs, CLDNN, Transformers) to FGSM, PGD, C&W attacks
   - Establish over-the-air (OTA) testbed to validate digital adversarial examples under realistic propagation
   - Quantify interaction between adversarial perturbations and channel impairments (fading, CFO, phase noise)

2. **Channel-Aware Attack Development (Phase 2):**
   - Design adversarial attacks that account for expected channel response (channel-aware perturbations)
   - Develop detection mechanisms for adversarial RF signals (extending Reference 17)
   - Establish metrics for physical realism of OTA adversarial examples

3. **Certified Robustness for RF (Phase 3):**
   - Extend randomized smoothing to RF-realistic noise models (Rayleigh/Rician fading)
   - Derive and empirically validate certified robustness bounds for modulation classifiers
   - Compare certified robustness with empirical adversarial training

4. **Practical Defense Mechanisms (Phase 4):**
   - Develop hybrid defense combining channel augmentation, adversarial training, and signal processing
   - Design SNR-adaptive defenses that preserve accuracy at low SNR while improving robustness
   - Evaluate defenses in representative EW operational scenarios

5. **EW System Integration (Phase 5):**
   - Integrate robust RF classifiers into cognitive EW frameworks
   - Assess impact of adversarial robustness on CEW decision timelines and spectrum situational awareness
   - Recommend acquisition/fielding strategies for military spectrum operations

---

## Conclusion

The literature establishes that:
- Deep learning RF classifiers are foundational for cognitive EW but are vulnerable to adversarial attacks
- Adversarial attack methodologies (FGSM, PGD, C&W) are well-established; RF-specific attacks are emerging
- Certified robustness techniques exist but are not adapted to RF channels
- Channel impairments naturally degrade some adversarial perturbations, but interactions are poorly understood
- Military spectrum operations (CEW, spectrum management) depend on robust RF ML systems

This research project addresses the critical gap at the intersection of adversarial ML and RF signal processing under realistic channel impairments, with direct relevance to Army cyber and EW operations.

---

## Key References by Category

| Category | Reference Numbers | Key Papers |
|----------|------------------|-----------|
| **RF/Modulation Classification** | 1–5 | O'Shea et al. (RadioML), CLDNN, Transformers, ResNets |
| **Adversarial Attacks** | 6–12 | Goodfellow (FGSM), Madry (PGD), Carlini–Wagner (C&W), RF-specific attacks |
| **Robustness & Defenses** | 13–17 | Randomized smoothing, adversarial training, data augmentation, detection defenses |
| **Channel-Aware Robustness** | 18–23 | RF impairments, fading channels, channel augmentation, OTA attacks |
| **EW & Cognitive Radio** | 24–28 | Cognitive EW, DARPA RF-ML, spectrum sensing, distributed defenses |
| **Advanced Topics** | 29–33 | RF fingerprinting, domain adaptation, RIS, robustness evaluation frameworks |

---

**Document Prepared for:** Cyber Defense Review 2026 EW Special Issue, U.S. Army
**Scope:** Literature review covering adversarial robustness, RF signal classification, and channel impairments
**Total References:** 33 peer-reviewed and authoritative sources
**Research Gaps Identified:** 7 major gaps with corresponding research opportunities
