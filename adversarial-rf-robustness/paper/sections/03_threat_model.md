# 3. Problem Formulation and Threat Model

## 3.1 Modulation Classification Task

We consider the supervised classification of RF modulation schemes from complex baseband I/Q samples. Let x in R^{2 x N} denote a windowed observation of N time-domain samples, where the two channels represent the in-phase (I) and quadrature (Q) components. A deep neural network classifier f_theta maps each observation to a predicted modulation class:

    f_theta(x) -> y,  y in {1, ..., K}

where K is the number of modulation classes and theta denotes the learned parameters.

## 3.2 Channel Model

The received signal passes through a composite wireless channel h(.) before reaching the classifier. We model three impairment types:

**AWGN.** Additive white Gaussian noise is added to achieve a target SNR:

    r = x + n,  where n ~ N(0, sigma^2 I)

and sigma^2 is calibrated such that SNR = P_signal / P_noise at the specified level in dB.

**Rayleigh Fading.** Flat (block) Rayleigh fading applies a random complex gain:

    r = h * x,  where h ~ CN(0, 1)

This models the multiplicative distortion caused by multipath propagation in non-line-of-sight environments. The complex multiplication is applied to the stacked I/Q representation via the standard real-valued equivalent.

**Carrier Frequency Offset (CFO).** A normalized frequency offset delta_f introduces a time-varying phase rotation:

    r[n] = x[n] * exp(j * 2 * pi * delta_f * n)

where delta_f is drawn uniformly from [-delta_max, delta_max]. This models oscillator mismatch between transmitter and receiver.

**Composite Channel.** The full channel applies impairments in sequence: Rayleigh fading, then CFO, then AWGN. This ordering reflects the physical signal path: the transmitted signal experiences multiplicative fading and frequency offset during propagation, followed by additive receiver noise.

## 3.3 Adversarial Threat Model

We consider an adversary who introduces a perturbation delta to the transmitted signal before it passes through the channel. The receiver observes:

    r = h(x + delta)

This "pre-channel" attack models an adversary who can manipulate the signal at or near the transmitter, or who injects a low-power interfering waveform into the spectrum. This is the operationally relevant scenario for electronic attack.

**Adversary Knowledge.** We primarily evaluate the white-box setting, where the adversary has full knowledge of the classifier architecture, weights, and the channel model used during training, but no channel state information (no-CSI). This is a strong attack model that provides upper bounds on vulnerability under pre-channel attacks without perfect channel estimation. A perfect-CSI adversary with the ability to estimate or observe the actual channel realization would be strictly stronger, but channel estimation capability is less realistic in many EW scenarios. We note that transfer attacks (black-box) are a natural extension but are deferred to future work unless a second model architecture is evaluated.

**Channel interaction.** Under this pre-channel attack model, the adversarial perturbation is subject to the same propagation effects as the signal itself. In the AWGN-only case, the perturbation passes through the channel with minimal distortion. Under fading channels, the PGD optimization uses a single stochastic channel sample per gradient step (a noisy single-sample approximation of Expectation over Transformation, not a full multi-sample EoT average), while the actual adversary evaluates against independent realizations at test time. This mismatch between the training-time channel samples used in gradient computation and the independent test-time channel realizations introduces a decorrelation between the crafted perturbation direction and the received signal geometry, a key factor in our experimental results, where fading channels provide substantial accidental robustness against pre-channel attacks.

**Perturbation Constraint.** Rather than adopting the L_inf or fixed-epsilon L_2 norms common in computer vision, we impose a power-ratio constraint that is physically meaningful in RF:

    ||delta||_2 / ||x||_2 <= rho

where rho represents the maximum ratio of perturbation L2 norm to signal L2 norm. The corresponding power ratios are given by rho^2; we evaluate rho in {0.5%, 1%, 2%, 5%}, corresponding to perturbation-to-signal power ratios of -46 dB to -26 dB (computed as 10*log10(rho^2)). These levels represent perturbations that would be difficult to detect via conventional energy detection.

**Rationale.** This constraint is more appropriate than norm-based bounds for three reasons: (1) it scales with signal energy, making the budget invariant to signal amplitude normalization; (2) it directly corresponds to a physical quantity -- interference-to-signal ratio -- that EW operators can reason about; and (3) it produces markedly different (and more operationally realistic) attack success rates than the L-inf or fixed-epsilon L-2 constraints used in prior RF adversarial studies [9, 10]. As we demonstrate in Section 5, this choice of constraint is a primary factor in the discrepancy between our results and previously reported attack effectiveness.

## 3.4 Attack Algorithms

We implement two standard gradient-based attacks adapted for the I/Q domain:

**FGSM (Fast Gradient Sign Method).** A single-step attack that computes the adversarial perturbation as:

    delta = epsilon * grad_x L(f_theta(x), y) / ||grad_x L||_2

where epsilon = rho * ||x||_2 and L is the cross-entropy loss. The gradient is L2-normalized (rather than sign-based) to produce a perturbation aligned with the power-ratio constraint.

**PGD (Projected Gradient Descent).** A multi-step iterative attack that applies T gradient ascent steps with projection back onto the constraint set after each step:

    delta_{t+1} = Pi_{B(rho)} [ delta_t + alpha * grad_delta L / ||grad_delta L||_2 ]

where alpha is the step size, T = 10 steps, and Pi projects onto the L2 ball of radius rho * ||x||_2. PGD with random initialization is widely considered the standard benchmark for adversarial robustness evaluation.

## 3.5 Defense Strategies

We evaluate three defense mechanisms chosen for their computational feasibility in edge-deployed EW systems:

**Channel Augmentation Training.** The model is trained with on-the-fly random channel impairments (Rayleigh fading and CFO) applied to each training batch. Because RadioML samples already contain AWGN at their labeled SNR, no additional AWGN is applied during augmentation to avoid double-noising (see Section 4.4). This implicitly regularizes the model against input perturbations by exposing it to a distribution of multiplicative signal distortions during training.

**Adversarial Training (PGD-lite).** Following Madry et al., we augment each training batch with adversarial examples generated by a reduced-step PGD attack (3-5 steps). The training loss combines clean and adversarial components:

    L_total = alpha * L(f(x), y) + (1 - alpha) * L(f(x + delta), y)

where alpha = 0.5 balances clean accuracy and robustness.

**Noise Injection (Gaussian Data Augmentation).** Gaussian noise is added to the input during training. This exposes the model to a distribution of input perturbations, building tolerance to small adversarial deviations. The trained model is evaluated normally at inference (single forward pass, no additional noise). While this training procedure is inspired by randomized smoothing, we do not implement the full Monte Carlo inference procedure; our evaluation measures only the robustness gained from noise-augmented training.
