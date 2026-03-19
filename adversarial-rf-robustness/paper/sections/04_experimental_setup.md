# 4. Experimental Setup

## 4.1 Datasets

We evaluate on two widely adopted benchmarks for automatic modulation classification (AMC), enabling analysis of how adversarial vulnerability scales with task complexity.

**RadioML 2016.10a.** The dataset contains 220,000 I/Q sample windows across 11 modulation schemes: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, 16-QAM, 64-QAM, QPSK, and WBFM. Each sample consists of 128 complex baseband time-domain samples represented as a (2, 128) tensor of in-phase and quadrature components. Samples span 20 SNR levels from -20 dB to +18 dB in 2 dB increments, with 1,000 samples per (modulation, SNR) pair. We filter to the SNR range [-10, 18] dB, yielding 165,000 samples.

**RadioML 2018.01a.** The dataset contains over 2.5 million I/Q sample windows across 24 modulation schemes: OOK, 4ASK, 8ASK, BPSK, QPSK, 8PSK, 16PSK, 32PSK, 16APSK, 32APSK, 64APSK, 128APSK, 16QAM, 32QAM, 64QAM, 128QAM, 256QAM, AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC, FM, GMSK, and OQPSK. Each sample consists of 1024 complex baseband time-domain samples represented as a (2, 1024) tensor. Samples span 26 SNR levels from -20 dB to +30 dB in 2 dB increments. This dataset presents a substantially harder classification task with more than twice as many modulation classes and eight times the temporal resolution per sample.

Both datasets apply a fixed 70/15/15 train/validation/test split with deterministic seeding (seed = 42) for reproducibility. All experiments use the same splits.

## 4.2 Model Architecture

Our baseline classifier is a four-block 1D convolutional neural network (CNN) operating on the (2, 128) I/Q input. The architecture consists of:

- **Block 1:** Conv1d(2, 64, kernel=7) -> BatchNorm -> ReLU -> MaxPool(2)
- **Block 2:** Conv1d(64, 128, kernel=5) -> BatchNorm -> ReLU -> MaxPool(2)
- **Block 3:** Conv1d(128, 128, kernel=3) -> BatchNorm -> ReLU -> MaxPool(2)
- **Block 4:** Conv1d(128, 64, kernel=3) -> BatchNorm -> ReLU -> AdaptiveAvgPool(1)
- **Classifier:** Linear(64, 128) -> ReLU -> Dropout(0.5) -> Linear(128, K)

where K = 11 for RadioML 2016.10a and K = 24 for RadioML 2018.01a. The 2016 model contains 126,475 trainable parameters; the 2018 model contains 128,152 (the increase reflects the larger output layer). The same architecture is used for both datasets, with the input layer adapting to the respective sample lengths (128 vs. 1024). This architecture is inspired by VT-CNN2 with modern improvements (batch normalization, adaptive pooling) and is representative of lightweight classifiers suitable for edge deployment.

## 4.3 Training Configuration

All models are trained using the Adam optimizer with learning rate 0.001, weight decay 1e-4, and cosine annealing learning rate schedule over 60 epochs. Batch size is 256. We report the best validation accuracy checkpoint.

For channel-augmented training, each training batch is passed through a random composite channel (Rayleigh fading + CFO) before being fed to the model. No additional AWGN is applied during training augmentation because RadioML samples already contain AWGN at their labeled SNR; applying additional AWGN would cause double-noising and degrade training quality.

For adversarial training, we use PGD with 5 inner steps and rho = 0.01 to generate adversarial examples during training, with alpha = 0.5 balancing clean and adversarial loss terms.

## 4.4 Channel Configurations

We evaluate under three channel conditions of increasing severity:

| Config | Components | Parameters |
|--------|-----------|------------|
| AWGN | Additive noise only | SNR sweep: -10 to 18 dB |
| Rayleigh + AWGN | Block Rayleigh fading + AWGN | Fading: CN(0,1), SNR sweep |
| Rayleigh + CFO + AWGN | Block fading + frequency offset + AWGN | CFO: U[-0.01, 0.01], SNR sweep |

Channel impairments are applied as differentiable PyTorch layers, enabling gradient-based attacks to account for channel effects.

**Evaluation methodology.** A critical detail of our evaluation is that RadioML samples already contain AWGN at their labeled SNR. To avoid double-noising (applying additional AWGN to already-noisy samples), we evaluate at each SNR by *filtering* test samples by their native SNR labels (with +/- 1 dB tolerance) rather than re-applying AWGN. For the Rayleigh and CFO channel conditions, only the non-AWGN impairments (fading and frequency offset) are applied on top of the native samples, since the additive noise component is already present in the data. This approach ensures that the evaluated SNR matches the dataset's labeled conditions and avoids the evaluation artifact that would inflate both clean accuracy degradation and attack success rates.

**Fading channel approximation.** A consequence of RadioML's baked-in AWGN is that when Rayleigh fading is applied, the fading multiplier acts on the combined signal-plus-noise: h * (x_clean + n) rather than the physically correct (h * x_clean) + n. In a real system, thermal noise is generated at the receiver and is not attenuated by the channel. Our approximation means that fading does not degrade the effective SNR as severely as it would in reality, because the noise is attenuated along with the signal. This makes our fading-channel results optimistic for the classifier (fading appears less harmful than in practice). However, the relative comparison between clean and adversarial performance under fading remains valid, as both are subject to the same approximation. Similarly, the adversary computes gradients on the noisy sample (signal plus native AWGN) rather than the clean signal alone; a real pre-channel attacker would not have access to the receiver's thermal noise realization. This effect is minor at high SNR but more significant at low SNR where the noise component is larger relative to the signal.

## 4.5 Attack Configuration

Both FGSM and PGD attacks use the L2 norm-ratio constraint ||delta||_2 / ||x||_2 <= rho. We evaluate four perturbation budgets: rho in {0.005, 0.01, 0.02, 0.05} (0.5%, 1%, 2%, 5% norm ratio, corresponding to power ratios of rho^2 = -46 dB to -26 dB). PGD uses 10 steps with step size rho/4 and random initialization.

For the primary "pre-channel" attack, the adversary perturbs the signal before channel propagation. The attacked signal then passes through the same stochastic channel as the clean signal before reaching the classifier.

## 4.6 Evaluation Metrics

- **Clean Accuracy:** Classification accuracy on unperturbed signals, evaluated per-SNR.
- **Robust Accuracy:** Classification accuracy on adversarially perturbed signals, evaluated per-SNR.
- **Attack Success Rate (ASR):** Fraction of correctly classified clean samples that become misclassified after attack.
- **Training Time:** Total GPU hours for model training.
- **Inference Latency:** Milliseconds per sample at inference.
- **Parameter Count:** Total trainable parameters.

All stochastic evaluations (involving Rayleigh fading, CFO, or adversarial attacks with random initialization) are averaged over 10 independent trials. We report means, standard deviations, and 95% confidence intervals computed via Student's t-distribution. Under AWGN-only conditions (no stochastic channel), the evaluation is deterministic and confidence intervals are zero.

## 4.7 Reproducibility

All experiments use fixed random seeds. Code, model checkpoints, and configuration files are provided. Experiments are implemented in PyTorch and executed on GPU.
