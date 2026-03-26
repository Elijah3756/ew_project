# RF Modulation Classification Dataset Comparison

## Summary Table

| Dataset | Modulations | Samples | SNR Range | Sample Length | Channel Models | PyTorch Support | Best For |
|---------|------------|---------|-----------|---------------|----------------|-----------------|----------|
| RadioML 2016.10a | 11 | 220K | -20 to +18 dB | 128 | AWGN only | Yes (RFML lib) | Fast prototyping, baseline comparability |
| RadioML 2016.10b | 10 | 60K | -20 to +18 dB | 128 | AWGN only | Yes (RFML lib) | Smallest option |
| RadioML 2018.01a | 24 | 2.5M | -20 to +30 dB | 1024 | AWGN, fading, offsets | Yes (RFML lib) | Comprehensive evaluation |
| HisarMod 2019.1 | 26 | 780K | -20 to +18 dB | 1024 | 5 fading types | Limited | Realistic channel testing |
| TorchSig (Sig53) | 50+ | Configurable | Configurable | 4096 | Configurable | Native PyTorch | Modern workflows, custom experiments |

## Recommendation for This Project

**Primary: RadioML 2016.10a**
- Most cited in adversarial RF literature (direct comparability)
- Fast to train (128 samples, 220K total)
- Well-understood baselines exist
- 11 modulations is manageable for initial experiments
- Known PyTorch loaders available

**Secondary (if time allows): RadioML 2018.01a**
- 24 modulations, 1024-length windows
- Larger scale validates generalization claims
- Already includes some channel effects in generation
- Stronger paper if results hold across both

## Download Sources

### RadioML 2016.10a
- Direct: http://opendata.deepsig.io/datasets/2016.10/RML2016.10a.tar.bz2
- GitHub mirrors: github.com/ianblenke/deepsig_dataset
- Format: Python pickle (dict keyed by (modulation, SNR))
- License: CC BY-NC-SA 4.0

### RadioML 2018.01a
- Kaggle: kaggle.com/datasets/aleksandrdubrovin/deepsigio-radioml-201801a-new
- Format: HDF5 (X=IQ, Y=labels, Z=SNR)
- Size: ~5 GB compressed

### TorchSig
- pip install torchsig
- GitHub: github.com/TorchDSP/torchsig
- Generates data on-the-fly (no download needed)

## Known Limitations of RadioML 2016.10a
1. AWGN-only channel model (we add our own channels -- this is actually an advantage for controlled experiments)
2. AM-SSB encoding issues documented in literature
3. SNR values may not perfectly match stated values
4. Short windows (128 samples) limit temporal feature extraction
5. Synthetic data only (no over-the-air captures)

## Data Format Details

### RadioML 2016.10a
```python
# Pickle dict: {(modulation_str, snr_int): np.array(N, 2, 128)}
# Modulations: 8PSK, AM-DSB, AM-SSB, BPSK, CPFSK, GFSK, PAM4, QAM16, QAM64, QPSK, WBFM
# SNR values: -20, -18, -16, ..., 16, 18 (20 levels, 2 dB steps)
# 1000 samples per (mod, SNR) pair = 11 * 20 * 1000 = 220,000 total
```

### RadioML 2018.01a
```python
# HDF5 file with datasets:
# X: shape (2555904, 1024, 2) -- I/Q data
# Y: shape (2555904, 24) -- one-hot labels
# Z: shape (2555904, 1) -- SNR values
# 24 modulations, SNR: -20 to +30 dB (26 levels, 2 dB steps)
# 4096 samples per (mod, SNR) pair
```
