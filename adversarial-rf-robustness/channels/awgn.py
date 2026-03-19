"""
Differentiable AWGN channel layer.
Adds white Gaussian noise calibrated to a target SNR.
"""

import torch
import torch.nn as nn


class AWGNChannel(nn.Module):
    """
    Additive White Gaussian Noise channel.

    Can operate in two modes:
    1. Dynamic SNR scaling (legacy): Adds noise scaled to achieve target SNR
       relative to signal power. Useful for adding noise to variable-power signals.
    2. Constant noise power (physically correct): Adds fixed thermal noise power
       independent of signal level. This models realistic receiver noise floors.

    When constant_noise_power is provided, it takes precedence and the SNR parameter
    is ignored. This is useful for modeling thermal noise after fading channels,
    where the signal attenuates but the noise floor remains constant.

    Operates on stacked I/Q tensors of shape (batch, 2, T).
    """

    def __init__(self, snr_db: float = 10.0, constant_noise_power: float = None):
        super().__init__()
        self.snr_db = snr_db
        self.constant_noise_power = constant_noise_power

    @classmethod
    def from_snr_and_signal_power(cls, snr_db: float, reference_signal_power: float):
        """
        Create an AWGNChannel with constant noise power derived from a reference
        SNR and signal power level.

        This is useful for setting up a channel where you know the SNR at a
        specific reference signal power (e.g., unfaded signal), but want to
        maintain that noise power even after fading reduces signal power.

        Args:
            snr_db: Target SNR in dB at the reference signal power.
            reference_signal_power: The signal power (linear) at which the SNR
                                     applies. Typically the power of an unfaded signal.

        Returns:
            AWGNChannel instance with constant_noise_power pre-computed.

        Example:
            # Create a channel that maintains SNR=10dB noise power, computed
            # from unfaded signal power of 0.5:
            channel = AWGNChannel.from_snr_and_signal_power(snr_db=10.0, reference_signal_power=0.5)
        """
        snr_linear = 10.0 ** (snr_db / 10.0)
        noise_power = reference_signal_power / snr_linear
        return cls(snr_db=snr_db, constant_noise_power=noise_power)

    def forward(self, x: torch.Tensor, snr_db: float = None) -> torch.Tensor:
        """
        Args:
            x: Input I/Q tensor, shape (batch, 2, T).
            snr_db: Override SNR in dB. Only used if constant_noise_power is None.
                   If None, uses self.snr_db.
        Returns:
            Noisy signal with same shape.
        """
        if self.constant_noise_power is not None:
            # Fixed thermal noise floor (physically correct for receiver thermal noise)
            noise_power = self.constant_noise_power
        else:
            # Dynamic scaling: compute noise power from current signal power (legacy behavior)
            if snr_db is None:
                snr_db = self.snr_db

            # Compute signal power per sample
            sig_power = torch.mean(x ** 2, dim=(-2, -1), keepdim=True)

            # Convert SNR from dB to linear
            snr_linear = 10.0 ** (snr_db / 10.0)

            # Noise power scaled to signal
            noise_power = sig_power / snr_linear

        # Generate noise
        noise = torch.randn_like(x) * torch.sqrt(noise_power)

        return x + noise

    def __repr__(self):
        if self.constant_noise_power is not None:
            return f"AWGNChannel(snr_db={self.snr_db}, constant_noise_power={self.constant_noise_power:.6f})"
        return f"AWGNChannel(snr_db={self.snr_db})"
