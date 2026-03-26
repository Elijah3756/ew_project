"""
Dataset loader for RF modulation classification.
Supports RadioML 2016.10a, 2018.01a, and custom synthetic datasets.

NOTE on SNR methodology: RadioML samples are generated at specific SNR levels.
The SNR label (batch["snr"]) represents the native signal quality of each sample.
Evaluation scripts should FILTER by this label rather than adding additional AWGN,
to avoid double-noising. Additional channel effects (fading, CFO) may be applied
on top of the native SNR.
"""

import os
import pickle
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List, Dict


class RadioMLDataset(Dataset):
    """
    PyTorch Dataset for RadioML modulation classification data.

    Supports:
        - RadioML 2016.10a (pickle format, 11 modulations, 128 I/Q samples)
        - RadioML 2018.01a (HDF5 format, 24 modulations, 1024 I/Q samples)
          Uses lazy HDF5 loading for 2018.01a to avoid OOM on large files.
    """

    def __init__(
        self,
        data_path: str,
        dataset_version: str = "2016.10a",
        snr_range: Optional[Tuple[int, int]] = None,
        split: str = "train",
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
        transform=None,
    ):
        self.data_path = data_path
        self.dataset_version = dataset_version
        self.snr_range = snr_range
        self.split = split
        self.split_ratios = split_ratios
        self.seed = seed
        self.transform = transform

        # For 2018.01a lazy loading
        self._hdf5_file = None
        self._hdf5_X = None
        self._lazy = False

        if self.dataset_version == "2016.10a":
            self.iq_data, self.labels, self.snrs, self.mod_classes = self._load_2016()
        elif self.dataset_version == "2018.01a":
            self.labels, self.snrs, self.mod_classes = self._load_2018_metadata()
            self.iq_data = None  # loaded lazily
            self._lazy = True
        else:
            raise ValueError(f"Unsupported dataset version: {self.dataset_version}")

        self._apply_split()

    def _load_2016(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """Load RadioML 2016.10a from pickle (small enough to fit in RAM)."""
        with open(self.data_path, "rb") as f:
            data_dict = pickle.load(f, encoding="latin1")

        mod_classes = sorted(set(k[0] for k in data_dict.keys()))

        iq_list, label_list, snr_list = [], [], []

        for (mod, snr), samples in data_dict.items():
            if self.snr_range and not (self.snr_range[0] <= snr <= self.snr_range[1]):
                continue
            n = samples.shape[0]
            iq_list.append(samples)  # shape: (n, 2, 128)
            label_list.append(np.full(n, mod_classes.index(mod)))
            snr_list.append(np.full(n, snr))

        iq_data = np.concatenate(iq_list, axis=0).astype(np.float32)
        labels = np.concatenate(label_list, axis=0).astype(np.int64)
        snrs = np.concatenate(snr_list, axis=0).astype(np.float32)

        return iq_data, labels, snrs, mod_classes

    def _load_2018_metadata(self) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Load only labels and SNRs from RadioML 2018.01a HDF5.
        I/Q data (X) is accessed lazily via HDF5 indexing to avoid OOM.
        Y and Z are small enough to load into RAM (~50 MB total).
        """
        self._hdf5_file = h5py.File(self.data_path, "r")
        self._hdf5_X = self._hdf5_file["X"]  # keep dataset ref open, do NOT load [:]

        # Y and Z are small, safe to load
        labels_onehot = self._hdf5_file["Y"][:]  # (N, 24)
        snrs = self._hdf5_file["Z"][:].flatten().astype(np.float32)  # (N,)
        labels = np.argmax(labels_onehot, axis=1).astype(np.int64)

        mod_classes = [
            "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
            "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
            "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC",
            "AM-DSB-SC", "FM", "GMSK", "OQPSK"
        ]

        return labels, snrs, mod_classes

    def _apply_split(self):
        """Split data into train/val/test deterministically."""
        rng = np.random.RandomState(self.seed)
        n = len(self.labels)
        indices = rng.permutation(n)

        train_end = int(self.split_ratios[0] * n)
        val_end = train_end + int(self.split_ratios[1] * n)

        if self.split == "train":
            idx = indices[:train_end]
        elif self.split == "val":
            idx = indices[train_end:val_end]
        elif self.split == "test":
            idx = indices[val_end:]
        else:
            raise ValueError(f"Unknown split: {self.split}")

        # Apply SNR range filter AFTER split for consistency
        if self.snr_range:
            snr_mask = (self.snrs[idx] >= self.snr_range[0]) & (self.snrs[idx] <= self.snr_range[1])
            idx = idx[snr_mask]

        if not self._lazy:
            self.iq_data = self.iq_data[idx]

        self.labels = self.labels[idx]
        self.snrs = self.snrs[idx]
        self._indices = idx  # store original indices for lazy HDF5 access

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self._lazy:
            # Lazy HDF5 access: read single sample from disk
            orig_idx = int(self._indices[idx])
            iq_raw = self._hdf5_X[orig_idx]  # shape: (1024, 2)
            iq = torch.from_numpy(iq_raw.T.astype(np.float32))  # -> (2, 1024)
        else:
            iq = torch.from_numpy(self.iq_data[idx])  # (2, T)

        label = torch.tensor(self.labels[idx], dtype=torch.long)
        snr = torch.tensor(self.snrs[idx], dtype=torch.float32)

        if self.transform:
            iq = self.transform(iq)

        return {"iq": iq, "label": label, "snr": snr}

    def __del__(self):
        """Clean up HDF5 file handle."""
        if self._hdf5_file is not None:
            try:
                self._hdf5_file.close()
            except Exception:
                pass


def get_dataloaders(
    data_path: str,
    dataset_version: str = "2016.10a",
    snr_range: Optional[Tuple[int, int]] = None,
    batch_size: int = 256,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test DataLoaders.

    NOTE: num_workers is forced to 0 for 2018.01a because h5py file handles
    cannot be shared across worker processes.
    """
    # HDF5 lazy loading requires single-process access
    if dataset_version == "2018.01a":
        num_workers = 0

    loaders = {}
    for split in ["train", "val", "test"]:
        ds = RadioMLDataset(
            data_path=data_path,
            dataset_version=dataset_version,
            snr_range=snr_range,
            split=split,
            seed=seed,
        )
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=(num_workers > 0),
            drop_last=(split == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]
