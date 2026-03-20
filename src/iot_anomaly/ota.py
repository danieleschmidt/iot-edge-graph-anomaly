"""OTA (Over-The-Air) model deployer with versioning."""

import json
import os
from pathlib import Path
from typing import Optional
import numpy as np

from .lstm_autoencoder import LSTMAutoencoder


class OTADeployer:
    """Simple OTA model versioning and deployment using numpy .npz format.
    
    Args:
        model_dir: Directory where model versions are stored.
    """

    def __init__(self, model_dir: str = "/tmp/ota_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self._manifest_path = self.model_dir / "manifest.json"

    def _load_manifest(self) -> list:
        if self._manifest_path.exists():
            with open(self._manifest_path) as f:
                return json.load(f)
        return []

    def _save_manifest(self, entries: list):
        with open(self._manifest_path, "w") as f:
            json.dump(entries, f, indent=2)

    def save_model(
        self,
        model: LSTMAutoencoder,
        version: str,
        metadata: Optional[dict] = None,
    ) -> str:
        """Save a model version to disk.
        
        Args:
            model: LSTMAutoencoder instance to save.
            version: Version string (e.g. "v1.0.0").
            metadata: Optional dict of metadata to store alongside model.
        
        Returns:
            Path to saved .npz file.
        """
        weights = model.get_weights()
        save_path = self.model_dir / f"{version}.npz"
        np.savez(str(save_path), **weights)

        # Update manifest
        entries = self._load_manifest()
        entry = {
            "version": version,
            "path": str(save_path),
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "seq_len": model.seq_len,
            "metadata": metadata or {},
        }
        # Replace existing entry for same version or append
        versions = [e["version"] for e in entries]
        if version in versions:
            entries[versions.index(version)] = entry
        else:
            entries.append(entry)
        self._save_manifest(entries)
        return str(save_path)

    def load_model(self, version: str) -> LSTMAutoencoder:
        """Load a model by version string.
        
        Args:
            version: Version string (e.g. "v1.0.0").
        
        Returns:
            LSTMAutoencoder with loaded weights.
        """
        entries = self._load_manifest()
        entry = next((e for e in entries if e["version"] == version), None)
        if entry is None:
            raise KeyError(f"Version {version!r} not found in manifest.")

        weights = dict(np.load(entry["path"], allow_pickle=False))
        model = LSTMAutoencoder(
            input_dim=entry["input_dim"],
            hidden_dim=entry["hidden_dim"],
            seq_len=entry["seq_len"],
        )
        model.set_weights(weights)
        return model

    def list_versions(self) -> list:
        """Return list of all saved version dicts (version, path, metadata, ...)."""
        return self._load_manifest()

    def get_latest_version(self) -> str:
        """Return the version string of the most recently saved model.
        
        Returns:
            Version string.
        
        Raises:
            ValueError: If no versions have been saved.
        """
        entries = self._load_manifest()
        if not entries:
            raise ValueError("No model versions saved yet.")
        return entries[-1]["version"]
