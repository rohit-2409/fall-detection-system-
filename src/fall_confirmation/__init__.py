"""
Fall Confirmation Package (Stage 4F).

Exposes TemporalConfirmationAdapter and load_confirmation_config for modular,
reversible integration with live and offline fall-detection pipelines.
"""

from src.fall_confirmation.config_loader import load_confirmation_config
from src.fall_confirmation.temporal_adapter import TemporalConfirmationAdapter

__all__ = ["TemporalConfirmationAdapter", "load_confirmation_config"]
