"""
Stage 6I Notification Configuration and Credential Isolation.
Standalone Fall Detection Project.

Strict Safety Rules:
- Credentials loaded ONLY from environment variables.
- NEVER hardcode credentials.
- NEVER log secret values (auth tokens, passwords, headers).
- Phone numbers MUST be masked in all logs (e.g., +91*****1234).
- If credentials are missing, system gracefully degrades to DISABLED state without crashing.
"""

import logging
import os
from typing import Optional

logger = logging.getLogger("Stage6I_NotificationConfig")


def mask_phone_number(phone: Optional[str]) -> str:
    """
    Masks recipient and sender phone numbers for secure logging.
    
    Example:
        '+919876543210' -> '+91*****3210'
        'whatsapp:+919876543210' -> 'whatsapp:+91*****3210'
    """
    if not phone:
        return "[UNSET]"
    
    prefix = ""
    clean = str(phone).strip()
    if clean.lower().startswith("whatsapp:"):
        prefix = clean[:9]
        clean = clean[9:]
        
    if len(clean) >= 7:
        masked = clean[:3] + "*****" + clean[-4:]
    elif len(clean) >= 4:
        masked = clean[:1] + "***" + clean[-2:]
    else:
        masked = "***"
        
    return prefix + masked


def mask_secret(secret: Optional[str]) -> str:
    """Redacts secret credentials completely."""
    if not secret:
        return "[UNSET]"
    return "[REDACTED]"


class NotificationConfig:
    """
    Isolated configuration container for WhatsApp notifications.
    Safely parses environment variables without exposing sensitive values.
    """

    def __init__(
        self,
        provider: str = "twilio",
        account_sid: Optional[str] = None,
        auth_token: Optional[str] = None,
        whatsapp_from: Optional[str] = None,
        whatsapp_recipient: Optional[str] = None,
        timeout_seconds: float = 10.0,
        queue_maxsize: int = 100,
        db_path: Optional[str] = None,
        max_attempts: int = 1,
        retry_base_seconds: float = 5.0,
        retry_max_seconds: float = 300.0
    ):
        self.provider = (provider or "twilio").lower().strip()
        self.account_sid = account_sid.strip() if account_sid else None
        self.auth_token = auth_token.strip() if auth_token else None
        self.whatsapp_from = whatsapp_from.strip() if whatsapp_from else None
        self.whatsapp_recipient = whatsapp_recipient.strip() if whatsapp_recipient else None
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.queue_maxsize = max(10, int(queue_maxsize))
        self.db_path = db_path
        self.max_attempts = max(1, int(max_attempts))
        self.retry_base_seconds = max(0.1, float(retry_base_seconds))
        self.retry_max_seconds = max(self.retry_base_seconds, float(retry_max_seconds))

        # Validate configuration completeness
        self.enabled = False
        self.status = "DISABLED"
        self._validate_and_set_status()

    def _validate_and_set_status(self) -> None:
        """Determines if the provider has all necessary parameters."""
        if self.provider == "twilio":
            if bool(self.account_sid and self.auth_token and self.whatsapp_from and self.whatsapp_recipient):
                self.enabled = True
                self.status = "CONFIGURED"
            else:
                self.enabled = False
                self.status = "DISABLED"
        elif self.provider in ("fake", "mock"):
            # Mock provider for testing without real credentials
            self.enabled = True
            self.status = "CONFIGURED"
        else:
            self.enabled = False
            self.status = "DISABLED"

    @classmethod
    def from_env(cls) -> "NotificationConfig":
        """
        Loads configuration strictly from environment variables.
        Never crashes on missing variables; defaults to DISABLED.
        """
        provider = os.environ.get("WHATSAPP_PROVIDER", "twilio")
        account_sid = os.environ.get("TWILIO_ACCOUNT_SID")
        auth_token = os.environ.get("TWILIO_AUTH_TOKEN")
        whatsapp_from = os.environ.get("TWILIO_WHATSAPP_FROM")
        whatsapp_recipient = os.environ.get("WHATSAPP_RECIPIENT")
        
        try:
            timeout_seconds = float(os.environ.get("NOTIFICATION_TIMEOUT", 10.0))
        except (ValueError, TypeError):
            timeout_seconds = 10.0

        try:
            queue_maxsize = int(os.environ.get("NOTIFICATION_QUEUE_MAXSIZE", 100))
        except (ValueError, TypeError):
            queue_maxsize = 100

        try:
            max_attempts = int(os.environ.get("NOTIFICATION_MAX_ATTEMPTS", 5))
        except (ValueError, TypeError):
            max_attempts = 5

        try:
            retry_base_seconds = float(os.environ.get("NOTIFICATION_RETRY_BASE_SECONDS", 5.0))
        except (ValueError, TypeError):
            retry_base_seconds = 5.0

        try:
            retry_max_seconds = float(os.environ.get("NOTIFICATION_RETRY_MAX_SECONDS", 300.0))
        except (ValueError, TypeError):
            retry_max_seconds = 300.0

        db_path = os.environ.get("NOTIFICATION_DB_PATH")

        instance = cls(
            provider=provider,
            account_sid=account_sid,
            auth_token=auth_token,
            whatsapp_from=whatsapp_from,
            whatsapp_recipient=whatsapp_recipient,
            timeout_seconds=timeout_seconds,
            queue_maxsize=queue_maxsize,
            db_path=db_path,
            max_attempts=max_attempts,
            retry_base_seconds=retry_base_seconds,
            retry_max_seconds=retry_max_seconds
        )

        if not instance.enabled and instance.provider == "twilio":
            logger.warning("WhatsApp notifications disabled: required configuration is missing.")
        elif instance.enabled:
            logger.info(
                "WhatsApp notifications active [provider: %s, recipient: %s, max_attempts: %d].",
                instance.provider,
                mask_phone_number(instance.whatsapp_recipient),
                instance.max_attempts
            )

        return instance

    def __repr__(self) -> str:
        return (
            f"<NotificationConfig provider={self.provider} status={self.status} "
            f"recipient={mask_phone_number(self.whatsapp_recipient)} "
            f"from={mask_phone_number(self.whatsapp_from)} "
            f"timeout={self.timeout_seconds}s "
            f"max_attempts={self.max_attempts}>"
        )
