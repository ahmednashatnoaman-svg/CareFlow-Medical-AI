"""Module 1: Speech Service.

Handles audio transcription, transcript normalization, and timestamps.
Integrates remote ASR endpoint (e.g. ngrok service) with fallback support.
"""

import logging
from typing import Any, Dict, Optional
import httpx
from careflow.core.arabic_text import strip_diacritics
from careflow.core.config import settings

logger = logging.getLogger(__name__)


class SpeechService:
    """Service for Speech-to-Text ASR processing."""

    def __init__(self, endpoint_url: Optional[str] = None):
        raw_url = endpoint_url or settings.ASR_ENDPOINT_URL
        base_url = raw_url.rstrip("/")
        self.endpoint_url = base_url if "/api/v1/transcribe" in base_url else f"{base_url}/api/v1/transcribe"

    async def transcribe_audio(self, audio_bytes: bytes, content_type: str = "audio/wav") -> Dict[str, Any]:
        """Transcribes Egyptian Arabic audio stream into normalized timestamped transcript.

        Args:
            audio_bytes (bytes): Raw audio binary data.
            content_type (str): MIME type of audio.

        Returns:
            Dict[str, Any]: Contains transcript text and metadata.
        """
        logger.info("Transcribing audio batch", extra={"endpoint": self.endpoint_url, "bytes_len": len(audio_bytes)})

        if not audio_bytes or len(audio_bytes) < 10:
            return {
                "transcript": "",
                "language": "ar-EG",
                "confidence": 0.0,
                "duration_sec": 0.0,
            }

        files = {"file": ("audio.wav", audio_bytes, content_type)}

        try:
            async with httpx.AsyncClient(timeout=settings.SPEECH_TIMEOUT, follow_redirects=True) as client:
                response = await client.post(self.endpoint_url, files=files)
                if response.status_code == 200:
                    data = response.json()
                    transcript_text = (
                        data.get("transcription")
                        or data.get("transcript")
                        or data.get("text")
                        or data.get("result")
                        or (data.get("data", {}).get("transcript") if isinstance(data.get("data"), dict) else None)
                        or ""
                    )
                    duration = data.get("duration_seconds") or data.get("duration_sec") or 3.0
                    return {
                        "transcript": self.normalize_transcript(str(transcript_text)),
                        "language": data.get("language", "ar-EG"),
                        "confidence": float(data.get("confidence", 0.95)),
                        "duration_sec": float(duration),
                        "filename": data.get("filename", ""),
                    }
                else:
                    logger.warning(f"Remote ASR returned HTTP status {response.status_code}: {response.text}")
        except Exception as exc:
            logger.error("Remote ASR transcription failed", exc_info=True)

        return {
            "transcript": "",
            "language": "ar-EG",
            "confidence": 0.0,
            "duration_sec": 0.0,
        }

    def normalize_transcript(self, text: str) -> str:
        """Normalizes Egyptian Arabic speech text (removing diacritics, redundant whitespace)."""
        if not text:
            return ""
        return strip_diacritics(text)
