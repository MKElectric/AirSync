import time
import threading
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    """Metadata for a timestamped PCM audio segment."""
    pts: float
    sequence: int
    sample_rate: int
    channels: int
    sample_width: int
    frames: int

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        return self.frames / self.sample_rate


class TimestampGenerator:
    """Generates monotonically increasing presentation timestamps from a single clock source."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2, sample_width: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._lock = threading.Lock()
        self._sequence = 0
        self._total_samples = 0
        self._base_pts: float = 0.0
        self._initialized = False

    def initialize(self):
        """Set the base PTS to current wall clock time."""
        with self._lock:
            self._base_pts = time.monotonic()
            self._total_samples = 0
            self._sequence = 0
            self._initialized = True

    @property
    def current_pts(self) -> float:
        """Current presentation timestamp computed from base + integer sample count."""
        with self._lock:
            if not self._initialized:
                raise RuntimeError("TimestampGenerator not initialized")
            return self._base_pts + self._total_samples / self.sample_rate

    def timestamp(self, frame_count: int) -> AudioChunk:
        """Create a timestamped audio chunk from frame count (no data copy)."""
        with self._lock:
            if not self._initialized:
                self._base_pts = time.monotonic()
                self._total_samples = 0
                self._sequence = 0
                self._initialized = True

            pts = self._base_pts + self._total_samples / self.sample_rate
            chunk = AudioChunk(
                pts=pts,
                sequence=self._sequence,
                sample_rate=self.sample_rate,
                channels=self.channels,
                sample_width=self.sample_width,
                frames=frame_count,
            )
            self._sequence += 1
            self._total_samples += frame_count
            return chunk

    def reset(self):
        with self._lock:
            self._initialized = False
