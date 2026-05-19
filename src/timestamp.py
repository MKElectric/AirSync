import time
from dataclasses import dataclass


@dataclass(frozen=True)
class AudioChunk:
    """A timestamped chunk of PCM audio data."""
    data: bytes
    pts: float
    sequence: int
    sample_rate: int
    channels: int
    sample_width: int

    @property
    def duration(self) -> float:
        """Duration of this chunk in seconds."""
        bytes_per_sample = self.channels * self.sample_width
        total_samples = len(self.data) / bytes_per_sample
        return total_samples / self.sample_rate

    @property
    def frames(self) -> int:
        """Number of audio frames in this chunk."""
        bytes_per_frame = self.channels * self.sample_width
        return len(self.data) // bytes_per_frame


class TimestampGenerator:
    """Generates monotonically increasing presentation timestamps."""

    def __init__(self, sample_rate: int = 48000, channels: int = 2, sample_width: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self._sequence = 0
        self._base_pts: float = 0.0
        self._accumulated_duration: float = 0.0
        self._initialized = False

    def initialize(self):
        """Set the base PTS to current wall clock time."""
        self._base_pts = time.monotonic()
        self._accumulated_duration = 0.0
        self._sequence = 0
        self._initialized = True

    @property
    def current_pts(self) -> float:
        if not self._initialized:
            raise RuntimeError("TimestampGenerator not initialized")
        return self._base_pts + self._accumulated_duration

    def timestamp(self, data: bytes) -> AudioChunk:
        """Create a timestamped audio chunk."""
        if not self._initialized:
            self.initialize()

        pts = self.current_pts
        chunk = AudioChunk(
            data=data,
            pts=pts,
            sequence=self._sequence,
            sample_rate=self.sample_rate,
            channels=self.channels,
            sample_width=self.sample_width,
        )
        self._sequence += 1
        self._accumulated_duration += chunk.duration
        return chunk

    def reset(self):
        self._initialized = False
