import time
import threading
from typing import Optional, Callable
from src.ring_buffer import RingBuffer
from src.timestamp import AudioChunk


class PlaybackScheduler:
    """Schedules audio chunk playback based on presentation timestamps."""

    def __init__(
        self,
        buffer: RingBuffer,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width: int = 2,
        frame_size: int = 4800,
    ):
        self._buffer = buffer
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frame_size = frame_size
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[bytes, float], None]] = None
        self._lock = threading.Lock()
        self._pts_offset: float = 0.0
        self._last_pts: float = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    def set_callback(self, callback: Callable[[bytes, float], None]):
        """Set the playback callback. Called with (pcm_data, pts) when chunk is due."""
        with self._lock:
            self._callback = callback

    def set_pts_offset(self, offset: float):
        """Add a fixed delay to all timestamps (e.g., to sync with WiFi outputs)."""
        with self._lock:
            self._pts_offset = offset

    def start(self):
        """Start the playback scheduler thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the playback scheduler."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self):
        """Main scheduling loop."""
        bytes_per_frame = self.channels * self.sample_width
        frame_bytes = self.frame_size * bytes_per_frame

        while self._running:
            data = self._buffer.read(frame_bytes, block=False)
            if data is None:
                time.sleep(0.001)
                continue

            with self._lock:
                callback = self._callback
                pts_offset = self._pts_offset

            now = time.monotonic()
            pts = now + pts_offset

            if callback and len(data) > 0:
                callback(data, pts)

            self._last_pts = pts

    @property
    def buffer_fill_ms(self) -> float:
        """Current buffer fill level in milliseconds."""
        bytes_per_sec = self.sample_rate * self.channels * self.sample_width
        return (self._buffer.filled / bytes_per_sec) * 1000

    @property
    def last_pts(self) -> float:
        return self._last_pts
