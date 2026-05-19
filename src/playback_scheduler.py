import time
import threading
from typing import Optional, Callable
from src.ring_buffer import RingBuffer
from src.timestamp import TimestampGenerator


class PlaybackScheduler:
    """Schedules audio playback based on presentation timestamps from a shared clock.

    Reads (data, pts) pairs from the ring buffer, waits until the PTS time arrives,
    then dispatches to the callback. This ensures frames are played at their
    intended presentation time, not as fast as possible.
    """

    def __init__(
        self,
        buffer: RingBuffer,
        timestamp_gen: TimestampGenerator,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width: int = 2,
        frame_size: int = 4800,
        min_fill_frames: int = 9600,
    ):
        self._buffer = buffer
        self._timestamp_gen = timestamp_gen
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.frame_size = frame_size
        self.min_fill_frames = min_fill_frames

        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[bytes, float], None]] = None
        self._lock = threading.Lock()
        self._pts_offset: float = 0.0
        self._last_pts: float = 0.0
        self._underrun_callback: Optional[Callable[[], None]] = None
        self._was_underrun = False
        self._started = False

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def set_callback(self, callback: Callable[[bytes, float], None]):
        """Set the playback callback. Called with (pcm_data, pts) when chunk is due."""
        with self._lock:
            self._callback = callback

    def set_underrun_callback(self, callback: Callable[[], None]):
        """Called when buffer empties and playback stalls."""
        with self._lock:
            self._underrun_callback = callback

    def set_pts_offset(self, offset: float):
        """Add a fixed delay to all timestamps (e.g., to sync with WiFi outputs)."""
        with self._lock:
            self._pts_offset = offset

    def start(self):
        """Start the playback scheduler thread."""
        if self._running.is_set():
            return
        self._running.set()
        self._was_underrun = False
        self._started = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop the playback scheduler."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self):
        """Main scheduling loop: reads frames with PTS, waits for presentation time, dispatches."""
        bytes_per_frame = self.channels * self.sample_width
        frame_bytes = self.frame_size * bytes_per_frame
        min_fill_bytes = self.min_fill_frames * bytes_per_frame

        while self._running.is_set():
            data, pts = self._buffer.read_with_pts(frame_bytes, block=True, timeout=0.1)
            if data is None:
                if self._running.is_set() and not self._was_underrun:
                    self._was_underrun = True
                    with self._lock:
                        cb = self._underrun_callback
                    if cb:
                        cb()
                continue

            self._was_underrun = False

            if not self._started:
                if self._buffer.filled < min_fill_bytes:
                    continue
                self._started = True

            with self._lock:
                callback = self._callback
                pts_offset = self._pts_offset

            target_time = pts + pts_offset
            now = time.monotonic()
            sleep_time = target_time - now

            if sleep_time > 0:
                time.sleep(sleep_time)

            actual_pts = time.monotonic()

            if callback and len(data) > 0:
                callback(data, actual_pts)

            self._last_pts = actual_pts

    @property
    def buffer_fill_ms(self) -> float:
        """Current buffer fill level in milliseconds."""
        bytes_per_sec = self.sample_rate * self.channels * self.sample_width
        return (self._buffer.filled / bytes_per_sec) * 1000

    @property
    def last_pts(self) -> float:
        return self._last_pts
