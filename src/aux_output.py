import time
import threading
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Callable
from collections import deque


class AudioSink(ABC):
    """Abstract audio output sink."""

    @abstractmethod
    def start(self, sample_rate: int, channels: int, sample_width: int):
        pass

    @abstractmethod
    def write(self, data: bytes) -> bool:
        """Write PCM data to the sink. Returns False on failure."""
        pass

    @abstractmethod
    def stop(self):
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Check if the sink is operational."""
        pass


class NullSink(AudioSink):
    """Sink that discards audio. Used for testing."""

    def __init__(self):
        self._started = False

    def start(self, sample_rate: int, channels: int, sample_width: int):
        self._started = True

    def write(self, data: bytes) -> bool:
        return self._started

    def stop(self):
        self._started = False

    def is_healthy(self) -> bool:
        return self._started


class PipeWireSink(AudioSink):
    """Audio sink that outputs to PipeWire via pw-play subprocess."""

    def __init__(self, device: Optional[str] = None, node_name: str = "AirSync-AUX"):
        self._device = device
        self._node_name = node_name
        self._process: Optional[subprocess.Popen] = None
        self._healthy = False

    def start(self, sample_rate: int, channels: int, sample_width: int):
        args = [
            "pw-play", "-",
            "--rate", str(sample_rate),
            "--channels", str(channels),
            "--format", "s16",
            "-P", f"node.name={self._node_name}",
        ]
        if self._device:
            args.extend(["--target", self._device])

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._healthy = True

    def write(self, data: bytes) -> bool:
        if not self._process or not self._healthy:
            return False
        try:
            self._process.stdin.write(data)
            self._process.stdin.flush()
            return True
        except (BrokenPipeError, OSError):
            self._healthy = False
            return False

    def stop(self):
        if self._process:
            try:
                self._process.stdin.close()
            except Exception:
                pass
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
        self._healthy = False

    def is_healthy(self) -> bool:
        if not self._process:
            return False
        if self._process.poll() is not None:
            self._healthy = False
        return self._healthy


class AuxOutputAdapter:
    """AUX playback adapter with configurable latency, underrun protection, and master clock sync.

    Receives scheduled (pcm_data, pts) tuples from PlaybackScheduler,
    holds them in a delay line to match WiFi latency, then outputs to an AudioSink.
    """

    def __init__(
        self,
        sink: AudioSink,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width: int = 2,
        target_latency_ms: float = 2000.0,
        frame_size: int = 4800,
    ):
        self.sink = sink
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.target_latency_ms = target_latency_ms
        self.frame_size = frame_size

        self._bytes_per_frame = channels * sample_width * frame_size
        self._target_frames = int(target_latency_ms / 1000.0 * sample_rate / frame_size)

        self._delay_buffer: deque = deque()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._underrun_count = 0
        self._frame_count = 0
        self._underrun_callback: Optional[Callable[[], None]] = None
        self._status_callback: Optional[Callable[[dict], None]] = None

    def set_underrun_callback(self, callback: Callable[[], None]):
        self._underrun_callback = callback

    def set_status_callback(self, callback: Callable[[dict], None]):
        """Called periodically with adapter status."""
        self._status_callback = callback

    def set_latency(self, latency_ms: float):
        """Reconfigure target latency."""
        self.target_latency_ms = latency_ms
        self._target_frames = int(latency_ms / 1000.0 * self.sample_rate / self.frame_size)

    def on_scheduled_frame(self, data: bytes, pts: float):
        """Callback from PlaybackScheduler. Queues frame for delayed playback."""
        with self._lock:
            self._delay_buffer.append((data, pts))

    def start(self):
        if self._running.is_set():
            return

        self.sink.start(self.sample_rate, self.channels, self.sample_width)
        self._running.set()
        self._underrun_count = 0
        self._frame_count = 0
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.sink.stop()

    def _playback_loop(self):
        """Main loop: drains delay buffer, detects underruns, outputs to sink."""
        silence = bytes(self._bytes_per_frame)

        while self._running.is_set():
            frame = self._next_frame()

            if frame is None:
                self._handle_underrun()
                self.sink.write(silence)
                self._emit_status()
                continue

            data, pts = frame
            self._frame_count += 1

            if not self.sink.write(data):
                self._handle_underrun()

            self._emit_status()

    def _next_frame(self) -> Optional[tuple]:
        """Get the next frame ready for playback, or None if buffer is empty."""
        with self._lock:
            if not self._delay_buffer:
                return None

            data, pts = self._delay_buffer[0]

            if len(self._delay_buffer) < self._target_frames:
                return None

            self._delay_buffer.popleft()
            return data, pts

    def _handle_underrun(self):
        """Handle an underrun condition."""
        self._underrun_count += 1
        if self._underrun_callback:
            self._underrun_callback()

    def _emit_status(self):
        """Emit status update if callback is set."""
        if not self._status_callback:
            return

        with self._lock:
            buffer_fill = len(self._delay_buffer)

        fill_ms = buffer_fill * self.frame_size / self.sample_rate * 1000

        self._status_callback({
            "buffer_fill_frames": buffer_fill,
            "buffer_fill_ms": fill_ms,
            "target_latency_ms": self.target_latency_ms,
            "underrun_count": self._underrun_count,
            "frames_played": self._frame_count,
            "sink_healthy": self.sink.is_healthy(),
        })

    @property
    def buffer_fill_frames(self) -> int:
        with self._lock:
            return len(self._delay_buffer)

    @property
    def buffer_fill_ms(self) -> float:
        with self._lock:
            return len(self._delay_buffer) * self.frame_size / self.sample_rate * 1000

    @property
    def underrun_count(self) -> int:
        return self._underrun_count

    @property
    def frames_played(self) -> int:
        return self._frame_count
