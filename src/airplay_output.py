import time
import math
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Optional, Callable

from src.aux_output import AudioSink, NullSink


class DriftCompensator:
    """SRC-based drift compensation that adjusts playback speed.

    Instead of inserting/deleting samples (which causes clicks),
    this adjusts the effective playback rate by stretching/compressing
    audio via a simple linear interpolation resampler.
    """

    MIN_SPEED = 0.995
    MAX_SPEED = 1.005
    DEFAULT_SPEED = 1.0

    def __init__(self, channels: int = 2, sample_width: int = 2):
        self._channels = channels
        self._sample_width = sample_width
        self._speed = self.DEFAULT_SPEED
        self._fractional_pos = 0.0
        self._carryover: Optional[bytes] = None

    @property
    def speed(self) -> float:
        return self._speed

    @speed.setter
    def speed(self, value: float):
        self._speed = max(self.MIN_SPEED, min(self.MAX_SPEED, value))

    def reset(self):
        self._speed = self.DEFAULT_SPEED
        self._fractional_pos = 0.0
        self._carryover = None

    def process(self, data: bytes) -> bytes:
        """Apply speed adjustment via linear interpolation resampling.

        If speed < 1.0, output is shorter (playback faster, buffer drains).
        If speed > 1.0, output is longer (playback slower, buffer fills).
        """
        if self._speed == self.DEFAULT_SPEED and self._carryover is None:
            return data

        if self._speed == self.DEFAULT_SPEED:
            result = self._carryover + data if self._carryover else data
            self._carryover = None
            return result

        bytes_per_sample = self._channels * self._sample_width
        input_samples = len(data) // bytes_per_sample

        if self._carryover:
            carry_samples = len(self._carryover) // bytes_per_sample
            input_samples += carry_samples
            data = self._carryover + data
            self._carryover = None

        output_samples = int(input_samples / self._speed)
        if output_samples < 1:
            self._carryover = data
            return b""

        result = bytearray(output_samples * bytes_per_sample)
        src_pos = self._fractional_pos

        for i in range(output_samples):
            src_idx = int(src_pos)
            frac = src_pos - src_idx

            byte_idx = src_idx * bytes_per_sample
            next_byte_idx = min((src_idx + 1) * bytes_per_sample, len(data) - bytes_per_sample)

            if byte_idx + bytes_per_sample <= len(data) and next_byte_idx + bytes_per_sample <= len(data):
                for c in range(bytes_per_sample):
                    a = data[byte_idx + c]
                    b = data[next_byte_idx + c]
                    result[i * bytes_per_sample + c] = int(a + (b - a) * frac) & 0xFF

            src_pos += self._speed

        remaining = int(src_pos) * bytes_per_sample
        if remaining < len(data):
            self._carryover = data[remaining:]
            self._fractional_pos = src_pos - int(src_pos)
        else:
            self._carryover = None
            self._fractional_pos = 0.0

        return bytes(result)


class AirPlayOutputAdapter:
    """AirPlay room output adapter with jitter buffering, drift compensation, and latency offset.

    Receives (pcm_data, pts) tuples from the network stream,
    buffers them in a jitter buffer to absorb network variance,
    applies drift compensation to stay synchronized with the master clock,
    and outputs to an AudioSink at the configured latency offset.

    Drift compensation works by monitoring jitter buffer fill level:
    - Buffer draining too fast -> client clock is fast -> slow down playback (speed < 1.0)
    - Buffer filling too fast -> client clock is slow -> speed up playback (speed > 1.0)
    """

    def __init__(
        self,
        sink: AudioSink,
        sample_rate: int = 48000,
        channels: int = 2,
        sample_width: int = 2,
        target_latency_ms: float = 2000.0,
        frame_size: int = 4800,
        jitter_buffer_ms: float = 1500.0,
        latency_offset_ms: float = 0.0,
    ):
        self.sink = sink
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width
        self.target_latency_ms = target_latency_ms
        self.frame_size = frame_size
        self.jitter_buffer_ms = jitter_buffer_ms
        self.latency_offset_ms = latency_offset_ms

        self._bytes_per_frame = channels * sample_width * frame_size
        self._frame_duration = frame_size / sample_rate
        self._target_frames = int(target_latency_ms / 1000.0 * sample_rate / frame_size)
        self._jitter_frames = int(jitter_buffer_ms / 1000.0 * sample_rate / frame_size)
        self._min_playback_frames = self._jitter_frames

        self._jitter_buffer: deque = deque()
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

        self._drift = DriftCompensator(channels=channels, sample_width=sample_width)
        self._drift_check_interval = 0.5
        self._last_drift_check = 0.0
        self._drift_target_fill = self._target_frames
        self._drift_p_gain = 0.001

        self._underrun_count = 0
        self._frame_count = 0
        self._dropped_frames = 0
        self._underrun_callback: Optional[Callable[[], None]] = None
        self._status_callback: Optional[Callable[[dict], None]] = None
        self._last_status_time = 0.0
        self._status_interval = 1.0

    def set_underrun_callback(self, callback: Callable[[], None]):
        self._underrun_callback = callback

    def set_status_callback(self, callback: Callable[[dict], None]):
        """Called periodically with adapter status (rate-limited to ~1Hz)."""
        self._status_callback = callback

    def set_latency(self, latency_ms: float):
        """Reconfigure target latency."""
        with self._lock:
            self.target_latency_ms = latency_ms
            self._target_frames = int(latency_ms / 1000.0 * self.sample_rate / self.frame_size)
            self._drift_target_fill = self._target_frames

    def set_latency_offset(self, offset_ms: float):
        """Set latency offset relative to master clock (for cross-room sync)."""
        with self._lock:
            self.latency_offset_ms = offset_ms

    def set_jitter_buffer(self, jitter_ms: float):
        """Reconfigure jitter buffer depth."""
        with self._lock:
            self.jitter_buffer_ms = jitter_ms
            self._jitter_frames = int(jitter_ms / 1000.0 * self.sample_rate / self.frame_size)
            self._min_playback_frames = self._jitter_frames

    def on_network_frame(self, data: bytes, pts: float):
        """Callback from network receiver. Queues frame in jitter buffer."""
        with self._lock:
            self._jitter_buffer.append((data, pts))

    def start(self):
        if self._running.is_set():
            return

        self.sink.start(self.sample_rate, self.channels, self.sample_width)
        self._drift.reset()
        self._running.set()
        self._underrun_count = 0
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_drift_check = 0.0
        self._last_status_time = 0.0
        self._thread = threading.Thread(target=self._playback_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None
        self.sink.stop()
        self._drift.reset()

    def _playback_loop(self):
        """Main loop: drains jitter buffer, applies drift compensation, outputs to sink."""
        silence = bytes(self._bytes_per_frame)

        while self._running.is_set():
            frame = self._next_frame()

            if frame is None:
                self._handle_underrun()
                self.sink.write(silence)
                self._emit_status()
                time.sleep(self._frame_duration / 4)
                continue

            data, pts = frame
            self._frame_count += 1

            data = self._drift.process(data)
            if not data:
                continue

            target_time = pts + self.latency_offset_ms / 1000.0
            now = time.monotonic()
            sleep_time = target_time - now

            if sleep_time > 0:
                time.sleep(sleep_time)

            if not self.sink.write(data):
                self._handle_underrun()

            self._apply_drift_correction()
            self._emit_status()

    def _next_frame(self) -> Optional[tuple]:
        """Get the next frame from jitter buffer, or None if below minimum fill."""
        with self._lock:
            if not self._jitter_buffer:
                return None

            if len(self._jitter_buffer) < self._min_playback_frames:
                return None

            now = time.monotonic()
            data, pts = self._jitter_buffer[0]

            if pts > now + 0.5:
                return None

            self._jitter_buffer.popleft()
            return data, pts

    def _apply_drift_correction(self):
        """Adjust playback speed based on jitter buffer fill level."""
        now = time.monotonic()
        if now - self._last_drift_check < self._drift_check_interval:
            return

        self._last_drift_check = now

        with self._lock:
            current_fill = len(self._jitter_buffer)

        error = self._drift_target_fill - current_fill
        correction = error * self._drift_p_gain
        new_speed = DriftCompensator.DEFAULT_SPEED + correction

        self._drift.speed = new_speed

    def _handle_underrun(self):
        """Handle an underrun condition."""
        with self._lock:
            self._underrun_count += 1
            cb = self._underrun_callback
        if cb:
            cb()

    def _emit_status(self):
        """Emit status update if callback is set, rate-limited to ~1Hz."""
        if not self._status_callback:
            return

        now = time.monotonic()
        if now - self._last_status_time < self._status_interval:
            return

        self._last_status_time = now

        with self._lock:
            buffer_fill = len(self._jitter_buffer)
            underruns = self._underrun_count
            frames = self._frame_count
            dropped = self._dropped_frames
            speed = self._drift.speed

        fill_ms = buffer_fill * self.frame_size / self.sample_rate * 1000

        self._status_callback({
            "buffer_fill_frames": buffer_fill,
            "buffer_fill_ms": fill_ms,
            "target_latency_ms": self.target_latency_ms,
            "jitter_buffer_ms": self.jitter_buffer_ms,
            "latency_offset_ms": self.latency_offset_ms,
            "underrun_count": underruns,
            "dropped_frames": dropped,
            "frames_played": frames,
            "drift_speed": speed,
            "sink_healthy": self.sink.is_healthy(),
        })

    @property
    def buffer_fill_frames(self) -> int:
        with self._lock:
            return len(self._jitter_buffer)

    @property
    def buffer_fill_ms(self) -> float:
        with self._lock:
            return len(self._jitter_buffer) * self.frame_size / self.sample_rate * 1000

    @property
    def underrun_count(self) -> int:
        with self._lock:
            return self._underrun_count

    @property
    def frames_played(self) -> int:
        with self._lock:
            return self._frame_count

    @property
    def drift_speed(self) -> float:
        return self._drift.speed

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_frames
