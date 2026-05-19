import threading
from typing import Optional, Callable

from src.ring_buffer import RingBuffer
from src.timestamp import TimestampGenerator, AudioChunk
from src.playback_scheduler import PlaybackScheduler


class AudioEngine:
    """Core audio engine: receives PCM, timestamps, buffers, and schedules playback."""

    DEFAULT_BUFFER_SECONDS = 3.0
    DEFAULT_SAMPLE_RATE = 48000
    DEFAULT_CHANNELS = 2
    DEFAULT_SAMPLE_WIDTH = 2

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        buffer_seconds: float = DEFAULT_BUFFER_SECONDS,
        min_fill_frames: int = 9600,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        capacity = int(sample_rate * channels * sample_width * buffer_seconds)
        self._buffer = RingBuffer(capacity_bytes=capacity)

        self._timestamp_gen = TimestampGenerator(
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
        )

        self._scheduler = PlaybackScheduler(
            buffer=self._buffer,
            timestamp_gen=self._timestamp_gen,
            sample_rate=sample_rate,
            channels=channels,
            sample_width=sample_width,
            min_fill_frames=min_fill_frames,
        )

        self._running = threading.Event()

    @property
    def buffer(self) -> RingBuffer:
        return self._buffer

    @property
    def scheduler(self) -> PlaybackScheduler:
        return self._scheduler

    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    def set_playback_callback(self, callback: Callable[[bytes, float], None]):
        self._scheduler.set_callback(callback)

    def set_underrun_callback(self, callback: Callable[[], None]):
        self._scheduler.set_underrun_callback(callback)

    def set_delay_offset(self, offset_seconds: float):
        self._scheduler.set_pts_offset(offset_seconds)

    def receive_pcm(self, data: bytes) -> AudioChunk:
        """Receive a PCM audio chunk, timestamp it, and write to buffer."""
        bytes_per_frame = self.channels * self.sample_width
        frame_count = len(data) // bytes_per_frame
        chunk = self._timestamp_gen.timestamp(frame_count)
        self._buffer.write(data, pts=chunk.pts)
        return chunk

    def start(self):
        if self._running.is_set():
            return
        self._timestamp_gen.initialize()
        self._scheduler.start()
        self._running.set()

    def stop(self):
        if not self._running.is_set():
            return
        self._scheduler.stop()
        self._timestamp_gen.reset()
        self._running.clear()

    def reset(self):
        self.stop()
        self._buffer.clear()

    @property
    def buffer_fill_ms(self) -> float:
        return self._scheduler.buffer_fill_ms

    @property
    def current_pts(self) -> float:
        return self._timestamp_gen.current_pts
