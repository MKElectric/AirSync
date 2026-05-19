import os
import re
import time
import json
import signal
import struct
import logging
import subprocess
import threading
from typing import Optional, Callable
from pathlib import Path

logger = logging.getLogger(__name__)


class ShairportState:
    """Represents the current state of a shairport-sync connection."""
    IDLE = "idle"
    CONNECTING = "connecting"
    PLAYING = "playing"
    STOPPING = "stopping"


class AirPlaySession:
    """Represents a single AirPlay connection session."""

    def __init__(self, client_name: str = ""):
        self.client_name = client_name
        self.connected_at = time.monotonic()
        self.sample_rate = 44100
        self.channels = 2
        self.sample_width = 2
        self.rtp_timestamp: Optional[int] = None
        self.pts_offset: float = 0.0


class ShairportReceiver:
    """Manages shairport-sync subprocess as AirPlay receiver input.

    Runs shairport-sync with pipe output, captures PCM audio,
    and feeds it into the audio engine with timestamp preservation.

    Handles reconnection automatically: when an AirPlay sender disconnects,
    shairport-sync stops writing PCM. The receiver detects this and resets
    for the next connection.
    """

    DEFAULT_NAME = "AirSync"
    DEFAULT_SAMPLE_RATE = 44100
    DEFAULT_CHANNELS = 2
    DEFAULT_SAMPLE_WIDTH = 2
    PCM_CHUNK_SIZE = 4096
    RESTART_DELAY = 1.0

    def __init__(
        self,
        name: str = DEFAULT_NAME,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
        sample_width: int = DEFAULT_SAMPLE_WIDTH,
        config_path: Optional[str] = None,
    ):
        self.name = name
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self._config_path = config_path
        self._process: Optional[subprocess.Popen] = None
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._meta_thread: Optional[threading.Thread] = None

        self._state = ShairportState.IDLE
        self._state_lock = threading.Lock()
        self._session: Optional[AirPlaySession] = None

        self._pcm_callback: Optional[Callable[[bytes, float], None]] = None
        self._state_callback: Optional[Callable[[str, dict], None]] = None

        self._meta_dir: Optional[Path] = None
        self._meta_file: Optional[Path] = None
        self._meta_fd: Optional[int] = None

        self._total_bytes_received = 0
        self._connection_count = 0
        self._last_error: Optional[str] = None

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    @property
    def session(self) -> Optional[AirPlaySession]:
        with self._state_lock:
            return self._session

    @property
    def total_bytes_received(self) -> int:
        return self._total_bytes_received

    @property
    def connection_count(self) -> int:
        return self._connection_count

    @property
    def last_error(self) -> Optional[str]:
        return self._last_error

    def set_pcm_callback(self, callback: Callable[[bytes, float], None]):
        """Set callback for PCM data. Called with (pcm_bytes, pts)."""
        self._pcm_callback = callback

    def set_state_callback(self, callback: Callable[[str, dict], None]):
        """Set callback for state changes. Called with (state, context)."""
        self._state_callback = callback

    def start(self):
        """Start the shairport-sync receiver."""
        if self._running.is_set():
            return

        self._running.set()
        self._total_bytes_received = 0
        self._connection_count = 0
        self._last_error = None

        self._setup_metadata()
        self._spawn_shairport()

        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

        if self._meta_file:
            self._meta_thread = threading.Thread(target=self._metadata_loop, daemon=True)
            self._meta_thread.start()

        self._set_state(ShairportState.IDLE)

    def stop(self):
        """Stop the shairport-sync receiver."""
        self._running.clear()

        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

        if self._thread:
            self._thread.join(timeout=3.0)
            self._thread = None

        if self._meta_thread:
            self._meta_thread.join(timeout=2.0)
            self._meta_thread = None

        self._cleanup_metadata()
        self._set_state(ShairportState.IDLE)

    def _setup_metadata(self):
        """Create temporary directory for shairport-sync metadata."""
        import tempfile
        self._meta_dir = Path(tempfile.mkdtemp(prefix="airsync_meta_"))
        self._meta_file = self._meta_dir / "metadata"
        self._meta_file.touch()

    def _cleanup_metadata(self):
        """Clean up metadata directory."""
        if self._meta_fd is not None:
            try:
                os.close(self._meta_fd)
            except OSError:
                pass
            self._meta_fd = None

        if self._meta_dir and self._meta_dir.exists():
            import shutil
            try:
                shutil.rmtree(self._meta_dir, ignore_errors=True)
            except OSError:
                pass
            self._meta_dir = None
            self._meta_file = None

    def _spawn_shairport(self):
        """Spawn shairport-sync subprocess with pipe output."""
        args = [
            "shairport-sync",
            "-a", self.name,
            "-o", "stdout",
            "--name", self.name,
        ]

        if self._meta_file:
            args.extend(["--meta-dir", str(self._meta_dir)])

        if self._config_path:
            args.extend(["-c", self._config_path])

        logger.info("Starting shairport-sync: %s", " ".join(args))

        self._process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
        )

    def _monitor_loop(self):
        """Monitor shairport-sync process and PCM output."""
        assert self._process is not None
        pipe = self._process.stdout

        while self._running.is_set():
            try:
                data = pipe.read(self.PCM_CHUNK_SIZE)
                if not data:
                    if self._process.poll() is not None:
                        self._handle_process_exit()
                        if self._running.is_set():
                            time.sleep(self.RESTART_DELAY)
                            self._spawn_shairport()
                            pipe = self._process.stdout
                    continue

                self._on_pcm_data(data)

            except (OSError, ValueError) as e:
                logger.error("PCM read error: %s", e)
                self._last_error = str(e)
                if self._process.poll() is not None:
                    self._handle_process_exit()
                    if self._running.is_set():
                        time.sleep(self.RESTART_DELAY)
                        self._spawn_shairport()
                        pipe = self._process.stdout
                else:
                    time.sleep(0.1)

    def _on_pcm_data(self, data: bytes):
        """Handle incoming PCM data from shairport-sync."""
        if self._state != ShairportState.PLAYING:
            self._on_connection_start()

        self._total_bytes_received += len(data)

        pts = self._compute_pts(len(data))

        if self._pcm_callback:
            self._pcm_callback(data, pts)

    def _on_connection_start(self):
        """Handle new AirPlay connection."""
        self._connection_count += 1
        self._session = AirPlaySession(client_name="unknown")
        self._session.pts_offset = time.monotonic()
        self._set_state(ShairportState.PLAYING)
        self._emit_state("connected", {"session": self._connection_count})
        logger.info("AirPlay connection #%d", self._connection_count)

    def _on_connection_stop(self):
        """Handle AirPlay disconnection."""
        self._session = None
        self._set_state(ShairportState.IDLE)
        self._emit_state("disconnected", {"session": self._connection_count})
        logger.info("AirPlay disconnected")

    def _compute_pts(self, data_length: int) -> float:
        """Compute presentation timestamp for PCM data."""
        bytes_per_sample = self.channels * self.sample_width
        frames = data_length // bytes_per_sample
        duration = frames / self.sample_rate

        with self._state_lock:
            if self._session is None:
                return time.monotonic()
            base = self._session.pts_offset

        elapsed = self._total_bytes_received / (self.sample_rate * bytes_per_sample)
        return base + elapsed - duration

    def _handle_process_exit(self):
        """Handle shairport-sync process exit."""
        returncode = self._process.returncode
        logger.info("shairport-sync exited with code %d", returncode)

        if self._state == ShairportState.PLAYING:
            self._on_connection_stop()

        self._last_error = f"Process exited with code {returncode}"

    def _metadata_loop(self):
        """Monitor shairport-sync metadata file for session info."""
        if not self._meta_file:
            return

        try:
            self._meta_fd = os.open(str(self._meta_file), os.O_RDONLY | os.O_NONBLOCK)
        except OSError:
            return

        buffer = b""

        while self._running.is_set():
            try:
                chunk = os.read(self._meta_fd, 4096)
                if not chunk:
                    time.sleep(0.5)
                    continue

                buffer += chunk
                lines = buffer.split(b"\n")
                buffer = lines[-1]

                for line in lines[:-1]:
                    self._process_metadata_line(line.decode(errors="replace"))

            except OSError:
                time.sleep(0.5)

        try:
            os.close(self._meta_fd)
        except OSError:
            pass

    def _process_metadata_line(self, line: str):
        """Parse a metadata line from shairport-sync."""
        if not line.strip():
            return

        try:
            meta = json.loads(line)
            event_type = meta.get("type", "")

            if event_type == "pbeg":
                self._on_connection_start()
            elif event_type == "pend":
                self._on_connection_stop()
            elif event_type == "prgr":
                self._update_session_progress(meta)

        except json.JSONDecodeError:
            pass

    def _update_session_progress(self, meta: dict):
        """Update session from progress metadata."""
        with self._state_lock:
            if self._session is None:
                return

            rtp_timestamp = meta.get("rtp_timestamp")
            if rtp_timestamp is not None:
                self._session.rtp_timestamp = rtp_timestamp

    def _set_state(self, state: str):
        """Update internal state."""
        with self._state_lock:
            self._state = state

    def _emit_state(self, event: str, context: dict):
        """Emit state change to callback."""
        if self._state_callback:
            with self._state_lock:
                self._state_callback(self._state, {
                    "event": event,
                    "session": self._connection_count,
                    **context,
                })
