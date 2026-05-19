import threading
from collections import deque
from typing import Optional


class RingBuffer:
    """Thread-safe ring buffer for PCM audio samples with PTS tracking."""

    def __init__(self, capacity_bytes: int):
        self._capacity = capacity_bytes
        self._buffer = bytearray(capacity_bytes)
        self._write_pos = 0
        self._read_pos = 0
        self._filled = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

        self._pts_queue: deque = deque()

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def filled(self) -> int:
        with self._lock:
            return self._filled

    @property
    def available(self) -> int:
        with self._lock:
            return self._capacity - self._filled

    @property
    def read_pts(self) -> float:
        """PTS of the next byte to be read."""
        with self._lock:
            if self._pts_queue:
                return self._pts_queue[0][0]
            return 0.0

    def write(self, data: bytes, pts: float = 0.0, block: bool = True) -> int:
        """Write data into the buffer with associated PTS. Returns bytes written."""
        offset = 0
        length = len(data)

        with self._not_full:
            while length - offset > 0:
                space = self._capacity - self._filled
                if space == 0:
                    if not block:
                        break
                    self._not_full.wait()
                    continue

                chunk = min(length - offset, space)
                pos = self._write_pos % self._capacity

                if pos + chunk <= self._capacity:
                    self._buffer[pos:pos + chunk] = data[offset:offset + chunk]
                else:
                    split = self._capacity - pos
                    self._buffer[pos:] = data[offset:offset + split]
                    self._buffer[:chunk - split] = data[offset + split:offset + chunk]

                self._write_pos = (self._write_pos + chunk) % self._capacity
                self._filled += chunk
                self._pts_queue.append((pts, chunk))
                offset += chunk
                self._not_empty.notify()

        return offset

    def read_with_pts(self, size: int, block: bool = True, timeout: Optional[float] = None):
        """Read exactly `size` bytes with the PTS of the last byte in the chunk.
        Returns (data, pts) or (None, None) if insufficient data."""
        with self._not_empty:
            if not self._wait_for_data(block, timeout):
                return None, None

            if self._filled < size:
                return None, None

            result = self._read_bytes(size)
            pts = self._consume_pts_for_bytes(size)
            self._not_full.notify()
            return result, pts

    def read(self, size: int, block: bool = True, timeout: Optional[float] = None) -> Optional[bytes]:
        """Read up to `size` bytes. Returns None if empty and non-blocking/timed out."""
        with self._not_empty:
            if not self._wait_for_data(block, timeout):
                return None

            to_read = min(size, self._filled)
            result = self._read_bytes(to_read)
            self._consume_pts_for_bytes(to_read)
            self._not_full.notify()
            return result

    def peek(self, size: int, offset: int = 0) -> Optional[bytes]:
        """Peek at data without advancing read pointer."""
        with self._lock:
            if self._filled < offset + size:
                return None

            result = bytearray(size)
            pos = (self._read_pos + offset) % self._capacity

            if pos + size <= self._capacity:
                result[:] = self._buffer[pos:pos + size]
            else:
                split = self._capacity - pos
                result[:split] = self._buffer[pos:]
                result[split:] = self._buffer[:size - split]

            return bytes(result)

    def clear(self):
        """Clear the buffer and wake all waiting threads."""
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._filled = 0
            self._pts_queue.clear()
            self._not_full.notify_all()
            self._not_empty.notify_all()

    def _wait_for_data(self, block: bool, timeout: Optional[float]) -> bool:
        """Wait until data is available. Returns False if timed out or non-blocking with no data."""
        while self._filled == 0:
            if not block:
                return False
            if timeout is not None:
                if not self._not_empty.wait(timeout=timeout):
                    return False
            else:
                self._not_empty.wait()
        return True

    def _read_bytes(self, size: int) -> bytes:
        """Read `size` bytes from the buffer. Caller must hold the lock."""
        result = bytearray(size)
        pos = self._read_pos % self._capacity

        if pos + size <= self._capacity:
            result[:] = self._buffer[pos:pos + size]
        else:
            split = self._capacity - pos
            result[:split] = self._buffer[pos:]
            result[split:] = self._buffer[:size - split]

        self._read_pos = (self._read_pos + size) % self._capacity
        self._filled -= size
        return bytes(result)

    def _consume_pts_for_bytes(self, size: int) -> float:
        """Consume PTS entries for `size` bytes. Returns the PTS of the last byte.
        Caller must hold the lock."""
        remaining = size
        pts = 0.0

        while remaining > 0 and self._pts_queue:
            entry_pts, entry_size = self._pts_queue[0]
            if entry_size <= remaining:
                pts = entry_pts
                remaining -= entry_size
                self._pts_queue.popleft()
            else:
                self._pts_queue[0] = (entry_pts, entry_size - remaining)
                pts = entry_pts
                remaining = 0

        return pts
