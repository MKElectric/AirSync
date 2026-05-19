import threading
import array
from typing import Optional


class RingBuffer:
    """Thread-safe ring buffer for PCM audio samples."""

    def __init__(self, capacity_bytes: int):
        self._capacity = capacity_bytes
        self._buffer = bytearray(capacity_bytes)
        self._write_pos = 0
        self._read_pos = 0
        self._filled = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)

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

    def write(self, data: bytes, block: bool = True) -> int:
        """Write data into the buffer. Returns bytes written."""
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
                offset += chunk
                self._not_empty.notify()

        return offset

    def read(self, size: int, block: bool = True) -> Optional[bytes]:
        """Read data from the buffer. Returns bytes or None if empty and non-blocking."""
        with self._not_empty:
            while self._filled == 0:
                if not block:
                    return None
                self._not_empty.wait()

            to_read = min(size, self._filled)
            result = bytearray(to_read)
            pos = self._read_pos % self._capacity

            if pos + to_read <= self._capacity:
                result[:] = self._buffer[pos:pos + to_read]
            else:
                split = self._capacity - pos
                result[:split] = self._buffer[pos:]
                result[split:] = self._buffer[:to_read - split]

            self._read_pos = (self._read_pos + to_read) % self._capacity
            self._filled -= to_read
            self._not_full.notify()

        return bytes(result)

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
        with self._lock:
            self._write_pos = 0
            self._read_pos = 0
            self._filled = 0
            self._not_full.notify()
