import time
import threading
import unittest

from src.ring_buffer import RingBuffer
from src.timestamp import TimestampGenerator, AudioChunk
from src.playback_scheduler import PlaybackScheduler
from src.audio_engine import AudioEngine


class TestRingBuffer(unittest.TestCase):
    def test_write_read(self):
        buf = RingBuffer(capacity_bytes=1024)
        data = b"\x00" * 256
        written = buf.write(data)
        self.assertEqual(written, 256)
        self.assertEqual(buf.filled, 256)

        result = buf.read(256)
        self.assertIsNotNone(result)
        self.assertEqual(result, data)
        self.assertEqual(buf.filled, 0)

    def test_read_exact(self):
        buf = RingBuffer(capacity_bytes=1024)
        buf.write(b"A" * 200)
        result = buf.read_exact(100)
        self.assertEqual(result, b"A" * 100)
        self.assertEqual(buf.filled, 100)

    def test_read_exact_insufficient_data(self):
        buf = RingBuffer(capacity_bytes=1024)
        buf.write(b"A" * 50)
        result = buf.read_exact(100, block=False)
        self.assertIsNone(result)
        self.assertEqual(buf.filled, 50)

    def test_wrap_around(self):
        buf = RingBuffer(capacity_bytes=100)
        buf.write(b"A" * 60)
        buf.read(60)
        buf.write(b"B" * 60)
        buf.read(60)
        buf.write(b"C" * 60)
        result = buf.read(60)
        self.assertEqual(result, b"C" * 60)

    def test_capacity_limit(self):
        buf = RingBuffer(capacity_bytes=100)
        written = buf.write(b"X" * 150, block=False)
        self.assertEqual(written, 100)
        self.assertEqual(buf.filled, 100)

    def test_peek(self):
        buf = RingBuffer(capacity_bytes=100)
        buf.write(b"HELLO")
        result = buf.peek(5)
        self.assertEqual(result, b"HELLO")
        self.assertEqual(buf.filled, 5)

    def test_clear_wakes_readers(self):
        buf = RingBuffer(capacity_bytes=1024)
        result_holder = [None]
        barrier = threading.Barrier(2)

        def reader():
            barrier.wait()
            result_holder[0] = buf.read(100, block=True, timeout=2.0)

        t = threading.Thread(target=reader)
        t.start()
        barrier.wait()
        time.sleep(0.05)
        buf.clear()
        t.join(timeout=3.0)
        self.assertFalse(t.is_alive())
        self.assertIsNone(result_holder[0])

    def test_clear(self):
        buf = RingBuffer(capacity_bytes=100)
        buf.write(b"DATA")
        buf.clear()
        self.assertEqual(buf.filled, 0)

    def test_read_timeout(self):
        buf = RingBuffer(capacity_bytes=1024)
        start = time.monotonic()
        result = buf.read(100, block=True, timeout=0.1)
        elapsed = time.monotonic() - start
        self.assertIsNone(result)
        self.assertGreaterEqual(elapsed, 0.09)


class TestTimestampGenerator(unittest.TestCase):
    def test_timestamps_monotonic(self):
        gen = TimestampGenerator(sample_rate=48000)
        chunk1 = gen.timestamp(480)
        chunk2 = gen.timestamp(480)
        self.assertLess(chunk1.pts, chunk2.pts)
        self.assertLess(chunk1.sequence, chunk2.sequence)

    def test_duration_calculation(self):
        gen = TimestampGenerator(sample_rate=48000, channels=2, sample_width=2)
        chunk = gen.timestamp(4800)
        self.assertAlmostEqual(chunk.duration, 0.1, places=3)

    def test_frames_count(self):
        gen = TimestampGenerator(sample_rate=48000, channels=2, sample_width=2)
        chunk = gen.timestamp(4800)
        self.assertEqual(chunk.frames, 4800)

    def test_no_data_copy(self):
        gen = TimestampGenerator(sample_rate=48000)
        chunk = gen.timestamp(1000)
        self.assertFalse(hasattr(chunk, "data"))
        self.assertEqual(chunk.frames, 1000)

    def test_pts_from_integer_samples(self):
        gen = TimestampGenerator(sample_rate=48000)
        gen.initialize()
        chunk1 = gen.timestamp(4800)
        chunk2 = gen.timestamp(4800)
        expected_diff = 4800 / 48000
        self.assertAlmostEqual(chunk2.pts - chunk1.pts, expected_diff, places=6)


class TestPlaybackScheduler(unittest.TestCase):
    def test_start_stop(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480)
        scheduler.start()
        self.assertTrue(scheduler.is_running)
        scheduler.stop()
        self.assertFalse(scheduler.is_running)

    def test_callback_invoked(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        ts_gen.initialize()
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480)

        results = []
        event = threading.Event()

        def callback(data, pts):
            results.append((data, pts))
            event.set()

        scheduler.set_callback(callback)
        scheduler.start()
        buf.write(b"\x00" * 1920)

        self.assertTrue(event.wait(timeout=1.0))
        scheduler.stop()
        self.assertGreater(len(results), 0)

    def test_underrun_callback(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        ts_gen.initialize()
        scheduler = PlaybackScheduler(
            buffer=buf,
            timestamp_gen=ts_gen,
            frame_size=480,
            min_fill_frames=9600,
        )

        event = threading.Event()
        scheduler.set_underrun_callback(lambda: event.set())
        scheduler.start()

        self.assertTrue(event.wait(timeout=1.0))
        scheduler.stop()

    def test_buffer_fill_ms(self):
        buf = RingBuffer(capacity_bytes=19200)
        ts_gen = TimestampGenerator(sample_rate=48000)
        ts_gen.initialize()
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, sample_rate=48000, channels=2, sample_width=2)
        buf.write(b"\x00" * 9600)
        fill_ms = scheduler.buffer_fill_ms
        self.assertAlmostEqual(fill_ms, 50.0, places=0)

    def test_pts_offset_applied(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        ts_gen.initialize()
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480)

        pts_holder = [None]
        event = threading.Event()

        def callback(data, pts):
            pts_holder[0] = pts
            event.set()

        scheduler.set_pts_offset(2.0)
        scheduler.set_callback(callback)
        scheduler.start()
        buf.write(b"\x00" * 1920)

        self.assertTrue(event.wait(timeout=1.0))
        scheduler.stop()
        self.assertIsNotNone(pts_holder[0])
        self.assertGreater(pts_holder[0], time.monotonic() + 1.5)


class TestAudioEngine(unittest.TestCase):
    def test_receive_pcm(self):
        engine = AudioEngine(sample_rate=48000, buffer_seconds=1.0)
        data = b"\x00" * 1920
        chunk = engine.receive_pcm(data)
        self.assertIsInstance(chunk, AudioChunk)
        self.assertEqual(chunk.frames, 480)
        self.assertEqual(chunk.sequence, 0)

    def test_start_stop(self):
        engine = AudioEngine(sample_rate=48000, buffer_seconds=1.0)
        engine.start()
        self.assertTrue(engine.is_running)
        engine.stop()
        self.assertFalse(engine.is_running)

    def test_full_pipeline(self):
        engine = AudioEngine(sample_rate=48000, channels=2, sample_width=2, buffer_seconds=1.0)

        playback_data = []
        event = threading.Event()

        def on_playback(data, pts):
            playback_data.append(data)
            if len(playback_data) >= 1:
                event.set()

        engine.set_playback_callback(on_playback)
        engine.start()

        chunk = b"\x00" * (4800 * 2 * 2)
        engine.receive_pcm(chunk)

        self.assertTrue(event.wait(timeout=2.0))
        engine.stop()
        self.assertGreater(len(playback_data), 0)

    def test_delay_offset(self):
        engine = AudioEngine(sample_rate=48000, buffer_seconds=1.0)
        engine.set_delay_offset(2.0)
        self.assertEqual(engine.scheduler._pts_offset, 2.0)

    def test_reset(self):
        engine = AudioEngine(sample_rate=48000, buffer_seconds=1.0)
        engine.receive_pcm(b"\x00" * 1000)
        engine.reset()
        self.assertEqual(engine.buffer.filled, 0)
        self.assertFalse(engine.is_running)

    def test_underrun_callback(self):
        engine = AudioEngine(sample_rate=48000, buffer_seconds=0.1)
        event = threading.Event()
        engine.set_underrun_callback(lambda: event.set())
        engine.start()

        self.assertTrue(event.wait(timeout=1.0))
        engine.stop()


if __name__ == "__main__":
    unittest.main()
