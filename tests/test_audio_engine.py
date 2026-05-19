import time
import threading
import unittest

from src.ring_buffer import RingBuffer
from src.timestamp import TimestampGenerator, AudioChunk
from src.playback_scheduler import PlaybackScheduler
from src.audio_engine import AudioEngine
from src.aux_output import AuxOutputAdapter, NullSink, PipeWireSink


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

    def test_read_with_pts(self):
        buf = RingBuffer(capacity_bytes=1024)
        buf.write(b"A" * 200, pts=1.5)
        data, pts = buf.read_with_pts(100)
        self.assertEqual(data, b"A" * 100)
        self.assertEqual(pts, 1.5)

    def test_read_with_pts_insufficient(self):
        buf = RingBuffer(capacity_bytes=1024)
        buf.write(b"A" * 50, pts=1.0)
        data, pts = buf.read_with_pts(100, block=False)
        self.assertIsNone(data)
        self.assertIsNone(pts)
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

    def test_reconcile_adjusts_base(self):
        gen = TimestampGenerator(sample_rate=48000)
        gen.initialize()
        gen.timestamp(4800)
        measured_pts = gen.current_pts
        measured_wall = time.monotonic() + 0.001
        gen.reconcile(measured_pts, measured_wall)
        new_pts = gen.current_pts
        self.assertAlmostEqual(new_pts, measured_wall, places=6)


class TestPlaybackScheduler(unittest.TestCase):
    def test_start_stop(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480)
        scheduler.start()
        self.assertTrue(scheduler.is_running)
        scheduler.stop()
        self.assertFalse(scheduler.is_running)

    def test_callback_invoked_with_frame_pts(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480, min_fill_frames=0)

        pts_holder = [None]
        event = threading.Event()

        def callback(data, pts):
            pts_holder[0] = pts
            event.set()

        scheduler.set_callback(callback)
        scheduler.start()
        buf.write(b"\x00" * 1920, pts=1.0)

        self.assertTrue(event.wait(timeout=2.0))
        scheduler.stop()
        self.assertIsNotNone(pts_holder[0])

    def test_underrun_callback(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
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
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, sample_rate=48000, channels=2, sample_width=2)
        buf.write(b"\x00" * 9600, pts=1.0)
        fill_ms = scheduler.buffer_fill_ms
        self.assertAlmostEqual(fill_ms, 50.0, places=0)

    def test_pts_offset_applied(self):
        buf = RingBuffer(capacity_bytes=10240)
        ts_gen = TimestampGenerator(sample_rate=48000)
        scheduler = PlaybackScheduler(buffer=buf, timestamp_gen=ts_gen, frame_size=480, min_fill_frames=0)

        pts_holder = [None]
        event = threading.Event()

        def callback(data, pts):
            pts_holder[0] = pts
            event.set()

        scheduler.set_pts_offset(0.5)
        scheduler.set_callback(callback)
        scheduler.start()
        buf.write(b"\x00" * 1920, pts=time.monotonic())

        self.assertTrue(event.wait(timeout=2.0))
        scheduler.stop()
        self.assertIsNotNone(pts_holder[0])


class TestAuxOutputAdapter(unittest.TestCase):
    def _make_adapter(self, target_latency_ms=100.0, frame_size=480):
        sink = NullSink()
        return AuxOutputAdapter(
            sink=sink,
            sample_rate=48000,
            channels=2,
            sample_width=2,
            target_latency_ms=target_latency_ms,
            frame_size=frame_size,
        )

    def test_start_stop(self):
        adapter = self._make_adapter()
        adapter.start()
        adapter.stop()

    def test_frame_queuing(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)

        adapter.on_scheduled_frame(frame, time.monotonic())
        self.assertEqual(adapter.buffer_fill_frames, 1)

        adapter.on_scheduled_frame(frame, time.monotonic())
        self.assertEqual(adapter.buffer_fill_frames, 2)

    def test_buffer_fill_ms(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)

        adapter.on_scheduled_frame(frame, time.monotonic())
        self.assertAlmostEqual(adapter.buffer_fill_ms, 10.0, places=0)

    def test_latency_drain(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)
        frames_needed = int(50.0 / 1000.0 * 48000 / 480)

        for _ in range(frames_needed + 5):
            adapter.on_scheduled_frame(frame, time.monotonic())

        adapter.start()
        time.sleep(0.3)
        adapter.stop()

        self.assertLess(adapter.buffer_fill_frames, frames_needed)

    def test_underrun_detection(self):
        adapter = self._make_adapter(target_latency_ms=1000.0, frame_size=480)
        event = threading.Event()
        adapter.set_underrun_callback(lambda: event.set())

        adapter.start()
        self.assertTrue(event.wait(timeout=1.0))
        adapter.stop()
        self.assertGreater(adapter.underrun_count, 0)

    def test_status_callback(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        status_holder = [None]
        event = threading.Event()

        def on_status(status):
            status_holder[0] = status
            event.set()

        adapter.set_status_callback(on_status)
        adapter.start()
        self.assertTrue(event.wait(timeout=1.0))
        adapter.stop()

        status = status_holder[0]
        self.assertIn("buffer_fill_frames", status)
        self.assertIn("buffer_fill_ms", status)
        self.assertIn("underrun_count", status)
        self.assertIn("frames_played", status)
        self.assertIn("sink_healthy", status)

    def test_latency_reconfiguration(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480)
        self.assertEqual(adapter.target_latency_ms, 100.0)

        adapter.set_latency(200.0)
        self.assertEqual(adapter.target_latency_ms, 200.0)

    def test_frames_played_counter(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)

        for _ in range(20):
            adapter.on_scheduled_frame(frame, time.monotonic())

        adapter.start()
        time.sleep(0.2)
        adapter.stop()

        self.assertGreater(adapter.frames_played, 0)

    def test_full_pipeline_with_engine(self):
        engine = AudioEngine(sample_rate=48000, channels=2, sample_width=2, buffer_seconds=1.0, min_fill_frames=0)
        adapter = AuxOutputAdapter(
            sink=NullSink(),
            sample_rate=48000,
            channels=2,
            sample_width=2,
            target_latency_ms=100.0,
            frame_size=4800,
        )

        engine.set_playback_callback(adapter.on_scheduled_frame)
        engine.start()
        adapter.start()

        chunk = b"\x00" * (4800 * 2 * 2)
        engine.receive_pcm(chunk)

        time.sleep(0.3)

        adapter.stop()
        engine.stop()

        self.assertGreater(adapter.frames_played, 0)

    def test_pts_paced_playback(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)
        frames_needed = int(50.0 / 1000.0 * 48000 / 480)

        for i in range(frames_needed + 2):
            adapter.on_scheduled_frame(frame, time.monotonic() + i * 0.1)

        adapter.start()
        time.sleep(0.5)
        adapter.stop()

        self.assertGreater(adapter.frames_played, 0)


class TestPipeWireSink(unittest.TestCase):
    def test_lifecycle(self):
        sink = PipeWireSink(node_name="AirSync-Test")
        sink.start(48000, 2, 2)
        self.assertTrue(sink.is_healthy())
        sink.stop()
        self.assertFalse(sink.is_healthy())


if __name__ == "__main__":
    unittest.main()
