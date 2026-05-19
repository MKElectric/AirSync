import time
import threading
import unittest

from src.aux_output import AuxOutputAdapter, NullSink, PipeWireSink
from src.audio_engine import AudioEngine


class TestNullSink(unittest.TestCase):
    def test_lifecycle(self):
        sink = NullSink()
        sink.start(48000, 2, 2)
        self.assertTrue(sink.is_healthy())
        self.assertTrue(sink.write(b"\x00" * 100))
        sink.stop()
        self.assertFalse(sink.is_healthy())


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
        engine = AudioEngine(sample_rate=48000, channels=2, sample_width=2, buffer_seconds=1.0)
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


class TestPipeWireSink(unittest.TestCase):
    def test_lifecycle(self):
        sink = PipeWireSink(node_name="AirSync-Test")
        sink.start(48000, 2, 2)
        self.assertTrue(sink.is_healthy())
        sink.stop()
        self.assertFalse(sink.is_healthy())


if __name__ == "__main__":
    unittest.main()
