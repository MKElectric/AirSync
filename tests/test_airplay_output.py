import time
import threading
import unittest

from src.airplay_output import AirPlayOutputAdapter, DriftCompensator
from src.aux_output import NullSink


class TestDriftCompensator(unittest.TestCase):
    def test_default_speed_passthrough(self):
        comp = DriftCompensator(channels=2, sample_width=2)
        data = b"\x00" * 100
        result = comp.process(data)
        self.assertEqual(result, data)

    def test_speed_limits(self):
        comp = DriftCompensator()
        comp.speed = 0.5
        self.assertEqual(comp.speed, comp.MIN_SPEED)
        comp.speed = 1.5
        self.assertEqual(comp.speed, comp.MAX_SPEED)

    def test_reset(self):
        comp = DriftCompensator()
        comp.speed = 1.002
        comp.reset()
        self.assertEqual(comp.speed, comp.DEFAULT_SPEED)

    def test_speed_change_affects_output(self):
        comp = DriftCompensator(channels=2, sample_width=2)
        comp.speed = 0.998
        data = bytes(range(256)) * 10
        result = comp.process(data)
        self.assertNotEqual(len(result), len(data))

    def test_carryover_preservation(self):
        comp = DriftCompensator(channels=2, sample_width=2)
        comp.speed = 1.002
        data = b"\x00" * 100
        result = comp.process(data)
        result2 = comp.process(b"\x00" * 100)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result2)


class TestAirPlayOutputAdapter(unittest.TestCase):
    def _make_adapter(self, target_latency_ms=100.0, frame_size=480, jitter_buffer_ms=50.0):
        sink = NullSink()
        return AirPlayOutputAdapter(
            sink=sink,
            sample_rate=48000,
            channels=2,
            sample_width=2,
            target_latency_ms=target_latency_ms,
            frame_size=frame_size,
            jitter_buffer_ms=jitter_buffer_ms,
        )

    def test_start_stop(self):
        adapter = self._make_adapter()
        adapter.start()
        adapter.stop()

    def test_frame_queuing(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)

        adapter.on_network_frame(frame, time.monotonic())
        self.assertEqual(adapter.buffer_fill_frames, 1)

        adapter.on_network_frame(frame, time.monotonic())
        self.assertEqual(adapter.buffer_fill_frames, 2)

    def test_buffer_fill_ms(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480)
        frame = b"\x00" * (480 * 2 * 2)

        adapter.on_network_frame(frame, time.monotonic())
        self.assertAlmostEqual(adapter.buffer_fill_ms, 10.0, places=0)

    def test_latency_drain(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480, jitter_buffer_ms=20.0)
        frame = b"\x00" * (480 * 2 * 2)
        frames_needed = int(50.0 / 1000.0 * 48000 / 480)

        for i in range(frames_needed + 5):
            adapter.on_network_frame(frame, time.monotonic() - i * 0.01)

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
        self.assertIn("drift_speed", status)
        self.assertIn("sink_healthy", status)
        self.assertIn("jitter_buffer_ms", status)
        self.assertIn("latency_offset_ms", status)

    def test_latency_reconfiguration(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480)
        self.assertEqual(adapter.target_latency_ms, 100.0)

        adapter.set_latency(200.0)
        self.assertEqual(adapter.target_latency_ms, 200.0)

    def test_latency_offset(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480)
        self.assertEqual(adapter.latency_offset_ms, 0.0)

        adapter.set_latency_offset(50.0)
        self.assertEqual(adapter.latency_offset_ms, 50.0)

    def test_jitter_buffer_reconfiguration(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480, jitter_buffer_ms=50.0)
        self.assertEqual(adapter.jitter_buffer_ms, 50.0)

        adapter.set_jitter_buffer(100.0)
        self.assertEqual(adapter.jitter_buffer_ms, 100.0)

    def test_frames_played_counter(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480, jitter_buffer_ms=20.0)
        frame = b"\x00" * (480 * 2 * 2)

        for i in range(20):
            adapter.on_network_frame(frame, time.monotonic() - i * 0.01)

        adapter.start()
        time.sleep(0.2)
        adapter.stop()

        self.assertGreater(adapter.frames_played, 0)

    def test_drift_correction_applied(self):
        adapter = self._make_adapter(target_latency_ms=100.0, frame_size=480, jitter_buffer_ms=50.0)
        frame = b"\x00" * (480 * 2 * 2)
        frames_needed = int(100.0 / 1000.0 * 48000 / 480)

        for i in range(frames_needed + 10):
            adapter.on_network_frame(frame, time.monotonic() - i * 0.01)

        adapter.start()
        time.sleep(0.6)

        speed = adapter.drift_speed
        adapter.stop()

        self.assertIsInstance(speed, float)
        self.assertGreaterEqual(speed, DriftCompensator.MIN_SPEED)
        self.assertLessEqual(speed, DriftCompensator.MAX_SPEED)

    def test_pts_future_frame_held(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480, jitter_buffer_ms=20.0)
        frame = b"\x00" * (480 * 2 * 2)

        future_pts = time.monotonic() + 10.0
        adapter.on_network_frame(frame, future_pts)

        adapter.start()
        time.sleep(0.1)
        adapter.stop()

        self.assertEqual(adapter.frames_played, 0)
        self.assertEqual(adapter.buffer_fill_frames, 1)

    def test_dropped_frames_tracking(self):
        adapter = self._make_adapter(target_latency_ms=50.0, frame_size=480, jitter_buffer_ms=20.0)
        self.assertEqual(adapter.dropped_frames, 0)

    def test_full_pipeline_with_engine(self):
        from src.audio_engine import AudioEngine

        engine = AudioEngine(sample_rate=48000, channels=2, sample_width=2, buffer_seconds=1.0, min_fill_frames=0)
        adapter = AirPlayOutputAdapter(
            sink=NullSink(),
            sample_rate=48000,
            channels=2,
            sample_width=2,
            target_latency_ms=100.0,
            frame_size=4800,
            jitter_buffer_ms=50.0,
        )

        engine.set_playback_callback(adapter.on_network_frame)
        engine.start()
        adapter.start()

        chunk = b"\x00" * (4800 * 2 * 2)
        engine.receive_pcm(chunk)

        time.sleep(0.3)

        adapter.stop()
        engine.stop()

        self.assertGreater(adapter.frames_played, 0)


if __name__ == "__main__":
    unittest.main()
