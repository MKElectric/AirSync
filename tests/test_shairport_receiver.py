import os
import time
import json
import threading
import unittest
from unittest.mock import patch, MagicMock
from io import BytesIO

from src.shairport_receiver import ShairportReceiver, ShairportState, AirPlaySession


class TestAirPlaySession(unittest.TestCase):
    def test_default_values(self):
        session = AirPlaySession()
        self.assertEqual(session.client_name, "")
        self.assertEqual(session.sample_rate, 44100)
        self.assertEqual(session.channels, 2)
        self.assertEqual(session.sample_width, 2)
        self.assertIsNone(session.rtp_timestamp)
        self.assertEqual(session.session_bytes, 0)

    def test_with_client_name(self):
        session = AirPlaySession(client_name="iPhone")
        self.assertEqual(session.client_name, "iPhone")

    def test_connected_at_set(self):
        before = time.monotonic()
        session = AirPlaySession()
        after = time.monotonic()
        self.assertGreaterEqual(session.connected_at, before)
        self.assertLessEqual(session.connected_at, after)


class TestShairportReceiver(unittest.TestCase):
    def _make_receiver(self, **kwargs):
        return ShairportReceiver(name="TestReceiver", **kwargs)

    def test_initial_state(self):
        receiver = self._make_receiver()
        self.assertEqual(receiver.state, ShairportState.IDLE)
        self.assertIsNone(receiver.session)
        self.assertEqual(receiver.total_bytes_received, 0)
        self.assertEqual(receiver.connection_count, 0)
        self.assertIsNone(receiver.last_error)

    def test_set_callbacks(self):
        receiver = self._make_receiver()
        pcm_called = []
        state_called = []
        reconnect_called = []

        receiver.set_pcm_callback(lambda data, pts: pcm_called.append((data, pts)))
        receiver.set_state_callback(lambda state, ctx: state_called.append((state, ctx)))
        receiver.set_reconnect_callback(lambda: reconnect_called.append(True))

        self.assertIsNotNone(receiver._pcm_callback)
        self.assertIsNotNone(receiver._state_callback)
        self.assertIsNotNone(receiver._reconnect_callback)

    def test_start_stop(self):
        receiver = self._make_receiver()

        with patch.object(receiver, "_spawn_shairport"), \
             patch.object(receiver, "_setup_metadata"), \
             patch.object(receiver, "_cleanup_metadata"), \
             patch.object(receiver, "_monitor_loop"), \
             patch.object(receiver, "_stderr_loop"):
            receiver.start()
            self.assertTrue(receiver._running.is_set())
            receiver.stop()
            self.assertFalse(receiver._running.is_set())

    def test_start_idempotent(self):
        receiver = self._make_receiver()

        with patch.object(receiver, "_spawn_shairport"), \
             patch.object(receiver, "_setup_metadata"), \
             patch.object(receiver, "_cleanup_metadata"), \
             patch.object(receiver, "_monitor_loop"), \
             patch.object(receiver, "_stderr_loop"):
            receiver.start()
            receiver.start()
            receiver.start()
            receiver.stop()

    def test_stop_idempotent(self):
        receiver = self._make_receiver()

        with patch.object(receiver, "_spawn_shairport"), \
             patch.object(receiver, "_setup_metadata"), \
             patch.object(receiver, "_cleanup_metadata"), \
             patch.object(receiver, "_monitor_loop"), \
             patch.object(receiver, "_stderr_loop"):
            receiver.stop()
            receiver.stop()

    def test_pcm_callback_invoked(self):
        receiver = self._make_receiver()
        results = []
        event = threading.Event()

        def on_pcm(data, pts):
            results.append((data, pts))
            if len(results) >= 1:
                event.set()

        receiver.set_pcm_callback(on_pcm)

        receiver._on_pcm_data(b"\x00" * 1764)

        self.assertTrue(event.wait(timeout=1.0))
        self.assertEqual(len(results), 1)
        self.assertEqual(len(results[0][0]), 1764)
        self.assertIsInstance(results[0][1], float)

    def test_state_callback_on_connection(self):
        receiver = self._make_receiver()
        states = []
        event = threading.Event()

        def on_state(state, ctx):
            states.append((state, ctx))
            if ctx.get("event") == "connected":
                event.set()

        receiver.set_state_callback(on_state)

        receiver._on_connection_start()

        self.assertTrue(event.wait(timeout=1.0))
        self.assertEqual(receiver.state, ShairportState.PLAYING)
        self.assertIsNotNone(receiver.session)
        self.assertEqual(receiver.connection_count, 1)

    def test_state_callback_on_disconnection(self):
        receiver = self._make_receiver()
        states = []
        disconnect_event = threading.Event()

        def on_state(state, ctx):
            states.append((state, ctx))
            if ctx.get("event") == "disconnected":
                disconnect_event.set()

        receiver.set_state_callback(on_state)

        receiver._on_connection_start()
        receiver._on_connection_stop()

        self.assertTrue(disconnect_event.wait(timeout=1.0))
        self.assertEqual(receiver.state, ShairportState.IDLE)
        self.assertIsNone(receiver.session)

    def test_pts_computation_session_relative(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()

        pts1 = receiver._compute_pts(1764)
        receiver._on_pcm_data(b"\x00" * 1764)
        pts2 = receiver._compute_pts(1764)

        self.assertLess(pts1, pts2)

    def test_pts_before_connection(self):
        receiver = self._make_receiver()
        pts = receiver._compute_pts(1764)
        self.assertIsInstance(pts, float)
        self.assertGreater(pts, 0)

    def test_total_bytes_tracking(self):
        receiver = self._make_receiver()
        receiver._on_pcm_data(b"\x00" * 1000)
        receiver._on_pcm_data(b"\x00" * 500)
        self.assertEqual(receiver.total_bytes_received, 1500)

    def test_connection_count_tracking(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()
        receiver._on_connection_stop()
        receiver._on_connection_start()
        self.assertEqual(receiver.connection_count, 2)

    def test_session_bytes_reset_on_reconnect(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()
        receiver._on_pcm_data(b"\x00" * 1000)
        first_session_bytes = receiver.session.session_bytes

        receiver._on_connection_stop()
        receiver._on_connection_start()
        second_session_bytes = receiver.session.session_bytes

        self.assertGreater(first_session_bytes, 0)
        self.assertEqual(second_session_bytes, 0)

    def test_metadata_parse_pbeg(self):
        receiver = self._make_receiver()
        event = threading.Event()
        receiver.set_state_callback(lambda s, c: event.set() if c.get("event") == "connected" else None)

        receiver._process_metadata_line(json.dumps({"type": "pbeg"}))

        self.assertTrue(event.wait(timeout=1.0))
        self.assertEqual(receiver.state, ShairportState.PLAYING)

    def test_metadata_parse_pend(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()
        event = threading.Event()
        receiver.set_state_callback(lambda s, c: event.set() if c.get("event") == "disconnected" else None)

        receiver._process_metadata_line(json.dumps({"type": "pend"}))

        self.assertTrue(event.wait(timeout=1.0))
        self.assertEqual(receiver.state, ShairportState.IDLE)

    def test_metadata_parse_prgr(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()

        receiver._process_metadata_line(json.dumps({
            "type": "prgr",
            "rtp_timestamp": 12345678,
        }))

        self.assertEqual(receiver.session.rtp_timestamp, 12345678)

    def test_metadata_invalid_json(self):
        receiver = self._make_receiver()
        receiver._process_metadata_line("not json")
        receiver._process_metadata_line("")
        receiver._process_metadata_line("   ")

    def test_process_exit_during_play(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()
        receiver._process = MagicMock()
        receiver._process.returncode = 1
        receiver._process.poll.return_value = 1

        receiver._handle_process_exit()

        self.assertEqual(receiver.state, ShairportState.IDLE)
        self.assertIsNotNone(receiver.last_error)

    def test_process_exit_during_idle(self):
        receiver = self._make_receiver()
        receiver._process = MagicMock()
        receiver._process.returncode = 0
        receiver._process.poll.return_value = 0

        receiver._handle_process_exit()

        self.assertEqual(receiver.state, ShairportState.IDLE)

    def test_full_pipeline_integration(self):
        receiver = self._make_receiver()
        pcm_results = []
        state_results = []
        play_event = threading.Event()

        receiver.set_pcm_callback(lambda data, pts: pcm_results.append((data, pts)))
        receiver.set_state_callback(lambda s, c: state_results.append((s, c)))

        receiver._on_connection_start()

        for i in range(10):
            receiver._on_pcm_data(b"\x00" * 1764)

        receiver._on_connection_stop()

        self.assertEqual(len(pcm_results), 10)
        self.assertEqual(receiver.connection_count, 1)
        self.assertEqual(receiver.state, ShairportState.IDLE)

    def test_session_pts_offset_reset_on_reconnect(self):
        receiver = self._make_receiver()

        receiver._on_connection_start()
        first_offset = receiver.session.pts_offset

        receiver._on_connection_stop()
        receiver._on_connection_start()
        second_offset = receiver.session.pts_offset

        self.assertGreaterEqual(second_offset, first_offset)

    def test_double_connection_start_prevented(self):
        receiver = self._make_receiver()
        receiver._on_connection_start()
        receiver._on_connection_start()

        self.assertEqual(receiver.connection_count, 1)
        self.assertEqual(receiver.state, ShairportState.PLAYING)

    def test_reconnect_callback_invoked(self):
        receiver = self._make_receiver()
        event = threading.Event()
        receiver.set_reconnect_callback(lambda: event.set())

        receiver._on_connection_start()

        self.assertTrue(event.wait(timeout=1.0))


if __name__ == "__main__":
    unittest.main()
