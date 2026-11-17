"""Synthetic tests for the possession state logic; these do not use match labels."""

import unittest

import numpy as np

from starter.possession import possession_observations, track_possession


class PossessionStateTests(unittest.TestCase):
    def test_first_clear_tracking_observation_initializes_owner(self):
        labels = track_possession(
            np.array([1, -1], dtype=np.int8), np.zeros(2), np.zeros(2, dtype=bool),
            np.full(2, 20))
        self.assertEqual(labels.tolist(), ["A", "A"])

    def test_requires_sustained_opponent_control_to_switch(self):
        observations = np.array([0] * 8 + [1] * 7 + [1], dtype=np.int8)
        labels = track_possession(
            observations, np.zeros(len(observations)), np.zeros(len(observations), dtype=bool),
            np.full(len(observations), 20), direct_confirm=8)
        self.assertEqual(labels[7], "H")
        self.assertEqual(labels[14], "H")
        self.assertEqual(labels[15], "A")

    def test_default_switch_confirmation_is_sustained(self):
        observations = np.array([0] + [1] * 14 + [1], dtype=np.int8)
        labels = track_possession(
            observations, np.zeros(len(observations)), np.zeros(len(observations), dtype=bool),
            np.full(len(observations), 20))
        self.assertEqual(labels[14], "H")
        self.assertEqual(labels[15], "A")

    def test_short_missing_ball_gap_does_not_erase_switch_evidence(self):
        observations = np.array([0] * 8 + [1, -1, 1, 1, -1, 1, 1, 1], dtype=np.int8)
        labels = track_possession(
            observations, np.zeros(len(observations)), np.zeros(len(observations), dtype=bool),
            np.full(len(observations), 20), direct_confirm=6)
        self.assertEqual(labels[-1], "A")

    def test_phase_hint_is_slower_than_direct_tracking(self):
        observations = np.full(25, -1, dtype=np.int8)
        phase = np.full(25, -0.2)
        labels = track_possession(
            observations, phase, np.zeros(25, dtype=bool), np.full(25, 20),
            direct_confirm=8, phase_confirm=20)
        self.assertTrue(np.all(labels[:19] == "X"))
        self.assertEqual(labels[19], "A")

    def test_dead_ball_label_does_not_reset_known_owner(self):
        observations = np.array([0] * 8 + [-1] * 3, dtype=np.int8)
        dead = np.array([False] * 8 + [True, True, False])
        labels = track_possession(
            observations, np.zeros(11), dead, np.full(11, 20))
        self.assertEqual(labels[8:10].tolist(), ["D", "D"])
        self.assertEqual(labels[10], "H")

    def test_sparse_tracking_abstains_but_keeps_owner(self):
        observations = np.array([0] * 8 + [-1, -1], dtype=np.int8)
        counts = np.array([20] * 8 + [5, 20])
        labels = track_possession(
            observations, np.zeros(10), np.zeros(10, dtype=bool), counts)
        self.assertEqual(labels[8], "X")
        self.assertEqual(labels[9], "H")

    def test_long_sparse_tracking_gap_resets_owner(self):
        observations = np.array([0] * 8 + [-1] * 50 + [1], dtype=np.int8)
        counts = np.array([20] * 8 + [5] * 50 + [20])
        labels = track_possession(
            observations, np.zeros(len(observations)),
            np.zeros(len(observations), dtype=bool), counts)
        self.assertEqual(labels[7], "H")
        self.assertEqual(labels[8], "X")
        self.assertEqual(labels[-1], "A")

    def test_long_missing_ball_gap_keeps_owner_when_players_remain_visible(self):
        observations = np.array([0] * 8 + [-1] * 60 + [0], dtype=np.int8)
        labels = track_possession(
            observations, np.zeros(len(observations)),
            np.zeros(len(observations), dtype=bool), np.full(len(observations), 20))
        self.assertTrue(np.all(labels[8:-1] == "H"))
        self.assertEqual(labels[-1], "H")

    def test_nearest_player_requires_close_unambiguous_evidence(self):
        distances = np.array([
            [2.0, 3.0],
            [4.0, 8.0],
            [1.0, 1.2],
            [np.inf, np.inf],
        ])
        self.assertEqual(
            possession_observations(distances).tolist(), [0, -1, -1, -1])


if __name__ == "__main__":
    unittest.main()
