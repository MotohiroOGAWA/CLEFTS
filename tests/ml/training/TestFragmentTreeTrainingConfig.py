from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clefts.ml.training.fragment_tree_training.training_model import (
    build_train_config,
    load_assignment_score_selection,
    load_and_validate_split_preprocessing,
)


class TestFragmentTreeTrainingConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.fragmenter = {
            "fragment_ion_tree_builder": {"cleavage_pattern_set": {"patterns": []}}
        }
        preprocessing = {
            "symbols": ["C", "H", "N", "O"],
            "fragmenter_params": self.fragmenter,
            "max_node": -1,
            "max_edge": -1,
        }
        config_dir = self.root / "config"
        config_dir.mkdir()
        (config_dir / "preprocessing_config.json").write_text(
            json.dumps(preprocessing), encoding="utf-8"
        )
        for split_name in ("train_structures", "validation_structures"):
            split_dir = self.root / split_name
            (split_dir / "data").mkdir(parents=True)
            (split_dir / "fragmenter.json").write_text(
                json.dumps(self.fragmenter), encoding="utf-8"
            )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_loads_saved_config_when_data_subdirectories_are_passed(self) -> None:
        preprocessing, config_path = load_and_validate_split_preprocessing(
            self.root / "train_structures" / "data",
            self.root / "validation_structures" / "data",
        )

        self.assertEqual(preprocessing["symbols"], ["C", "H", "N", "O"])
        self.assertEqual(config_path, self.root / "config" / "preprocessing_config.json")

    def test_prefers_dedicated_pftprep_extension_over_legacy_plain_json(self) -> None:
        # setUp already wrote the legacy plain-named file; also write the
        # dedicated-extension file with different (but still valid) content
        # and confirm it is the one actually loaded and returned.
        dedicated_preprocessing = {
            "symbols": ["C", "H", "N", "O", "S"],
            "fragmenter_params": self.fragmenter,
            "max_node": -1,
            "max_edge": -1,
        }
        (self.root / "config" / "preprocessing_config.pftprep.json").write_text(
            json.dumps(dedicated_preprocessing), encoding="utf-8"
        )

        preprocessing, config_path = load_and_validate_split_preprocessing(
            self.root / "train_structures" / "data",
            self.root / "validation_structures" / "data",
        )

        self.assertEqual(preprocessing["symbols"], ["C", "H", "N", "O", "S"])
        self.assertEqual(
            config_path, self.root / "config" / "preprocessing_config.pftprep.json"
        )

    def test_rejects_different_validation_fragmenter(self) -> None:
        validation_fragmenter = self.root / "validation_structures" / "fragmenter.json"
        validation_fragmenter.write_text(
            json.dumps(
                {
                    "fragment_ion_tree_builder": {
                        "cleavage_pattern_set": {"patterns": []},
                        "max_depth": 9,
                    }
                }
            ),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(ValueError, "different fragmenter.json"):
            load_and_validate_split_preprocessing(
                self.root / "train_structures",
                self.root / "validation_structures",
            )

    def test_default_train_tensorboard_interval_is_fifty_steps(self) -> None:
        config = build_train_config(project_dir=self.root / "project")
        self.assertEqual(config["train_log_interval_steps"], 50)

    def test_anomaly_detection_is_disabled_by_default_and_can_be_enabled(self) -> None:
        default_config = build_train_config(project_dir=self.root / "default_project")
        diagnostic_config = build_train_config(
            project_dir=self.root / "diagnostic_project", detect_anomaly=True
        )
        self.assertFalse(default_config["detect_anomaly"])
        self.assertTrue(diagnostic_config["detect_anomaly"])
        self.assertFalse(default_config["profile_performance"])

    def test_assignment_score_threshold_defaults_to_point_eight(self) -> None:
        config = build_train_config(project_dir=self.root / "project")
        self.assertEqual(config["assignment_score_threshold"], 0.8)

    def test_assignment_scores_are_partitioned_by_sample_within_each_file(self) -> None:
        score_file = self.root / "assignment_scores.tsv"
        score_file.write_text(
            "structure_file\tassignment_score\n"
            "one.preft.pt\t0.95\n"
            "one.preft.pt\t0.20\n"
            "two.preft.pt\t0.80\n",
            encoding="utf-8",
        )
        included, excluded, scores = load_assignment_score_selection(score_file, 0.8)
        self.assertEqual(included, {"one.preft.pt": {0}, "two.preft.pt": {0}})
        self.assertEqual(excluded, {"one.preft.pt": {1}})
        self.assertEqual(scores, [0.95, 0.2, 0.8])


if __name__ == "__main__":
    unittest.main()
