from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from clefts.ml.training.fragment_tree_training.training_model import (
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


if __name__ == "__main__":
    unittest.main()
