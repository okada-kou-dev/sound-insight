import json
from pathlib import Path
import tempfile
import unittest
from scripts.runtime import digest
from src.demo_selection import create_selection, fixed_rows, reorder_selection, common_selection, common_rows


class FixedDemoTests(unittest.TestCase):
    def test_shipped_order_keeps_calibration_membership_and_bundle_identity(self):
        original_path = Path("configs/demo_selection_v1.json")
        original = json.loads(original_path.read_text(encoding="utf-8"))
        current = json.loads(Path("configs/demo_selection.json").read_text(encoding="utf-8"))
        bundle = json.loads(Path("demo_assets/bundle.json").read_text(encoding="utf-8"))
        self.assertEqual(current, reorder_selection(original, original["public20"][19], original["public20"][0], digest(original_path)))
        self.assertEqual(current["public20"], [r["record_id"] for r in bundle["samples"][:20]])
        expected = common_selection(current, digest(Path("configs/demo_selection.json")))
        self.assertEqual(expected, json.loads(Path("configs/demo50.json").read_text(encoding="utf-8")))
        self.assertEqual(expected["sample_ids"], [r["record_id"] for r in bundle["samples"]])
        self.assertEqual(len(bundle["samples"]), 50)
        self.assertEqual(sum(r["label"] for r in bundle["samples"]), 25)
        metadata = json.loads((Path("demo_assets/models") / bundle["temporal_model_id"] / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["demo_selection_sha256"], digest(original_path))

    def rows(self):
        return [{"record_id": f"{label}_{i}", "relative_path": f"data/{label}/{i:03}.wav", "selected_channel_sha256": f"pcm_{label}_{i}", "label": label, "split": "development"} for label in (0, 1) for i in range(60)]

    def test_balanced_stable_identity_and_order_independent_of_manifest_order(self):
        rows = self.rows()
        public = [r["record_id"] for r in rows[:10] + rows[60:70]]
        a = create_selection(rows, public, "hash")
        b = create_selection(rows[::-1], public, "hash")
        self.assertEqual(a, b)
        self.assertEqual(a["local100"][:20], public)
        self.assertEqual(len(set(a["local100"])), 100)
        self.assertEqual(sum(r["label"] for r in a["records"]), 50)
        self.assertEqual(create_selection(rows, public, "hash"), a)

    def test_fixed_selection_rejects_changed_audio_or_manifest(self):
        rows = self.rows()
        public = [r["record_id"] for r in rows[:10] + rows[60:70]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "data").mkdir(); (root / "configs").mkdir()
            manifest = root / "data/manifest.csv"; manifest.write_text("synthetic manifest", encoding="utf-8")
            selection = create_selection(rows, public, digest(manifest))
            (root / "configs/demo_selection.json").write_text(json.dumps(selection), encoding="utf-8")
            self.assertEqual(len(fixed_rows(root, rows)), 100)
            self.assertEqual(len(fixed_rows(root, rows, public=True)), 20)
            common = common_selection(selection, digest(root / "configs/demo_selection.json"))
            common_path = root / "configs/demo50.json"
            common_path.write_text(json.dumps(common), encoding="utf-8")
            chosen = common_rows(root, rows)
            self.assertEqual(len(chosen), 50)
            self.assertEqual(sum(int(r["label"]) for r in chosen), 25)
            self.assertEqual([r["record_id"] for r in chosen[:20]], public)
            common["sample_ids"][-1] = common["sample_ids"][0]
            common_path.write_text(json.dumps(common), encoding="utf-8")
            with self.assertRaises(ValueError): common_rows(root, rows)
            changed = [dict(r) for r in rows]; changed[0]["selected_channel_sha256"] = "changed"
            with self.assertRaises(ValueError): fixed_rows(root, changed)
            manifest.write_text("different", encoding="utf-8")
            with self.assertRaises(ValueError): fixed_rows(root, rows)

    def test_insufficient_or_imbalanced_data_is_not_silently_reselected(self):
        rows = self.rows(); public = [r["record_id"] for r in rows[:20]]
        with self.assertRaises(ValueError): create_selection(rows, public, "hash")
        public = [r["record_id"] for r in rows[:10] + rows[60:70]]
        with self.assertRaises(ValueError): create_selection(rows[:30]+rows[60:], public, "hash")

    def test_reordering_preserves_membership_provenance_and_non_alternation(self):
        rows = self.rows(); public = [r["record_id"] for r in rows[:10] + rows[60:70]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "data").mkdir(); (root / "configs").mkdir()
            manifest = root / "data/manifest.csv"; manifest.write_text("manifest", encoding="utf-8")
            original = create_selection(rows, public, digest(manifest))
            archive = root / "configs/demo_selection_v1.json"
            archive.write_text(json.dumps(original), encoding="utf-8")
            reordered = reorder_selection(original, public[-1], public[0], digest(archive))
            self.assertEqual(set(original["local100"]), set(reordered["local100"]))
            self.assertEqual(set(original["public20"]), set(reordered["public20"]))
            self.assertEqual(reordered["local100"][:2], [public[-1], public[0]])
            path = root / "configs/demo_selection.json"
            path.write_text(json.dumps(reordered), encoding="utf-8")
            chosen = fixed_rows(root, rows)
            labels = [r["label"] for r in chosen[:20]]
            self.assertTrue(any(a == b for a, b in zip(labels, labels[1:])))
            self.assertEqual(chosen, fixed_rows(root, rows[::-1]))
            reordered["local100"][2:4] = reordered["local100"][2:4][::-1]
            path.write_text(json.dumps(reordered), encoding="utf-8")
            with self.assertRaises(ValueError): fixed_rows(root, rows)
