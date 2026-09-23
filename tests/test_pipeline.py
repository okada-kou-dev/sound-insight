"""実データを参照しないCLI選択規則の検証。"""
import unittest
from scripts.train import meets_goals,ranking
from scripts.verify_demo import demo_rows
from scripts.runtime import config_from,ROOT


class PipelineTests(unittest.TestCase):
    def test_selection_goals_ties_and_missing(self):
        good = {"roc_auc":.75,"recall":.60,"false_positive_rate":.10}
        self.assertTrue(meets_goals(good))
        self.assertFalse(meets_goals(dict(good,recall=None)))
        self.assertGreater(ranking({"metrics":good,"n_clusters":8}),ranking({"metrics":good,"n_clusters":16}))
        lower_fpr = dict(good,false_positive_rate=.05)
        self.assertGreater(ranking({"metrics":lower_fpr,"n_clusters":32}),ranking({"metrics":good,"n_clusters":8}))

    def test_demo_excludes_final_and_does_not_use_predictions(self):
        rows = [{"record_id":f"{split}_{label}_{i}","relative_path":f"{split}/{label}/{i}.wav","split":split,"label":label}
                for split in ("development","final_test") for label in (0,1) for i in range(14)]
        selected = demo_rows(rows)
        self.assertEqual(len(selected),20)
        self.assertEqual(selected,demo_rows(list(reversed(rows))))
        self.assertEqual(sum(r["label"] for r in selected),10)
        self.assertTrue(all(r["split"]=="development" for r in selected))

    def test_fixed_config(self):
        self.assertEqual(config_from(ROOT / "configs/default.json")["feature"]["context_frames"],5)


if __name__ == "__main__":
    unittest.main()
