"""Unit tests for ml_engine.evaluation.report.ModelReportGenerator.

Bug-hunting, not coverage padding. Highlights:
  - THE TRAP: the oracle-mode caveat in generate_summary_text is gated on
    report["evaluation_mode"], but generate_report used to DROP that key from the
    evaluation results, so the caveat could never fire (silently hiding that
    oracle-mode mIoU is optimistic). test_oracle_caveat_* fails on the pre-fix code.
  - The recommendation tiers are probed at each boundary, including the [70,80)
    band that produces NO recommendation (a documented logic gap).
"""

from __future__ import annotations

import json

import pytest

from ml_engine.evaluation.report import ModelReportGenerator

# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _results(model_type="detection", overall_score=85, per_class=None, technical=None, **extra):
    r = {
        "model_type": model_type,
        "technical_metrics": technical or {},
        "simple_metrics": {
            "overall_score": overall_score,
            "grade": "B",
            "summary": "a summary",
            "per_class": per_class or [],
        },
        "samples": {"success": [{"file_name": "ok.jpg"}], "failure": [{"file_name": "bad.jpg"}]},
    }
    r.update(extra)
    return r


def _pc(name, score, sample_count=100, grade="B"):
    return {"class": name, "score": score, "grade": grade, "sample_count": sample_count}


@pytest.fixture
def gen():
    return ModelReportGenerator()


# --------------------------------------------------------------------------- #
# THE TRAP: oracle-mode caveat propagation
# --------------------------------------------------------------------------- #


class TestOracleModeCaveat:
    def test_evaluation_mode_is_propagated_into_report(self, gen):
        report = gen.generate_report(
            _results(model_type="segmentation", evaluation_mode="oracle_gt_prompts"),
            model_name="sam",
            test_set_size=10,
        )
        assert report.get("evaluation_mode") == "oracle_gt_prompts"

    def test_oracle_caveat_appears_in_summary_when_mode_set(self, gen):
        # BUG TRAP: pre-fix, generate_report dropped evaluation_mode, so this
        # caveat never rendered even for oracle-mode segmentation evals.
        report = gen.generate_report(
            _results(model_type="segmentation", evaluation_mode="oracle_gt_prompts"),
            model_name="sam",
            test_set_size=10,
        )
        text = gen.generate_summary_text(report)
        assert "oracle mode" in text
        assert "real mIoU will be lower" in text

    def test_no_oracle_caveat_when_mode_absent(self, gen):
        report = gen.generate_report(_results(model_type="segmentation"), model_name="sam", test_set_size=10)
        assert "evaluation_mode" not in report
        assert "oracle mode" not in gen.generate_summary_text(report)

    def test_no_evaluation_mode_key_for_plain_detection(self, gen):
        report = gen.generate_report(_results(model_type="detection"), model_name="gd", test_set_size=10)
        assert "evaluation_mode" not in report


# --------------------------------------------------------------------------- #
# generate_report structure
# --------------------------------------------------------------------------- #


class TestGenerateReport:
    def test_defaults_when_metrics_missing(self, gen):
        report = gen.generate_report({}, model_name="m", test_set_size=5)
        assert report["model_type"] == "detection"  # default
        assert report["simple_metrics"]["overall_score"] == 0
        assert report["simple_metrics"]["grade"] == "Unknown"
        assert report["samples"] == {
            "success_count": 0,
            "failure_count": 0,
            "success_files": [],
            "failure_files": [],
        }

    def test_sample_counts_and_files(self, gen):
        results = _results()
        results["samples"] = {
            "success": [{"file_name": "a.jpg"}, {"file_name": "b.jpg"}],
            "failure": [{"file_name": "c.jpg"}],
        }
        report = gen.generate_report(results, model_name="m", test_set_size=3)
        assert report["samples"]["success_count"] == 2
        assert report["samples"]["failure_count"] == 1
        assert report["samples"]["success_files"] == ["a.jpg", "b.jpg"]
        assert report["samples"]["failure_files"] == ["c.jpg"]

    def test_segmentation_adds_coverage_and_quality(self, gen):
        results = _results(model_type="segmentation")
        results["simple_metrics"]["coverage_rate"] = 88.0
        results["simple_metrics"]["quality_rate"] = 77.0
        report = gen.generate_report(results, model_name="sam", test_set_size=10)
        assert report["simple_metrics"]["coverage_rate"] == 88.0
        assert report["simple_metrics"]["quality_rate"] == 77.0

    def test_detection_omits_coverage_and_quality(self, gen):
        report = gen.generate_report(_results(model_type="detection"), model_name="gd", test_set_size=10)
        assert "coverage_rate" not in report["simple_metrics"]

    def test_extra_info_included_only_when_provided(self, gen):
        r1 = gen.generate_report(_results(), model_name="m", test_set_size=1, extra_info={"k": "v"})
        assert r1["extra_info"] == {"k": "v"}
        r2 = gen.generate_report(_results(), model_name="m", test_set_size=1)
        assert "extra_info" not in r2


# --------------------------------------------------------------------------- #
# Recommendation tiers (probed at boundaries)
# --------------------------------------------------------------------------- #


class TestRecommendationTiers:
    def _recs(self, gen, **kw):
        return gen.generate_report(_results(**kw), model_name="m", test_set_size=100)["recommendations"]

    def test_low_score_recommends_more_data(self, gen):
        recs = self._recs(gen, overall_score=40)
        assert any("performance is low" in r for r in recs)

    def test_average_score_recommends_weak_classes(self, gen):
        recs = self._recs(gen, overall_score=65)
        assert any("performance is average" in r for r in recs)

    def test_good_performance_message_in_70_to_80_band(self, gen):
        # REGRESSION: 75 used to yield ZERO recommendations — a gap between the
        # "average" (<70) and "very good" (>=80) tiers. Now it gets "Good
        # performance" guidance instead of silence.
        recs = self._recs(gen, overall_score=75)
        assert any("Good performance" in r for r in recs)

    def test_very_good_fallback_at_80(self, gen):
        recs = self._recs(gen, overall_score=85)
        assert any("Very good performance" in r for r in recs)

    def test_excellent_fallback_at_90(self, gen):
        recs = self._recs(gen, overall_score=95)
        assert any("Excellent performance" in r for r in recs)

    def test_weak_classes_listed_worst_first_top3(self, gen):
        pc = [_pc("a", 10), _pc("b", 45), _pc("c", 5), _pc("d", 30), _pc("e", 95)]
        recs = self._recs(gen, overall_score=85, per_class=pc)
        weak = next(r for r in recs if "Weak performing classes" in r)
        # worst-first, only the 3 worst (<50): c(5), a(10), d(30) — not b(45)? b<50 too.
        # 4 are <50 (a,b,c,d); top-3 worst by score: c(5), a(10), d(30)
        assert weak.index("'c'") < weak.index("'a'") < weak.index("'d'")
        assert "'b'" not in weak  # trimmed to top-3 worst
        assert "'e'" not in weak  # 95 is not weak

    def test_low_sample_classes_flagged(self, gen):
        pc = [_pc("rare", 80, sample_count=10)]
        recs = self._recs(gen, overall_score=85, per_class=pc)
        assert any("few training samples" in r and "'rare'" in r for r in recs)

    def test_class_imbalance_flagged_when_range_over_40(self, gen):
        pc = [_pc("hi", 95), _pc("lo", 50)]  # range 45 > 40
        recs = self._recs(gen, overall_score=85, per_class=pc)
        assert any("performance gap between classes" in r for r in recs)

    def test_no_imbalance_when_range_within_40(self, gen):
        pc = [_pc("hi", 90), _pc("lo", 60)]  # range 30
        recs = self._recs(gen, overall_score=85, per_class=pc)
        assert not any("performance gap between classes" in r for r in recs)

    def test_detection_localization_hint(self, gen):
        recs = self._recs(gen, overall_score=85, technical={"mAP50": 0.9, "mAP50_95": 0.4})
        assert any("localization" in r for r in recs)


# --------------------------------------------------------------------------- #
# Summary text + combine + save
# --------------------------------------------------------------------------- #


class TestSummaryText:
    def test_detection_shows_map_lines(self, gen):
        report = gen.generate_report(
            _results(technical={"mAP50": 0.812, "mAP50_95": 0.55}), model_name="gd", test_set_size=10
        )
        text = gen.generate_summary_text(report)
        assert "mAP@50: 0.812" in text and "mAP@50-95: 0.550" in text

    def test_segmentation_shows_iou_lines(self, gen):
        report = gen.generate_report(
            _results(model_type="segmentation", technical={"mIoU": 0.7, "mean_dice": 0.8}),
            model_name="sam",
            test_set_size=10,
        )
        text = gen.generate_summary_text(report)
        assert "mIoU: 0.700" in text

    def test_per_class_and_recommendation_lines_render(self, gen):
        pc = [_pc("cat", 42, sample_count=20)]
        report = gen.generate_report(
            _results(overall_score=40, per_class=pc), model_name="gd", test_set_size=100
        )
        text = gen.generate_summary_text(report)
        assert "cat" in text and "PER-CLASS PERFORMANCE" in text
        assert "RECOMMENDATIONS" in text


class TestCombineReports:
    def test_average_score_rounded(self, gen):
        reports = [
            {"model_name": "a", "simple_metrics": {"overall_score": 80}, "recommendations": ["ra"]},
            {"model_name": "b", "simple_metrics": {"overall_score": 91}, "recommendations": ["rb"]},
        ]
        combined = gen.combine_reports(reports)
        assert combined["summary"]["average_score"] == 85.5
        assert combined["summary"]["num_models"] == 2
        assert combined["summary"]["model_names"] == ["a", "b"]

    def test_empty_reports_no_zero_division(self, gen):
        combined = gen.combine_reports([])
        assert combined["summary"]["average_score"] == 0
        assert combined["summary"]["num_models"] == 0

    def test_recommendations_prefixed_by_model(self, gen):
        reports = [{"model_name": "a", "simple_metrics": {"overall_score": 80}, "recommendations": ["fix x"]}]
        combined = gen.combine_reports(reports)
        assert combined["combined_recommendations"] == ["[a] fix x"]


class TestSaveReport:
    def test_writes_json_and_text(self, gen, tmp_path):
        report = gen.generate_report(_results(), model_name="gd", test_set_size=10)
        out = tmp_path / "sub" / "report.json"
        gen.save_report(report, str(out))
        assert out.exists()
        assert json.loads(out.read_text())["model_name"] == "gd"
        assert out.with_suffix(".txt").exists()  # text summary written alongside

    def test_no_text_when_disabled(self, gen, tmp_path):
        report = gen.generate_report(_results(), model_name="gd", test_set_size=10)
        out = tmp_path / "report.json"
        gen.save_report(report, str(out), also_save_text=False)
        assert out.exists() and not out.with_suffix(".txt").exists()
