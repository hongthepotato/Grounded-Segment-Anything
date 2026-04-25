"""
Unit tests for ml_engine.export.packager — written adversarially.

Covers the orchestration logic in `create_export_package`, the README
templating in `_create_readme`, and the minimal-fallback helpers
`_create_minimal_inference_script` + `_get_minimal_readme`.

Bugs surfaced (tracked via xfail):
- `mAP50=0.0` / `mIoU=0.0` are silently rendered as "N/A" because of a
  truthiness-vs-presence check. A genuinely-zero-score model is hidden.
- The orchestrator has no try/finally cleanup, so a mid-pipeline failure
  leaves a half-built package_dir on disk.
- `class_names` containing newlines silently corrupts the round-trip via
  class_names.txt — no escaping or validation.

Heavy dependencies (`merge_lora_weights`, `save_merged_model`) are mocked
so this file stays pure unit-level and fast.
"""

import zipfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from torch import nn

from ml_engine.export import packager
from ml_engine.export.packager import (
    _create_minimal_inference_script,
    _create_readme,
    _get_minimal_readme,
    create_export_package,
)


class _TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


# ===========================================================================
# _get_minimal_readme — pure string formatter
# ===========================================================================


class TestGetMinimalReadme:
    def test_default_model_name_renders_in_top_heading(self):
        readme = _get_minimal_readme()
        assert "# Fine-tuned grounding_dino Model" in readme

    def test_custom_model_name_renders(self):
        readme = _get_minimal_readme("sam")
        assert "# Fine-tuned sam Model" in readme

    def test_template_tokens_remain_for_later_replacement(self):
        """The minimal README is a TEMPLATE — token strings must remain literal."""
        readme = _get_minimal_readme()
        assert "{model_name}" in readme
        assert "{class_names}" in readme
        assert "{epochs}" in readme
        assert "{generation_date}" in readme


# ===========================================================================
# _create_minimal_inference_script — file emitter
# ===========================================================================


class TestCreateMinimalInferenceScript:
    def test_writes_python_file_with_argparse(self, tmp_path: Path):
        out = tmp_path / "inference.py"
        _create_minimal_inference_script(out)

        content = out.read_text()
        assert content.startswith("#!/usr/bin/env python3")
        assert "import argparse" in content
        assert "def main():" in content
        assert '__name__ == "__main__"' in content

    def test_model_name_propagates_into_script(self, tmp_path: Path):
        out = tmp_path / "inference.py"
        _create_minimal_inference_script(out, model_name="sam")

        content = out.read_text()
        assert "sam" in content

    def test_emitted_script_is_syntactically_valid_python(self, tmp_path: Path):
        """The generated file must compile, otherwise we ship a broken artifact."""
        out = tmp_path / "inference.py"
        _create_minimal_inference_script(out)

        # compile() raises SyntaxError on invalid Python — no file load needed
        compile(out.read_text(), str(out), "exec")


# ===========================================================================
# _create_readme — template substitution
# ===========================================================================


class TestCreateReadme:
    def test_renders_with_no_template_dir_falls_through_to_minimal(self, tmp_path: Path, monkeypatch):
        """No template files on disk → fallback to _get_minimal_readme."""
        # Point TEMPLATES_DIR at an empty directory
        empty_templates = tmp_path / "no_templates"
        empty_templates.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty_templates)

        out = tmp_path / "README.md"
        _create_readme(out, class_names=["dog", "cat"], model_name="grounding_dino")

        content = out.read_text()
        # Top-of-file f-string substitution
        assert "# Fine-tuned grounding_dino Model" in content
        # Token-replacement substitution
        assert "dog, cat" in content
        # No raw template tokens leak through after substitution
        assert "{class_names}" not in content
        assert "{epochs}" not in content
        assert "{generation_date}" not in content

    def test_uses_model_specific_template_when_present(self, tmp_path: Path, monkeypatch):
        """`{model_name}_README_template.md` wins over the generic one."""
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "sam_README_template.md").write_text(
            "SAM-SPECIFIC: classes={class_names}, epochs={epochs}"
        )
        (templates / "README_template.md").write_text("GENERIC: should NOT be picked")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(
            out,
            class_names=["a"],
            training_info={"epochs": 5},
            model_name="sam",
        )

        content = out.read_text()
        assert content.startswith("SAM-SPECIFIC")
        assert "GENERIC" not in content

    def test_falls_through_to_generic_template_when_model_specific_missing(self, tmp_path: Path, monkeypatch):
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "README_template.md").write_text("GENERIC: classes={class_names}")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(
            out,
            class_names=["wombat"],
            model_name="unknown_model_with_no_specific_template",
        )

        assert "GENERIC: classes=wombat" in out.read_text()

    def test_training_info_none_renders_na_for_metrics(self, tmp_path: Path, monkeypatch):
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        out = tmp_path / "README.md"
        _create_readme(out, class_names=[], training_info=None)

        content = out.read_text()
        # Even with None training_info, the substitution shouldn't leak tokens
        assert "{epochs}" not in content
        assert "{map50}" not in content
        assert "{miou}" not in content

    def test_training_info_truthy_metrics_render_as_percentages(self, tmp_path: Path, monkeypatch):
        templates = tmp_path / "tpl"
        templates.mkdir()
        (templates / "README_template.md").write_text("mAP50: {map50}, mIoU: {miou}")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(
            out,
            class_names=[],
            training_info={"mAP50": 0.835, "mIoU": 0.612},
        )

        content = out.read_text()
        assert "mAP50: 83.5%" in content
        assert "mIoU: 61.2%" in content

    def test_zero_metric_renders_zero_percent_not_na(self, tmp_path: Path, monkeypatch):
        """
        Regression: a catastrophically-scoring model (mAP50=0.0) used to render
        as "N/A" because of a truthiness check; now renders honestly as "0.0%".
        Distinguishes "metric not computed" (None / missing) from "metric was zero".
        """
        templates = tmp_path / "tpl"
        templates.mkdir()
        (templates / "README_template.md").write_text("mAP50: {map50}")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(out, class_names=[], training_info={"mAP50": 0.0})

        assert "mAP50: 0.0%" in out.read_text()

    def test_missing_metric_still_renders_na(self, tmp_path: Path, monkeypatch):
        """
        Counterpart to the zero-metric fix: when mAP50 is genuinely absent (key
        missing or None), still render "N/A". The fix should only differentiate
        zero from None — not break the missing-metric case.
        """
        templates = tmp_path / "tpl"
        templates.mkdir()
        (templates / "README_template.md").write_text("mAP50: {map50}, mIoU: {miou}")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(out, class_names=[], training_info={})  # both metrics missing

        content = out.read_text()
        assert "mAP50: N/A" in content
        assert "mIoU: N/A" in content

    def test_generation_date_substituted_with_iso_timestamp(self, tmp_path: Path, monkeypatch):
        templates = tmp_path / "tpl"
        templates.mkdir()
        (templates / "README_template.md").write_text("Generated: {generation_date}")
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates)

        out = tmp_path / "README.md"
        _create_readme(out, class_names=[])

        content = out.read_text()
        # Just verify it's a date-like string, not the literal token
        assert "{generation_date}" not in content
        # Format is "%Y-%m-%d %H:%M:%S"
        assert datetime.now().strftime("%Y-%m-%d") in content


# ===========================================================================
# create_export_package — orchestration smoke tests
# ===========================================================================


class TestCreateExportPackage:
    def test_produces_zip_with_expected_contents(self, tmp_path: Path, monkeypatch):
        """End-to-end (mocked merger): ZIP exists and contains all 4 artifacts."""
        # Mock the heavy bits: merge_lora_weights returns the model unchanged
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        # Empty templates dir → uses minimal fallbacks
        templates_dir = tmp_path / "templates_unused"
        templates_dir.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", templates_dir)

        zip_path = create_export_package(
            model=_TinyModel(),
            output_dir=tmp_path / "experiment",
            class_names=["dog", "cat"],
            model_name="grounding_dino",
            training_info={"epochs": 3, "mAP50": 0.81},
        )

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"

        # Inspect ZIP contents
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())

        assert "merged_model.pth" in names
        assert "inference.py" in names
        assert "README.md" in names
        assert "class_names.txt" in names

    def test_class_names_txt_contains_one_class_per_line(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        zip_path = create_export_package(
            model=_TinyModel(),
            output_dir=tmp_path,
            class_names=["aardvark", "buffalo", "capybara"],
        )

        with zipfile.ZipFile(zip_path) as zf:
            content = zf.read("class_names.txt").decode("utf-8")

        assert content == "aardvark\nbuffalo\ncapybara"

    def test_package_dir_cleaned_up_after_success(self, tmp_path: Path, monkeypatch):
        """The intermediate package_dir is rmtree'd at the end of a successful run."""
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        output_dir = tmp_path / "experiment"
        create_export_package(
            model=_TinyModel(),
            output_dir=output_dir,
            class_names=[],
        )

        package_dir = output_dir / "exports" / "grounding_dino_package"
        assert not package_dir.exists()
        # The ZIP should remain
        assert (output_dir / "exports" / "grounding_dino_package.zip").exists()

    def test_existing_package_dir_is_silently_destroyed(self, tmp_path: Path, monkeypatch):
        """
        Documents a destructive behavior: if `{model_name}_package/` already exists
        in exports/, it gets rmtree'd without warning. A user (or another tool) that
        happens to use the same directory name loses data silently.
        """
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        output_dir = tmp_path / "experiment"
        package_dir = output_dir / "exports" / "grounding_dino_package"
        package_dir.mkdir(parents=True)
        # Plant a "user file" the user might have placed here
        important = package_dir / "user_data_DO_NOT_DELETE.txt"
        important.write_text("important")

        create_export_package(
            model=_TinyModel(),
            output_dir=output_dir,
            class_names=[],
        )

        # The user file is gone — silent destruction
        assert not important.exists()

    @pytest.mark.xfail(
        reason=(
            "BUG: create_export_package has no try/finally around the cleanup "
            "rmtree(package_dir). If any of steps 1-4 raises (e.g. save_merged_model "
            "fails because output_dir isn't writable, or zipfile.write hits an OS "
            "error), the function exits with package_dir leaked on disk. Fix: wrap "
            "the body in try/finally so cleanup always runs."
        ),
        strict=True,
    )
    def test_package_dir_cleaned_up_even_when_zip_step_fails(self, tmp_path: Path, monkeypatch):
        """If the ZIP step fails, package_dir should still be cleaned up."""
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        # Simulate a failure mid-zip — patch.object restores ZipFile after the with-block
        class _FailingZip:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("simulated failure")

        with patch.object(zipfile, "ZipFile", _FailingZip):
            with pytest.raises(RuntimeError, match="simulated failure"):
                create_export_package(
                    model=_TinyModel(),
                    output_dir=tmp_path / "experiment",
                    class_names=["x"],
                )

        # FAILS today: package_dir is leaked because there's no try/finally
        package_dir = tmp_path / "experiment" / "exports" / "grounding_dino_package"
        assert not package_dir.exists()

    @pytest.mark.parametrize(
        "bad_name",
        [
            pytest.param("multi\nline_class", id="LF"),
            pytest.param("carriage\rreturn_class", id="CR"),
            pytest.param("crlf\r\nclass", id="CRLF"),
        ],
    )
    def test_class_names_with_embedded_newlines_raise_valueerror(
        self, tmp_path: Path, monkeypatch, bad_name: str
    ):
        """
        Regression: previously, class_names.txt's '\\n'.join(...) silently
        corrupted the round-trip when a class name contained '\\n' / '\\r'.
        Now the function rejects such names at the boundary with a clear
        ValueError naming the offending index — no half-built artifact, no
        silent corruption.
        """
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        with pytest.raises(ValueError, match=r"class_names\[1\].*newline"):
            create_export_package(
                model=_TinyModel(),
                output_dir=tmp_path,
                class_names=["normal_class", bad_name, "another"],
            )

    def test_create_export_package_no_artifacts_left_after_validation_failure(
        self, tmp_path: Path, monkeypatch
    ):
        """
        Validation should run BEFORE any filesystem mutation — a rejected
        class_names input must not leave a half-created exports/ subdirectory.
        """
        monkeypatch.setattr(packager, "merge_lora_weights", lambda m: m)
        empty = tmp_path / "tpl"
        empty.mkdir()
        monkeypatch.setattr(packager, "TEMPLATES_DIR", empty)

        with pytest.raises(ValueError):
            create_export_package(
                model=_TinyModel(),
                output_dir=tmp_path / "experiment",
                class_names=["bad\nname"],
            )

        # No exports/ directory should have been created
        assert not (tmp_path / "experiment" / "exports").exists()
