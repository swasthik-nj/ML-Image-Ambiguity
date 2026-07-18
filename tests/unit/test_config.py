"""Unit tests for configuration loading."""

from __future__ import annotations

from pathlib import Path

from image_ambiguity.config import PROJECT_ROOT, get_settings, load_yaml_config


def test_load_yaml_config_has_expected_keys() -> None:
    data = load_yaml_config()
    assert data["sample_size"] == 500
    assert data["experiment_name"] == "image_ambiguity_baseline"


def test_get_settings_resolves_paths_under_project_root() -> None:
    get_settings.cache_clear()
    settings = get_settings()

    assert settings.project_root == PROJECT_ROOT.resolve()
    assert settings.annotation_file.is_absolute()
    assert "captions_val2017.json" in str(settings.annotation_file)
    assert settings.sample_size == 500
    assert settings.log_level.upper() == "INFO"


def test_env_override_sample_size(monkeypatch) -> None:
    monkeypatch.setenv("IAP_SAMPLE_SIZE", "123")
    get_settings.cache_clear()

    settings = get_settings()
    assert settings.sample_size == 123

    get_settings.cache_clear()


def test_ensure_directories_creates_artifacts(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("IAP_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("IAP_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("IAP_LOG_DIR", str(tmp_path / "results" / "logs"))
    get_settings.cache_clear()

    settings = get_settings()
    settings.ensure_directories()

    assert settings.models_dir.is_dir()
    assert settings.results_dir.is_dir()
    assert settings.log_dir.is_dir()
    get_settings.cache_clear()
