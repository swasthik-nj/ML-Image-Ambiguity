"""Application configuration loaded from environment variables and YAML."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, field_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load a YAML configuration file into a plain dictionary.

    Args:
        path: Optional path to a YAML file. Defaults to ``configs/default.yaml``.

    Returns:
        Parsed YAML mapping, or an empty dict if the file is missing.
    """
    config_path = path or (PROJECT_ROOT / "configs" / "default.yaml")
    if not config_path.is_file():
        return {}

    with config_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {config_path}")
    return data


def flatten_config(data: dict[str, Any], parent_key: str = "") -> dict[str, Any]:
    """Flatten one level of nested YAML sections into Settings field names."""
    flat: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict) and parent_key == "":
            # Support sections like paths:, experiment:, api:
            flat.update(flatten_config(value, parent_key=key))
        else:
            flat[key] = value
    return flat


class Settings(BaseSettings):
    """Central runtime settings for the ambiguity prediction project.

    Priority (highest to lowest):
    1. Process environment variables (``IAP_*``)
    2. ``.env`` file
    3. ``configs/default.yaml``
    4. Model defaults
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        env_prefix="IAP_",
        extra="ignore",
    )

    # Environment
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"
    random_seed: int = 42

    # Paths
    project_root: Path = PROJECT_ROOT
    config_path: Path = PROJECT_ROOT / "configs" / "default.yaml"
    dataset_dir: Path = PROJECT_ROOT / "dataset"
    annotation_file: Path = (
        PROJECT_ROOT / "dataset" / "annotations" / "captions_val2017.json"
    )
    image_dir: Path = PROJECT_ROOT / "dataset" / "val2017"
    models_dir: Path = PROJECT_ROOT / "models"
    results_dir: Path = PROJECT_ROOT / "results"
    log_dir: Path = PROJECT_ROOT / "results" / "logs"
    embeddings_dir: Path = PROJECT_ROOT / "results" / "embeddings"

    # Experiment
    experiment_name: str = "image_ambiguity_baseline"
    sample_size: int = 500
    train_ratio: float = Field(default=0.7, ge=0.0, le=1.0)
    val_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.15, ge=0.0, le=1.0)

    # API
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    api_reload: bool = True
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:5173"])

    # Model / feature defaults
    sentence_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    embedding_batch_size: int = Field(default=32, ge=1)

    @field_validator(
        "project_root",
        "config_path",
        "dataset_dir",
        "annotation_file",
        "image_dir",
        "models_dir",
        "results_dir",
        "log_dir",
        "embeddings_dir",
        mode="before",
    )
    @classmethod
    def _coerce_path(cls, value: Any) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )

    def ensure_directories(self) -> None:
        """Create writable artifact directories if they do not exist."""
        for path in (
            self.models_dir,
            self.results_dir,
            self.log_dir,
            self.embeddings_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


class YamlConfigSettingsSource(PydanticBaseSettingsSource):
    """Load flat/nested keys from ``configs/default.yaml``."""

    def get_field_value(
        self, field: Any, field_name: str
    ) -> tuple[Any, str, bool]:
        data = flatten_config(load_yaml_config())
        return data.get(field_name), field_name, False

    def __call__(self) -> dict[str, Any]:
        data = flatten_config(load_yaml_config())
        return {
            key: value
            for key, value in data.items()
            if key in self.settings_cls.model_fields
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` instance."""
    return Settings()
