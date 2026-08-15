"""Visual simplicity assessment from OpenCV features (inference gating only)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class VisualSimplicityConfig:
    """Thresholds for voting whether an image is visually simple.

    Each criterion that holds contributes one vote. The image is treated as
    simple when ``simple_votes / n_criteria >= min_vote_fraction`` (or when
    ``simple_votes >= min_simple_votes`` if that is set and stricter).

    These thresholds only select BLIP generation / clustering behaviour.
    They never override Low / Medium / High class definitions.
    """

    enabled: bool = True
    edge_density_max: float = 0.040
    entropy_max: float = 7.25
    texture_max: float = 900.0
    contrast_max: float = 55.0
    color_variance_max: float = 3200.0
    # Optional: plain/bright backgrounds (e.g. product on white). None disables.
    brightness_min: float | None = 165.0
    min_simple_votes: int = 3
    min_vote_fraction: float = 0.5


@dataclass(frozen=True)
class VisualSimplicityAssessment:
    """Result of comparing OpenCV features to simplicity thresholds."""

    is_simple: bool
    simplicity_score: float
    simple_votes: int
    n_criteria: int
    votes: dict[str, bool]
    thresholds_used: dict[str, float | None]
    enabled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def assess_visual_simplicity(
    opencv_features: Mapping[str, float],
    config: VisualSimplicityConfig | None = None,
) -> VisualSimplicityAssessment:
    """Vote on visual simplicity using existing OpenCV feature values.

    Args:
        opencv_features: Mapping with keys such as ``edge_density``,
            ``entropy``, ``texture``, ``contrast``, ``color_variance``,
            ``brightness``.
        config: Optional threshold configuration.

    Returns:
        :class:`VisualSimplicityAssessment`. When ``config.enabled`` is
        false, ``is_simple`` is always ``False`` (diverse BLIP path).
    """
    cfg = config or VisualSimplicityConfig()
    votes: dict[str, bool] = {
        "low_edge_density": float(opencv_features.get("edge_density", 1.0))
        <= cfg.edge_density_max,
        "low_entropy": float(opencv_features.get("entropy", 8.0)) <= cfg.entropy_max,
        "low_texture": float(opencv_features.get("texture", 1e9)) <= cfg.texture_max,
        "low_contrast": float(opencv_features.get("contrast", 1e9))
        <= cfg.contrast_max,
        "low_color_variance": float(opencv_features.get("color_variance", 1e9))
        <= cfg.color_variance_max,
    }
    if cfg.brightness_min is not None:
        votes["high_brightness"] = (
            float(opencv_features.get("brightness", 0.0)) >= cfg.brightness_min
        )

    simple_votes = sum(1 for held in votes.values() if held)
    n_criteria = len(votes)
    fraction = simple_votes / n_criteria if n_criteria else 0.0
    meets_votes = simple_votes >= cfg.min_simple_votes
    meets_fraction = fraction >= cfg.min_vote_fraction
    is_simple = bool(cfg.enabled and meets_votes and meets_fraction)

    thresholds = {
        "edge_density_max": cfg.edge_density_max,
        "entropy_max": cfg.entropy_max,
        "texture_max": cfg.texture_max,
        "contrast_max": cfg.contrast_max,
        "color_variance_max": cfg.color_variance_max,
        "brightness_min": cfg.brightness_min,
        "min_simple_votes": float(cfg.min_simple_votes),
        "min_vote_fraction": cfg.min_vote_fraction,
    }
    return VisualSimplicityAssessment(
        is_simple=is_simple,
        simplicity_score=float(fraction),
        simple_votes=int(simple_votes),
        n_criteria=int(n_criteria),
        votes=votes,
        thresholds_used=thresholds,
        enabled=bool(cfg.enabled),
    )
