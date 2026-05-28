from __future__ import annotations

from dataclasses import dataclass, field

_FITZPATRICK_NUMERAL = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}

ABCDE_FEATURES: tuple[str, ...] = (
    "asymmetric outline",
    "irregular border",
    "variegated colour",
    "diameter beyond six millimetres",
    "evolving surface",
    "irregular pigment network",
    "blue-white veil",
    "atypical vascular pattern",
)

ANATOMICAL_SITES: tuple[str, ...] = (
    "the back",
    "the forearm",
    "the lower leg",
    "the face",
    "the scalp",
    "the trunk",
)

BENIGN_DIAGNOSES: tuple[str, ...] = (
    "nevus",
    "seborrheic keratosis",
    "dermatofibroma",
    "benign lesion",
)


@dataclass
class LesionMeta:
    diagnosis: str
    location: str
    fitzpatrick: int
    features: tuple[str, ...] = field(default_factory=tuple)
    age: int | None = None
    sex: str | None = None


def _feature_clause(features: tuple[str, ...]) -> str:
    if not features:
        return "no distinguishing dermoscopic features"
    if len(features) == 1:
        return features[0]
    return ", ".join(features[:-1]) + " and " + features[-1]


def render_descriptor(meta: LesionMeta, mode: str = "patient") -> str:
    numeral = _FITZPATRICK_NUMERAL.get(meta.fitzpatrick, "III")
    clause = _feature_clause(meta.features)
    if mode == "class":
        return (
            f"{meta.diagnosis} lesion presenting {clause} "
            f"described with ABCDE dermoscopic terminology"
        )
    age = meta.age if meta.age is not None else "unknown age"
    sex = meta.sex if meta.sex is not None else "unspecified"
    age_clause = f"{age} year old" if isinstance(age, int) else age
    return (
        f"{meta.diagnosis} lesion on {meta.location} presenting {clause}; "
        f"in a {age_clause} {sex} individual with Fitzpatrick type {numeral}"
    )
