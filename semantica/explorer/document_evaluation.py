"""Exact-match evaluation helpers for the offline document extractor."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .routes.documents import (
    _ONTOLOGY_PROFILES,
    _cjk_entities,
    _cjk_relations,
    _entity_canonical_text,
)


def _metrics(predicted: set[Tuple[str, ...]], expected: set[Tuple[str, ...]]) -> Dict[str, float | int]:
    true_positive = len(predicted & expected)
    false_positive = len(predicted - expected)
    false_negative = len(expected - predicted)
    precision = true_positive / (true_positive + false_positive) if predicted else (1.0 if not expected else 0.0)
    recall = true_positive / (true_positive + false_negative) if expected else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def evaluate_samples(samples: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    predicted_entities: set[Tuple[str, ...]] = set()
    expected_entities: set[Tuple[str, ...]] = set()
    predicted_relations: set[Tuple[str, ...]] = set()
    expected_relations: set[Tuple[str, ...]] = set()
    details: List[Dict[str, Any]] = []

    for sample in samples:
        sample_id = str(sample["id"])
        profile = str(sample.get("profile", "general"))
        definition = _ONTOLOGY_PROFILES[profile]
        entities = _cjk_entities(str(sample["text"]), definition["entity_types"])
        relations = _cjk_relations(str(sample["text"]), entities, definition["relation_types"])

        sample_predicted_entities = {
            (_entity_canonical_text(entity), str(entity.label).upper())
            for entity in entities
        }
        sample_expected_entities = {
            (str(text), str(label).upper())
            for text, label in sample.get("entities", [])
        }
        sample_predicted_relations = {
            (
                _entity_canonical_text(relation.subject),
                str(relation.predicate).lower(),
                _entity_canonical_text(relation.object),
            )
            for relation in relations
        }
        sample_expected_relations = {
            (str(source), str(predicate).lower(), str(target))
            for source, predicate, target in sample.get("relations", [])
        }

        predicted_entities.update((sample_id, *item) for item in sample_predicted_entities)
        expected_entities.update((sample_id, *item) for item in sample_expected_entities)
        predicted_relations.update((sample_id, *item) for item in sample_predicted_relations)
        expected_relations.update((sample_id, *item) for item in sample_expected_relations)
        details.append({
            "id": sample_id,
            "entity_metrics": _metrics(sample_predicted_entities, sample_expected_entities),
            "relation_metrics": _metrics(sample_predicted_relations, sample_expected_relations),
            "unexpected_entities": sorted(sample_predicted_entities - sample_expected_entities),
            "missing_entities": sorted(sample_expected_entities - sample_predicted_entities),
            "unexpected_relations": sorted(sample_predicted_relations - sample_expected_relations),
            "missing_relations": sorted(sample_expected_relations - sample_predicted_relations),
        })

    return {
        "sample_count": len(details),
        "entities": _metrics(predicted_entities, expected_entities),
        "relations": _metrics(predicted_relations, expected_relations),
        "samples": details,
    }


def evaluate_file(path: Path) -> Dict[str, Any]:
    samples: Sequence[Dict[str, Any]] = json.loads(path.read_text(encoding="utf-8"))
    return evaluate_samples(samples)
