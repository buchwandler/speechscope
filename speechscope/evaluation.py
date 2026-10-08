"""Policy outcomes are distinct from observations."""

import math
from dataclasses import fields

from .errors import InvalidThresholdPolicyError
from .types import ThresholdPolicy, VerificationMetrics


def parse_policy(value: dict) -> ThresholdPolicy:
    allowed = {f.name for f in fields(ThresholdPolicy)}
    if not isinstance(value, dict) or set(value) - allowed:
        raise InvalidThresholdPolicyError("Unknown threshold policy fields")
    try:
        policy = ThresholdPolicy(**value)
    except (ValueError, TypeError) as exc:
        raise InvalidThresholdPolicyError(str(exc)) from exc
    if policy.schema_version != "1" or policy.missing_timing not in ("review", "fail"):
        raise InvalidThresholdPolicyError("Unsupported policy schema or missing_timing action")
    for key in ("max_wer", "max_cer", "max_mer", "min_reference_timestamp_coverage"):
        v = getattr(policy, key)
        if v is not None and (
            not isinstance(v, (float, int))
            or isinstance(v, bool)
            or not math.isfinite(v)
            or v < 0
            or (key == "min_reference_timestamp_coverage" and v > 1)
        ):
            raise InvalidThresholdPolicyError(f"Invalid value for {key}")
    v = policy.max_invalid_timestamp_count
    if v is not None and (type(v) is not int or v < 0):
        raise InvalidThresholdPolicyError("Invalid max_invalid_timestamp_count")
    return policy


def evaluate(metrics: VerificationMetrics, policy: ThresholdPolicy | None) -> dict:
    if policy is None:
        return {"status": "not_evaluated", "rules": []}
    parse_policy({f.name: getattr(policy, f.name) for f in fields(ThresholdPolicy)})
    rules = []
    for key, op in (
        ("max_wer", "max"),
        ("max_cer", "max"),
        ("max_mer", "max"),
        ("min_reference_timestamp_coverage", "min"),
        ("max_invalid_timestamp_count", "max"),
    ):
        limit = getattr(policy, key)
        if limit is None:
            continue
        metric_key = {
            "max_wer": "wer",
            "max_cer": "cer",
            "max_mer": "mer",
            "min_reference_timestamp_coverage": "reference_timestamp_coverage_ratio",
            "max_invalid_timestamp_count": "invalid_timestamp_count",
        }[key]
        observed = getattr(metrics, metric_key)
        if (
            key == "min_reference_timestamp_coverage"
            and metrics.hypothesis_timestamp_coverage_ratio == 0
        ):
            outcome = policy.missing_timing
        else:
            outcome = (
                "pass" if (observed <= limit if op == "max" else observed >= limit) else "fail"
            )
        rules.append({"name": key, "observed": observed, "expected": limit, "status": outcome})
    status = (
        "fail"
        if any(r["status"] == "fail" for r in rules)
        else ("review" if any(r["status"] == "review" for r in rules) else "pass")
    )
    return {"status": status, "rules": rules}
