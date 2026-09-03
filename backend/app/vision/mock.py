"""Mock vision provider.

In mock mode we cannot genuinely "see" an image, so we build a description from
the surrounding context (the caption / nearby text the extractor attaches as a
hint) and from layout heuristics. This is honest about its limits: the hint is
*document text*, which is exactly the kind of signal a real VLM would draw on.
The result is deterministic and dependency-free, so the full multimodal pipeline
runs out of the box.

For charts it also parses a compact numeric summary from the hint (e.g. "2022 10M,
2023 14M, 2024 19M") so chart QA like "which year had the highest revenue?" can
be answered from the chart itself.
"""
from __future__ import annotations

import re

from pydantic import Field

from app.vision.base import VisionAnnotation, VisionProvider

_ARCH_PATTERN = re.compile(
    r"\b(client|api gateway|backend|server|frontend|database|load balancer|cache|gateway|auth)\b",
    re.IGNORECASE,
)
_YEAR_VALUE = re.compile(r"(\d{4})\s*[:=-]?\s*([\d.,]+)\s*(M|k|K|B|million|billion|%)?")


class MockVisionProvider(VisionProvider):
    async def describe(self, image_bytes: bytes, hint: str = "") -> VisionAnnotation:
        hint = hint or ""
        objects = [o.title() for o in dict.fromkeys(_ARCH_PATTERN.findall(hint))]
        chart_type = self._detect_chart_type(hint)
        relationships = self._build_relations(objects)

        chart_data: dict = {}
        summary = ""
        if chart_type:
            pairs = self._parse_year_values(hint)
            chart_data = {"chart_type": chart_type, "pairs": pairs}
            summary = self._chart_summary(chart_type, hint, pairs)

        description = hint.strip() or "An unattributed image with no surrounding caption."
        if summary:
            description = f"{description} Summary: {summary}"

        return VisionAnnotation(
            description=description,
            objects=objects,
            labels=objects,
            chart_type="diagram" if chart_type == "diagram" else chart_type,
            chart_data=chart_data,
            relationships=relationships,
            summary=summary,
            confidence=0.9 if hint else 0.5,
        )

    @staticmethod
    def _detect_chart_type(hint: str) -> str | None:
        low = hint.lower()
        if "architecture" in low or "diagram" in low or "flowchart" in low or "flow" in low:
            return "diagram"
        if "bar" in low:
            return "bar"
        if "line" in low or "trend" in low:
            return "line"
        if "pie" in low:
            return "pie"
        if "chart" in low or "graph" in low or "plot" in low:
            return "chart"
        return None

    @staticmethod
    def _parse_year_values(hint: str) -> list[dict]:
        rows: list[dict] = []
        for m in _YEAR_VALUE.finditer(hint):
            year, val, unit = m.group(1), m.group(2), (m.group(3) or "")
            rows.append({"label": year, "value": float(val.replace(",", "")), "unit": unit})
        return rows

    @staticmethod
    def _chart_summary(chart_type: str, hint: str, pairs: list[dict]) -> str:
        if chart_type == "diagram":
            return "architecture flow diagram"
        if not pairs:
            return f"{chart_type.replace('_', ' ')} chart"
        values = [p["value"] for p in pairs]
        # determine highest/lowest for "which year had the highest..." style QA
        top = max(pairs, key=lambda p: p["value"])
        low = min(pairs, key=lambda p: p["value"])
        unit = top.get("unit", "")
        parts = ", ".join(f"{p['label']} {p['value']}{p['unit']}" for p in pairs)
        return (f"{chart_type.replace('_', ' ').title()} of {len(pairs)} categories — "
                f"values {parts}. The highest value is {top['label']} at {top['value']}{unit} "
                f"and the lowest is {low['label']} at {low['value']}{low.get('unit','')}.")

    @staticmethod
    def _build_relations(objects: list[str]) -> list[dict]:
        flow = ["client", "api gateway", "gateway", "backend", "server", "auth", "database"]
        tokens = [o.lower() for o in objects]
        filtered = [t for t in tokens if t in flow]
        rel = []
        if len(filtered) >= 2:
            for i in range(len(filtered) - 1):
                rel.append({"from": filtered[i], "to": filtered[i + 1]})
        return rel
