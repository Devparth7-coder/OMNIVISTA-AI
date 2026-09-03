"""Query understanding.

Classifies the question's required modality (text/table/image/diagram/mixed),
detects page references, entities, and whether visual evidence is needed. This
drives the retrieval router.
"""
from __future__ import annotations

import re

from app.schemas.domain import ContentType, QueryIntent

_PAGE_RE = re.compile(r"(?:page|pg\.?)\s*(\d+)", re.IGNORECASE)
_VISUAL_WORDS = re.compile(
    r"\b(diagram|figure|fig\.|chart|graph|image|screenshot|flowchart|architecture|"
    r"floor ?plan|illustration|plot|table\s+\d|visual|picture|diagram|blueprint)\b",
    re.IGNORECASE,
)
_TABLE_WORDS = re.compile(r"\btable\b|row|column|columns", re.IGNORECASE)
_DIAGRAM_WORDS = re.compile(
    r"\b(architecture|flowchart|flow|diagram|network|schema|topology|circuit|uml|org chart|process)\b",
    re.IGNORECASE,
)
_CHART_WORDS = re.compile(r"\b(chart|graph|plot|bar chart|line graph|pie chart)\b", re.IGNORECASE)
_IMAGE_WORDS = re.compile(r"\b(image|picture|screenshot|photo|snapshot|figure)\b", re.IGNORECASE)
_NUMBER_WORDS = re.compile(
    r"\b(which year|highest|largest|compare|how many|sum|total|increase|decrease|percentage|trend|revenue)\b",
    re.IGNORECASE,
)
_CROSS_MODAL_WORDS = re.compile(
    r"\b(match|matches|matches?|compatib|consistent|verbs|described in (the )?text|"
    r"text (and|vs|versus|against)|differenc|agreement|conflict|verify)\b",
    re.IGNORECASE,
)


def analyze_query(query: str, doc_ids: list[str] | None = None) -> QueryIntent:
    q = query.strip()
    intent = QueryIntent(original=q, page=None)

    # page reference
    m = _PAGE_RE.search(q)
    if m:
        intent.page = int(m.group(1))

    # modality detection
    has_visual = bool(_VISUAL_WORDS.search(q))
    has_table = bool(_TABLE_WORDS.search(q))
    has_diagram = bool(_DIAGRAM_WORDS.search(q))
    has_chart = bool(_CHART_WORDS.search(q))
    has_image = bool(_IMAGE_WORDS.search(q))
    numeric = bool(_NUMBER_WORDS.search(q))
    cross = bool(_CROSS_MODAL_WORDS.search(q))
    imps = [has_diagram, has_chart, has_image, has_table]
    visual_need = has_visual or has_image or has_diagram or has_chart
    intent.needs_visual = visual_need

    # Explicit cross-modal reasoning (compare text vs diagram) → mixed.
    if cross and (has_diagram or has_image or has_chart):
        intent.modality = "mixed"
        intent.content_type = None
    elif sum(imps) >= 2:
        intent.modality = "mixed"
        intent.content_type = None
    elif has_diagram:
        intent.modality = "diagram"
        intent.content_type = ContentType.DIAGRAM
    elif has_chart:
        intent.modality = "chart"
        intent.content_type = ContentType.CHART
    elif has_table or (numeric and not visual_need):
        intent.modality = "table"
        intent.content_type = ContentType.TABLE
    elif has_image:
        intent.modality = "image"
        intent.content_type = ContentType.IMAGE
    else:
        intent.modality = "text"
        intent.content_type = ContentType.TEXT

    # entities
    intent.entities = _extract_entities(q)

    # document references
    if doc_ids:
        intent.filters["document_ids"] = doc_ids
    if intent.page is not None:
        intent.filters["page"] = intent.page
    if intent.content_type:
        intent.filters["content_type"] = intent.content_type.value
    return intent


_T_TEXT = re.compile(r"\b(describe|explain|summarize|what|which|how|why|compare|difference|methodology|demonstrate)\b", re.IGNORECASE)
_TEXT_ANALYSIS_WORDS = _T_TEXT


def _extract_entities(q: str) -> list[str]:
    # naive proper-noun / capitalized token extraction
    tokens = re.findall(r"\b[A-Z][A-Za-z0-9\-]+(?:\s+[A-Z][A-Za-z0-9\-]+)*\b", q)
    # de-dupe and trim multi-word to single
    seen: list[str] = []
    for t in tokens:
        first = t.split()[0]
        if first not in seen and len(first) > 1:
            seen.append(first)
    return seen[:6]
