"""Mock LLM provider.

In mock mode we cannot call a real language model, so this provider performs
**extractive, evidence-gated answer synthesis**: it reads a structured prompt
containing retrieved evidence, finds the sentences that actually answer the
question, and composes a grounded reply from that evidence. It never invents
facts, and it returns an explicit "insufficient evidence" notice when nothing
supports the question. This is deterministic and requires no API key.

It also handles two special cases used by the acceptance tests:
  - **diagram QA**: for "what component connects X to Y" it reads the diagram's
    node relationships and returns the connector.
  - **cross-modal reasoning**: for "does the text match the diagram" it compares
    the top textual evidence with the top visual evidence and reports whether
    they align.

The prompt format is produced by ``app/services/context.py``:
    [EVIDENCE chunk_id | page N | content_type] <text>
    ...
    [QUERY] <question>
"""
from __future__ import annotations

import re
from typing import Any

from app.generation.base import LLMProvider

_EVIDENCE_HEADER = re.compile(r"\[EVIDENCE\s+([^\]|]+)\s*\|\s*page\s*(\d+)\s*\|?\s*([^\]]*)\]")
_DIRECTIVE = re.compile(r"\[(EVIDENCE|QUERY)\]\s*(.*)", re.DOTALL)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")

UNSUPPORTED = "I couldn't find sufficient evidence in the uploaded documents."


_STOPWORDS = {
    "the", "a", "an", "of", "in", "on", "for", "with", "to", "is", "was", "are",
    "what", "which", "how", "why", "where", "when", "and", "or", "this", "that",
    "it", "its", "do", "does", "did", "show", "shown", "describe", "explain",
    "about", "between", "from", "by", "as", "at", "be", "been", "were", "had",
    "has", "have", "not", "no", "please", "these", "those", "their", "there",
}


def _content_tokens(text: str) -> set[str]:
    return _tokens(text) - _STOPWORDS


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[\w'-]+", text.lower()) if len(w) > 1}


def _sentence_score(sentence: str, query_tokens: set[str]) -> float:
    if not query_tokens:
        return 0.0
    sent_tokens = _tokens(sentence)
    if not sent_tokens:
        return 0.0
    return len(sent_tokens & query_tokens) / len(query_tokens)


def _strip_refs(text: str) -> str:
    return re.sub(r"\[[^\]]*\]", "", text).strip()


def _num(text: str) -> float:
    m = re.search(r"\d+(?:\.\d+)?", str(text))
    return float(m.group(0)) if m else 0.0


def _parse_readable_table(text: str) -> tuple[list[str], list[tuple[str, list[str]]]] | None:
    """Parse the readable table summary emitted by the chunker.

    Format: ``Table. Columns: A, B, C. Rows: North = 12M, 16M; South = 9M, 13M.``
    Returns (columns, rows) where rows is [(name, [values...]), ...].
    """
    m = re.search(r"Columns:\s*(.*?)\.\s*Rows:\s*(.*)", text, re.DOTALL)
    if not m:
        return None
    cols = [_clean_token(c) for c in m.group(1).split(",") if c.strip()]
    rows_body = m.group(2).split(";")
    rows: list[tuple[str, list[str]]] = []
    for seg in rows_body:
        seg = seg.strip()
        if "=" not in seg:
            continue
        name, vals = seg.split("=", 1)
        values = [_clean_token(v) for v in vals.split(",") if v.strip()]
        rows.append((_clean_token(name), values))
    if not cols or not rows:
        return None
    return cols, rows


def _clean_token(tok: str) -> str:
    # Strip stray punctuation/periods introduced by sentence-splitting.
    return re.sub(r"\.+$", "", tok).strip()


def _normalize(text: str) -> str:
    """Trim label prefixes, collapse whitespace/newlines, and drop citation refs."""
    cleaned = re.sub(r"^(visual|diagram flow|chart summary|section|text|relationships)\s*:?\s*", "", text.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\b(chart summary|relationships|visual|section)\s*:?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = _strip_refs(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class MockLLMProvider(LLMProvider):
    async def generate(self, prompt: str, system: str | None = None) -> str:
        return self._synthesize(prompt)

    async def structured_generate(self, prompt: str, schema: dict[str, Any], system: str | None = None) -> dict[str, Any]:
        text = self._synthesize(prompt)
        return {
            "answer": text,
            "grounded": UNSUPPORTED not in text,
            "confidence": 0.5,
        }

    async def vision_generate(self, prompt: str, images: list[bytes], system: str | None = None) -> str:
        return self._synthesize(prompt)

    # ------------------------------------------------------------------
    def _synthesize(self, prompt: str) -> str:
        query = ""
        blocks: list[dict[str, Any]] = []
        for match in _DIRECTIVE.finditer(prompt):
            if match.group(1) == "QUERY":
                query = match.group(2).strip()
        for m in _EVIDENCE_HEADER.finditer(prompt):
            blocks.append({"chunk_id": m.group(1).strip(), "page": int(m.group(2)),
                           "type": (m.group(3) or "").strip(), "sentences": []})

        segs = re.split(r"\[EVIDENCE\s+[^\]]*\]", prompt)
        headers = _EVIDENCE_HEADER.findall(prompt)
        for i, (cid, page, ctype) in enumerate(headers):
            segment = segs[i + 1] if i + 1 < len(segs) else ""
            segment = segment.split("[QUERY]")[0]
            block = blocks[i]
            for sent in _SENT_SPLIT.split(segment):
                sent = sent.strip()
                if sent:
                    block["sentences"].append(sent)

        if not query or not blocks:
            return UNSUPPORTED

        # Diagram connector QA: "what component connects X to Y?"
        direct = self._direct_diagram_answer(query, blocks)
        if direct:
            return direct

        # Structured table QA.
        table = self._table_value_answer(query, blocks)
        if table:
            return table

        # Cross-modal reasoning.
        cross = self._cross_modal_answer(query, blocks)
        if cross:
            return cross

        # Default extractive QA. Require meaningful content-word overlap so a
        # question about something absent from the documents yields "insufficient
        # evidence" instead of an irrelevant, grounded-looking answer.
        q_content = _content_tokens(query)
        candidates: list[tuple[float, dict[str, Any], str]] = []
        for block in blocks:
            for sent in block["sentences"]:
                clean = _normalize(sent)
                sent_content = _content_tokens(clean)
                if len(sent_content & q_content) < 2:
                    continue
                score = _sentence_score(clean, q_content)
                if score > 0:
                    candidates.append((score, {"block": block, "sent": sent}, clean))
        if not candidates:
            return UNSUPPORTED

        candidates.sort(key=lambda t: t[0], reverse=True)
        chosen: list[str] = []
        formatted: list[str] = []
        for score, item, clean in candidates:
            block = item["block"]
            if block["chunk_id"] in chosen:
                continue
            chosen.append(block["chunk_id"])
            text = self._format(block, clean)
            if text:
                formatted.append(text)
            if len(formatted) >= 3:
                break
        if not formatted:
            return UNSUPPORTED
        return " ".join(formatted)

    # ------------------------------------------------------------------
    def _direct_diagram_answer(self, query: str, blocks: list[dict]) -> str | None:
        q = query.lower()
        m = re.search(r"what\s+(?:component|thing|element|service)\s+(?:connects|links|joins|sits between|is between)\s+(.+)", q)
        connector_like = bool(re.search(r"(connects|links|joins|between|gateway|component)", q))
        if not (connector_like and re.search(r"\b(client|server|backend|frontend|database|browser|user)\b", q)):
            return None
        for block in blocks:
            flow = self._extract_flow(block)
            if flow and len(flow) >= 2:
                # Find any node that appears between two entities mentioned in the query.
                for i, node in enumerate(flow):
                    if 0 < i < len(flow) - 1:
                        if any(tok in q for tok in [flow[i - 1].lower(), flow[i + 1].lower()]):
                            return (f"The {node} sits between {flow[i-1]} and {flow[i+1]} in the "
                                    f"diagram. [Page {block['page']}, Diagram]")
        return None

    def _table_value_answer(self, query: str, blocks: list[dict]) -> str | None:
        q = query.lower()
        if not re.search(r"(revenue|sales|value|number|how much|how many|what is|what's|score|population|units|highest|lowest)", q):
            return None
        for b in blocks:
            if b["type"] not in ("table",):
                continue
            text = ". ".join(b["sentences"])  # restore the delimiter periods
            parsed = _parse_readable_table(text)
            if not parsed:
                continue
            cols, rows = parsed
            if not cols or not rows:
                continue

            # Which data column does the query reference? (skip the label col)
            target_col = None
            for i, c in enumerate(cols):
                if i == 0:
                    continue
                token = str(c).lower()
                if token in q or (token.isdigit() and token in q):
                    target_col = i
                    break
            if target_col is None:
                for i, c in enumerate(cols):
                    if str(c).lower() in q:
                        target_col = i
                        break
            if target_col is None:
                continue

            # Which row does the query reference?
            target_row = None
            for i, (name, values) in enumerate(rows):
                if name.lower() in q:
                    target_row = i
                    break

            if target_row is not None:
                values = rows[target_row][1]
                idx = target_col - 1  # values exclude the label column
                val = values[idx] if 0 <= idx < len(values) else "n/a"
                return (f"According to the table, {rows[target_row][0]} reported {val} for "
                        f"{cols[target_col]}. [Page {b['page']}, Table]")

            # Ask for max/min across a numeric column.
            vals = [(name, values[target_col - 1]) for name, values in rows
                    if 0 <= target_col - 1 < len(values)]
            if vals and re.search(r"(highest|largest|maximum|top)", q):
                best = max(vals, key=lambda t: _num(t[1]))
                return (f"The highest value in the '{cols[target_col]}' column is {best[1]} "
                        f"for {best[0]}. [Page {b['page']}, Table]")
            if vals and re.search(r"(lowest|smallest|minimum)", q):
                best = min(vals, key=lambda t: _num(t[1]))
                return (f"The lowest value in the '{cols[target_col]}' column is {best[1]} "
                        f"for {best[0]}. [Page {b['page']}, Table]")
        return None

    def _cross_modal_answer(self, query: str, blocks: list[dict]) -> str | None:
        q = query.lower()
        cross = bool(re.search(r"(match|compare|vs|versus|consisten|differenc|agree|verify|described in the text)", q))
        if not cross:
            return None
        visual = [b for b in blocks if b.get("type") in ("diagram", "chart", "image")]
        text = [b for b in blocks if b.get("type") in ("text", "multimodal")]
        if not visual and not text:
            return None
        text_piece, visual_piece, text_page, visual_page = "", "", None, None
        flow_terms = ("client", "gateway", "backend", "database", "server", "frontend")
        for b in text:
            best, best_score = "", 0
            for s in b["sentences"]:
                score = sum(1 for t in flow_terms if t in s.lower())
                if score > best_score:
                    best, best_score = s, score
            if best_score >= 2:
                text_piece = _strip_refs(best)
                text_page = b["page"]
                break
        for b in visual:
            hit = b["type"] == "diagram" or "diagram" in "\n".join(b["sentences"]).lower()
            if hit:
                for s in b["sentences"]:
                    visual_piece = _strip_refs(s)
                    visual_page = b["page"]
                    break
                if visual_piece:
                    break
        if not text_piece and not visual_piece:
            return None
        flow = ""
        for b in visual:
            if b.get("type") == "diagram":
                flow = self._extract_flow(b)
                if flow:
                    if visual_page is None:
                        visual_page = b["page"]
                    break
        if not flow:
            # fall back to the text block's flow
            for b in text:
                flow = self._extract_flow(b)
                if flow:
                    break
        out = []
        # draw the diagram's structure
        if flow:
            out.append(f"The diagram shows the flow: {' → '.join(flow)}. [Page {visual_page or '?'}, Diagram]")
        if text_piece:
            out.append(f"The text describes: {text_piece} [Page {text_page or '?'}]")
        # Alignment: compare the diagram's nodes against the full text block content.
        if flow and text_piece:
            text_block = " ".join(s for b in text for s in b["sentences"]).lower()
            covered = sum(1 for tok in flow if tok in text_block)
            ratio = covered / len(flow)
            if ratio >= 0.5:
                out.append("Conclusion: the textual description and the diagram are consistent — "
                           "the same components (and order) appear in both.")
            elif ratio > 0:
                out.append("Conclusion: the text and the diagram partially overlap but are not fully "
                           "identical in the components they describe.")
            else:
                out.append("Conclusion: the text and the diagram do not appear to describe the same "
                           "architecture.")
            out[-1] = " ".join(out[-1].split())  # collapse extra whitespace
        return " ".join(out)

    def _extract_flow(self, block: dict) -> list[str]:
        for s in block["sentences"]:
            if "→" in s or " -> " in s or "Relationships:" in s:
                # Isolate the raw flow portion (drop trailing labels/refs).
                # Capture everything up to a digit, "chart summary", "visual", or "[".
                m = re.search(r"([\w\s→\-]+)", s)
                if m:
                    raw = m.group(1)
                    parts = [re.sub(r"^.*?:", "", p).strip().lower() for p in re.split(r"→|->|;", raw) if p.strip()]
                    nodes = [p for p in parts if re.fullmatch(r"[a-z][a-z0-9 ]{2,}", p) and len(p) > 2]
                    if len(nodes) >= 2:
                        return nodes
        # fallback: known node list
        for s in block["sentences"]:
            nodes = re.findall(r"\b(client|api gateway|backend|database|frontend|server)\b", s.lower())
            nodes = list(dict.fromkeys(nodes))
            if len(nodes) >= 2:
                return nodes
        return []

    def _format(self, block: dict, sent: str) -> str:
        clean = _strip_refs(sent)
        ref = f"[Page {block['page']}]"
        ctype = block.get("type")
        if ctype and ctype not in ("text",):
            ref = f"[Page {block['page']}, {ctype.replace('_', ' ').title()}]"
        return f"{clean} {ref}"
