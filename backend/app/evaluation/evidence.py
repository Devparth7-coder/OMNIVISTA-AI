"""Evidence validator + hallucination metrics.

Before an answer is returned, we verify that its claims are grounded in the
retrieved evidence, that citations resolve to real chunks, and that any numbers
quoted actually appear in the source. This is a deterministic approximation; a
production deployment would add an LLM-based faithfulness scorer alongside it.
"""
from __future__ import annotations

import re

from app.schemas.domain import AnswerEvidence, RetrievedChunk

_STOP = set(
    "the a an of to in and for on with is are was were this that it as at be by from or "
    "its their there which what when how why not no yes i you we they he she has have had "
    "using use used does do did according based only states stated can could will would might "
    "likely section page diagram figure table text following".split()
)


class EvidenceValidator:
    def __init__(self) -> None:
        self._valid_chunk_ids: dict[str, bool] = {}

    def validate(self, answer: str, chunks: list[RetrievedChunk], used_chunk_ids: list[str]) -> AnswerEvidence:
        evidence_text = " ".join(c.text + " " + c.visual_context for c in chunks).lower()
        evidence_sentences = _sentences(evidence_text)

        if not chunks:
            return AnswerEvidence(grounded=False, confidence=0.1, citations_valid=False, used_chunks=0)

        # Citation validity
        citations_valid = all(any(cid == c.chunk_id for c in chunks) for cid in used_chunk_ids) if used_chunk_ids else True

        # Groundedness: coverage of answer content words by the evidence corpus.
        answer_words = _content_words(answer)
        if not answer_words:
            return AnswerEvidence(grounded=False, confidence=0.1, citations_valid=citations_valid, used_chunks=len(chunks))

        covered = sum(1 for w in answer_words if w in evidence_sentences)
        coverage = covered / len(answer_words)

        # Numeric consistency: every numeric value mentioned in the answer must
        # appear in the evidence (compare the base number, ignoring units and
        # trailing decimal formatting like 10 vs 10.0).
        answer_nums = _numbers(answer)
        evidence_nums = _numbers(evidence_text)
        numeric_ok = answer_nums.issubset(evidence_nums)

        unsupported = _find_unsupported(answer, evidence_sentences)
        grounded = coverage >= 0.4 and numeric_ok and len(unsupported) == 0 and citations_valid
        confidence = round(0.4 + 0.4 * coverage + (0.1 if numeric_ok else 0.0), 2)
        confidence = min(0.97, max(0.2, confidence))

        return AnswerEvidence(
            grounded=grounded,
            confidence=confidence,
            unsupported_claims=unsupported,
            citations_valid=citations_valid,
            used_chunks=len(chunks),
        )


def _content_words(text: str) -> list[str]:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'-]{1,}", text.lower()) if w not in _STOP]


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    """Normalized set of numeric values (e.g. '10', '10.0' -> {'10'})."""
    out: set[str] = set()
    for m in _NUMBER_RE.findall(text):
        out.add(str(int(float(m))) if "." in m else m)
    return out


def _sentences(text: str) -> set[str]:
    words = set(re.findall(r"[a-z0-9'-]+", text))
    return words


# Meta/disclosure clauses that are not factual claims about the documents.
_META_CLAUSE_TOKENS = {"retrieved", "evidence", "stated", "document", "content", "reliance", "stated"}


def _find_unsupported(answer: str, evidence_words: set[str]) -> list[str]:
    """Heuristic: find factual clauses whose content words are mostly absent from evidence.

    Meta/disclosure clauses (e.g. "based on the retrieved evidence") are ignored
    because they are not claims about the documents.
    """
    clauses = re.split(r"[.;]\s+", answer)
    bad: list[str] = []
    for clause in clauses:
        cw = _content_words(clause)
        if len(cw) < 3:
            continue
        meta_ratio = sum(1 for w in cw if w in _META_CLAUSE_TOKENS) / len(cw)
        if meta_ratio > 0.5:
            continue
        present = sum(1 for w in cw if w in evidence_words)
        if present / len(cw) < 0.4:
            bad.append(clause.strip())
    return bad[:5]
