"""Plain-text and Markdown parser.

Both are treated as a single-page linear document. Markdown headings become
``heading`` regions with a hierarchy; everything else is paragraph/list.
"""
from __future__ import annotations

import re

from app.parsers.base import DocumentParser, ParseError
from app.schemas.domain import ParsedDocument, Region, RegionType, SourceType

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_LIST_RE = re.compile(r"^\s*[-*+]\s+(.*)")
_ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+(.*)")


class TextParser(DocumentParser):
    source_type = SourceType.TXT

    async def parse(self, data: bytes) -> ParsedDocument:
        try:
            text = data.decode("utf-8", errors="replace")
        except Exception as exc:
            raise ParseError(str(exc)) from exc
        return self._parse_text(text)

    def _parse_text(self, text: str, source_type: SourceType | None = None, document_id: str | None = None) -> ParsedDocument:
        source_type = source_type or self.source_type
        document_id = document_id or "doc_placeholder"
        regions: list[Region] = []
        section_stack: list[str] = []
        order = 0
        for line in text.splitlines():
            line = line.rstrip()
            if not line.strip():
                continue

            m = _HEADING_RE.match(line)
            if m:
                level = m.group(1).count("#")
                title = m.group(2).strip()
                hierarchy = {f"h{i+1}": lvl for i, lvl in enumerate(section_stack)}
                regions.append(
                    Region(
                        document_id=document_id,
                        page_number=1,
                        type=RegionType.HEADING,
                        content=title,
                        reading_order=order,
                        hierarchy=hierarchy,
                        metadata={"level": level},
                        source_type=source_type,
                    )
                )
                # Maintain heading tree (truncate to depth)
                while len(section_stack) >= level:
                    section_stack.pop()
                section_stack.append(title)
                order += 1
            elif _LIST_RE.match(line) or _ORDERED_LIST_RE.match(line):
                content = _LIST_RE.sub(r"\1", line) or _ORDERED_LIST_RE.sub(r"\1", line)
                regions.append(
                    Region(
                        document_id=document_id,
                        page_number=1,
                        type=RegionType.LIST,
                        content=content.strip(),
                        reading_order=order,
                        hierarchy={f"h{i+1}": lvl for i, lvl in enumerate(section_stack)},
                        source_type=source_type,
                    )
                )
                order += 1
            else:
                regions.append(
                    Region(
                        document_id=document_id,
                        page_number=1,
                        type=RegionType.PARAGRAPH,
                        content=line.strip(),
                        reading_order=order,
                        hierarchy={f"h{i+1}": lvl for i, lvl in enumerate(section_stack)},
                        source_type=source_type,
                    )
                )
                order += 1

        return ParsedDocument(
            document_id=document_id,
            source_type=source_type,
            pages=max(1, 1 if regions else 1),
            regions=regions,
            metadata={"format": source_type.value},
        )


class MarkdownParser(TextParser):
    source_type = SourceType.MD
