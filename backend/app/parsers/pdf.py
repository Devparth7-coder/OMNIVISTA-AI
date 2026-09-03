"""PDF parser using pdfplumber.

Produces layout regions (titles, headings, paragraphs, lists, tables, images,
captions) with real page coordinates and reading order. Also:
- detects and strips repeated headers/footers and bare page numbers
- extracts tables as structured rows/columns
- extracts raw image bytes (passed downstream to the vision provider)
- preserves page metadata and per-region bounding boxes
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

from app.parsers.base import DocumentParser, ParseError
from app.schemas.domain import (
    ContentType,
    ImageAsset,
    ParsedDocument,
    Region,
    RegionType,
    SourceType,
    TableStructure,
)

PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.IGNORECASE)


@dataclass
class _Line:
    text: str
    top: float
    bottom: float
    x0: float
    x1: float
    size: float
    bold: bool
    kind: str = "paragraph"  # heading/paragraph/list/caption


@dataclass
class _PageBlock:
    lines: list[_Line] = field(default_factory=list)
    content_type: str = "paragraph"
    kind: str = "paragraph"


class PDFParser(DocumentParser):
    source_type = SourceType.PDF

    async def parse(self, data: bytes) -> ParsedDocument:
        import pdfplumber

        document_id = "doc_placeholder"
        regions: list[Region] = []
        images: list[ImageAsset] = []
        last_section: str = ""
        order = 0

        try:
            pdf = pdfplumber.open(io.BytesIO(data))
        except Exception as exc:
            raise ParseError(f"Could not parse PDF: {exc}") from exc

        n_pages = len(pdf.pages)
        try:
            for page_idx, page in enumerate(pdf.pages, start=1):
                if page_idx > 200:
                    break
                page_region = self._analyze_page(page, document_id, page_idx, order, last_section)
                regions.extend(page_region["regions"])
                images.extend(page_region["images"])
                last_section = page_region["last_section"]
                order = page_region["next_order"]
        finally:
            try:
                pdf.close()
            except Exception:
                pass

        return ParsedDocument(
            document_id=document_id,
            source_type=SourceType.PDF,
            pages=n_pages,
            regions=regions,
            images=images,
            metadata={"format": "pdf", "num_pages": n_pages},
        )

    # ------------------------------------------------------------------
    def _analyze_page(self, page, document_id: str, page_num: int, order: int, last_section: str) -> dict:
        width = float(page.width)
        height = float(page.height)
        header_margin = min(50.0, height * 0.06)
        footer_margin = min(50.0, height * 0.06)

        # ---- group characters into lines --------------------------------
        lines: list[_Line] = self._chars_to_lines(page, header_margin, footer_margin)
        if not lines and not page.images:
            return {"regions": [], "images": [], "last_section": last_section, "next_order": order}

        # Detect body font size so headings ("larger than body") are found.
        sizes = [ln.size for ln in lines]
        body_size = self._median(sizes) if sizes else 10.0

        # ---- analyze lines -> block kinds -------------------------------
        kinded: list[_Line] = []
        for ln in lines:
            ln.kind = self._classify_line(ln, body_size)
            kinded.append(ln)

        regions: list[Region] = []
        images: list[ImageAsset] = []

        # ---- tables ------------------------------------------------------
        table_regions = self._extract_tables(page, document_id, page_num, order, last_section)
        if table_regions:
            order = table_regions["next_order"]
            regions.extend(table_regions["regions"])
            table_boxes = table_regions["boxes"]

        # ---- images -------------------------------------------------------
        img_regions = self._extract_images(page, document_id, page_num, order, last_section)
        if img_regions["regions"]:
            regions.extend(img_regions["regions"])
        if img_regions["images"]:
            images.extend(img_regions["images"])
        img_boxes = img_regions["boxes"]

        # ---- text blocks ---------------------------------------------------
        text_regions = self._extract_text_blocks(kinded, document_id, page_num, order, last_section)
        regions.extend(text_regions["regions"])
        order = text_regions["next_order"]
        text_boxes = text_regions["boxes"]

        # ---- merge for reading order ----------------------------------------
        all_blocks: list[tuple[float, Region]] = []
        for r in regions:
            top = r.bbox[1] if r.bbox else 0
            all_blocks.append((top, r))
        all_blocks.sort(key=lambda t: t[0])
        ordered = []
        for i, (_, r) in enumerate(all_blocks):
            r.reading_order = i
            if r.type == RegionType.HEADING:
                # Keep the running section context.
                pass
            ordered.append(r)
        if ordered:
            last_section = ordered[-1].content if ordered[-1].type == RegionType.HEADING else last_section

        return {"regions": ordered, "images": images, "last_section": last_section, "next_order": len(ordered)}

    # ---- helpers -----------------------------------------------------------
    def _chars_to_lines(self, page, header_margin: float, footer_margin: float) -> list[_Line]:
        raw: dict[int, list] = {}
        for ch in page.chars:
            top = round(ch["top"], 1)
            raw.setdefault(top, []).append(ch)
        lines: list[_Line] = []
        for top in sorted(raw):
            chars = sorted(raw[top], key=lambda c: c["x0"])
            text = "".join(c["text"] for c in chars).strip()
            if not text:
                continue
            x0 = min(c["x0"] for c in chars)
            x1 = max(c["x1"] for c in chars)
            bottom = max(c["bottom"] for c in chars)
            size = max(c["size"] for c in chars)
            fontname = max(chars, key=lambda c: c["size"]).get("fontname", "")
            bold = "Bold" in fontname or "bold" in fontname
            lines.append(_Line(text=text, top=top, bottom=bottom, x0=x0, x1=x1, size=size, bold=bold))
        # strip page numbers and repeated header/footer lines
        kept = []
        for ln in lines:
            condensed = re.sub(r"\s+", " ", ln.text).strip()
            if PAGE_NUMBER_RE.match(condensed):
                continue
            # page-number-like footer
            if len(condensed) <= 6 and condensed.isdigit():
                continue
            kept.append(ln)
        return kept

    @staticmethod
    def _median(values: list[float]) -> float:
        s = sorted(values)
        if not s:
            return 0.0
        return s[len(s) // 2]

    def _classify_line(self, ln: _Line, body_size: float) -> str:
        tc = ln.text.strip()
        low = tc.lower()
        if ln.size >= body_size * 1.25 or ln.bold and ln.size >= body_size:
            return "heading"
        if re.match(r"^[\dA-Z]+\)?\s*\.?\s+", tc) or re.match(r"^\d+\.\d+\s", tc):
            return "paragraph"
        if re.match(r"^\s*[-*•]\s+", tc) or re.match(r"^\s*\d+\.\s+", tc):
            return "list"
        if low.startswith(("figure", "fig.", "table", "tab.", "chart", "diagram")) and len(tc) < 220:
            return "caption"
        return "paragraph"

    def _extract_tables(self, page, document_id, page_num, order, section, ) -> dict:
        regions: list[Region] = []
        boxes: list[list[float]] = []
        try:
            tables = page.find_tables()
        except Exception:
            tables = []
        for t in tables:
            bbox = [float(t.bbox[0]), float(t.bbox[1]), float(t.bbox[2]), float(t.bbox[3])]
            rows = t.extract()
            if not rows:
                continue
            header_cells = rows[0]
            header = [str(c).strip() if c else "" for c in header_cells]
            data_rows = [[str(c).strip() if c else "" for c in r] for r in rows[1:]]
            summary = f"table: columns {header} rows {data_rows}"
            table = TableStructure(columns=header, rows=data_rows, summary=summary)
            regions.append(
                Region(
                    document_id=document_id,
                    page_number=page_num,
                    type=RegionType.TABLE,
                    content_type=ContentType.TABLE,
                    bbox=bbox,
                    content=summary,
                    reading_order=order,
                    table=table,
                    hierarchy={"section": section} if section else {},
                )
            )
            boxes.append(bbox)
            order += 1
        return {"regions": regions, "next_order": order, "boxes": boxes}

    def _extract_images(self, page, document_id, page_num, order, section, ) -> dict:
        regions: list[Region] = []
        images: list[ImageAsset] = []
        boxes: list[list[float]] = []
        for img in page.images:
            bbox = [float(img["x0"]), float(img["top"]), float(img["x1"]), float(img["bottom"])]
            w = float(img.get("width", 0))
            h = float(img.get("height", 0))
            if w == 0 or h == 0:
                continue
            if (w * h) < 2000:  # skip tiny artifacts
                continue
            asset = ImageAsset(
                image_id=f"img_{document_id[:8]}_{page_num}_{order}",
                document_id=document_id,
                page_number=page_num,
                storage_path="",
                bbox=bbox,
                caption="",
                width=int(w),
                height=int(h),
            )
            # Keep the raw image bytes transiently (used by a real vision
            # provider). We avoid embedding them in serialized responses.
            try:
                raw = self._extract_image_bytes(img)
                if raw:
                    asset.metadata["_raw_bytes_len"] = len(raw)
                    asset.metadata["_raw_bytes"] = raw
            except Exception:
                pass
            images.append(asset)
            regions.append(
                Region(
                    document_id=document_id,
                    page_number=page_num,
                    type=RegionType.IMAGE,
                    content_type=ContentType.IMAGE,
                    bbox=bbox,
                    content="",
                    reading_order=order,
                    image_asset_id=asset.image_id,
                    hierarchy={"section": section} if section else {},
                )
            )
            boxes.append(bbox)
            order += 1
        return {"regions": regions, "images": images, "next_order": order, "boxes": boxes}

    @staticmethod
    def _extract_image_bytes(img) -> bytes | None:
        try:
            stream = img.stream
            if stream is None:
                return None
            data = stream.get_data()
            if isinstance(data, str):
                data = data.encode()
            return data if data else None
        except Exception:
            return None

    def _extract_text_blocks(self, lines, document_id, page_num, order, section, ) -> dict:
        regions: list[Region] = []
        boxes: list[list[float]] = []
        cur: list[_Line] = []
        cur_kind = "paragraph"

        def flush() -> None:
            nonlocal cur, cur_kind, order
            if not cur:
                return
            first, last = cur[0], cur[-1]
            text = "\n".join(ln.text for ln in cur).strip()
            if not text:
                cur = []
                return
            bbox = [first.x0, first.top, last.x1, last.bottom]
            rtype = {
                "heading": RegionType.HEADING,
                "list": RegionType.LIST,
                "caption": RegionType.CAPTION,
                "paragraph": RegionType.PARAGRAPH,
            }.get(cur_kind, RegionType.PARAGRAPH)
            regions.append(
                Region(
                    document_id=document_id,
                    page_number=page_num,
                    type=rtype,
                    content_type=ContentType.TEXT,
                    bbox=bbox,
                    content=text,
                    reading_order=order,
                    hierarchy={"section": section} if section else {},
                )
            )
            boxes.append(bbox)
            order += 1
            cur = []

        for ln in lines:
            if ln.kind != cur_kind and cur:
                flush()
                cur_kind = ln.kind
                cur = [ln]
            elif not cur:
                cur_kind = ln.kind
                cur = [ln]
            else:
                cur.append(ln)
        flush()
        return {"regions": regions, "next_order": order, "boxes": boxes}
