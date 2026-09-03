"""Assigns vision annotations to the images detected in a parsed document.

For each image/chart/diagram region we derive a *hint* from the surrounding
document text (the nearest caption, or the section heading) and ask the vision
provider to annotate it. The annotation (description/objects/labels/chart_type)
is stored in ``image_meta`` (consumed by the chunker) and written back onto the
:class:`ImageAsset`.
"""
from __future__ import annotations

from app.schemas.domain import ImageAsset, ParsedDocument, Region, RegionType
from app.vision.base import VisionProvider

_CAPTION_HINTS = ("figure", "fig.", "table", "chart", "diagram", "architecture")


async def annotate_images(parsed: ParsedDocument, vision: VisionProvider) -> dict:
    # Build a per-page index of caption and heading regions with bboxes.
    page_ctx: dict[int, list[Region]] = {}
    for r in parsed.regions:
        page_ctx.setdefault(r.page_number, []).append(r)

    image_meta: dict[str, dict] = {}
    assets: list[ImageAsset] = []

    image_regions = [r for r in parsed.regions if r.image_asset_id]
    known_assets = {img.image_id: img for img in parsed.images}

    for region in image_regions:
        image_id = region.image_asset_id
        asset = known_assets.get(image_id)
        hint = _find_hint(region, page_ctx.get(region.page_number, []), asset)
        raw = (asset.metadata.get("_raw_bytes") if asset and asset.metadata else None) or b""
        annotation = await vision.describe(raw if isinstance(raw, bytes) else b"", hint=hint)
        meta = annotation.model_dump()
        image_meta[image_id] = meta
        if asset is not None:
            asset.description = meta["description"]
            asset.objects = meta["objects"]
            asset.labels = meta["labels"]
            asset.caption = hint if hint.startswith(("figure", "fig", "diagram", "architecture")) else ""
            if meta.get("chart_type"):
                asset.chart_data = {"chart_type": meta["chart_type"], "relationships": meta["relationships"]}
            assets.append(asset)
        else:
            assets.append(
                ImageAsset(
                    image_id=image_id, document_id=parsed.document_id,
                    page_number=region.page_number, storage_path="", bbox=region.bbox,
                    description=meta["description"], objects=meta["objects"],
                    labels=meta["labels"], caption=hint,
                    chart_data={"chart_type": meta["chart_type"], "relationships": meta["relationships"]} if meta.get("chart_type") else None,
                )
            )

    return {"image_meta": image_meta, "images": assets}


def _find_hint(region: Region, page_regions: list[Region], asset: ImageAsset | None) -> str:
    # 1. Nearest caption region on the same page (below/around the image fully).
    if asset and asset.caption:
        return asset.caption
    bbox = region.bbox or asset.bbox if asset else None
    candidates = []
    for other in page_regions:
        if other.type == RegionType.CAPTION and other.region_id != region.region_id and other.bbox:
            if bbox:
                gap = other.bbox[1] - (bbox[3] if len(bbox) > 3 else 0)  # caption top vs image bottom
                if -60 <= gap <= 220:
                    candidates.append((abs(gap), other.content))
            else:
                candidates.append((0, other.content))
    s = sorted(candidates, key=lambda c: c[0])
    if s:
        return s[0][1].strip()
    # 2. Any caption region (regardless of position).
    for other in page_regions:
        if other.type == RegionType.CAPTION:
            return other.content.strip()
    # 3. The current section heading.
    hierarchy = region.hierarchy or {}
    h1 = hierarchy.get("h1") or hierarchy.get("section")
    if h1:
        return h1
    return ""
