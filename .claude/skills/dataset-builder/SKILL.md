---
name: dataset-builder
description: Use when changing or adding a dataset builder (src/kvsel/gqa.py, textvqa.py or a new dataset), the item manifest schema, the funnel, or image fetching. Covers versioning, the funnel record, manifest fields and the item-construction rules.
---

# Dataset builders and manifests

Builders turn a raw dataset into frozen JSONL manifests in `manifests/` plus a funnel JSON that
records every filter's count.

## Manifest item fields

`item_id`, `dataset`, `builder`, `split` (`dev` or confirmatory), `image_id`, `image_file`,
`image_size`, `question`, `a_true`, `a_alt`, `descriptions` (`mis`, `match`, `irr`),
`evidence_box` (`x`, `y`, `w`, `h` in image pixels), plus dataset-specific fields (`object`,
`attribute_kind`, `conflict_type`, ...). Downstream code (runner, verify, geometry) relies on these.

## Rules

- **Bump the builder version** (`gqa-2` → `gqa-3`) for any change that can alter which items are
  produced or their content. The version is stored on every item and in the funnel.
- The funnel JSON lists each filter stage in order with its count, the source file, the builder
  version, and the sha256 of each output manifest. Keep it complete; the plan quotes it.
- Splits are deterministic (fixed seed). Never reshuffle dev vs confirmatory after outcomes have been
  seen (see `research-integrity`).
- Item-construction rules already decided (plan §1):
  - Closed vocabulary: values from 11 clear colours and 8 materials for GQA.
  - TextVQA: a_alt and irrelevant strings can't contain, or be contained in, an accepted answer or a_alt.
  - Matching and misleading descriptions differ only in the value; verb agreement follows the
    dataset's question or answer ("are" for plural objects); irrelevant descriptions use singular objects.
  - Description lengths within ±3 tokens of each other.
- New rules that refine the proposal must be written into the plan (they go into `PREREG.md` later).

## After a rebuild

1. Run the builder tests (`tests/test_gqa.py`, `tests/test_textvqa.py`) on small synthetic inputs.
2. The raw question files live on the user's side; hand them the `kvsel build-...` command and ask for
   the printed funnel (`gpu-handoff` style, though no GPU is needed).
3. Any new images need `kvsel fetch-images <manifest> ...`, which downloads only referenced files
   and checks CRCs.
4. Update the funnel numbers quoted in the implementation plan.
