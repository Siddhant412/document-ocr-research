#!/usr/bin/env python3
"""Create a deterministic, source-first OmniDocBench smoke-test manifest.

The sampler fixes source quotas first, then seeks the requested language and
layout *marginals*.  It never claims coverage of every source/language/layout
joint cell: nonempty cells absent from the sample are written to coverage.json.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True, help="Official OmniDocBench JSON")
    parser.add_argument("--output", type=Path, required=True, help="Manifest JSONL to create")
    parser.add_argument("--coverage-output", type=Path, required=True)
    parser.add_argument("--pages", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260906)
    parser.add_argument("--minimum-per-source", type=int, default=5)
    parser.add_argument(
        "--language",
        help="Restrict selection to one exact OmniDocBench page language label (for example: english).",
    )
    parser.add_argument(
        "--allow-margin-residuals",
        action="store_true",
        help="Write a manifest even if exact language/layout marginal targets are infeasible.",
    )
    return parser.parse_args()


def label(sample: dict[str, Any], key: str) -> str:
    attributes = sample.get("page_info", {}).get("page_attribute", {})
    if not isinstance(attributes, dict):
        attributes = {}
    value = attributes.get(key, sample.get(key))
    if value is None or value == "":
        raise ValueError(f"page has no {key!r} label: {sample.get('page_info', {}).get('image_path')}")
    return str(value)


def image_name(sample: dict[str, Any]) -> str:
    value = sample.get("page_info", {}).get("image_path")
    if not value:
        raise ValueError("page has no page_info.image_path")
    return Path(str(value)).name


def priority(seed: int, page_id: str) -> str:
    return hashlib.sha256(f"{seed}:{page_id}".encode("utf-8")).hexdigest()


def quotas(counts: collections.Counter[str], total: int, minimum: int) -> dict[str, int]:
    """Largest-remainder allocation with a capped minimum per nonempty group."""
    if total > sum(counts.values()):
        raise ValueError(f"Requested {total} pages but dataset contains only {sum(counts.values())}")
    groups = sorted(counts)
    allocation = {group: min(minimum, counts[group]) for group in groups}
    # Reduce minima deterministically if callers ask for fewer pages than minima require.
    while sum(allocation.values()) > total:
        group = max(groups, key=lambda item: (allocation[item], item))
        allocation[group] -= 1

    ideal = {group: total * counts[group] / sum(counts.values()) for group in groups}
    while sum(allocation.values()) < total:
        eligible = [group for group in groups if allocation[group] < counts[group]]
        if not eligible:
            raise RuntimeError("No eligible pages remain while allocating quotas")
        group = max(eligible, key=lambda item: (ideal[item] - allocation[item], counts[item], item))
        allocation[group] += 1
    return allocation


def objective(
    selected: Iterable[dict[str, Any]], language_targets: dict[str, int], layout_targets: dict[str, int]
) -> int:
    selected = list(selected)
    languages = collections.Counter(page["language"] for page in selected)
    layouts = collections.Counter(page["layout"] for page in selected)
    return sum(abs(languages[key] - target) for key, target in language_targets.items()) + sum(
        abs(layouts[key] - target) for key, target in layout_targets.items()
    )


def select_pages(
    pages: list[dict[str, Any]], source_targets: dict[str, int], language_targets: dict[str, int],
    layout_targets: dict[str, int], seed: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for page in pages:
        by_source[page["data_source"]].append(page)
    for group in by_source.values():
        group.sort(key=lambda page: priority(seed, page["page_id"]))

    remaining = dict(source_targets)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    language_counts: collections.Counter[str] = collections.Counter()
    layout_counts: collections.Counter[str] = collections.Counter()
    joint_counts: collections.Counter[tuple[str, str, str]] = collections.Counter()

    while sum(remaining.values()):
        options: list[tuple[tuple[int, int, int, int, str], dict[str, Any]]] = []
        for source, need in remaining.items():
            if need <= 0:
                continue
            for page in by_source[source]:
                if page["page_id"] in selected_ids:
                    continue
                language_need = max(language_targets[page["language"]] - language_counts[page["language"]], 0)
                layout_need = max(layout_targets[page["layout"]] - layout_counts[page["layout"]], 0)
                new_joint = int(joint_counts[(source, page["language"], page["layout"])] == 0)
                # Source counts are hard constraints; the remaining terms make the
                # deterministic choice satisfy language/layout marginals where possible.
                score = (
                    language_need + layout_need,
                    language_need,
                    layout_need,
                    new_joint,
                    -int(priority(seed, page["page_id"]), 16),
                )
                options.append((score, page))
        if not options:
            raise RuntimeError("Sampler exhausted candidates before satisfying source quotas")
        _, chosen = max(options, key=lambda item: item[0])
        selected.append(chosen)
        selected_ids.add(chosen["page_id"])
        remaining[chosen["data_source"]] -= 1
        language_counts[chosen["language"]] += 1
        layout_counts[chosen["layout"]] += 1
        joint_counts[(chosen["data_source"], chosen["language"], chosen["layout"])] += 1

    # Deterministic same-source swaps preserve the primary source quota and reduce
    # marginal error.  It never injects hand-picked pages.
    current = objective(selected, language_targets, layout_targets)
    for _ in range(len(pages) * 2):
        best: tuple[int, int, dict[str, Any], dict[str, Any]] | None = None
        selected_by_source: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for page in selected:
            selected_by_source[page["data_source"]].append(page)
        for source, chosen_pages in selected_by_source.items():
            unchosen = [page for page in by_source[source] if page["page_id"] not in selected_ids]
            for old in chosen_pages:
                for new in unchosen:
                    trial = [page for page in selected if page["page_id"] != old["page_id"]] + [new]
                    candidate = objective(trial, language_targets, layout_targets)
                    if candidate < current:
                        tie = int(priority(seed, new["page_id"]), 16)
                        proposal = (candidate, tie, old, new)
                        if best is None or proposal[:2] < best[:2]:
                            best = proposal
        if best is None:
            break
        _, _, old, new = best
        selected = [page for page in selected if page["page_id"] != old["page_id"]]
        selected.append(new)
        selected_ids.remove(old["page_id"])
        selected_ids.add(new["page_id"])
        current = objective(selected, language_targets, layout_targets)

    return sorted(selected, key=lambda page: (page["data_source"], priority(seed, page["page_id"])))


def select_pages_milp(
    pages: list[dict[str, Any]],
    source_targets: dict[str, int],
    language_targets: dict[str, int],
    layout_targets: dict[str, int],
    seed: int,
) -> tuple[list[dict[str, Any]], bool]:
    """Select a globally feasible sample, or the closest documented relaxation.

    Source and layout targets remain hard constraints in the fallback. Language
    absolute error is minimized only after confirming that all three target
    families cannot be satisfied together. A tiny seeded secondary objective
    makes otherwise equivalent solutions reproducible.
    """
    try:
        import numpy as np
        from scipy.optimize import Bounds, LinearConstraint, milp
    except ImportError as error:  # pragma: no cover - environment guard
        raise RuntimeError("Install scipy>=1.11 to build a production smoke manifest") from error

    sources = sorted(source_targets)
    languages = sorted(language_targets)
    layouts = sorted(layout_targets)
    page_count = len(pages)
    ranks = {page["page_id"]: rank + 1 for rank, page in enumerate(sorted(pages, key=lambda p: priority(seed, p["page_id"]))) }
    secondary_cost = np.array([ranks[page["page_id"]] for page in pages], dtype=float) / (page_count * page_count * 1000)

    def hard_rows(include_language: bool, variable_count: int) -> tuple[list[list[float]], list[float]]:
        rows: list[list[float]] = []
        values: list[float] = []
        for key, targets, field in (
            (sources, source_targets, "data_source"),
            (layouts, layout_targets, "layout"),
        ):
            for value in key:
                rows.append([float(page[field] == value) for page in pages] + [0.0] * (variable_count - page_count))
                values.append(float(targets[value]))
        if include_language:
            for value in languages:
                rows.append([float(page["language"] == value) for page in pages] + [0.0] * (variable_count - page_count))
                values.append(float(language_targets[value]))
        return rows, values

    rows, values = hard_rows(include_language=True, variable_count=page_count)
    exact = milp(
        c=secondary_cost,
        integrality=np.ones(page_count),
        bounds=Bounds(np.zeros(page_count), np.ones(page_count)),
        constraints=LinearConstraint(np.array(rows), np.array(values), np.array(values)),
    )
    if exact.success:
        selected = [page for page, value in zip(pages, np.rint(exact.x).astype(int)) if value]
        return sorted(selected, key=lambda page: (page["data_source"], priority(seed, page["page_id"]))), True

    # Exact marginals are impossible. Preserve source/layout quotas, cover each
    # represented language at least once, and minimize language residuals.
    variable_count = page_count + 2 * len(languages)
    rows, values = hard_rows(include_language=False, variable_count=variable_count)
    lower = list(values)
    upper = list(values)
    for language_index, language in enumerate(languages):
        page_row = [float(page["language"] == language) for page in pages]
        plus = [-float(index == language_index) for index in range(len(languages))]
        minus = [float(index == language_index) for index in range(len(languages))]
        rows.append(page_row + plus + minus)
        lower.append(float(language_targets[language]))
        upper.append(float(language_targets[language]))
        # All target languages must still be represented; this prevents a rare
        # label from disappearing merely because exact proportions are impossible.
        rows.append(page_row + [0.0] * (2 * len(languages)))
        lower.append(1.0)
        upper.append(np.inf)
    relaxed = milp(
        c=np.concatenate((secondary_cost, np.ones(2 * len(languages)))),
        integrality=np.ones(variable_count),
        bounds=Bounds(np.zeros(variable_count), np.concatenate((np.ones(page_count), np.full(2 * len(languages), page_count)))),
        constraints=LinearConstraint(np.array(rows), np.array(lower), np.array(upper)),
    )
    if not relaxed.success:
        raise RuntimeError(f"Unable to solve smoke selection: {relaxed.message}")
    selected = [page for page, value in zip(pages, np.rint(relaxed.x[:page_count]).astype(int)) if value]
    return sorted(selected, key=lambda page: (page["data_source"], priority(seed, page["page_id"]))), False


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    options = args()
    raw = json.loads(options.annotations.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("OmniDocBench annotations must be a JSON list")

    pages: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for original_index, sample in enumerate(raw):
        if options.language is not None and label(sample, "language") != options.language:
            continue
        page_id = image_name(sample)
        if page_id in seen_names:
            raise ValueError(f"Duplicate image basename is not supported: {page_id}")
        seen_names.add(page_id)
        pages.append(
            {
                "source_index": original_index,
                "page_id": page_id,
                "image_path": sample["page_info"]["image_path"],
                "prediction_filename": f"{Path(page_id).stem}.md",
                "data_source": label(sample, "data_source"),
                "language": label(sample, "language"),
                "layout": label(sample, "layout"),
                "page_attributes": sample.get("page_info", {}).get("page_attribute", {}),
            }
        )

    source_counts = collections.Counter(page["data_source"] for page in pages)
    language_counts = collections.Counter(page["language"] for page in pages)
    layout_counts = collections.Counter(page["layout"] for page in pages)
    source_targets = quotas(source_counts, options.pages, options.minimum_per_source)
    language_targets = quotas(language_counts, options.pages, 1)
    layout_targets = quotas(layout_counts, options.pages, 1)
    selected, exact_margins_feasible = select_pages_milp(
        pages, source_targets, language_targets, layout_targets, options.seed
    )

    actual_languages = collections.Counter(page["language"] for page in selected)
    actual_layouts = collections.Counter(page["layout"] for page in selected)
    residuals = {
        "language": {key: actual_languages[key] - target for key, target in language_targets.items() if actual_languages[key] != target},
        "layout": {key: actual_layouts[key] - target for key, target in layout_targets.items() if actual_layouts[key] != target},
    }
    all_joints = collections.Counter((page["data_source"], page["language"], page["layout"]) for page in pages)
    selected_joints = collections.Counter((page["data_source"], page["language"], page["layout"]) for page in selected)
    coverage = {
        "selection": {
            "pages": options.pages,
            "seed": options.seed,
            "minimum_per_source": options.minimum_per_source,
            "language_filter": options.language,
            "solver": "scipy.optimize.milp (HiGHS)",
            "exact_source_language_layout_margins_feasible": exact_margins_feasible,
        },
        "targets": {"data_source": source_targets, "language": language_targets, "layout": layout_targets},
        "actual": {
            "data_source": dict(collections.Counter(page["data_source"] for page in selected)),
            "language": dict(actual_languages),
            "layout": dict(actual_layouts),
        },
        "marginal_residuals": residuals,
        "uncovered_nonempty_source_language_layout": [
            {"data_source": source, "language": language, "layout": layout, "dataset_pages": count}
            for (source, language, layout), count in sorted(all_joints.items())
            if selected_joints[(source, language, layout)] == 0
        ],
    }
    write_json(options.coverage_output, coverage)
    if (residuals["language"] or residuals["layout"]) and not options.allow_margin_residuals:
        print(
            "Exact language/layout marginal targets were infeasible; coverage report written. "
            "Inspect it or pass --allow-margin-residuals explicitly.",
            file=sys.stderr,
        )
        return 2

    options.output.parent.mkdir(parents=True, exist_ok=True)
    with options.output.open("x", encoding="utf-8") as handle:
        for sequence, page in enumerate(selected):
            page["sequence"] = sequence
            handle.write(json.dumps(page, ensure_ascii=False, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
