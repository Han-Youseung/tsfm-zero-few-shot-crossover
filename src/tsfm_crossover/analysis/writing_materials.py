"""Export a writing kit from verified existing final tests; never run a model."""

import argparse
import csv
import html
import json
import shutil
import statistics as st
import subprocess
from collections import Counter
from pathlib import Path

from tsfm_crossover.analysis.paper import (
    METRICS,
    analyze,
    condition_key,
    digest,
    load_scope,
    load_verified,
    read,
)
from tsfm_crossover.analysis.paper_report import LABELS, bounds, plot, table
from tsfm_crossover.experiments.pilot_review import require


def interval_label(value):
    return bounds(value) if value is not None else "관측 범위 내 지속 개선 기준 미충족"


def json_file(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def csv_file(path, rows):
    require(bool(rows), "empty table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def flatten(result, records, prepared):
    """Pure transformations; retain numeric precision and null for inapplicable fields."""
    curves, endpoints, channels, budget = [], [], [], []
    for g in result["groups"]:
        key = {"model": g["family"], "dataset": g["dataset"]}
        for metric in METRICS:
            p = g["metrics"][metric]
            for c in p["curve"]:
                row = key | {"metric": metric} | {k: v for k, v in c.items() if k != "paired"}
                for pair in c["paired"]:
                    for field in ("zero", "few", "relative_improvement"):
                        row[f"{field}_{pair['seed']}"] = pair[field]
                curves.append(row)
            endpoints.append(
                key
                | {"metric": metric}
                | {k: v for k, v in p["curve"][-1].items() if k != "paired"}
                | {
                    "mean_sustained_interval": interval_label(p["mean"]["sustained_improvement"]),
                    "all_seed_sustained_interval": interval_label(
                        p["all_seeds"]["sustained_improvement"]
                    ),
                }
            )
        for ch in g["channels"]:
            channels.append(
                key
                | {"channel": ch["channel"]}
                | {k: v for k, v in ch["endpoint"].items() if k != "paired"}
                | {
                    "mean_sustained_interval": interval_label(ch["mean"]["sustained_improvement"]),
                    "all_seed_sustained_interval": interval_label(
                        ch["all_seeds"]["sustained_improvement"]
                    ),
                }
            )
        b = g["budget"]
        budget.append(
            key
            | {
                "fine_tuned_conditions": b["fine_tuned_conditions"],
                "reached_max_steps": b["reached_max_steps"],
                "best_step_at_budget_cap": b["best_step_at_budget_cap"],
                "actual_steps_min": b["actual_steps"]["min"],
                "actual_steps_max": b["actual_steps"]["max"],
                "full_pool_visited_fraction_min": b["full_pool_unique_fraction"]["min"],
                "full_pool_visited_fraction_max": b["full_pool_unique_fraction"]["max"],
                "convergence_established": False,
            }
        )
    conditions, history, environments = [], [], {}
    for r in sorted(records, key=condition_key):
        c, t, p = r["identity"]["condition"], r["training"], r["provenance"]
        conditions.append(
            {
                "condition_id": c["id"],
                "model": c["family"],
                "dataset": c["dataset"],
                "horizon": c["horizon"],
                "seed": c["seed"],
                "requested_rate": c["rate"],
                **r["test"]["metrics"]["macro"],
                "train_pool_total": p["total_train_windows"],
                "selected_pool": p["selected_train_windows"],
                "actual_optimizer_steps": t["actual_optimizer_steps"],
                "best_validation_step": t.get("best_validation_step"),
                "stopping_step": t.get("stopping_step"),
                "actual_unique_windows": t.get("actual_unique_train_windows"),
                "equivalent_epochs": t.get("equivalent_epochs"),
                "average_window_exposures": t.get("average_window_exposures"),
                "stop_reason": t.get("stop_reason"),
                "train_seconds": t.get("train_seconds"),
                "validation_inference_seconds": t.get("inference_seconds"),
                "test_inference_seconds": r["test"]["timing"]["inference_seconds"],
                "train_peak_allocated_mb": t.get("peak_allocated_mb"),
                "train_peak_reserved_mb": t.get("peak_reserved_mb"),
                "test_windows": r["test"]["number_of_test_windows"],
                "gpu": r["runtime"]["gpu"],
                "runtime_hash": r["runtime_hash"],
                "trainable_parameters": r["trainable_parameters"],
                "total_parameters": r["total_parameters"],
                "execution_commit": r["identity"]["commit"],
                "data_sha256": r["identity"]["data_sha256"],
                "test_window_hash": r["test"]["window_hash"],
                "sampling_manifest_hash": p["sampling_manifest_hash"],
            }
        )
        for point in t.get("validation_history", []):
            history.append({"condition_id": c["id"], "step": point["step"], **point["macro"]})
        rh = r["runtime_hash"]
        require(rh not in environments or environments[rh] == r["runtime"], "runtime hash conflict")
        environments[rh] = r["runtime"]
    datasets = []
    for name in result["scope"]["datasets"]:
        q = prepared[name]["qc"]
        g = next(g for g in result["groups"] if g["dataset"] == name)
        d = g["train_descriptor"]
        require(q["sha256"] == d["dataset_sha256"], "prepared/descriptor fingerprint")
        datasets.append(
            {
                "dataset": name,
                "source_group": g["source_group"],
                "rows": q["rows"],
                "channels": q["channels"],
                "frequency_minutes": q["frequency_seconds"] / 60,
                "context_hours": 512 * q["frequency_seconds"] / 3600,
                "horizon_hours": 96 * q["frequency_seconds"] / 3600,
                **{
                    f"{s}_rows": q["split"][s]["end"] - q["split"][s]["start"]
                    for s in ("train", "validation", "test")
                },
                "train_candidate_windows": q["split"]["train"]["end"] - 512 - 96 + 1,
                "validation_windows_used": 64,
                "test_windows": g["test_windows"],
                "missing_cells": q["missing_values"],
                "duplicate_timestamps": q["duplicate_timestamps"],
                "train_lag1_channel_mean": d["channel_macro_lag1_correlation"],
                "train_abs_trend_in_std_channel_mean": d[
                    "channel_macro_absolute_linear_trend_in_train_std"
                ],
                "data_sha256": q["sha256"],
                "channels_in_order": " | ".join(q["channel_names"]),
                "source_urls": " | ".join(s["url"] for s in prepared[name]["source_files"]),
                "license_recorded": prepared[name]["license"],
            }
        )
    model_settings = []
    for family in result["scope"]["families"]:
        row = next(r for r in records if r["identity"]["condition"]["family"] == family)
        q = row["adapter_settings"]
        model_settings.append(
            {"model": family}
            | row["identity"]["model_pin"]
            | {
                k: q[k]
                for k in (
                    "context",
                    "horizon",
                    "dtype",
                    "learning_rate",
                    "weight_decay",
                    "max_optimizer_steps",
                    "num_samples",
                    "point_statistic",
                    "gradient_clip",
                    "external_scaler",
                )
            }
            | {
                "total_parameters": row["total_parameters"],
                "trainable_parameters": row["trainable_parameters"],
            }
        )
    return {
        "datasets": datasets,
        "endpoints": endpoints,
        "curves": curves,
        "conditions": conditions,
        "channels": channels,
        "budget": budget,
        "validation_history": history,
        "model_settings": model_settings,
    }, environments


def results_paragraphs(result):
    paragraphs = []
    for g in result["groups"]:
        p = g["metrics"]["normalized_mae"]
        e = p["curve"][-1]
        paragraphs.append(
            f"{LABELS[g['family']]} / {g['dataset']}: Zero-Shot normalized MAE "
            f"{e['zero_mean']:.5f}, 100% pool {e['few_mean']:.5f}, "
            f"평균 상대 개선 {e['mean_relative_improvement'] * 100:.2f}% "
            f"(seed 최소–최대 {e['seed_min_relative_improvement'] * 100:.2f}–"
            f"{e['seed_max_relative_improvement'] * 100:.2f}%). "
            f"평균 지속 개선 구간 {interval_label(p['mean']['sustained_improvement'])}; "
            "세 seed 모두 지속 개선 구간 "
            f"{interval_label(p['all_seeds']['sustained_improvement'])}."
        )
    return "\n\n".join(paragraphs)


STYLE = """
body{font-family:'Malgun Gothic',Arial,sans-serif;color:#203040;margin:0;background:#f6f8fb}
main{max-width:1120px;margin:30px auto;background:white;padding:35px;box-sizing:border-box}
h1{font-size:29px}h2{margin-top:36px;color:#173c64}a{color:#075ba8}
p{line-height:1.85}pre{white-space:pre-wrap;font:15px/1.9 'Malgun Gothic',sans-serif}
table{border-collapse:collapse;width:100%;font-size:13px}
td,th{padding:10px;border-bottom:1px solid #ddd}
th{background:#173c64;color:white;text-align:left}.scroll{overflow:auto}.note{background:#edf4fa;padding:20px}
img{max-width:100%;height:auto}.figures{display:grid;grid-template-columns:1fr 1fr;gap:10px}
@media(max-width:650px){main{margin:0;padding:18px}.figures{grid-template-columns:1fr}}
@media print{body{background:white}main{margin:0;padding:0}a{color:inherit}pre{font-size:10pt}}
"""


def page(title, body):
    return (
        '<!doctype html><html lang="ko"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{STYLE}</style><main>{body}</main></html>"
    )


def build(root, archives, output):
    require(not output.exists(), "use a new output directory; do not overwrite evidence")
    scope = load_scope(root / "configs/analysis/primary5_h96.yaml")
    records, archive_hashes = load_verified(scope, root, archives)
    ref = scope["descriptors"]
    require(digest(root / ref["path"]) == ref["sha256"], "descriptor bytes changed")
    descriptors = read(root / ref["path"])
    require(
        descriptors["descriptor_source_sha256"] == digest(root / descriptors["descriptor_source"]),
        "descriptor source changed",
    )
    result = read(root / "results/summaries/primary5_h96_two_model_analysis.json")
    require(
        analyze(records, scope, descriptors)
        == {k: v for k, v in result.items() if k != "evidence"},
        "analysis no longer matches original results",
    )
    tables, environments = flatten(
        result, records, read(root / "results/manifests/pilot/prepared_primary9.json")
    )
    output.mkdir(parents=True)
    for name, rows in tables.items():
        csv_file(output / "tables" / f"{name}.csv", rows)
    json_file(output / "tables/workbook_inputs.json", tables)
    json_file(output / "evidence/final_results.json", records)
    json_file(output / "evidence/environments.json", environments)
    json_file(output / "evidence/source_archive_hashes.json", archive_hashes)
    for name in (
        "primary5_h96_ttm_review.json",
        "primary5_h96_moirai1_review.json",
        "primary5_h96_train_descriptors.json",
        "primary5_h96_analysis_validation.json",
    ):
        shutil.copy2(root / "results/manifests/study" / name, output / "evidence" / name)
    shutil.copy2(
        root / "results/summaries/primary5_h96_two_model_analysis.json",
        output / "evidence/analysis.json",
    )
    shutil.copy2(
        root / "configs/analysis/primary5_h96.yaml", output / "evidence/analysis_scope.yaml"
    )
    # Historical original plan, not a declaration that all nine datasets/horizons are complete.
    shutil.copy2(root / "configs/study/main.yaml", output / "evidence/original_execution_plan.yaml")
    templates = root / "docs/paper-writing"
    for source in templates.iterdir():
        if source.suffix in {".txt", ".bib", ".json"} and "template" not in source.name:
            shutil.copy2(source, output / source.name)
    refs = read(templates / "references.json")
    bibliography = "\n\n".join(
        f"[{r['number']}] {r['authors']}. {r['title']}. {r['venue']}. "
        + (str(r["year"]) if r["year"] else "Publication year not specified")
        + f".\n{r['url']}\n적용 범위: {r['note']}"
        for r in refs
    )
    (output / "references.txt").write_text(bibliography + "\n", encoding="utf-8")
    draft = (templates / "draft-ko.template.txt").read_text(encoding="utf-8")
    draft = draft.replace("{{RESULTS}}", results_paragraphs(result))
    require("{{" not in draft, "unresolved draft template")
    (output / "paper_draft_ko.txt").write_text(draft, encoding="utf-8")
    (output / "paper_draft_ko.html").write_text(
        page("논문 본문 초안", f"<h1>논문 본문 초안</h1><pre>{html.escape(draft)}</pre>"),
        encoding="utf-8",
    )
    (output / "figures").mkdir()
    for d in scope["datasets"]:
        (output / "figures" / f"relative-{d}.svg").write_text(
            plot(d, result["groups"]), encoding="utf-8"
        )
    main = [e for e in tables["endpoints"] if e["metric"] == "normalized_mae"]
    grid = table(
        ["모델", "데이터", "ZS NMAE", "100% NMAE", "평균 상대 개선", "평균 지속", "3 seed 지속"],
        [
            [
                LABELS[e["model"]],
                e["dataset"],
                f"{e['zero_mean']:.5f}",
                f"{e['few_mean']:.5f}",
                f"{e['mean_relative_improvement'] * 100:.2f}%",
                e["mean_sustained_interval"],
                e["all_seed_sustained_interval"],
            ]
            for e in main
        ],
    )
    links = [
        ("paper_draft_ko.html", "1. 한국어 본문·국/영문 초록 초안"),
        ("paper_draft_ko.txt", "동일 초안 TXT — 한글/Word 편집용"),
        ("paper_tables.xlsx", "2. 계산 대조된 Excel 수치표"),
        ("methods_and_reproducibility.txt", "3. 수식·정확한 설정·재현 및 지표 해설"),
        ("claims_and_checklist.txt", "4. 주장-근거 대응표·한계·제출 전 체크리스트"),
        ("figure_captions.txt", "5. 그림·표 캡션과 본문 배치 안내"),
        ("references.bib", "6. 참고문헌 BibTeX"),
        ("references.txt", "참고문헌 일반 텍스트 — 한글/Word 편집용"),
        ("references.json", "참고문헌 원문 링크·적용 범위"),
        ("evidence/analysis.json", "7. 모든 부가 분석 JSON"),
        ("evidence/final_results.json", "270개 검증된 원 실행 결과 JSON"),
        ("tables/validation_history.csv", "기존 validation 이력 — 새 실험 아님"),
        ("verification.json", "무결성 및 범위 검증"),
    ]
    body = "<h1>TSFM 논문 작성 자료집</h1>"
    body += (
        '<p class="note">TTM-R3 · MOIRAI 1.1-R-small / 5개 데이터 변형 / H96 / 3 seeds / 270조건. '
    )
    body += "기존 결과를 CPU에서 재검증·정리했습니다. 신규 학습·test 재평가·설정 변경은 없습니다. "
    body += "저자·소속·학회 양식은 제출자가 추가해야 합니다.</p>"
    body += (
        "<h2>읽는 순서</h2><ul>"
        + "".join(f'<li><a href="{u}">{t}</a></li>' for u, t in links)
        + "</ul>"
    )
    body += "<h2>본문용 핵심 결과</h2>" + grid
    body += "<p>100%는 선택 가능한 train-window pool 크기이며 완전 순회·수렴을 뜻하지 않습니다. "
    body += "지속 개선 구간은 격자상의 기술적 요약이지 연속 임계점이나 신뢰구간이 아닙니다.</p>"
    body += '<h2>모델 내부 상대 개선 곡선</h2><div class="figures">'
    body += (
        "".join(
            f'<img src="figures/relative-{d}.svg" alt="{d} 상대 개선 곡선">'
            for d in scope["datasets"]
        )
        + "</div>"
    )
    body += (
        "<p>음영: 3 seed 최소–최대(신뢰구간 아님). x축: log 비율. 패널마다 y축 범위가 다릅니다. "
    )
    body += "모델 간 절대 오차는 위 표와 absolute-nmae.svg를 함께 보십시오.</p>"
    (output / "index.html").write_text(page("TSFM 논문 작성 자료집", body), encoding="utf-8")
    counts = Counter((r["identity"]["condition"]["family"], r["runtime"]["gpu"]) for r in records)
    json_file(
        output / "verification.json",
        {
            "status": "verified_existing_final_tests_only",
            "conditions": len(records),
            "source_archives": len(archive_hashes),
            "table_rows": {k: len(v) for k, v in tables.items()},
            "gpu_condition_counts": [
                {"model": f, "gpu": gpu, "conditions": n} for (f, gpu), n in sorted(counts.items())
            ],
            "execution_commit": scope["execution_commit"],
            "analysis_baseline_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "new_training": False,
            "test_re_evaluation": False,
            "weights_downloaded": False,
            "analysis_matches_verified_records": True,
            "scope_after_results_available": True,
            "mean_relative_improvement_is_mean_of_paired_ratios": True,
            "figure_bands_are_confidence_intervals": False,
            "runtime_comparisons_are_controlled_hardware_benchmarks": False,
            "full_pool_unique_fraction_range": [
                min(b["full_pool_visited_fraction_min"] for b in tables["budget"]),
                max(b["full_pool_visited_fraction_max"] for b in tables["budget"]),
            ],
            "mean_final_nmae": {
                f: st.fmean(e["few_mean"] for e in main if e["model"] == f)
                for f in scope["families"]
            },
            "mean_final_nmae_note": (
                "unweighted descriptive average; not an independent-dataset significance test"
            ),
        },
    )
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, action="append", default=[])
    parser.add_argument("--archive", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    archives = args.archive + [p for d in args.archive_dir for p in sorted(d.glob("*.zip"))]
    tables = build(Path.cwd(), archives, args.output_dir)
    print(json.dumps({k: len(v) for k, v in tables.items()}))


if __name__ == "__main__":
    main()
