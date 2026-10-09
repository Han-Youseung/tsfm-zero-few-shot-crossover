"""Small static, accessible publication tables and vector plots from verified JSON."""

import html
import math
import statistics as st
from pathlib import Path

LABELS = {"ttm": "TTM-R3", "moirai1": "MOIRAI 1.1-R-small"}


def bounds(value):
    if value is None:
        return "미관측"
    return f"({value['lower_exclusive'] * 100:g}%, {value['upper_inclusive'] * 100:g}%]"


def pct(value):
    return f"{100 * value:.2f}%"


def table(headers, rows):
    head = "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in rows
    )
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def plot(dataset, groups):
    selected = [g for g in groups if g["dataset"] == dataset]
    curves = [g["metrics"]["normalized_mae"]["curve"] for g in selected]
    values = [0.0] + [
        100 * c[k]
        for curve in curves
        for c in curve
        for k in ("seed_min_relative_improvement", "seed_max_relative_improvement")
    ]
    lo, hi = min(values), max(values)
    margin = max((hi - lo) * 0.07, 0.1)
    lo, hi = lo - margin, hi + margin

    def x(rate):
        return 62 + 430 * math.log(rate / 0.005) / math.log(200)

    def y(value):
        return 215 - (value - lo) / (hi - lo) * 165

    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 530 275" ',
        f'role="img" aria-label="{dataset} paired seed relative improvement">',
        '<rect width="530" height="275" fill="white"/>',
        '<g font-family="Arial,sans-serif" font-size="11" fill="#243743">',
        f'<text x="62" y="24" font-size="17" font-weight="bold">{dataset}</text>',
        f'<line x1="62" x2="492" y1="{y(0):.2f}" y2="{y(0):.2f}" '
        'stroke="#888" stroke-dasharray="4 4"/>',
    ]
    for value in (lo, (lo + hi) / 2, hi):
        parts.append(f'<text x="53" y="{y(value) + 4:.2f}" text-anchor="end">{value:.1f}%</text>')
    for i, (g, color) in enumerate(zip(selected, ("#2166ac", "#b34d20"), strict=True)):
        curve = g["metrics"]["normalized_mae"]["curve"]
        band = [(x(c["rate"]), y(100 * c["seed_min_relative_improvement"])) for c in curve]
        band += [
            (x(c["rate"]), y(100 * c["seed_max_relative_improvement"])) for c in reversed(curve)
        ]
        poly = " ".join(f"{a:.2f},{b:.2f}" for a, b in band)
        line = " ".join(
            f"{x(c['rate']):.2f},{y(100 * c['mean_relative_improvement']):.2f}" for c in curve
        )
        parts.extend(
            [
                f'<polygon points="{poly}" fill="{color}" opacity=".12"/>',
                f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2"/>',
                f'<text x="{282 + i * 95}" y="24" fill="{color}">{g["family"]}</text>',
            ]
        )
    for rate in curves[0]:
        r = rate["rate"]
        parts.append(f'<text x="{x(r):.2f}" y="238" text-anchor="middle">{r * 100:g}</text>')
    parts.append(
        '<text x="272" y="263" text-anchor="middle">Train window pool (%) '
        "· log scale</text></g></svg>"
    )
    return "".join(parts)


def manuscript(result):
    groups = result["groups"]
    text = [
        "논문 결과·논의 초안: 두 모델 / 5개 데이터셋 / H96",
        "이 문서는 이미 관측한 결과를 요약한 초안이다. "
        "신규 사전등록 또는 전체 연구 완료를 뜻하지 않는다.",
        "연구 목적",
        "고정된 TTM-R3와 MOIRAI 1.1-R-small 각각에 대해, 대상 train-window pool의 크기에 따른 "
        "Zero-Shot–Few-Shot 오차의 전환 구간을 제한된 계산 예산 아래 비교하였다.",
        "방법",
        "ETTh1·ETTh2·ETTm1·ETTm2·Tetouan, context 512, horizon 96, 세 seed를 사용했다. "
        "60/20/20 시간순 분할, 중첩 시간층화 sampling, 동일 test origins, native scaling 및 "
        "최대 1000 optimizer steps/validation early stopping 규칙을 유지했다. "
        "주 지표는 train 표준편차로 오차만 정규화한 채널 macro MAE이다. "
        "각 모델·데이터셋·seed의 Zero-Shot과 같은 seed의 Few-Shot을 비교했다.",
        "결과",
    ]
    for g in groups:
        p = g["metrics"]["normalized_mae"]
        endpoint = p["curve"][-1]
        text.append(
            f"{LABELS[g['family']]} / {g['dataset']}: 100% pool에서 평균 상대 오차 감소율 "
            f"{pct(endpoint['mean_relative_improvement'])}, seed 범위 "
            f"{pct(endpoint['seed_min_relative_improvement'])}–"
            f"{pct(endpoint['seed_max_relative_improvement'])}. "
            f"평균 지속 개선 구간 {bounds(p['mean']['sustained_improvement'])}; "
            f"세 seed 모두 지속 개선 구간 {bounds(p['all_seeds']['sustained_improvement'])}."
        )
    text.extend(
        [
            "해석과 한계",
            "전환 구간은 관측된 학습 비율 격자에 대한 기술통계이다. 범위 내 연속 함수의 단조성, "
            "정확한 임계점 또는 통계적 유의성을 의미하지 않는다. 0.5%에서 이미 개선한 경우 "
            "그보다 작은 비율에서 언제 전환됐는지는 관측하지 않았다.",
            "100%는 전체 적격 train pool을 sampling 대상으로 허용했다는 뜻이다. batch 1 및 "
            "최대 1000 step 때문에 모든 후보 window를 방문하거나 전체 데이터로 수렴할 때까지 "
            "학습했다는 뜻은 아니다. 선택 pool 수, 실제 unique 방문 수, 반복 노출을 구분한다.",
            "두 ETT 변압기의 시간/분 데이터는 서로 독립된 네 출처가 아니다. "
            "5개 변형은 3개 출처군으로 묶되 출처군 간 독립성도 보장하지 않는다. "
            "H96의 실제 시간 범위도 ETTh 96시간, ETTm 24시간, Tetouan 16시간으로 다르다. "
            "특성별 차이는 사례 관찰로 해석하며 일반화 회귀, 인과 설명 또는 "
            "독립 5표본 검정을 하지 않았다.",
            "seed 제외, 다른 오차 지표, 1%/2% 상대 개선 마진 및 출처군 제외 분석은 결과 확인 후의 "
            "기술적 민감도 분석이다. 1%와 2%는 도메인에서 정당화된 유의성 기준이 아니며, "
            "사전 주 분석의 0% 규칙이나 학습 설정을 대체하지 않는다.",
            "학습 seed와 MOIRAI의 확률적 예측 seed가 함께 변하므로 관측 seed 변동은 두 효과를 "
            "분리하지 못한다. 시간 block 불확실성, 동일 epoch 재학습 및 수렴성은 검증하지 않았다.",
            "Timer는 구현·실행 provenance 검토 후 사용자가 두 원래 모델에 집중하기로 하여 본문 "
            "주 분석에서 제외했다. 원본은 보존한다. 채널 독립 처리 자체가 배제 사유는 아니다. "
            "TTM은 내부 channel-independent/shared-weight 구조, MOIRAI는 joint-target 구조이다.",
            "범위 선택은 결과 이용 가능 후 이루어졌음을 공개한다. 원래 primary9·네 horizon "
            "1944조건 계획 전체의 완료나 모든 TSFM에 대한 결론으로 확대하지 않는다.",
        ]
    )
    return "\n\n".join(text) + "\n"


def render(result, output):
    groups = result["groups"]
    main_rows, sensitivity, exposure, channel_rows = [], [], [], []
    for g in groups:
        primary = g["metrics"]["normalized_mae"]
        e = primary["curve"][-1]
        label = (LABELS[g["family"]], g["dataset"])
        main_rows.append(
            (
                *label,
                f"{e['zero_mean']:.5f}",
                f"{e['few_mean']:.5f}",
                pct(e["mean_relative_improvement"]),
                bounds(primary["mean"]["sustained_improvement"]),
                bounds(primary["all_seeds"]["sustained_improvement"]),
            )
        )
        margin = g["posthoc_relative_margins"][1]
        seed_intervals = [
            bounds(r["mean"]["sustained_improvement"]) for r in g["posthoc_leave_one_seed_out"]
        ]
        sensitivity.append(
            (
                *label,
                bounds(g["metrics"]["normalized_mse"]["all_seeds"]["sustained_improvement"]),
                bounds(margin["all_seeds_relative"]["sustained_improvement"]),
                " / ".join(seed_intervals),
            )
        )
        b = g["budget"]
        u = b["full_pool_unique_fraction"]
        exposure.append(
            (
                *label,
                b["reached_max_steps"],
                b["best_step_at_budget_cap"],
                b["full_pool_conditions"][0]["selected_pool"],
                f"{pct(u['min'])}–{pct(u['max'])}",
            )
        )
        harmed = [c["channel"] for c in g["channels"] if c["endpoint"]["mean_delta"] < 0]
        all_improve = sum(c["endpoint"]["improved_seed_count"] == 3 for c in g["channels"])
        channel_rows.append(
            (*label, len(g["channels"]), all_improve, ", ".join(harmed) if harmed else "없음")
        )
    feature_rows = []
    for d in result["scope"]["datasets"]:
        g = next(g for g in groups if g["dataset"] == d)
        v = g["train_descriptor"]
        feature_rows.append(
            (
                d,
                g["source_group"],
                v["channels"],
                v["train_rows"],
                v["frequency_seconds"] / 60,
                96 * v["frequency_seconds"] / 3600,
                f"{v['channel_macro_lag1_correlation']:.4f}",
                f"{v['channel_macro_absolute_linear_trend_in_train_std']:.4f}",
            )
        )
    aggregate_rows = []
    for row in result["source_aggregates"]:
        if row["rate"] == 1.0:
            vals = row["leave_one_source_out_means"].values()
            aggregate_rows.append(
                (
                    LABELS[row["family"]],
                    pct(row["dataset_equal_weight_mean"]),
                    pct(row["source_equal_weight_mean"]),
                    f"{pct(min(vals))}–{pct(max(vals))}",
                )
            )
    figures = []
    for dataset in result["scope"]["datasets"]:
        svg = plot(dataset, groups)
        (output / f"figure-{dataset}.svg").write_text(svg, encoding="utf-8")
        figures.append(svg)
    sections = {
        "MAIN_TABLE": table(
            [
                "모델",
                "데이터",
                "ZS NMAE",
                "100% FT NMAE",
                "평균 감소율",
                "평균 지속 구간",
                "3 seed 모두 지속 구간",
            ],
            main_rows,
        ),
        "SENSITIVITY_TABLE": table(
            [
                "모델",
                "데이터",
                "NMSE 3 seed 지속",
                "NMAE >1% 3 seed 지속",
                "seed 한 개 제외 후 평균 지속 구간",
            ],
            sensitivity,
        ),
        "EXPOSURE_TABLE": table(
            [
                "모델",
                "데이터",
                "1000 step 도달 /24",
                "best=1000 /24",
                "100% 선택 pool",
                "100% pool 실제 unique 방문 비율",
            ],
            exposure,
        ),
        "CHANNEL_TABLE": table(
            [
                "모델",
                "데이터",
                "채널 수",
                "100%에서 3 seed 모두 개선 채널",
                "100%에서 평균 악화 채널",
            ],
            channel_rows,
        ),
        "FEATURE_TABLE": table(
            [
                "데이터",
                "출처군",
                "채널",
                "train 행",
                "간격(분)",
                "H96(시간)",
                "lag1 상관",
                "정규화 절대 추세",
            ],
            feature_rows,
        ),
        "AGGREGATE_TABLE": table(
            ["모델", "5 dataset 동일 가중", "3 출처군 동일 가중", "출처군 하나 제외 범위"],
            aggregate_rows,
        ),
        "FIGURES": "".join(figures),
    }
    template = Path(__file__).with_name("paper_template.html").read_text(encoding="utf-8")
    for name, value in sections.items():
        template = template.replace("{{" + name + "}}", value)
    (output / "report.html").write_text(template, encoding="utf-8")
    (output / "manuscript_sections.txt").write_text(manuscript(result), encoding="utf-8")
    brief = []
    for family in result["scope"]["families"]:
        selected = [g for g in groups if g["family"] == family]
        brief.append(
            {
                "family": family,
                "all_seed_sustained_datasets": sum(
                    g["metrics"]["normalized_mae"]["all_seeds"]["sustained_improvement"] is not None
                    for g in selected
                ),
                "mean_endpoint_reduction_across_datasets": st.fmean(
                    g["metrics"]["normalized_mae"]["curve"][-1]["mean_relative_improvement"]
                    for g in selected
                ),
            }
        )
    return brief
