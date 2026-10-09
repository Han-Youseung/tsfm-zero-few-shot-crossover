"""Read back scientific tables and exported workbook, without editing either."""

import argparse
import csv
import json
import math
from pathlib import Path


def check(output):
    import openpyxl

    data = json.loads((output / "tables/workbook_inputs.json").read_text("utf-8"))
    book = openpyxl.load_workbook(output / "paper_tables.xlsx", data_only=True)
    formulas = openpyxl.load_workbook(output / "paper_tables.xlsx", data_only=False)
    errors = []
    for sheet in book:
        for row in sheet:
            errors.extend(
                f"{sheet.title}!{c.coordinate}: {c.value}" for c in row if c.data_type == "e"
            )
    assert not errors, errors

    def near(a, b):
        return isinstance(a, (int, float)) and math.isclose(a, b, rel_tol=0, abs_tol=1e-11)

    curves = [r for r in data["curves"] if r["metric"] == "normalized_mae"]
    for i, row in enumerate(curves, 5):
        for col, key in enumerate(
            (
                "zero_mean",
                "few_mean",
                "mean_relative_improvement",
                "seed_min_relative_improvement",
                "seed_max_relative_improvement",
            ),
            4,
        ):
            assert near(book["Curves"].cell(i, col).value, row[key]), (i, key)
            assert formulas["Curves"].cell(i, col).data_type == "f"
    ends = [r for r in data["endpoints"] if r["metric"] == "normalized_mae"]
    for i, row in enumerate(ends, 5):
        assert near(book["Summary"].cell(i, 5).value, row["mean_relative_improvement"])
        assert book["Summary"].cell(i, 9).value == row["all_seed_sustained_interval"]
    for i, row in enumerate(data["conditions"], 5):
        assert near(book["Conditions"].cell(i, 5).value, row["normalized_mae"])
        assert near(book["Conditions"].cell(i, 6).value, row["normalized_mse"])
        assert book["Conditions"].cell(i, 10).value == row["actual_optimizer_steps"]
        assert book["Conditions"].cell(i, 12).value == row["actual_unique_windows"]
    for name, rows in data.items():
        with (output / "tables" / f"{name}.csv").open(encoding="utf-8-sig", newline="") as handle:
            recovered = list(csv.DictReader(handle))
        assert len(recovered) == len(rows), name
        for actual, original in zip(recovered, rows, strict=True):
            for key, value in original.items():
                if isinstance(value, float):
                    assert float(actual[key]) == value, (name, key)
                else:
                    assert actual[key] == ("" if value is None else str(value)), (name, key)
    raw = json.loads((output / "evidence/final_results.json").read_text("utf-8"))
    assert len(raw) == 270
    assert all(r["status"] == "completed" and r["test_evaluation"] for r in raw)
    assert len({r["identity"]["condition"]["id"] for r in raw}) == 270
    actual_formulas = sum(c.data_type == "f" for s in formulas for row in s for c in row)
    assert actual_formulas == 690
    return {
        "xlsx_readback": "passed",
        "xlsx_formula_errors": errors,
        "formulas": actual_formulas,
        "curve_rows_checked": 80,
        "endpoint_rows_checked": 10,
        "conditions_checked": 270,
        "csv_round_trip_all_fields": "passed",
        "original_final_test_records": 270,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
