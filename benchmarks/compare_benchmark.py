import json
from pathlib import Path
from statistics import mean, median, stdev
import csv
import math

def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def pick_time_value(entry: dict):
    """Robustly pick a numeric timing from an entry.
    Prefer mean_time_ms, then median_time_ms, then max_time_ms, then any numeric field.
    Returns (value_ms: float or None, size: int or None).
    """
    for key in ("mean_time_ms", "median_time_ms", "max_time_ms"):
        if key in entry:
            try:
                return float(entry[key]), entry.get("size")
            except Exception:
                pass
    # fallback: search for first numeric value
    for k, v in entry.items():
        if isinstance(v, (int, float)):
            # ignore small integer metadata like iteration counts if name-like keys present
            if k in ("iterations", "size", "runs"):
                continue
            try:
                return float(v), entry.get("size")
            except Exception:
                pass
    return None, entry.get("size")

def build_map(entries):
    # allow entries that might already be a dict of results
    out = {}
    for e in entries:
        name = e.get("name") or e.get("id") or e.get("test") or None
        if not name:
            # generate fallback unique name if missing
            name = json.dumps(e, sort_keys=True)
        out[name] = e
    return out

def compare_category(fastlowess_entries, stats_entries):
    fastlowess_map = build_map(fastlowess_entries)
    stats_map = build_map(stats_entries)
    common = sorted(set(fastlowess_map.keys()) & set(stats_map.keys()))
    rows = []
    speedups = []
    for name in common:
        fl_entry = fastlowess_map[name]
        s_entry = stats_map[name]
        fl_val, fl_size = pick_time_value(fl_entry)
        s_val, s_size = pick_time_value(s_entry)

        row = {
            "name": name,
            "fastlowess_value_ms": fl_val,
            "stats_value_ms": s_val,
            "fastlowess_size": fl_size,
            "stats_size": s_size,
            "notes": []
        }

        if fl_val is None or s_val is None:
            row["notes"].append("missing_metric")
            rows.append(row)
            continue

        # core comparisons
        if fl_val == 0 or s_val == 0:
            speedup = None
        else:
            speedup = s_val / fl_val  # >1 => Statsmodels faster by this factor
        row["speedup_stats_over_fastlowess"] = speedup
        if speedup is not None:
            row["log2_speedup"] = math.log2(speedup) if speedup > 0 else None
            row["percent_change_stats_vs_fastlowess"] = ((s_val - fl_val) / fl_val) * 100.0
            speedups.append(speedup)

        # absolute diffs
        row["absolute_diff_ms"] = None if fl_val is None or s_val is None else (s_val - fl_val)
        row["abs_percent_vs_fastlowess"] = None if fl_val == 0 else abs(row["absolute_diff_ms"]) / fl_val * 100.0

        # per-point normalization if size available and >0
        size = fl_size or s_size
        if size:
            try:
                size_i = int(size)
                row["fastlowess_ms_per_point"] = fl_val / size_i
                row["stats_ms_per_point"] = s_val / size_i
                row["speedup_per_point"] = None if row["fastlowess_ms_per_point"] == 0 else row["stats_ms_per_point"] / row["fastlowess_ms_per_point"]
            except Exception:
                row["notes"].append("bad_size")

        rows.append(row)
    summary = {
        "compared": len(common),
        "mean_speedup": mean(speedups) if speedups else None,
        "median_speedup": median(speedups) if speedups else None,
        "count_with_metrics": len(speedups),
    }
    return rows, summary

def main():
    repo_root = Path(__file__).resolve().parent
    # walk up to workspace root (same heuristic as other scripts)
    workspace = repo_root
    for _ in range(6):
        if (workspace / "output").exists():
            break
        if workspace.parent == workspace:
            break
        workspace = workspace.parent
    out_dir = workspace / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    fastlowess_path = out_dir / "fastLowess_benchmark.json"
    stats_path = out_dir / "statsmodels_benchmark.json"

    if not fastlowess_path.exists() or not stats_path.exists():
        missing = []
        if not fastlowess_path.exists():
            missing.append(str(fastlowess_path))
        if not stats_path.exists():
            missing.append(str(stats_path))
        print("Missing files:", ", ".join(missing))
        return

    fastlowess = load_json(fastlowess_path)
    stats = load_json(stats_path)

    all_keys = sorted(set(fastlowess.keys()) | set(stats.keys()))
    comparison = {}
    overall_speedups = []

    # detailed rows for CSV
    csv_rows = []
    csv_fieldnames = [
        "category","name","fastlowess_value_ms","stats_value_ms","speedup_stats_over_fastlowess",
        "log2_speedup","percent_change_stats_vs_fastlowess","absolute_diff_ms","abs_percent_vs_fastlowess",
        "fastlowess_size","stats_size","fastlowess_ms_per_point","stats_ms_per_point","speedup_per_point","notes"
    ]

    for key in all_keys:
        fl_entries = fastlowess.get(key, [])
        s_entries = stats.get(key, [])
        rows, summary = compare_category(fl_entries, s_entries)
        comparison[key] = {"rows": rows, "summary": summary}
        if summary["median_speedup"] is not None:
            overall_speedups.append(summary["median_speedup"])
        for row in rows:
            csv_rows.append({
                "category": key,
                "name": row.get("name"),
                "fastlowess_value_ms": row.get("fastlowess_value_ms"),
                "stats_value_ms": row.get("stats_value_ms"),
                "speedup_stats_over_fastlowess": row.get("speedup_stats_over_fastlowess"),
                "log2_speedup": row.get("log2_speedup"),
                "percent_change_stats_vs_fastlowess": row.get("percent_change_stats_vs_fastlowess"),
                "absolute_diff_ms": row.get("absolute_diff_ms"),
                "abs_percent_vs_fastlowess": row.get("abs_percent_vs_fastlowess"),
                "fastlowess_size": row.get("fastlowess_size"),
                "stats_size": row.get("stats_size"),
                "fastlowess_ms_per_point": row.get("fastlowess_ms_per_point"),
                "stats_ms_per_point": row.get("stats_ms_per_point"),
                "speedup_per_point": row.get("speedup_per_point"),
                "notes": ";".join(row.get("notes", []))
            })

    print("\nBenchmark comparison (statsmodels_ms / fastlowess_ms):")
    for key, data in comparison.items():
        s = data["summary"]
        print(f"- {key}: compared={s['compared']}, median_speedup={s['median_speedup']}, mean_speedup={s['mean_speedup']}")

    # Top wins and regressions across all categories
    all_rows = [r for cat in comparison.values() for r in cat["rows"] if r.get("speedup_stats_over_fastlowess") is not None]
    if all_rows:
        sorted_by_speed = sorted(all_rows, key=lambda r: r["speedup_stats_over_fastlowess"] or 0, reverse=True)
        sorted_by_regression = sorted(all_rows, key=lambda r: r["speedup_stats_over_fastlowess"] or 0)

        print("\nTop 10 FastLowess wins (largest stats_ms / fastlowess_ms):")
        for r in sorted_by_speed[:10]:
            print(f"  {r['name']}: stats={r['stats_value_ms']:.4f}ms, fastlowess={r['fastlowess_value_ms']:.4f}ms, speedup={r['speedup_stats_over_fastlowess']:.2f}x")

        print("\nTop 10 regressions (statsmodels faster than FastLowess):")
        for r in sorted_by_regression[:10]:
            if r["speedup_stats_over_fastlowess"] < 1.0:
                print(f"  {r['name']}: stats={r['stats_value_ms']:.4f}ms, fastlowess={r['fastlowess_value_ms']:.4f}ms, speedup={r['speedup_stats_over_fastlowess']:.2f}x")

    # Print detailed per-category rows to console
    print("\nDetailed per-category results:")
    for cat, data in comparison.items():
        rows = data["rows"]
        if not rows:
            continue
        print(f"\nCategory: {cat} (compared={data['summary']['compared']})")
        # header
        print(f"{'name':60} {'fastlowess_ms':>10} {'stats_ms':>10} {'speedup':>8} {'%chg':>8} {'notes'}")
        for r in rows:
            name = (r.get("name") or "")[:60].ljust(60)
            fastlowess_v = r.get("fastlowess_value_ms")
            stats_v = r.get("stats_value_ms")
            sp = r.get("speedup_stats_over_fastlowess")
            pct = r.get("percent_change_stats_vs_fastlowess")
            notes = ";".join(r.get("notes", []))
            fastlowess_s = f"{fastlowess_v:.4f}" if isinstance(fastlowess_v, (int, float)) else "N/A"
            stats_s = f"{stats_v:.4f}" if isinstance(stats_v, (int, float)) else "N/A"
            sp_s = f"{sp:.2f}x" if isinstance(sp, (int, float)) else "N/A"
            pct_s = f"{pct:.1f}%" if isinstance(pct, (int, float)) else "N/A"
            print(f"{name} {fastlowess_s:>10} {stats_s:>10} {sp_s:>8} {pct_s:>8} {notes}")

if __name__ == "__main__":
    main()
