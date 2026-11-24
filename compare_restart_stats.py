#!/usr/bin/env python3
"""Compares the Turbo runs of a restart series (run_restart_series.sh).

Usage:
    python3 compare_restart_stats.py logs/restart_series_<host>_<date>
    python3 compare_restart_stats.py DIR [--maximize] [--no-timeline] [--out PREFIX]

Reads every DIR/<instance>/*.log (any Turbo -s log works: the timeout and the -restart value are
read from the "% command:" line, not from the file name) and writes:
    DIR/comparison.csv   every metric of every run, one row per run
(the markdown tables are no longer written or printed; analyze_mznc_results.py / generate_pdf_report.py cover them)

For each instance and timeout, the restart runs are compared with the run without restart:
nodes ratio, idle difference, objective difference. The objective is minimized unless --maximize.
"""
import argparse, csv, glob, os, re, sys

STAT = re.compile(r"^%%%mzn-stat: ([A-Za-z0-9_]+)=(.*)$")
SLICE = re.compile(r"^%\s+(\d+)-\s*(\d+)% of kernel = \[\s*([\d.]+),\s*([\d.]+)\) s\s+idle\s+([\d.]+)%\s+busy blocks at end\s+(\d+)")
EXIT = re.compile(r"^% exit_code: (-?\d+)\s+wall_time_s: (\d+)")


def num(v):
    if v is None:
        return None
    v = v.strip().strip('"')
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


# Since 2026-11-23 Turbo prints the restart statistics with an "enr_" prefix (enr_tree_nodes, enr_restarts, ...).
# Logs from before keep the old names: new names are mapped back to the old keys, so both kinds of logs are read.
RENAMED_STATS = {"tree_nodes", "tree_failures", "redundant_dive_nodes", "restarts", "restart_items_sum", "restart_items_max",
                 "restart_items_last", "restart_subproblems_sum", "restart_s_max", "restart_truncated_blocks",
                 "restart_drain_ms_avg", "restart_drain_ms_max", "restart_first_sec", "restart_last_sec",
                 "restart_idle_pct", "restart_min_ms", "restart_subfactor", "restart_max"}


def stat_key(k):
    return k[4:] if k.startswith("enr_") and k[4:] in RENAMED_STATS else k


def parse(path):
    st, slices, r = {}, [], {"log": path}
    cmd = ""
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = STAT.match(line)
            if m:
                st[stat_key(m.group(1))] = num(m.group(2))  # last value wins (num_blocks is printed twice)
                continue
            m = SLICE.match(line)
            if m:
                slices.append((float(m.group(3)), float(m.group(4)), float(m.group(5)), int(m.group(6))))
                continue
            m = EXIT.match(line)
            if m:
                r["exit"], r["wall_s"] = int(m.group(1)), int(m.group(2))
            elif line.startswith("% command:"):
                cmd = line
            elif line.strip() == "==========":
                r["proved"] = True
            elif "=====UNSATISFIABLE=====" in line:
                r["unsat"] = True
    t = re.search(r"-t (\d+)", cmd)
    rs = re.search(r"-restart (\d+)", cmd)
    r["timeout_s"] = int(t.group(1)) // 1000 if t else st.get("timeout_ms", 0) // 1000
    r["restart_pct"] = int(rs.group(1)) if rs else int(st.get("restart_idle_pct", 0) or 0)
    r["config"] = f"T{r['timeout_s']} " + (f"restart {r['restart_pct']}%" if r["restart_pct"] else "no restart")
    r["stats"], r["slices"] = st, slices
    return r


# (column title, key or function, format)
def g(k):
    return lambda r: r["stats"].get(k)


def kernel_busy_rate(r):
    # nodes per second of a busy block: nodes / (block time actually spent working)
    s = r["stats"]
    nb, T, idle, nodes = s.get("num_blocks"), s.get("enr_kernel_time_sec"), s.get("enr_pct_idle"), s.get("nodes")
    if not all(isinstance(x, (int, float)) for x in (nb, T, idle, nodes)) or T == 0 or idle >= 100:
        return None
    return nodes / (nb * T * (1 - idle / 100))


METRICS = [
    ("objective", g("objective"), "{}"),
    ("best obj time s", g("best_obj_time"), "{:.0f}"),
    ("proved", lambda r: "yes" if r.get("proved") or r.get("unsat") else "no", "{}"),
    ("solutions", g("num_solutions"), "{}"),
    ("nodes", g("nodes"), "{:,}"),
    ("tree nodes", g("tree_nodes"), "{:,}"),
    ("failures", g("failures"), "{:,}"),
    ("peak depth", g("peakDepth"), "{}"),
    ("kernel nodes/s", g("enr_kernel_nodes_per_sec"), "{:,.0f}"),
    ("busy-block nodes/s", kernel_busy_rate, "{:,.0f}"),
    ("node us avg", g("enr_node_time_us_avg"), "{:.1f}"),
    ("fixpoint us avg", g("enr_node_fixpoint_us_avg"), "{:.1f}"),
    ("idle %", g("enr_pct_idle"), "{:.2f}"),
    ("fixpoint % of time", g("enr_pct_fixpoint_all"), "{:.1f}"),
    ("blocks", g("num_blocks"), "{}"),
    ("blocks done", g("enr_blocks_done"), "{}"),
    ("first block done s", lambda r: ("none" if isinstance(r["stats"].get("enr_kernel_first_block_done_sec"), (int, float))
                                      and r["stats"]["enr_kernel_first_block_done_sec"] < 0
                                      else r["stats"].get("enr_kernel_first_block_done_sec")), "{:.0f}"),
    ("EPS subproblems", g("eps_num_subproblems"), "{}"),
    ("subproblems solved", g("eps_solved_subproblems"), "{}"),
    ("subproblems skipped", g("eps_skipped_subproblems"), "{}"),
    ("subproblems/block min-max", lambda r: (f"{r['stats'].get('enr_block_subproblems_min')}-{r['stats'].get('enr_block_subproblems_max')}"
                                             if "enr_block_subproblems_min" in r["stats"] else None), "{}"),
    ("nodes/block min-max", lambda r: (f"{r['stats'].get('enr_block_nodes_min'):,}-{r['stats'].get('enr_block_nodes_max'):,}"
                                       if isinstance(r["stats"].get("enr_block_nodes_min"), int) else None), "{}"),
    ("restarts", g("restarts"), "{}"),
    ("items sum / max", lambda r: (f"{r['stats'].get('restart_items_sum')} / {r['stats'].get('restart_items_max')}"
                                   if "restart_items_sum" in r["stats"] else None), "{}"),
    ("split depth s max", g("restart_s_max"), "{}"),
    ("truncated blocks", g("restart_truncated_blocks"), "{}"),
    ("drain ms avg / max", lambda r: (f"{r['stats'].get('restart_drain_ms_avg'):.3f} / {r['stats'].get('restart_drain_ms_max'):.3f}"
                                      if isinstance(r["stats"].get("restart_drain_ms_avg"), (int, float)) else None), "{}"),
    ("first / last restart s", lambda r: (f"{r['stats'].get('restart_first_sec'):.0f} / {r['stats'].get('restart_last_sec'):.0f}"
                                          if isinstance(r["stats"].get("restart_first_sec"), (int, float)) and r["stats"].get("restarts") else None), "{}"),
    ("kernel time s", g("enr_kernel_time_sec"), "{:.0f}"),
    ("exit / wall s", lambda r: f"{r.get('exit', '?')} / {r.get('wall_s', '?')}", "{}"),
]


def fmt(v, f):
    if v is None or v == "":
        return "–"
    try:
        return f.format(v)
    except (ValueError, TypeError):
        return str(v)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--maximize", action="store_true", help="the objective is maximized (default: minimized)")
    ap.add_argument("--no-timeline", action="store_true", help="do not print the idle timelines")
    ap.add_argument("--out", help="output prefix (default: DIR/comparison)")
    a = ap.parse_args()

    logs = sorted(glob.glob(os.path.join(a.dir, "*", "*.log")))
    if not logs:
        sys.exit(f"no DIR/<instance>/*.log found in {a.dir}")
    by_inst = {}
    for p in logs:
        r = parse(p)
        if not r["stats"]:
            print(f"warning: no statistics in {p} (run crashed or still running?)", file=sys.stderr)
            continue
        by_inst.setdefault(os.path.basename(os.path.dirname(p)), []).append(r)

    better = (lambda x, y: x > y) if a.maximize else (lambda x, y: x < y)
    md = [f"# Restart series comparison: `{a.dir}`", ""]
    info = os.path.join(a.dir, "build_info.txt")
    if os.path.exists(info):
        md += ["```", open(info).read().strip(), "```", ""]

    csv_rows = []
    for inst, runs in sorted(by_inst.items()):
        runs.sort(key=lambda r: (r["timeout_s"], r["restart_pct"]))
        md += [f"## {inst}", ""]
        header = ["metric"] + [r["config"] for r in runs]
        rows = [[name] + [fmt(fn(r), f) for r in runs] for name, fn, f in METRICS]
        md += [md_table(header, rows), ""]

        # Comparison with the run without restart of the same timeout.
        comp = []
        for T in sorted({r["timeout_s"] for r in runs}):
            base = next((r for r in runs if r["timeout_s"] == T and r["restart_pct"] == 0), None)
            for r in runs:
                if r["timeout_s"] != T or r["restart_pct"] == 0 or base is None:
                    continue
                s, b = r["stats"], base["stats"]
                def ratio(k):
                    x, y = s.get(k), b.get(k)
                    return f"{x / y:.3f}×" if isinstance(x, (int, float)) and isinstance(y, (int, float)) and y else "–"
                def diff(k, f="{:+.2f}"):
                    x, y = s.get(k), b.get(k)
                    return f.format(x - y) if isinstance(x, (int, float)) and isinstance(y, (int, float)) else "–"
                ox, oy = s.get("objective"), b.get("objective")
                if isinstance(ox, (int, float)) and isinstance(oy, (int, float)):
                    verdict = "better" if better(ox, oy) else ("same" if ox == oy else "worse")
                    obj = f"{ox} vs {oy} ({verdict})"
                else:
                    obj = "–"
                comp.append([f"T{T} restart {r['restart_pct']}%", obj, ratio("nodes"), ratio("tree_nodes"),
                             ratio("failures"), diff("enr_pct_idle") + " pt", ratio("enr_node_time_us_avg"),
                             diff("best_obj_time", "{:+.0f}") + " s"])
        if comp:
            md += ["**Restart vs no restart, same timeout**", "",
                   md_table(["run", "objective (this vs no restart)", "nodes", "tree nodes", "failures",
                             "idle", "node time", "best obj time"], comp), ""]

        # Idle timelines side by side (one table per timeout).
        if not a.no_timeline:
            for T in sorted({r["timeout_s"] for r in runs}):
                rt = [r for r in runs if r["timeout_s"] == T and r["slices"]]
                if not rt:
                    continue
                n = max(len(r["slices"]) for r in rt)
                rows = []
                for i in range(n):
                    lo = next((r["slices"][i][0] for r in rt if i < len(r["slices"])), 0)
                    hi = next((r["slices"][i][1] for r in rt if i < len(r["slices"])), 0)
                    rows.append([f"{lo:.0f}–{hi:.0f}"] + [
                        (f"{r['slices'][i][2]:.1f}% ({r['slices'][i][3]})" if i < len(r["slices"]) else "–") for r in rt])
                md += [f"**Idle timeline, T{T}** (idle % of the block slots in the slice, busy blocks at its end)", "",
                       md_table(["slice s"] + [r["config"] for r in rt], rows), ""]

        for r in runs:
            row = {"instance": inst, "config": r["config"], "timeout_s": r["timeout_s"], "restart_pct": r["restart_pct"]}
            for name, fn, f in METRICS:
                row[name] = fn(r)
            row["log"] = r["log"]
            csv_rows.append(row)

    prefix = a.out or os.path.join(a.dir, "comparison")
    with open(prefix + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
        w.writeheader()
        w.writerows(csv_rows)
    print(f"written: {prefix}.csv", file=sys.stderr)


if __name__ == "__main__":
    main()
