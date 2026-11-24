#!/usr/bin/env python3
"""Summarize a run_restart_series.sh results folder (e.g. logs/mznc2024_5474391).

Per run: status, nodes, solve time (if search completed), idle %, best objective,
nodes/s.  Writes <DIR>/summary.csv and <DIR>/summary.md and prints the markdown.

usage: analyze_mznc_results.py DIR [--baseline BDIR] [--fzn-dir benchmarks/mznc2024/fzn]
  --baseline BDIR: take the norestart logs from BDIR/<instance>/ (DIR then only needs the restart logs).
"""
import csv, math, os, re, sys
from collections import Counter, defaultdict

STAT = re.compile(r"^%%%mzn-stat: (\w+)=(.*)$")
MODES = ["norestart", "restart20"]   # second entry replaced by the restartN found in DIR


def solve_kind(fzn):
    """'minimize' | 'maximize' | 'satisfy' | None, read from the fzn tail."""
    try:
        with open(fzn, "rb") as f:
            f.seek(max(0, os.path.getsize(fzn) - 4000))
            tail = f.read().decode(errors="replace")
    except OSError:
        return None
    m = re.findall(r"solve\b[^;]*?\b(satisfy|minimize|maximize)", tail)
    return m[-1] if m else None


def num(v):
    try:
        return float(v.strip().strip('"'))
    except (ValueError, AttributeError):
        return None


# Since 2026-11-23 Turbo prints the restart statistics with an "enr_" prefix (enr_tree_nodes, enr_restarts, ...).
# Logs from before keep the old names: new names are mapped back to the old keys, so both kinds of logs are read.
RENAMED_STATS = {"tree_nodes", "tree_failures", "redundant_dive_nodes", "restarts", "restart_items_sum", "restart_items_max",
                 "restart_items_last", "restart_subproblems_sum", "restart_s_max", "restart_truncated_blocks",
                 "restart_drain_ms_avg", "restart_drain_ms_max", "restart_first_sec", "restart_last_sec",
                 "restart_idle_pct", "restart_min_ms", "restart_subfactor", "restart_max"}


def stat_key(k):
    return k[4:] if k.startswith("enr_") and k[4:] in RENAMED_STATS else k


def parse_log(path):
    st, marker, sols, exit_code, wall = {}, None, 0, None, None
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            m = STAT.match(line)
            if m:
                st[stat_key(m.group(1))] = m.group(2)        # keep the last value
            elif line == "----------":
                sols += 1
            elif line in ("==========", "=====UNSATISFIABLE=====", "=====UNKNOWN=====",
                          "=====ERROR====="):
                marker = line
            elif line.startswith("% exit_code:"):
                mm = re.match(r"% exit_code:\s*(\d+)\s+wall_time_s:\s*(\d+)", line)
                if mm:
                    exit_code, wall = int(mm.group(1)), int(mm.group(2))
    return st, marker, sols, exit_code, wall


def classify(st, marker, sols, exit_code, kind, timeout_s):
    if exit_code not in (0, None):
        sig = {136: "SIGFPE", 137: "SIGKILL", 139: "SIGSEGV"}.get(exit_code, f"exit {exit_code}")
        return f"CRASH({sig})"
    if marker == "==========":
        return "OPTIMAL"
    if marker == "=====UNSATISFIABLE=====":
        return "UNSAT"
    if marker == "=====ERROR=====":
        return "ERROR"
    if sols > 0:
        t = num(st.get("solveTime"))
        if kind == "satisfy" and t is not None and t < timeout_s - 1:
            return "SOLVED"
        return "SAT(timeout)"
    return "UNKNOWN"


def fmt(v, nd=0, dash="–"):
    if v is None:
        return dash
    if nd == 0:
        return f"{v:,.0f}"
    return f"{v:,.{nd}f}"


def geomean(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    d = args[0]
    fzn_dir = "benchmarks/mznc2024/fzn"
    if "--fzn-dir" in args:
        fzn_dir = args[args.index("--fzn-dir") + 1]

    base = args[args.index("--baseline") + 1] if "--baseline" in args else None
    found = sorted({m.group(1) for _, _, fs in os.walk(d) for f in fs
                    for m in [re.match(r"T\d+_(restart(?:\d+|_each))\.log$", f)] if m})
    if found:
        MODES[1] = found[0]
    RM = MODES[1]
    rows = []
    sources = [(base, {"norestart"}), (d, {RM})] if base else [(d, None)]
    for inst in sorted(os.listdir(d)):
        if not os.path.isdir(os.path.join(d, inst)):
            continue
        kind = solve_kind(os.path.join(fzn_dir, inst + ".fzn"))
        for src, want in sources:
          p = os.path.join(src, inst)
          for fn in (sorted(os.listdir(p)) if os.path.isdir(p) else []):
            m = re.match(r"T(\d+)_(\w+)\.log$", fn)
            if not m or (want and m.group(2) not in want):
                continue
            T, mode = int(m.group(1)), m.group(2)
            st, marker, sols, ec, wall = parse_log(os.path.join(p, fn))
            status = classify(st, marker, sols, ec, kind, T)
            nodes = num(st.get("nodes"))
            t = num(st.get("solveTime"))                  # real wall solve time (solve_time is /40, upstream bug)
            done = status in ("OPTIMAL", "UNSAT", "SOLVED")
            obj = num(st.get("objective")) if sols > 0 else None
            rows.append(dict(
                instance=inst, family=inst.split("__")[0], mode=mode, T=T, kind=kind or "?",
                status=status, nodes=nodes,
                time_s=t if done else None,
                run_s=t if t is not None else wall,
                best_obj=obj,
                best_obj_time_s=num(st.get("best_obj_time")) if obj is not None else None,
                idle_pct=num(st.get("enr_pct_idle")),
                nodes_per_s=(nodes / t) if (nodes and t) else None,
                kernel_nodes_per_s=num(st.get("enr_kernel_nodes_per_sec")),
                restarts=num(st.get("restarts")),
                blocks_done=num(st.get("num_blocks_done")),
                solutions=sols, exit_code=ec, wall_s=wall,
            ))

    # ---- CSV
    keys = list(rows[0].keys())
    with open(os.path.join(d, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    by = defaultdict(dict)
    for r in rows:
        by[r["instance"]][r["mode"]] = r

    out = []
    P = out.append
    P(f"# MZC 2024 results: `{d}`" + (f" vs baseline `{base}`" if base else "") + "\n")
    P("Columns: **status**; **nodes** explored; **time** = solve time in s, only when search completed "
      "(optimal / unsat / solved), else `–`; **idle** = enr_pct_idle; **best** = best objective found; "
      "**n/s** = nodes / solveTime (whole run).\n")

    # status counts
    P("## Outcome counts\n")
    P("| status | " + " | ".join(MODES) + " |")
    P("|---|" + "---|" * len(MODES))
    cnt = {m: Counter(r["status"] for r in rows if r["mode"] == m) for m in MODES}
    for s in sorted(set().union(*cnt.values())):
        P(f"| {s} | " + " | ".join(str(cnt[m][s]) for m in MODES) + " |")
    P("")

    # comparison
    better = Counter()
    nps_ratio, node_ratio, idle_diff = [], [], []
    faster = Counter()
    for inst, mm in by.items():
        a, b = mm.get("norestart"), mm.get(RM)
        if not (a and b):
            continue
        if a["status"].startswith("CRASH") or b["status"].startswith("CRASH"):
            continue
        if a["nodes_per_s"] and b["nodes_per_s"]:
            nps_ratio.append(b["nodes_per_s"] / a["nodes_per_s"])
        if a["nodes"] and b["nodes"] and a["time_s"] is None and b["time_s"] is None:
            node_ratio.append(b["nodes"] / a["nodes"])
        if a["idle_pct"] is not None and b["idle_pct"] is not None:
            idle_diff.append(b["idle_pct"] - a["idle_pct"])
        if a["time_s"] is not None and b["time_s"] is not None:
            faster["r20 faster" if b["time_s"] < a["time_s"] * 0.95 else
                   "nr faster" if a["time_s"] < b["time_s"] * 0.95 else "same (±5%)"] += 1
        elif a["time_s"] is not None:
            faster["only nr completed"] += 1
        elif b["time_s"] is not None:
            faster["only r20 completed"] += 1
        oa, ob = a["best_obj"], b["best_obj"]
        if oa is None and ob is None:
            better["neither has a solution"] += 1
        elif oa is None:
            better["only r20 has a solution"] += 1
        elif ob is None:
            better["only nr has a solution"] += 1
        elif oa == ob:
            better["same objective"] += 1
        else:
            sign = -1 if a["kind"] == "minimize" else 1
            better["r20 better" if sign * (ob - oa) > 0 else "nr better"] += 1

    P(f"## {RM} vs norestart (crashed instances excluded)\n")
    P("| | count |\n|---|---|")
    for k, v in sorted(better.items()):
        P(f"| {k} | {v} |")
    for k, v in sorted(faster.items()):
        P(f"| completion: {k} | {v} |")
    gn, gr = geomean(nps_ratio), geomean(node_ratio)
    P(f"| geomean nodes/s ratio r20/nr ({len(nps_ratio)} inst.) | {fmt(gn, 3)} |")
    P(f"| geomean nodes ratio r20/nr, both timed out ({len(node_ratio)} inst.) | {fmt(gr, 3)} |")
    if idle_diff:
        P(f"| mean idle change r20 − nr (pt) | {sum(idle_diff) / len(idle_diff):+.2f} |")
        for m in MODES:
            n10 = sum(1 for r in rows if r["mode"] == m and (r["idle_pct"] or 0) > 10)
            P(f"| {m} runs with idle > 10% | {n10} |")
    P("")

    # per-instance table
    P("## Per instance (nr = norestart, r20 = the restart run)\n")
    P("| instance | kind | nr status | r20 status | nr nodes | r20 nodes | nr time | r20 time | "
      "nr idle% | r20 idle% | nr best | r20 best | nr n/s | r20 n/s | r20 restarts |")
    P("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for inst in sorted(by):
        a, b = by[inst].get("norestart", {}), by[inst].get(RM, {})
        g = lambda r, k, nd=0: fmt(r.get(k), nd)
        P(f"| {inst} | {(a or b).get('kind', '?')[:3]} | {a.get('status', '–')} | {b.get('status', '–')} | "
          f"{g(a, 'nodes')} | {g(b, 'nodes')} | {g(a, 'time_s', 1)} | {g(b, 'time_s', 1)} | "
          f"{g(a, 'idle_pct', 1)} | {g(b, 'idle_pct', 1)} | {g(a, 'best_obj')} | {g(b, 'best_obj')} | "
          f"{g(a, 'nodes_per_s')} | {g(b, 'nodes_per_s')} | {g(b, 'restarts')} |")

    md = "\n".join(out) + "\n"
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()
