#!/usr/bin/env python3
"""PDF report of a run_restart_series.sh results folder.

usage: generate_pdf_report.py DIR [--baseline BDIR] [--fzn-dir benchmarks/mznc2024/fzn] [-o DIR/report.pdf]
  --baseline BDIR: take the norestart logs from BDIR/<instance>/ (DIR then only needs the restart log).
Reuses the parsing of analyze_mznc_results.py.
"""
import os, re, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_mznc_results import parse_log, classify, solve_kind, num  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

MODES = ("norestart", "restart")          # restartN logs are loaded as "restart"; N goes into the label
LABEL = {"norestart": "no restart", "restart": "restart"}
# default dataviz palette, categorical slots 1-2 (validated pair), light surface
COLOR = {"norestart": "#2a78d6", "restart": "#eb6834"}
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
HL_GOOD_BG, HL_GOOD_INK = "#e3f3e3", "#006300"     # light tint of status good + success text
HL_BOTH_BG, HL_NEUTRAL_BG = "#eaf1fb", "#f0efec"

# slice "lo- hi% of kernel = [t0, t1) s  idle x%": t0/t1 are printed with 3 decimals (they round to 0 on ms kernels),
# so the slice is placed from its lo/hi % and the full-precision enr_kernel_time_sec instead.
SLICE = re.compile(r"^%\s+(\d+)-\s*(\d+)% of kernel = \[\s*([\d.]+),\s*([\d.]+)\) s\s+idle\s+([\d.]+)%")
DONE = ("OPTIMAL", "UNSAT", "SOLVED")

plt.rcParams.update({
    "font.size": 7.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.titlesize": 8, "axes.titleweight": "bold", "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def timeline(path, kend):
    """[(t0, t1, active%)] of the last idle timeline printed in the log; t0/t1 = lo/hi % of kend (kernel time)."""
    sl = []
    with open(path, errors="replace") as f:
        for line in f:
            if "idle timeline" in line:
                sl = []
            m = SLICE.match(line)
            if m:
                lo, hi, t0, t1, idle = map(float, m.groups())
                if kend:
                    t0, t1 = lo / 100.0 * kend, hi / 100.0 * kend
                sl.append((t0, t1, 100.0 - idle))
    return sl


def load(d, fzn_dir, base=None):
    runs = defaultdict(dict)
    sources = [(base, "norestart"), (d, r"restart(?:\d+|_each)")] if base else [(d, r"norestart|restart(?:\d+|_each)")]
    for inst in sorted(os.listdir(d)):
        if not os.path.isdir(os.path.join(d, inst)):
            continue
        kind = solve_kind(os.path.join(fzn_dir, inst + ".fzn")) or "?"
        for src, pat in sources:
          p = os.path.join(src, inst)
          for fn in (sorted(os.listdir(p)) if os.path.isdir(p) else []):
            m = re.match(r"T(\d+)_(\w+)\.log$", fn)
            if not m or not re.fullmatch(pat, m.group(2)):
                continue
            T, mode, path = int(m.group(1)), m.group(2), os.path.join(p, fn)
            if mode != "norestart":
                LABEL["restart"] = "restart each block" if mode == "restart_each" else f"restart {mode[7:]}%"
                mode = "restart"
            st, marker, sols, ec, wall = parse_log(path)
            status = classify(st, marker, sols, ec, kind, T)
            nodes, t = num(st.get("nodes")), num(st.get("solveTime"))
            runs[inst][mode] = dict(
                T=T, kind=kind, status=status, wall=wall, nodes=nodes,
                time=t if status in DONE else None,
                nps=nodes / t if nodes and t else None,
                obj=num(st.get("objective")) if sols else None,
                restarts=num(st.get("restarts")),
                first_restart=num(st.get("restart_first_sec")),
                prep=num(st.get("preprocessing_time")),
                nblocks=num(st.get("num_blocks")),
                kend=num(st.get("enr_kernel_time_sec")),
                idle=num(st.get("enr_pct_idle")),
                slices=timeline(path, num(st.get("enr_kernel_time_sec"))))
    return runs


def category(a, b):
    """'crash' | 'r20_only' | 'nr_only' | 'both' | 'neither'."""
    if not a or not b or a["status"].startswith("CRASH") or b["status"].startswith("CRASH"):
        return "crash"
    if a["status"] == "UNSAT" or b["status"] == "UNSAT":
        return "unsat"
    da, db = a["time"] is not None, b["time"] is not None
    return {(False, True): "r20_only", (True, False): "nr_only",
            (True, True): "both"}.get((da, db), "neither")


def f_int(v):
    return "–" if v is None else f"{v:,.0f}"


def f_time(r):
    if r is None:
        return "–"
    if r["status"].startswith("CRASH"):
        return f"{r['status'][6:-1]} @ {r['wall']} s"
    if r["status"] == "UNSAT":
        return f"unsat in preproc. ({r['prep']:,.0f} s)" if r["prep"] is not None else "unsat"
    if r["time"] is None:
        return f"timeout ({r['T']} s)"
    return f"{r['time']:,.1f} s"


def f_obj(r):
    if r is None or r["status"].startswith("CRASH"):
        return "–"
    if r["status"] == "UNSAT":
        return "unsat"
    return "no solution" if r["obj"] is None else f"{r['obj']:,.0f}"


def ratio(x, y, nd=2):
    return f"{y / x:.{nd}f}×" if x and y else ""


def obj_cmp(a, b):
    if not a or not b or a["obj"] is None or b["obj"] is None:
        return ""
    if a["obj"] == b["obj"]:
        return "same"
    sign = -1 if a["kind"] == "minimize" else 1
    return "r20 better" if sign * (b["obj"] - a["obj"]) > 0 else "nr better"


def banner(cat, a, b):
    if cat == "r20_only":
        return (f"{LABEL['restart']} finished in {b['time']:,.1f} s — no restart timed out ({a['T']} s)",
                HL_GOOD_BG, HL_GOOD_INK)
    if cat == "nr_only":
        return (f"no restart finished in {a['time']:,.1f} s — {LABEL['restart']} timed out", HL_NEUTRAL_BG, INK)
    if cat == "unsat":
        return ("proved unsat during preprocessing (0 nodes, check: MZC instance)", HL_NEUTRAL_BG, INK)
    if cat == "crash":
        parts = [f"{LABEL[m]}: {f_time(r)}" for m, r in (("norestart", a), ("restart", b))
                 if r and r["status"].startswith("CRASH")]
        return ("crashed — " + "; ".join(parts), HL_NEUTRAL_BG, INK)
    return None


def draw_instance(sf, inst, mm, page_of, NB):
    a, b = mm.get("norestart"), mm.get("restart")
    cat = category(a, b)
    fam, _, name = inst.partition("__")
    kind = (a or b)["kind"]
    gs = sf.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.45, wspace=0.18,
                         left=0.07, right=0.98, top=0.86, bottom=0.08)
    sf.text(0.02, 0.965, name, fontsize=10.5, fontweight="bold", va="top")
    sf.text(0.02, 0.925, f"family: {fam}    ({kind})", fontsize=7.5, color=INK2, va="top")
    bn = banner(cat, a, b)
    if bn:
        sf.text(0.98, 0.955, bn[0], ha="right", va="top", fontsize=7.5, color=bn[2],
                fontweight="bold" if cat == "r20_only" else "normal",
                bbox=dict(boxstyle="round,pad=0.35", fc=bn[1], ec="none"))

    ax = sf.add_subplot(gs[0, :])
    ax.axis("off")
    rows = [
        ["time", f_time(a), f_time(b), ratio(b and b["time"], a and a["time"]) if cat == "both" else ""],
        ["nodes explored", f_int(a and a["nodes"]), f_int(b and b["nodes"]), ratio(a and a["nodes"], b and b["nodes"])],
        ["nodes / s", f_int(a and a["nps"]), f_int(b and b["nps"]), ratio(a and a["nps"], b and b["nps"])],
        ["best objective", f_obj(a), f_obj(b), obj_cmp(a, b)],
        ["restarts", "off" if a else "–", f_int(b and b["restarts"]),
         f"first at {b['first_restart']:,.1f} s" if b and b["restarts"] and b["first_restart"] is not None else ""],
        ["mean idle", f"{a['idle']:.1f} %" if a and a["idle"] is not None else "–",
         f"{b['idle']:.1f} %" if b and b["idle"] is not None else "–", ""],
    ]
    tb = ax.table(cellText=rows, colLabels=["", LABEL["norestart"], LABEL["restart"], "restart / no restart"],
                  colWidths=[0.22, 0.27, 0.27, 0.2], loc="center", cellLoc="right", edges="horizontal")
    tb.auto_set_font_size(False)
    tb.set_fontsize(7.5)
    for (r, c), cell in tb.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_height(0.15)
        if c == 0:
            cell.set_text_props(ha="left", color=INK2)
        if r == 0:
            cell.set_text_props(fontweight="bold", color=INK2)
        if r == 0 and c in (1, 2):
            pass
            cell.set_text_props(color=INK, fontweight="bold")
    if cat == "r20_only":
        tb[1, 2].set_text_props(color=HL_GOOD_INK, fontweight="bold")

    for i, (mode, r) in enumerate((("norestart", a), ("restart", b))):
        axp = sf.add_subplot(gs[1, i])
        axp.set_title(LABEL[mode] + (f"  ·  {r['restarts']:.0f} restarts" if mode == "restart" and r and r["restarts"] else ""),
                      color=INK, loc="left")
        axp.set_ylim(0, NB * 1.2)
        # x range: this run's own kernel time, so the 20 slices have the same width in both plots
        T = max(r["slices"][-1][1] if r and r["slices"] else 0.0, 1e-3)
        axp.set_xlim(0, T * 1.01)
        axp.set_yticks([round(NB * k / 4) for k in range(5)])
        axp.grid(axis="y", color=GRID, lw=0.5)
        axp.set_axisbelow(True)
        for s in ("top", "right"):
            axp.spines[s].set_visible(False)
        axp.set_xlabel("time (s)")
        if i == 0:
            axp.set_ylabel(f"active blocks (of {NB})")
        if r and r["slices"]:
            x = [s[0] for s in r["slices"]]
            w = [(s[1] - s[0]) for s in r["slices"]]
            nb = r["nblocks"] or NB
            h = [s[2] / 100.0 * nb for s in r["slices"]]
            axp.bar(x, h, width=[wi * 0.9 for wi in w], align="edge", color=COLOR[mode], lw=0)
            if nb < NB:
                axp.axhline(nb, color=MUTED, lw=0.8, ls="--")
                axp.text(T * 0.01, nb + NB * 0.025, f"{nb:.0f} blocks launched", color=INK2, fontsize=6.5, va="bottom")
            if r["time"] is not None:
                te = r["kend"] if r["kend"] is not None else r["slices"][-1][1]
                axp.axvline(te, color="#d03b3b", lw=1.5)
                left = te < T * 0.7
                axp.text(te + (T * 0.01 if left else -T * 0.01), NB * 1.13, ("search ended " + (f"{te * 1000:.1f} ms" if te < 1 else f"{te:,.1f} s")),
                         color=INK, fontsize=6.5, va="center", ha="left" if left else "right")
        else:
            msg = "no timeline (crashed)" if r and r["status"].startswith("CRASH") else "no timeline"
            axp.text(0.5, 0.5, msg, transform=axp.transAxes, ha="center", color=MUTED)
    sf.text(0.98, 0.005, f"p. {page_of[inst]}", ha="right", va="bottom", fontsize=6.5, color=MUTED)


def overview(pdf, d, runs, page_of, base=None):
    fig = plt.figure(figsize=(8.27, 11.69))
    y = 0.95
    fig.text(0.07, y, "Restart vs no restart — MZC 2024", fontsize=16, fontweight="bold")
    y -= 0.03
    info = ""
    bi = os.path.join(d, "build_info.txt")
    if os.path.exists(bi):
        with open(bi) as f:
            info = [l[2:].strip() for l in f if l.startswith("% host") or l.startswith("% gpu:")
                    or l.startswith("% instances") or l.startswith("% gpu_barebones")]
        info = "\n".join(info)
    head = f"run:      {d}\nbaseline: {base}" if base else d
    fig.text(0.07, y, f"{head}\n{info}", fontsize=7, color=INK2, va="top", family="monospace")
    y -= 0.09

    cats = Counter(category(mm.get("norestart"), mm.get("restart")) for mm in runs.values())
    better = Counter(obj_cmp(mm.get("norestart"), mm.get("restart")) or "n/a" for mm in runs.values()
                     if category(mm.get("norestart"), mm.get("restart")) != "crash")
    lines = [
        f"instances: {len(runs)}",
        f"finished by both: {cats['both']}      only with restart: {cats['r20_only']}      "
        f"only without restart: {cats['nr_only']}      neither: {cats['neither']}      unsat: {cats['unsat']}      crashed: {cats['crash']}",
        f"best objective (r20 vs nr): better {better['r20 better']}, same {better['same']}, "
        f"worse {better['nr better']}, no solution in one/both {better['n/a']}",
    ]
    fig.text(0.07, y, "\n".join(lines), fontsize=8.5, va="top", linespacing=1.7)
    y -= 0.08

    def listing(title, insts, bg, cols, rowf):
        nonlocal y
        fig.text(0.07, y, title, fontsize=10, fontweight="bold", va="top")
        y -= 0.022
        if not insts:
            fig.text(0.07, y, "none", color=MUTED, va="top")
            y -= 0.03
            return
        h = 0.0165 * (len(insts) + 1)
        ax = fig.add_axes([0.07, y - h, 0.86, h])
        ax.axis("off")
        tb = ax.table(cellText=[rowf(i) for i in insts], colLabels=cols, loc="upper center",
                      cellLoc="right", edges="horizontal", bbox=[0, 0, 1, 1],
                      colWidths=[0.50, 0.17, 0.17, 0.09, 0.07])
        tb.auto_set_font_size(False)
        tb.set_fontsize(6.5)
        for (r, c), cell in tb.get_celld().items():
            cell.set_edgecolor(GRID)
            if c == 0:
                cell.set_text_props(ha="left")
            if r == 0:
                cell.set_text_props(fontweight="bold", color=INK2)
            elif bg:
                cell.set_facecolor(bg)
        y -= h + 0.035

    r20_only = sorted(i for i, mm in runs.items() if category(mm.get("norestart"), mm.get("restart")) == "r20_only")
    both = sorted(i for i, mm in runs.items() if category(mm.get("norestart"), mm.get("restart")) == "both")
    crash = sorted(i for i, mm in runs.items() if category(mm.get("norestart"), mm.get("restart")) == "crash")
    unsat = sorted(i for i, mm in runs.items() if category(mm.get("norestart"), mm.get("restart")) == "unsat")
    nr_only = sorted(i for i, mm in runs.items() if category(mm.get("norestart"), mm.get("restart")) == "nr_only")

    def row(i):
        a, b = runs[i]["norestart"], runs[i]["restart"]
        sp = f"{a['time'] / b['time']:.2f}×" if a["time"] and b["time"] else ""
        return [i, f_time(a), f_time(b), sp, str(page_of[i])]

    cols = ["instance", "no restart", LABEL["restart"], "speedup", "page"]
    listing("Finished only with restart (no restart timed out)", r20_only, HL_GOOD_BG, cols, row)
    if nr_only:
        listing("Finished only without restart", nr_only, None, cols, row)
    listing("Finished by both", both, None, cols, row)
    listing("Crashed", crash, None, cols,
            lambda i: [i, f_time(runs[i].get("norestart")), f_time(runs[i].get("restart")), "", str(page_of[i])])
    fig.text(0.07, 0.02, "time = solveTime when the search completed (optimal / unsat); nodes/s = nodes / solveTime; "
             "active blocks = (100 − idle %) × launched blocks, per 5% slice of the kernel time;\n"
             "red line = end of the search (kernel time); each plot's x axis = its own run's kernel time; dashed line = blocks launched when fewer than the maximum.", fontsize=6.5, color=MUTED)
    pdf.savefig(fig)
    plt.close(fig)


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    d = args[0].rstrip("/")
    fzn_dir = args[args.index("--fzn-dir") + 1] if "--fzn-dir" in args else "benchmarks/mznc2024/fzn"
    out = args[args.index("-o") + 1] if "-o" in args else os.path.join(d, "report.pdf")

    base = args[args.index("--baseline") + 1].rstrip("/") if "--baseline" in args else None
    runs = load(d, fzn_dir, base)
    NB = int(max([r["nblocks"] or 0 for mm in runs.values() for r in mm.values()] + [1]))
    insts = sorted(runs)
    page_of = {inst: 2 + k // 2 for k, inst in enumerate(insts)}
    with PdfPages(out) as pdf:
        overview(pdf, d, runs, page_of, base)
        for k in range(0, len(insts), 2):
            fig = plt.figure(figsize=(8.27, 11.69))
            sfs = fig.subfigures(2, 1, hspace=0.02)
            for sf, inst in zip(sfs, insts[k:k + 2]):
                draw_instance(sf, inst, runs[inst], page_of, NB)
            fig.add_artist(plt.Line2D([0.04, 0.96], [0.5, 0.5], color=GRID, lw=0.8))
            pdf.savefig(fig)
            plt.close(fig)
    print(f"wrote {out}  ({1 + (len(insts) + 1) // 2} pages, {len(insts)} instances)")


if __name__ == "__main__":
    main()
