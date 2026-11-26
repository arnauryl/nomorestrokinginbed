"""Animation (GIF) of the two accumulators, bout by bout, using the real evidence() function.
HAND-MADE sequence of bouts (datos sintéticos), only to explain k and h.
Usage (repo root): python -m src.eval.animate_k  -> results/explain/k_h_animation.gif"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from src.models.evidence_accumulator import Params, evidence

OUT = Path(__file__).resolve().parents[2] / "results/explain/k_h_animation.gif"
# (kind, amp_left, amp_right) in g. First 5 bouts: healthy. Then the LEFT arm becomes weak.
SEQ = [("right_only", 0, .20), ("left_only", .15, 0), ("both", .20, .25), ("right_only", 0, .10), ("left_only", .20, 0),
       ("right_only", 0, .20), ("both", .02, .25), ("right_only", 0, .15), ("right_only", 0, .30), ("both", .03, .20),
       ("right_only", 0, .25), ("right_only", 0, .10), ("both", .02, .30), ("right_only", 0, .20), ("right_only", 0, .15)]
NAME = {"right_only": "only the RIGHT arm moves", "left_only": "only the LEFT arm moves", "both": "both arms move"}


def main():
    p = Params()
    S = {"left": 0.0, "right": 0.0}
    steps = [dict(S=dict(S), txt={"left": "", "right": ""}, alarm=None)]
    for kind, al, ar in SEQ:
        txt = {}
        for side in S:
            own = f"{side}_only"
            k_ = "both" if kind == "both" else "self_only" if kind == own else "contra_only"
            amp_s, amp_c = (al, ar) if side == "left" else (ar, al)
            e = evidence(k_, amp_s, amp_c, p)
            new = max(0.0, S[side] + e - p.k)
            txt[side] = f"{S[side]:.1f}  {e:+.1f} (evidence)  -{p.k} (k)  = {S[side] + e - p.k:.1f}" + ("  -> floor at 0" if new == 0 and S[side] + e - p.k < 0 else "")
            S[side] = new
        alarm = max(S, key=S.get) if max(S.values()) > p.h else None
        steps.append(dict(S=dict(S), txt=txt, alarm=alarm, kind=kind))
        if alarm:
            S = {"left": 0.0, "right": 0.0}
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    title = fig.suptitle("")

    def draw(i):
        for ax, side, color in zip(axes, ("left", "right"), ("tab:blue", "tab:orange")):
            ax.clear()
            xs = list(range(i + 1))
            ax.step(xs, [s["S"][side] for s in steps[:i + 1]], where="post", color=color, lw=2.5)
            ax.axhline(p.h, color="r", ls="--"); ax.text(len(SEQ) + .1, p.h, "h (alarm)", color="r", va="bottom", ha="right")
            ax.set_xlim(0, len(SEQ) + .2); ax.set_ylim(-.3, 6)
            ax.set_ylabel(f"suspicion: {side.upper()} arm weak")
            if i:
                ax.text(.02, .93, steps[i]["txt"][side], transform=ax.transAxes, va="top", fontsize=10, family="monospace",
                        bbox=dict(fc="w", ec=color))
            if steps[i]["alarm"] == side:
                ax.text(i, p.h + .6, "ALARM! (then reset to 0 and 30 min pause)", color="r", fontsize=12, ha="right")
        axes[1].set_xlabel("bout number (k is subtracted once per bout; time without movement changes nothing)")
        title.set_text("Datos sintéticos (hand-made example)\n" + (f"bout {i}: {NAME[steps[i]['kind']]}" if i else "start: both counts at 0")
                       + ("   [healthy phase]" if 0 < i <= 5 else "   [left arm weak]" if i > 5 else ""))

    last = len(steps) - 1
    anim = FuncAnimation(fig, lambda f: draw(min(f, last)), frames=len(steps) + 2, interval=1500)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    anim.save(OUT, writer=PillowWriter(fps=0.7))
    draw(next(i for i, s in enumerate(steps) if s["alarm"])); fig.savefig(OUT.with_name("k_h_animation_alarm_frame.png"), dpi=110)
    print([round(s["S"]["left"], 1) for s in steps], [round(s["S"]["right"], 1) for s in steps])


if __name__ == "__main__":
    main()
