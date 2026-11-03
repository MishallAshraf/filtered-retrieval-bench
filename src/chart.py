"""Draw the README chart from the hosted results on the guide's schema: results returned per user, before and after the fix."""
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import db  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BEFORE, AFTER = "#eb6834", "#2a78d6"
USERS = [("u_50pct", "50%"), ("u_10pct", "10%"), ("u_5pct", "5%"), ("u_1pct", "1%"), ("u_0_1pct", "0.1%")]


def returned(rows, config, setting):
    return [next(r["mean_returned"] for r in rows if r["variant"] == "single owner" and r["config"] == config
                 and r["setting"] == setting and r["user"] == user)
            for user, _ in USERS]


def main():
    rows = json.loads((db.ROOT / "results" / "guide_schema_supabase.json").read_text(encoding="utf-8"))["rows"]
    before = returned(rows, "guide policy only", "default")
    after = returned(rows, "document_id filter + indexes, re-sorted", "iterative")

    fig, ax = plt.subplots(figsize=(8.4, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(USERS))
    width = 0.34
    gap = 0.02
    ax.bar(x - width / 2 - gap, before, width, color=BEFORE, label="Default settings")
    ax.bar(x + width / 2 + gap, after, width, color=AFTER, label="document_id filter, indexes and iterative scan")
    for i, value in enumerate(before):
        ax.text(i - width / 2 - gap, value + 0.2, f"{value:.2f}".rstrip("0").rstrip("."), ha="center", va="bottom", fontsize=9, color=INK)
    for i, value in enumerate(after):
        ax.text(i + width / 2 + gap, value + 0.2, f"{value:.0f}", ha="center", va="bottom", fontsize=9, color=INK)

    ax.set_xticks(x, [label for _, label in USERS], fontsize=10, color=INK)
    ax.set_xlabel("Share of the table the user owns", fontsize=10, color=MUTED, labelpad=8)
    ax.set_ylabel("Average results returned, of 10 requested", fontsize=10, color=MUTED, labelpad=8)
    ax.set_ylim(0, 11.5)
    ax.set_yticks([0, 2, 4, 6, 8, 10])
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=9)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    fig.suptitle("Smaller users lose search results, with no error", x=0.075, y=0.97, ha="left",
                 fontsize=14, fontweight="bold", color=INK)
    fig.text(0.075, 0.885, "Supabase's RAG with Permissions schema, 57,600 sections, 648 questions, hosted project",
             fontsize=10, color=MUTED)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, frameon=False, fontsize=9.5, labelcolor=INK)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.2)
    path = db.ROOT / "results" / "results_by_user.png"
    fig.savefig(path, facecolor=SURFACE)
    print(path)


if __name__ == "__main__":
    main()
