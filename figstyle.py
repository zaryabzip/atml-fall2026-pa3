# shared plain figure style for the report: stock matplotlib font, few colors, no top or right frame, direct line labels
import matplotlib
import matplotlib.pyplot as plt

INK, GRAY, LIGHT = "#222222", "#8a8a8a", "#c4c4c4"
BLUE, ORANGE, GREEN, RED = "#1f5fa8", "#d9772b", "#3a8a4a", "#c0392b"


def apply():
    plt.rcParams.update({
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "lines.linewidth": 1.8, "xtick.labelsize": 9, "ytick.labelsize": 9,
    })


def end_labels(ax, items, x, gap, fontsize=9):
    # items: (y, text, color) at the right end of the lines; spread so the labels never overlap
    items = sorted(items, key=lambda t: t[0])
    ys = [t[0] for t in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (y0, text, color), y in zip(items, ys):
        ax.annotate(text, xy=(x, y0), xytext=(x, y), textcoords="data", xycoords="data", fontsize=fontsize, color=color,
                    va="center", ha="left", annotation_clip=False)
