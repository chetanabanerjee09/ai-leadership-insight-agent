import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def generate_plot(decision: Dict[str, Any], output_dir: str = ".") -> Optional[str]:
    """
    Generate a plot from a validated plot spec and save it as a PNG.

    Parameters
    ----------
    decision   : validated dict from EnginePipeline.decide_plot(), e.g.:
                 {
                     "plot_type": "line",
                     "title":     "Revenue Trend",
                     "x_label":   "Quarter",
                     "y_label":   "Revenue ($B)",
                     "data":      [{"x": "Q1 2024", "y": 5.2}, ...]
                 }
    output_dir : directory to save the plot (default: current directory)

    Returns
    -------
    str  — absolute path to saved PNG file
    None — if plotting fails
    """
    try:
        os.environ["MPLBACKEND"] = "Agg"  # override any Jupyter backend set in env
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe for scripts
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required: pip install matplotlib")

    plot_type = decision.get("plot_type", "bar")
    title     = decision.get("title", "Chart")
    x_label   = decision.get("x_label", "")
    y_label   = decision.get("y_label", "")
    data      = decision.get("data", [])

    x_values = [str(d["x"]) for d in data]
    y_values = [float(d["y"]) for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))

    if plot_type == "line":
        ax.plot(x_values, y_values, marker="o", linewidth=2, color="#2196F3")
        ax.fill_between(range(len(x_values)), y_values, alpha=0.1, color="#2196F3")

    elif plot_type == "bar":
        bars = ax.bar(x_values, y_values, color="#4CAF50", edgecolor="white", width=0.6)
        # Add value labels on top of each bar
        for bar, val in zip(bars, y_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(y_values) * 0.01,
                f"{val:g}",
                ha="center", va="bottom", fontsize=9,
            )

    elif plot_type == "pie":
        ax.pie(
            y_values,
            labels=x_values,
            autopct="%1.1f%%",
            startangle=140,
            colors=["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0",
                    "#00BCD4", "#FF5722", "#607D8B"],
        )
        ax.set_aspect("equal")

    else:
        # Default to bar for unknown types
        logger.warning(f"Unknown plot_type '{plot_type}' — defaulting to bar chart.")
        ax.bar(x_values, y_values, color="#4CAF50")

    # Styling
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if plot_type != "pie":
        ax.set_xlabel(x_label, fontsize=11)
        ax.set_ylabel(y_label, fontsize=11)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    safe_title = title.lower().replace(" ", "_").replace("/", "_")[:50]
    output_path = os.path.abspath(os.path.join(output_dir, f"{safe_title}.png"))
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Plot saved: {output_path}")
    return output_path