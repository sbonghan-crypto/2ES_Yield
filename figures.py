"""Plot synthetic planets, detected planets, and RV mass limits."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


X_AXIS_FIELDS = {
    "period": ("period_days", "Period [days]"),
    "sma": ("sma_au", "Semi-major axis [AU]"),
}


def _resolve_star_file(directory: str | Path, star_name: str) -> Path:
    directory = Path(directory)
    exact = directory / f"{star_name}.npy"
    if exact.exists():
        return exact

    matches = [
        path
        for path in directory.glob("*.npy")
        if star_name.lower() in path.stem.lower()
    ]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"No .npy file for {star_name!r} in {directory}")
    raise ValueError(f"Multiple files match {star_name!r}: {[path.name for path in matches]}")


def _require_fields(array: np.ndarray, fields: list[str], path: Path) -> None:
    names = array.dtype.names or ()
    missing = [field for field in fields if field not in names]
    if missing:
        raise KeyError(f"{path} is missing fields: {missing}")


def check_one_star(
    star_name: str,
    gen_planet_path: str | Path,
    det_limit_path: str | Path,
    det_planet_path: str | Path,
    x_axis: str = "period",
    mc_run: int | None = None,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str = "Planet mass [M$_\\oplus$]",
    generated_label: str = "Undetected planets",
    detected_label: str = "Detected planets",
    limit_label: str = "50% RV mass limit",
    figsize: tuple[float, float] = (7.0, 5.0),
    dpi: int = 140,
    generated_color: str = "0.65",
    detected_color: str = "tab:red",
    limit_color: str = "black",
    generated_marker: str = "o",
    detected_marker: str = "o",
    generated_size: float = 28.0,
    detected_size: float = 36.0,
    generated_alpha: float = 0.45,
    detected_alpha: float = 0.9,
    limit_linestyle: str = "-",
    limit_linewidth: float = 2.0,
    xscale: str = "linear",
    yscale: str = "linear",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    font_size: float = 11.0,
    title_font_size: float | None = None,
    label_font_size: float | None = None,
    tick_font_size: float | None = None,
    legend_font_size: float | None = None,
    legend_loc: str = "best",
    grid: bool = True,
    grid_alpha: float = 0.25,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """Plot one star's generated planets, detected planets, and mass limit.

    Parameters
    ----------
    star_name
        Source ID/file stem to plot. A unique case-insensitive partial match is
        also accepted.
    gen_planet_path
        Directory containing generated planet ``.npy`` files from ``spawn.run``.
    det_limit_path
        Directory containing detection-limit ``.npy`` files from
        ``detection.get_mass_limit``.
    det_planet_path
        Directory containing detected-planet ``.npy`` files from
        ``detection.run_yield``.
    x_axis
        ``"period"`` or ``"sma"``.
    mc_run
        If set, plot only planets from this Monte Carlo run.

    Returns
    -------
    tuple
        ``(fig, ax)`` for further customization.
    """
    if x_axis not in X_AXIS_FIELDS:
        raise ValueError(f"x_axis must be one of {list(X_AXIS_FIELDS)}")

    x_field, default_xlabel = X_AXIS_FIELDS[x_axis]
    xlabel = xlabel or default_xlabel
    title = title if title is not None else star_name
    title_font_size = title_font_size or font_size + 2
    label_font_size = label_font_size or font_size
    tick_font_size = tick_font_size or font_size - 1
    legend_font_size = legend_font_size or font_size - 1

    gen_file = _resolve_star_file(gen_planet_path, star_name)
    limit_file = _resolve_star_file(det_limit_path, star_name)
    detected_file = _resolve_star_file(det_planet_path, star_name)

    generated = np.load(gen_file)
    limits = np.load(limit_file)
    detected_planets = np.load(detected_file)

    _require_fields(generated, [x_field, "mass_mearth", "mc_run"], gen_file)
    _require_fields(limits, [x_field, "mass_limit_mearth"], limit_file)
    _require_fields(
        detected_planets,
        [x_field, "mass_mearth", "mc_run", "detected"],
        detected_file,
    )

    if mc_run is not None:
        generated = generated[generated["mc_run"] == mc_run]
        detected_planets = detected_planets[detected_planets["mc_run"] == mc_run]

    undetected_only = detected_planets[~detected_planets["detected"]]
    detected_only = detected_planets[detected_planets["detected"]]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    if len(undetected_only) > 0:
        ax.scatter(
            undetected_only[x_field],
            undetected_only["mass_mearth"],
            s=generated_size,
            c=generated_color,
            marker=generated_marker,
            alpha=generated_alpha,
            edgecolors="none",
            label=generated_label,
        )

    if len(detected_only) > 0:
        ax.scatter(
            detected_only[x_field],
            detected_only["mass_mearth"],
            s=detected_size,
            c=detected_color,
            marker=detected_marker,
            alpha=detected_alpha,
            edgecolors="black",
            linewidths=0.4,
            label=detected_label,
            zorder=3,
        )

    ax.plot(
        limits[x_field],
        limits["mass_limit_mearth"],
        color=limit_color,
        linestyle=limit_linestyle,
        linewidth=limit_linewidth,
        label=limit_label,
        zorder=2,
    )

    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    if grid:
        ax.grid(True, alpha=grid_alpha)
    ax.legend(loc=legend_loc, fontsize=legend_font_size)
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax


def yield_histogram(
    cat_path: str | Path,
    det_planet_path: str | Path,
    mass_mearth_cut: tuple[float | None, float | None] | None = None,
    radius_rearth_cut: tuple[float | None, float | None] | None = None,
    bins: int | np.ndarray = 10,
    hist_range: tuple[float, float] | None = None,
    title: str | None = None,
    xlabel: str = "Detectable planets per star",
    ylabel: str = "Number of stars",
    total_yield_label: str = "Total yield = {total_yield:.2f}",
    print_total_yield: bool = True,
    show_total_yield: bool = False,
    total_yield_position: tuple[float, float] = (0.97, 0.95),
    total_yield_ha: str = "right",
    total_yield_va: str = "top",
    total_yield_bbox: dict | None = None,
    figsize: tuple[float, float] = (7.0, 4.5),
    dpi: int = 140,
    color: str = "tab:blue",
    edgecolor: str = "black",
    linewidth: float = 0.8,
    alpha: float = 0.75,
    histtype: str = "bar",
    font_size: float = 11.0,
    title_font_size: float | None = None,
    label_font_size: float | None = None,
    tick_font_size: float | None = None,
    text_font_size: float | None = None,
    grid: bool = True,
    grid_alpha: float = 0.25,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ax: plt.Axes | None = None,
    save_path: str | Path | None = None,
    show: bool = True,
) -> tuple[plt.Figure, plt.Axes, pd.DataFrame]:
    """Plot a histogram of per-star detectable planet yield.

    The yield for one star is:

    ``number of detected planets satisfying the cuts across all MC runs / n_mc``.

    Parameters
    ----------
    cat_path
        Input stellar catalogue CSV. Every ``source_id`` in this catalogue is
        included, even when the star has zero detected planets.
    det_planet_path
        Directory containing detected-planet ``.npy`` files and
        ``yield_by_mc_run.csv`` from ``detection.run_yield``.
    mass_mearth_cut
        Optional inclusive ``(min, max)`` cut on detected planet mass.
        Use ``None`` for an open bound, e.g. ``(0.7, None)``.
    radius_rearth_cut
        Optional inclusive ``(min, max)`` cut on detected planet radius.
        Use ``None`` for an open bound, e.g. ``(None, 2.0)``.
    print_total_yield
        If ``True``, print the summed yield across all catalogue stars.
    show_total_yield
        If ``True``, annotate the summed yield on the plot.

    Returns
    -------
    tuple
        ``(fig, ax, yield_table)`` where ``yield_table`` contains one row per
        catalogue star plus a ``yield`` column.
    """
    cat = pd.read_csv(cat_path)
    if "source_id" not in cat.columns:
        raise KeyError("Catalogue is missing required column: source_id")

    det_planet_path = Path(det_planet_path)
    yield_by_run_path = det_planet_path / "yield_by_mc_run.csv"
    if not yield_by_run_path.exists():
        raise FileNotFoundError(f"Missing yield-by-run file: {yield_by_run_path}")

    n_mc = len(pd.read_csv(yield_by_run_path))
    if n_mc <= 0:
        raise ValueError(f"No Monte Carlo runs found in {yield_by_run_path}")

    title = title if title is not None else "Detectable Planet Yield"
    title_font_size = title_font_size or font_size + 2
    label_font_size = label_font_size or font_size
    tick_font_size = tick_font_size or font_size - 1
    text_font_size = text_font_size or font_size
    if total_yield_bbox is None:
        total_yield_bbox = {
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.9,
        }

    rows = []
    for source_id in cat["source_id"].astype(str):
        detected_file = det_planet_path / f"{source_id}.npy"
        if not detected_file.exists():
            raise FileNotFoundError(f"Missing detected planet file: {detected_file}")

        detected_planets = np.load(detected_file)
        _require_fields(
            detected_planets,
            ["detected", "mass_mearth", "radius_rearth"],
            detected_file,
        )

        selected = detected_planets["detected"].astype(bool)
        if mass_mearth_cut is not None:
            lower, upper = mass_mearth_cut
            if lower is not None:
                selected &= detected_planets["mass_mearth"] >= lower
            if upper is not None:
                selected &= detected_planets["mass_mearth"] <= upper
        if radius_rearth_cut is not None:
            lower, upper = radius_rearth_cut
            if lower is not None:
                selected &= detected_planets["radius_rearth"] >= lower
            if upper is not None:
                selected &= detected_planets["radius_rearth"] <= upper

        n_detected = int(selected.sum())
        rows.append(
            {
                "source_id": source_id,
                "n_detected": n_detected,
                "yield": n_detected / n_mc,
            }
        )

    yield_table = pd.DataFrame(rows)
    total_yield = float(yield_table["yield"].sum())
    if print_total_yield:
        print(total_yield_label.format(total_yield=total_yield, n_mc=n_mc))

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    else:
        fig = ax.figure

    ax.hist(
        yield_table["yield"],
        bins=bins,
        range=hist_range,
        color=color,
        edgecolor=edgecolor,
        linewidth=linewidth,
        alpha=alpha,
        histtype=histtype,
    )
    ax.set_title(title, fontsize=title_font_size)
    ax.set_xlabel(xlabel, fontsize=label_font_size)
    ax.set_ylabel(ylabel, fontsize=label_font_size)
    ax.tick_params(axis="both", labelsize=tick_font_size)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    if grid:
        ax.grid(True, axis="y", alpha=grid_alpha)

    if show_total_yield:
        ax.text(
            total_yield_position[0],
            total_yield_position[1],
            total_yield_label.format(total_yield=total_yield, n_mc=n_mc),
            transform=ax.transAxes,
            ha=total_yield_ha,
            va=total_yield_va,
            fontsize=text_font_size,
            bbox=total_yield_bbox,
        )

    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    return fig, ax, yield_table
