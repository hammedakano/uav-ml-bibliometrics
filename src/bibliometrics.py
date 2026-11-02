"""Bibliometric analysis of UAV and machine learning studies in environmental remote sensing.

Input: a Scopus CSV export of the included studies (data/raw/scopus_full_inclusion.csv).
Output: figures and tables in outputs/.

Run from the repository root:  python src/bibliometrics.py
"""
from __future__ import annotations

import itertools
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

RAW = Path("data/raw/scopus_full_inclusion.csv")
FIG = Path("outputs/figures")
TAB = Path("outputs/tables")

# Synonyms mapped to one keyword so counts are not split across spellings.
SYNONYMS = {
    "unmanned aerial vehicle": "uav", "unmanned aerial vehicles": "uav", "uavs": "uav",
    "unmanned aerial system": "uav", "unmanned aircraft system": "uav", "uas": "uav",
    "drone": "uav", "drones": "uav", "rpas": "uav", "uav remote sensing": "uav",
    "unmanned aerial vehicle (uav)": "uav", "unmanned aerial vehicles (uav)": "uav",
    "rf": "random forest", "random forests": "random forest",
    "svm": "support vector machine", "support vector machines": "support vector machine",
    "cnn": "convolutional neural network", "convolutional neural networks": "convolutional neural network",
    "convolutional neural network (cnn)": "convolutional neural network",
    "deep-learning": "deep learning", "machine-learning": "machine learning",
    "lidar data": "lidar", "light detection and ranging": "lidar",
    "multispectral imagery": "multispectral", "multispectral images": "multispectral",
    "hyperspectral imagery": "hyperspectral", "hyperspectral images": "hyperspectral",
    "ndvi": "vegetation index", "vegetation indices": "vegetation index",
    "remote-sensing": "remote sensing",
    "unmanned aerial systems": "uav", "uav imagery": "uav", "uav images": "uav", "uav-based": "uav",
    "chm": "canopy height model", "canopy height model (chm)": "canopy height model",
    "sfm": "structure from motion", "structure from motion (sfm)": "structure from motion",
    "structure-from-motion": "structure from motion",
    "object-based image analysis": "obia", "geographic object-based image analysis": "obia",
    "geobia": "obia", "biomass": "aboveground biomass", "above-ground biomass": "aboveground biomass",
    "agb": "aboveground biomass",
}
GENERIC = {"uav", "remote sensing", "machine learning"}  # too common to be informative in the network

# Regex groups searched in title, abstract and keywords.
SENSORS = {
    "RGB": r"\brgb\b|visible[- ]light|true[- ]colou?r",
    "Multispectral": r"multi-?spectral",
    "Hyperspectral": r"hyper-?spectral",
    "LiDAR": r"lidar|laser scann",
    "Thermal": r"thermal|infrared therm",
}
METHODS = {
    "Random forest": r"random forest",
    "Support vector machine": r"support vector|\bsvm\b|\bsvr\b",
    "Gradient boosting": r"boost|xgboost|lightgbm|catboost",
    "Neural network (shallow)": r"artificial neural network|\bann\b|multilayer perceptron|\bmlp\b|bp neural",
    "Deep learning": r"deep learning|convolutional|\bcnn\b|u-net|unet|yolo|resnet|transformer|mask r-cnn|faster r-cnn",
}
DOMAINS = {
    "Agriculture": r"\\bcrops?\\b|wheat|maize|\\brice\\b|soybean|\\byield|agricultur|orchard|vineyard|\\bfarm",
    "Forest": r"forest|\\btrees?\\b|\\bstands?\\b|timber|mangrove",
    "Grassland and rangeland": r"grassland|pasture|rangeland|meadow",
    "Wetland and coast": r"wetland|coast|marsh|estuar|seagrass|shoreline",
    "Wildlife": r"wildlife|\\banimals?\\b|\\bbirds?\\b|mammal|koala|\\bdeer\\b|livestock",
}


def normalise_keyword(k: str) -> str:
    k = re.sub(r"\s+", " ", k.strip().lower())
    return SYNONYMS.get(k, k)


def split_keywords(cell) -> list[str]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = [normalise_keyword(k) for k in cell.split(";")]
    return sorted({k for k in out if k})


def flag_groups(text: str, groups: dict[str, str]) -> dict[str, bool]:
    t = text.lower()
    return {name: bool(re.search(pat, t)) for name, pat in groups.items()}


def load(path: Path = RAW) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[["Title", "Year", "Source title", "Cited by", "DOI", "Abstract", "Author Keywords"]].copy()
    d["Year"] = d["Year"].astype("Int64")
    d["Cited by"] = d["Cited by"].fillna(0).astype(int)
    d["keywords"] = d["Author Keywords"].apply(split_keywords)
    d["text"] = (d["Title"].fillna("") + " " + d["Abstract"].fillna("") + " " + d["Author Keywords"].fillna(""))
    for groups in (SENSORS, METHODS, DOMAINS):
        flags = d["text"].apply(lambda t: pd.Series(flag_groups(t, groups)))
        d = pd.concat([d, flags], axis=1)
    return d


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)
    d = load()
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

    # 1. Publications per year
    yr = d["Year"].value_counts().sort_index()
    yr.rename_axis("year").rename("studies").to_csv(TAB / "studies_per_year.csv")
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(yr.index.astype(int), yr.values, color="#2F6F8F")
    ax.set_xlabel("Year"); ax.set_ylabel("Studies")
    ax.set_title("Included studies per year (search run September 2023)")
    fig.tight_layout(); fig.savefig(FIG / "studies_per_year.png", dpi=200); plt.close(fig)

    # 2. Journals
    d["Source title"].value_counts().head(15).rename_axis("journal").rename("studies").to_csv(TAB / "top_journals.csv")

    # 3. Sensor by method
    sens, meth = list(SENSORS), list(METHODS)
    m = pd.DataFrame([[int((d[s] & d[k]).sum()) for k in meth] for s in sens], index=sens, columns=meth)
    m.to_csv(TAB / "sensor_by_method.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    im = ax.imshow(m.values, cmap="Blues")
    ax.set_xticks(range(len(meth)), meth, rotation=30, ha="right"); ax.set_yticks(range(len(sens)), sens)
    for i, j in itertools.product(range(len(sens)), range(len(meth))):
        v = m.values[i, j]
        ax.text(j, i, v, ha="center", va="center", color="white" if v > m.values.max() / 2 else "black")
    ax.set_title("Studies by sensor and method"); fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(FIG / "sensor_by_method.png", dpi=200); plt.close(fig)

    # 4. Method share over time
    recent = d[d["Year"] >= 2016]
    share = recent.groupby("Year")[meth].mean().mul(100).round(1)
    share.to_csv(TAB / "method_share_by_year.csv")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    for k in meth:
        ax.plot(share.index.astype(int), share[k], marker="o", label=k)
    ax.set_ylabel("Studies using method (%)"); ax.set_xlabel("Year")
    ax.legend(frameon=False, fontsize=8); ax.set_title("Methods used over time")
    fig.tight_layout(); fig.savefig(FIG / "method_share_by_year.png", dpi=200); plt.close(fig)

    # 5. Domains and sensors overall
    pd.Series({k: int(d[k].sum()) for k in list(DOMAINS) + sens}).rename("studies").to_csv(TAB / "domain_sensor_counts.csv")

    # 6. Keyword co-occurrence network (top 30 keywords, generic terms removed)
    counts = Counter(k for ks in d["keywords"] for k in ks if k not in GENERIC)
    top = [k for k, _ in counts.most_common(30)]
    pd.Series(dict(counts.most_common(40))).rename("studies").to_csv(TAB / "top_keywords.csv")
    edges = Counter()
    for ks in d["keywords"]:
        ks = [k for k in ks if k in top]
        edges.update(itertools.combinations(sorted(ks), 2))
    g = nx.Graph()
    for k in top:
        g.add_node(k, n=counts[k])
    for (a, b), w in edges.items():
        if w >= 2:
            g.add_edge(a, b, w=w)
    pos = nx.spring_layout(g, seed=7, k=0.9, weight="w")
    fig, ax = plt.subplots(figsize=(8, 7))
    nx.draw_networkx_edges(g, pos, ax=ax, width=[0.4 * g[u][v]["w"] for u, v in g.edges], alpha=0.35, edge_color="grey")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=[25 * g.nodes[n]["n"] for n in g.nodes], node_color="#2F6F8F", alpha=0.85)
    nx.draw_networkx_labels(g, pos, ax=ax, font_size=8)
    ax.set_title("Author keyword co-occurrence (top 30, links in 2 or more studies)"); ax.axis("off")
    ax.margins(0.12)
    fig.tight_layout(); fig.savefig(FIG / "keyword_network.png", dpi=200); plt.close(fig)

    print(f"{len(d)} studies, {int(d['Year'].min())} to {int(d['Year'].max())}")
    print(m)
    print(share.tail(4))


if __name__ == "__main__":
    main()
