# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install
# ─────────────────────────────────────────────────────────────────────────────
!pip install osmnx -q

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import osmnx as ox
import networkx as nx
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.gridspec import GridSpec
from shapely.geometry import Point
import warnings
from datetime import datetime
from google.colab import files

warnings.filterwarnings("ignore")
print("Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Configuration
# ─────────────────────────────────────────────────────────────────────────────
CITIES = {
    "Mumbai":    {"center": (19.0760,  72.8777)},
    "Delhi":     {"center": (28.6139,  77.2090)},
    "Kolkata":   {"center": (22.5726,  88.3639)},
    "Chennai":   {"center": (13.0827,  80.2707)},
    "Bengaluru": {"center": (12.9716,  77.5946)},
    "Hyderabad": {"center": (17.3850,  78.4867)},
    "Ahmedabad": {"center": (23.0225,  72.5714)},
    "Pune":      {"center": (18.5204,  73.8567)},
    "Jaipur":    {"center": (26.9124,  75.7873)},
    "Lucknow":   {"center": (26.8467,  80.9462)},
    "Patna":     {"center": (25.5941,  85.1376)},
    "Bhopal":    {"center": (23.2599,  77.4126)},
}

RADIUS_M = 5000   # fetch radius in metres

HOSPITAL_TAGS = {
    "amenity":    ["hospital", "clinic", "doctors", "pharmacy",
                   "dentist", "health_post", "health_centre"],
    "healthcare": True,
}

# ── Scoring ────────────────────────────────────────────────────────────────
# How many hospitals/clinics within 5km = a "fully served" city centre?
# Based on actual counts from your previous run:
#   Bengaluru 839, Chennai 436, Pune 473, Ahmedabad 441, Hyderabad 521
#   Mumbai 165, Delhi 153, Kolkata 224, Lucknow 137, Bhopal 147
#   Patna 238, Jaipur 408
# Saturation = 400 means "400+ facilities = 100% served" proportionally
SATURATION = 400   # facilities at which score = 100

# ── Light theme colours ────────────────────────────────────────────────────
BG         = "#F7F9FC"   # soft off-white background
PANEL      = "#FFFFFF"   # pure white panels
TVAL       = "#1A1A2E"   # dark navy — titles
TDIM       = "#6B7280"   # muted grey — subtitles

ROAD_MINOR = "#D1D9E6"   # very light — minor streets
ROAD_MAIN  = "#7C8FA6"   # medium slate — primary roads
ROAD_TRUNK = "#3B5272"   # dark navy — trunk/motorway
HOSP_DOT   = "#E63946"   # vivid red — hospitals
CENTRE_COL = "#1D3557"   # deep navy — centre marker

MAP_HALF_LAT = 0.045
MAP_HALF_LON = 0.060

def score_color(s):
    """Colour of score badge based on value."""
    if s >= 75:  return "#2A9D8F"   # teal  — excellent
    if s >= 45:  return "#E9C46A"   # amber — moderate
    return               "#E76F51"  # coral — low

def accessibility_score(count):
    """Linear score 0–100. Saturates at SATURATION facilities."""
    return round(min(count / SATURATION, 1.0) * 100, 1)

print(f"Config ready — {len(CITIES)} cities")
print(f"Saturation target: {SATURATION} facilities = 100 score")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Fetch functions
# ─────────────────────────────────────────────────────────────────────────────
def fetch_network(lat, lon, radius):
    G = ox.graph_from_point((lat, lon), dist=radius, network_type="all")
    nodes, edges = ox.graph_to_gdfs(G)
    return G, nodes, edges

def fetch_hospitals(lat, lon, radius):
    try:
        gdf = ox.features_from_point((lat, lon), tags=HOSPITAL_TAGS, dist=radius)
        gdf = gdf[gdf.geometry.geom_type.isin(
            ["Point", "Polygon", "MultiPolygon"])].copy()
        gdf["geometry"] = gdf.geometry.centroid
        return gdf[["geometry"]].set_crs("EPSG:4326", allow_override=True)
    except Exception:
        return gpd.GeoDataFrame(columns=["geometry"])

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Run analysis for all cities
# ─────────────────────────────────────────────────────────────────────────────
results = []

for name, cfg in CITIES.items():
    print(f"\n--- {name} ---")
    lat, lon = cfg["center"]
    try:
        print("  Network...", end=" ", flush=True)
        G, nodes, edges = fetch_network(lat, lon, RADIUS_M)
        print(f"{len(nodes):,} nodes")

        print("  Hospitals...", end=" ", flush=True)
        hospitals = fetch_hospitals(lat, lon, RADIUS_M)
        count = len(hospitals)
        print(f"{count} found")

        score = accessibility_score(count)
        print(f"  Score: {score}/100")

        results.append(dict(
            name=name, lat=lat, lon=lon,
            edges=edges, hospitals=hospitals,
            count=count, score=score,
        ))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

print("\n" + "="*45)
print(f"  {'City':<14} {'Hospitals':>10} {'Score':>8}")
print("-"*45)
for r in sorted(results, key=lambda x: -x["score"]):
    bar = "█" * int(r["score"] / 5)
    print(f"  {r['name']:<14} {r['count']:>10}   {r['score']:>5.1f}  {bar}")
print("="*45)

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Draw function (light theme)
# ─────────────────────────────────────────────────────────────────────────────
def draw_map(ax, r):
    lat, lon  = r["lat"], r["lon"]
    edges     = r["edges"]
    hospitals = r["hospitals"]

    ax.set_facecolor(BG)

    # Road network — 3-tier visual hierarchy
    main_types  = {"primary", "secondary", "trunk", "motorway",
                   "primary_link", "secondary_link"}
    trunk_types = {"trunk", "motorway"}

    for _, row in edges.iterrows():
        hw = row.get("highway", "")
        if isinstance(hw, list): hw = hw[0] if hw else ""
        if row.geometry is None: continue
        xs = list(row.geometry.xy[0])
        ys = list(row.geometry.xy[1])
        if hw in trunk_types:
            ax.plot(xs, ys, c=ROAD_TRUNK, lw=1.2, alpha=0.95, zorder=3)
        elif hw in main_types:
            ax.plot(xs, ys, c=ROAD_MAIN,  lw=0.7, alpha=0.85, zorder=2)
        else:
            ax.plot(xs, ys, c=ROAD_MINOR, lw=0.3, alpha=0.70, zorder=1)

    # Hospital dots
    if not hospitals.empty:
        ax.scatter(
            hospitals.geometry.x, hospitals.geometry.y,
            s=14, c=HOSP_DOT, alpha=0.80, zorder=5, linewidths=0,
        )

    # Centre marker
    ax.scatter([lon], [lat], s=100, c=CENTRE_COL, zorder=8,
               edgecolors="#FFFFFF", linewidths=1.5)

    # Bounds
    ax.set_xlim(lon - MAP_HALF_LON, lon + MAP_HALF_LON)
    ax.set_ylim(lat - MAP_HALF_LAT, lat + MAP_HALF_LAT)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#D1D9E6")

    # Corner info
    ax.text(0.02, 0.03,
            f"Radius {RADIUS_M//1000} km  |  {r['count']} facilities",
            transform=ax.transAxes, fontsize=6.5,
            color=TDIM, va="bottom", ha="left", zorder=9)




def legend_handles():
    return [
        mpatches.Patch(color=HOSP_DOT,   label="Hospital / Clinic"),
        mpatches.Patch(color=ROAD_TRUNK,  label="Trunk / Motorway"),
        mpatches.Patch(color=ROAD_MAIN,   label="Primary road"),
        mpatches.Patch(color=ROAD_MINOR,  label="Street"),
    ]

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Individual city images
# ─────────────────────────────────────────────────────────────────────────────
city_files = []

for r in results:
    name = r["name"]
    sc   = r["score"]
    print(f"Rendering {name} (score {sc})...")

    fig, ax = plt.subplots(figsize=(10, 9), facecolor=BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01)

    draw_map(ax, r)

    ax.legend(
        handles=legend_handles(),
        loc="upper left", fontsize=7,
        framealpha=0.85, edgecolor="#D1D9E6",
        labelcolor=TVAL, facecolor=PANEL,
        handlelength=1.0,
    )

    fig.text(0.50, 0.945,
             f"{name.upper()}  —  Hospital Accessibility",
             ha="center", va="center", fontsize=20,
             fontweight="bold", color=TVAL,
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    fig.text(0.50, 0.915,
             f"Healthcare facilities within {RADIUS_M//1000} km  |  "
             f"OpenStreetMap  |  {datetime.now().strftime('%B %Y')}",
             ha="center", va="center", fontsize=9,
             color=TDIM, style="italic")

    fname = f"hospital_{name.lower()}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    plt.close(fig)
    city_files.append(fname)
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Combined 3×4 overview + score bar chart
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering combined overview...")

n_cities = len(results)
fig = plt.figure(figsize=(24, 26), facecolor=BG)

gs = GridSpec(
    4, 1, figure=fig,
    height_ratios=[0.5, 8, 8, 3.5],
    hspace=0.08,
    top=0.94, bottom=0.03,
    left=0.03, right=0.97,
)

# ── Row 0: title spacer (handled by fig.text) ──
# ── Rows 1–2: 3×4 map grid ────────────────────────────────────────────────
map_gs1 = gs[1].subgridspec(1, 4, wspace=0.04)
map_gs2 = gs[2].subgridspec(1, 4, wspace=0.04)
map_gs3 = gs[3].subgridspec(1, 1)   # score bar chart

row1 = results[:4]
row2 = results[4:8]
row3 = results[8:]

for i, r in enumerate(row1):
    ax = fig.add_subplot(map_gs1[i])
    draw_map(ax, r)
    ax.set_title(f"{r['name']}  —  {r['count']} facilities",
                 color=TVAL, fontsize=11, fontweight="bold", pad=5)

for i, r in enumerate(row2):
    ax = fig.add_subplot(map_gs2[i])
    draw_map(ax, r)
    ax.set_title(f"{r['name']}  —  {r['count']} facilities",
                 color=TVAL, fontsize=11, fontweight="bold", pad=5)

# ── Row 3: horizontal score bar chart ─────────────────────────────────────
ax_bar = fig.add_subplot(map_gs3[0])
ax_bar.set_facecolor(PANEL)

sorted_r = sorted(results, key=lambda x: -x["score"])
names_s  = [r["name"]  for r in sorted_r]
scores_s = [r["score"] for r in sorted_r]
counts_s = [r["count"] for r in sorted_r]
colors_s = [score_color(s) for s in scores_s]

y = np.arange(len(names_s))
bars = ax_bar.barh(y, scores_s, color=colors_s, alpha=0.88,
                   height=0.6, zorder=2)

# Value labels
for bar, sc, cnt in zip(bars, scores_s, counts_s):
    ax_bar.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height()/2,
                f"{sc}  ({cnt} facilities)",
                va="center", ha="left",
                fontsize=9, color=TVAL, fontweight="bold")

# Reference lines
for xref, label in [(75, "Excellent"), (45, "Moderate")]:
    ax_bar.axvline(xref, color="#9CA3AF", lw=1.0, ls="--", alpha=0.7, zorder=1)
    ax_bar.text(xref + 0.5, len(names_s) - 0.3, label,
                fontsize=7, color="#9CA3AF", va="top")

ax_bar.set_yticks(y)
ax_bar.set_yticklabels(names_s, fontsize=10, color=TVAL, fontweight="bold")
ax_bar.set_xlim(0, 130)
ax_bar.set_xlabel("Accessibility Score  (facilities within 5 km / saturation target)",
                  fontsize=9, color=TDIM)
ax_bar.tick_params(axis="x", colors=TDIM, labelsize=8)
ax_bar.tick_params(axis="y", length=0)
for sp in ["top", "right"]: ax_bar.spines[sp].set_visible(False)
for sp in ["left", "bottom"]: ax_bar.spines[sp].set_color("#D1D9E6")
ax_bar.xaxis.grid(True, color="#E5E9F0", lw=0.5, zorder=0)
ax_bar.set_axisbelow(True)
ax_bar.set_title("Score Comparison — sorted by accessibility",
                 color=TVAL, fontsize=11, pad=6, fontweight="bold")

# Score legend
legend_patches = [
    mpatches.Patch(color="#2A9D8F", label=">= 75  Excellent"),
    mpatches.Patch(color="#E9C46A", label="45-74  Moderate"),
    mpatches.Patch(color="#E76F51", label="< 45   Low"),
]
ax_bar.legend(handles=legend_patches, loc="lower right",
              fontsize=8, framealpha=0.8,
              edgecolor="#D1D9E6", labelcolor=TVAL, facecolor=PANEL)

# ── Third row cities (8–11) — add below row 2 ─────────────────────────────
# Re-use map_gs2 area for row3 cities by adjusting subgridspec
map_gs_r3 = gs[2].subgridspec(2, 4, wspace=0.04, hspace=0.10)
# Already used top half — place row3 in bottom 4 of the 2-row grid
for i, r in enumerate(row3):
    ax = fig.add_subplot(map_gs_r3[1, i])
    draw_map(ax, r)
    ax.set_title(f"{r['name']}  —  {r['count']} facilities",
                 color=TVAL, fontsize=11, fontweight="bold", pad=5)

# ── Titles ─────────────────────────────────────────────────────────────────
fig.text(0.50, 0.965,
         "HOSPITAL ACCESSIBILITY  —  INDIA (12 CITIES)",
         ha="center", va="center", fontsize=22,
         fontweight="bold", color=TVAL,
         path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
fig.text(0.50, 0.950,
         f"Healthcare facilities within {RADIUS_M//1000} km radius  |  "
         f"Score = count / {SATURATION} × 100  |  "
         f"OpenStreetMap  |  {datetime.now().strftime('%B %Y')}",
         ha="center", va="center", fontsize=9,
         color=TDIM, style="italic")

overview_fname = "hospital_overview.png"
fig.savefig(overview_fname, dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.show()
plt.close(fig)
print(f"Saved: {overview_fname}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Download all
# ─────────────────────────────────────────────────────────────────────────────
print("\nDownloading all images...")
for fname in city_files + [overview_fname]:
    files.download(fname)
    print(f"  {fname}")
print("Done!")
