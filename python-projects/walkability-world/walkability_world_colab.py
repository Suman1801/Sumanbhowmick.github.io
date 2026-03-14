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

# Mumbai  — CST / Fort area
# London  — Covent Garden (dense central London)
# Dubai   — Downtown Dubai / Burj Khalifa area
# Paris   — Le Marais (dense central Paris)

CITIES = {
    "Mumbai": {"center": (18.9388,  72.8354)},   # Fort / CST
    "London": {"center": (51.5117,  -0.1240)},   # Covent Garden
    "Dubai":  {"center": (25.1972,  55.2744)},   # Downtown Dubai
    "Paris":  {"center": (48.8566,   2.3522)},   # Le Marais
}

WALK_DIST_M = 1200   # ~15 min walk at 5 km/h

POI_TAGS = {
    "School":      {"amenity": ["school", "kindergarten", "university", "college"]},
    "Shop":        {"shop": True},
    "Clinic":      {"amenity": ["clinic", "hospital", "pharmacy", "doctors", "dentist"]},
    "Gym":         {"leisure": ["fitness_centre", "sports_centre", "swimming_pool"]},
    "Restaurant":  {"amenity": ["restaurant", "cafe", "fast_food", "bar", "pub", "food_court"]},
    "Park":        {"leisure": ["park", "garden", "nature_reserve", "playground"]},
    "Transit":     {
        "public_transport": ["stop_position", "station", "platform"],
        "railway":          ["station", "subway_entrance", "tram_stop"],
        "highway":          ["bus_stop"],
        "amenity":          ["bus_station"],
    },
    "Supermarket": {"shop": ["supermarket", "convenience", "grocery", "greengrocer"]},
    "Hotel":       {"tourism": ["hotel", "hostel", "guest_house"]},
    "ATM":         {"amenity": ["atm", "bank"]},
}

COLORS = {
    "School":      "#4FC3F7",
    "Shop":        "#FFB74D",
    "Clinic":      "#EF5350",
    "Gym":         "#66BB6A",
    "Restaurant":  "#AB47BC",
    "Park":        "#26A69A",
    "Transit":     "#FFA726",
    "Supermarket": "#EC407A",
    "Hotel":       "#FF8A65",
    "ATM":         "#80CBC4",
}

SCORE_WEIGHTS = {
    "Supermarket": 1.2, "Transit": 1.2, "Restaurant": 1.0,
    "Park":        1.0, "Clinic":  1.0, "Shop":       0.9,
    "Gym":         0.7, "School":  0.8, "Hotel":      0.5, "ATM": 0.6,
}

# Raised saturation targets — calibrated so a world-class city centre
# scores 70-90, not 100, keeping real differentiation between cities
SATURATION = {
    "Supermarket": 15,  "Transit": 50,  "Restaurant": 150,
    "Park":        20,  "Clinic":  25,  "Shop":       200,
    "Gym":         20,  "School":  20,  "Hotel":      30,  "ATM": 25,
}

BG    = "#0A0A0F"
PANEL = "#0D0D1A"
TVAL  = "#E8E8F0"
TDIM  = "#8888AA"

MAP_HALF_LAT = 0.015
MAP_HALF_LON = 0.020

def score_color(s):
    return "#2ECC71" if s >= 70 else "#F39C12" if s >= 45 else "#E74C3C"

print(f"Config ready — cities: {', '.join(CITIES)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Analysis functions
# ─────────────────────────────────────────────────────────────────────────────
def fetch_network(lat, lon, dist):
    G = ox.graph_from_point((lat, lon), dist=dist, network_type="walk")
    G = ox.add_edge_speeds(G)
    G = ox.add_edge_travel_times(G)
    return G

def fetch_pois(lat, lon, dist):
    out = {}
    for cat, tags in POI_TAGS.items():
        try:
            gdf = ox.features_from_point((lat, lon), tags=tags, dist=dist)
            gdf = gdf[gdf.geometry.geom_type.isin(
                ["Point","Polygon","MultiPolygon","LineString"])].copy()
            gdf["geometry"] = gdf.geometry.centroid
            out[cat] = gdf[["geometry"]].copy()
        except Exception:
            out[cat] = gpd.GeoDataFrame(columns=["geometry"])
    return out

def isochrone_nodes(G, lat, lon, dist):
    origin = ox.nearest_nodes(G, lon, lat)
    lengths = nx.single_source_dijkstra_path_length(
        G, origin, cutoff=dist, weight="length")
    return set(lengths.keys()), origin

def count_pois(G, reachable, poi_dict):
    pts = [Point(G.nodes[n]["x"], G.nodes[n]["y"]) for n in reachable]
    if len(pts) < 3:
        return {c: 0 for c in poi_dict}
    hull = gpd.GeoSeries(pts, crs="EPSG:4326").unary_union.convex_hull
    return {
        cat: int(gdf.set_crs("EPSG:4326", allow_override=True)
                    .geometry.within(hull).sum())
        for cat, gdf in poi_dict.items()
    }

def walkability_score(counts):
    tw = sum(SCORE_WEIGHTS.values())
    s  = sum(
        min(1.0, counts.get(c, 0) / SATURATION[c]) * w
        for c, w in SCORE_WEIGHTS.items()
    )
    return round(s / tw * 100, 1)

print("Functions defined")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Run analysis
# ─────────────────────────────────────────────────────────────────────────────
results = []
dist = WALK_DIST_M + 300

for name, cfg in CITIES.items():
    print(f"\n--- {name} ---")
    lat, lon = cfg["center"]
    try:
        print("  Street network...", end=" ", flush=True)
        G = fetch_network(lat, lon, dist)
        nodes, edges = ox.graph_to_gdfs(G)
        print(f"{len(nodes):,} nodes, {len(edges):,} edges")

        print("  POIs...")
        poi_dict = fetch_pois(lat, lon, dist)
        for cat, gdf in poi_dict.items():
            if len(gdf) > 0:
                print(f"    {cat}: {len(gdf):,}")

        print("  Isochrone...", end=" ", flush=True)
        reachable, origin = isochrone_nodes(G, lat, lon, WALK_DIST_M)
        print(f"{len(reachable):,} reachable nodes")

        counts = count_pois(G, reachable, poi_dict)
        sc     = walkability_score(counts)
        print(f"  Score: {sc}/100")
        print(f"  Key counts: Shop={counts.get('Shop',0)} "
              f"Restaurant={counts.get('Restaurant',0)} "
              f"Transit={counts.get('Transit',0)} "
              f"Park={counts.get('Park',0)}")

        results.append(dict(
            name=name, G=G, poi_dict=poi_dict,
            reachable=reachable, origin=origin,
            counts=counts, score=sc, lat=lat, lon=lon,
        ))
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {e}")

print("\n" + "="*50)
print(f"  {'City':<12}  {'Score':>8}")
print("-"*50)
for r in sorted(results, key=lambda x: -x["score"]):
    bar = "#" * int(r["score"] / 5)
    print(f"  {r['name']:<12}  {r['score']:>5.1f} / 100  {bar}")
print("="*50)
print("\nNote: Scores reflect OSM data density as much as real walkability.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Draw functions
# ─────────────────────────────────────────────────────────────────────────────
def draw_map(ax, r):
    G, reachable = r["G"], r["reachable"]
    lat, lon = r["lat"], r["lon"]
    ax.set_facecolor(BG)

    for u, v, _ in G.edges(data=True):
        xs = [G.nodes[u]["x"], G.nodes[v]["x"]]
        ys = [G.nodes[u]["y"], G.nodes[v]["y"]]
        if u in reachable and v in reachable:
            ax.plot(xs, ys, c="#5555FF", lw=1.0, alpha=0.9, zorder=2)
        else:
            ax.plot(xs, ys, c="#1C1C3A", lw=0.4, alpha=0.6, zorder=1)

    for cat, gdf in r["poi_dict"].items():
        if not gdf.empty:
            ax.scatter(gdf.geometry.x, gdf.geometry.y,
                       s=10, c=COLORS[cat], alpha=0.75, zorder=3, linewidths=0)

    ax.scatter([lon], [lat], s=160, c="#FFFFFF", zorder=6,
               edgecolors="#FFD700", linewidths=2.2)

    ax.set_xlim(lon - MAP_HALF_LON, lon + MAP_HALF_LON)
    ax.set_ylim(lat - MAP_HALF_LAT, lat + MAP_HALF_LAT)
    ax.set_aspect("equal")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for sp in ax.spines.values():
        sp.set_edgecolor("#2A2A4A")
    ax.text(0.02, 0.03, f"Centre  |  Walk radius {WALK_DIST_M}m",
            transform=ax.transAxes, fontsize=6.5,
            color=TDIM, va="bottom", ha="left", zorder=8)


def draw_bars(ax, r):
    ax.set_facecolor(PANEL)
    cats   = list(SCORE_WEIGHTS)
    values = [r["counts"].get(c, 0) for c in cats]
    sats   = [SATURATION[c] for c in cats]
    y      = np.arange(len(cats))

    ax.barh(y, sats, color="#FFFFFF", alpha=0.07, height=0.65, zorder=1)
    bars = ax.barh(y, values, color=[COLORS[c] for c in cats],
                   alpha=0.87, height=0.65, zorder=2)

    mx = max(max(values) if values else 1, max(sats))
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_width() + mx * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    str(val), va="center", ha="left",
                    fontsize=9, color="#CCCCDD", fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=9, color="#AAAACC")
    ax.set_xlim(0, mx * 1.32)
    ax.set_xlabel("# POIs reachable on foot", fontsize=8, color=TDIM)
    ax.tick_params(axis="x", colors="#555577", labelsize=8)
    ax.tick_params(axis="y", length=0)
    for s in ["top","right"]: ax.spines[s].set_visible(False)
    for s in ["left","bottom"]: ax.spines[s].set_color("#222244")
    ax.xaxis.grid(True, color="#1E1E3A", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.01, "light bar = saturation target",
            transform=ax.transAxes, fontsize=6.5,
            color=TDIM, ha="right", va="bottom", style="italic")


def poi_legend_handles():
    return [mpatches.Patch(color=COLORS[c], label=c) for c in COLORS]

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Individual city images
# ─────────────────────────────────────────────────────────────────────────────
city_files = []

for r in results:
    name = r["name"]
    sc   = r["score"]
    col  = score_color(sc)
    print(f"Rendering {name}...")

    fig = plt.figure(figsize=(14, 7), facecolor=BG)
    gs  = GridSpec(1, 2, figure=fig,
                   left=0.03, right=0.97, top=0.82, bottom=0.08,
                   wspace=0.10, width_ratios=[46, 54])

    ax_map = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    draw_map(ax_map, r)
    draw_bars(ax_bar, r)

    ax_map.legend(
        handles=poi_legend_handles(),
        loc="upper left", fontsize=7, ncol=2,
        framealpha=0.30, edgecolor="#333355",
        labelcolor=TVAL, facecolor="#0D0D1A",
        title="POI types", title_fontsize=7,
        handlelength=0.9, handleheight=0.8,
    )

    # Score badge — top-right of figure (not axes)
    badge_x, badge_y = 0.97, 0.93
    badge_w, badge_h = 0.085, 0.10
    fig.patches.append(mpatches.FancyBboxPatch(
        (badge_x - badge_w, badge_y - badge_h),
        badge_w, badge_h,
        boxstyle="round,pad=0.01",
        transform=fig.transFigure,
        facecolor=col, alpha=0.93,
        zorder=20, edgecolor="none", clip_on=False,
    ))
    fig.text(badge_x - badge_w / 2, badge_y - badge_h * 0.40,
             f"{sc}", ha="center", va="center",
             fontsize=18, fontweight="bold", color="white",
             transform=fig.transFigure, zorder=21)
    fig.text(badge_x - badge_w / 2, badge_y - badge_h * 0.82,
             "/ 100", ha="center", va="center",
             fontsize=8, color="white", alpha=0.75,
             transform=fig.transFigure, zorder=21)

    fig.text(0.50, 0.945,
             f"{name.upper()}  -  Urban Walkability",
             ha="center", va="center", fontsize=20,
             fontweight="bold", color=TVAL,
             path_effects=[pe.withStroke(linewidth=4, foreground=BG)])
    fig.text(0.50, 0.912,
             f"Daily-life amenities within {WALK_DIST_M}m on foot  |  "
             f"OpenStreetMap / OSMnx  |  {datetime.now().strftime('%B %Y')}",
             ha="center", va="center", fontsize=9,
             color=TDIM, style="italic")

    ax_bar.set_title("Walkability breakdown",
                     color="#9999BB", fontsize=10, pad=6)

    fname = f"walkability_{name.lower().replace(' ','_')}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.show()
    plt.close(fig)
    city_files.append(fname)
    print(f"  Saved: {fname}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Combined overview image
# ─────────────────────────────────────────────────────────────────────────────
print("\nRendering combined overview...")
n   = len(results)
fig = plt.figure(figsize=(max(22, 5.8*n), 22), facecolor=BG)

gs = GridSpec(3, 1, figure=fig, hspace=0.08,
              top=0.925, bottom=0.04, left=0.04, right=0.96,
              height_ratios=[4.5, 2.8, 1.8])
map_gs = gs[0].subgridspec(1, n, wspace=0.05)
bar_gs = gs[1].subgridspec(1, n, wspace=0.18)

for i, r in enumerate(results):
    ax_m = fig.add_subplot(map_gs[i])
    draw_map(ax_m, r)
    ax_m.set_title(r["name"], color=TVAL, fontsize=13, fontweight="bold", pad=5)

    sc  = r["score"]
    col = score_color(sc)
    ax_m.add_patch(mpatches.FancyBboxPatch(
        (0.73, 0.04), 0.24, 0.11,
        boxstyle="round,pad=0.01",
        transform=ax_m.transAxes,
        facecolor=col, alpha=0.93, zorder=10, edgecolor="none"))
    ax_m.text(0.85, 0.096, f"{sc}", transform=ax_m.transAxes,
              ha="center", va="center", fontsize=11,
              fontweight="bold", color="white", zorder=11)
    ax_m.text(0.85, 0.052, "/ 100", transform=ax_m.transAxes,
              ha="center", va="center", fontsize=6.5,
              color="white", alpha=0.75, zorder=11)

    draw_bars(fig.add_subplot(bar_gs[i]), r)

# Score comparison
ax_sc = fig.add_subplot(gs[2])
ax_sc.set_facecolor(PANEL)
names  = [r["name"]  for r in results]
scores = [r["score"] for r in results]
x      = np.arange(len(names))
bars   = ax_sc.bar(x, scores, color=[score_color(s) for s in scores],
                   alpha=0.9, width=0.5, zorder=2, edgecolor=PANEL)
for bar, sc in zip(bars, scores):
    ax_sc.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1,
               f"{sc}", ha="center", va="bottom",
               fontsize=15, fontweight="bold", color=TVAL)
for yref, label, alpha in [(70,"Very Walkable",0.45),(45,"Walkable",0.3)]:
    ax_sc.axhline(yref, color="#FFF", lw=0.8, alpha=alpha, ls="--", zorder=1)
    ax_sc.text(n-0.42, yref+1, label, fontsize=7, color="#FFF",
               alpha=alpha+0.1, va="bottom", ha="right")
ax_sc.set_xticks(x)
ax_sc.set_xticklabels(names, fontsize=13, color="#CCCCDD", fontweight="bold")
ax_sc.set_ylim(0, 112)
ax_sc.set_ylabel("Score", fontsize=10, color=TDIM)
ax_sc.set_title("Mumbai vs World Cities  -  Walkability Comparison",
                color="#CCCCDD", fontsize=13, pad=8, fontweight="bold")
ax_sc.tick_params(axis="y", colors="#555577", labelsize=8)
ax_sc.tick_params(axis="x", length=0)
for s in ["top","right"]: ax_sc.spines[s].set_visible(False)
for s in ["left","bottom"]: ax_sc.spines[s].set_color("#222244")
ax_sc.yaxis.grid(True, color="#1A1A30", lw=0.5, zorder=0)
ax_sc.set_axisbelow(True)
ax_sc.legend(
    handles=[
        mpatches.Patch(color="#2ECC71", label=">= 70  Excellent"),
        mpatches.Patch(color="#F39C12", label="45-69  Good"),
        mpatches.Patch(color="#E74C3C", label="< 45   Needs improvement"),
    ],
    loc="upper left", fontsize=8.5, framealpha=0.15,
    edgecolor="#333355", labelcolor="#CCCCDD", facecolor=PANEL,
)

fig.legend(handles=poi_legend_handles(),
           loc="upper right", bbox_to_anchor=(0.99, 0.965),
           fontsize=8, ncol=2, framealpha=0.15,
           edgecolor="#333355", labelcolor=TVAL, facecolor=PANEL,
           title="POI categories", title_fontsize=8,
           handlelength=1, handleheight=0.9)

fig.text(0.5, 0.967, "URBAN WALKABILITY  -  MUMBAI vs WORLD",
         ha="center", va="center", fontsize=26, fontweight="bold", color=TVAL,
         path_effects=[pe.withStroke(linewidth=5, foreground=BG)])
fig.text(0.5, 0.953,
         f"Daily-life amenities within {WALK_DIST_M}m on foot  |  "
         f"OpenStreetMap / OSMnx  |  {datetime.now().strftime('%B %Y')}",
         ha="center", va="center", fontsize=10, color=TDIM, style="italic")

combined_fname = "walkability_overview.png"
fig.savefig(combined_fname, dpi=150, bbox_inches="tight",
            facecolor=BG, edgecolor="none")
plt.show()
plt.close(fig)
print(f"Saved: {combined_fname}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Download all
# ─────────────────────────────────────────────────────────────────────────────
print("\nDownloading all images...")
for fname in city_files + [combined_fname]:
    files.download(fname)
    print(f"  Downloading: {fname}")
print("Done! Check your Downloads folder.")
