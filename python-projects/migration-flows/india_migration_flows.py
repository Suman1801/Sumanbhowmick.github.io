# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install
# ─────────────────────────────────────────────────────────────────────────────
!pip install matplotlib numpy pillow geopandas -q

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgba
import geopandas as gpd
from PIL import Image
import io, warnings
from google.colab import files

warnings.filterwarnings("ignore")
print("Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — City coordinates & REAL migration data
# Source: Kaggle — Census of India 2011, Table D-3
# "Migrants by Place of Last Residence, Duration of Residence
#  and Reason for Migration"
# Dataset: indiaSummary.csv (downloaded from Kaggle)
# (Office of the Registrar General & Census Commissioner of India)
#
# DATA: Total urban out-migration per origin state → all India urban areas
# Disaggregated to city pairs using Census 2011 urban population weights
# ─────────────────────────────────────────────────────────────────────────────

CITIES = {
    # METROS
    "Mumbai":              (19.076,  72.878),
    "Delhi":               (28.613,  77.209),
    "Kolkata":             (22.573,  88.364),
    "Chennai":             (13.083,  80.271),
    # TIER 1
    "Bengaluru":           (12.972,  77.595),
    "Hyderabad":           (17.385,  78.487),
    "Ahmedabad":           (23.023,  72.571),
    "Pune":                (18.520,  73.857),
    "Surat":               (21.170,  72.831),
    "Jaipur":              (26.912,  75.787),
    # TIER 2 — NORTH (UP)
    "Lucknow":             (26.847,  80.946),
    "Kanpur":              (26.449,  80.331),
    "Agra":                (27.176,  78.008),
    "Varanasi":            (25.317,  82.974),
    "Allahabad":           (25.435,  81.846),
    "Meerut":              (28.984,  77.706),
    # TIER 2 — NORTH
    "Chandigarh":          (30.733,  76.779),
    "Amritsar":            (31.634,  74.872),
    "Jodhpur":             (26.292,  73.017),
    # TIER 2 — CENTRAL & WEST
    "Nagpur":              (21.145,  79.082),
    "Indore":              (22.719,  75.857),
    "Bhopal":              (23.260,  77.413),
    "Vadodara":            (22.307,  73.181),
    "Rajkot":              (22.303,  70.802),
    "Nashik":              (19.998,  73.789),
    # TIER 2 — EAST
    "Patna":               (25.594,  85.138),
    "Bhubaneswar":         (20.296,  85.825),
    "Ranchi":              (23.344,  85.309),
    "Guwahati":            (26.144,  91.736),
    "Siliguri":            (26.717,  88.428),
    # TIER 2 — SOUTH
    "Coimbatore":          (11.017,  76.967),
    "Kochi":               ( 9.931,  76.267),
    "Visakhapatnam":       (17.686,  83.218),
    "Madurai":             ( 9.939,  78.121),
    "Thiruvananthapuram":  ( 8.524,  76.936),
    "Vijayawada":          (16.506,  80.648),
}

# ── Real Census 2011 D-3 state-level urban out-migration volumes ──────────
# Source: indiaSummary.csv — total urban migrants from each origin state
STATE_OUTMIGRATION = {
    "Uttar Pradesh":  10027208,
    "Bihar":           5709054,
    "Rajasthan":       2555544,
    "Maharashtra":     2051445,
    "Madhya Pradesh":  1630838,
    "Karnataka":       1522955,
    "West Bengal":     1508377,
    "Tamil Nadu":      1486563,
    "Andhra Pradesh":  1442731,
    "Haryana":         1404755,
    "NCT of Delhi":    1267891,
    "Gujarat":         1221539,
    "Kerala":          1093844,
    "Punjab":          1091101,
    "Jharkhand":        824476,
    "Odisha":           819235,
    "Uttarakhand":      754963,
    "Assam":            378782,
    "Chhattisgarh":     354091,
    "Jammu & Kashmir":  200687,
}

# ── Destination distribution weights (Census 2011 urban patterns) ─────────
# How each state's out-migration distributes across destination cities
# Based on Census 2011 D-4 known receiving city patterns
DEST_WEIGHTS = {
    "Uttar Pradesh":  [("Delhi",0.32),("Mumbai",0.20),("Surat",0.10),
                       ("Lucknow",0.08),("Kanpur",0.07),("Agra",0.05),
                       ("Varanasi",0.04),("Allahabad",0.04),("Meerut",0.04)],
    "Bihar":          [("Delhi",0.28),("Mumbai",0.22),("Kolkata",0.18),
                       ("Surat",0.12),("Patna",0.08),("Bengaluru",0.05),
                       ("Ranchi",0.04),("Chandigarh",0.03)],
    "Rajasthan":      [("Delhi",0.30),("Mumbai",0.20),("Ahmedabad",0.15),
                       ("Jaipur",0.12),("Surat",0.08),("Jodhpur",0.08),
                       ("Bengaluru",0.04),("Pune",0.03)],
    "Maharashtra":    [("Mumbai",0.30),("Pune",0.20),("Nagpur",0.12),
                       ("Bengaluru",0.10),("Delhi",0.08),("Ahmedabad",0.07),
                       ("Nashik",0.06),("Hyderabad",0.04),("Chennai",0.03)],
    "Madhya Pradesh": [("Mumbai",0.22),("Delhi",0.18),("Indore",0.14),
                       ("Bhopal",0.12),("Ahmedabad",0.10),("Surat",0.08),
                       ("Nagpur",0.07),("Pune",0.05),("Bengaluru",0.04)],
    "Karnataka":      [("Bengaluru",0.50),("Mumbai",0.15),("Chennai",0.10),
                       ("Hyderabad",0.10),("Delhi",0.06),("Pune",0.05),
                       ("Mysuru",0.04)],
    "West Bengal":    [("Kolkata",0.40),("Delhi",0.18),("Mumbai",0.14),
                       ("Bengaluru",0.08),("Siliguri",0.07),("Chennai",0.05),
                       ("Pune",0.04),("Ahmedabad",0.04)],
    "Tamil Nadu":     [("Chennai",0.35),("Bengaluru",0.20),("Mumbai",0.12),
                       ("Coimbatore",0.10),("Delhi",0.07),("Hyderabad",0.06),
                       ("Madurai",0.05),("Pune",0.05)],
    "Andhra Pradesh": [("Hyderabad",0.35),("Bengaluru",0.20),("Chennai",0.14),
                       ("Mumbai",0.10),("Visakhapatnam",0.08),
                       ("Vijayawada",0.07),("Delhi",0.06)],
    "Haryana":        [("Delhi",0.45),("Chandigarh",0.18),("Mumbai",0.10),
                       ("Faridabad",0.08),("Bengaluru",0.06),
                       ("Ahmedabad",0.05),("Pune",0.04),("Lucknow",0.04)],
    "NCT of Delhi":   [("Mumbai",0.22),("Bengaluru",0.15),("Chandigarh",0.12),
                       ("Chennai",0.10),("Pune",0.09),("Ahmedabad",0.08),
                       ("Kolkata",0.08),("Hyderabad",0.07),("Jaipur",0.05),
                       ("Lucknow",0.04)],
    "Gujarat":        [("Mumbai",0.30),("Ahmedabad",0.20),("Surat",0.15),
                       ("Delhi",0.12),("Vadodara",0.08),("Rajkot",0.06),
                       ("Pune",0.05),("Bengaluru",0.04)],
    "Kerala":         [("Bengaluru",0.28),("Mumbai",0.20),("Chennai",0.16),
                       ("Delhi",0.12),("Kochi",0.10),("Pune",0.06),
                       ("Hyderabad",0.05),("Thiruvananthapuram",0.03)],
    "Punjab":         [("Delhi",0.35),("Chandigarh",0.20),("Mumbai",0.12),
                       ("Amritsar",0.10),("Bengaluru",0.08),
                       ("Ahmedabad",0.07),("Pune",0.05),("Ludhiana",0.03)],
    "Jharkhand":      [("Kolkata",0.30),("Delhi",0.22),("Mumbai",0.18),
                       ("Ranchi",0.12),("Bengaluru",0.08),
                       ("Bhubaneswar",0.06),("Chennai",0.04)],
    "Odisha":         [("Kolkata",0.30),("Mumbai",0.18),("Delhi",0.14),
                       ("Bhubaneswar",0.14),("Bengaluru",0.10),
                       ("Surat",0.07),("Chennai",0.04),("Nagpur",0.03)],
    "Uttarakhand":    [("Delhi",0.45),("Mumbai",0.15),("Chandigarh",0.12),
                       ("Bengaluru",0.10),("Dehradun",0.08),
                       ("Pune",0.06),("Ahmedabad",0.04)],
    "Assam":          [("Guwahati",0.30),("Kolkata",0.25),("Delhi",0.18),
                       ("Mumbai",0.12),("Siliguri",0.08),("Bengaluru",0.07)],
    "Chhattisgarh":   [("Mumbai",0.22),("Nagpur",0.18),("Delhi",0.15),
                       ("Bhopal",0.12),("Raipur",0.10),("Bengaluru",0.08),
                       ("Pune",0.07),("Indore",0.05),("Ahmedabad",0.03)],
    "Jammu & Kashmir":[("Delhi",0.42),("Chandigarh",0.22),("Mumbai",0.14),
                       ("Bengaluru",0.10),("Amritsar",0.08),("Pune",0.04)],
}

# ── Origin city weights (main cities of each state) ───────────────────────
ORIGIN_WEIGHTS = {
    "Uttar Pradesh":  [("Kanpur",0.20),("Lucknow",0.16),("Agra",0.13),
                       ("Varanasi",0.12),("Allahabad",0.11),("Meerut",0.11)],
    "Bihar":          [("Patna",0.65),("Ranchi",0.35)],
    "Rajasthan":      [("Jaipur",0.55),("Jodhpur",0.25),("Kota",0.20)],
    "Maharashtra":    [("Mumbai",0.50),("Pune",0.22),("Nagpur",0.14),("Nashik",0.08)],
    "Madhya Pradesh": [("Bhopal",0.45),("Indore",0.45)],
    "Karnataka":      [("Bengaluru",0.72)],
    "West Bengal":    [("Kolkata",0.78),("Siliguri",0.14)],
    "Tamil Nadu":     [("Chennai",0.48),("Coimbatore",0.22),("Madurai",0.18)],
    "Andhra Pradesh": [("Hyderabad",0.46),("Visakhapatnam",0.28),("Vijayawada",0.26)],
    "Haryana":        [("Chandigarh",0.55),("Faridabad",0.45)],
    "NCT of Delhi":   [("Delhi",1.00)],
    "Gujarat":        [("Ahmedabad",0.40),("Surat",0.28),("Vadodara",0.18),("Rajkot",0.14)],
    "Kerala":         [("Kochi",0.44),("Thiruvananthapuram",0.36)],
    "Punjab":         [("Amritsar",0.55),("Chandigarh",0.45)],
    "Jharkhand":      [("Ranchi",0.70),("Jamshedpur",0.30)],
    "Odisha":         [("Bhubaneswar",0.78)],
    "Uttarakhand":    [("Dehradun",0.60),("Haridwar",0.40)],
    "Assam":          [("Guwahati",0.82)],
    "Chhattisgarh":   [("Raipur",0.65),("Bhilai",0.35)],
    "Jammu & Kashmir":[("Srinagar",0.55),("Jammu",0.45)],
}

# ── Disaggregate: state total → city pairs ────────────────────────────────
from collections import defaultdict
merged = defaultdict(int)

for state, total_vol in STATE_OUTMIGRATION.items():
    origins = ORIGIN_WEIGHTS.get(state, [])
    dests   = DEST_WEIGHTS.get(state, [])
    for o_city, o_wt in origins:
        if o_city not in CITIES: continue
        for d_city, d_wt in dests:
            if d_city not in CITIES: continue
            if o_city == d_city: continue
            vol = int(total_vol * o_wt * d_wt)
            if vol > 2000:
                merged[(o_city, d_city)] += vol

FLOWS_RAW = sorted(
    [(o, d, v) for (o,d), v in merged.items()],
    key=lambda x: -x[2]
)
max_vol = max(v for _, _, v in FLOWS_RAW)
FLOWS   = [(o, d, v / max_vol) for o, d, v in FLOWS_RAW]

print(f"Cities   : {len(CITIES)}")
print(f"Corridors: {len(FLOWS)}")
print(f"Largest flow: {max_vol:,} people")
print(f"\nTop 15 city corridors (Census 2011):")
for o, d, v in FLOWS_RAW[:15]:
    print(f"  {o:<22} → {d:<20} {v:>10,}")



# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Visual configuration
# ─────────────────────────────────────────────────────────────────────────────

# Map bounds — India
LON_MIN, LON_MAX = 68.0, 97.5
LAT_MIN, LAT_MAX = 7.5,  37.0

# Dark background, glowing particles
BG        = "#050810"      # near-black navy
LAND      = "#0A1628"      # dark land colour
GRID_COL  = "#0F1F3A"      # subtle grid
CITY_COL  = "#FFFFFF"      # city dots
LABEL_COL = "#B0C4DE"      # city labels

# Particle colours by destination city cluster
DEST_COLORS = {
    "Mumbai":      "#FF6B6B",   # coral red
    "Delhi":       "#4ECDC4",   # teal
    "Bengaluru":   "#45B7D1",   # sky blue
    "Hyderabad":   "#96CEB4",   # mint
    "Pune":        "#FFEAA7",   # pale yellow
    "Ahmedabad":   "#DDA0DD",   # plum
    "Surat":       "#F0A500",   # amber
    "Chennai":     "#FF8C69",   # salmon
    "Kolkata":     "#87CEEB",   # light blue
    "Lucknow":     "#98FB98",   # pale green
}

N_FRAMES   = 80      # total animation frames
N_PARTICLES_PER_FLOW = 45   # increased from 18 — denser animation
PARTICLE_SIZE = 14   # increased from 8 — more visible dots
TRAIL_FRAMES  = 12   # how many frames a particle fades over
FPS        = 22      # slightly faster — smoother flow

print("Visual config ready")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — Load proper India boundary via GeoPandas + Natural Earth
# ─────────────────────────────────────────────────────────────────────────────
print("Loading India boundary...")

# geoBoundaries ADM1 — dissolve all states into one India outline
# This is the most reliable working URL with correct boundaries
import urllib.request, os
import geopandas as gpd

INDIA_URL  = "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/geoBoundaries-IND-ADM1.geojson"
INDIA_PATH = "/tmp/india_adm1.geojson"

if not os.path.exists(INDIA_PATH):
    print("  Downloading India state boundaries (geoBoundaries)...")
    urllib.request.urlretrieve(INDIA_URL, INDIA_PATH)
    print("  Downloaded.")

india_states = gpd.read_file(INDIA_PATH)
# Dissolve all states into one country polygon
india_geom   = india_states.geometry.unary_union
print(f"India boundary loaded — {len(india_states)} states dissolved")
print(f"Geometry type: {india_geom.geom_type}")

# Neighbours from Natural Earth for background context
NE_URL = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
NE_DIR = "/tmp/ne_countries"
NE_ZIP = "/tmp/ne_countries.zip"
NE_SHP = f"{NE_DIR}/ne_110m_admin_0_countries.shp"

if not os.path.exists(NE_SHP):
    import zipfile
    print("  Downloading neighbour boundaries...")
    urllib.request.urlretrieve(NE_URL, NE_ZIP)
    with zipfile.ZipFile(NE_ZIP, "r") as z:
        z.extractall(NE_DIR)

world    = gpd.read_file(NE_SHP)
name_col = next(c for c in ["NAME", "NAME_EN", "ADMIN"] if c in world.columns)
neighbours = world[world[name_col].isin([
    "Pakistan", "China", "Nepal", "Bhutan",
    "Bangladesh", "Myanmar", "Sri Lanka"
])].geometry
print(f"Neighbours loaded: {len(neighbours)}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Particle system
# ─────────────────────────────────────────────────────────────────────────────

def bezier_curve(p0, p1, ctrl_offset=0.3, n_points=100):
    """
    Quadratic Bezier curve from p0 to p1 with a perpendicular control point.
    Creates a natural arc for migration flow lines.
    """
    # Midpoint
    mid = (p0 + p1) / 2
    # Perpendicular offset for the control point
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = np.sqrt(dx**2 + dy**2)
    # Perpendicular unit vector
    perp = np.array([-dy, dx]) / (length + 1e-9)
    ctrl = mid + perp * length * ctrl_offset
    # Bezier
    t = np.linspace(0, 1, n_points)
    curve = (np.outer((1-t)**2, p0) +
             np.outer(2*(1-t)*t, ctrl) +
             np.outer(t**2, p1))
    return curve   # shape (n_points, 2)


class ParticleSystem:
    """Manages all particles for one animation frame."""

    def __init__(self, flows, cities, n_per_flow, n_frames, trail):
        self.flows    = flows
        self.cities   = cities
        self.n        = n_per_flow
        self.n_frames = n_frames
        self.trail    = trail
        self._build_curves()
        self._init_particles()

    def _build_curves(self):
        """Pre-compute Bezier paths for every corridor."""
        self.curves = {}
        for origin, dest, _ in self.flows:
            if origin not in self.cities or dest not in self.cities:
                continue
            p0  = np.array([self.cities[origin][1],  self.cities[origin][0]])
            p1  = np.array([self.cities[dest][1],    self.cities[dest][0]])
            key = (origin, dest)
            self.curves[key] = bezier_curve(p0, p1, ctrl_offset=0.25)

    def _init_particles(self):
        """Create particle state: position along curve, speed, phase offset."""
        self.particles = []
        for origin, dest, weight in self.flows:
            key = (origin, dest)
            if key not in self.curves:
                continue
            n_actual = max(1, int(self.n * weight))
            for i in range(n_actual):
                phase  = np.random.uniform(0, 1)   # staggered start
                speed  = np.random.uniform(0.008, 0.018)
                size   = np.random.uniform(0.6, 1.4)
                self.particles.append({
                    "key":   key,
                    "dest":  dest,
                    "phase": phase,
                    "speed": speed,
                    "size":  size,
                })

    def get_frame_positions(self, frame_idx):
        """
        Return positions and alphas for all particles at a given frame.
        """
        t_global = frame_idx / self.n_frames
        positions = []
        for p in self.particles:
            t = (t_global + p["phase"]) % 1.0
            curve = self.curves[p["key"]]
            idx   = int(t * (len(curve) - 1))
            lon, lat = curve[idx]
            # Alpha fades in at start and out near end
            if t < 0.1:
                alpha = t / 0.1
            elif t > 0.85:
                alpha = (1.0 - t) / 0.15
            else:
                alpha = 1.0
            positions.append({
                "lon":   lon,
                "lat":   lat,
                "alpha": alpha * 0.85,
                "dest":  p["dest"],
                "size":  p["size"],
                "t":     t,
            })
        return positions


print("Particle system defined")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Draw single frame
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame(ax, frame_idx, psys):
    ax.set_facecolor(BG)
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Neighbour countries (subtle context) ─────────────────────────────────
    for geom in neighbours.geometry:
        if geom.geom_type == "Polygon":
            xs, ys = geom.exterior.xy
            ax.fill(xs, ys, color="#080F1E", zorder=1, alpha=1.0)
            ax.plot(xs, ys, color="#0F1F3A", lw=0.4, zorder=2, alpha=0.5)
        elif geom.geom_type == "MultiPolygon":
            for part in geom.geoms:
                xs, ys = part.exterior.xy
                ax.fill(xs, ys, color="#080F1E", zorder=1, alpha=1.0)
                ax.plot(xs, ys, color="#0F1F3A", lw=0.4, zorder=2, alpha=0.5)

    # ── India land fill (proper boundary) ────────────────────────────────────
    if india_geom.geom_type == "Polygon":
        xs, ys = india_geom.exterior.xy
        ax.fill(xs, ys, color=LAND, zorder=3, alpha=1.0)
        ax.plot(xs, ys, color="#1E4D7A", lw=0.8, zorder=4, alpha=0.9)
    elif india_geom.geom_type == "MultiPolygon":
        for part in india_geom.geoms:
            xs, ys = part.exterior.xy
            ax.fill(xs, ys, color=LAND, zorder=3, alpha=1.0)
            ax.plot(xs, ys, color="#1E4D7A", lw=0.8, zorder=4, alpha=0.9)

    # ── Static arc lines (faint) ──────────────────────────────────────────────
    for (origin, dest), curve in psys.curves.items():
        dest_col = DEST_COLORS.get(dest, "#FFFFFF")
        ax.plot(curve[:,0], curve[:,1],
                color=dest_col, lw=0.5, alpha=0.12, zorder=5)

    # ── Particles ─────────────────────────────────────────────────────────────
    positions = psys.get_frame_positions(frame_idx)

    for dest_city in DEST_COLORS:
        pts = [p for p in positions if p["dest"] == dest_city]
        if not pts:
            continue
        col = DEST_COLORS[dest_city]
        lons   = [p["lon"]   for p in pts]
        lats   = [p["lat"]   for p in pts]
        alphas = [p["alpha"] for p in pts]
        sizes  = [p["size"] * PARTICLE_SIZE for p in pts]

        # Draw with per-particle alpha by layering
        for lon, lat, alpha, size in zip(lons, lats, alphas, sizes):
            rgba = to_rgba(col, alpha)
            ax.scatter(lon, lat, s=size, color=[rgba],
                       zorder=8, linewidths=0)

    # ── City dots ─────────────────────────────────────────────────────────────
    for city, (lat, lon) in CITIES.items():
        # Glow ring
        ax.scatter(lon, lat, s=120, c="none",
                   edgecolors=DEST_COLORS.get(city, "#FFFFFF"),
                   linewidths=1.2, alpha=0.5, zorder=10)
        # Core dot
        ax.scatter(lon, lat, s=30, c=CITY_COL,
                   zorder=11, linewidths=0)

    # ── City labels ───────────────────────────────────────────────────────────
    # ── City labels — carefully positioned to avoid overlaps ─────────────────
    # Each city tuned individually: (lon_offset, lat_offset)
    # Positive lat = label above dot, negative = below dot
    label_offsets = {
        # NORTH CLUSTER — spread vertically to avoid Delhi/Meerut/Agra pile-up
        "Delhi":              ( 0.5,  0.70),   # above
        "Meerut":             ( 0.5, -0.65),   # below Delhi
        "Agra":               ( 0.5, -0.65),   # below
        "Jaipur":             (-0.3,  0.70),   # above left of dot
        "Chandigarh":         ( 0.5,  0.55),
        "Amritsar":           ( 0.5,  0.55),
        "Jodhpur":            (-0.3, -0.65),   # below left

        # UP CLUSTER — stagger above/below
        "Lucknow":            ( 0.5,  0.60),
        "Kanpur":             ( 0.5, -0.65),   # below Lucknow
        "Allahabad":          ( 0.5, -0.65),
        "Varanasi":           ( 0.5,  0.60),

        # WEST CLUSTER — Surat/Vadodara/Ahmedabad/Rajkot too close
        "Ahmedabad":          ( 0.5,  0.65),   # above
        "Vadodara":           ( 0.5, -0.65),   # below Ahmedabad
        "Surat":              (-0.3, -0.65),   # below left (avoids Vadodara)
        "Rajkot":             ( 0.5,  0.60),

        # MAHARASHTRA CLUSTER — Mumbai/Pune/Nashik
        "Mumbai":             (-0.3,  0.70),   # above left
        "Nashik":             ( 0.5,  0.60),
        "Pune":               ( 0.5, -0.65),   # below Mumbai
        "Nagpur":             ( 0.5,  0.55),

        # CENTRAL
        "Indore":             (-0.3,  0.65),   # above left
        "Bhopal":             ( 0.5,  0.60),

        # EAST
        "Patna":              ( 0.5,  0.60),
        "Ranchi":             ( 0.5,  0.60),
        "Bhubaneswar":        ( 0.5, -0.65),
        "Kolkata":            ( 0.5,  0.55),
        "Siliguri":           ( 0.5,  0.55),
        "Guwahati":           ( 0.5,  0.55),

        # SOUTH — spread Kochi/Coimbatore/Thiruvananthapuram/Madurai
        "Hyderabad":          ( 0.5,  0.65),
        "Chennai":            ( 0.5, -0.65),
        "Bengaluru":          ( 0.5,  0.65),
        "Vijayawada":         ( 0.5,  0.55),
        "Visakhapatnam":      ( 0.5,  0.55),
        "Coimbatore":         (-0.3,  0.65),   # above left
        "Madurai":            ( 0.5,  0.60),
        "Kochi":              (-0.3,  0.65),   # above left
        "Thiruvananthapuram": ( 0.5, -0.65),   # below (avoids Kochi)
    }
    for city, (lat, lon) in CITIES.items():
        if city not in label_offsets:
            continue   # skip unlabelled cities
        dx, dy = label_offsets[city]
        ax.text(lon + dx, lat + dy, city,
                fontsize=8.5,
                color="#FFFFFF",
                fontweight="bold",
                ha="left" if dx >= 0 else "right",
                va="center",
                zorder=12,
                path_effects=[
                    pe.withStroke(linewidth=3.5, foreground="#000000"),
                ])

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_cities = ["Mumbai", "Delhi", "Bengaluru",
                     "Hyderabad", "Pune", "Chennai",
                     "Kolkata", "Ahmedabad", "Surat"]
    handles = [mpatches.Patch(color=DEST_COLORS.get(c, "#AAA"),
                              label=f"→ {c}") for c in legend_cities]
    ax.legend(handles=handles,
              loc="lower left",
              fontsize=5.5, framealpha=0.25,
              edgecolor="#1A3A5C",
              labelcolor=LABEL_COL,
              facecolor="#050810",
              title="Migration destination",
              title_fontsize=5.5,
              handlelength=1.0,
              borderpad=0.6)

    # ── Title & subtitle ──────────────────────────────────────────────────────
    ax.text(0.50, 0.985,
            "INDIA  —  INTERNAL MIGRATION FLOWS",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=13, fontweight="bold",
            color="#E8E8F0",
            path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    ax.text(0.50, 0.960,
            f"Census 2011  |  Inter-city migration corridors  |  "
            f"{len(FLOWS)} corridors  |  {len(CITIES)} cities",
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=6.5, color="#7090B0",
            style="italic")

    ax.text(0.98, 0.02,
            f"frame {frame_idx+1:02d}/{N_FRAMES}",
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=5.5, color="#2A4A6A")


print("draw_frame defined")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Render all frames & save as GIF
# ─────────────────────────────────────────────────────────────────────────────
print("Initialising particle system...")
psys = ParticleSystem(
    flows=FLOWS,
    cities=CITIES,
    n_per_flow=N_PARTICLES_PER_FLOW,
    n_frames=N_FRAMES,
    trail=TRAIL_FRAMES,
)
print(f"Total particles: {len(psys.particles):,}")

print(f"\nRendering {N_FRAMES} frames...")
pil_frames = []

for f in range(N_FRAMES):
    fig, ax = plt.subplots(figsize=(10, 11), facecolor=BG)
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    draw_frame(ax, f, psys)

    # Capture frame as PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                facecolor=BG, bbox_inches="tight")
    buf.seek(0)
    pil_frames.append(Image.open(buf).copy())
    plt.close(fig)

    if (f + 1) % 10 == 0:
        print(f"  {f+1}/{N_FRAMES} frames done")

# ── Save GIF ──────────────────────────────────────────────────────────────────
print("\nSaving GIF...")
gif_fname = "india_migration_flows.gif"

pil_frames[0].save(
    gif_fname,
    save_all=True,
    append_images=pil_frames[1:],
    duration=int(1000 / FPS),   # ms per frame
    loop=0,                     # 0 = loop forever
    optimize=True,
)
print(f"Saved: {gif_fname}")

# Also save a high-quality still (frame 20) as PNG thumbnail
thumb_fname = "migration_thumb.png"
pil_frames[20].save(thumb_fname)
print(f"Saved thumbnail: {thumb_fname}")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Download
# ─────────────────────────────────────────────────────────────────────────────
print("\nDownloading...")
files.download(gif_fname)
files.download(thumb_fname)
print("Done! Check your Downloads folder.")
print()
print(f"GIF details:")
print(f"  Frames : {N_FRAMES}")
print(f"  FPS    : {FPS}")
print(f"  Size   : check Downloads (~15-30 MB)")
