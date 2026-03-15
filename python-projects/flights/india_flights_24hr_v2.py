# ─────────────────────────────────────────────────────────────────────────────
# CELL 1 — Install
# ─────────────────────────────────────────────────────────────────────────────
!pip install matplotlib pandas pillow geopandas -q

# ─────────────────────────────────────────────────────────────────────────────
# CELL 2 — Imports
# ─────────────────────────────────────────────────────────────────────────────
import gc, io, os, shutil, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgba
import geopandas as gpd
import urllib.request
from PIL import Image
from google.colab import files

warnings.filterwarnings("ignore")
plt.rcParams["font.family"]  = "DejaVu Sans"
print("Imports OK")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 3 — Config
# ─────────────────────────────────────────────────────────────────────────────

# Animation — smooth and clear
TIME_STEP   = 5           # minutes per frame → 288 frames (very smooth)
FRAMES      = 1440 // TIME_STEP
TRAIL_STEPS = 6           # longer trail for smooth streaks
DPI         = 90
FIG_W, FIG_H = 9, 10

# Map bounds
LON_MIN, LON_MAX = 66.0, 99.0
LAT_MIN, LAT_MAX =  6.0, 38.0

# ── Map colour theme — dark navy with high contrast ─────────────────────────
BG        = "#030610"     # deep navy black
STATE_COL = "#070E20"     # dark navy state fills
BORDER    = "#0D2040"     # visible state borders
OUTER_BDR = "#00D4FF"     # bright cyan boundary
TVAL      = "#FFFFFF"     # pure white titles
TDIM      = "#4A6A9A"     # muted blue subtitles

# ── Airline colours — maximum brightness and distinction ───────────────────
AIRLINE_COLORS = {
    "IndiGo":        "#886EFF",   # bright lavender indigo
    "SpiceJet":      "#FF3B3B",   # vivid red
    "Air India":     "#FFB300",   # bright amber
    "GoAir":         "#00E5FF",   # electric cyan
    "AirAsia India": "#FF1493",   # deep pink
    "Vistara":       "#DA70D6",   # orchid purple
    "Jet Airways":   "#FFE600",   # bright yellow
    "TruJet":        "#00FF7F",   # spring green
    "Jetlite":       "#FF7F00",   # vivid orange
}
DEFAULT_COLOR = "#4A6A9A"

# ── Major hub airports — get larger dots + labels ──────────────────────────
HUB_AIRPORTS = {
    "Delhi", "Mumbai", "Bengaluru", "Chennai",
    "Hyderabad", "Kolkata", "Pune", "Ahmedabad",
    "Goa", "Kochi", "Jaipur", "Lucknow",
    "Patna", "Guwahati", "Srinagar",
}

# Short display names for labels
LABEL_NAMES = {
    "Delhi": "Delhi", "Mumbai": "Mumbai", "Bengaluru": "Bengaluru",
    "Chennai": "Chennai", "Hyderabad": "Hyderabad", "Kolkata": "Kolkata",
    "Pune": "Pune", "Ahmedabad": "Ahmedabad", "Goa": "Goa",
    "Kochi": "Kochi", "Jaipur": "Jaipur", "Lucknow": "Lucknow",
    "Patna": "Patna", "Guwahati": "Guwahati", "Srinagar": "Srinagar",
}

# ── Airport coordinates ─────────────────────────────────────────────────────
AIRPORTS = {
    "Agartala":(23.887,91.240),"Agra":(27.156,77.961),
    "Ahmedabad":(23.074,72.635),"Aizwal":(23.841,92.820),
    "Allahabad":(25.440,81.734),"Amritsar":(31.710,74.797),
    "Aurangabad":(19.863,75.398),"Bagdogra":(26.681,88.329),
    "Belgaum":(15.859,74.618),"Bengaluru":(13.198,77.706),
    "Bhopal":(23.288,77.337),"Bhubaneswar":(20.244,85.818),
    "Calicut":(11.137,75.955),"Chandigarh":(30.674,76.789),
    "Chennai":(12.990,80.169),"Coimbatore":(11.030,77.043),
    "Darbhanga":(26.192,85.917),"Dehradun":(30.190,78.180),
    "Delhi":(28.556,77.100),"Dibrugarh":(27.484,95.017),
    "Dimapur":(25.884,93.771),"Gaya":(24.744,84.951),
    "Goa":(15.381,73.831),"Guwahati":(26.106,91.586),
    "Gwalior":(26.293,78.228),"Hubli":(15.362,75.085),
    "Hyderabad":(17.231,78.430),"Imphal":(24.760,93.897),
    "Indore":(22.722,75.801),"Jabalpur":(23.178,80.052),
    "Jaipur":(26.824,75.812),"Jammu":(32.689,74.837),
    "Jodhpur":(26.251,73.049),"Jorhat":(26.732,94.176),
    "Kadapa":(14.513,78.772),"Kandla":(23.113,70.100),
    "Kannur":(11.919,75.547),"Kanpur":(26.404,80.410),
    "Keshod":(21.317,70.270),"Khajuraho":(24.817,79.919),
    "Kochi":(9.945,76.271),"Kolhapur":(16.665,74.289),
    "Kolkata":(22.652,88.446),"Leh":(34.136,77.547),
    "Lilabari":(27.295,94.098),"Lucknow":(26.761,80.889),
    "Madurai":(9.835,78.093),"Mangalore":(12.961,74.890),
    "Mumbai":(19.090,72.866),"Mysore":(12.231,76.652),
    "Nagpur":(21.092,79.047),"Patna":(25.591,85.088),
    "Pondicherry":(11.968,79.812),"Porbandar":(21.649,69.657),
    "Port Blair":(11.641,92.730),"Pune":(18.582,73.920),
    "Raipur":(21.180,81.739),"Rajahmundry":(17.110,81.818),
    "Rajkot":(22.309,70.780),"Ranchi":(23.314,85.322),
    "Salem":(11.783,78.066),"Shillong":(25.704,91.979),
    "Silchar":(24.913,92.979),"Srinagar":(33.987,74.774),
    "Surat":(21.114,72.742),
    "Thiruvananthapuram":(8.482,76.920),
    "Tiruchirappalli":(10.765,78.709),
    "Tirupati":(13.633,79.543),"Tuticorin":(8.724,78.026),
    "Udaipur":(24.618,73.896),"Vadodara":(22.336,73.226),
    "Varanasi":(25.452,82.859),"Vijayawada":(16.530,80.798),
    "Visakhapatnam":(17.721,83.225),
}

print(f"Config OK — {FRAMES} frames, {len(AIRPORTS)} airports")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 4 — Load & prepare data — SINGLE DAY: Friday 01 March 2019
# ─────────────────────────────────────────────────────────────────────────────
from datetime import datetime

TARGET_DATE = datetime(2019, 3, 1)
DAY_NAME    = "Friday"

df_raw = pd.read_csv("Flight_Schedule_without_missing.csv")
df_raw.columns = [c.strip() for c in df_raw.columns]

# Only rows with arrival time
df_raw = df_raw.dropna(subset=["scheduledDepartureTime","scheduledArrivalTime"])
df_raw = df_raw[df_raw["scheduledArrivalTime"].str.strip() != ""]

# Only airports in coordinate dict
df_raw = df_raw[df_raw["origin"].isin(AIRPORTS) & df_raw["destination"].isin(AIRPORTS)]

# Only flights valid on 01 March 2019
def parse_date(s):
    try: return datetime.strptime(str(s).strip(), "%d-%m-%Y")
    except: return None

def parse_date(s):
    """Parse DD-MM-YYYY dates robustly."""
    try:
        return pd.to_datetime(str(s).strip(), dayfirst=True)
    except:
        return None

df_raw["vf"] = df_raw["validFrom"].apply(parse_date)
df_raw["vt"] = df_raw["validTo"].apply(parse_date)

# Debug — show date range
print(f"validFrom range: {df_raw['vf'].min()} to {df_raw['vf'].max()}")
print(f"validTo range:   {df_raw['vt'].min()} to {df_raw['vt'].max()}")

# Must operate on Friday AND be valid on 01 Mar 2019
mask = (
    df_raw["dayOfWeek"].str.contains(DAY_NAME, na=False) &
    df_raw["vf"].notna() & df_raw["vt"].notna() &
    (df_raw["vf"] <= TARGET_DATE) & (df_raw["vt"] >= TARGET_DATE)
)
print(f"Rows matching date filter: {mask.sum():,}")
df_raw = df_raw[mask]

def t2m(t):
    try:
        h, m = str(t).strip().split(":")
        return int(h)*60 + int(m)
    except:
        return None

df_raw["dep_min"] = df_raw["scheduledDepartureTime"].apply(t2m)
df_raw["arr_min"] = df_raw["scheduledArrivalTime"].apply(t2m)
df_raw = df_raw.dropna(subset=["dep_min","arr_min"]).copy()
df_raw["dep_min"] = df_raw["dep_min"].astype(int)
df_raw["arr_min"] = df_raw["arr_min"].astype(int)
df_raw.loc[df_raw["arr_min"] < df_raw["dep_min"], "arr_min"] += 1440
df_raw = df_raw.drop_duplicates(subset=["origin","destination","dep_min","arr_min"])

df = df_raw.copy()
# Rename TestIndigo → IndiGo (dataset quirk)
df["airline"] = df["airline"].replace("TestIndigo", "IndiGo")
df["orig_lat"] = df["origin"].map(lambda x: AIRPORTS[x][0])
df["orig_lon"] = df["origin"].map(lambda x: AIRPORTS[x][1])
df["dest_lat"] = df["destination"].map(lambda x: AIRPORTS[x][0])
df["dest_lon"] = df["destination"].map(lambda x: AIRPORTS[x][1])
df["color"]    = df["airline"].map(AIRLINE_COLORS).fillna(DEFAULT_COLOR)
df["duration"] = df["arr_min"] - df["dep_min"]
df = df[(df["duration"] >= 30) & (df["duration"] <= 480)]

def haversine(la1, lo1, la2, lo2):
    R=6371; la1,lo1,la2,lo2=map(np.radians,[la1,lo1,la2,lo2])
    a=np.sin((la2-la1)/2)**2+np.cos(la1)*np.cos(la2)*np.sin((lo2-lo1)/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

df["dist_km"] = haversine(df["orig_lat"],df["orig_lon"],df["dest_lat"],df["dest_lon"])

print(f"Date  : {TARGET_DATE.strftime('%d %B %Y')} ({DAY_NAME})")
print(f"Flights: {len(df):,}")
print(f"\nBy airline:")
print(df.groupby("airline").size().sort_values(ascending=False).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# CELL 5 — India boundary + static arcs
# ─────────────────────────────────────────────────────────────────────────────
INDIA_PATH = "/tmp/india_adm1.geojson"
if not os.path.exists(INDIA_PATH):
    print("Downloading boundary...")
    urllib.request.urlretrieve(
        "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/IND/ADM1/geoBoundaries-IND-ADM1.geojson",
        INDIA_PATH)
india_states = gpd.read_file(INDIA_PATH)
india_union  = india_states.geometry.unary_union

def get_rings(g):
    if g.geom_type == "Polygon": return [list(g.exterior.coords)]
    return [list(p.exterior.coords) for p in g.geoms]
INDIA_RINGS = get_rings(india_union)

def make_arc(olon, olat, dlon, dlat, n=30):   # reduced from 60
    pts = []
    p0=np.array([olon,olat]); p1=np.array([dlon,dlat])
    mid=(p0+p1)/2; dx,dy=p1-p0; L=np.hypot(dx,dy)
    ctrl=mid+np.array([-dy,dx])/(L+1e-9)*L*0.22
    for i in range(n+1):
        t=i/n
        pts.append((1-t)**2*p0+2*(1-t)*t*ctrl+t**2*p1)
    return np.array(pts)

# Top 50 busiest routes as static arcs
top_routes = (df.groupby(["orig_lon","orig_lat","dest_lon","dest_lat"])
                .size().sort_values(ascending=False).head(50))
STATIC_ARCS = []
for (olon,olat,dlon,dlat), cnt in top_routes.items():
    arc = make_arc(olon,olat,dlon,dlat)
    STATIC_ARCS.append((arc, cnt))
max_cnt = max(c for _,c in STATIC_ARCS)

print(f"Boundary + {len(STATIC_ARCS)} route arcs ready")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 6 — Bezier + active flights
# ─────────────────────────────────────────────────────────────────────────────
def bezier_pos(lon0,lat0,lon1,lat1,t):
    p0=np.array([lon0,lat0]); p1=np.array([lon1,lat1])
    mid=(p0+p1)/2; dx,dy=p1-p0; L=np.hypot(dx,dy)
    ctrl=mid+np.array([-dy,dx])/(L+1e-9)*L*0.22
    pos  = (1-t)**2*p0+2*(1-t)*t*ctrl+t**2*p1
    deriv= 2*(1-t)*(ctrl-p0)+2*t*(p1-ctrl)
    angle= np.degrees(np.arctan2(deriv[1], deriv[0]))
    return float(pos[0]), float(pos[1]), float(angle)

def active_flights(cur_min, trail_step=0):
    t_now = cur_min - trail_step*TIME_STEP
    mask  = (df["dep_min"] <= t_now) & (df["arr_min"] > t_now)
    out   = []
    for _, r in df[mask].iterrows():
        t = float(np.clip((t_now-r["dep_min"])/r["duration"], 0, 1))
        lon,lat,ang = bezier_pos(r["orig_lon"],r["orig_lat"],
                                  r["dest_lon"],r["dest_lat"], t)
        # fade-in first 5% and fade-out last 5%
        fade = float(np.clip(min(t/0.05, (1-t)/0.05, 1.0), 0.0, 1.0))
        out.append((lon, lat, t, r["color"], r["dist_km"], ang, fade))
    return out

# Peak hour annotation text
def peak_label(cur_min):
    h = (cur_min % 1440) // 60
    if  5 <= h <  7: return "Early morning departures"
    if  7 <= h < 10: return "Morning peak — busiest period"
    if 10 <= h < 14: return "Midday operations"
    if 14 <= h < 17: return "Afternoon lull"
    if 17 <= h < 21: return "Evening peak — second busiest"
    if 21 <= h < 24: return "Night — flights winding down"
    return                   "Night — minimal operations"

print("Helper functions defined")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 7 — Draw single frame
# ─────────────────────────────────────────────────────────────────────────────
TITLE = "INDIA DOMESTIC FLIGHTS  —  01 March 2019 (Friday)"
SUB   = (f"Kaggle / Indian Flight Schedules  |  {len(df):,} flights  |  "
         f"{len(AIRPORTS)} airports  |  Suman Bhowmick")

def draw_frame(cur_min):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor=BG)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.06)
    ax.set_facecolor(BG)

    # ── State fills ───────────────────────────────────────────────────────────
    india_states.plot(ax=ax, color=STATE_COL,
                      edgecolor=BORDER, linewidth=0.5, zorder=1)

    # ── Outer boundary — bright double glow ───────────────────────────────────
    for ring in INDIA_RINGS:
        xs = [c[0] for c in ring]; ys = [c[1] for c in ring]
        ax.plot(xs, ys, color=OUTER_BDR, lw=0.5, alpha=0.4, zorder=3)
        ax.plot(xs, ys, color=OUTER_BDR, lw=1.5, alpha=1.0, zorder=3)

    # ── Static route arcs ─────────────────────────────────────────────────────
    for arc, cnt in STATIC_ARCS:
        alpha = 0.06 + 0.18 * (cnt / max_cnt)
        lw    = 0.25 + 0.6  * (cnt / max_cnt)
        ax.plot(arc[:,0], arc[:,1],
                c="#1A0066", lw=lw, alpha=alpha, zorder=3)

    # ── Airport dots ──────────────────────────────────────────────────────────
    for name, (lat, lon) in AIRPORTS.items():
        is_hub = name in HUB_AIRPORTS
        if is_hub:
            # Outer glow ring
            ax.scatter(lon, lat, s=160, c="none",
                       edgecolors="#00D4FF", linewidths=1.2,
                       alpha=0.6, zorder=4)
            # Bright core dot
            ax.scatter(lon, lat, s=30, c="#FFFFFF",
                       alpha=1.0, zorder=5, linewidths=0)
        else:
            ax.scatter(lon, lat, s=6, c="#1A3A6A",
                       alpha=0.6, zorder=4, linewidths=0)

    # ── City labels — large, bright, clearly readable ────────────────────────
    label_offsets = {
        "Delhi":       ( 0.5,  0.6), "Mumbai":    (-0.6, -0.9),
        "Bengaluru":   ( 0.5, -0.8), "Chennai":   ( 0.5, -0.8),
        "Hyderabad":   (-4.0,  0.5), "Kolkata":   ( 0.5,  0.5),
        "Pune":        ( 0.5, -0.8), "Ahmedabad": (-1.0,  0.7),
        "Goa":         (-2.5,  0.5), "Kochi":     (-3.5, -0.5),
        "Jaipur":      (-3.8,  0.5), "Lucknow":   ( 0.5,  0.5),
        "Patna":       ( 0.5,  0.5), "Guwahati":  ( 0.5,  0.5),
        "Srinagar":    ( 0.5,  0.5),
    }
    for name, (lat, lon) in AIRPORTS.items():
        if name not in HUB_AIRPORTS: continue
        dx, dy = label_offsets.get(name, (0.5, 0.5))
        ax.text(lon+dx, lat+dy, LABEL_NAMES.get(name, name),
                fontsize=8.5, color="#FFFFFF", alpha=1.0,
                fontweight="bold",
                ha="left" if dx >= 0 else "right", va="center",
                zorder=10,
                path_effects=[
                    pe.withStroke(linewidth=3.0, foreground=BG),
                ])

    # ── Flight trails ─────────────────────────────────────────────────────────
    for step in range(TRAIL_STEPS, 0, -1):
        alpha = 0.02 + 0.04*(TRAIL_STEPS - step)
        size  = 1.5  + 3.5*(TRAIL_STEPS - step)/TRAIL_STEPS
        trail = active_flights(cur_min, trail_step=step)
        if trail:
            ax.scatter(
                [f[0] for f in trail], [f[1] for f in trail],
                s=size,
                color=[to_rgba(f[3], alpha) for f in trail],
                zorder=7, linewidths=0
            )

    # ── Aircraft — clean 2-layer glow ─────────────────────────────────────────
    planes = active_flights(cur_min)
    for lon, lat, t, col, dist, angle, fade in planes:
        fsize = float(np.clip(6.5 + dist * 0.004, 6.5, 11))
        # Soft outer glow
        ax.text(lon, lat, "✈", fontsize=fsize*1.6, color=col,
                alpha=fade*0.18, ha="center", va="center",
                rotation=angle, zorder=8)
        # Sharp core
        ax.text(lon, lat, "✈", fontsize=fsize, color=col,
                alpha=fade*0.95, ha="center", va="center",
                rotation=angle, zorder=9,
                path_effects=[pe.withStroke(linewidth=1.0,
                                            foreground=BG)])

    # ── Map frame ─────────────────────────────────────────────────────────────
    ax.set_xlim(LON_MIN, LON_MAX)
    ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect("equal")
    ax.axis("off")

    # ── Clock ─────────────────────────────────────────────────────────────────
    h, m = (cur_min % 1440)//60, cur_min % 60
    ax.text(0.97, 0.97, f"{h:02d}:{m:02d}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=24, fontweight="bold", color="#FF2079",
            path_effects=[pe.withStroke(linewidth=5, foreground=BG)],
            zorder=12)
    ax.text(0.97, 0.905, "IST",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color=TDIM, zorder=12)

    # ── Flight count + peak label ─────────────────────────────────────────────
    n = len(planes)
    ax.text(0.03, 0.07, f"{n}  flights airborne",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9.5, color="#00E5FF",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG)],
            zorder=12)
    ax.text(0.03, 0.04, peak_label(cur_min),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=TDIM, style="italic", zorder=12)

    # ── Progress bar ──────────────────────────────────────────────────────────
    bar = fig.add_axes([0.05, 0.03, 0.90, 0.013])
    bar.set_xlim(0, 1); bar.set_ylim(0, 1)
    bar.set_facecolor("#0A0020")
    progress = (cur_min % 1440) / 1440
    bar.barh(0.5, progress, height=1.0, color="#FF2079", alpha=0.90)
    for th in range(0, 25, 3):
        tp = th/24
        bar.axvline(tp, color="#1A004A", lw=0.8)
        bar.text(tp, -1.0, f"{th:02d}:00",
                 ha="center", va="top", fontsize=5, color=TDIM,
                 transform=bar.transAxes)
    # Peak hour shading
    bar.axvspan(7/24, 10/24, color="#FF2079", alpha=0.18)
    bar.axvspan(17/24, 21/24, color="#FF2079", alpha=0.12)
    bar.axis("off")

    # ── Airline legend ────────────────────────────────────────────────────────
    active_airlines = df.loc[
        (df["dep_min"] <= cur_min) & (df["arr_min"] > cur_min),
        "airline"
    ].unique()
    handles = [
        mpatches.Patch(color=c, label=a, alpha=0.90)
        for a, c in AIRLINE_COLORS.items()
        if a in df["airline"].values
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", fontsize=7,
        framealpha=0.35, edgecolor="#4400AA",
        labelcolor=TVAL, facecolor="#06001A",
        handlelength=1.2, borderpad=0.8,
        title="Airline", title_fontsize=7.5, ncol=1
    )

    # ── Titles ────────────────────────────────────────────────────────────────
    fig.text(0.5, 0.966, TITLE,
             ha="center", fontsize=15, fontweight="bold", color=TVAL,
             path_effects=[pe.withStroke(linewidth=3, foreground=BG)])
    fig.text(0.5, 0.940, SUB,
             ha="center", fontsize=7.5, color=TDIM, style="italic")

    return fig

print("draw_frame defined")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 8 — Parallel render to /tmp PNGs using multiprocessing (~4x faster)
# Before running: Runtime → Change runtime type → T4 GPU (for extra RAM)
# ─────────────────────────────────────────────────────────────────────────────
import multiprocessing as mp
import os, gc, shutil
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — required for multiprocessing

TMP_DIR = "/tmp/flight_frames"
if os.path.exists(TMP_DIR): shutil.rmtree(TMP_DIR)
os.makedirs(TMP_DIR)

# Build task list — (frame_index, current_minute, hold_ms)
tasks     = []
durations = []
for i in range(FRAMES):
    cur = i * TIME_STEP
    if   420  <= cur <= 600:  hold = 60    # morning peak — slightly slower
    elif 1020 <= cur <= 1260: hold = 60    # evening peak — slightly slower
    elif cur < 300:           hold = 20    # dead of night — fast
    else:                     hold = 40    # normal hours
    durations.append(hold)
    tasks.append((i, cur, hold))


def render_one(args):
    """Render a single frame and save to disk. Called in worker process."""
    i, cur, hold = args

    # Re-import inside worker (each process needs its own state)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import gc

    fig = draw_frame(cur)
    fig.savefig(
        os.path.join(TMP_DIR, f"f{i:04d}.png"),
        dpi=DPI, facecolor=BG, bbox_inches="tight"
    )
    plt.close(fig)
    gc.collect()
    return i


# Detect CPU count — Colab usually has 2-4 cores
N_WORKERS = min(mp.cpu_count(), 4)
print(f"Rendering {FRAMES} frames with {N_WORKERS} parallel workers...")
print(f"(Change runtime to T4 GPU for extra RAM if it crashes)\n")

gc.collect()
plt.close("all")

completed = 0
with mp.Pool(processes=N_WORKERS) as pool:
    for done in pool.imap_unordered(render_one, tasks):
        completed += 1
        if completed % 24 == 0 or completed == FRAMES:
            pct = int(completed / FRAMES * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            print(f"  [{bar}] {completed}/{FRAMES} frames  ({pct}%)")

print(f"\nAll {FRAMES} frames saved to disk.")

# ─────────────────────────────────────────────────────────────────────────────
# CELL 9 — Assemble GIF + download (GIF only, no thumbnail)
# ─────────────────────────────────────────────────────────────────────────────
gif_fname = "india_flights_24hr.gif"

png_paths = sorted([
    os.path.join(TMP_DIR, f)
    for f in os.listdir(TMP_DIR) if f.endswith(".png")
])

print(f"Assembling GIF from {len(png_paths)} frames...")

# Load all frames as RGBA first for best colour accuracy
# Then convert to palette with 256 colours per frame
frames_out = []
for i, path in enumerate(png_paths):
    img  = Image.open(path).convert("RGBA")
    # Convert to palette — 256 colours, no dither for clean dark backgrounds
    pal  = img.convert("P", palette=Image.ADAPTIVE,
                       colors=256, dither=Image.NONE)
    frames_out.append(pal)
    img.close()
    if (i+1) % 60 == 0:
        print(f"  Converted {i+1}/{len(png_paths)} frames")

print("Saving GIF...")
frames_out[0].save(
    gif_fname,
    save_all=True,
    append_images=frames_out[1:],
    duration=durations,
    loop=0,
    optimize=True,
)

for f in frames_out: f.close()
del frames_out; gc.collect()
shutil.rmtree(TMP_DIR)

import os as _os
size_mb = _os.path.getsize(gif_fname) / 1024 / 1024
print(f"GIF saved: {gif_fname}  ({size_mb:.1f} MB)")
files.download(gif_fname)
print("Done!")
print()
print("Key moments:")
print("  00:00-05:30  Empty skies — near-zero flights")
print("  05:30-07:00  First morning wave departs")
print("  07:00-10:00  MORNING PEAK — sky fills up rapidly")
print("  12:00-15:00  Midday steady operations")
print("  17:00-21:00  EVENING PEAK — second surge")
print("  22:00+       Flights taper off to silence")
