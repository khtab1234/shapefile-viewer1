"""
app.py  —  Shapefile Viewer (Streamlit Web App)
Design & Developed by Eng. Mhmd Samir

Run locally:
    streamlit run app.py

Deploy free on Streamlit Cloud:
    https://streamlit.io/cloud
"""

import math, zipfile, tempfile, shutil, io

from pathlib import Path

import streamlit as st
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Arabic fix ────────────────────────────────────────
try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def ar(text):
        if not text: return text
        return get_display(arabic_reshaper.reshape(str(text)))
except ImportError:
    def ar(text): return str(text) if text else text

# ── Page config ───────────────────────────────────────
st.set_page_config(
    page_title="Shapefile Viewer — Eng. Mhmd Samir",
    page_icon="🗺️",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────
st.markdown("""
<style>
  /* Dark sidebar */
  section[data-testid="stSidebar"] { background: #1e293b; }
  section[data-testid="stSidebar"] * { color: #f1f5f9 !important; }
  section[data-testid="stSidebar"] .stSlider > div { color: #f1f5f9; }
  section[data-testid="stSidebar"] input { background:#0f172a !important; color:#f1f5f9 !important; }
  /* Header bar */
  .top-bar {
    background: linear-gradient(90deg,#1e3a5f,#6366f1);
    padding: 10px 20px; border-radius: 8px;
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }
  .top-bar h2 { color:white; margin:0; font-size:1.2rem; }
  .top-bar small { color:#c7d2fe; font-size:0.75rem; }
</style>
""", unsafe_allow_html=True)

# ── Top bar ───────────────────────────────────────────
st.markdown("""
<div class="top-bar">
  <h2>🗺️ Shapefile Viewer</h2>
  <small>Design &amp; Developed by Eng. Mhmd Samir</small>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════

def haversine_m(lon1, lat1, lon2, lat2):
    R = 6_371_000
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(df/2)**2 + math.cos(f1)*math.cos(f2)*math.sin(dl/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def fmt(m):
    return f"{m/1000:.2f} km" if m >= 1000 else f"{m:.2f} m"

def collect_vertices(gdf):
    vertices = []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        gt = geom.geom_type
        rings = []
        if gt == "Polygon":
            rings.append(list(geom.exterior.coords))
            for i in geom.interiors: rings.append(list(i.coords))
        elif gt == "MultiPolygon":
            for poly in geom.geoms:
                rings.append(list(poly.exterior.coords))
                for i in poly.interiors: rings.append(list(i.coords))
        elif gt == "LineString":   rings.append(list(geom.coords))
        elif gt == "MultiLineString":
            rings += [list(g.coords) for g in geom.geoms]
        for ring in rings:
            pts = ring[:-1] if len(ring)>1 and ring[0]==ring[-1] else ring
            vertices.extend(pts)
    return vertices

def load_shp(uploaded_file):
    """يقرأ SHP أو ZIP من الـ uploader ويرجع GeoDataFrame."""
    tmp = tempfile.mkdtemp(prefix="shpweb_")
    try:
        name = uploaded_file.name
        data = uploaded_file.read()
        fpath = Path(tmp) / name
        fpath.write_bytes(data)

        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(fpath) as z:
                z.extractall(tmp)
        # ابحث عن .shp
        shps = list(Path(tmp).rglob("*.shp"))
        if not shps:
            raise FileNotFoundError("No .shp found inside the archive.")
        gdf = gpd.read_file(shps[0])
        if gdf.crs and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        return gdf, shps[0].stem
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ════════════════════════════════════════════════════
#  Drawing
# ════════════════════════════════════════════════════

def draw_map(ax, gdf, font_size, line_color, label_color, lw,
             map_bg, label_bg, show_vertices, show_lengths):
    ax.clear()
    ax.set_facecolor(map_bg)

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        gt = geom.geom_type

        lines = []
        if gt == "LineString":       lines = [list(geom.coords)]
        elif gt == "MultiLineString":lines = [list(g.coords) for g in geom.geoms]
        elif gt == "Polygon":
            lines = [list(geom.exterior.coords)]
            for i in geom.interiors: lines.append(list(i.coords))
        elif gt == "MultiPolygon":
            for poly in geom.geoms:
                lines.append(list(poly.exterior.coords))
                for i in poly.interiors: lines.append(list(i.coords))

        for coords in lines:
            xs = [c[0] for c in coords]
            ys = [c[1] for c in coords]
            ax.plot(xs, ys, color=line_color, linewidth=lw,
                    solid_capstyle="round", solid_joinstyle="round")

            if show_lengths:
                all_xs = [c[0] for cc in lines for c in cc]
                all_ys = [c[1] for cc in lines for c in cc]
                data_range = max(
                    max(all_xs)-min(all_xs) if len(all_xs)>1 else 1,
                    max(all_ys)-min(all_ys) if len(all_ys)>1 else 1
                )
                DIM_OFFSET = data_range * 0.035
                EXT_OVER   = data_range * 0.012
                cx_shape = (max(all_xs)+min(all_xs))/2
                cy_shape = (max(all_ys)+min(all_ys))/2
                placed_labels = []

                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    d = haversine_m(p1[0], p1[1], p2[0], p2[1])
                    if d < 0.01: continue
                    x1,y1,x2,y2 = p1[0],p1[1],p2[0],p2[1]
                    mx,my = (x1+x2)/2,(y1+y2)/2
                    seg_angle = math.atan2(y2-y1, x2-x1)
                    nx,ny = -math.sin(seg_angle), math.cos(seg_angle)
                    to_mid_x,to_mid_y = mx-cx_shape, my-cy_shape
                    if (nx*to_mid_x + ny*to_mid_y) < 0: nx,ny = -nx,-ny
                    mult = 1.0
                    for (px_p, py_p) in placed_labels:
                        if math.hypot(mx+nx*DIM_OFFSET-px_p,
                                      my+ny*DIM_OFFSET-py_p) < DIM_OFFSET*1.8:
                            mult = 2.2; break
                    eff = DIM_OFFSET * mult
                    d1x,d1y = x1+nx*eff, y1+ny*eff
                    d2x,d2y = x2+nx*eff, y2+ny*eff
                    dmx,dmy = mx+nx*eff, my+ny*eff
                    placed_labels.append((dmx,dmy))

                    for (px,py,dpx,dpy) in [(x1,y1,d1x,d1y),(x2,y2,d2x,d2y)]:
                        ax.plot([px, dpx+nx*EXT_OVER],[py, dpy+ny*EXT_OVER],
                                color=label_color, linewidth=0.7,
                                linestyle="--", alpha=0.7, zorder=8)
                    ax.annotate("", xy=(d2x,d2y), xytext=(d1x,d1y),
                                xycoords="data", textcoords="data",
                                arrowprops=dict(arrowstyle="<->",
                                    color=label_color, lw=1.1,
                                    mutation_scale=10), zorder=9)
                    angle_deg = math.degrees(seg_angle)
                    if angle_deg>90:  angle_deg-=180
                    if angle_deg<-90: angle_deg+=180
                    bbox = dict(boxstyle="round,pad=0.18", fc=label_bg,
                                alpha=0.90, ec=label_color,
                                linewidth=0.5) if label_bg!="none" else None
                    ax.text(dmx, dmy, fmt(d),
                            fontsize=font_size, color=label_color,
                            ha="center", va="center", rotation=angle_deg,
                            bbox=bbox, fontweight="bold", zorder=10)

        if "Point" in gt:
            xs = [geom.x] if gt=="Point" else [g.x for g in geom.geoms]
            ys = [geom.y] if gt=="Point" else [g.y for g in geom.geoms]
            ax.scatter(xs, ys, color=line_color, s=18, zorder=5)

    # ── Bounds ────────────────────────────────────────
    all_x, all_y = [], []
    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty: continue
        try:
            b = geom.bounds
            all_x += [b[0], b[2]]; all_y += [b[1], b[3]]
        except: pass

    if all_x and all_y:
        xmin,xmax = min(all_x),max(all_x)
        ymin,ymax = min(all_y),max(all_y)
        xpad = max((xmax-xmin)*0.18, 1e-6)
        ypad = max((ymax-ymin)*0.18, 1e-6)
        ax.set_xlim(xmin-xpad, xmax+xpad)
        ax.set_ylim(ymin-ypad, ymax+ypad)

    ax.set_aspect("equal", adjustable="datalim")
    ax.tick_params(labelsize=7, colors="#aaa")
    for sp in ax.spines.values(): sp.set_edgecolor("#333")

    # ── Vertex numbers ────────────────────────────────
    if show_vertices:
        vertices = collect_vertices(gdf)
        if vertices and all_x:
            x_lo,x_hi = ax.get_xlim()
            y_lo,y_hi = ax.get_ylim()
            off_x = (x_hi-x_lo)/600*8
            off_y = (y_hi-y_lo)/600*8
            vxs = [v[0] for v in vertices]
            vys = [v[1] for v in vertices]
            ax.scatter(vxs, vys, color="#f59e0b", s=50,
                       edgecolors="#92400e", linewidths=1.2,
                       zorder=60, clip_on=False)
            for idx,(vx,vy) in enumerate(vertices, start=1):
                ax.text(vx+off_x, vy+off_y, str(idx),
                        fontsize=max(font_size,7), color="white",
                        fontweight="bold", ha="left", va="bottom",
                        zorder=61, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.22",
                                  fc="#1e3a5f", ec="#f59e0b",
                                  linewidth=0.9, alpha=0.95))

    # ── North Arrow ───────────────────────────────────
    from matplotlib.patches import Circle
    na_x, na_y = 0.93, 0.82
    circ = Circle((na_x, na_y+0.055), 0.065,
                  transform=ax.transAxes,
                  facecolor="white", edgecolor="#1e3a5f",
                  linewidth=1.5, zorder=25)
    ax.add_patch(circ)
    ax.annotate("", xy=(na_x,na_y+0.105), xytext=(na_x,na_y+0.055),
                xycoords="axes fraction", textcoords="axes fraction",
                arrowprops=dict(arrowstyle="-|>", color="#1e3a5f",
                                lw=2.0, mutation_scale=14), zorder=26)
    ax.plot([na_x,na_x],[na_y+0.005,na_y+0.055],
            color="#1e3a5f", lw=2, transform=ax.transAxes, zorder=26)
    ax.text(na_x, na_y+0.12, "N", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=10,
            fontweight="bold", color="#1e3a5f", zorder=27)


def draw_table(ax_tbl, gdf):
    ax_tbl.clear(); ax_tbl.axis("off")
    vertices = collect_vertices(gdf)
    if not vertices:
        ax_tbl.text(0.5,0.5,"No vertices",ha="center",va="center",
                    fontsize=9,color="#64748b",transform=ax_tbl.transAxes)
        return
    col_labels = ["#","X (Lon)","Y (Lat)"]
    table_data = [[str(i+1),f"{v[0]:.6f}",f"{v[1]:.6f}"]
                  for i,v in enumerate(vertices)]
    row_colors = [["#f8fafc"]*3 if i%2==0 else ["#e2e8f0"]*3
                  for i in range(len(table_data))]
    tbl = ax_tbl.table(cellText=table_data, colLabels=col_labels,
                       cellLoc="center", loc="upper center",
                       cellColours=row_colors)
    tbl.auto_set_font_size(False); tbl.set_fontsize(7); tbl.scale(1,1.15)
    for j in range(3):
        tbl[(0,j)].set_facecolor("#1e3a5f")
        tbl[(0,j)].set_text_props(color="white", fontweight="bold")
    ax_tbl.set_title(ar("إحداثيات الكسرات"), fontsize=8,
                     fontweight="bold", color="#1e293b", pad=3)
    ax_tbl.set_xlim(0,1); ax_tbl.set_ylim(0,1)


def draw_title_block(ax_tb, name, ref, center):
    ax_tb.clear(); ax_tb.set_xlim(0,1); ax_tb.set_ylim(0,1); ax_tb.axis("off")
    ax_tb.set_facecolor("#dbeafe")
    from matplotlib.patches import FancyBboxPatch
    outer = FancyBboxPatch((0.005,0.05),0.99,0.90,
                           boxstyle="square,pad=0",
                           linewidth=1.5, edgecolor="#1e3a5f",
                           facecolor="#dbeafe", zorder=1,
                           transform=ax_tb.transAxes)
    ax_tb.add_patch(outer)
    cols = [
        (ar("اسم مقدم الطلب"), ar(name   or "—")),
        (ar("رقم الطلب"),       ar(ref    or "—")),
        (ar("المركز"),           ar(center or "—")),
    ]
    for i,(header,value) in enumerate(cols):
        col_w = 1.0/3
        x0 = i*col_w
        if i>0: ax_tb.axvline(x=x0,ymin=0.05,ymax=0.95,
                               color="#1e3a5f",linewidth=1.2,zorder=2)
        cx = x0+col_w/2
        ax_tb.text(cx,0.72,header,ha="center",va="center",
                   fontsize=7,color="#475569",fontweight="bold",
                   transform=ax_tb.transAxes,zorder=3)
        ax_tb.plot([x0+0.01,x0+col_w-0.01],[0.55,0.55],
                   color="#94a3b8",linewidth=0.6,
                   transform=ax_tb.transAxes,zorder=3)
        ax_tb.text(cx,0.28,value,ha="center",va="center",
                   fontsize=10,color="#1e293b",fontweight="bold",
                   transform=ax_tb.transAxes,zorder=3,clip_on=True)
    ax_tb.text(0.998,0.12,"Design & Developed by Eng. Mhmd Samir",
               ha="right",va="center",fontsize=6.5,color="#64748b",
               fontstyle="italic",transform=ax_tb.transAxes,zorder=4)


def build_figure(gdf, opts):
    fig = plt.figure(figsize=(14, 9), facecolor="#0a1628")
    gs = gridspec.GridSpec(2, 2, figure=fig,
                           width_ratios=[3,1], height_ratios=[10,1],
                           left=0.04, right=0.98,
                           top=0.97, bottom=0.03,
                           wspace=0.04, hspace=0.06)
    ax     = fig.add_subplot(gs[0,0])
    ax_tbl = fig.add_subplot(gs[0,1])
    ax_tb  = fig.add_subplot(gs[1,:])

    draw_map(ax, gdf,
             opts["font_size"], opts["line_color"], opts["label_color"],
             opts["lw"], opts["map_bg"], opts["label_bg"],
             opts["show_vertices"], opts["show_lengths"])

    if opts["show_table"]:
        draw_table(ax_tbl, gdf)
    else:
        ax_tbl.axis("off")

    draw_title_block(ax_tb, opts["tb_name"], opts["tb_ref"], opts["tb_center"])

    return fig


# ════════════════════════════════════════════════════
#  Sidebar
# ════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🗂️ Open File")
    uploaded = st.file_uploader(
        "Upload SHP or ZIP",
        type=["shp","zip"],
        help="Upload a .shp file or a .zip containing the shapefile components"
    )

    st.markdown("---")
    st.markdown("## ⚙️ Display Options")
    show_vertices = st.checkbox("Show vertex numbers", value=True)
    show_lengths  = st.checkbox("Show segment lengths", value=True)
    show_table    = st.checkbox("Show coords table", value=True)

    st.markdown("---")
    font_size  = st.slider("Label font size", 4, 22, 8)
    lw         = st.slider("Line width", 0.5, 8.0, 1.5, step=0.5)
    dpi        = st.slider("Save DPI", 72, 600, 300, step=50)

    st.markdown("---")
    st.markdown("## 🎨 Colors")
    line_color  = st.color_picker("Line color",  "#1d4ed8")
    label_color = st.color_picker("Label color", "#b91c1c")
    label_bg    = st.color_picker("Label bg",    "#fffde7")
    map_bg      = st.color_picker("Map bg",      "#f8fafc")

    st.markdown("---")
    st.markdown("## 📋 بيانات الطلب")
    tb_name   = st.text_input("اسم مقدم الطلب", placeholder="مثال: أحمد محمد")
    tb_ref    = st.text_input("رقم الطلب",       placeholder="مثال: 2024/123")
    tb_center = st.text_input("المركز",           placeholder="مثال: مركز أبوحمص")


# ════════════════════════════════════════════════════
#  Main area
# ════════════════════════════════════════════════════

if uploaded is None:
    st.info("⬆️  Upload a Shapefile (SHP or ZIP) from the sidebar to begin.")
    st.stop()

# Load
with st.spinner("Loading shapefile..."):
    try:
        gdf, layer_name = load_shp(uploaded)
    except Exception as e:
        st.error(f"Failed to load file: {e}")
        st.stop()

st.caption(f"**Layer:** {layer_name}  •  **Features:** {len(gdf)}  •  **CRS:** EPSG:4326")

# Build options dict
opts = dict(
    font_size=font_size, line_color=line_color, label_color=label_color,
    lw=lw, map_bg=map_bg, label_bg=label_bg,
    show_vertices=show_vertices, show_lengths=show_lengths, show_table=show_table,
    tb_name=tb_name, tb_ref=tb_ref, tb_center=tb_center,
)

# Draw
with st.spinner("Rendering..."):
    fig = build_figure(gdf, opts)

st.pyplot(fig, use_container_width=True)

# ── Download button ───────────────────────────────────
buf = io.BytesIO()
fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
            facecolor=fig.get_facecolor())
buf.seek(0)
st.download_button(
    label="⬇️ Download PNG",
    data=buf,
    file_name=f"{layer_name}_output.png",
    mime="image/png",
    use_container_width=True,
)

# ── Coords table (interactive) ────────────────────────
with st.expander("📊 Vertex Coordinates Table", expanded=False):
    vertices = collect_vertices(gdf)
    if vertices:
        import pandas as pd
        df = pd.DataFrame(
            [[i+1, f"{v[0]:.6f}", f"{v[1]:.6f}"] for i,v in enumerate(vertices)],
            columns=["#", "X (Lon)", "Y (Lat)"]
        )
        st.dataframe(df, use_container_width=True, height=300)

        # تحميل الجدول كـ CSV
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Coords CSV", csv,
                           f"{layer_name}_coords.csv", "text/csv")
    else:
        st.warning("No vertices found.")

plt.close(fig)
