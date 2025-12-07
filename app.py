import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

st.set_page_config(page_title="Week 10 • Geography & Channels", layout="wide")
st.title("Week 10 — Geography & Channels Deep Dive")
st.caption("World map • Geography × Channels • Shipping lag by Country + City • Stats • $ CAD")

BASE = pathlib.Path(__file__).parent
DATA_FILE = BASE / "Combined_Sales_2025 (2).csv"

ESSENTIAL = [
    "Sale ID", "Date", "Country", "City", "Channel",
    "Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Shipped Date"
]

@st.cache_data(show_spinner=False)
def load_csv(p: pathlib.Path) -> pd.DataFrame:
    try:
        d = pd.read_csv(p)
    except Exception:
        d = pd.read_csv(p, encoding="utf-8-sig")
    d.columns = d.columns.str.strip()
    return d

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

def normalize_country(x: str) -> str:
    s = "" if x is None else str(x).strip()
    if not s:
        return ""
    patches = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.": "United States",
        "us": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
    }
    return patches.get(s.lower(), s)

def cad(x, decimals=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    try:
        return f"${float(x):,.{decimals}f} CAD"
    except Exception:
        return "-"

def p_fmt(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "-"
    p = float(p)
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"

def rank_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "#", range(1, len(out) + 1))
    return out

def download_html(fig: go.Figure, filename: str):
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button(
        label="Download HTML",
        data=html,
        file_name=filename,
        mime="text/html",
        key=f"dl_{filename}"
    )

def heatmap_from_pivot(pv: pd.DataFrame, title: str, ztitle: str):
    fig = go.Figure(
        data=go.Heatmap(
            z=pv.values,
            x=pv.columns.tolist(),
            y=pv.index.tolist(),
            colorbar=dict(title=ztitle)
        )
    )
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=60, b=0))
    return fig

if not DATA_FILE.exists():
    st.error("Dataset file not found. Put 'Combined_Sales_2025 (2).csv' in the SAME folder as app.py in your repo.")
    st.stop()

df = load_csv(DATA_FILE)
missing = [c for c in ESSENTIAL if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

text_cols = ["Country", "City", "Channel", "Customer Type", "Product Type", "Lead Source", "Consignment? (Y/N)"]
for c in text_cols:
    if c in df.columns:
        df[c] = _clean_str(df[c])

df["Country"] = df["Country"].apply(normalize_country)

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Shipped Date"] = pd.to_datetime(df["Shipped Date"], errors="coerce")

num_cols = ["Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)",
            "Color Count (#)", "length", "width", "weight"]
for c in num_cols:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["Net Sales (CAD)"] = (df["Price (CAD)"] - df["Discount (CAD)"]).clip(lower=0)
df["Total Collected (CAD)"] = (df["Net Sales (CAD)"] + df["Shipping (CAD)"].fillna(0) + df["Taxes Collected (CAD)"].fillna(0)).clip(lower=0)
df["Discount Rate"] = np.where(df["Price (CAD)"] > 0, df["Discount (CAD)"] / df["Price (CAD)"], np.nan)

df["Ship Lag Raw (days)"] = (df["Shipped Date"] - df["Date"]).dt.days
df["Ship Lag Clean (days)"] = np.where(df["Ship Lag Raw (days)"] >= 0, df["Ship Lag Raw (days)"], np.nan)
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

st.sidebar.header("Filters")
min_d = df["Date"].min()
max_d = df["Date"].max()
if pd.isna(min_d) or pd.isna(max_d):
    st.error("Date column could not be parsed.")
    st.stop()

dr = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
if not isinstance(dr, tuple):
    dr = (dr, dr)
start = pd.to_datetime(dr[0])
end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

metric = st.sidebar.selectbox("Metric ($ CAD)", ["Total Collected (CAD)", "Net Sales (CAD)", "Price (CAD)"], index=0)
exclude_negative_lag = st.sidebar.toggle("Exclude negative ship lag", value=True)
top_n = st.sidebar.slider("Top N (countries)", 5, 30, 12)

countries = sorted([c for c in df["Country"].dropna().unique().tolist() if c])
channels = sorted([c for c in df["Channel"].dropna().unique().tolist() if c])
sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])

base = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
if sel_countries:
    base = base[base["Country"].isin(sel_countries)]
if sel_channels:
    base = base[base["Channel"].isin(sel_channels)]

cities = sorted([c for c in base["City"].dropna().unique().tolist() if c])
sel_cities = st.sidebar.multiselect("Cities (optional)", cities, default=[])

f = base.copy()
if sel_cities:
    f = f[f["City"].isin(sel_cities)]

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

lag_col = "Ship Lag Clean (days)" if exclude_negative_lag else "Ship Lag Raw (days)"

total = float(f[metric].sum())
orders = int(len(f))
aov = float(f[metric].mean())
median_val = float(f[metric].median())

country_totals = f.groupby("Country")[metric].sum().sort_values(ascending=False)
channel_totals = f.groupby("Channel")[metric].sum().sort_values(ascending=False)

top_country = country_totals.index[0] if len(country_totals) else "-"
top_channel = channel_totals.index[0] if len(channel_totals) else "-"

cons_rate = float((f["Consignment? (Y/N)"].astype(str).str.upper().eq("Y").mean()) * 100) if "Consignment? (Y/N)" in f.columns else np.nan
neg_lag_rows = int((f["Ship Lag Raw (days)"] < 0).sum())
avg_lag = float(np.nanmean(f[lag_col].values)) if f[lag_col].notna().any() else np.nan

k1, k2, k3, k4, k5, k6, k7, k8, k9 = st.columns(9)
k1.metric("Orders", f"{orders:,}")
k2.metric("Total", cad(total, 0))
k3.metric("Avg Order", cad(aov, 0))
k4.metric("Median", cad(median_val, 0))
k5.metric("Top Country", top_country if top_country else "-")
k6.metric("Top Channel", top_channel if top_channel else "-")
k7.metric("Consignment", f"{cons_rate:.1f}%" if np.isfinite(cons_rate) else "-")
k8.metric("Avg Ship Lag", f"{avg_lag:.1f} days" if np.isfinite(avg_lag) else "-")
k9.metric("Neg Ship Lag Rows", f"{neg_lag_rows:,}")

tabs = st.tabs(["Overview", "World Map", "Geography × Channels", "Shipping Lag (Country + City)", "Time", "Stats", "Data"])

with tabs[0]:
    st.subheader("Insights")
    share_top = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
    bullets = []
    if np.isfinite(share_top):
        bullets.append(f"- Concentration: **{top_country}** contributes ~**{share_top*100:.1f}%** of {metric}.")
    bullets.append(f"- Top channel by {metric}: **{top_channel}**.")
    if np.isfinite(cons_rate):
        bullets.append(f"- Consignment share: **{cons_rate:.1f}%** of orders.")
    if neg_lag_rows > 0:
        bullets.append(f"- Data note: **{neg_lag_rows}** rows have negative ship lag (excluded from lag charts if toggle is ON).")
    st.markdown("\n".join(bullets) if bullets else "-")

    st.subheader("Recommendations")
    recs = []
    if np.isfinite(share_top) and share_top >= 0.5:
        recs += [
            "- Protect the anchor country with inventory + fulfillment reliability + channel execution.",
            "- Scale 2–3 next-tier countries using the channels that already win there (see heatmap).",
        ]
    else:
        recs += ["- Use market tiers (anchor/growth/test) and align channel strategy per tier."]
    if neg_lag_rows > 0:
        recs += ["- Fix negative ship lag (date entry / shipping record) so operational KPIs are reliable."]
    st.markdown("\n".join(recs) if recs else "-")

with tabs[1]:
    st.subheader(f"World map — {metric} ($ CAD)")
    agg = country_totals.reset_index().rename(columns={metric: "value"})
    agg["share"] = agg["value"] / agg["value"].sum()

    fig = px.choropleth(
        agg,
        locations="Country",
        locationmode="country names",
        color="value",
        hover_name="Country",
        custom_data=["share"],
        projection="natural earth",
        title=f"World Map — {metric} ($ CAD)"
    )
    fig.update_traces(
        hovertemplate="<b>%{location}</b><br>"
                      "Value: %{z:$,.0f} CAD<br>"
                      "Share: %{customdata[0]:.1%}<extra></extra>"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "01_world_map.html")

    st.subheader("Top markets")
    top_tbl = agg.sort_values("value", ascending=False).head(15).copy()
    top_tbl = rank_df(top_tbl)
    st.dataframe(top_tbl.set_index("#")[["Country", "value", "share"]], use_container_width=True)

with tabs[2]:
    colA, colB = st.columns(2)

    with colA:
        st.subheader(f"Top countries by {metric}")
        top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "value"})
        fig1 = px.bar(top_c, x="Country", y="value", title=f"Top {top_n} Countries ({metric})")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        st.plotly_chart(fig1, use_container_width=True)
        download_html(fig1, "02_top_countries.html")

    with colB:
        st.subheader(f"{metric} by channel")
        ch = channel_totals.reset_index().rename(columns={metric: "value"})
        fig2 = px.bar(ch, x="Channel", y="value", title=f"{metric} by Channel ($ CAD)")
        fig2.update_layout(xaxis={"categoryorder": "total descending"})
        fig2.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)
        download_html(fig2, "03_channel_bar.html")

    st.subheader("Country × Channel heatmap (Top countries)")
    top_idx = country_totals.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
    fig3 = heatmap_from_pivot(pv, f"Heatmap: {metric} ($ CAD)", "$ CAD")
    st.plotly_chart(fig3, use_container_width=True)
    download_html(fig3, "04_country_channel_heatmap.html")

    st.subheader("Channel mix share by country (Top countries)")
    mix = df_top.groupby(["Country", "Channel"])[metric].sum().reset_index().rename(columns={metric: "value"})
    mix["country_total"] = mix.groupby("Country")["value"].transform("sum")
    mix["share"] = mix["value"] / mix["country_total"]
    fig4 = px.bar(mix, x="Country", y="share", color="Channel", barmode="stack", title="Channel Mix (Share of Country Total)")
    fig4.update_layout(yaxis_tickformat=".0%", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig4, use_container_width=True)
    download_html(fig4, "05_channel_mix_share.html")

with tabs[3]:
    st.subheader("Shipping lag tied to both Country + City")

    lag_df = f.dropna(subset=[lag_col]).copy()
    if lag_df.empty:
        st.info("No usable shipping lag values after filters.")
    else:
        lag_df["Ship Lag (days)"] = lag_df[lag_col].astype(float)

        col1, col2 = st.columns(2)

        with col1:
            by_country = (lag_df.groupby("Country")["Ship Lag (days)"]
                          .mean().sort_values(ascending=False).head(20).reset_index())
            fig1 = px.bar(by_country, x="Country", y="Ship Lag (days)", title="Avg Ship Lag by Country (days)")
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig1, use_container_width=True)
            download_html(fig1, "06_ship_lag_by_country.html")

            pick = st.selectbox("Pick a country (city drilldown)", sorted(lag_df["Country"].unique().tolist()))
            by_city = (lag_df[lag_df["Country"] == pick]
                       .groupby("City")["Ship Lag (days)"].mean()
                       .sort_values(ascending=False).head(15).reset_index())
            fig2 = px.bar(by_city, x="City", y="Ship Lag (days)", title=f"Avg Ship Lag by City in {pick} (Top 15)")
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig2, use_container_width=True)
            download_html(fig2, "07_ship_lag_by_city.html")

        with col2:
            min_orders = st.slider("Minimum orders per Country+City", 2, 15, 5)

            cc = (lag_df.groupby(["Country", "City"]).agg(
                orders=("Sale ID", "count"),
                avg_lag=("Ship Lag (days)", "mean"),
                med_lag=("Ship Lag (days)", "median"),
                total_metric=(metric, "sum")
            ).reset_index())
            cc = cc[cc["orders"] >= min_orders].copy()
            cc = cc.sort_values(["avg_lag", "orders"], ascending=[False, False]).head(25)
            cc["total_metric"] = cc["total_metric"].round(0)
            cc["avg_lag"] = cc["avg_lag"].round(1)
            cc["med_lag"] = cc["med_lag"].round(1)
            cc = rank_df(cc).rename(columns={"total_metric": f"Total ({metric})"})
            st.dataframe(cc.set_index("#")[["Country", "City", "orders", "avg_lag", "med_lag", f"Total ({metric})"]], use_container_width=True)

            top_countries = (lag_df.groupby("Country")[metric].sum().sort_values(ascending=False).head(12).index)
            sub = lag_df[lag_df["Country"].isin(top_countries)].copy()
            top_cities = (sub.groupby("City")[metric].sum().sort_values(ascending=False).head(20).index)
            sub = sub[sub["City"].isin(top_cities)].copy()

            pv2 = sub.pivot_table(values="Ship Lag (days)", index="Country", columns="City", aggfunc="mean")
            fig3 = heatmap_from_pivot(pv2, "Avg Ship Lag Heatmap (Country × City)", "days")
            st.plotly_chart(fig3, use_container_width=True)
            download_html(fig3, "08_ship_lag_heatmap_country_city.html")

            samp = lag_df.copy()
            if len(samp) > 2500:
                samp = samp.sample(2500, random_state=7)
            fig4 = px.scatter(samp, x="Ship Lag (days)", y=metric, color="Channel",
                              title=f"Ship Lag vs {metric} ($ CAD)", hover_data=["Country", "City"])
            fig4.update_traces(hovertemplate="Lag: %{x:.0f} days<br>Value: %{y:$,.0f} CAD<extra></extra>")
            st.plotly_chart(fig4, use_container_width=True)
            download_html(fig4, "09_ship_lag_scatter.html")

        st.subheader("Shipping recommendations")
        st.markdown(
            "- Use the **Country+City hotspot table** to identify the biggest delay areas.\n"
            "- If a **few cities** are slow inside a country: fix routes/carriers for those city clusters.\n"
            "- If a **whole country** is slow: treat it as a different fulfillment region or SLA."
        )

with tabs[4]:
    st.subheader(f"Time trend — {metric} ($ CAD)")
    ts_df = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "value"})
    fig = px.line(ts_df, x="Month", y="value", title=f"Monthly {metric} ($ CAD)")
    fig.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "10_monthly_trend.html")

with tabs[5]:
    st.subheader("Stats")

    st.markdown("### 1) Do channels differ on order value?")
    grp = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
    if len(grp) >= 2:
        _, p = stats.kruskal(*grp.tolist())
        st.write(f"p-value: **{p_fmt(p)}** → " + ("Different typical values across channels" if p < 0.05 else "No strong evidence of difference"))
    else:
        st.write("Not enough data for this test with current filters.")

    st.markdown("### 2) Is channel mix different by country?")
    top_for_test = country_totals.head(min(10, len(country_totals))).index
    tmp = f.copy()
    tmp["Country (top)"] = np.where(tmp["Country"].isin(top_for_test), tmp["Country"], "Other")
    ct = pd.crosstab(tmp["Country (top)"], tmp["Channel"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        _, p2, _, _ = stats.chi2_contingency(ct)
        st.write(f"p-value: **{p_fmt(p2)}** → " + ("Different mixes by country" if p2 < 0.05 else "No strong evidence of different mixes"))
    else:
        st.write("Not enough data for this test with current filters.")

    st.markdown("### 3) Strongest numeric relationships (Spearman)")
    driver_candidates = [
        "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)",
        "Color Count (#)", "length", "width", "weight", lag_col
    ]
    drivers = [c for c in driver_candidates if c in f.columns]
    rows = []
    for c in drivers:
        x = f[c]
        y = f[metric]
        ok = x.notna() & y.notna()
        if ok.sum() >= 30:
            r, pv = stats.spearmanr(x[ok], y[ok])
            rows.append((c, float(r), float(pv), int(ok.sum())))
    if rows:
        out = pd.DataFrame(rows, columns=["variable", "spearman_r", "p_value", "n"])
        out["abs_r"] = out["spearman_r"].abs()
        out = out.sort_values("abs_r", ascending=False).drop(columns=["abs_r"]).head(10).reset_index(drop=True)
        out["spearman_r"] = out["spearman_r"].round(3)
        out["p_value"] = out["p_value"].apply(lambda v: "<0.0001" if float(v) < 1e-4 else f"{float(v):.4f}")
        out = rank_df(out)
        st.dataframe(out.set_index("#"), use_container_width=True)
    else:
        st.write("Not enough data for this test with current filters.")

with tabs[6]:
    st.subheader("Clean tables + download")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Top Countries")
        t = country_totals.reset_index().rename(columns={metric: "Total ($ CAD)"}).head(25)
        t["Total ($ CAD)"] = t["Total ($ CAD)"].round(0)
        st.dataframe(rank_df(t).set_index("#"), use_container_width=True)

        st.markdown("### Top Cities (Country + City)")
        cct = f.groupby(["Country", "City"])[metric].sum().sort_values(ascending=False).head(25).reset_index().rename(columns={metric: "Total ($ CAD)"})
        cct["Total ($ CAD)"] = cct["Total ($ CAD)"].round(0)
        st.dataframe(rank_df(cct).set_index("#"), use_container_width=True)

    with colB:
        st.markdown("### Country × Channel KPI")
        kpi = f.groupby(["Country", "Channel"]).agg(
            orders=("Sale ID", "count"),
            total=(metric, "sum"),
            avg=(metric, "mean"),
            median=(metric, "median"),
            avg_ship_lag=(lag_col, "mean"),
            avg_discount_rate=("Discount Rate", "mean")
        ).reset_index()

        kpi["total"] = kpi["total"].round(0)
        kpi["avg"] = kpi["avg"].round(0)
        kpi["median"] = kpi["median"].round(0)
        kpi["avg_ship_lag"] = kpi["avg_ship_lag"].round(1)
        kpi["avg_discount_rate"] = (kpi["avg_discount_rate"] * 100).round(1)

        kpi = kpi.sort_values("total", ascending=False).head(40)
        kpi = rank_df(kpi).rename(columns={
            "total": "Total ($ CAD)",
            "avg": "Avg ($ CAD)",
            "median": "Median ($ CAD)",
            "avg_ship_lag": "Avg Ship Lag (days)",
            "avg_discount_rate": "Avg Discount (%)"
        })
        st.dataframe(kpi.set_index("#"), use_container_width=True)

        st.download_button(
            "Download filtered data (CSV)",
            data=f.to_csv(index=False).encode("utf-8"),
            file_name="filtered_data.csv",
            mime="text/csv"
        )

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
