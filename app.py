import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Week 10 • Geography & Channels", layout="wide")
st.title("Week 10 — Geography & Channels Deep Dive")
st.caption("World maps + deep visuals + statistical tests + recommendations (auto-load from repo).")

BASE = pathlib.Path(__file__).parent
DATA_FILE = BASE / "Combined_Sales_2025 (2).csv"

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

@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    try:
        d = pd.read_csv(path)
    except Exception:
        d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = d.columns.str.strip()
    return d

def country_to_iso3(name: str):
    try:
        import pycountry
    except Exception:
        return None
    s = normalize_country(name)
    if not s:
        return None
    direct = {
        "United States": "USA",
        "United Kingdom": "GBR",
        "Russia": "RUS",
        "South Korea": "KOR",
        "Hong Kong": "HKG",
    }
    if s in direct:
        return direct[s]
    try:
        c = pycountry.countries.lookup(s)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None

def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return float((H - k + 1) / (n - k))

def cramers_v(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = ct.shape
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt((chi2 / n) / denom))

def holm_adjust(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    prev = 0.0
    for i, idx in enumerate(order):
        val = (m - i) * pvals[idx]
        val = min(1.0, max(val, prev))
        adj[idx] = val
        prev = val
    return adj.tolist()

def gini(x: np.ndarray) -> float:
    x = np.array(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    x = np.sort(np.maximum(x, 0))
    s = x.sum()
    if s == 0:
        return 0.0
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() / (n * s)) - (n + 1) / n)

def download_html(fig: go.Figure, name: str, label: str):
    import plotly.io as pio
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button(label, data=html, file_name=name, mime="text/html")

if not DATA_FILE.exists():
    st.error("Dataset file not found. Put 'Combined_Sales_2025 (2).csv' in the SAME folder as app.py in your repo.")
    st.stop()

df = load_data(str(DATA_FILE))

required = ["Sale ID", "Date", "Country", "Channel", "Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Shipped Date"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Shipped Date"] = pd.to_datetime(df["Shipped Date"], errors="coerce")
df["Country"] = _clean_str(df["Country"]).apply(normalize_country)
df["Channel"] = _clean_str(df["Channel"])
df["Customer Type"] = _clean_str(df.get("Customer Type", pd.Series([np.nan] * len(df))))
df["Product Type"] = _clean_str(df.get("Product Type", pd.Series([np.nan] * len(df))))
df["Lead Source"] = _clean_str(df.get("Lead Source", pd.Series([np.nan] * len(df))))
df["City"] = _clean_str(df.get("City", pd.Series([np.nan] * len(df))))
df["Consignment? (Y/N)"] = _clean_str(df.get("Consignment? (Y/N)", pd.Series([np.nan] * len(df))))

for c in ["Price (CAD)", "Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "length", "width", "weight", "Color Count (#)"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

df["Net Sales (CAD)"] = (df["Price (CAD)"] - df["Discount (CAD)"]).clip(lower=0)
df["Total Collected (CAD)"] = (df["Net Sales (CAD)"] + df["Shipping (CAD)"].fillna(0) + df["Taxes Collected (CAD)"].fillna(0)).clip(lower=0)
df["Ship Lag (days)"] = (df["Shipped Date"] - df["Date"]).dt.days
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

st.sidebar.header("Filters")
min_d, max_d = df["Date"].min(), df["Date"].max()
dr = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))

countries = sorted([c for c in df["Country"].dropna().unique().tolist() if c])
channels = sorted([c for c in df["Channel"].dropna().unique().tolist() if c])
cust_types = sorted([c for c in df["Customer Type"].dropna().unique().tolist() if c])
prod_types = sorted([c for c in df["Product Type"].dropna().unique().tolist() if c])

sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])
sel_cust = st.sidebar.multiselect("Customer Type", cust_types, default=[])
sel_prod = st.sidebar.multiselect("Product Type", prod_types, default=[])

metric_choice = st.sidebar.selectbox("Metric", ["Total Collected (CAD)", "Net Sales (CAD)", "Price (CAD)"], index=0)
top_n = st.sidebar.slider("Top N (countries)", 5, 30, 12)

start = pd.to_datetime(dr[0])
end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

f = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
if sel_countries:
    f = f[f["Country"].isin(sel_countries)]
if sel_channels:
    f = f[f["Channel"].isin(sel_channels)]
if sel_cust:
    f = f[f["Customer Type"].isin(sel_cust)]
if sel_prod:
    f = f[f["Product Type"].isin(sel_prod)]

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

metric = metric_choice
total = float(f[metric].sum())
orders = int(len(f))
aov = float(f[metric].mean())
median = float(f[metric].median())

country_totals = f.groupby("Country")[metric].sum().sort_values(ascending=False)
channel_totals = f.groupby("Channel")[metric].sum().sort_values(ascending=False)

anchor_country = country_totals.index[0] if len(country_totals) else "-"
anchor_share = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
top3_share = float(country_totals.head(3).sum() / country_totals.sum()) if country_totals.sum() else np.nan
gini_c = gini(country_totals.values) if len(country_totals) else np.nan

anchor_channel = channel_totals.index[0] if len(channel_totals) else "-"
anchor_channel_share = float(channel_totals.iloc[0] / channel_totals.sum()) if channel_totals.sum() else np.nan

cons_rate = float((f["Consignment? (Y/N)"].str.upper().eq("Y").mean()) * 100) if "Consignment? (Y/N)" in f.columns else np.nan
avg_ship_lag = float(f["Ship Lag (days)"].mean()) if f["Ship Lag (days)"].notna().any() else np.nan

k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.metric("Orders", f"{orders:,}")
k2.metric("Total", f"{total:,.0f}")
k3.metric("Avg Order", f"{aov:,.0f}")
k4.metric("Median", f"{median:,.0f}")
k5.metric("Top Country", anchor_country if anchor_country else "-")
k6.metric("Top Channel", anchor_channel if anchor_channel else "-")
k7.metric("Consignment %", f"{cons_rate:.1f}%" if np.isfinite(cons_rate) else "-")
k8.metric("Avg Ship Lag", f"{avg_ship_lag:.1f}d" if np.isfinite(avg_ship_lag) else "-")

st.divider()

tabs = st.tabs(["1) Overview", "2) World Maps", "3) Geography", "4) Channels", "5) Mix", "6) Time", "7) Stats", "8) Data"])

with tabs[0]:
    st.subheader("Key insights")
    lines = []
    if np.isfinite(anchor_share):
        lines.append(f"- Concentration: **{anchor_country}** contributes ~**{anchor_share:.1%}** of {metric}.")
    if np.isfinite(top3_share):
        lines.append(f"- Top 3 countries contribute ~**{top3_share:.1%}** of total.")
    if np.isfinite(gini_c):
        lines.append(f"- Country concentration (Gini): **{gini_c:.2f}** (higher = more concentrated).")
    if np.isfinite(anchor_channel_share):
        lines.append(f"- Top channel (**{anchor_channel}**) contributes ~**{anchor_channel_share:.1%}** of total.")
    st.markdown("\n".join(lines) if lines else "- -")

    st.subheader("Recommendations")
    recs = []
    if np.isfinite(anchor_share) and anchor_share >= 0.5:
        recs.append("- Protect the anchor market (inventory + fulfillment + pricing discipline) because it drives most results.")
        recs.append("- Scale 2–3 secondary markets using their strongest channel (see heatmap + mix).")
    else:
        recs.append("- Set market tiers (anchor/growth/test) and align channel strategy by tier.")
    if np.isfinite(gini_c) and gini_c >= 0.6:
        recs.append("- Concentration is high: define growth targets for the next tier markets and track weekly progress.")
    if np.isfinite(avg_ship_lag) and avg_ship_lag >= 14:
        recs.append("- Shipping lag is material: diagnose bottlenecks by channel and country; set SLA targets.")
    st.markdown("\n".join(recs) if recs else "- -")

with tabs[1]:
    st.subheader("World revenue map (choropleth)")
    agg = country_totals.reset_index().rename(columns={metric: "metric"})
    agg["share"] = agg["metric"] / agg["metric"].sum()
    agg["iso3"] = agg["Country"].apply(country_to_iso3)

    mapped = agg[agg["iso3"].notna()].copy()
    if len(mapped) >= 2:
        fig = px.choropleth(
            mapped,
            locations="iso3",
            color="metric",
            hover_name="Country",
            hover_data={"metric": ":,.0f", "share": ".1%"},
            projection="natural earth",
            title=f"World Map — {metric} by Country"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "01_world_map.html", "Download world map (HTML)")
    else:
        fig = px.choropleth(
            agg,
            locations="Country",
            locationmode="country names",
            color="metric",
            hover_name="Country",
            hover_data={"metric": ":,.0f", "share": ".1%"},
            projection="natural earth",
            title=f"World Map — {metric} by Country"
        )
        fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(fig, use_container_width=True)
        download_html(fig, "01_world_map_country_names.html", "Download world map (HTML)")

    st.markdown("**Insight:** This shows global revenue concentration at a glance.\n\n"
                "**Recommendation:** Defend anchor markets; choose 2–3 growth markets and scale the channels that already win locally.")

    st.subheader("Top markets table")
    top_tbl = agg.sort_values("metric", ascending=False).head(15).copy()
    top_tbl["metric"] = top_tbl["metric"].round(0)
    top_tbl.insert(0, "#", range(1, len(top_tbl) + 1))
    st.dataframe(top_tbl.set_index("#")[["Country", "metric", "share"]], use_container_width=True)

    st.subheader("Bubble map (top countries)")
    top_b = agg.sort_values("metric", ascending=False).head(top_n).copy()
    top_b["iso3"] = top_b["Country"].apply(country_to_iso3)
    use_iso = top_b["iso3"].notna().mean() >= 0.7
    fig2 = px.scatter_geo(
        top_b,
        locations="iso3" if use_iso else "Country",
        locationmode="ISO-3" if use_iso else "country names",
        size="metric",
        hover_name="Country",
        hover_data={"metric": ":,.0f", "share": ".1%"},
        title=f"Bubble Map — Top {top_n} Countries"
    )
    fig2.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    st.plotly_chart(fig2, use_container_width=True)
    download_html(fig2, "02_bubble_map.html", "Download bubble map (HTML)")

with tabs[2]:
    st.subheader("Top countries (bar)")
    top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "metric"})
    fig = px.bar(top_c, x="Country", y="metric", title=f"Top {top_n} Countries by {metric}")
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "03_top_countries.html", "Download chart (HTML)")
    st.markdown("**Insight:** Identifies anchor vs secondary markets.\n\n"
                "**Recommendation:** Protect anchor markets and build a specific growth plan for the next tier.")

    st.subheader("Pareto concentration (cumulative share)")
    pareto = country_totals.reset_index().rename(columns={metric: "metric"})
    pareto["rank"] = np.arange(1, len(pareto) + 1)
    pareto["cum_share"] = pareto["metric"].cumsum() / pareto["metric"].sum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=pareto["rank"][:top_n], y=pareto["metric"][:top_n], name="Metric"))
    fig2.add_trace(go.Scatter(x=pareto["rank"][:top_n], y=pareto["cum_share"][:top_n], name="Cumulative Share", yaxis="y2"))
    fig2.update_layout(
        title=f"Pareto — Top {top_n} Country Ranks",
        xaxis=dict(title="Country rank"),
        yaxis=dict(title=f"{metric}"),
        yaxis2=dict(title="Cumulative share", overlaying="y", side="right", tickformat=".0%")
    )
    st.plotly_chart(fig2, use_container_width=True)
    download_html(fig2, "04_pareto.html", "Download chart (HTML)")
    t5 = float(country_totals.head(min(5, len(country_totals))).sum() / country_totals.sum()) if country_totals.sum() else np.nan
    st.markdown(f"**Insight:** Top {min(5, len(country_totals))} countries contribute ~**{t5:.1%}**.\n\n"
                "**Recommendation:** If concentration is high, reduce dependency by scaling the next tier markets.")

    st.subheader("Cities (top 15)")
    if f["City"].notna().any():
        city_totals = f.groupby("City")[metric].sum().sort_values(ascending=False).head(15).reset_index().rename(columns={metric: "metric"})
        fig3 = px.bar(city_totals, x="City", y="metric", title="Top Cities by Metric (Top 15)")
        fig3.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "05_top_cities.html", "Download chart (HTML)")
        st.markdown("**Insight:** Shows which cities are the biggest pockets of demand.\n\n"
                    "**Recommendation:** Target events/partners/logistics in these cities first.")
    else:
        st.info("City column is empty for the filtered subset.")

with tabs[3]:
    st.subheader("Revenue by channel (bar)")
    ch = channel_totals.reset_index().rename(columns={metric: "metric"})
    fig = px.bar(ch, x="Channel", y="metric", title=f"{metric} by Channel")
    fig.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "06_channel_bar.html", "Download chart (HTML)")
    st.markdown("**Insight:** Shows which channel is actually driving value.\n\n"
                "**Recommendation:** Invest in the top channels; fix or redesign low-performing ones.")

    st.subheader("Order value distribution by channel (box)")
    fig2 = px.box(f, x="Channel", y=metric, points="outliers", title=f"Distribution of {metric} by Channel")
    fig2.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig2, use_container_width=True)
    download_html(fig2, "07_channel_box.html", "Download chart (HTML)")
    st.markdown("**Insight:** Shows volatility/outliers by channel.\n\n"
                "**Recommendation:** For high-variance channels, set pricing/discount guardrails and forecasting buffers.")

    st.subheader("Lead source (top 12)")
    if f["Lead Source"].notna().any():
        ls = f.groupby("Lead Source")[metric].sum().sort_values(ascending=False).head(12).reset_index().rename(columns={metric: "metric"})
        fig3 = px.bar(ls, x="Lead Source", y="metric", title=f"Top Lead Sources by {metric} (Top 12)")
        fig3.update_layout(xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "08_lead_source.html", "Download chart (HTML)")
        st.markdown("**Insight:** Identifies which sources generate the most value.\n\n"
                    "**Recommendation:** Shift marketing and partnerships toward high-value lead sources.")
    else:
        st.info("Lead Source column is empty for the filtered subset.")

with tabs[4]:
    st.subheader("Country × Channel heatmap")
    top_idx = country_totals.head(top_n).index
    df_top = f[f["Country"].isin(top_idx)]
    pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
    fig = px.imshow(pv, title=f"Heatmap: {metric} by Country & Channel (Top {top_n} countries)",
                    labels=dict(x="Channel", y="Country", color=metric))
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "09_heatmap.html", "Download heatmap (HTML)")
    st.markdown("**Insight:** Shows which channels dominate per country.\n\n"
                "**Recommendation:** Build a per-market channel playbook instead of one global playbook.")

    st.subheader("Channel mix share by country (stacked)")
    mix = df_top.groupby(["Country", "Channel"])[metric].sum().reset_index().rename(columns={metric: "metric"})
    mix["country_total"] = mix.groupby("Country")["metric"].transform("sum")
    mix["share"] = mix["metric"] / mix["country_total"]
    fig2 = px.bar(mix, x="Country", y="share", color="Channel", barmode="stack",
                  title="Channel Mix by Country (Share of Total)", labels={"share": "Share"})
    fig2.update_layout(yaxis_tickformat=".0%", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig2, use_container_width=True)
    download_html(fig2, "10_mix_share.html", "Download mix chart (HTML)")
    st.markdown("**Insight:** Shows dependence on single channels per market.\n\n"
                "**Recommendation:** Where one channel is >70% of a market, diversify with a controlled test of a second channel.")

    st.subheader("Treemap: country → channel")
    treemap = df_top.groupby(["Country", "Channel"])[metric].sum().reset_index().rename(columns={metric: "metric"})
    fig3 = px.treemap(treemap, path=["Country", "Channel"], values="metric", title="Treemap: Contribution by Country & Channel")
    st.plotly_chart(fig3, use_container_width=True)
    download_html(fig3, "11_treemap.html", "Download treemap (HTML)")

with tabs[5]:
    st.subheader("Monthly trend (overall)")
    ts = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "metric"})
    fig = px.line(ts, x="Month", y="metric", title=f"Monthly {metric}")
    st.plotly_chart(fig, use_container_width=True)
    download_html(fig, "12_monthly_trend.html", "Download trend (HTML)")
    st.markdown("**Insight:** Shows seasonality and spikes.\n\n"
                "**Recommendation:** Investigate spikes by drilling into country/channel filters and matching to events/campaigns.")

    st.subheader("Monthly trend by channel (top 6)")
    top_channels = channel_totals.head(6).index
    ts2 = f[f["Channel"].isin(top_channels)].groupby(["Month", "Channel"])[metric].sum().reset_index()
    fig2 = px.line(ts2, x="Month", y=metric, color="Channel", title=f"Monthly {metric} — Top 6 Channels")
    st.plotly_chart(fig2, use_container_width=True)
    download_html(fig2, "13_monthly_channels.html", "Download chart (HTML)")

    st.subheader("Ship lag distribution")
    if f["Ship Lag (days)"].notna().any():
        fig3 = px.histogram(f, x="Ship Lag (days)", nbins=30, title="Shipping Lag (days) Distribution")
        st.plotly_chart(fig3, use_container_width=True)
        download_html(fig3, "14_ship_lag.html", "Download chart (HTML)")
        st.markdown("**Insight:** Shows operational speed spread.\n\n"
                    "**Recommendation:** Segment lag by channel/country and set SLA improvements.")
    else:
        st.info("Ship lag not available for the filtered subset.")

with tabs[6]:
    st.subheader("A) Country ↔ Channel association (chi-square + Cramer's V)")
    top_for_test = country_totals.head(min(15, len(country_totals))).index
    f_test = f.copy()
    f_test["Country (top)"] = np.where(f_test["Country"].isin(top_for_test), f_test["Country"], "Other")
    ct = pd.crosstab(f_test["Country (top)"], f_test["Channel"])
    if ct.shape[0] >= 2 and ct.shape[1] >= 2:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v(ct)
        st.write({"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": float(v)})
        st.markdown("**Insight:** If p < 0.05, channel mix differs across countries.\n\n"
                    "**Recommendation:** Use market-specific channel strategies when significant.")
    else:
        st.info("Not enough categories after filters to run chi-square.")

    st.subheader("B) Do channels differ on order value? (Kruskal–Wallis + post-hoc)")
    groups = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
    if len(groups) >= 2:
        H, p_kw = stats.kruskal(*groups.tolist())
        eps = epsilon_squared_kruskal(H, n=len(f), k=len(groups))
        st.write({"H": float(H), "p_value": float(p_kw), "epsilon_squared": float(eps)})

        chans = list(groups.index)
        pairs, pvals = [], []
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                a = f.loc[f["Channel"] == chans[i], metric].dropna()
                b = f.loc[f["Channel"] == chans[j], metric].dropna()
                if len(a) >= 10 and len(b) >= 10:
                    _, pu = stats.mannwhitneyu(a, b, alternative="two-sided")
                    pairs.append((chans[i], chans[j], len(a), len(b)))
                    pvals.append(pu)

        if pvals:
            adj = holm_adjust(pvals)
            out = pd.DataFrame({
                "channel_a": [p[0] for p in pairs],
                "channel_b": [p[1] for p in pairs],
                "n_a": [p[2] for p in pairs],
                "n_b": [p[3] for p in pairs],
                "p_raw": pvals,
                "p_holm": adj
            }).sort_values("p_holm")
            out.insert(0, "#", range(1, len(out) + 1))
            st.dataframe(out.set_index("#"), use_container_width=True)
    else:
        st.info("Not enough channels after filters.")

    st.subheader("C) Numeric drivers (Spearman correlation with metric)")
    num_cols = [c for c in ["Discount (CAD)", "Shipping (CAD)", "Taxes Collected (CAD)", "Color Count (#)", "length", "width", "weight", "Ship Lag (days)"] if c in f.columns]
    corr_rows = []
    for c in num_cols:
        x = f[c]
        y = f[metric]
        ok = x.notna() & y.notna()
        if ok.sum() >= 25:
            r, pval = stats.spearmanr(x[ok], y[ok])
            corr_rows.append((c, float(r), float(pval), int(ok.sum())))
    if corr_rows:
        corr_df = pd.DataFrame(corr_rows, columns=["variable", "spearman_r", "p_value", "n"])
        corr_df = corr_df.sort_values("spearman_r", key=lambda s: s.abs(), ascending=False)
        corr_df.insert(0, "#", range(1, len(corr_df) + 1))
        st.dataframe(corr_df.set_index("#"), use_container_width=True)
        st.markdown("**Insight:** Variables with larger |r| have stronger monotonic association with the metric.\n\n"
                    "**Recommendation:** Use this to justify where to focus (discount policy, shipping pricing, product attributes).")
    else:
        st.info("Not enough numeric data for correlation (need ~25+ usable rows).")

with tabs[7]:
    st.subheader("Filtered data (preview)")
    st.dataframe(f.head(300), use_container_width=True)

    st.subheader("Download filtered data")
    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

    st.subheader("Country × Channel KPI table")
    kpi = f.groupby(["Country", "Channel"]).agg(
        orders=("Sale ID", "count"),
        total=(metric, "sum"),
        avg=(metric, "mean"),
        median=(metric, "median"),
        consignment_rate=("Consignment? (Y/N)", lambda s: (s.str.upper().eq("Y").mean() * 100) if s.notna().any() else np.nan),
        avg_ship_lag=("Ship Lag (days)", "mean")
    ).reset_index()
    kpi["share"] = kpi["total"] / kpi["total"].sum() if kpi["total"].sum() else np.nan
    kpi = kpi.sort_values("total", ascending=False).head(40)
    kpi.insert(0, "#", range(1, len(kpi) + 1))
    st.dataframe(kpi.set_index("#"), use_container_width=True)
