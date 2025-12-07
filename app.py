import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Week 10 • Geography & Channels", layout="wide")
st.title("Week 10 — Geography & Channels Deep Dive")
st.caption("Auto-loaded dataset from repo → world maps + deep visuals + statistical tests + recommendations.")

def _clean_str(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})

def pick_col(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in cols:
            return cand
        key = str(cand).lower()
        if key in lower_map:
            return lower_map[key]
    for cand in candidates:
        pat = str(cand).lower()
        for c in cols:
            if pat in c.lower():
                return c
    return None

def numeric_candidates(df: pd.DataFrame):
    nums = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            nums.append(c)
    for c in df.columns:
        if c in nums:
            continue
        if df[c].dtype == "object":
            sample = df[c].dropna().astype(str).head(80)
            if sample.empty:
                continue
            looks = sample.str.replace(r"[^0-9\.\-]", "", regex=True)
            ok = pd.to_numeric(looks, errors="coerce").notna().mean()
            if ok >= 0.6:
                nums.append(c)
    return nums

def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s
    cleaned = s.astype(str).str.replace(r"[^0-9\.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")

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

def cramers_v_from_crosstab(ct: pd.DataFrame) -> float:
    chi2, _, _, _ = stats.chi2_contingency(ct)
    n = ct.to_numpy().sum()
    if n == 0:
        return np.nan
    r, k = ct.shape
    phi2 = chi2 / n
    denom = min(k - 1, r - 1)
    if denom <= 0:
        return np.nan
    return float(np.sqrt(phi2 / denom))

def epsilon_squared_kruskal(H: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return float((H - k + 1) / (n - k))

def top_share(series: pd.Series, top_n: int) -> float:
    s = series.sort_values(ascending=False)
    total = s.sum()
    if total == 0:
        return np.nan
    return float(s.head(top_n).sum() / total)

def gini_from_values(x: np.ndarray) -> float:
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

def fig_download_button(fig: go.Figure, filename: str, label: str):
    import plotly.io as pio
    html = pio.to_html(fig, include_plotlyjs="cdn", full_html=True).encode("utf-8")
    st.download_button(label, data=html, file_name=filename, mime="text/html")

def normalize_country_name(name: str) -> str:
    if name is None:
        return ""
    s = str(name).strip()
    if not s:
        return ""
    patches = {
        "usa": "United States",
        "u.s.a.": "United States",
        "u.s.": "United States",
        "us": "United States",
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "russia": "Russian Federation",
        "south korea": "Korea, Republic of",
        "north korea": "Korea, Democratic People's Republic of",
        "iran": "Iran, Islamic Republic of",
        "venezuela": "Venezuela, Bolivarian Republic of",
        "bolivia": "Bolivia, Plurinational State of",
        "tanzania": "Tanzania, United Republic of",
        "vietnam": "Viet Nam",
        "laos": "Lao People's Democratic Republic",
        "syria": "Syrian Arab Republic",
        "moldova": "Moldova, Republic of",
        "brunei": "Brunei Darussalam",
        "czech republic": "Czechia",
        "palestine": "Palestine, State of",
        "taiwan": "Taiwan, Province of China",
        "hong kong": "Hong Kong",
    }
    key = s.lower()
    return patches.get(key, s)

def country_to_iso3(name: str):
    try:
        import pycountry
    except Exception:
        return None
    s = normalize_country_name(name)
    if not s:
        return None
    direct = {
        "United States": "USA",
        "United Kingdom": "GBR",
        "Russia": "RUS",
        "Russian Federation": "RUS",
        "Korea, Republic of": "KOR",
        "Korea, Democratic People's Republic of": "PRK",
        "Viet Nam": "VNM",
        "Czechia": "CZE",
        "Hong Kong": "HKG",
    }
    if s in direct:
        return direct[s]
    try:
        c = pycountry.countries.lookup(s)
        return getattr(c, "alpha_3", None)
    except Exception:
        return None

BASE = pathlib.Path(__file__).parent
DATA_FILE = BASE / "Combined_Sales_2025 (2).csv"

if not DATA_FILE.exists():
    st.error("Dataset file not found. Put 'Combined_Sales_2025 (2).csv' in the SAME folder as app.py in your repo.")
    st.stop()

try:
    df_raw = pd.read_csv(DATA_FILE)
except Exception:
    df_raw = pd.read_csv(DATA_FILE, encoding="utf-8-sig")

df_raw.columns = df_raw.columns.str.strip()
st.caption("Loaded: Combined_Sales_2025 (2).csv")

with st.expander("Peek at raw columns"):
    st.write(df_raw.columns.tolist())
    st.dataframe(df_raw.head(10), use_container_width=True)

st.sidebar.header("Column mapping")
auto_country = pick_col(df_raw, ["country", "ship_country", "billing_country", "destination_country", "market"])
auto_channel = pick_col(df_raw, ["channel", "sales_channel", "source", "platform"])
auto_date = pick_col(df_raw, ["date", "sale_date", "order_date", "invoice_date", "created_at"])
auto_metric = pick_col(df_raw, ["revenue", "sales", "net_sales", "amount", "total", "cad", "price"])

num_cols = numeric_candidates(df_raw)
if auto_metric is None and num_cols:
    auto_metric = num_cols[0]

country_col = st.sidebar.selectbox("Country", ["(none)"] + df_raw.columns.tolist(),
                                   index=(df_raw.columns.get_loc(auto_country) + 1) if auto_country else 0)
channel_col = st.sidebar.selectbox("Channel", ["(none)"] + df_raw.columns.tolist(),
                                   index=(df_raw.columns.get_loc(auto_channel) + 1) if auto_channel else 0)
date_col = st.sidebar.selectbox("Date (optional)", ["(none)"] + df_raw.columns.tolist(),
                                index=(df_raw.columns.get_loc(auto_date) + 1) if auto_date else 0)
metric_col = st.sidebar.selectbox("Numeric metric", ["(none)"] + df_raw.columns.tolist(),
                                  index=(df_raw.columns.get_loc(auto_metric) + 1) if auto_metric else 0)

if metric_col == "(none)":
    st.error("Pick a numeric metric column (Revenue/Sales/Amount/etc.).")
    st.stop()
if country_col == "(none)" or channel_col == "(none)":
    st.error("Pick BOTH Country and Channel columns.")
    st.stop()

df = df_raw.copy()
df[country_col] = _clean_str(df[country_col]).apply(normalize_country_name)
df[channel_col] = _clean_str(df[channel_col])
df[metric_col] = to_numeric_series(df[metric_col])

if date_col != "(none)":
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = df[date_col].dt.to_period("M").dt.to_timestamp()
else:
    df["month"] = pd.NaT

df = df.dropna(subset=[country_col, channel_col, metric_col])

st.sidebar.header("Filters")
countries = sorted(df[country_col].dropna().unique().tolist())
channels = sorted(df[channel_col].dropna().unique().tolist())

sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])

f = df.copy()
if sel_countries:
    f = f[f[country_col].isin(sel_countries)]
if sel_channels:
    f = f[f[channel_col].isin(sel_channels)]

if date_col != "(none)" and f[date_col].notna().any():
    dmin, dmax = f[date_col].min(), f[date_col].max()
    dr = st.sidebar.date_input("Date range", value=(dmin.date(), dmax.date()))
    if isinstance(dr, tuple) and len(dr) == 2:
        start = pd.to_datetime(dr[0])
        end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        f = f[(f[date_col] >= start) & (f[date_col] <= end)]

top_n = st.sidebar.slider("Top N (countries)", min_value=5, max_value=30, value=12)

total_metric = f[metric_col].sum()
rows = len(f)
u_countries = f[country_col].nunique()
u_channels = f[channel_col].nunique()

country_totals = f.groupby(country_col)[metric_col].sum().sort_values(ascending=False)
channel_totals = f.groupby(channel_col)[metric_col].sum().sort_values(ascending=False)

anchor_country = country_totals.index[0] if len(country_totals) else "—"
anchor_share = (country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan
top3_share = top_share(country_totals, 3)
gini = gini_from_values(country_totals.values)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Rows", f"{rows:,}")
c2.metric("Total (metric)", f"{total_metric:,.0f}")
c3.metric("Countries", f"{u_countries:,}")
c4.metric("Channels", f"{u_channels:,}")
c5.metric("Anchor", f"{anchor_country}", help=f"Top share: {anchor_share:.1%}" if pd.notna(anchor_share) else None)
c6.metric("Concentration (Gini)", f"{gini:.2f}" if pd.notna(gini) else "—")

st.divider()

tab_overview, tab_maps, tab_geo, tab_channel, tab_time, tab_stats, tab_data = st.tabs(
    ["Overview", "World Maps", "Geography", "Channels", "Time", "Stats", "Data"]
)

with tab_overview:
    st.subheader("Executive Summary")
    bullets = []
    if pd.notna(anchor_share):
        bullets.append(f"Concentration: **{anchor_country}** contributes ~**{anchor_share:.1%}** of total.")
    if pd.notna(top3_share):
        bullets.append(f"Top 3 countries contribute ~**{top3_share:.1%}** of total.")
    if pd.notna(gini):
        bullets.append(f"Country concentration (Gini): **{gini:.2f}** (higher = more concentrated).")
    if len(channel_totals):
        bullets.append(f"Top channel: **{channel_totals.index[0]}** contributes ~**{(channel_totals.iloc[0]/channel_totals.sum()):.1%}** of total.")
    st.markdown("\n".join([f"- {b}" for b in bullets]) if bullets else "- Not enough data to summarize.")

    st.subheader("Recommendations")
    recs = []
    if pd.notna(anchor_share) and anchor_share >= 0.5:
        recs.append("Protect the anchor market with stock/service reliability and channel excellence.")
        recs.append("De-risk by scaling 2–3 mid-tier countries via their locally winning channels (use mix/heatmap).")
    else:
        recs.append("Define market tiers (anchor vs growth vs test) and align channel investment to each tier.")
    if pd.notna(gini) and gini >= 0.6:
        recs.append("Concentration risk is high: add explicit targets for secondary markets and track weekly channel mix.")
    st.markdown("\n".join([f"- {r}" for r in recs]))

with tab_maps:
    st.subheader("World Revenue Map (Choropleth)")
    agg = country_totals.reset_index().rename(columns={metric_col: "metric"})
    agg["share"] = agg["metric"] / agg["metric"].sum()
    agg["iso3"] = agg[country_col].apply(country_to_iso3)

    mapped = agg[agg["iso3"].notna()].copy()
    unmapped = agg[agg["iso3"].isna()].copy()

    left, right = st.columns([3, 2])

    with left:
        if len(mapped) >= 2:
            figm1 = px.choropleth(
                mapped,
                locations="iso3",
                color="metric",
                hover_name=country_col,
                hover_data={"metric": ":,.0f", "share": ".1%"},
                projection="natural earth",
                title=f"World Map — {metric_col} by Country"
            )
            figm1.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(figm1, use_container_width=True)
            fig_download_button(figm1, "world_map_metric.html", "Download world map (HTML)")
        else:
            figm1b = px.choropleth(
                agg,
                locations=country_col,
                locationmode="country names",
                color="metric",
                hover_name=country_col,
                hover_data={"metric": ":,.0f", "share": ".1%"},
                projection="natural earth",
                title=f"World Map (country-name mode) — {metric_col}"
            )
            figm1b.update_layout(margin=dict(l=0, r=0, t=50, b=0))
            st.plotly_chart(figm1b, use_container_width=True)
            fig_download_button(figm1b, "world_map_country_names.html", "Download world map (HTML)")

        st.markdown("**Insight:** Shows global revenue concentration and where the business actually makes money.\n\n"
                    "**Recommendation:** Treat top markets as anchors; select 2–3 growth markets and invest in their best-performing channels.")

    with right:
        st.subheader("Top Markets Table")
        top_tbl = agg.sort_values("metric", ascending=False).head(15).copy()
        top_tbl["metric"] = top_tbl["metric"].round(0)
        st.dataframe(top_tbl[[country_col, "metric", "share"]], use_container_width=True)

        if len(unmapped):
            st.subheader("Unmapped Countries (ISO)")
            st.dataframe(unmapped[[country_col, "metric"]].sort_values("metric", ascending=False), use_container_width=True)
            st.caption("If the world map misses countries, add `pycountry` in requirements.txt and keep country names clean.")

    st.subheader("Bubble Geo Map (Top N countries)")
    top_b = agg.sort_values("metric", ascending=False).head(top_n).copy()
    if len(top_b) >= 2:
        figm2 = px.scatter_geo(
            top_b,
            locations="iso3" if top_b["iso3"].notna().mean() > 0.5 else country_col,
            locationmode="ISO-3" if top_b["iso3"].notna().mean() > 0.5 else "country names",
            size="metric",
            hover_name=country_col,
            hover_data={"metric": ":,.0f", "share": ".1%"},
            title=f"Bubble Map — Top {top_n} Countries by {metric_col}"
        )
        figm2.update_layout(margin=dict(l=0, r=0, t=50, b=0))
        st.plotly_chart(figm2, use_container_width=True)
        fig_download_button(figm2, "bubble_geo_map.html", "Download bubble map (HTML)")

with tab_geo:
    st.subheader("Revenue by Country (Top N)")
    top_countries = country_totals.head(top_n).reset_index().rename(columns={metric_col: "metric"})
    fig1 = px.bar(top_countries, x=country_col, y="metric", title=f"Top {top_n} Countries by {metric_col}")
    fig1.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig1, use_container_width=True)
    fig_download_button(fig1, "rev_by_country.html", "Download chart (HTML)")
    st.markdown("**Insight:** Shows where performance is concentrated.\n\n"
                "**Recommendation:** Protect anchor markets; build growth plans for the next tier markets using their best channel mix.")

    st.subheader("Concentration (Pareto)")
    pareto = country_totals.reset_index().rename(columns={metric_col: "metric"})
    pareto["rank"] = np.arange(1, len(pareto) + 1)
    pareto["cum_share"] = pareto["metric"].cumsum() / pareto["metric"].sum()
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(x=pareto["rank"][:top_n], y=pareto["metric"][:top_n], name="Metric"))
    fig2.add_trace(go.Scatter(x=pareto["rank"][:top_n], y=pareto["cum_share"][:top_n], name="Cumulative Share", yaxis="y2"))
    fig2.update_layout(
        title=f"Pareto Concentration (Top {top_n} Countries)",
        xaxis=dict(title="Country rank"),
        yaxis=dict(title=f"{metric_col}"),
        yaxis2=dict(title="Cumulative share", overlaying="y", side="right", tickformat=".0%")
    )
    st.plotly_chart(fig2, use_container_width=True)
    fig_download_button(fig2, "pareto_concentration.html", "Download chart (HTML)")
    t5 = top_share(country_totals, min(5, len(country_totals)))
    st.markdown(f"**Insight:** Top {min(5, len(country_totals))} countries contribute ~**{t5:.1%}** of total.\n\n"
                "**Recommendation:** If concentration is high, reduce risk by scaling 2–3 mid-tier markets and tracking monthly progress.")

with tab_channel:
    st.subheader("Revenue by Channel")
    ch = channel_totals.reset_index().rename(columns={metric_col: "metric"})
    fig3 = px.bar(ch, x=channel_col, y="metric", title=f"{metric_col} by Channel")
    fig3.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig3, use_container_width=True)
    fig_download_button(fig3, "metric_by_channel.html", "Download chart (HTML)")
    st.markdown("**Insight:** Identifies which channel currently drives most value.\n\n"
                "**Recommendation:** Invest in the strongest channel where it wins; don’t force identical strategy across every country.")

    st.subheader("Heatmap — Country × Channel")
    top_idx = country_totals.head(top_n).index
    df_top = f[f[country_col].isin(top_idx)]
    pv = df_top.pivot_table(values=metric_col, index=country_col, columns=channel_col, aggfunc="sum", fill_value=0)
    fig4 = px.imshow(pv, title=f"Heatmap: {metric_col} by Country & Channel (Top {top_n} countries)",
                     labels=dict(x="Channel", y="Country", color=metric_col))
    st.plotly_chart(fig4, use_container_width=True)
    fig_download_button(fig4, "heatmap_country_channel.html", "Download chart (HTML)")
    st.markdown("**Insight:** Shows dominant channels per country.\n\n"
                "**Recommendation:** Build a per-market channel playbook based on what already performs locally.")

    st.subheader("Channel Mix — Share of Country Total")
    mix = (df_top.groupby([country_col, channel_col])[metric_col].sum()
           .reset_index().rename(columns={metric_col: "metric"}))
    mix["country_total"] = mix.groupby(country_col)["metric"].transform("sum")
    mix["share"] = mix["metric"] / mix["country_total"]
    fig5 = px.bar(mix, x=country_col, y="share", color=channel_col, barmode="stack",
                  title="Channel Mix by Country (Share of Total)", labels={"share": "Share"})
    fig5.update_layout(yaxis_tickformat=".0%", xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig5, use_container_width=True)
    fig_download_button(fig5, "channel_mix_share.html", "Download chart (HTML)")
    st.markdown("**Insight:** Countries show different channel dependence.\n\n"
                "**Recommendation:** Optimize the winning channel in each market first; then test a secondary channel to diversify.")

    st.subheader("Treemap — Country → Channel contribution")
    treemap = (df_top.groupby([country_col, channel_col])[metric_col].sum()
               .reset_index().rename(columns={metric_col: "metric"}))
    fig6 = px.treemap(treemap, path=[country_col, channel_col], values="metric", title="Treemap: Market & Channel Contribution")
    st.plotly_chart(fig6, use_container_width=True)
    fig_download_button(fig6, "treemap_market_channel.html", "Download chart (HTML)")
    st.markdown("**Insight:** Shows which market-channel combos make up the business.\n\n"
                "**Recommendation:** Focus on the 2–3 biggest combos for short-term wins; build experiments for smaller combos.")

    st.subheader("Distribution — Metric by Channel (boxplot)")
    fig7 = px.box(f, x=channel_col, y=metric_col, points="outliers", title=f"Distribution of {metric_col} by Channel")
    fig7.update_layout(xaxis={"categoryorder": "total descending"})
    st.plotly_chart(fig7, use_container_width=True)
    fig_download_button(fig7, "box_metric_by_channel.html", "Download chart (HTML)")
    st.markdown("**Insight:** Some channels can be volatile with outliers.\n\n"
                "**Recommendation:** Add guardrails (pricing/order thresholds) and forecasting buffers for high-variance channels.")

with tab_time:
    st.subheader("Time Series")
    if date_col == "(none)" or not f[date_col].notna().any():
        st.info("No usable date column selected.")
    else:
        ts_df = f.groupby("month")[metric_col].sum().reset_index().rename(columns={metric_col: "metric"})
        figt1 = px.line(ts_df, x="month", y="metric", title=f"Monthly {metric_col} (Overall)")
        st.plotly_chart(figt1, use_container_width=True)
        fig_download_button(figt1, "time_monthly_overall.html", "Download chart (HTML)")
        st.markdown("**Insight:** Shows overall trend + spikes.\n\n"
                    "**Recommendation:** Investigate spikes/drops by filtering to specific markets/channels and checking operational causes.")

        top5 = country_totals.head(5).index
        ts2 = f[f[country_col].isin(top5)].groupby(["month", country_col])[metric_col].sum().reset_index()
        figt2 = px.line(ts2, x="month", y=metric_col, color=country_col, title=f"Monthly {metric_col} — Top 5 Countries")
        st.plotly_chart(figt2, use_container_width=True)
        fig_download_button(figt2, "time_top5_countries.html", "Download chart (HTML)")
        st.markdown("**Insight:** Identifies growth vs decline markets.\n\n"
                    "**Recommendation:** Allocate effort to consistently upward markets; fix declines via channel mix or availability improvements.")

with tab_stats:
    st.subheader("Statistical Analysis")
    st.markdown("### A) Chi-square: Country ↔ Channel association")
    ct = pd.crosstab(f[country_col], f[channel_col])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        st.warning("Need at least 2 countries and 2 channels after filters.")
    else:
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        v = cramers_v_from_crosstab(ct)
        st.write({"chi2": float(chi2), "p_value": float(p), "dof": int(dof), "cramers_v": float(v)})
        if p < 0.05:
            st.success("Channel mix differs significantly by country.")
        else:
            st.info("No strong evidence channel mix differs by country (within current filtered subset).")
        st.markdown("**Recommendation:** If significant (and V non-trivial), use market-specific channel strategies.")

    st.markdown("### B) Kruskal–Wallis: Metric differs by Channel?")
    g = f.dropna(subset=[metric_col, channel_col]).groupby(channel_col)[metric_col].apply(lambda x: x.values)
    if len(g) < 2:
        st.warning("Need at least 2 channels to run Kruskal–Wallis.")
    else:
        H, p_kw = stats.kruskal(*g.tolist())
        eps = epsilon_squared_kruskal(H, n=len(f), k=len(g))
        st.write({"H": float(H), "p_value": float(p_kw), "epsilon_squared": float(eps)})
        if p_kw < 0.05:
            st.success("Channels differ significantly on the metric.")
        else:
            st.info("No strong evidence channels differ on the metric (within filtered subset).")
        st.markdown("**Recommendation:** If significant, prioritize scaling or improving channels with higher typical value and stable variance.")

        st.markdown("#### Post-hoc pairwise Mann–Whitney U (Holm correction)")
        chans = list(g.index)
        pairs, pvals = [], []
        for i in range(len(chans)):
            for j in range(i + 1, len(chans)):
                a = f.loc[f[channel_col] == chans[i], metric_col].dropna()
                b = f.loc[f[channel_col] == chans[j], metric_col].dropna()
                if len(a) >= 10 and len(b) >= 10:
                    _, p_u = stats.mannwhitneyu(a, b, alternative="two-sided")
                    pairs.append((chans[i], chans[j], len(a), len(b)))
                    pvals.append(p_u)
        if not pvals:
            st.info("Not enough observations per channel for pairwise tests (need ~10+ per group).")
        else:
            adj = holm_adjust(pvals)
            out = pd.DataFrame({
                "channel_a": [p[0] for p in pairs],
                "channel_b": [p[1] for p in pairs],
                "n_a": [p[2] for p in pairs],
                "n_b": [p[3] for p in pairs],
                "p_raw": pvals,
                "p_holm": adj
            }).sort_values("p_holm")
            st.dataframe(out, use_container_width=True)
            st.markdown("**Recommendation:** Focus action on pairs with smallest adjusted p-values (most reliable differences).")

    st.markdown("### C) Kruskal–Wallis: Metric differs by Country (top countries)?")
    topk = country_totals.head(min(top_n, 15)).index
    f_top = f[f[country_col].isin(topk)]
    g2 = f_top.groupby(country_col)[metric_col].apply(lambda x: x.values)
    if len(g2) < 2:
        st.warning("Need at least 2 countries to run Kruskal–Wallis.")
    else:
        H2, p2 = stats.kruskal(*g2.tolist())
        eps2 = epsilon_squared_kruskal(H2, n=len(f_top), k=len(g2))
        st.write({"H": float(H2), "p_value": float(p2), "epsilon_squared": float(eps2)})
        st.markdown("**Recommendation:** If significant, tailor targets and channel focus per market tier; don’t assume one-size-fits-all across countries.")

with tab_data:
    st.subheader("Filtered data preview")
    st.dataframe(f.head(300), use_container_width=True)
    st.subheader("Download filtered data (CSV)")
    csv = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")
    st.subheader("Summaries")
    colA, colB = st.columns(2)
    with colA:
        st.write("By Country")
        st.dataframe(country_totals.reset_index().rename(columns={metric_col: "total"}).head(30), use_container_width=True)
    with colB:
        st.write("By Channel")
        st.dataframe(channel_totals.reset_index().rename(columns={metric_col: "total"}).head(30), use_container_width=True)
