import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

st.set_page_config(page_title="Week 10 • Geography & Channels", layout="wide")
st.title("Week 10 — Geography & Channels Deep Dive")
st.caption("Auto-load from repo • World map • Country/City shipping lag • Simple stats • CAD ($) everywhere")

BASE = pathlib.Path(__file__).parent
DATA_FILE = BASE / "Combined_Sales_2025 (2).csv"

REQ = [
    "Sale ID","Date","Product Type","Customer Type","Country","City","Channel","Lead Source",
    "Price (CAD)","Discount (CAD)","Shipping (CAD)","Taxes Collected (CAD)","Shipped Date",
    "Consignment? (Y/N)","Color Count (#)","length","width","weight"
]

def dash(x):
    if x is None:
        return "-"
    if isinstance(x, float) and not np.isfinite(x):
        return "-"
    s = str(x).strip()
    return s if s else "-"

def money(x, decimals=0):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    try:
        return f"${float(x):,.{decimals}f} CAD"
    except Exception:
        return "-"

def pct(x, decimals=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "-"
    try:
        return f"{float(x)*100:.{decimals}f}%"
    except Exception:
        return "-"

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

def rankify(df: pd.DataFrame, start=1):
    out = df.copy()
    out.insert(0, "#", range(start, start + len(out)))
    return out

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

if not DATA_FILE.exists():
    st.error("Dataset file not found. Put 'Combined_Sales_2025 (2).csv' in the SAME folder as app.py in your repo.")
    st.stop()

df = load_data(str(DATA_FILE))
missing = [c for c in REQ if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["Shipped Date"] = pd.to_datetime(df["Shipped Date"], errors="coerce")

for c in ["Country","City","Channel","Customer Type","Product Type","Lead Source","Consignment? (Y/N)"]:
    df[c] = df[c].astype(str).str.strip()

df["Country"] = df["Country"].apply(normalize_country)
df["Consignment? (Y/N)"] = df["Consignment? (Y/N)"].str.upper()

for c in ["Price (CAD)","Discount (CAD)","Shipping (CAD)","Taxes Collected (CAD)","Color Count (#)","length","width","weight"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df["Net Sales (CAD)"] = (df["Price (CAD)"] - df["Discount (CAD)"]).clip(lower=0)
df["Total Collected (CAD)"] = (df["Net Sales (CAD)"] + df["Shipping (CAD)"].fillna(0) + df["Taxes Collected (CAD)"].fillna(0)).clip(lower=0)
df["Discount Rate"] = np.where(df["Price (CAD)"] > 0, df["Discount (CAD)"] / df["Price (CAD)"], np.nan)

df["Ship Lag Raw (days)"] = (df["Shipped Date"] - df["Date"]).dt.days
df["Ship Lag Valid (days)"] = np.where(df["Ship Lag Raw (days)"] >= 0, df["Ship Lag Raw (days)"], np.nan)
df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()

st.sidebar.header("Filters")
min_d = df["Date"].min()
max_d = df["Date"].max()
dr = st.sidebar.date_input("Date range", value=(min_d.date(), max_d.date()))
if not isinstance(dr, tuple):
    dr = (dr, dr)
start = pd.to_datetime(dr[0])
end = pd.to_datetime(dr[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

metric = st.sidebar.selectbox("Metric ($ CAD)", ["Total Collected (CAD)", "Net Sales (CAD)", "Price (CAD)"], index=0)
exclude_negative_shiplag = st.sidebar.toggle("Exclude negative ship lag (recommended)", value=True)

countries = sorted([c for c in df["Country"].dropna().unique().tolist() if c])
channels = sorted([c for c in df["Channel"].dropna().unique().tolist() if c])
cust_types = sorted([c for c in df["Customer Type"].dropna().unique().tolist() if c])
prod_types = sorted([c for c in df["Product Type"].dropna().unique().tolist() if c])

sel_countries = st.sidebar.multiselect("Countries", countries, default=[])
sel_channels = st.sidebar.multiselect("Channels", channels, default=[])
sel_cust = st.sidebar.multiselect("Customer Type", cust_types, default=[])
sel_prod = st.sidebar.multiselect("Product Type", prod_types, default=[])

df0 = df[(df["Date"] >= start) & (df["Date"] <= end)].copy()
if sel_countries:
    df0 = df0[df0["Country"].isin(sel_countries)]
if sel_channels:
    df0 = df0[df0["Channel"].isin(sel_channels)]
if sel_cust:
    df0 = df0[df0["Customer Type"].isin(sel_cust)]
if sel_prod:
    df0 = df0[df0["Product Type"].isin(sel_prod)]

cities = sorted([c for c in df0["City"].dropna().unique().tolist() if c])
sel_cities = st.sidebar.multiselect("Cities (optional)", cities, default=[])

f = df0.copy()
if sel_cities:
    f = f[f["City"].isin(sel_cities)]

if f.empty:
    st.warning("No rows match the current filters.")
    st.stop()

lag_col = "Ship Lag Valid (days)" if exclude_negative_shiplag else "Ship Lag Raw (days)"

total = float(f[metric].sum())
orders = int(len(f))
aov = float(f[metric].mean())
median_val = float(f[metric].median())
u_countries = int(f["Country"].nunique())
u_channels = int(f["Channel"].nunique())

country_totals = f.groupby("Country")[metric].sum().sort_values(ascending=False)
channel_totals = f.groupby("Channel")[metric].sum().sort_values(ascending=False)

top_country = country_totals.index[0] if len(country_totals) else "-"
top_channel = channel_totals.index[0] if len(channel_totals) else "-"

neg_lag_rows = int((f["Ship Lag Raw (days)"] < 0).sum())
avg_lag = float(np.nanmean(f[lag_col].values)) if f[lag_col].notna().any() else np.nan
cons_rate = float((f["Consignment? (Y/N)"].eq("Y").mean()) * 100)

gini_c = gini(country_totals.values) if len(country_totals) else np.nan
top_share = float(country_totals.iloc[0] / country_totals.sum()) if country_totals.sum() else np.nan

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
c1.metric("Orders", f"{orders:,}")
c2.metric("Total", money(total, 0).replace(" CAD",""))
c3.metric("Avg Order", money(aov, 0).replace(" CAD",""))
c4.metric("Median", money(median_val, 0).replace(" CAD",""))
c5.metric("Top Country", dash(top_country))
c6.metric("Top Channel", dash(top_channel))
c7.metric("Avg Ship Lag", f"{avg_lag:.1f} days" if np.isfinite(avg_lag) else "-")
c8.metric("Neg Ship Lag Rows", f"{neg_lag_rows:,}")

st.divider()

tabs = st.tabs(["1) Overview","2) World Map","3) Geography & Channels","4) Shipping Lag (Country+City)","5) Time","6) Stats (Simple)","7) Data (Clean)"])

with tabs[0]:
    st.subheader("What this says (simple)")
    a = []
    if np.isfinite(top_share):
        a.append(f"- Concentration: **{top_country}** contributes about **{top_share*100:.1f}%** of {metric}.")
    if np.isfinite(gini_c):
        a.append(f"- Country concentration (Gini): **{gini_c:.2f}** (higher = more concentrated).")
    a.append(f"- Consignment share: **{cons_rate:.1f}%** of orders.")
    if neg_lag_rows > 0:
        a.append(f"- Data note: **{neg_lag_rows}** rows have negative ship lag (chart uses {'clean' if exclude_negative_shiplag else 'raw'} lag).")
    st.markdown("\n".join(a) if a else "-")

    st.subheader("Recommendations")
    r = []
    if np.isfinite(top_share) and top_share >= 0.5:
        r.append("- Protect the anchor country: inventory + fulfillment + channel execution.")
        r.append("- Grow 2–3 next-tier countries using their strongest channel (see heatmap + mix).")
    else:
        r.append("- Set market tiers (anchor/growth/test) and align channel strategy by tier.")
    if neg_lag_rows > 0:
        r.append("- Fix date logic causing negative ship lag so operational KPIs are reliable.")
    st.markdown("\n".join(r) if r else "-")

with tabs[1]:
    st.subheader(f"World revenue map — {metric} ($ CAD)")
    agg = country_totals.reset_index().rename(columns={metric: "metric"})
    agg["share"] = agg["metric"] / agg["metric"].sum()
    fig = px.choropleth(
        agg,
        locations="Country",
        locationmode="country names",
        color="metric",
        hover_name="Country",
        hover_data={"metric": ":$,.0f", "share": ".1%"},
        projection="natural earth",
        title=f"World Map — {metric}"
    )
    fig.update_layout(margin=dict(l=0, r=0, t=60, b=0))
    fig.update_traces(hovertemplate="<b>%{hovertext}</b><br>"+f"{metric}: %{customdata[0]:$,.0f} CAD<br>Share: %{customdata[1]:.1%}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top markets table")
    top_tbl = agg.sort_values("metric", ascending=False).head(15).copy()
    top_tbl = rankify(top_tbl, 1)
    top_tbl["metric"] = top_tbl["metric"].round(0)
    st.dataframe(top_tbl.set_index("#")[["Country","metric","share"]], use_container_width=True)

with tabs[2]:
    left, right = st.columns(2)

    with left:
        st.subheader(f"Top countries by {metric} ($ CAD)")
        top_n = st.slider("Top N countries (for charts)", 5, 30, 12, key="topn_c")
        top_c = country_totals.head(top_n).reset_index().rename(columns={metric: "metric"})
        fig1 = px.bar(top_c, x="Country", y="metric", title=f"Top {top_n} Countries")
        fig1.update_layout(xaxis={"categoryorder": "total descending"})
        fig1.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Country concentration (Pareto)")
        pareto = country_totals.reset_index().rename(columns={metric: "metric"})
        pareto["rank"] = np.arange(1, len(pareto) + 1)
        pareto["cum_share"] = pareto["metric"].cumsum() / pareto["metric"].sum()
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=pareto["rank"][:top_n], y=pareto["metric"][:top_n], name="Value"))
        fig2.add_trace(go.Scatter(x=pareto["rank"][:top_n], y=pareto["cum_share"][:top_n], name="Cumulative Share", yaxis="y2"))
        fig2.update_layout(
            title=f"Pareto (Top {top_n} Country Ranks)",
            xaxis=dict(title="Country rank (1 = biggest)"),
            yaxis=dict(title=f"{metric} ($ CAD)"),
            yaxis2=dict(title="Cumulative share", overlaying="y", side="right", tickformat=".0%")
        )
        st.plotly_chart(fig2, use_container_width=True)

    with right:
        st.subheader(f"{metric} by channel ($ CAD)")
        ch = channel_totals.reset_index().rename(columns={metric: "metric"})
        fig3 = px.bar(ch, x="Channel", y="metric", title=f"{metric} by Channel")
        fig3.update_layout(xaxis={"categoryorder": "total descending"})
        fig3.update_traces(hovertemplate="<b>%{x}</b><br>Value: %{y:$,.0f} CAD<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader("Country × Channel heatmap (Top countries)")
        top_idx = country_totals.head(top_n).index
        df_top = f[f["Country"].isin(top_idx)]
        pv = df_top.pivot_table(values=metric, index="Country", columns="Channel", aggfunc="sum", fill_value=0)
        fig4 = px.imshow(pv, title=f"Heatmap: {metric} ($ CAD)", labels=dict(x="Channel", y="Country", color=metric))
        st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Channel mix share by country")
        mix = df_top.groupby(["Country","Channel"])[metric].sum().reset_index().rename(columns={metric:"metric"})
        mix["country_total"] = mix.groupby("Country")["metric"].transform("sum")
        mix["share"] = mix["metric"] / mix["country_total"]
        fig5 = px.bar(mix, x="Country", y="share", color="Channel", barmode="stack", title="Channel Mix (Share of Country Total)")
        fig5.update_layout(yaxis_tickformat=".0%", xaxis={"categoryorder": "total descending"})
        st.plotly_chart(fig5, use_container_width=True)

with tabs[3]:
    st.subheader("Shipping Lag tied to BOTH Country + City")
    top_countries_for_lag = st.slider("Top countries for shipping analysis", 5, 25, 12, key="topn_lag")
    dlag = f.dropna(subset=[lag_col]).copy()
    if dlag.empty:
        st.info("No usable shipping lag values after filters.")
    else:
        dlag["Ship Lag (days)"] = dlag[lag_col]

        colA, colB = st.columns(2)

        with colA:
            st.subheader("Avg ship lag by country (Top)")
            lag_by_country = dlag.groupby("Country")["Ship Lag (days)"].mean().sort_values(ascending=False).head(top_countries_for_lag).reset_index()
            fig1 = px.bar(lag_by_country, x="Country", y="Ship Lag (days)", title="Avg Ship Lag by Country")
            fig1.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig1, use_container_width=True)

            st.subheader("Country → City drilldown")
            pick_country = st.selectbox("Pick a country", sorted(dlag["Country"].unique().tolist()))
            lag_by_city = (dlag[dlag["Country"] == pick_country]
                           .groupby("City")["Ship Lag (days)"].mean()
                           .sort_values(ascending=False).head(15).reset_index())
            fig2 = px.bar(lag_by_city, x="City", y="Ship Lag (days)", title=f"Avg Ship Lag by City in {pick_country} (Top 15)")
            fig2.update_layout(xaxis={"categoryorder": "total descending"})
            st.plotly_chart(fig2, use_container_width=True)

        with colB:
            st.subheader("Country × City heatmap (top countries + top cities)")
            top_countries = dlag.groupby("Country")[metric].sum().sort_values(ascending=False).head(top_countries_for_lag).index
            sub = dlag[dlag["Country"].isin(top_countries)].copy()

            top_cities = (sub.groupby("City")[metric].sum().sort_values(ascending=False).head(20).index)
            sub = sub[sub["City"].isin(top_cities)]

            pv = sub.pivot_table(values="Ship Lag (days)", index="Country", columns="City", aggfunc="mean")
            fig3 = px.imshow(pv, title="Avg Ship Lag Heatmap (Country × City)", labels=dict(x="City", y="Country", color="days"))
            st.plotly_chart(fig3, use_container_width=True)

            st.subheader("Lag vs order value scatter")
            samp = sub.copy()
            if len(samp) > 2500:
                samp = samp.sample(2500, random_state=7)
            fig4 = px.scatter(samp, x="Ship Lag (days)", y=metric, color="Channel",
                              title=f"Ship Lag vs {metric} ($ CAD)", hover_data=["Country","City"])
            fig4.update_traces(hovertemplate="Lag: %{x:.0f} days<br>"+f"Value: %{y:$,.0f} CAD"+"<extra></extra>")
            st.plotly_chart(fig4, use_container_width=True)

        st.subheader("Operational recommendations (based on lag)")
        st.markdown(
            "- If certain **cities** within a country have high lag: investigate carrier/fulfillment routes for those city clusters.\n"
            "- If one **country** has consistently high lag across many cities: treat it as a separate fulfillment region (local stock, partner shipping, or SLA adjustment).\n"
            "- Track lag by **Country × City × Channel** (some channels may be slower operationally)."
        )

with tabs[4]:
    st.subheader(f"Time trends — {metric} ($ CAD)")
    ts = f.groupby("Month")[metric].sum().reset_index().rename(columns={metric: "metric"})
    fig1 = px.line(ts, x="Month", y="metric", title=f"Monthly {metric}")
    fig1.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>Value: %{y:$,.0f} CAD<extra></extra>")
    st.plotly_chart(fig1, use_container_width=True)

    top5 = country_totals.head(5).index
    ts2 = f[f["Country"].isin(top5)].groupby(["Month","Country"])[metric].sum().reset_index()
    fig2 = px.line(ts2, x="Month", y=metric, color="Country", title=f"Monthly {metric} — Top 5 Countries")
    fig2.update_traces(hovertemplate="Month: %{x|%Y-%m}<br>"+f"Value: %{y:$,.0f} CAD"+"<extra></extra>")
    st.plotly_chart(fig2, use_container_width=True)

    dlag = f.dropna(subset=[lag_col]).copy()
    if not dlag.empty:
        dlag["Ship Lag (days)"] = dlag[lag_col]
        fig3 = px.histogram(dlag, x="Ship Lag (days)", nbins=30, title="Shipping Lag Distribution (days)")
        st.plotly_chart(fig3, use_container_width=True)

with tabs[5]:
    st.subheader("Stats (simple + explainable)")
    st.caption("Only 3 checks. You can screenshot this tab and paste into your notebook.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 1) Do channels differ on order value?")
        groups = f.groupby("Channel")[metric].apply(lambda x: x.dropna().values)
        if len(groups) >= 2:
            H, p = stats.kruskal(*groups.tolist())
            st.metric("p-value", f"{p:.4f}")
            st.write("Meaning:", "Different" if p < 0.05 else "Similar")
            st.write("Recommendation:", "Scale high-performing channels; fix low performers." if p < 0.05 else "Focus on volume + country mix.")
        else:
            st.write("-")

    with col2:
        st.markdown("### 2) Is channel mix different by country?")
        top_for_test = country_totals.head(min(12, len(country_totals))).index
        tmp = f.copy()
        tmp["Country (top)"] = np.where(tmp["Country"].isin(top_for_test), tmp["Country"], "Other")
        ct = pd.crosstab(tmp["Country (top)"], tmp["Channel"])
        if ct.shape[0] >= 2 and ct.shape[1] >= 2:
            chi2, p, _, _ = stats.chi2_contingency(ct)
            st.metric("p-value", f"{p:.4f}")
            st.write("Meaning:", "Different mixes" if p < 0.05 else "No strong difference")
            st.write("Recommendation:", "Use country-specific channel strategy." if p < 0.05 else "Start from a standard mix.")
        else:
            st.write("-")

    with col3:
        st.markdown("### 3) Does consignment change value?")
        a = f.loc[f["Consignment? (Y/N)"] == "Y", metric].dropna()
        b = f.loc[f["Consignment? (Y/N)"] == "N", metric].dropna()
        if len(a) >= 15 and len(b) >= 15:
            U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
            st.metric("p-value", f"{p:.4f}")
            st.write("Median (Y):", money(a.median(), 0))
            st.write("Median (N):", money(b.median(), 0))
            st.write("Recommendation:", "Track consignment separately if different." if p < 0.05 else "Compare together if similar.")
        else:
            st.write("-")

with tabs[6]:
    st.subheader("Clean tables (rank starts at 1)")
    st.caption("Everything is formatted for screenshots and notebook copy/paste.")

    colA, colB = st.columns(2)

    with colA:
        st.markdown("### Top Countries")
        t = country_totals.reset_index().rename(columns={metric: "Total ($ CAD)"})
        t = t.head(25)
        t["Total ($ CAD)"] = t["Total ($ CAD)"].round(0)
        t = rankify(t, 1)
        st.dataframe(t.set_index("#"), use_container_width=True)

        st.markdown("### Top Cities")
        ct = f.groupby(["Country","City"])[metric].sum().sort_values(ascending=False).head(25).reset_index()
        ct = ct.rename(columns={metric: "Total ($ CAD)"})
        ct["Total ($ CAD)"] = ct["Total ($ CAD)"].round(0)
        ct = rankify(ct, 1)
        st.dataframe(ct.set_index("#"), use_container_width=True)

    with colB:
        st.markdown("### Country × Channel KPI")
        kpi = f.groupby(["Country","Channel"]).agg(
            orders=("Sale ID","count"),
            total=(metric,"sum"),
            avg=(metric,"mean"),
            median=(metric,"median"),
            avg_ship_lag=(lag_col,"mean"),
            avg_discount_rate=("Discount Rate","mean")
        ).reset_index()
        kpi["total"] = kpi["total"].round(0)
        kpi["avg"] = kpi["avg"].round(0)
        kpi["median"] = kpi["median"].round(0)
        kpi["avg_ship_lag"] = kpi["avg_ship_lag"].round(1)
        kpi["avg_discount_rate"] = (kpi["avg_discount_rate"] * 100).round(1)
        kpi = kpi.sort_values("total", ascending=False).head(40)
        kpi = rankify(kpi, 1)
        kpi = kpi.rename(columns={
            "total": "Total ($ CAD)",
            "avg": "Avg ($ CAD)",
            "median": "Median ($ CAD)",
            "avg_ship_lag": "Avg Ship Lag (days)",
            "avg_discount_rate": "Avg Discount (%)"
        })
        st.dataframe(kpi.set_index("#"), use_container_width=True)

        st.markdown("### Download")
        st.download_button("Download filtered data (CSV)", data=f.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_data.csv", mime="text/csv")

    with st.expander("Preview (first 200 rows)"):
        st.dataframe(f.head(200), use_container_width=True)
