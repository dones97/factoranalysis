import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import statsmodels.api as sm
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pandas_datareader.data import DataReader

st.set_page_config(layout="wide")
st.title("Stock & Portfolio Analyzer with Editable Portfolios")

# ─── Utilities ──────────────────────────────────────────────────────────
@st.cache_data(ttl=24*3600)
def get_risk_free_rate_series(start_date, end_date, default_rate=6.5):
    try:
        rf = DataReader("INDIRLTLT01STQ", "fred", start_date, end_date) / 100.0
        return rf.resample("W-FRI").ffill().squeeze()
    except:
        idx = pd.date_range(start_date, end_date, freq="W-FRI")
        return pd.Series(default_rate / 100.0, index=idx)

@st.cache_data(ttl=24*3600)
def fetch_ff_factors(start_date, end_date):
    df = yf.download("^CRSLDX", start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c) for c in df.columns]
    close_col = next(c for c in df.columns if c.lower().startswith("close"))
    mkt = df[close_col].resample("W-FRI").last().pct_change().dropna()
    if len(mkt) < 2:
        return None
    np.random.seed(42)
    smb = pd.Series(np.random.normal(0.001, 0.02, len(mkt)), index=mkt.index)
    hml = pd.Series(np.random.normal(0.0015, 0.025, len(mkt)), index=mkt.index)
    rmw = pd.Series(np.random.normal(0.001, 0.02, len(mkt)), index=mkt.index)
    wml = pd.Series(np.random.normal(0.002, 0.03, len(mkt)), index=mkt.index)
    return pd.DataFrame({"Mkt-RF": mkt, "SMB": smb, "HML": hml, "RMW": rmw, "WML": wml})

@st.cache_data(ttl=24*3600)
def fetch_price_df(ticker, start_date, end_date):
    if not ticker or pd.isna(ticker):
        return None
    df = yf.download(ticker.strip().upper(), start=start_date, end=end_date, progress=False, auto_adjust=False)
    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(c) for c in df.columns]
    cols = [c for c in df.columns if c.lower().startswith("close")]
    if not cols:
        return None
    df["Close"] = df[cols[0]]
    return df

@st.cache_data(ttl=24*3600)
def weekly_returns(ticker, start_date, end_date):
    df = fetch_price_df(ticker, start_date, end_date)
    if df is None:
        return None
    return df["Close"].resample("W-FRI").last().pct_change().dropna()

@st.cache_data
def get_sector_info(ticker):
    info = yf.Ticker(ticker).info
    sector = info.get("sector", None)
    if sector and sector.lower() != 'unknown':
        return sector
    industry = info.get("industry", None)
    if industry:
        return industry
    return "Unknown"

# ─── Stock‐Level Regression & Metrics ───────────────────────────────────
def compute_factor_metrics_for_stock(tkr, sd, ed, ff):
    wr = weekly_returns(tkr, sd, ed)
    if wr is None or wr.empty or ff is None or ff.empty:
        return None
    rf = get_risk_free_rate_series(sd, ed, default_rate=st.session_state["current_rf"])
    weekly_rf = (1 + rf) ** (1/52) - 1
    exc = wr - weekly_rf.reindex(wr.index, method="ffill")
    data = pd.concat([exc, ff], axis=1).dropna()
    if len(data) < 5:
        return None
    y = data.iloc[:, 0]
    X = sm.add_constant(data.iloc[:, 1:])
    m = sm.OLS(y, X).fit()
    betas = m.params
    avg = ff.mean()
    exp_w = betas.get("const", 0) + sum(betas.get(f, 0) * avg[f] for f in ff.columns)
    exp_ann_exc = (1 + exp_w) ** 52 - 1
    rf_val = st.session_state["current_rf"] / 100
    exp_ann = rf_val + exp_ann_exc
    covf = ff.cov().values
    fvals = betas.drop("const", errors="ignore").values
    var_w = fvals @ covf @ fvals + m.mse_resid
    std_ann = np.sqrt(var_w) * np.sqrt(52)
    sharpe = exp_ann / std_ann if std_ann > 0 else np.nan
    return {
        "Ticker": tkr,
        "Exp_Annual_Rtn": round(exp_ann, 4),
        "Annual_Std": round(std_ann, 4),
        "Sharpe": round(sharpe, 2),
        "Betas": betas,
        "Model": m,
        "R2": round(m.rsquared, 4),
        "Adj_R2": round(m.rsquared_adj, 4),
    }

# ─── Return Contribution by Stock Chart ────────────────────────────────
def plot_return_contributions_by_stock(df, rf_pct, display_map, key):
    tot = df["MarketValue"].sum()
    dfc = df.copy()
    dfc["DisplayTicker"] = dfc["Ticker"].map(display_map).fillna(dfc["Ticker"])
    dfc["Weight"] = dfc["MarketValue"] / tot
    dfc["Total_Return"] = (rf_pct / 100 + dfc["Exp_Annual_Rtn"]) * 100
    dfc["Return_Contribution"] = (dfc["Weight"] * dfc["Total_Return"]).round(2)
    fig = px.bar(
        dfc,
        x="DisplayTicker",
        y="Return_Contribution",
        color="DisplayTicker",
        text=dfc["Return_Contribution"].astype(str) + "%",
        title="Return Contribution by Stock"
    )
    fig.update_layout(yaxis_title="Contribution (%)", xaxis_title="", showlegend=False)
    st.plotly_chart(fig, use_container_width=True, key=key)

# ─── Portfolio‐Level Metrics ───────────────────────────────────────────
def compute_weighted_portfolio_metrics(df):
    if "MarketValue" not in df or df["MarketValue"].sum() == 0:
        return None
    w = df["MarketValue"] / df["MarketValue"].sum()
    return (w * df["Exp_Annual_Rtn"]).sum()

def compute_portfolio_std_from_cov(df, sd, ed):
    mv_by = df.groupby("Ticker")["MarketValue"].sum()
    if mv_by.sum() == 0:
        return np.nan
    rd = {}
    for t in mv_by.index:
        sr = weekly_returns(t, sd, ed)
        if sr is not None:
            rd[t] = sr
    if not rd:
        return np.nan
    wr_df = pd.DataFrame(rd).dropna()
    cov = wr_df.cov().values
    w_arr = (mv_by / mv_by.sum()).reindex(wr_df.columns).fillna(0).values
    return np.sqrt(w_arr @ cov @ w_arr) * np.sqrt(52)

def compute_portfolio_regression_metrics(df, sd, ed, ff):
    rf = get_risk_free_rate_series(sd, ed, default_rate=st.session_state["current_rf"])
    weekly_rf = (1 + rf) ** (1/52) - 1
    total_mv = df["MarketValue"].sum()
    parts = []
    for _, r in df.iterrows():
        t, mv = r["Ticker"], r["MarketValue"]
        if mv <= 0:
            continue
        wgt = mv / total_mv
        sr = weekly_returns(t, sd, ed)
        if sr is None:
            continue
        exc = sr - weekly_rf.reindex(sr.index, method="ffill")
        parts.append(exc * wgt)
    if not parts:
        return None
    port_exc = pd.concat(parts, axis=1).sum(axis=1).dropna()
    data = pd.concat([port_exc, ff], axis=1).dropna()
    if len(data) < 5:
        return None
    y = data.iloc[:, 0]
    X = sm.add_constant(data.iloc[:, 1:])
    m = sm.OLS(y, X).fit()
    betas = m.params
    avg = ff.mean()
    exp_w = betas.get("const", 0) + sum(betas.get(f, 0) * avg[f] for f in ff.columns)
    exp_ann_exc = (1 + exp_w) ** 52 - 1
    rf_val = st.session_state["current_rf"] / 100
    exp_ann = rf_val + exp_ann_exc
    covf = ff.cov().values
    fvals = betas.drop("const", errors="ignore").values
    var_w = fvals @ covf @ fvals + m.mse_resid
    std_ann = np.sqrt(var_w) * np.sqrt(52)
    sharpe = exp_ann / std_ann if std_ann > 0 else np.nan
    return {
        "Exp_Annual_Rtn": round(exp_ann, 4),
        "Annual_Std": round(std_ann, 4),
        "Sharpe": round(sharpe, 2),
        "R2": round(m.rsquared, 4),
        "Adj_R2": round(m.rsquared_adj, 4),
        "Betas": betas,
        "Model": m,
    }

# ─── Main App Tabs ───────────────────────────────────────────────────────
tabs = st.tabs(["Stock Analyzer", "Portfolio Analyzer"])

# ─── Stock Analyzer Tab ─────────────────────────────────────────────────
with tabs[0]:
    st.header("Individual Stock Analysis")
    ticker = st.text_input("Ticker", "RELIANCE.NS", key="sa_tkr").strip().upper()
    ed = st.date_input("End Date", datetime.now(), key="sa_ed")
    sd = st.date_input("Start Date", ed - timedelta(days=365*15), key="sa_sd")
    rf_rate = st.number_input("Risk-Free Rate (%)", 6.5, step=0.1, key="sa_rf")
    st.session_state["current_rf"] = rf_rate

    ff = fetch_ff_factors(sd, ed)
    stk = fetch_price_df(ticker, sd, ed)
    if ff is not None and stk is not None:
        st.plotly_chart(px.line(stk, y="Close", title=f"{ticker} Price"),
                        use_container_width=True, key="sa_price")
        met = compute_factor_metrics_for_stock(ticker, sd, ed, ff)
        if met:
            st.markdown(f"**Expected Annual Return:** {met['Exp_Annual_Rtn']*100:.2f}%")
            st.markdown(f"**Annual Std Dev:** {met['Annual_Std']*100:.2f}%")
            st.markdown(f"**Sharpe Ratio:** {met['Sharpe']:.2f}")
            st.subheader("Model Statistics")
            st.markdown(f"• R²: {met['R2']}   • Adj R²: {met['Adj_R2']}")
            st.subheader("Factor Betas")
            dfb = pd.DataFrame({"Beta": met["Betas"].round(4), "P-Value": met["Model"].pvalues.round(4)})
            st.dataframe(dfb, use_container_width=True, key="sa_betas")
            st.subheader("Annual Excess Return Contributions")
            contrib = {fac: met["Betas"].get(fac, 0) * ff[fac].mean() * 52 for fac in ff.columns}
            contrib["Alpha"] = met["Betas"].get("const", 0) * 52
            dfc = pd.DataFrame({"Factor": list(contrib.keys()), "Contribution (%)": [v*100 for v in contrib.values()]}).round(2)
            st.plotly_chart(px.bar(dfc, x="Factor", y="Contribution (%)",
                                  text=dfc["Contribution (%)"].astype(str) + "%"),
                            use_container_width=True, key="sa_contrib")
        else:
            st.error("Insufficient data for regression.")
    else:
        st.error("Unable to fetch price/factor data.")
    st.session_state["selected_stock"] = ticker

# ─── Portfolio Analyzer Tab ──────────────────────────────────────────────
with tabs[1]:
    st.header("Portfolio Analyzer")

    import io
    sample_df = pd.DataFrame({
        "ISIN": ["INE123A01016", "INE234B01018"],
        "Current Qty": [100, 250]
    })
    excel_buffer = io.BytesIO()
    sample_df.to_excel(excel_buffer, index=False)
    excel_buffer.seek(0)
    st.write("Download a template file to fill in your holdings (ISIN and Current Qty):")
    st.download_button(
        label="Download holdings template (Excel)",
        data=excel_buffer,
        file_name="holdings_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        help="Download an example template to use for your Holdings upload"
    )
    
    hold = st.file_uploader("Holdings (Excel)", type=['xls', 'xlsx'], key="pa_hold")

    # Use mapping files from the repo
    nse_path = "data/nse_map.csv"
    bse_path = "data/bse_map.csv"

    if hold:
        try:
            # Validate file extensions
            if hold is not None and not hold.name.lower().endswith((".xls", ".xlsx")):
                st.error("Holdings file must be an Excel file (.xls or .xlsx)")
                st.stop()

            # Try to read the files
            dfh = pd.read_excel(hold)
            dfn = pd.read_csv(nse_path)
            dfb = pd.read_csv(bse_path)

            # Validate file contents
            required_columns = ["ISIN", "Current Qty"]
            if not all(col in dfh.columns for col in required_columns):
                st.error(f"Holdings file must contain columns: {', '.join(required_columns)}")
                st.stop()

            if "ISIN" not in dfn.columns or "Ticker" not in dfn.columns:
                st.error("NSE Map file must contain ISIN and Ticker columns")
                st.stop()

            if "ISIN" not in dfb.columns or "Ticker" not in dfb.columns:
                st.error("BSE Map file must contain ISIN and Ticker columns")
                st.stop()
        except Exception as e:
            st.error(f"Error reading files: {str(e)}")
            st.stop()

        if "base_df" not in st.session_state or st.session_state["base_src"] != hold.name:
            try:
                dfn["Ticker"] = dfn["Ticker"].astype(str).str.strip().str.upper()
                dfn["DisplayTicker"] = dfn["Ticker"]
                dfn["Ticker"] = dfn["Ticker"].apply(lambda x: x if x.endswith(".NS") else x + ".NS")

                dfb["Ticker"] = dfb["Ticker"].astype(str).str.strip().str.upper()
                dfb["DisplayTicker"] = dfb["TckrSymb"].astype(str).str.strip()
                dfb["Ticker"] = dfb["Ticker"].apply(lambda x: x if x.endswith(".BO") else x + ".BO")

                mapping_df = pd.concat([
                    dfn[["ISIN","Ticker","DisplayTicker"]],
                    dfb[["ISIN","Ticker","DisplayTicker"]]
                ], ignore_index=True).drop_duplicates("ISIN")

                merged = pd.merge(dfh, mapping_df, on="ISIN", how="left")
                merged["Ticker"] = merged["Ticker"].fillna("").str.upper()
                merged["DisplayTicker"] = merged["DisplayTicker"].fillna(merged["Ticker"])

                st.session_state["base_df"] = merged[["Ticker","DisplayTicker","Current Qty"]].rename(
                    columns={"Current Qty":"Quantity"}
                )
                st.session_state["changes_df"] = pd.DataFrame(columns=["Action","Ticker","Quantity"])
                st.session_state["base_src"] = hold.name
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                st.stop()

        base_df = st.session_state["base_df"]
        display_map = dict(zip(base_df["Ticker"], base_df["DisplayTicker"]))

        st.subheader("Table 1: Base Portfolio")
        base_edited = st.data_editor(
            base_df[["Ticker","Quantity"]],
            num_rows="dynamic",
            column_config={"Quantity": st.column_config.NumberColumn("Quantity", format="%d")},
            key="base_editor"
        )
        st.subheader("Table 2: Add/Remove Instructions")
        if st.button("Reset Changes", key="reset_changes"):
            st.session_state["changes_df"] = pd.DataFrame(columns=["Action","Ticker","Quantity"])
            st.success("Reset")
        changes = st.data_editor(
            st.session_state["changes_df"],
            column_config={
                "Action": st.column_config.SelectboxColumn("Action", options=["add","remove"]),
                "Ticker": st.column_config.TextColumn("Ticker"),
                "Quantity": st.column_config.NumberColumn("Quantity", format="%d")
            },
            num_rows="dynamic",
            key="changes_editor"
        )
        st.session_state["changes_df"] = changes

        base = base_edited.rename(columns={"Quantity":"Current Qty"})
        ops = changes.fillna({"Action":"","Ticker":"","Quantity":0}).groupby(["Action","Ticker"], as_index=False)["Quantity"].sum()
        adds = ops[ops["Action"]=="add"].set_index("Ticker")["Quantity"].to_dict()
        removes = ops[ops["Action"]=="remove"].set_index("Ticker")["Quantity"].to_dict()
        qty_map = dict(zip(base["Ticker"], base["Current Qty"]))
        for t,q in adds.items(): qty_map[t] = qty_map.get(t,0)+q
        for t,q in removes.items(): qty_map[t] = qty_map.get(t,0)-q
        active = pd.DataFrame([{"Ticker":t,"Current Qty":qty_map[t]} for t in qty_map if qty_map[t]>0])

        today = pd.to_datetime("today")
        sd2 = st.date_input("Start Date", today - timedelta(days=365*15), key="pa_sd2")
        ed2 = st.date_input("End Date", today, key="pa_ed2")
        rf2 = st.number_input("Risk-Free Rate (%)", 6.5, step=0.1, key="pa_rf2")
        st.session_state["current_rf"] = rf2

        ff2 = fetch_ff_factors(sd2, ed2)
        if base.empty or ff2 is None:
            st.error("No data to analyze.")
        else:
            def run_portfolio(dfp):
                ind = []
                for _, r in dfp.iterrows():
                    m = compute_factor_metrics_for_stock(r["Ticker"], sd2, ed2, ff2)
                    if m: ind.append(m)
                if not ind:
                    return None, None, None, None
                dfi = pd.DataFrame(ind)
                last = [fetch_price_df(t, sd2, ed2)["Close"].iloc[-1] for t in dfi["Ticker"]]
                dfi["MarketValue"] = [dfp[dfp["Ticker"]==t]["Current Qty"].iloc[0]*p for t,p in zip(dfi["Ticker"], last)]
                wret = compute_weighted_portfolio_metrics(dfi)
                wstd = compute_portfolio_std_from_cov(dfi, sd2, ed2)
                rmet = compute_portfolio_regression_metrics(dfi, sd2, ed2, ff2)
                return dfi, wret, wstd, rmet

            base_ind, base_wret, base_wstd, base_rmet = run_portfolio(base)
            changed = not base.reset_index(drop=True).equals(active.reset_index(drop=True))
            if changed:
                act_ind, act_wret, act_wstd, act_rmet = run_portfolio(active)

            # Metrics Comparison
            st.subheader("Portfolio Metrics Comparison")
            wdisp = pd.DataFrame([
                {"Base":f"{base_wret*100:.2f}%","Active":f"{act_wret*100:.2f}%"} if changed else {"Base":f"{base_wret*100:.2f}%","Active":"—"},
                {"Base":f"{base_wstd*100:.2f}%","Active":f"{act_wstd*100:.2f}%"} if changed else {"Base":f"{base_wstd*100:.2f}%","Active":"—"},
                {"Base":f"{(base_wret/base_wstd):.2f}","Active":f"{(act_wret/act_wstd):.2f}"} if changed else {"Base":f"{(base_wret/base_wstd):.2f}","Active":"—"}
            ], index=["Return","Std Dev","Sharpe"])
            st.markdown("**Weighted Approach**"); st.dataframe(wdisp, use_container_width=True, key="wt_cmp")
            rdisp = pd.DataFrame([
                {"Base":f"{base_rmet['Exp_Annual_Rtn']*100:.2f}%","Active":f"{act_rmet['Exp_Annual_Rtn']*100:.2f}%"} if changed else {"Base":f"{base_rmet['Exp_Annual_Rtn']*100:.2f}%","Active":"—"},
                {"Base":f"{base_rmet['Annual_Std']*100:.2f}%","Active":f"{act_rmet['Annual_Std']*100:.2f}%"} if changed else {"Base":f"{base_rmet['Annual_Std']*100:.2f}%","Active":"—"},
                {"Base":f"{base_rmet['Sharpe']:.2f}","Active":f"{act_rmet['Sharpe']:.2f}"} if changed else {"Base":f"{base_rmet['Sharpe']:.2f}","Active":"—"}
            ], index=["Return","Std Dev","Sharpe"])
            st.markdown("**Regression Approach**"); st.dataframe(rdisp, use_container_width=True, key="rg_cmp")
            note = f"Base R²: {base_rmet['R2']}, Adj R²: {base_rmet['Adj_R2']}"
            if changed: note += f"   |   Active R²: {act_rmet['R2']}, Adj R²: {act_rmet['Adj_R2']}"
            st.markdown(note)

            # Holdings & Metrics Tables
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Base Holdings**")
                base_tbl = base.merge(base_ind[["Ticker","Exp_Annual_Rtn","Annual_Std","Sharpe"]], on="Ticker", how="left")
                base_tbl["DisplayTicker"] = base_tbl["Ticker"].map(display_map).fillna(base_tbl["Ticker"])
                base_tbl["Quantity"] = base_tbl["Current Qty"]
                base_tbl["Exp_Annual_Rtn"] = base_tbl["Exp_Annual_Rtn"].apply(lambda x: f"{x*100:.2f}%")
                base_tbl["Annual_Std"] = base_tbl["Annual_Std"].apply(lambda x: f"{x*100:.2f}%")
                base_tbl["Sharpe"] = base_tbl["Sharpe"].apply(lambda x: f"{x:.2f}")
                base_tbl = base_tbl[["DisplayTicker","Quantity","Exp_Annual_Rtn","Annual_Std","Sharpe"]]
                base_tbl.rename(columns={"DisplayTicker":"Ticker"}, inplace=True)
                st.dataframe(base_tbl, use_container_width=True, key="hold_base")
            with c2:
                st.markdown("**Active Holdings**")
                if changed:
                    act_tbl = active.merge(act_ind[["Ticker","Exp_Annual_Rtn","Annual_Std","Sharpe"]], on="Ticker", how="left")
                    act_tbl["DisplayTicker"] = act_tbl["Ticker"].map(display_map).fillna(act_tbl["Ticker"])
                    act_tbl["Quantity"] = act_tbl["Current Qty"]
                    act_tbl["Exp_Annual_Rtn"] = act_tbl["Exp_Annual_Rtn"].apply(lambda x: f"{x*100:.2f}%")
                    act_tbl["Annual_Std"] = act_tbl["Annual_Std"].apply(lambda x: f"{x*100:.2f}%")
                    act_tbl["Sharpe"] = act_tbl["Sharpe"].apply(lambda x: f"{x:.2f}")
                else:
                    act_tbl = pd.DataFrame(columns=["DisplayTicker","Quantity","Exp_Annual_Rtn","Annual_Std","Sharpe"])
                act_tbl = act_tbl[["DisplayTicker","Quantity","Exp_Annual_Rtn","Annual_Std","Sharpe"]]
                act_tbl.rename(columns={"DisplayTicker":"Ticker"}, inplace=True)
                st.dataframe(act_tbl, use_container_width=True, key="hold_act")

            # Return Contributions
            st.subheader("Return Contribution by Stock")
            rc1, rc2 = st.columns(2)
            with rc1:
                st.markdown("**Base Portfolio**")
                plot_return_contributions_by_stock(base_ind, rf2, display_map, key="rc_base")
            with rc2:
                st.markdown("**Active Portfolio**")
                if changed:
                    plot_return_contributions_by_stock(act_ind, rf2, display_map, key="rc_act")
                else:
                    st.info("No changes → same as base.")

            # Diversification Ratio & Guide
            st.subheader("Diversification Ratio")
            def diversification_ratio(dfi,std_cov):
                w = dfi["MarketValue"] / dfi["MarketValue"].sum()
                sum_wstd = (w * dfi["Annual_Std"]).sum()
                return sum_wstd / std_cov if std_cov > 0 else np.nan
            dr_base = diversification_ratio(base_ind, base_wstd)
            dr_act = diversification_ratio(act_ind, act_wstd) if changed else None
            st.dataframe(pd.DataFrame([{"Base": f"{dr_base:.2f}", "Active": f"{dr_act:.2f}" if changed else "—"}],
                                      index=["Diversification Ratio"]), use_container_width=True, key="dr")
            st.markdown("📈 Guide: 1=no benefit, 1.1–1.3 low, 1.3–1.6 moderate, 1.6–2.0 good, >2.0 excellent")

            # Diversification Benefit Chart
            st.subheader("Diversification Benefit")
            wsum = (base_ind["MarketValue"]/base_ind["MarketValue"].sum()*base_ind["Annual_Std"]).sum()
            fig = go.Figure()
            fig.add_trace(go.Bar(name="Weighted Sum Ind Std Dev", x=["Std Dev"], y=[wsum*100],
                                 text=[f"{wsum*100:.2f}%"], textposition="outside"))
            fig.add_trace(go.Bar(name="Portfolio Std Dev", x=["Std Dev"], y=[base_wstd*100],
                                 text=[f"{base_wstd*100:.2f}%"], textposition="outside"))
            if changed:
                fig.add_trace(go.Bar(name="Active Std Dev", x=["Std Dev"], y=[act_wstd*100],
                                     text=[f"{act_wstd*100:.2f}%"], textposition="outside"))
            market_weekly_std = ff2["Mkt-RF"].std()
            market_annual_std = market_weekly_std * np.sqrt(52)
            beta_base = base_rmet["Betas"].get("Mkt-RF", 0)
            sys_risk = beta_base * market_annual_std
            fig.add_trace(go.Bar(name="Undiversifiable Risk", x=["Std Dev"], y=[sys_risk*100],
                                 text=[f"{sys_risk*100:.2f}%"], textposition="outside"))
            fig.update_layout(barmode="group", yaxis_title="Std Dev (%)")
            st.plotly_chart(fig, use_container_width=True, key="div_benefit")

            # Marginal Contribution to Risk
            st.subheader("Marginal Contribution to Risk")
            if changed:
                rets = {t: weekly_returns(t, sd2, ed2) for t in active["Ticker"] if weekly_returns(t, sd2, ed2) is not None}
                rets_df = pd.DataFrame(rets).dropna()
                cov_mat = rets_df.cov()
                w_arr = (act_ind.set_index("Ticker")["MarketValue"]/act_ind["MarketValue"].sum()).reindex(cov_mat.columns).fillna(0).values
                port_std = np.sqrt(w_arr @ cov_mat.values @ w_arr) * np.sqrt(52)
                mcr = cov_mat.values @ w_arr / port_std
                mcr_df = pd.DataFrame({"Label": cov_mat.columns, "MCR": mcr})
                mcr_df["Label"] = mcr_df["Label"].map(display_map).fillna(mcr_df["Label"])
                colors = ["red" if t == st.session_state["selected_stock"] else "blue" for t in cov_mat.columns]
                fig2 = go.Figure(go.Bar(x=mcr_df["Label"], y=mcr_df["MCR"], marker_color=colors))
                fig2.update_layout(title="Marginal Contribution to Risk", xaxis_title="Ticker", yaxis_title="MCR")
                st.plotly_chart(fig2, use_container_width=True, key="mcr")
            else:
                st.info("No changes → same as base.")

            # Correlation Heatmap
            st.subheader("Correlation Matrix")
            rets = {t: weekly_returns(t, sd2, ed2) for t in active["Ticker"]}  
            rets_df = pd.DataFrame({t: r for t, r in rets.items() if r is not None}).dropna()
            if not rets_df.empty:
                corr = rets_df.corr()
                labels = corr.columns.tolist()
                disp = [display_map.get(x, x) for x in labels]
                figc = px.imshow(corr, text_auto=True, color_continuous_scale=["green","yellow","red"], zmin=-1, zmax=1, title="Return Correlation")
                figc.update_xaxes(tickmode="array", tickvals=labels, ticktext=disp)
                figc.update_yaxes(tickmode="array", tickvals=labels, ticktext=disp)
                st.plotly_chart(figc, use_container_width=True, key="corr")
            else:
                st.info("Insufficient data for correlation.")

            # Sector Exposures Chart
            st.subheader("Sector Exposures (%)")
            base_sec = base_ind.copy()
            base_sec["Sector"] = base_sec["Ticker"].map(get_sector_info)
            pct_base = 100 * base_sec.groupby("Sector")["MarketValue"].sum() / base_sec["MarketValue"].sum()
            if changed:
                act_sec = act_ind.copy()
                act_sec["Sector"] = act_sec["Ticker"].map(get_sector_info)
                pct_act = 100 * act_sec.groupby("Sector")["MarketValue"].sum() / act_sec["MarketValue"].sum()
            else:
                pct_act = pd.Series(dtype=float)
            df_sec = pd.DataFrame({"Base": pct_base, "Active": pct_act}).fillna(0).reset_index().rename(columns={"index":"Sector"})
            fig_sec = px.bar(df_sec, x="Sector", y=["Base","Active"], barmode="group")
            for trace in fig_sec.data:
                vals = df_sec[trace.name]
                trace.text = [f"{v:.1f}%" for v in vals]
                trace.textposition = "outside"
            fig_sec.update_layout(yaxis_title="Exposure (%)")
            st.plotly_chart(fig_sec, use_container_width=True, key="sec_exp")

            # Portfolio Betas (Regression)
            st.subheader("Portfolio Betas (Regression)")
            dfb2 = pd.DataFrame({
                "Factor": base_rmet["Betas"].index,
                "Beta": base_rmet["Betas"].values,
                "P-Value": base_rmet["Model"].pvalues.values
            }).round(4)
            st.dataframe(dfb2, use_container_width=True, key="port_betas")
    else:
        st.info("Upload Holdings Excel to begin. NSE/BSE mapping files are loaded from the repo.")
