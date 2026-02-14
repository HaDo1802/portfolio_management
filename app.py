from datetime import date, timedelta

import pandas as pd
import streamlit as st

from Source_Code.Load_Data import calculated_result, ef_graph, get_data


st.set_page_config(page_title="Portfolio Optimization", layout="wide")
st.title("Stock Portfolio Optimization")
st.caption("Mean-variance optimization with Maximum Sharpe and Minimum Volatility portfolios.")


def _parse_tickers(raw_value: str) -> list[str]:
    tickers = [t.strip().upper() for t in raw_value.split(",") if t.strip()]
    return list(dict.fromkeys(tickers))


@st.cache_data(show_spinner=False)
def _load_statistics(tickers: tuple[str, ...], start_date: date, end_date: date):
    return get_data(list(tickers), start_date, end_date)


with st.sidebar:
    st.header("Inputs")
    raw_tickers = st.text_input("Tickers (comma-separated)", value="MSFT,GOOGL,NVDA")
    tickers = _parse_tickers(raw_tickers)

    today = date.today()
    default_start = today - timedelta(days=365)
    start_date = st.date_input("Start date", value=default_start, max_value=today - timedelta(days=1))
    end_date = st.date_input("End date", value=today, min_value=start_date + timedelta(days=1), max_value=today)

    risk_free_rate = st.number_input(
        "Risk-free rate (annual, decimal)",
        min_value=0.0,
        max_value=0.2,
        value=0.02,
        step=0.005,
        format="%.3f",
    )

    allow_short = st.checkbox("Allow short selling", value=False)
    lower_bound = st.number_input("Lower bound per asset", value=-1.0 if allow_short else 0.0, step=0.05)
    upper_bound = st.number_input("Upper bound per asset", value=1.0, step=0.05)
    run_clicked = st.button("Run Optimization", type="primary")


if run_clicked:
    if len(tickers) < 2:
        st.error("Please enter at least 2 tickers.")
    elif lower_bound >= upper_bound:
        st.error("Lower bound must be smaller than upper bound.")
    else:
        constraint_set = (lower_bound, upper_bound)
        try:
            mean_return, covar = _load_statistics(tuple(tickers), start_date, end_date)
            (
                opt_return,
                opt_std,
                optimal_allocation,
                min_return,
                min_std,
                min_allocation,
                _,
                _,
            ) = calculated_result(mean_return, covar, risk_free_rate=risk_free_rate, constraint_set=constraint_set)
            fig = ef_graph(mean_return, covar, risk_free_rate=risk_free_rate, constraint_set=constraint_set)
        except Exception as exc:
            st.error(f"Optimization failed: {exc}")
        else:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Max Sharpe Return", f"{opt_return:.2f}%")
            c2.metric("Max Sharpe Volatility", f"{opt_std:.2f}%")
            c3.metric("Min Vol Return", f"{min_return:.2f}%")
            c4.metric("Min Volatility", f"{min_std:.2f}%")

            st.plotly_chart(fig, use_container_width=True)

            left, right = st.columns(2)
            with left:
                st.subheader("Maximum Sharpe Allocation")
                display_opt = optimal_allocation.copy()
                display_opt["Allocation"] = (display_opt["Allocation"] * 100).round(2)
                display_opt = display_opt.rename(columns={"Allocation": "Allocation (%)"})
                st.dataframe(display_opt, use_container_width=True)
            with right:
                st.subheader("Minimum Volatility Allocation")
                display_min = min_allocation.copy()
                display_min["Allocation"] = (display_min["Allocation"] * 100).round(2)
                display_min = display_min.rename(columns={"Allocation": "Allocation (%)"})
                st.dataframe(display_min, use_container_width=True)

            export_df = pd.concat(
                [
                    display_opt.rename(columns={"Allocation (%)": "Max Sharpe Allocation (%)"}),
                    display_min.rename(columns={"Allocation (%)": "Min Vol Allocation (%)"}),
                ],
                axis=1,
            )
            st.download_button(
                "Download allocations (CSV)",
                data=export_df.to_csv().encode("utf-8"),
                file_name="portfolio_allocations.csv",
                mime="text/csv",
            )
else:
    st.info("Set your portfolio parameters in the sidebar, then click `Run Optimization`.")
