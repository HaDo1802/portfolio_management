import numpy as np
import pandas as pd
import pandas_datareader.data as web
from scipy import optimize as sc
import plotly.graph_objects as go

TRADING_DAYS = 252


def get_data(stocks, start, end):
    """Load close prices from Stooq and return daily mean returns and covariance."""
    if not stocks:
        raise ValueError("At least one ticker is required.")
    if start >= end:
        raise ValueError("Start date must be earlier than end date.")

    stk_data = web.DataReader(stocks, "stooq", start=start, end=end)
    stk_data = stk_data[[name for name in stk_data.columns if name[0] == "Close"]]
    stk_data.columns = stk_data.columns.droplevel()
    stk_data = stk_data.sort_index()

    simreturns = stk_data.pct_change().dropna()
    if simreturns.empty:
        raise ValueError("No return data available for the selected inputs.")

    mean_return = simreturns.mean()
    covar = simreturns.cov()
    return mean_return, covar


def port_performance(weights, mean_return, covar):
    """Annualized portfolio return and volatility."""
    port_return = np.sum(mean_return * weights) * TRADING_DAYS
    port_std = np.sqrt(np.dot(weights.T, np.dot(covar, weights))) * np.sqrt(TRADING_DAYS)
    return port_return, port_std


def negative_sharpe(weights, mean_return, covar, risk_free_rate=0.0):
    """Negative Sharpe ratio (used as objective for minimization)."""
    port_return, port_std = port_performance(weights, mean_return, covar)
    if np.isclose(port_std, 0):
        return np.inf
    sharpe_ratio = (port_return - risk_free_rate) / port_std
    return -sharpe_ratio


def max_sharpe_ratio(mean_return, covar, risk_free_rate=0.0, constraint_set=(0, 1)):
    num_assets = len(mean_return)
    initial_weights = np.array([1.0 / num_assets] * num_assets)
    args = (mean_return, covar, risk_free_rate)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple(constraint_set for _ in range(num_assets))
    return sc.minimize(
        negative_sharpe,
        initial_weights,
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def portfolio_var(weights, mean_return, covar):
    """Portfolio volatility as optimization objective for min-vol portfolio."""
    return port_performance(weights, mean_return, covar)[1]


def min_portfolio_var(mean_return, covar, constraint_set=(0, 1)):
    num_assets = len(mean_return)
    args = (mean_return, covar)
    constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1},)
    bounds = tuple(constraint_set for _ in range(num_assets))
    return sc.minimize(
        portfolio_var,
        num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def portfolio_return(weights, mean_return, covar):
    return port_performance(weights, mean_return, covar)[0]


def efficient_frontier(mean_return, covar, target_return, constraint_set=(0, 1)):
    """For each target return, find the lowest volatility portfolio."""
    num_assets = len(mean_return)
    args = (mean_return, covar)
    constraints = (
        {
            "type": "eq",
            "fun": lambda x: portfolio_return(x, mean_return, covar) - target_return,
        },
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
    )
    bounds = tuple(constraint_set for _ in range(num_assets))
    return sc.minimize(
        portfolio_var,
        num_assets * [1.0 / num_assets],
        args=args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )


def calculated_result(mean_return, covar, risk_free_rate=0.0, constraint_set=(0, 1)):
    max_sharpe_result = max_sharpe_ratio(mean_return, covar, risk_free_rate, constraint_set)
    min_vol_result = min_portfolio_var(mean_return, covar, constraint_set)
    if not max_sharpe_result.success:
        raise ValueError(f"Max Sharpe optimization failed: {max_sharpe_result.message}")
    if not min_vol_result.success:
        raise ValueError(f"Min volatility optimization failed: {min_vol_result.message}")

    optimal_weights = max_sharpe_result.x
    minimal_weights = min_vol_result.x

    opt_return, opt_std = port_performance(optimal_weights, mean_return, covar)
    minimal_return, minimal_std = port_performance(minimal_weights, mean_return, covar)

    optimal_allocation = pd.DataFrame(
        optimal_weights, index=mean_return.index, columns=["Allocation"]
    )
    minimal_allocation = pd.DataFrame(
        minimal_weights, index=mean_return.index, columns=["Allocation"]
    )

    ef_vols = []
    target_returns = np.linspace(minimal_return, opt_return, 20)
    for target in target_returns:
        ef_result = efficient_frontier(mean_return, covar, target, constraint_set)
        if ef_result.success:
            ef_vols.append(ef_result.fun)
        else:
            ef_vols.append(np.nan)

    minimal_return, minimal_std = round(minimal_return * 100, 2), round(minimal_std * 100, 2)
    opt_return, opt_std = round(opt_return * 100, 2), round(opt_std * 100, 2)

    return (
        opt_return,
        opt_std,
        optimal_allocation,
        minimal_return,
        minimal_std,
        minimal_allocation,
        ef_vols,
        target_returns,
    )


def ef_graph(mean_return, covar, risk_free_rate=0.0, constraint_set=(0, 1)):
    """Return figure with efficient frontier, min-vol and max-sharpe portfolios."""
    (
        opt_return,
        opt_std,
        _,
        minimal_return,
        minimal_std,
        _,
        ef_vols,
        target_returns,
    ) = calculated_result(mean_return, covar, risk_free_rate, constraint_set)

    max_sharpe_point = go.Scatter(
        name="Maximum Sharpe Ratio",
        mode="markers",
        x=[opt_std],
        y=[opt_return],
        marker=dict(color="red", size=14, line=dict(width=3, color="black")),
    )

    min_vol_point = go.Scatter(
        name="Minimum Volatility",
        mode="markers",
        x=[minimal_std],
        y=[minimal_return],
        marker=dict(color="green", size=14, line=dict(width=3, color="black")),
    )

    ef_curve = go.Scatter(
        name="Efficient Frontier",
        mode="lines+markers",
        x=[round(vol * 100, 2) if not np.isnan(vol) else None for vol in ef_vols],
        y=[round(ret * 100, 2) for ret in target_returns],
        line=dict(color="black", width=4, dash="dashdot"),
    )

    layout = go.Layout(
        title="Portfolio Optimization with the Efficient Frontier",
        yaxis=dict(title="Annualized Return (%)"),
        xaxis=dict(title="Annualized Volatility (%)"),
        showlegend=True,
        legend=dict(
            x=0.72,
            y=0.02,
            traceorder="normal",
            bgcolor="#E2E2E2",
            bordercolor="black",
            borderwidth=1,
        ),
        width=900,
        height=600,
    )

    return go.Figure(data=[max_sharpe_point, min_vol_point, ef_curve], layout=layout)
