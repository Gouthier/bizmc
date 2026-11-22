"""Ready-made models for common small-business questions.

They are ordinary functions taking a :class:`~bizmc.simulation.Context`, so
they double as examples for writing your own.
"""

from __future__ import annotations

import numpy as np

from .simulation import Context

__all__ = ["cash_runway", "subscription"]


def _require(ctx: Context, observed: tuple[str, ...], assumed: tuple[str, ...]) -> None:
    missing_obs = [k for k in observed if k not in ctx.observed]
    missing_ass = [k for k in assumed if k not in ctx.draws]
    if missing_obs or missing_ass:
        parts = []
        if missing_obs:
            parts.append(f"observed {missing_obs}")
        if missing_ass:
            parts.append(f"assumptions {missing_ass}")
        raise KeyError("missing " + " and ".join(parts))


def _full(ctx: Context, name: str) -> np.ndarray:
    return np.broadcast_to(ctx.draws[name], (ctx.n, ctx.periods))


def cash_runway(ctx: Context) -> dict[str, np.ndarray]:
    """Cash balance from uncertain monthly revenue and costs.

    Observed: ``starting_cash``.
    Assumptions: ``revenue`` and ``costs`` per period.

    Outputs: ``net_cash_flow`` and ``cash`` (end-of-period balance).
    """
    _require(ctx, ("starting_cash",), ("revenue", "costs"))
    net = _full(ctx, "revenue") - _full(ctx, "costs")
    cash = ctx.observed["starting_cash"] + np.cumsum(net, axis=1)
    return {"net_cash_flow": net, "cash": cash}


def subscription(ctx: Context) -> dict[str, np.ndarray]:
    """A subscription business (SaaS, membership, retainer services).

    Observed:
        ``starting_cash``, ``starting_customers``, ``arpa`` (average monthly
        revenue per account today).
    Assumptions:
        ``churn`` - monthly probability that a customer leaves (0-1).
        ``new_customers`` - expected new customers per period; the actual
        number is drawn from a Poisson distribution around it.
        ``cac`` - acquisition cost per new customer.
        ``gross_margin`` - share of revenue left after delivery costs (0-1).
        ``fixed_costs`` - fixed costs per period.
    Optional assumption:
        ``arpa_growth`` - monthly relative change of ARPA (default 0).

    Customers churn individually (binomial draws), so small customer bases
    show the lumpiness they have in reality.

    Outputs: ``customers``, ``new_customers``, ``churned``, ``mrr``, ``arr``,
    ``gross_profit``, ``acquisition_spend``, ``net_cash_flow``, ``cash``.
    """
    _require(
        ctx,
        ("starting_cash", "starting_customers", "arpa"),
        ("churn", "new_customers", "cac", "gross_margin", "fixed_costs"),
    )
    n, periods = ctx.n, ctx.periods
    customers = np.empty((n, periods))
    new = np.empty((n, periods))
    churned = np.empty((n, periods))
    arpa = np.empty((n, periods))

    current = np.full(n, int(round(ctx.observed["starting_customers"])), dtype=np.int64)
    if np.any(current < 0):
        raise ValueError("starting_customers must be >= 0")
    current_arpa = np.full(n, ctx.observed["arpa"])

    for t in range(periods):
        churn = np.clip(ctx.period_values("churn", t), 0.0, 1.0)
        lost = ctx.rng.binomial(current, churn)
        rate = np.clip(ctx.period_values("new_customers", t), 0.0, None)
        gained = ctx.rng.poisson(rate)
        current = current - lost + gained
        if "arpa_growth" in ctx.draws:
            current_arpa = current_arpa * (1.0 + ctx.period_values("arpa_growth", t))
        customers[:, t] = current
        new[:, t] = gained
        churned[:, t] = lost
        arpa[:, t] = current_arpa

    mrr = customers * arpa
    margin = np.clip(_full(ctx, "gross_margin"), 0.0, 1.0)
    gross_profit = mrr * margin
    acquisition = new * _full(ctx, "cac")
    net = gross_profit - acquisition - _full(ctx, "fixed_costs")
    cash = ctx.observed["starting_cash"] + np.cumsum(net, axis=1)
    return {
        "customers": customers,
        "new_customers": new,
        "churned": churned,
        "mrr": mrr,
        "arr": mrr * 12,
        "gross_profit": gross_profit,
        "acquisition_spend": acquisition,
        "net_cash_flow": net,
        "cash": cash,
    }
