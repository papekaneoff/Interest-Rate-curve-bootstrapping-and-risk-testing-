import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA


# ==========================================================
# 1. DATA LOADING & CLEANING
# ==========================================================

def load_fred_csv(filepath, maturity):
    """
    Load a FRED CSV file and clean it.
    Handles both column naming conventions:
      - 'observation_date' / 'DGSx'  (direct FRED download)
      - 'date' / 'rate'              (renamed manually)

    Parameters
    ----------
    filepath : str
        Path to CSV file
    maturity : int
        Maturity in years

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)

    # Normalise column names regardless of FRED format
    df.columns = ["date", "rate"]

    df["rate"] = pd.to_numeric(df["rate"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    df["rate"] = df["rate"] / 100
    df["maturity"] = maturity
    return df


def load_us_treasury_from_csv(data_dir):
    """
    Load multiple Treasury maturities and concatenate them.
    """
    files = {
        1: "DGS1.csv",
        2: "DGS2.csv",
        5: "DGS5.csv",
        10: "DGS10.csv",
    }
    frames = []
    for maturity, filename in files.items():
        path = f"{data_dir}/{filename}"
        df = load_fred_csv(path, maturity)
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    return data


def extract_zero_rates_at_date(data, date):
    """
    Extract snapshot of zero rates at given date.
    """
    date = pd.to_datetime(date)
    snapshot = data[data["date"] == date]
    if snapshot.empty:
        raise ValueError("No data available for this date")
    return dict(zip(snapshot["maturity"], snapshot["rate"]))


# ==========================================================
# 2. BOOTSTRAPPING
# ==========================================================

def bootstrap_discount_factors(bonds, face_value=100):
    """
    Bootstrap discount factors from coupon bonds.
    """
    discount_factors = {}
    bonds = sorted(bonds, key=lambda x: x["maturity"])
    for bond in bonds:
        T = bond["maturity"]
        coupon_rate = bond["coupon"]
        price = bond["price"]
        coupon = coupon_rate * face_value
        pv_known_coupons = 0
        for t in range(1, T):
            if t not in discount_factors:
                raise ValueError(f"Missing discount factor for maturity {t}")
            pv_known_coupons += coupon * discount_factors[t]
        final_cash_flow = coupon + face_value
        DF_T = (price - pv_known_coupons) / final_cash_flow
        discount_factors[T] = DF_T
    return discount_factors


def discount_factor_to_zero_rates(discount_factors):
    """
    Convert discount factors to zero rates.
    """
    zero_rates = {}
    for T, DF in discount_factors.items():
        zero_rate = DF ** (-1 / T) - 1
        zero_rates[T] = zero_rate
    return zero_rates


# ==========================================================
# 3. INTERPOLATION
# ==========================================================

def interpolate_zero_curve_spline(zero_rates):
    """
    Cubic spline interpolation.
    """
    maturities = np.array(sorted(zero_rates.keys()))
    rates = np.array([zero_rates[T] for T in maturities])
    spline = CubicSpline(maturities, rates, bc_type="natural")
    return spline


def nelson_siegel(T, beta0, beta1, beta2, tau):
    """
    Nelson-Siegel parametric curve.
    """
    T = np.array(T)
    term1 = (1 - np.exp(-T / tau)) / (T / tau)
    term2 = term1 - np.exp(-T / tau)
    return beta0 + beta1 * term1 + beta2 * term2


def fit_nelson_siegel(zero_rates):
    """
    Fit Nelson-Siegel parameters.
    """
    maturities = np.array(sorted(zero_rates.keys()))
    rates = np.array([zero_rates[T] for T in maturities])
    params, _ = curve_fit(
        nelson_siegel,
        maturities,
        rates,
        p0=[0.03, -0.02, 0.02, 2.0],
        maxfev=10000
    )
    return params


def steepener_curve(curve_function, shock_short=0.0, shock_long=0.01, pivot=5):
    """
    Apply steepening effect around pivot maturity.
    """
    def new_curve(t):
        if t <= pivot:
            return curve_function(t) + shock_short
        else:
            return curve_function(t) + shock_long
    return new_curve


# ==========================================================
# 4. BOND PRICING & RISK
# ==========================================================

def bond_price_from_curve(maturity, coupon_rate, face_value, curve_function):
    """
    Price bond from zero curve.
    """
    price = 0
    coupon = coupon_rate * face_value
    for t in range(1, maturity + 1):
        r_t = curve_function(t)
        discount_factor = (1 + r_t) ** (-t)
        if t < maturity:
            cashflow = coupon
        else:
            cashflow = coupon + face_value
        price += cashflow * discount_factor
    return price


def bond_duration(maturity, coupon_rate, face_value, curve_function, bump=0.0001):
    """
    Modified duration (finite difference).
    """
    price0 = bond_price_from_curve(maturity, coupon_rate, face_value, curve_function)
    def bumped_curve(t):
        return curve_function(t) + bump
    price_up = bond_price_from_curve(maturity, coupon_rate, face_value, bumped_curve)
    duration = -(price_up - price0) / (price0 * bump)
    return duration


def bond_convexity(maturity, coupon_rate, face_value, curve_function, bump=0.0001):
    """
    Convexity (finite difference).
    """
    price0 = bond_price_from_curve(maturity, coupon_rate, face_value, curve_function)
    def curve_up(t):
        return curve_function(t) + bump
    def curve_down(t):
        return curve_function(t) - bump
    price_up = bond_price_from_curve(maturity, coupon_rate, face_value, curve_up)
    price_down = bond_price_from_curve(maturity, coupon_rate, face_value, curve_down)
    convexity = (price_up - 2 * price0 + price_down) / (price0 * bump ** 2)
    return convexity


# ==========================================================
# 5. FORWARD & SWAP RATES
# ==========================================================

def zero_to_discount_factors(zero_rates):
    """
    Convert zero rates to discount factors.
    """
    discount_factors = {}
    for T, r in zero_rates.items():
        discount_factors[T] = 1 / ((1 + r) ** T)
    return discount_factors


def forward_rate(zero_rates, T1, T2):
    """
    Compute forward rate between T1 and T2.
    """
    if T2 <= T1:
        raise ValueError("T2 must be greater than T1")
    discount_factors = zero_to_discount_factors(zero_rates)
    DF1 = discount_factors[T1]
    DF2 = discount_factors[T2]
    f = (DF1 / DF2) ** (1 / (T2 - T1)) - 1
    return f


def full_forward_curve(zero_rates):
    forwards = {}
    maturities = sorted(zero_rates.keys())
    for i in range(len(maturities) - 1):
        T1 = maturities[i]
        T2 = maturities[i + 1]
        forwards[(T1, T2)] = forward_rate(zero_rates, T1, T2)
    return forwards


def swap_rate(zero_rates, maturity, curve_function=None):
    """
    Compute par swap rate for a given maturity.
    
    If curve_function is provided, uses it to get rates at all annual tenors.
    Otherwise, assumes annual payments only at observed maturities (simpler but less accurate).
    """
    if curve_function is not None:
        # Proper calculation: get discount factors at ALL annual payment dates
        discount_factors = {}
        for t in range(1, maturity + 1):
            r_t = curve_function(t)
            discount_factors[t] = 1 / ((1 + r_t) ** t)
        
        numerator = 1 - discount_factors[maturity]
        denominator = sum(discount_factors[t] for t in range(1, maturity + 1))
        return numerator / denominator
    else:
        # Simplified: use only observed maturities (less accurate)
        discount_factors = zero_to_discount_factors(zero_rates)
        available_maturities = [t for t in sorted(discount_factors.keys()) if t <= maturity]
        
        if maturity not in discount_factors:
            raise ValueError(f"Maturity {maturity} not available. Use curve_function for interpolation.")
        
        numerator = 1 - discount_factors[maturity]
        denominator = sum(discount_factors[t] for t in available_maturities)
        return numerator / denominator


# ==========================================================
# 6. STRESS TESTING & VAR
# ==========================================================

portfolio = [
    {"maturity": 2,  "coupon": 0.02, "weight": 0.4},
    {"maturity": 5,  "coupon": 0.03, "weight": 0.3},
    {"maturity": 10, "coupon": 0.04, "weight": 0.3},
]


def portfolio_value(portfolio, face_value, curve_function):
    total_value = 0
    for bond in portfolio:
        price = bond_price_from_curve(
            bond["maturity"], bond["coupon"], face_value, curve_function
        )
        total_value += bond["weight"] * price
    return total_value


def parallel_shift_curve(curve_function, shock):
    """
    Parallel shift of yield curve.
    """
    def shifted_curve(t):
        return curve_function(t) + shock
    return shifted_curve


def key_rate_duration(maturity, coupon_rate, face_value,
                      curve_function, key_maturity, bump=0.0001, width=0.5):
    """
    Key rate duration around specific maturity.
    """
    def local_bumped_curve(t):
        if abs(t - key_maturity) <= width:
            return curve_function(t) + bump
        else:
            return curve_function(t)
    price0 = bond_price_from_curve(maturity, coupon_rate, face_value, curve_function)
    price_k = bond_price_from_curve(maturity, coupon_rate, face_value, local_bumped_curve)
    krd = -(price_k - price0) / (price0 * bump)
    return krd


def simulate_var(portfolio, face_value, curve_function, shock_std=0.01, n_sim=1000):
    np.random.seed(42)
    base_value = portfolio_value(portfolio, face_value, curve_function)
    losses = []
    for _ in range(n_sim):
        shock = np.random.normal(0, shock_std)
        shocked_curve = parallel_shift_curve(curve_function, shock)
        new_value = portfolio_value(portfolio, face_value, shocked_curve)
        losses.append(base_value - new_value)
    var_95 = np.percentile(losses, 95)
    return var_95, losses


def scenario_grid(portfolio, face_value, curve):
    scenarios = {}
    scenarios["Base"] = portfolio_value(portfolio, face_value, curve)
    scenarios["Parallel +100 bps"] = portfolio_value(
        portfolio, face_value, parallel_shift_curve(curve, 0.01)
    )
    scenarios["Parallel -100 bps"] = portfolio_value(
        portfolio, face_value, parallel_shift_curve(curve, -0.01)
    )
    scenarios["Steepener"] = portfolio_value(
        portfolio, face_value, steepener_curve(curve, 0, 0.01)
    )
    return scenarios


# ==========================================================
# 7. PCA
# ==========================================================

def perform_pca(data):
    """
    Perform PCA on daily yield curve changes.
    Returns pca object, maturities, explained variance, and returns dataframe.
    """
    pivot = data.pivot(index="date", columns="maturity", values="rate")
    pivot = pivot.sort_index()
    pivot = pivot.interpolate(method="linear")
    returns = pivot.diff()
    returns = returns.dropna(how="all")
    if returns.shape[0] < 2:
        raise ValueError(
            f"Not enough observations for PCA. returns shape = {returns.shape}"
        )
    pca = PCA()
    pca.fit(returns)
    return pca, returns.columns, returns


# ==========================================================
# 8. PLOTTING — PLOTLY INTERACTIVE
# ==========================================================

def plot_yield_curve_interactive(zero_rates, spline_curve, ns_curve):
    """
    Interactive Plotly yield curve with spline and Nelson-Siegel fits.
    """
    T = np.linspace(0.5, max(zero_rates.keys()), 200)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=list(zero_rates.keys()),
        y=[r * 100 for r in zero_rates.values()],
        mode="markers",
        name="Observed rates (FRED)",
        marker=dict(size=10, color="#E63946", symbol="circle"),
    ))

    fig.add_trace(go.Scatter(
        x=T,
        y=spline_curve(T) * 100,
        mode="lines",
        name="Cubic Spline",
        line=dict(color="#457B9D", width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=T,
        y=nelson_siegel(T, *fit_nelson_siegel(zero_rates)) * 100,
        mode="lines",
        name="Nelson–Siegel",
        line=dict(color="#2A9D8F", width=2.5, dash="dash"),
    ))

    fig.update_layout(
        title=dict(text="U.S. Treasury Zero-Coupon Yield Curve", font=dict(size=18)),
        xaxis_title="Maturity (years)",
        yaxis_title="Zero Rate (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        height=450,
    )

    fig.show()
    return fig


def plot_forward_rates_interactive(zero_rates):
    """
    Bar chart of implied forward rates.
    """
    forwards = full_forward_curve(zero_rates)
    labels = [f"{T1}Y–{T2}Y" for (T1, T2) in forwards.keys()]
    values = [v * 100 for v in forwards.values()]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=["#457B9D", "#2A9D8F", "#E9C46A"],
        text=[f"{v:.2f}%" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title=dict(text="Implied Forward Rates", font=dict(size=18)),
        xaxis_title="Period",
        yaxis_title="Forward Rate (%)",
        template="plotly_white",
        height=400,
        yaxis=dict(range=[0, max(values) * 1.3]),
    )

    fig.show()
    return fig


def plot_stress_test_interactive(maturity, coupon_rate, face_value, curve, shock=0.01):
    """
    Bar chart comparing base vs shocked bond price.
    """
    base_price = bond_price_from_curve(maturity, coupon_rate, face_value, curve)
    shocked_curve = parallel_shift_curve(curve, shock)
    shocked_price = bond_price_from_curve(maturity, coupon_rate, face_value, shocked_curve)
    change_pct = (shocked_price - base_price) / base_price * 100

    colors = ["#457B9D", "#E63946"]
    labels = ["Base Price", f"+{int(shock * 10000)} bps Shock"]
    values = [base_price, shocked_price]

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
        width=0.4,
    ))

    fig.add_annotation(
        x=1, y=shocked_price,
        text=f"Δ = {change_pct:.2f}%",
        showarrow=True,
        arrowhead=2,
        font=dict(size=13, color="#E63946"),
        ay=-40,
    )

    fig.update_layout(
        title=dict(text=f"Bond Price Stress Test — {maturity}Y Bond", font=dict(size=18)),
        yaxis_title="Price (per 100 face value)",
        template="plotly_white",
        height=420,
        yaxis=dict(range=[min(values) * 0.97, max(values) * 1.03]),
    )

    fig.show()
    return fig


def plot_scenario_grid_interactive(portfolio, face_value, curve):
    """
    Horizontal bar chart of portfolio values under different scenarios.
    """
    scenarios = scenario_grid(portfolio, face_value, curve)
    labels = list(scenarios.keys())
    values = list(scenarios.values())
    base = values[0]
    colors = ["#457B9D" if v == base else ("#E63946" if v < base else "#2A9D8F") for v in values]

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}" for v in values],
        textposition="outside",
    ))

    fig.add_vline(
        x=base, line_dash="dash", line_color="grey",
        annotation_text="Base", annotation_position="top"
    )

    fig.update_layout(
        title=dict(text="Portfolio Value — Scenario Grid", font=dict(size=18)),
        xaxis_title="Portfolio Value",
        template="plotly_white",
        height=380,
        xaxis=dict(range=[min(values) * 0.97, max(values) * 1.02]),
    )

    fig.show()
    return fig


def plot_var_distribution_interactive(losses, var_95):
    """
    Histogram of simulated P&L with VaR threshold.
    """
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=losses,
        nbinsx=60,
        name="Simulated Losses",
        marker_color="#457B9D",
        opacity=0.75,
    ))

    fig.add_vline(
        x=var_95,
        line_dash="dash",
        line_color="#E63946",
        line_width=2.5,
        annotation_text=f"VaR 95% = {var_95:.2f}",
        annotation_position="top right",
        annotation_font=dict(color="#E63946", size=13),
    )

    fig.update_layout(
        title=dict(text="Monte Carlo VaR — Loss Distribution (2 000 simulations)", font=dict(size=18)),
        xaxis_title="Portfolio Loss",
        yaxis_title="Frequency",
        template="plotly_white",
        height=420,
    )

    fig.show()
    return fig


def plot_pca_components_interactive(pca, maturities):
    """
    Line chart of the first 3 PCA components (level, slope, curvature).
    """
    labels = ["PC1 — Level", "PC2 — Slope", "PC3 — Curvature"]
    colors = ["#457B9D", "#E63946", "#2A9D8F"]

    fig = go.Figure()

    for i in range(3):
        var_exp = pca.explained_variance_ratio_[i] * 100
        fig.add_trace(go.Scatter(
            x=list(maturities),
            y=pca.components_[i],
            mode="lines+markers",
            name=f"{labels[i]} ({var_exp:.1f}%)",
            line=dict(color=colors[i], width=2.5),
            marker=dict(size=8),
        ))

    fig.update_layout(
        title=dict(
            text="PCA of Yield Curve Movements — Factor Loadings",
            font=dict(size=16),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title="Maturity (years)",
        yaxis_title="Factor Loading",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        template="plotly_white",
        height=500,
        margin=dict(t=80, b=100, l=60, r=40),
    )

    fig.show()
    return fig


def plot_pca_variance_interactive(pca):
    """
    Bar chart of explained variance by PC.
    """
    n = min(len(pca.explained_variance_ratio_), 5)
    labels = [f"PC{i+1}" for i in range(n)]
    values = pca.explained_variance_ratio_[:n] * 100
    cumulative = np.cumsum(values)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        name="Explained Variance (%)",
        marker_color="#457B9D",
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))

    fig.add_trace(go.Scatter(
        x=labels,
        y=cumulative,
        mode="lines+markers",
        name="Cumulative",
        yaxis="y2",
        line=dict(color="#E63946", width=2.5),
        marker=dict(size=8),
    ))

    fig.update_layout(
        title=dict(
            text="PCA — Explained Variance per Component",
            font=dict(size=16),
            y=0.95,
            x=0.5,
            xanchor='center',
            yanchor='top'
        ),
        xaxis_title="Principal Component",
        yaxis=dict(title="Explained Variance (%)"),
        yaxis2=dict(title="Cumulative (%)", overlaying="y", side="right", range=[0, 110]),
        template="plotly_white",
        height=500,
        margin=dict(t=80, b=80, l=60, r=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )

    fig.show()
    return fig
