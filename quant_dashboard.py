import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Leon's Quant Dashboard", layout="wide")

# OMXS30 tickers (fetched from yfinance) (simplified list, expand to 30)
omxs30_tickers = [
    "ERIC-B.ST",
    "HM-B.ST",
    "VOLV-B.ST",
    "TELIA.ST",
    "SAND.ST",
    "ABB.ST",
]

# Streamlit app
st.title("Leon's Quant Dashboard")
st.markdown("Select an OMXS30 stock to view quant metrics and charts.")


# Sidebar for stock selection
st.sidebar.title("Stock Selection")
ticker = st.sidebar.multiselect(
    "Select a stock from OMXS30", omxs30_tickers, default=["ERIC-B.ST"]
)
# three buttons: one for custom weights, one for equally weighted portfolio, one for minimum variance portfolio
weights = pd.Series(
    [0] * len(ticker), index=ticker
)  # Default equally weighted portfolio
st.sidebar.title("Portfolio Weights")
if st.sidebar.button("Equally Weighted Portfolio"):
    weights = pd.Series(
        [1 / len(ticker)] * len(ticker), index=ticker
    )  # Equally weighted portfolio
    st.sidebar.write("Weights: ", weights)
if st.sidebar.button("Minimum Variance Portfolio"):
    weights = pd.Series(
        [1 / len(ticker)] * len(ticker), index=ticker
    )  # Minimum variance portfolio
    st.sidebar.write("Weights: ", weights)

st.sidebar.title("Market Simulation")
st.sidebar.write(
    "Simulate the market by changing the date range and the risk-free rate."
)


# Cache data to avoid repeated API calls
@st.cache_data
def load_data(ticker):
    """Load stock data from Yahoo Finance."""
    stock = yf.download(
        ticker, start="2020-01-01", end="2025-04-25", progress=False
    )
    stock = stock.droplevel(
        1, axis=1
    )  # Remove the ticker information from the multiindex
    return stock


# Load OMXS30 index for Beta
@st.cache_data
def load_index():
    """Load OMXS30 index data from Yahoo Finance."""
    index = yf.download(
        "^OMX", start="2020-01-01", end="2025-04-25", progress=False
    )
    return index


def load_risk_free_rate():
    """Load risk-free rate (3-month T-bill) from Yahoo Finance."""
    risk_free_rate = yf.download(
        "^IRX", start="2020-01-01", end="2025-04-25", progress=False
    )
    return risk_free_rate


# Load data
try:
    # if multiple tickers, create an equally weighted portfolio
    if len(ticker) > 1:
        stock = yf.download(
            ticker, start="2020-01-01", end="2025-04-25", progress=False
        )
        # for each column Close, High, Low, Open, Volume, calculate the mean between the tickers
        stock = stock.groupby(level=0, axis=1).mean()
        print(
            "🐍 File: stock-quantification/quant_dashboard.py | Line: 71 | undefined ~ stock",
            stock,
        )
    else:
        stock = load_data(ticker)
    index = load_index()
    risk_free_rate = load_risk_free_rate()
    mean_risk_free_rate = risk_free_rate.values.mean() / 100
    returns = stock["Close"].pct_change().dropna()
    index_returns = index["Close"].pct_change().dropna()
    stock_returns = stock["Close"].pct_change().dropna()

    start_date = st.sidebar.date_input(
        "Start Date", pd.to_datetime("2020-01-01")
    )
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2025-04-25"))
    risk_free_rate = st.sidebar.number_input(
        "Risk-Free Rate (%)",
        min_value=0.0,
        max_value=100.0,
        value=mean_risk_free_rate * 100,
        step=0.01,
        format="%.2f",
    )
    # Calculate VaR and CVaR
    var_95 = np.percentile(returns, 5) * 100  # 95% VaR
    cvar_95_vals = stock_returns[
        stock_returns.values <= (var_95 / 100)
    ].values  # CVaR
    cvar_95 = cvar_95_vals.mean() * 100
    # Calculate Beta
    merged = pd.merge(
        stock_returns, index_returns, left_index=True, right_index=True
    )
    merged.columns = ["Stock", "Index"]
    covariance = merged.cov().iloc[0, 1]
    variance = merged["Index"].var()
    beta = covariance / variance
    # Calculate Alpha
    alpha = (
        (
            (returns.mean() - mean_risk_free_rate)
            - beta * (index_returns.mean() - mean_risk_free_rate)
        )
        * 100
    ).values[0]
    # Calculate annualized volatility
    volatility = returns.std() * np.sqrt(252)
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** 252 - 1
    print(annualized_return)
    print(volatility)
    print(mean_risk_free_rate)
    # Calculate Sharpe ratio
    sharpe_ratio = (annualized_return - mean_risk_free_rate) / volatility
    # Calculate maximum drawdown
    cumulative_returns = (1 + stock_returns).cumprod()
    rolling_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min() * 100
    # Calculate Sortino ratio
    downside_returns = returns[returns < 0]
    downside_deviation = np.sqrt((downside_returns**2).mean()) * np.sqrt(252)
    sortino_ratio = (
        annualized_return - mean_risk_free_rate
    ) / downside_deviation
    # Calculate Treynor ratio
    treynor_ratio = (annualized_return - mean_risk_free_rate) / beta

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quant Metrics")
        col3, col4 = st.columns(2)
        with col3:
            st.write(f"**Ticker(s)**: {ticker}")
            st.write(f"**Beta**: {beta:.2f}")
            st.caption(
                r"$\beta = \frac{\text{Cov}(\text{Stock}, \text{Index})}{\text{Var}(\text{Index})}$"
            )
            st.caption(
                "Beta measures the stock's volatility relative to the market. "
            )
            st.write(f"**Alpha**: {alpha:.2f}")
            st.caption(
                r"$\alpha = \text{Return} - \beta \cdot \text{Market Return}$"
            )
            st.caption(
                "Alpha measures the stock's performance relative to the market. "
            )
            st.write(f"**Sharpe Ratio**: {sharpe_ratio:.2f}")
            st.caption(
                r"$\text{Sharpe Ratio} = \frac{\text{Return} - \text{Risk-Free Rate}}{\text{Volatility}}$"
            )
            st.caption(
                "The Sharpe Ratio measures the risk-adjusted return of an investment. "
            )
            st.write(f"**Sortino Ratio**: {sortino_ratio:.2f}")
            st.caption(
                r"$\text{Sortino Ratio} = \frac{\text{Return} - \text{Risk-Free Rate}}{\text{Downside Deviation}}$"
            )
            st.caption(
                "The Sortino Ratio is a variation of the Sharpe Ratio that only considers downside risk. "
            )
            st.write(f"**Treynor Ratio**: {treynor_ratio:.2f}")
            st.caption(
                r"$\text{Treynor Ratio} = \frac{\text{Return} - \text{Risk-Free Rate}}{\beta}$"
            )
            st.caption(
                "The Treynor Ratio measures the return earned in excess of that which could have been earned on a riskless investment per each unit of market risk. "
            )
        with col4:
            st.write(f"**Max Drawdown**: {max_drawdown:.2f}%")
            st.caption(
                r"$\text{Max Drawdown} = \frac{\text{Cumulative Returns} - \text{Rolling Max}}{\text{Rolling Max}}$"
            )
            st.caption(
                "Max Drawdown measures the largest peak-to-trough decline in the stock's price. "
            )
            st.write(f"**VaR (95%)**: {var_95:.2f}%")
            st.caption(
                r"$\text{VaR} = \text{Percentile}(\text{Returns}, 5\%)$"
            )
            st.caption(
                "Value at Risk (VaR) estimates the potential loss in value of an asset or portfolio over a defined period for a given confidence interval. "
            )
            st.write(f"**CVaR (95%)**: {cvar_95:.2f}%")
            st.caption(
                r"$\text{CVaR} = \text{Mean}(\text{Returns} | \text{Returns} \leq \text{VaR})$"
            )
            st.caption(
                "Conditional Value at Risk (CVaR) is the expected loss on days when there is a VaR breach. "
            )
            st.write(f"**Annualized Return**: {annualized_return * 100:.2f}%")
            st.caption(
                r"$\text{Annualized Return} = (1 + \text{Mean Daily Return})^{252} - 1$"
            )
            st.caption(
                "Annualized Return is the return on an investment over a year, assuming the investment is held for the entire year. "
            )
            st.write(f"**Annualized Volatility**: {volatility * 100:.2f}%")
            st.caption(
                r"$\text{Annualized Volatility} = \text{Standard Deviation}(\text{Daily Returns}) \cdot \sqrt{252}$"
            )
            st.caption(
                "Annualized Volatility is a measure of the amount of variation in the stock's price over time. "
            )
    with col2:
        st.subheader("Quant Charts")
        # Portfolio division
        st.write("Portfolio Division")
        fig = px.pie(
            names=["Stock", "Risk-Free Rate"],
            values=[1, 1],
            title="Portfolio Division",
            color_discrete_sequence=["#636EFA", "#EF553B"],
        )
        fig.update_traces(
            textinfo="percent+label",
            textfont_size=15,
            marker=dict(line=dict(color="#000000", width=2)),
        )
        st.plotly_chart(fig)
        # Returns distribution plot
        fig = px.histogram(
            x=returns.values * 100,  # Use raw return values multiplied by 100
            nbins=200,  # Adjust the number of bins for better granularity
            title=f"{ticker} Returns Distribution (%)",
            labels={"x": "Daily Returns (%)"},  # Label the x-axis
        )
        # add a better hover template
        fig.update_traces(
            hovertemplate="Daily Returns: %{x:.2f}%<extra></extra>"
        )
        fig.add_vline(
            x=var_95,
            line_dash="dot",
            line_color="red",
            annotation_text="VaR (95%)",
            annotation_position="top",
        )
        st.plotly_chart(fig)
    # Price trend plot
    st.subheader("Price Trend (Last 6 Months)")
    recent = stock[-126:]  # ~6 months (126 trading days)
    recent = recent["Close"].dropna()
    fig = px.line(
        recent,
        x=recent.index,
        y="Close",
        title=f"{ticker} Price",
    )
    st.plotly_chart(fig)
except Exception as e:
    st.error(
        f"Error loading data for {ticker}: {str(e)}. Try another stock. On line {e.__traceback__.tb_lineno}"
    )
