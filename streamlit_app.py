Okay, here is the complete, full Python code for the Streamlit application, incorporating the `StreamlitLogger` and the necessary debugging suggestions for the `MarketDataAgent` that we discussed.

This code is ready to be saved as a `.py` file (e.g., `fund_pricing_app.py`) and run with `streamlit run fund_pricing_app.py`.

```python
import streamlit as st
import pandas as pd
import yfinance as yf
import requests # Though not directly used for simulated holdings, good to keep if you expand
from io import StringIO
from datetime import datetime, timedelta

# --- Configuration (can be made user-configurable in Streamlit) ---
DEFAULT_FUND_TICKER = "SPY"

# Determine a robust default date for the workflow to run on.
# Today's date in Ireland is Thursday, June 19, 2025.
# Market data for the current day is typically not complete until after market close.
# Let's default to a past trading day (e.g., Wednesday, June 18, 2025)
# or Tuesday, June 17, 2025, to ensure data is available.
# We'll calculate based on the current time in Ireland to set a sensible default.

# Get current time in Ireland (IST is typically UTC+1 in summer, UTC+0 in winter)
# However, `datetime.now()` usually gives local time, so let's stick to that for simplicity
# and adjust days relative to it.
today = datetime.now()

# Define a function to find the previous trading day
def get_previous_trading_day(current_date):
    # Go back one day at a time until a weekday (Monday-Friday) is found.
    # This doesn't account for public holidays, which would require an external holiday calendar.
    prev_day = current_date - timedelta(days=1)
    while prev_day.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        prev_day -= timedelta(days=1)
    return prev_day

# Set the default target date for the UI
# If today is a weekday after market close, maybe use today. Otherwise, use yesterday.
# For simplicity, let's just use the previous trading day as the default to increase success rate.
default_date_obj = get_previous_trading_day(today)
DEFAULT_TARGET_DATE = default_date_obj.strftime('%Y-%m-%d')


# --- Helper for logging messages to Streamlit ---
class StreamlitLogger:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.log_messages = []

    def log(self, message):
        self.log_messages.append(message)
        # Using a text_area allows the log to be scrollable and grow
        self.placeholder.text_area("Workflow Log", "\n".join(self.log_messages), height=300)
        # Ensure the UI updates immediately
        st.experimental_rerun() # Not strictly necessary for text_area, but ensures rapid updates if content changes


# --- Agent 1: HoldingsDataAgent ---
class HoldingsDataAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print
        # NOTE: The actual URL for daily holdings CSV/Excel on SSGA changes frequently.
        # It's not a stable API. For a real system, you'd need a direct data feed
        # or a sophisticated, regularly updated web scraper.
        # For this demonstration, we are simulating the holdings data.

    def fetch_holdings(self, fund_ticker, target_date_str):
        self.logger(f"HoldingsDataAgent: Attempting to fetch holdings for {fund_ticker} on {target_date_str}...")

        # --- SIMULATED HOLDINGS DATA ---
        # This is the critical simplification. In a real scenario, you'd
        # download the Excel/CSV from SSGA's website for the specific date
        # and parse it. Example of how a downloaded CSV might look (Ticker, Quantity):
        mock_holdings_csv = """Ticker,Quantity
MSFT,100000
AAPL,90000
NVDA,50000
GOOGL,70000
AMZN,80000
META,60000
TSLA,40000
BRK.B,30000
JPM,25000
XOM,20000
UNH,15000
JNJ,12000
V,10000
PG,9000
MA,8000
HD,7000
BAC,6000
ADBE,5000
CSCO,4000
PEP,3000
"""
        try:
            holdings_df = pd.read_csv(StringIO(mock_holdings_csv))
            self.logger(f"HoldingsDataAgent: Successfully fetched/simulated holdings for {len(holdings_df)} securities.")
            return holdings_df
        except Exception as e:
            self.logger(f"HoldingsDataAgent: Error fetching holdings - {e}")
            return None

# --- Agent 2: MarketDataAgent ---
class MarketDataAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print

    def fetch_prices(self, tickers, target_date_str):
        self.logger(f"MarketDataAgent: Fetching closing prices for {len(tickers)} tickers on {target_date_str}...")
        try:
            start_date = datetime.strptime(target_date_str, '%Y-%m-%d')
            end_date = start_date + timedelta(days=1) # yfinance end date is exclusive

            # Use a context manager for spinner to show activity in UI
            with st.spinner(f"Fetching market data for {len(tickers)} tickers..."):
                data = yf.download(tickers, start=start_date, end=end_date, progress=False, show_errors=False)
                # show_errors=False suppresses yfinance internal console errors,
                # we'll handle `data.empty` for our logic.

            # --- DEBUGGING LINE ---
            self.logger(f"MarketDataAgent: Raw data from yfinance.download for {target_date_str}:\n{data.to_string()}")
            # --- END DEBUGGING ---

            if data.empty:
                self.logger("MarketDataAgent: No data retrieved for the specified date.")
                self.logger("MarketDataAgent: This often means the date was a weekend or public holiday, or no data exists for these tickers/date.")
                return pd.DataFrame() # Return empty DataFrame

            # yfinance returns a MultiIndex DataFrame if multiple tickers are fetched.
            # We need the 'Adj Close' column for the specific `target_date_str`.
            # .xs() is good for selecting specific levels from a MultiIndex.
            # `drop_level=True` removes the date level after selection.
            try:
                closing_prices = data['Adj Close'].xs(target_date_str, level=0, drop_level=True)
            except KeyError:
                # This can happen if some tickers have data but not for the specific date requested,
                # or if all tickers have no data for the date but yfinance didn't return an empty DF initially.
                self.logger(f"MarketDataAgent: Could not find '{target_date_str}' in 'Adj Close' index for any ticker. Data might be incomplete or missing for the target date.")
                return pd.DataFrame() # Return empty DataFrame


            # Handle case where only one ticker was fetched (yfinance returns a Series then)
            if isinstance(closing_prices, pd.Series):
                 # Convert Series to DataFrame and then extract, ensuring consistent structure
                 closing_prices_series = closing_prices
            else: # It's already a DataFrame, likely with a single row for the date
                # We expect a single row of prices for the target date.
                # If there are multiple rows (e.g., from a range), we take the first.
                closing_prices_series = closing_prices.iloc[0] if not closing_prices.empty else pd.Series()


            self.logger(f"MarketDataAgent: Successfully fetched prices for {len(closing_prices_series)} tickers.")
            return closing_prices_series.rename('ClosePrice') # Rename for clarity
        except Exception as e:
            self.logger(f"MarketDataAgent: Error fetching prices - {e}")
            self.logger("MarketDataAgent: Common reasons for this error include network issues or yfinance API changes.")
            return None

# --- Agent 3: ValuationAgent ---
class ValuationAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print

    def value_portfolio(self, holdings_df, prices_series):
        self.logger("ValuationAgent: Valuing portfolio...")
        if holdings_df is None or holdings_df.empty:
            self.logger("ValuationAgent: Missing holdings data. Cannot value portfolio.")
            return 0, pd.DataFrame()
        if prices_series is None or prices_series.empty:
            self.logger("ValuationAgent: Missing price data. Cannot value portfolio.")
            return 0, pd.DataFrame()

        # Clean up ticker names for merging
        holdings_df['Ticker'] = holdings_df['Ticker'].str.strip().str.upper()
        prices_series.index = prices_series.index.str.strip().str.upper()

        # Merge holdings with prices. Use 'left' merge to keep all holdings.
        merged_df = pd.merge(holdings_df, prices_series.to_frame(), left_on='Ticker', right_index=True, how='left')
        merged_df = merged_df.rename(columns={'ClosePrice': 'Price'})

        # Identify missing prices and report them
        missing_prices = merged_df[merged_df['Price'].isna()]
        if not missing_prices.empty:
            missing_tickers_list = missing_prices['Ticker'].tolist()
            self.logger(f"ValuationAgent: WARNING! Missing prices for {len(missing_tickers_list)} tickers: {', '.join(missing_tickers_list)}")
            self.logger("ValuationAgent: These securities will be valued at $0 for this calculation. In a real system, this would trigger a critical alert/fallback.")
            # For this demo, we'll proceed by treating missing prices as 0 or dropping them.
            # Dropping them is safer for a calculation.
            merged_df.dropna(subset=['Price'], inplace=True)

        if merged_df.empty:
            self.logger("ValuationAgent: No valid securities with prices to value after handling missing data.")
            return 0, pd.DataFrame()

        merged_df['SecurityValue'] = merged_df['Quantity'] * merged_df['Price']
        total_portfolio_value = merged_df['SecurityValue'].sum()

        self.logger(f"ValuationAgent: Total portfolio value calculated: ${total_portfolio_value:,.2f}")
        return total_portfolio_value, merged_df[['Ticker', 'Quantity', 'Price', 'SecurityValue']]

# --- Agent 4: NAVCalculationAgent ---
class NAVCalculationAgent:
    def __init__(self, logger=None, assumed_shares_outstanding=500_000_000):
        self.logger = logger if logger else print
        # SPY shares outstanding is dynamically changing, this is a fixed estimate for demo.
        self.assumed_shares_outstanding = assumed_shares_outstanding
        self.logger(f"NAVCalculationAgent: Using assumed shares outstanding: {self.assumed_shares_outstanding:,}")

    def calculate_nav_per_share(self, total_portfolio_value, cash_component=0):
        # In a real fund, cash is a significant component, as are liabilities (fees, expenses).
        # For SPY, cash is usually a very small percentage of NAV.
        total_assets = total_portfolio_value + cash_component
        # Simplified: no liabilities considered.
        net_asset_value = total_assets

        if self.assumed_shares_outstanding <= 0:
            self.logger("NAVCalculationAgent: Error - Shares outstanding cannot be zero or negative.")
            return 0.0

        nav_per_share = net_asset_value / self.assumed_shares_outstanding
        self.logger(f"NAVCalculationAgent: Calculated NAV per Share: ${nav_per_share:,.4f}")
        return nav_per_share

# --- Agent 5: ReportingAgent ---
class ReportingAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print

    def generate_report(self, nav_per_share, detailed_valuation_df, fund_ticker, target_date_str):
        self.logger("\n--- ReportingAgent: Generating Fund Unit Price Report ---")
        st.subheader(f"Fund Unit Price Report for {fund_ticker}")
        st.write(f"**Date:** {target_date_str}")
        st.metric(label="Calculated NAV per Share", value=f"${nav_per_share:,.4f}")

        self.logger("\n--- Detailed Portfolio Valuation ---")
        st.write("### Detailed Portfolio Valuation")
        if not detailed_valuation_df.empty:
            st.dataframe(detailed_valuation_df)
        else:
            st.info("No detailed valuation data available.")
        self.logger("\n-----------------------------------------------------")


# --- Orchestrator (The main workflow logic) ---
def run_fund_pricing_workflow_streamlit(fund_ticker, target_date_str, logger):
    logger(f"Orchestrator: Starting workflow for {fund_ticker} on {target_date_str}...")

    # Initialize Agents with the Streamlit logger
    holdings_agent = HoldingsDataAgent(logger)
    market_data_agent = MarketDataAgent(logger)
    valuation_agent = ValuationAgent(logger)
    nav_calc_agent = NAVCalculationAgent(logger)
    reporting_agent = ReportingAgent(logger)

    st.markdown("---") # Separator in UI for better readability

    # Step 1: Fetch Holdings
    st.info("Step 1: Fetching Holdings Data...")
    holdings_df = holdings_agent.fetch_holdings(fund_ticker, target_date_str)
    if holdings_df is None or holdings_df.empty:
        st.error("Orchestrator: Failed to get holdings. Aborting workflow.")
        return

    tickers_to_fetch = holdings_df['Ticker'].tolist()
    if not tickers_to_fetch:
        st.error("Orchestrator: No tickers found in holdings. Aborting workflow.")
        return

    # Step 2: Fetch Market Data (Prices)
    st.info("Step 2: Fetching Market Data...")
    prices_series = market_data_agent.fetch_prices(tickers_to_fetch, target_date_str)
    if prices_series is None or prices_series.empty:
        st.error("Orchestrator: Failed to get market prices. Aborting workflow.")
        return

    # Step 3: Value Portfolio
    st.info("Step 3: Valuing Portfolio...")
    total_value, detailed_valuation = valuation_agent.value_portfolio(holdings_df, prices_series)
    if total_value <= 0: # Check for non-positive value
         st.error("Orchestrator: Portfolio valuation resulted in zero or negative value. Aborting workflow.")
         return

    # Step 4: Calculate NAV per Share
    st.info("Step 4: Calculating NAV per Share...")
    nav_per_share = nav_calc_agent.calculate_nav_per_share(total_value)
    if nav_per_share <= 0:
        st.error("Orchestrator: Calculated NAV per share is zero or negative. Aborting workflow.")
        return

    # Step 5: Generate Report
    st.info("Step 5: Generating Report...")
    reporting_agent.generate_report(nav_per_share, detailed_valuation, fund_ticker, target_date_str)

    st.success("Orchestrator: Workflow completed successfully!")

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Fund Unit Price Calculator (Agentic Demo)")

st.title("💰 Fund Unit Price Calculator (Agentic Workflow Demo)")
st.write("This application demonstrates a simplified agentic workflow for calculating a fund's unit price using publicly available data.")
st.warning("⚠️ **Disclaimer:** This is a simplified educational demo. Real fund pricing is highly complex, involves proprietary data, extensive validation, and strict regulatory compliance. **Do NOT use for actual financial decisions.** The holdings data is simulated for demonstration purposes as live scraping of daily holdings from public sites is unreliable and prone to breaking.")

# User inputs in the sidebar
st.sidebar.header("Configuration")
fund_ticker_input = st.sidebar.text_input("Fund Ticker (e.g., SPY)", DEFAULT_FUND_TICKER).upper()

# Ensure the date picker defaults to the `datetime` object, not just a string
initial_date_picker_value = datetime.strptime(DEFAULT_TARGET_DATE, '%Y-%m-%d').date()

target_date_input = st.sidebar.date_input(
    "Target Date for Calculation",
    value=initial_date_picker_value,
    max_value=today.date(), # Cannot pick future date
    help="Select a past trading day. Weekends and public holidays will result in no data."
)
target_date_str = target_date_input.strftime('%Y-%m-%d')

st.sidebar.markdown("---")

# Placeholder for logging messages (this will be filled by the StreamlitLogger)
log_placeholder = st.empty()
streamlit_logger_instance = StreamlitLogger(log_placeholder)

# Run button
if st.sidebar.button("Run Unit Price Calculation", key="run_button"):
    # Clear previous logs when a new run starts
    streamlit_logger_instance.log_messages = []
    log_placeholder.empty() # Clear the text_area visually before populating
    with st.spinner("Starting workflow..."):
        run_fund_pricing_workflow_streamlit(fund_ticker_input, target_date_str, streamlit_logger_instance.log)

st.sidebar.markdown("---")
st.sidebar.info("The holdings data for SPY is **simulated** in this demo due to the dynamic nature of public website scraping. For a real system, you'd need a robust method to acquire daily fund holdings.")
st.sidebar.caption(f"Defaulting to previous trading day: {DEFAULT_TARGET_DATE}")
```
