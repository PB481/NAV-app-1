import streamlit as st
import pandas as pd
import yfinance as yf
import requests
from io import StringIO
from datetime import datetime, timedelta

# --- Configuration (can be made user-configurable in Streamlit) ---
DEFAULT_FUND_TICKER = "SPY"
# Default to yesterday's date for robustness, as today's data might not be ready
DEFAULT_TARGET_DATE = "2025-06-18"

# --- Helper for logging messages to Streamlit ---
class StreamlitLogger:
    def __init__(self, placeholder):
        self.placeholder = placeholder
        self.log_messages = []

    def log(self, message):
        self.log_messages.append(message)
        self.placeholder.text_area("Workflow Log", "\n".join(self.log_messages), height=300)
        # Scroll to bottom (Streamlit handles this automatically for text_area most times)

# --- Agent 1: HoldingsDataAgent ---
class HoldingsDataAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print
        # NOTE: The actual URL for daily holdings CSV/Excel changes frequently.
        # You would need to dynamically find this on the SSGA website or use a more robust API.
        # For this example, we'll use a hardcoded small sample to illustrate.
        # In a real app, this would involve web scraping or a direct API.

    def fetch_holdings(self, fund_ticker, target_date_str):
        self.logger(f"HoldingsDataAgent: Attempting to fetch holdings for {fund_ticker} on {target_date_str}...")

        # --- SIMULATED HOLDINGS DATA (replace with actual web scraping/parsing) ---
        # For a truly dynamic solution, you'd implement a scraper here to get the real daily holdings.
        # Given the volatility of public website structures, this is a simplification.
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

            # Use a context manager for spinner
            with st.spinner(f"Fetching market data for {len(tickers)} tickers..."):
                data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if data.empty:
                self.logger("MarketDataAgent: No data retrieved for the specified date. Check if it was a trading day.")
                return pd.DataFrame() # Return empty DataFrame

            # Extract Adj Close price for the specific date
            closing_prices = data['Adj Close'].xs(target_date_str, level=0, drop_level=True)
            if isinstance(closing_prices, pd.Series): # For a single ticker lookup
                 closing_prices = closing_prices.to_frame().T # Make it a DataFrame
            # Ensure it's a Series indexed by ticker
            closing_prices_series = closing_prices.iloc[0] if isinstance(closing_prices, pd.DataFrame) else closing_prices

            self.logger(f"MarketDataAgent: Successfully fetched prices for {len(closing_prices_series)} tickers.")
            return closing_prices_series.rename('ClosePrice') # Rename for clarity
        except Exception as e:
            self.logger(f"MarketDataAgent: Error fetching prices - {e}")
            return None

# --- Agent 3: ValuationAgent ---
class ValuationAgent:
    def __init__(self, logger=None):
        self.logger = logger if logger else print

    def value_portfolio(self, holdings_df, prices_series):
        self.logger("ValuationAgent: Valuing portfolio...")
        if holdings_df is None or prices_series is None or holdings_df.empty or prices_series.empty:
            self.logger("ValuationAgent: Missing holdings or price data. Cannot value portfolio.")
            return 0, pd.DataFrame()

        holdings_df['Ticker'] = holdings_df['Ticker'].str.strip().str.upper()
        prices_series.index = prices_series.index.str.strip().str.upper()

        merged_df = pd.merge(holdings_df, prices_series.to_frame(), left_on='Ticker', right_index=True, how='left')
        merged_df = merged_df.rename(columns={'ClosePrice': 'Price'})

        missing_prices = merged_df[merged_df['Price'].isna()]
        if not missing_prices.empty:
            self.logger(f"ValuationAgent: WARNING! Missing prices for the following tickers: {missing_prices['Ticker'].tolist()}")
            merged_df.dropna(subset=['Price'], inplace=True)

        if merged_df.empty:
            self.logger("ValuationAgent: No valid securities with prices to value.")
            return 0, pd.DataFrame()

        merged_df['SecurityValue'] = merged_df['Quantity'] * merged_df['Price']
        total_portfolio_value = merged_df['SecurityValue'].sum()

        self.logger(f"ValuationAgent: Total portfolio value calculated: ${total_portfolio_value:,.2f}")
        return total_portfolio_value, merged_df[['Ticker', 'Quantity', 'Price', 'SecurityValue']]

# --- Agent 4: NAVCalculationAgent ---
class NAVCalculationAgent:
    def __init__(self, logger=None, assumed_shares_outstanding=500_000_000): # A large, but fixed number for SPY shares
        self.logger = logger if logger else print
        self.assumed_shares_outstanding = assumed_shares_outstanding
        self.logger(f"NAVCalculationAgent: Using assumed shares outstanding: {self.assumed_shares_outstanding:,}")

    def calculate_nav_per_share(self, total_portfolio_value, cash_component=0):
        total_assets = total_portfolio_value + cash_component
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

    st.markdown("---") # Separator in UI

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
    if total_value == 0 and not detailed_valuation.empty:
         st.error("Orchestrator: Portfolio valuation resulted in zero value (unlikely if data valid). Aborting workflow.")
         return
    elif total_value == 0:
        st.error("Orchestrator: Portfolio valuation failed or resulted in zero value. Aborting workflow.")
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

# User inputs
st.sidebar.header("Configuration")
fund_ticker_input = st.sidebar.text_input("Fund Ticker (e.g., SPY)", DEFAULT_FUND_TICKER).upper()
target_date_input = st.sidebar.date_input(
    "Target Date for Calculation",
    datetime.strptime(DEFAULT_TARGET_DATE, '%Y-%m-%d'),
    max_value=datetime.now() # Cannot pick future date
)
target_date_str = target_date_input.strftime('%Y-%m-%d')

st.sidebar.markdown("---")

# Placeholder for logging messages
log_placeholder = st.empty()
streamlit_logger = StreamlitLogger(log_placeholder)

# Run button
if st.sidebar.button("Run Unit Price Calculation"):
    with st.spinner("Starting workflow..."):
        run_fund_pricing_workflow_streamlit(fund_ticker_input, target_date_str, streamlit_logger.log)

st.sidebar.markdown("---")
st.sidebar.info("The holdings data for SPY is **simulated** in this demo due to the dynamic nature of public website scraping. For a real system, you'd need a robust method to acquire daily fund holdings.")
