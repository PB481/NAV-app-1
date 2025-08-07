import streamlit as st
import pandas as pd
import math
import json

# Conditional imports for visualization libraries
# --- Page Configuration ---
st.set_page_config(
    page_title="Periodic Table of Asset Types",
    page_icon="📊",
    layout="wide", # Use the full screen width
)

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available - some advanced visualizations will be disabled")

try:
    import altair as alt
    ALTAIR_AVAILABLE = True
except ImportError:
    ALTAIR_AVAILABLE = False

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    st.error("NumPy is required for calculations")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("SciPy not available - portfolio optimization will be disabled")

try:
    import yfinance as yf
    import requests
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    st.warning("yfinance not available - real-time market data will be disabled")

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("scikit-learn not available - AI predictions will be disabled")


# --- Data Curation ---
# This is the core data for our application.
# We've created a list of dictionaries, where each dictionary is an asset.
# 'GridRow' and 'GridCol' are used to position the asset in our CSS grid.
# The scores (Risk, Liquidity, OpCost, OpRisk) are illustrative, on a 1-10 scale.
asset_data = [
    # Group 1 & 2: Currencies & Gov Bonds (High Liquidity, Low Risk/Cost)
    {'Symbol': 'USD', 'Name': 'US Dollar', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'UST', 'Name': 'US Treasury Bill', 'Category': 'Fixed Income', 'GridRow': 2, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 'OpCost': 2, 'OpRisk': 2},
    {'Symbol': 'EUR', 'Name': 'Euro', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 2, 'Risk': 2, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'Bund', 'Name': 'German Bund', 'Category': 'Fixed Income', 'GridRow': 2, 'GridCol': 2, 'Risk': 2, 'Liquidity': 9, 'OpCost': 2, 'OpRisk': 2},

    # Transition Metals Block: Corporate Bonds, Equities, Funds
    {'Symbol': 'IGC', 'Name': 'Investment Grade Corp Bond', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 4, 'Risk': 4, 'Liquidity': 7, 'OpCost': 3, 'OpRisk': 3},
    {'Symbol': 'HYC', 'Name': 'High-Yield Corp Bond', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 5, 'Risk': 6, 'Liquidity': 6, 'OpCost': 4, 'OpRisk': 4},
    {'Symbol': 'ETF', 'Name': 'Equity ETF (e.g., SPY)', 'Category': 'Fund', 'GridRow': 2, 'GridCol': 6, 'Risk': 5, 'Liquidity': 9, 'OpCost': 1, 'OpRisk': 2},
    {'Symbol': 'MFt', 'Name': 'Active Mutual Fund', 'Category': 'Fund', 'GridRow': 2, 'GridCol': 7, 'Risk': 6, 'Liquidity': 8, 'OpCost': 3, 'OpRisk': 3},
    {'Symbol': 'EMD', 'Name': 'Emerging Market Debt', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 8, 'Risk': 7, 'Liquidity': 5, 'OpCost': 5, 'OpRisk': 6},
    {'Symbol': 'EMC', 'Name': 'Emerging Market Currency', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 9, 'Risk': 8, 'Liquidity': 6, 'OpCost': 4, 'OpRisk': 5},
    
    # Non-metals Block: Derivatives
    {'Symbol': 'Fut', 'Name': 'Futures (Listed)', 'Category': 'Derivative', 'GridRow': 2, 'GridCol': 13, 'Risk': 7, 'Liquidity': 9, 'OpCost': 3, 'OpRisk': 4},
    {'Symbol': 'Opt', 'Name': 'Options (Listed)', 'Category': 'Derivative', 'GridRow': 2, 'GridCol': 14, 'Risk': 8, 'Liquidity': 8, 'OpCost': 4, 'OpRisk': 5},
    {'Symbol': 'Sw', 'Name': 'OTC Interest Rate Swap', 'Category': 'Derivative', 'GridRow': 3, 'GridCol': 15, 'Risk': 6, 'Liquidity': 5, 'OpCost': 8, 'OpRisk': 8},
    {'Symbol': 'CDS', 'Name': 'Credit Default Swap', 'Category': 'Derivative', 'GridRow': 3, 'GridCol': 16, 'Risk': 8, 'Liquidity': 4, 'OpCost': 9, 'OpRisk': 9},
    {'Symbol': 'SP', 'Name': 'Structured Product (CLO)', 'Category': 'Structured Product', 'GridRow': 4, 'GridCol': 17, 'Risk': 9, 'Liquidity': 3, 'OpCost': 9, 'OpRisk': 9},
    
    # Lanthanides/Actinides Block: Alternatives (Illiquid, High Risk/Cost)
    {'Symbol': 'HF', 'Name': 'Hedge Fund', 'Category': 'Fund', 'GridRow': 6, 'GridCol': 4, 'Risk': 8, 'Liquidity': 4, 'OpCost': 7, 'OpRisk': 7},
    {'Symbol': 'PE', 'Name': 'Private Equity', 'Category': 'Private Equity', 'GridRow': 6, 'GridCol': 5, 'Risk': 9, 'Liquidity': 2, 'OpCost': 8, 'OpRisk': 8},
    {'Symbol': 'VC', 'Name': 'Venture Capital', 'Category': 'Private Equity', 'GridRow': 6, 'GridCol': 6, 'Risk': 10, 'Liquidity': 1, 'OpCost': 8, 'OpRisk': 8},
    {'Symbol': 'CRE', 'Name': 'Commercial Real Estate', 'Category': 'Real Estate', 'GridRow': 7, 'GridCol': 4, 'Risk': 7, 'Liquidity': 2, 'OpCost': 7, 'OpRisk': 6},
    {'Symbol': 'Inf', 'Name': 'Infrastructure', 'Category': 'Infrastructure', 'GridRow': 7, 'GridCol': 5, 'Risk': 6, 'Liquidity': 2, 'OpCost': 8, 'OpRisk': 7},
    {'Symbol': 'Au', 'Name': 'Gold (Physical)', 'Category': 'Commodity', 'GridRow': 7, 'GridCol': 7, 'Risk': 5, 'Liquidity': 7, 'OpCost': 5, 'OpRisk': 6},
    {'Symbol': 'Oil', 'Name': 'Crude Oil (Futures)', 'Category': 'Commodity', 'GridRow': 7, 'GridCol': 8, 'Risk': 8, 'Liquidity': 8, 'OpCost': 4, 'OpRisk': 5},
    {'Symbol': 'Art', 'Name': 'Fine Art', 'Category': 'Collectable', 'GridRow': 7, 'GridCol': 9, 'Risk': 9, 'Liquidity': 1, 'OpCost': 6, 'OpRisk': 7},
    {'Symbol': 'Wn', 'Name': 'Fine Wine', 'Category': 'Collectable', 'GridRow': 7, 'GridCol': 10, 'Risk': 8, 'Liquidity': 1, 'OpCost': 6, 'OpRisk': 7},
]

# Convert the list of dictionaries to a Pandas DataFrame for easier manipulation
df = pd.DataFrame(asset_data)

# --- Load Real Financial Data ---
@st.cache_data
def load_real_financial_data():
    """Load real financial data from CSV files"""
    try:
        # Load assets data
        assets_df = pd.read_csv('Asset and Fund Types/Can you create a machine readable format of the g... - Assets.csv')
        
        # Load funds data  
        funds_df = pd.read_csv('Asset and Fund Types/Can you create a machine readable format of the g... - Funds.csv')
        
        # Clean up column names
        assets_df.columns = ['GICS_Sector', 'Asset_Class', 'Asset_Type', 'Asset_SubType', 
                           'Reference_Details', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']
        
        funds_df.columns = ['Regulatory_Framework', 'Fund_Type', 'Legal_Structure', 'Key_Characteristics',
                          'Sample_Assets', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']
        
        # Remove empty rows
        assets_df = assets_df.dropna(subset=['Asset_Class'])
        funds_df = funds_df.dropna(subset=['Fund_Type'])
        
        return assets_df, funds_df
        
    except Exception as e:
        st.error(f"Error loading real financial data: {str(e)}")
        return None, None

# Load real data
real_assets_df, real_funds_df = load_real_financial_data()

# --- Load Operational Data ---
@st.cache_data
def load_operational_data():
    """Load and process operational fund data from CSV files"""
    try:
        # Load operational data files
        nav_df = pd.read_csv('datapoints samples/genie_fund_daily_nav.csv')
        characteristics_df = pd.read_csv('datapoints samples/genie_fund_characteristics.csv')
        holdings_df = pd.read_csv('datapoints samples/genie_custody_holdings.csv')
        
        # Convert date columns to datetime
        nav_df['nav_date'] = pd.to_datetime(nav_df['nav_date'])
        characteristics_df['inception_date'] = pd.to_datetime(characteristics_df['inception_date'])
        holdings_df['snapshot_date'] = pd.to_datetime(holdings_df['snapshot_date'])
        
        return nav_df, characteristics_df, holdings_df
    except FileNotFoundError:
        st.warning("Operational data files not found in 'datapoints samples' folder.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading operational data: {str(e)}")
        return None, None, None

# Load operational data
nav_data, fund_characteristics, custody_holdings = load_operational_data()

# --- Operational Workstreams Data ---
# Structure the workstreams data from the file
workstreams_data = {
    "NAV Calculation": {
        "processes": [
            "Capstock processing", "NAV Calculation and Publication", "Income Equalisation",
            "Month End Close of business Performance NAV for intraday funds", 
            "Fixed Income Fund - yield calculation and publication", "Distributions Calculations",
            "Rebates", "Swing Pricing", "NAV Tax Calculations"
        ],
        "applications": ["MultiFond", "ICON", "Global Invest One (GIO)", "INATE", "EMS", "OWM", "NAVCOM"],
        "complexity": 9, "operational_risk": 8, "automation": 6, "client_impact": 9,
        "row": 1, "col": 1
    },
    "Portfolio Valuation": {
        "processes": [
            "Exchange Listed Securities", "FX Rates", "Loans", "Over the Counter Securities", "Fund"
        ],
        "applications": ["POP", "AIP", "Vendor FVP App", "EMS"],
        "complexity": 7, "operational_risk": 7, "automation": 7, "client_impact": 8,
        "row": 1, "col": 2
    },
    "Special Portfolio Valuation": {
        "processes": ["Vendor FVP", "Instructed FVP / Client Direct"],
        "applications": ["Vendor FVP App", "EMS"],
        "complexity": 8, "operational_risk": 9, "automation": 4, "client_impact": 7,
        "row": 1, "col": 3
    },
    "Income Accounting": {
        "processes": [
            "Cash Accrual", "Fixed Income Accrual", "Dividend Income (capture and processing)",
            "OTC Income", "Sec Lending Income Accrual"
        ],
        "applications": ["GIO", "Fund Master"],
        "complexity": 6, "operational_risk": 6, "automation": 7, "client_impact": 6,
        "row": 2, "col": 1
    },
    "Trade Capture": {
        "processes": [
            "Cash", "Investor Subs and Reds", "Exchange Listed Security", "Fx Hedging",
            "FX", "Exchange Trades Derivative", "Loans", "Unlisted Listed Security",
            "Over the Counter Derivatives", "Fund"
        ],
        "applications": ["GIO", "T-Hub", "Omnium", "Murex", "AIP", "Fund Master"],
        "complexity": 8, "operational_risk": 7, "automation": 8, "client_impact": 8,
        "row": 2, "col": 2
    },
    "Reconciliation": {
        "processes": [
            "Stock", "Cash", "Investor Subs and Reds", "Exchange Listed Security",
            "Fx Hedging", "FX", "Exchange Trades Derivative", "Loans",
            "Unlisted Listed Security", "Over the Counter Derivatives", "Fund"
        ],
        "applications": ["GIO", "TMLP", "Xceptor", "Fund Master", "Omnium"],
        "complexity": 7, "operational_risk": 8, "automation": 6, "client_impact": 7,
        "row": 2, "col": 3
    },
    "Corporate Actions": {
        "processes": ["Mandatory Corp Actions", "Voluntary Corp Actions"],
        "applications": ["GIO", "E-HUB", "CARD"],
        "complexity": 6, "operational_risk": 6, "automation": 7, "client_impact": 6,
        "row": 2, "col": 4
    },
    "Expense Accounting": {
        "processes": ["Performance Fees", "Budgets", "Invoice Mgt", "Rate Cards", "Rebates"],
        "applications": ["Global Invest One (GIO)", "Broadridge – Revport", "Xceptor"],
        "complexity": 5, "operational_risk": 5, "automation": 6, "client_impact": 5,
        "row": 3, "col": 1
    },
    "Expense Reporting": {
        "processes": [
            "Other Ongoing Cost calculation", "Total Expense Ratio Reporting",
            "Fund / Client be-spoke fund fee calculations"
        ],
        "applications": ["Passport PRFA", "GIO OLE Spectre", "EUC"],
        "complexity": 6, "operational_risk": 5, "automation": 7, "client_impact": 6,
        "row": 3, "col": 2
    },
    "Tax Accounting": {
        "processes": [
            "Tax Reclaim Income Capture", "Emerging Markets CGT Accrual and Capture",
            "Withholding Tax Accrual"
        ],
        "applications": ["GIO", "Fund Master"],
        "complexity": 7, "operational_risk": 7, "automation": 5, "client_impact": 7,
        "row": 3, "col": 3
    },
    "Tax Reporting": {
        "processes": [
            "German Tax Reporting", "Austrian Tax Reporting", "Belgian Tax", "K1 / PFIC Reporting"
        ],
        "applications": ["Passport PRFA", "GIO OLE Spectre"],
        "complexity": 8, "operational_risk": 6, "automation": 6, "client_impact": 8,
        "row": 3, "col": 4
    },
    "Financial Reporting": {
        "processes": [
            "Annual and semi-annual financial statements",
            "GAAP, IFRS and IAS standards compliance",
            "Fund regulatory reporting"
        ],
        "applications": ["Passport PRFA", "EUC", "Xceptor"],
        "complexity": 7, "operational_risk": 6, "automation": 7, "client_impact": 9,
        "row": 3, "col": 5
    },
    "New Business": {
        "processes": ["Fund Setups", "Project Management", "Document review", "Data Review"],
        "applications": ["GIO", "Fund Master", "EUC"],
        "complexity": 6, "operational_risk": 5, "automation": 5, "client_impact": 8,
        "row": 4, "col": 1
    },
    "Customized Reporting": {
        "processes": [
            "Mark to Market & Liquidity Reporting", "ESMA Money Market Fund Returns",
            "AIFMD Annex IV Reporting", "MiFIR transaction reporting", "Dutch Regulatory Reporting"
        ],
        "applications": ["Passport PRFA", "GIO OLE Spectre", "EUC", "Xceptor"],
        "complexity": 8, "operational_risk": 7, "automation": 6, "client_impact": 8,
        "row": 4, "col": 2
    }
}

# Capital Portfolio Projects
capital_projects = {
    "FA - GIO Off-Mainframe Initiative": {
        "classification": "Rock", "value_stream": "Multiple", "budget": "High"
    },
    "Portfolio Analytics & Compliance (PLX)": {
        "classification": "Sand", "value_stream": "FA Workflow", "budget": "Medium"
    },
    "Entitlements (EHub) - Announcement Feed": {
        "classification": "Sand", "value_stream": "Corporate Actions", "budget": "Medium"
    },
    "Upstream Enablement - FACP": {
        "classification": "Sand", "value_stream": "Trade Capture", "budget": "Medium"
    },
    "GFS Data Mesh": {
        "classification": "Sand", "value_stream": "Customized Reporting", "budget": "Medium"
    },
    "FACT - E2E FA Recs Transformation": {
        "classification": "Sand", "value_stream": "Reconciliation", "budget": "High"
    },
    "Control Center Upgrade": {
        "classification": "Sand", "value_stream": "FA Workflow", "budget": "Medium"
    },
    "Central Bank of Ireland Strategic Reporting": {
        "classification": "Sand", "value_stream": "Financial Reporting", "budget": "Medium"
    },
    "Semi-Liquid Enhancements": {
        "classification": "Sand", "value_stream": "NAV Calculation", "budget": "Medium"
    },
    "ETF Strategic Growth Initiative": {
        "classification": "Sand", "value_stream": "New Business", "budget": "High"
    },
    "TLMP FA Strategic Data Feed Build": {
        "classification": "Sand", "value_stream": "Reconciliation", "budget": "Medium"
    }
}

# Identified Gaps mapped to workstreams
identified_gaps = {
    "NAV Calculation": [
        "Swing Pricing - Enhanced threshold and factor capabilities",
        "NDC Automation - Provide accurate swing rates with flexibility",
        "Bond maturity limitations - Ability for GIO to mature at different rates",
        "Dummy Lines - Strategic solution within GIO",
        "Accounting Interfaces to GIO - Link Yardi & Investran"
    ],
    "Special Portfolio Valuation": [
        "Fair Value Processes - Automated client directed fair value price consumption"
    ],
    "Income Accounting": [
        "REIT classification/Special Dividend/Capital Reduction - Better accounting of reclassification"
    ],
    "Trade Capture": [
        "Trades - Standardization of trade blotters",
        "Transaction Tax flags - Accurate reflection in security static data"
    ],
    "Reconciliation": [
        "Reclaims reconciliation",
        "Harmonise Custody Accounts - Single custody account solution"
    ],
    "New Business": [
        "Merger calculations - Automated fund merger capabilities"
    ],
    "Expense Accounting": [
        "Fee/Expense Calculation - Complex fee calculations not supported by GIO",
        "OCF Capping capabilities",
        "Umbrella Fees support"
    ],
    "Tax Accounting": [
        "CGT - Enhanced CGT processing and MACRO removal",
        "Taxation Linkages - Better links between FA & Custody"
    ],
    "Customized Reporting": [
        "Reporting enhancements - Improve PRFA calculation capabilities",
        "Regulatory Reporting - Enhanced regulatory reporting within FA",
        "MBOR/IBOR - Performance NAVs and XD NAVs completion",
        "Income Forecasting - Produce income projections",
        "GIO to PACE Upgrade - Remove MR data dependency",
        "FAILs enhancements - Enhanced reporting capabilities"
    ]
}

# Client Change Distribution
client_change_data = {
    "Fund Change": 37.0,
    "Reporting Change": 34.0,
    "Calculation Enhancements": 12.0,
    "Expenses": 10.0,
    "Transaction Capture": 3.54,
    "Pricing": 1.77
}

# --- Helper Functions ---

@st.cache_data
def get_color_for_value(value, metric):
    """
    Returns a background color based on the score (1-10).
    Higher scores for Risk, OpCost, OpRisk get "more red".
    Higher scores for Liquidity get "more green".
    """
    if pd.isna(value):
        return "#f0f2f6"  # Default background color for empty cells
    
    # Normalize the value from 1-10 to a 0-1 range for color mapping
    val_norm = (value - 1) / 9.0
    
    if metric == 'Liquidity':
        # Green scale for liquidity: low liquidity is reddish, high is greenish
        red = int(255 * (1 - val_norm))
        green = int(255 * val_norm)
        blue = 40
    else:
        # Red scale for risk/cost: low is greenish, high is reddish
        red = int(255 * val_norm)
        green = int(255 * (1 - val_norm))
        blue = 40
        
    return f"rgb({red}, {green}, {blue})"

def create_interactive_periodic_table(df, color_metric, selected_category="All", search_term=""):
    """
    Creates an authentic periodic table layout using CSS Grid with hover tooltips
    """
    try:
        # Filter data
        filtered_df = df.copy()
        if selected_category != 'All':
            filtered_df = filtered_df[filtered_df['Category'] == selected_category]
        if search_term:
            search_mask = (
                filtered_df['Symbol'].str.contains(search_term, case=False, na=False) |
                filtered_df['Name'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        max_row, max_col = int(df['GridRow'].max()), int(df['GridCol'].max())
        
        # Create a much simpler HTML structure that Streamlit can handle better
        html = f"""
        <div style="width: 100%; overflow-x: auto; padding: 10px;">
            <div style="display: grid; 
                        grid-template-columns: repeat({max_col}, 70px); 
                        grid-template-rows: repeat({max_row}, 70px); 
                        gap: 2px; 
                        justify-content: center;
                        background: #f0f0f0;
                        padding: 20px;
                        border-radius: 10px;">
        """
        
        # Add elements to the grid
        for _, asset in df.iterrows():
            color = get_color_for_value(asset[color_metric], color_metric)
            
            # Check if this asset should be filtered out
            is_filtered_out = (
                (selected_category != 'All' and asset['Category'] != selected_category) or
                (search_term and not (
                    search_term.lower() in asset['Symbol'].lower() or 
                    search_term.lower() in asset['Name'].lower()
                ))
            )
            
            opacity = "0.3" if is_filtered_out else "1.0"
            
            html += f'''
            <div style="grid-row: {asset['GridRow']}; 
                        grid-column: {asset['GridCol']}; 
                        background-color: {color}; 
                        border: 2px solid #333; 
                        border-radius: 5px; 
                        display: flex; 
                        flex-direction: column; 
                        justify-content: center; 
                        align-items: center; 
                        text-align: center; 
                        opacity: {opacity};
                        cursor: pointer;
                        font-family: Arial, sans-serif;"
                 title="{asset['Name']} ({asset['Category']}) | Risk: {asset['Risk']}/10 | Liquidity: {asset['Liquidity']}/10 | Op Cost: {asset['OpCost']}/10 | Op Risk: {asset['OpRisk']}/10">
                <div style="font-size: 14px; font-weight: bold; color: white; text-shadow: 1px 1px 2px black;">
                    {asset['Symbol']}
                </div>
                <div style="font-size: 10px; color: white; text-shadow: 1px 1px 2px black;">
                    {asset[color_metric]}/10
                </div>
            </div>
            '''
        
        html += """
            </div>
        </div>
        """
        
        return html
        
    except Exception as e:
        # Return a simple error message if grid fails
        return f'<div style="color: red;">Error creating periodic table: {str(e)}</div>'

@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_market_data():
    """
    Simulated market data loading - replace with real API calls
    """
    try:
        # This would be replaced with actual market data API
        # import yfinance as yf
        # tickers = ['SPY', 'GLD', 'TLT', 'DX-Y.NYB', 'CL=F']  # ETF, Gold, Treasury, Dollar Index, Oil
        # data = yf.download(tickers, period='1d', interval='1m')
        # return process_yfinance_data(data)
        
        # Simulated data for demo with more realistic variation
        import numpy as np
        np.random.seed(42)  # For consistent demo data
        
        market_data = {
            'USD': {'price': 1.0, 'change': np.random.normal(0, 0.1)},
            'EUR': {'price': 1.08, 'change': np.random.normal(0.2, 0.3)},
            'ETF': {'price': 445.32, 'change': np.random.normal(1.2, 1.5)},
            'Au': {'price': 2034.50, 'change': np.random.normal(-0.5, 0.8)},
            'Oil': {'price': 78.45, 'change': np.random.normal(2.1, 2.0)},
            'UST': {'price': 100.12, 'change': np.random.normal(0.05, 0.2)},
            'Bund': {'price': 98.45, 'change': np.random.normal(0.08, 0.25)},
        }
        return market_data
    except Exception:
        return {}

@st.cache_data
def calculate_portfolio_optimization(portfolio_data, optimization_method="max_sharpe"):
    """
    Portfolio optimization using modern portfolio theory
    """
    try:
        if not portfolio_data or len(portfolio_data) < 2 or not SCIPY_AVAILABLE:
            return None
            
        import numpy as np
        from scipy.optimize import minimize
        
        # Create synthetic return data based on asset characteristics
        assets = []
        returns = []
        volatilities = []
        
        for asset in portfolio_data:
            assets.append(asset['Symbol'])
            # Synthetic expected return based on risk (higher risk = higher expected return)
            expected_return = 0.02 + (asset['Risk'] / 10) * 0.12  # 2% to 14% range
            # Volatility based on risk and liquidity
            volatility = (asset['Risk'] / 10) * 0.3 * (1 - asset['Liquidity'] / 20)  # Up to 30%
            
            returns.append(expected_return)
            volatilities.append(volatility)
        
        returns = np.array(returns)
        volatilities = np.array(volatilities)
        
        # Simple correlation matrix (more sophisticated would use historical data)
        n_assets = len(assets)
        correlation_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                # Assets in same category have higher correlation
                if portfolio_data[i]['Category'] == portfolio_data[j]['Category']:
                    correlation_matrix[i, j] = correlation_matrix[j, i] = 0.7
                else:
                    correlation_matrix[i, j] = correlation_matrix[j, i] = 0.3
        
        # Covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        def portfolio_stats(weights):
            portfolio_return = np.sum(returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            sharpe_ratio = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            return portfolio_return, portfolio_vol, sharpe_ratio
        
        def objective(weights):
            _, vol, sharpe = portfolio_stats(weights)
            if optimization_method == "max_sharpe":
                return -sharpe  # Negative for maximization
            elif optimization_method == "min_vol":
                return vol
            else:
                return -sharpe
        
        # Constraints and bounds
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        bounds = tuple((0.05, 0.5) for _ in range(n_assets))  # 5% to 50% per asset
        
        # Initial guess
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Optimize
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if result.success:
            optimal_weights = result.x
            opt_return, opt_vol, opt_sharpe = portfolio_stats(optimal_weights)
            
            return {
                'assets': assets,
                'optimal_weights': optimal_weights,
                'expected_return': opt_return,
                'volatility': opt_vol,
                'sharpe_ratio': opt_sharpe,
                'optimization_method': optimization_method
            }
        else:
            return None
            
    except Exception as e:
        st.error(f"Portfolio optimization error: {str(e)}")
        return None

@st.cache_data
def calculate_efficient_frontier(portfolio_data, n_portfolios=50):
    """
    Calculate efficient frontier for portfolio visualization
    """
    try:
        if not portfolio_data or len(portfolio_data) < 2:
            return None
            
        import numpy as np
        
        # Use same return/risk logic as optimization function
        returns = []
        volatilities = []
        
        for asset in portfolio_data:
            expected_return = 0.02 + (asset['Risk'] / 10) * 0.12
            volatility = (asset['Risk'] / 10) * 0.3 * (1 - asset['Liquidity'] / 20)
            returns.append(expected_return)
            volatilities.append(volatility)
        
        returns = np.array(returns)
        volatilities = np.array(volatilities)
        
        # Simple correlation
        n_assets = len(portfolio_data)
        correlation_matrix = np.eye(n_assets)
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                if portfolio_data[i]['Category'] == portfolio_data[j]['Category']:
                    correlation_matrix[i, j] = correlation_matrix[j, i] = 0.7
                else:
                    correlation_matrix[i, j] = correlation_matrix[j, i] = 0.3
        
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Generate random portfolios for frontier
        frontier_data = []
        for _ in range(n_portfolios):
            weights = np.random.random(n_assets)
            weights /= weights.sum()
            
            portfolio_return = np.sum(returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            frontier_data.append({
                'return': portfolio_return,
                'volatility': portfolio_vol,
                'sharpe': portfolio_return / portfolio_vol if portfolio_vol > 0 else 0
            })
        
        return frontier_data
        
    except Exception:
        return None


# --- App UI ---

st.title("🚀 Iluvalcar 2.0")
st.markdown("""
Advanced fund accounting and asset management platform with comprehensive analytics, visualization tools, and operational insights.
""")

# Create main tabs
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🏦 Fund Accounting Product Overview", 
    "📊 Asset Profiling and Fund Insights", 
    "🤖 MLOps for Fund Accounting"
])

# Tab 1: Fund Accounting Product Overview
with main_tab1:
    st.header("🏗️ Operational Workstreams - Fund Administration Periodic Table")
    st.markdown("""
    ### Comprehensive Fund Administration Operations
    
    Overview of key processing value streams, underlying processes, applications, and capital portfolio positioning within our fund administration framework.
    """)
    
    # Create workstream tabs
    workstream_tab1, workstream_tab2, workstream_tab3, workstream_tab4 = st.tabs([
        "📊 Processing Value Streams", 
        "💼 Portfolio Analysis", 
        "🔍 Identified Gaps", 
        "📈 Client Change Analytics"
    ])
    
    with workstream_tab1:
        st.subheader("🔄 Core Processing Value Streams")
        
        # Create columns for different workstreams
        ws_col1, ws_col2 = st.columns(2)
        
        with ws_col1:
            st.markdown("""
            #### 🧮 Net Asset Value (NAV) Calculation
            1. Capstock processing
            2. NAV Calculation and Publication
            3. Income Equalisation
            4. Month End Close Performance NAV
            5. Fixed Income Fund yield calculation
            6. Distributions Calculations
            7. Rebates & Swing Pricing
            8. NAV Tax Calculations
            
            #### 💰 Portfolio Valuation
            - Exchange Listed Securities
            - FX Rates & Loans
            - Over the Counter Securities
            - Fund Valuations
            
            #### 📋 Special Portfolio Valuation
            - Vendor FVP
            - Instructed FVP / Client Direct
            
            #### 💵 Income Accounting
            - Cash & Fixed Income Accrual
            - Dividend Income Processing
            - OTC Income & Sec Lending
            """)
        
        with ws_col2:
            st.markdown("""
            #### 🔄 Trade Capture
            - Cash & Investor Subs/Reds
            - Exchange Listed Securities
            - FX Hedging & FX Trades
            - Exchange Traded Derivatives
            - Loans & Unlisted Securities
            - OTC Derivatives & Funds
            
            #### 🔗 Reconciliation
            - Stock & Cash Reconciliation
            - Investor Subscriptions/Redemptions
            - Exchange Listed & FX Reconciliation
            - Derivative & Fund Reconciliation
            
            #### 🏢 Corporate Actions
            - Mandatory Corp Actions
            - Voluntary Corp Actions
            
            #### 💳 Expense Management
            - Performance Fees & Budgets
            - Invoice Management
            - Rate Cards & Rebates
            """)
        
        st.markdown("---")
        
        # Application Mapping
        st.subheader("💻 Application Portfolio Mapping")
        
        app_col1, app_col2, app_col3 = st.columns(3)
        
        with app_col1:
            st.markdown("""
            **Transaction Processing (6 Apps)**
            - GIO
            - T-Hub
            - Omnium
            - Murex
            - AIP
            - Fund Master
            """)
        
        with app_col2:
            st.markdown("""
            **Fund Valuation (4 Apps)**
            - INATE
            - EMS
            - OWM
            - NAVCOM
            """)
        
        with app_col3:
            st.markdown("""
            **Data & Reporting (5 Apps)**
            - Passport PRFA
            - GIO OLE Spectre
            - EUC
            - Xceptor
            """)
    
    with workstream_tab2:
        st.subheader("📈 Portfolio Analysis: Projects by Value Stream")
        
        # Capital Portfolio Overview
        st.info("💰 **Capital Portfolio**: USD 26m allocated for 2025 initiatives")
        
        # Project classifications
        project_col1, project_col2 = st.columns(2)
        
        with project_col1:
            st.markdown("""
            #### 🗿 Rock Projects (Strategic)
            - **FA - GIO Off-Mainframe Initiative**
              - Value Stream: Multiple
              - Strategic modernization initiative
            
            #### 🏗️ Sand Projects (Tactical)
            - **Portfolio Analytics & Compliance (PLX)**
            - **Entitlements (EHub) - Announcement Feed**
            - **Upstream Enablement - FACP**
            - **GFS Data Mesh**
            - **FACT - E2E FA Recs Transformation**
            """)
        
        with project_col2:
            st.markdown("""
            #### 📊 Additional Sand Projects
            - **Control Center Upgrade**
            - **Central Bank of Ireland Strategic Reporting**
            - **Semi-Liquid Enhancements**
            - **ETF Strategic Growth Initiative**
            - **TLMP FA Strategic Data Feed Build**
            
            #### 🎯 Value Stream Distribution
            - FA Workflow
            - Events Corp Actions
            - Trade Capture
            - Reporting
            - Reconciliation
            """)
        
        st.markdown("---")
        
        # 3D Positioning Analysis
        st.subheader("🎯 3D Capital Portfolio Positioning Analysis")
        
        if PLOTLY_AVAILABLE:
            # Create 3D positioning data for projects
            projects_3d = [
                # Rock Projects
                {"name": "FA - GIO Off-Mainframe", "complexity": 9, "strategic_value": 10, "completion": 25, "category": "Rock", "investment": 12.0},
                
                # Sand Projects
                {"name": "PLX Migration", "complexity": 7, "strategic_value": 8, "completion": 60, "category": "Sand", "investment": 3.2},
                {"name": "EHub Enhancement", "complexity": 5, "strategic_value": 6, "completion": 40, "category": "Sand", "investment": 2.1},
                {"name": "FACP Enablement", "complexity": 6, "strategic_value": 7, "completion": 30, "category": "Sand", "investment": 2.8},
                {"name": "GFS Data Mesh", "complexity": 8, "strategic_value": 9, "completion": 15, "category": "Sand", "investment": 3.5},
                {"name": "FACT E2E Transformation", "complexity": 7, "strategic_value": 8, "completion": 50, "category": "Sand", "investment": 2.4},
                {"name": "Control Center Upgrade", "complexity": 4, "strategic_value": 5, "completion": 80, "category": "Sand", "investment": 1.0},
                {"name": "CBI Strategic Reporting", "complexity": 5, "strategic_value": 6, "completion": 70, "category": "Sand", "investment": 1.5},
                {"name": "Semi-Liquid Enhancements", "complexity": 6, "strategic_value": 7, "completion": 35, "category": "Sand", "investment": 2.2},
                {"name": "ETF Growth Initiative", "complexity": 8, "strategic_value": 9, "completion": 20, "category": "Sand", "investment": 4.0},
            ]
            
            # Create current state vs future state 3D plot
            col_3d1, col_3d2 = st.columns(2)
            
            with col_3d1:
                st.markdown("**Current State (2024)**")
                
                fig_current = go.Figure(data=go.Scatter3d(
                    x=[p["complexity"] for p in projects_3d],
                    y=[p["strategic_value"] for p in projects_3d],
                    z=[p["completion"] for p in projects_3d],
                    mode='markers+text',
                    marker=dict(
                        size=[p["investment"] * 2 for p in projects_3d],
                        color=['red' if p["category"] == "Rock" else 'blue' for p in projects_3d],
                        opacity=0.6,
                        colorscale='RdYlBu'
                    ),
                    text=[p["name"] for p in projects_3d],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Complexity: %{x}<br>' +
                                  'Strategic Value: %{y}<br>' +
                                  'Completion: %{z}%<br>' +
                                  '<extra></extra>'
                ))
                
                fig_current.update_layout(
                    scene=dict(
                        xaxis_title="Complexity (1-10)",
                        yaxis_title="Strategic Value (1-10)",
                        zaxis_title="Completion %"
                    ),
                    height=500,
                    title="Current Portfolio Position"
                )
                
                st.plotly_chart(fig_current, use_container_width=True)
            
            with col_3d2:
                st.markdown("**Future State (2025 Target)**")
                
                # Adjust future completion rates
                future_projects = []
                for p in projects_3d:
                    future_p = p.copy()
                    if p["category"] == "Rock":
                        future_p["completion"] = min(100, p["completion"] + 60)
                    else:
                        future_p["completion"] = min(100, p["completion"] + 40)
                    future_projects.append(future_p)
                
                fig_future = go.Figure(data=go.Scatter3d(
                    x=[p["complexity"] for p in future_projects],
                    y=[p["strategic_value"] for p in future_projects],
                    z=[p["completion"] for p in future_projects],
                    mode='markers+text',
                    marker=dict(
                        size=[p["investment"] * 2 for p in future_projects],
                        color=['green' if p["category"] == "Rock" else 'lightgreen' for p in future_projects],
                        opacity=0.8,
                        colorscale='RdYlGn'
                    ),
                    text=[p["name"] for p in future_projects],
                    textposition="top center",
                    hovertemplate='<b>%{text}</b><br>' +
                                  'Complexity: %{x}<br>' +
                                  'Strategic Value: %{y}<br>' +
                                  'Completion: %{z}%<br>' +
                                  '<extra></extra>'
                ))
                
                fig_future.update_layout(
                    scene=dict(
                        xaxis_title="Complexity (1-10)",
                        yaxis_title="Strategic Value (1-10)",
                        zaxis_title="Completion %"
                    ),
                    height=500,
                    title="2025 Target Position"
                )
                
                st.plotly_chart(fig_future, use_container_width=True)
            
            # Timeline visualization
            st.markdown("#### 📅 Project Completion Timeline")
            
            # Create timeline data
            timeline_data = []
            for p in projects_3d:
                months_remaining = int((100 - p["completion"]) / 10)  # Rough estimate
                target_date = pd.Timestamp.now() + pd.DateOffset(months=months_remaining)
                timeline_data.append({
                    'Project': p["name"],
                    'Current %': p["completion"],
                    'Target Date': target_date,
                    'Investment ($M)': p["investment"],
                    'Category': p["category"]
                })
            
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('Target Date')
            
            fig_timeline = go.Figure()
            
            for _, row in timeline_df.iterrows():
                color = 'red' if row['Category'] == 'Rock' else 'blue'
                fig_timeline.add_trace(go.Scatter(
                    x=[row['Target Date']],
                    y=[row['Investment ($M)']],
                    mode='markers+text',
                    marker=dict(size=row['Current %']/2, color=color, opacity=0.7),
                    text=row['Project'],
                    textposition="top center",
                    name=row['Project'],
                    showlegend=False
                ))
            
            fig_timeline.update_layout(
                title="Project Timeline by Investment Size and Completion Target",
                xaxis_title="Target Completion Date",
                yaxis_title="Investment ($M)",
                height=400
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        else:
            st.warning("3D positioning analysis requires Plotly for interactive visualizations.")
        
        st.markdown("---")
        st.subheader("🏆 Competitive Impact Analysis")
        st.markdown("""
        **Major Fund Administration Players & Their Advantages:**
        - **State Street**: Advanced automation in NAV calculations and reporting
        - **JPMorgan**: Integrated custody and fund accounting platform  
        - **BNY Mellon**: Cloud-native fund operations suite
        - **HSBC**: RegTech integration for automated compliance
        - **Citi**: Real-time reconciliation and risk monitoring
        
        **Investment Reallocation Recommendations:**
        1. **Increase Rock Project Investment**: GIO Off-Mainframe needs 40% more funding
        2. **Accelerate ETF Growth**: Market opportunity closing rapidly
        3. **Prioritize Data Mesh**: Foundation for future automation
        
        **Risk of Non-Investment:**
        - 15-20% market share loss in ETF administration
        - Regulatory compliance gaps in emerging markets
        - 30% higher operational costs vs competitors by 2026
        """)
    
    with workstream_tab3:
        st.subheader("🔍 Identified Operational Gaps")
        
        gap_col1, gap_col2 = st.columns(2)
        
        with gap_col1:
            st.markdown("""
            #### 🧮 NAV Calculation Gaps
            - **Swing Pricing**: Threshold-based swing factors
            - **NDC Automation**: End-to-end process automation
            - **Bond Maturity**: Flexible maturity rate handling
            - **Dummy Lines**: Strategic GIO solution needed
            
            #### 💰 Valuation & Income Gaps
            - **Fair Value Processes**: Automated client-directed pricing
            - **REIT Classification**: Better income/capital event accounting
            - **Trade Standardization**: Unified trade blotter approach
            
            #### 🔗 Reconciliation Gaps
            - **Reclaims Reconciliation**: Automated processing
            - **Custody Account Harmonization**: Single account structure
            """)
        
        with gap_col2:
            st.markdown("""
            #### 💼 Business Process Gaps
            - **Merger Calculations**: One-button fund merger capability
            - **Fee/Expense Calculation**: Complex fee structure support
            - **OCF Capping & Umbrella Fees**: Enhanced calculations
            
            #### 📊 Reporting Gaps  
            - **PRFA Enhancement**: Calculation + data provision
            - **Regulatory Reporting**: Enhanced FA support
            - **Performance NAVs**: MBOR/IBOR capability
            - **Income Forecasting**: Automated projections
            
            #### 🏛️ Tax & Compliance Gaps
            - **CGT Processing**: Automated capital gains tax
            - **Taxation Linkages**: Better FA-Custody integration
            """)
        
        st.warning("⚠️ **Critical Impact**: These gaps represent operational risk and manual intervention points that reduce scalability and increase error probability.")
    
    with workstream_tab4:
        st.subheader("📈 Client Change Request Analytics")
        
        # Client change distribution
        if PLOTLY_AVAILABLE:
            change_data = {
                'Change Type': ['Fund Change', 'Reporting Change', 'Calculation Enhancements', 
                               'Expenses', 'Transaction Capture', 'Pricing'],
                'Percentage': [37, 34, 12, 10, 3.54, 1.77]
            }
            
            fig_changes = go.Figure(data=[go.Pie(
                labels=change_data['Change Type'],
                values=change_data['Percentage'],
                hole=.3,
                textinfo='label+percent',
                textposition='outside'
            )])
            
            fig_changes.update_layout(
                title="Client Change Request Distribution",
                height=500
            )
            
            st.plotly_chart(fig_changes, use_container_width=True)
        
        # Key insights
        st.markdown("""
        ### 🎯 Key Change Request Insights
        
        **Top Change Drivers:**
        1. **Fund Changes (37%)**: Structure modifications, investment policy updates
        2. **Reporting Changes (34%)**: Custom reporting requirements, regulatory updates
        3. **Calculation Enhancements (12%)**: Fee structures, NAV calculation refinements
        
        **Operational Impact:**
        - Fund and Reporting changes represent 71% of all requests
        - High manual intervention required for complex changes
        - Need for more flexible, configurable systems
        
        **Strategic Recommendations:**
        - Invest in self-service client portals
        - Enhance system configurability
        - Automate common change patterns
        """)
        
        # Change request metrics
        change_metrics_col1, change_metrics_col2, change_metrics_col3 = st.columns(3)
        
        with change_metrics_col1:
            st.metric("Avg. Processing Time", "14 days", "-3 days vs Q3")
        
        with change_metrics_col2:
            st.metric("Monthly Requests", "147", "+12% MoM")
        
        with change_metrics_col3:
            st.metric("Automation Rate", "23%", "+5% target")

# Tab 2: Asset Profiling and Fund Insights  
with main_tab2:
    st.markdown("""
    This section contains the periodic table of asset types and comprehensive analysis tools for asset profiling and fund insights.
    """)
    
    # --- Sidebar Controls ---
    st.sidebar.header("⚙️ Controls")

# Color metric selector
color_metric = st.sidebar.selectbox(
    "Color Code By:",
    options=['Risk', 'Liquidity', 'OpCost', 'OpRisk'],
    format_func=lambda x: {
        'Risk': 'Market Risk',
        'Liquidity': 'Liquidity',
        'OpCost': 'Operational Cost',
        'OpRisk': 'Operational Risk'
    }[x]
)

# Category filter
categories = ['All'] + sorted(df['Category'].unique().tolist())
selected_category = st.sidebar.selectbox(
    "Filter by Category:",
    options=categories
)

# Search functionality
search_term = st.sidebar.text_input(
    "Search Assets:",
    placeholder="Enter symbol or name..."
)

# Color scale legend
st.sidebar.markdown("---")
st.sidebar.header("🎨 Color Scale")
legend_html = """
<div style='display: flex; flex-direction: column; gap: 10px;'>
    <div style='display: flex; align-items: center; gap: 10px;'>
        <div style='width: 100px; height: 20px; background: linear-gradient(to right, rgb(255,40,40), rgb(255,142,40), rgb(255,255,40), rgb(142,255,40), rgb(40,255,40)); border: 1px solid #ccc;'></div>
        <span style='font-size: 12px;'>""" + ("Low → High Liquidity" if color_metric == 'Liquidity' else "Low → High " + color_metric) + """</span>
    </div>
    <div style='display: flex; justify-content: space-between; font-size: 10px; color: #666;'>
        <span>1</span><span>3</span><span>5</span><span>7</span><span>10</span>
    </div>
</div>
"""
st.sidebar.markdown(legend_html, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.header("📊 Metric Definitions")
st.sidebar.info(
    """
    - **Market Risk**: Potential for investment loss due to factors that affect the overall financial market (1=Low, 10=High).
    - **Liquidity**: The ease with which an asset can be converted into cash (1=Low, 10=High).
    - **Operational Cost**: The cost to process, settle, and manage the asset (1=Low, 10=High).
    - **Operational Risk**: Risk of loss from failed internal processes, people, or systems (1=Low, 10=High).
    """
)


# --- Filter Data Based on User Selection ---

# Apply category filter
filtered_df = df.copy()
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

# Apply search filter
if search_term:
    search_mask = (
        filtered_df['Symbol'].str.contains(search_term, case=False, na=False) |
        filtered_df['Name'].str.contains(search_term, case=False, na=False)
    )
    filtered_df = filtered_df[search_mask]

# Display statistics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Assets", len(df))
with col2:
    st.metric("Filtered Assets", len(filtered_df))
with col3:
    avg_metric_value = filtered_df[color_metric].mean() if len(filtered_df) > 0 else 0
    st.metric(f"Avg {color_metric}", f"{avg_metric_value:.1f}")

# --- Generate the Interactive Periodic Table ---

st.subheader("🧪 The Periodic Table of Asset Types")


# --- Advanced Asset Visualizations ---
st.subheader("📊 Interactive Asset Analysis Dashboard")

# Apply filters to df
display_df = df.copy()
if selected_category != 'All':
    display_df = display_df[display_df['Category'] == selected_category]
if search_term:
    search_mask = (
        display_df['Symbol'].str.contains(search_term, case=False, na=False) |
        display_df['Name'].str.contains(search_term, case=False, na=False)
    )
    display_df = display_df[search_mask]

# Create multiple visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔬 Risk-Liquidity Matrix", "🌡️ Heatmaps", "📈 Interactive Charts", "🎯 Asset Positioning"])

with tab1:
    st.write("### Risk vs Liquidity Analysis")
    
    if PLOTLY_AVAILABLE:
        # Create bubble chart showing risk vs liquidity
        fig_bubble = px.scatter(
            display_df,
            x='Risk',
            y='Liquidity', 
            size='OpCost',
            color=color_metric,
            hover_name='Symbol',
            hover_data={
                'Name': True,
                'Category': True,
                'OpRisk': True,
                'GridRow': True,
                'GridCol': True
            },
            title=f"Asset Risk-Liquidity Profile (Size=OpCost, Color={color_metric})",
            labels={
                'Risk': 'Market Risk Level (1-10)',
                'Liquidity': 'Liquidity Level (1-10)',
                'OpCost': 'Operational Cost'
            },
            color_continuous_scale='Viridis' if color_metric != 'Liquidity' else 'Viridis_r',
            size_max=30
        )
        
        # Add quadrant lines
        fig_bubble.add_hline(y=5.5, line_dash="dash", line_color="gray", opacity=0.5)
        fig_bubble.add_vline(x=5.5, line_dash="dash", line_color="gray", opacity=0.5)
        
        # Add quadrant annotations
        fig_bubble.add_annotation(x=2.5, y=8.5, text="💚 Safe Haven<br>(Low Risk, High Liquidity)", 
                                 showarrow=False, font=dict(size=10), bgcolor="lightgreen", opacity=0.8)
        fig_bubble.add_annotation(x=8.5, y=8.5, text="🟡 High Risk Liquid<br>(High Risk, High Liquidity)", 
                                 showarrow=False, font=dict(size=10), bgcolor="yellow", opacity=0.8)
        fig_bubble.add_annotation(x=2.5, y=2.5, text="🔵 Conservative Illiquid<br>(Low Risk, Low Liquidity)", 
                                 showarrow=False, font=dict(size=10), bgcolor="lightblue", opacity=0.8)
        fig_bubble.add_annotation(x=8.5, y=2.5, text="🔴 High Risk Illiquid<br>(High Risk, Low Liquidity)", 
                                 showarrow=False, font=dict(size=10), bgcolor="lightcoral", opacity=0.8)
        
        fig_bubble.update_layout(height=600, hovermode='closest')
        st.plotly_chart(fig_bubble, use_container_width=True)
        
        # Asset positioning insights
        st.write("#### 🎯 Asset Positioning Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            # Best and worst performers by selected metric
            best_assets = display_df.nlargest(3, color_metric)
            worst_assets = display_df.nsmallest(3, color_metric)
            
            st.write(f"**🏆 Highest {color_metric}:**")
            for _, asset in best_assets.iterrows():
                st.write(f"• **{asset['Symbol']}** ({asset['Name']}): {asset[color_metric]}/10")
            
            st.write(f"**⚠️ Lowest {color_metric}:**")
            for _, asset in worst_assets.iterrows():
                st.write(f"• **{asset['Symbol']}** ({asset['Name']}): {asset[color_metric]}/10")
        
        with col2:
            # Quadrant analysis
            safe_haven = display_df[(display_df['Risk'] <= 5) & (display_df['Liquidity'] >= 6)]
            high_risk_liquid = display_df[(display_df['Risk'] > 5) & (display_df['Liquidity'] >= 6)]
            conservative_illiquid = display_df[(display_df['Risk'] <= 5) & (display_df['Liquidity'] < 6)]
            high_risk_illiquid = display_df[(display_df['Risk'] > 5) & (display_df['Liquidity'] < 6)]
            
            st.write("**📊 Quadrant Distribution:**")
            st.write(f"• 💚 Safe Haven: {len(safe_haven)} assets")
            st.write(f"• 🟡 High Risk Liquid: {len(high_risk_liquid)} assets")
            st.write(f"• 🔵 Conservative Illiquid: {len(conservative_illiquid)} assets")
            st.write(f"• 🔴 High Risk Illiquid: {len(high_risk_illiquid)} assets")
    
    elif SEABORN_AVAILABLE:
        st.info("Using Matplotlib/Seaborn visualization")
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot
        scatter = ax.scatter(
            display_df['Risk'], 
            display_df['Liquidity'],
            s=display_df['OpCost'] * 20,  # Size based on OpCost
            c=display_df[color_metric],
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        
        # Add asset labels
        for _, asset in display_df.iterrows():
            ax.annotate(asset['Symbol'], 
                       (asset['Risk'], asset['Liquidity']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Market Risk Level (1-10)')
        ax.set_ylabel('Liquidity Level (1-10)')
        ax.set_title(f'Asset Risk-Liquidity Profile (Size=OpCost, Color={color_metric})')
        ax.grid(True, alpha=0.3)
        
        # Add quadrant lines
        ax.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
        
        plt.colorbar(scatter, label=color_metric)
        st.pyplot(fig, use_container_width=True)

with tab2:
    st.write("### Asset Metrics Heatmaps")
    
    if SEABORN_AVAILABLE:
        # Create correlation heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Metrics Correlation Matrix")
            corr_metrics = display_df[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].corr()
            
            fig_corr, ax_corr = plt.subplots(figsize=(8, 6))
            sns.heatmap(corr_metrics, annot=True, cmap='RdBu_r', center=0,
                       square=True, ax=ax_corr, cbar_kws={"shrink": .8})
            ax_corr.set_title('Asset Metrics Correlation')
            st.pyplot(fig_corr, use_container_width=True)
        
        with col2:
            st.write("#### Asset Category Heatmap")
            # Create category-wise average metrics
            category_metrics = display_df.groupby('Category')[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].mean()
            
            fig_cat, ax_cat = plt.subplots(figsize=(10, 6))
            sns.heatmap(category_metrics.T, annot=True, cmap='YlOrRd', 
                       cbar_kws={"shrink": .8}, ax=ax_cat)
            ax_cat.set_title('Average Metrics by Asset Category')
            ax_cat.set_xlabel('Asset Category')
            ax_cat.set_ylabel('Metrics')
            plt.xticks(rotation=45)
            st.pyplot(fig_cat, use_container_width=True)
    
    elif PLOTLY_AVAILABLE:
        # Plotly heatmaps
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("#### Metrics Correlation Matrix")
            corr_metrics = display_df[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].corr()
            
            fig_corr = px.imshow(corr_metrics, 
                               text_auto=True, 
                               aspect="auto",
                               color_continuous_scale='RdBu_r',
                               title="Asset Metrics Correlation")
            st.plotly_chart(fig_corr, use_container_width=True)
        
        with col2:
            st.write("#### Asset Metrics by Category")
            category_metrics = display_df.groupby('Category')[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].mean()
            
            fig_cat = px.imshow(category_metrics.T, 
                              text_auto=True, 
                              aspect="auto",
                              color_continuous_scale='YlOrRd',
                              title="Average Metrics by Category")
            st.plotly_chart(fig_cat, use_container_width=True)

with tab3:
    st.write("### Interactive Asset Charts")
    
    chart_type = st.selectbox("Choose Chart Type:", 
                             ["Scatter Matrix", "Parallel Coordinates", "Radar Chart", "Box Plots"])
    
    if chart_type == "Scatter Matrix" and PLOTLY_AVAILABLE:
        try:
            fig_matrix = px.scatter_matrix(
                display_df,
                dimensions=['Risk', 'Liquidity', 'OpCost', 'OpRisk'],
                color='Category',
                hover_name='Symbol',
                title="Asset Metrics Scatter Matrix"
            )
            fig_matrix.update_layout(height=600)
            st.plotly_chart(fig_matrix, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating scatter matrix: {str(e)}")
    
    elif chart_type == "Parallel Coordinates" and PLOTLY_AVAILABLE:
        try:
            fig_parallel = px.parallel_coordinates(
                display_df,
                dimensions=['Risk', 'Liquidity', 'OpCost', 'OpRisk'],
                color=color_metric,
                labels={'Symbol': 'Asset Symbol'},
                title=f"Parallel Coordinates Plot (Colored by {color_metric})"
            )
            st.plotly_chart(fig_parallel, use_container_width=True)
        except Exception as e:
            st.error(f"Error creating parallel coordinates: {str(e)}")
    
    elif chart_type == "Radar Chart" and PLOTLY_AVAILABLE:
        # Select assets for radar comparison
        selected_assets = st.multiselect(
            "Select assets to compare:",
            options=display_df['Symbol'].tolist(),
            default=display_df['Symbol'].tolist()[:5]
        )
        
        if selected_assets:
            fig_radar = go.Figure()
            
            for symbol in selected_assets:
                asset_data = display_df[display_df['Symbol'] == symbol].iloc[0]
                categories = ['Risk', 'Liquidity', 'OpCost', 'OpRisk']
                values = [asset_data[cat] for cat in categories]
                values.append(values[0])  # Close the radar chart
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=f"{symbol} ({asset_data['Name'][:20]}...)",
                    line=dict(width=2),
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 10])
                ),
                showlegend=True,
                title="Asset Comparison Radar Chart",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
    
    elif chart_type == "Box Plots":
        if PLOTLY_AVAILABLE:
            try:
                # Box plots for each metric
                metrics = ['Risk', 'Liquidity', 'OpCost', 'OpRisk']
                
                for metric in metrics:
                    fig_box = px.box(
                        display_df, 
                        x='Category', 
                        y=metric,
                        points="all",
                        hover_name='Symbol',
                        title=f"{metric} Distribution by Category"
                    )
                    fig_box.update_layout(xaxis_tickangle=45)
                    st.plotly_chart(fig_box, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating Plotly box plots: {str(e)}")
                st.info("Falling back to basic visualization...")
        
        elif SEABORN_AVAILABLE:
            metrics = ['Risk', 'Liquidity', 'OpCost', 'OpRisk']
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                sns.boxplot(data=display_df, x='Category', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric} Distribution by Category')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)

with tab4:
    st.write("### Asset Positioning Analysis")
    
    if PLOTLY_AVAILABLE:
        # 3D scatter plot
        fig_3d = px.scatter_3d(
            display_df,
            x='Risk',
            y='Liquidity', 
            z='OpCost',
            color=color_metric,
            size='OpRisk',
            hover_name='Symbol',
            hover_data={'Name': True, 'Category': True},
            title=f"3D Asset Positioning (Size=OpRisk, Color={color_metric})",
            labels={
                'Risk': 'Market Risk',
                'Liquidity': 'Liquidity Level',
                'OpCost': 'Operational Cost'
            }
        )
        
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        # Sunburst chart for category breakdown
        st.write("#### Asset Category Breakdown")
        fig_sunburst = px.sunburst(
            display_df,
            path=['Category', 'Symbol'],
            values='Risk',  # Use risk as the size metric
            color=color_metric,
            title=f"Asset Category Hierarchy (Size=Risk, Color={color_metric})",
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_sunburst, use_container_width=True)
    
    # Asset recommendations based on current selection
    st.write("### 🎯 Smart Asset Recommendations")
    
    if len(display_df) > 0:
        # Calculate composite scores
        display_df_copy = display_df.copy()
        
        # Liquidity score (higher is better)
        display_df_copy['Liquidity_Score'] = display_df_copy['Liquidity'] / 10
        
        # Risk score (lower is better for conservative investors)
        display_df_copy['Risk_Score'] = (11 - display_df_copy['Risk']) / 10
        
        # OpCost score (lower is better)
        display_df_copy['OpCost_Score'] = (11 - display_df_copy['OpCost']) / 10
        
        # OpRisk score (lower is better)
        display_df_copy['OpRisk_Score'] = (11 - display_df_copy['OpRisk']) / 10
        
        # Overall score
        display_df_copy['Overall_Score'] = (
            display_df_copy['Liquidity_Score'] * 0.3 +
            display_df_copy['Risk_Score'] * 0.3 +
            display_df_copy['OpCost_Score'] * 0.2 +
            display_df_copy['OpRisk_Score'] * 0.2
        )
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**🏆 Top Overall Assets:**")
            top_assets = display_df_copy.nlargest(5, 'Overall_Score')
            for i, (_, asset) in enumerate(top_assets.iterrows(), 1):
                st.write(f"{i}. **{asset['Symbol']}** - {asset['Name'][:30]}...")
                st.write(f"   Score: {asset['Overall_Score']:.3f}")
        
        with col2:
            st.write("**💧 Most Liquid Assets:**")
            liquid_assets = display_df_copy.nlargest(5, 'Liquidity')
            for i, (_, asset) in enumerate(liquid_assets.iterrows(), 1):
                st.write(f"{i}. **{asset['Symbol']}** - Liquidity: {asset['Liquidity']}/10")
        
        with col3:
            st.write("**🛡️ Lowest Risk Assets:**")
            safe_assets = display_df_copy.nsmallest(5, 'Risk')
            for i, (_, asset) in enumerate(safe_assets.iterrows(), 1):
                st.write(f"{i}. **{asset['Symbol']}** - Risk: {asset['Risk']}/10")

# --- Alternative Visualization with Altair ---
if ALTAIR_AVAILABLE:
    st.markdown("---")
    st.subheader("🎨 Alternative Interactive Visualization (Altair)")
    
    # Create an interactive scatter plot with selection
    brush = alt.selection_interval()
    
    base = alt.Chart(display_df).add_selection(brush)
    
    # Main scatter plot
    scatter_alt = base.mark_circle(size=200, opacity=0.8).encode(
        x=alt.X('Risk:Q', scale=alt.Scale(domain=[0, 11]), title='Market Risk Level'),
        y=alt.Y('Liquidity:Q', scale=alt.Scale(domain=[0, 11]), title='Liquidity Level'), 
        color=alt.Color(f'{color_metric}:Q', 
                       scale=alt.Scale(scheme='viridis'),
                       legend=alt.Legend(title=f"{color_metric} Level")),
        size=alt.Size('OpCost:Q', scale=alt.Scale(range=[100, 400]), legend=alt.Legend(title="Op Cost")),
        tooltip=['Symbol:N', 'Name:N', 'Category:N', 'Risk:Q', 'Liquidity:Q', 'OpCost:Q', 'OpRisk:Q'],
        stroke=alt.value('black'),
        strokeWidth=alt.value(1)
    ).properties(
        title=f"Interactive Asset Risk-Liquidity Analysis (Color={color_metric}, Size=OpCost)",
        width=700,
        height=400
    )
    
    # Bar chart showing category distribution of selected points
    bars = base.mark_bar().encode(
        x=alt.X('count():Q', title='Number of Assets'),
        y=alt.Y('Category:N', title='Asset Category'),
        color=alt.condition(brush, alt.Color('Category:N'), alt.value('lightgray')),
        tooltip=['Category:N', 'count():Q']
    ).transform_filter(
        brush
    ).properties(
        title="Selected Assets by Category",
        width=300,
        height=400
    )
    
    # Combine charts
    combined_alt = alt.hconcat(scatter_alt, bars).resolve_legend(
        color="independent",
        size="independent"
    )
    
    st.altair_chart(combined_alt, use_container_width=True)
    st.info("💡 **Interactive Feature**: Select an area in the left chart to filter the category breakdown on the right!")

# Add enhanced legend and instructions
st.markdown("---")
st.markdown("""
## 📚 **Comprehensive Asset Analysis Guide**

### **💡 How to Navigate:**
- **🔬 Risk-Liquidity Matrix**: Interactive bubble chart with quadrant analysis
- **🌡️ Heatmaps**: Correlation analysis and category-wise metric averages  
- **📈 Interactive Charts**: Multiple chart types including scatter matrix, parallel coordinates, radar charts, and box plots
- **🎯 Asset Positioning**: 3D visualization and hierarchical category breakdown
- **🎨 Alternative Visualization**: Altair-powered interactive selection charts

### **🎯 Key Insights:**
- **Safe Haven Assets** 💚: Low risk, high liquidity (top-left quadrant)
- **High Risk Liquid** 🟡: Suitable for active trading (top-right quadrant)  
- **Conservative Illiquid** 🔵: Long-term, stable investments (bottom-left quadrant)
- **High Risk Illiquid** 🔴: Speculative, alternative investments (bottom-right quadrant)

### **📊 Visual Encoding:**
- **Bubble Size**: Represents operational cost or risk
- **Color**: Represents the selected metric intensity
- **Position**: Risk (X-axis) vs Liquidity (Y-axis)
- **Hover Details**: Complete asset information and metrics

### **🚀 Advanced Features:**
- **Real-time Filtering**: Search and category filters update all visualizations
- **Interactive Selection**: Brush selection in Altair charts
- **Multi-dimensional Analysis**: 3D plots and parallel coordinates
- **Smart Recommendations**: Algorithm-based asset scoring and ranking
""")

# --- Real Financial Data Analysis ---
if real_assets_df is not None and real_funds_df is not None:
    st.markdown("---")
    st.header("🏦 Real Financial Data Analysis")
    
    # Create tabs for real data analysis
    tab_assets, tab_funds, tab_combined = st.tabs(["📊 Asset Classes", "🏛️ Fund Types", "🔗 Combined Analysis"])
    
    with tab_assets:
        st.subheader("Real Asset Classes Analysis")
        
        if PLOTLY_AVAILABLE:
            # Risk vs Liquidity scatter for real assets
            fig_real_assets = px.scatter(
                real_assets_df,
                x='Risk_Score',
                y='Liquidity_Score',
                size='Cost_Score', 
                color='Ops_Risk_Score',
                hover_name='Asset_Type',
                hover_data={'Asset_Class': True, 'GICS_Sector': True, 'Reference_Details': True},
                title="Real Asset Classes: Risk vs Liquidity Profile",
                labels={
                    'Risk_Score': 'Risk Level (1-5)',
                    'Liquidity_Score': 'Liquidity Level (1-5)',
                    'Ops_Risk_Score': 'Operational Risk',
                    'Cost_Score': 'Cost Level'
                },
                color_continuous_scale='Reds'
            )
            
            fig_real_assets.update_layout(height=600)
            st.plotly_chart(fig_real_assets, use_container_width=True)
        
        # Asset class breakdown
        st.write("### Asset Class Distribution")
        asset_class_counts = real_assets_df['Asset_Class'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(asset_class_counts)
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig_pie = px.pie(
                    values=asset_class_counts.values,
                    names=asset_class_counts.index,
                    title="Asset Classes Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        # GICS Sector analysis
        st.write("### GICS Sector Breakdown")
        gics_sectors = real_assets_df[real_assets_df['GICS_Sector'] != 'N/A']['GICS_Sector'].value_counts()
        if len(gics_sectors) > 0:
            st.bar_chart(gics_sectors)
        else:
            st.info("Most assets are sector-agnostic (N/A)")
        
        # Risk-Return Matrix
        st.write("### Risk Profile Analysis")
        risk_analysis = real_assets_df.groupby('Asset_Class')[['Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']].mean()
        
        if SEABORN_AVAILABLE:
            fig_heatmap, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(risk_analysis.T, annot=True, cmap='RdYlBu_r', ax=ax, cbar_kws={"shrink": .8})
            ax.set_title('Average Risk Metrics by Asset Class')
            plt.xticks(rotation=45)
            st.pyplot(fig_heatmap, use_container_width=True)
        
        # Detailed asset table
        st.write("### Detailed Asset Information")
        st.dataframe(
            real_assets_df.style.background_gradient(
                subset=['Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score'], 
                cmap='RdYlGn_r'
            ),
            use_container_width=True,
            height=500
        )
    
    with tab_funds:
        st.subheader("Fund Types Analysis")
        
        if PLOTLY_AVAILABLE:
            # Fund risk analysis
            fig_funds = px.scatter(
                real_funds_df,
                x='Risk_Score',
                y='Liquidity_Score',
                size='Cost_Score',
                color='Regulatory_Framework',
                hover_name='Fund_Type',
                hover_data={'Legal_Structure': True, 'Key_Characteristics': True},
                title="Fund Types: Risk vs Liquidity Profile",
                labels={
                    'Risk_Score': 'Risk Level (1-5)',
                    'Liquidity_Score': 'Liquidity Level (1-5)',
                    'Cost_Score': 'Cost Level'
                }
            )
            
            fig_funds.update_layout(height=600)
            st.plotly_chart(fig_funds, use_container_width=True)
        
        # Regulatory framework analysis
        st.write("### Regulatory Framework Distribution")
        regulatory_counts = real_funds_df['Regulatory_Framework'].value_counts()
        
        col1, col2 = st.columns(2)
        with col1:
            st.bar_chart(regulatory_counts)
            
            # Show framework details
            st.write("**Framework Characteristics:**")
            for framework in regulatory_counts.index:
                fund_count = regulatory_counts[framework]
                st.write(f"• **{framework}**: {fund_count} fund types")
        
        with col2:
            if PLOTLY_AVAILABLE:
                fig_reg_pie = px.pie(
                    values=regulatory_counts.values,
                    names=regulatory_counts.index,
                    title="Regulatory Framework Distribution"
                )
                st.plotly_chart(fig_reg_pie, use_container_width=True)
        
        # Fund complexity analysis
        st.write("### Fund Complexity Matrix")
        fund_metrics = real_funds_df.groupby('Regulatory_Framework')[['Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']].mean()
        
        if SEABORN_AVAILABLE:
            fig_fund_heat, ax = plt.subplots(figsize=(8, 4))
            sns.heatmap(fund_metrics.T, annot=True, cmap='RdYlBu_r', ax=ax, cbar_kws={"shrink": .8})
            ax.set_title('Average Metrics by Regulatory Framework')
            st.pyplot(fig_fund_heat, use_container_width=True)
        
        # Detailed fund table
        st.write("### Detailed Fund Information")
        st.dataframe(
            real_funds_df.style.background_gradient(
                subset=['Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score'],
                cmap='RdYlGn_r'
            ),
            use_container_width=True,
            height=500
        )
    
    with tab_combined:
        st.subheader("Combined Assets & Funds Analysis")
        
        # Create combined dataset for analysis
        assets_combined = real_assets_df[['Asset_Type', 'Asset_Class', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']].copy()
        assets_combined['Type'] = 'Asset'
        assets_combined['Category'] = assets_combined['Asset_Class']
        assets_combined['Name'] = assets_combined['Asset_Type']
        
        funds_combined = real_funds_df[['Fund_Type', 'Regulatory_Framework', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']].copy()
        funds_combined['Type'] = 'Fund'
        funds_combined['Category'] = funds_combined['Regulatory_Framework']
        funds_combined['Name'] = funds_combined['Fund_Type']
        
        # Combine datasets
        combined_df = pd.concat([
            assets_combined[['Name', 'Category', 'Type', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']],
            funds_combined[['Name', 'Category', 'Type', 'Risk_Score', 'Liquidity_Score', 'Ops_Risk_Score', 'Cost_Score']]
        ], ignore_index=True)
        
        if PLOTLY_AVAILABLE:
            # Combined risk-liquidity analysis
            fig_combined = px.scatter(
                combined_df,
                x='Risk_Score',
                y='Liquidity_Score',
                size='Cost_Score',
                color='Type',
                hover_name='Name',
                hover_data={'Category': True, 'Ops_Risk_Score': True},
                title="Complete Financial Universe: Assets vs Funds",
                labels={
                    'Risk_Score': 'Risk Level (1-5)',
                    'Liquidity_Score': 'Liquidity Level (1-5)',
                    'Cost_Score': 'Cost Level'
                }
            )
            
            fig_combined.update_layout(height=600)
            st.plotly_chart(fig_combined, use_container_width=True)
        
        # Summary statistics
        st.write("### Comparative Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Asset Summary:**")
            st.metric("Total Asset Types", len(real_assets_df))
            st.metric("Avg Risk Score", f"{real_assets_df['Risk_Score'].mean():.2f}")
            st.metric("Avg Liquidity", f"{real_assets_df['Liquidity_Score'].mean():.2f}")
        
        with col2:
            st.write("**Fund Summary:**")
            st.metric("Total Fund Types", len(real_funds_df))
            st.metric("Avg Risk Score", f"{real_funds_df['Risk_Score'].mean():.2f}")
            st.metric("Avg Liquidity", f"{real_funds_df['Liquidity_Score'].mean():.2f}")
        
        with col3:
            st.write("**Combined Insights:**")
            high_risk_assets = len(combined_df[combined_df['Risk_Score'] >= 4])
            high_liquidity_assets = len(combined_df[combined_df['Liquidity_Score'] <= 2])  # Lower score = higher liquidity
            
            st.metric("High Risk Items", high_risk_assets)
            st.metric("High Liquidity Items", high_liquidity_assets)
        
        # Risk distribution comparison
        st.write("### Risk Distribution Comparison")
        
        if PLOTLY_AVAILABLE:
            fig_risk_dist = px.histogram(
                combined_df,
                x='Risk_Score',
                color='Type',
                title="Risk Score Distribution: Assets vs Funds",
                nbins=5,
                barmode='group'
            )
            st.plotly_chart(fig_risk_dist, use_container_width=True)

# --- Operational Data Analysis ---
if nav_data is not None and fund_characteristics is not None and custody_holdings is not None:
    st.markdown("---")
    st.header("🏢 Operational Fund Data Analysis")
    st.info("Real operational data from fund administration systems including NAV, holdings, and fund characteristics.")
    
    # Create tabs for operational data analysis
    op_tab_nav, op_tab_holdings, op_tab_characteristics, op_tab_dashboard, op_tab_workstreams = st.tabs([
        "📈 NAV Performance", "📊 Portfolio Holdings", "🏛️ Fund Characteristics", "📋 Operations Dashboard", "🔗 Workstream Network"
    ])
    
    with op_tab_nav:
        st.subheader("NAV Performance Analysis")
        
        if PLOTLY_AVAILABLE:
            # NAV time series analysis
            st.write("**Daily NAV Performance by Fund**")
            
            # Create NAV time series chart
            fig_nav = px.line(
                nav_data,
                x='nav_date',
                y='nav_per_share',
                color='fund_id',
                title="Daily NAV Per Share - All Funds",
                labels={'nav_date': 'Date', 'nav_per_share': 'NAV Per Share', 'fund_id': 'Fund ID'}
            )
            fig_nav.update_layout(height=500)
            st.plotly_chart(fig_nav, use_container_width=True)
            
            # NAV statistics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**NAV Performance Metrics**")
                nav_stats = nav_data.groupby('fund_id').agg({
                    'nav_per_share': ['min', 'max', 'mean', 'std'],
                    'total_nav': ['mean'],
                    'total_shares_outstanding': ['mean']
                }).round(4)
                nav_stats.columns = ['Min NAV', 'Max NAV', 'Avg NAV', 'NAV Volatility', 'Avg Total NAV', 'Avg Shares Outstanding']
                st.dataframe(nav_stats, use_container_width=True)
            
            with col2:
                st.write("**NAV Volatility Analysis**")
                fig_vol = px.box(
                    nav_data,
                    x='fund_id',
                    y='nav_per_share',
                    title="NAV Per Share Distribution by Fund"
                )
                fig_vol.update_layout(height=400)
                st.plotly_chart(fig_vol, use_container_width=True)
        else:
            st.write("**NAV Data Summary**")
            st.dataframe(nav_data.head(10), use_container_width=True)
    
    with op_tab_holdings:
        st.subheader("Portfolio Holdings Analysis")
        
        if PLOTLY_AVAILABLE:
            # Holdings analysis
            st.write("**Portfolio Composition by Asset Class**")
            
            # Aggregate holdings by asset class
            holdings_summary = custody_holdings.groupby('asset_class').agg({
                'market_value': 'sum',
                'quantity': 'count',
                'unrealized_gain_loss': 'sum'
            }).reset_index()
            
            # Asset class pie chart
            fig_holdings = px.pie(
                holdings_summary,
                values='market_value',
                names='asset_class',
                title="Portfolio Value by Asset Class",
                hover_data=['quantity']
            )
            st.plotly_chart(fig_holdings, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Holdings by Currency**")
                currency_summary = custody_holdings.groupby('currency')['market_value'].sum().reset_index()
                fig_currency = px.bar(
                    currency_summary,
                    x='currency',
                    y='market_value',
                    title="Total Holdings by Currency"
                )
                st.plotly_chart(fig_currency, use_container_width=True)
            
            with col2:
                st.write("**P&L Analysis**")
                pnl_summary = custody_holdings.groupby('asset_class')['unrealized_gain_loss'].sum().reset_index()
                fig_pnl = px.bar(
                    pnl_summary,
                    x='asset_class',
                    y='unrealized_gain_loss',
                    title="Unrealized P&L by Asset Class",
                    color='unrealized_gain_loss',
                    color_continuous_scale='RdYlGn'
                )
                st.plotly_chart(fig_pnl, use_container_width=True)
            
            # Holdings detail table
            st.write("**Holdings Detail**")
            holdings_display = custody_holdings.copy()
            holdings_display['market_value'] = holdings_display['market_value'].round(2)
            holdings_display['unrealized_gain_loss'] = holdings_display['unrealized_gain_loss'].round(2)
            st.dataframe(holdings_display, use_container_width=True, height=300)
            
        else:
            st.write("**Holdings Data Summary**")
            st.dataframe(custody_holdings.head(10), use_container_width=True)
    
    with op_tab_characteristics:
        st.subheader("Fund Characteristics Analysis")
        
        if PLOTLY_AVAILABLE:
            # Fund characteristics analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Fund Type Distribution**")
                fund_type_counts = fund_characteristics['fund_type'].value_counts().reset_index()
                fund_type_counts.columns = ['Fund Type', 'Count']
                fig_fund_types = px.pie(
                    fund_type_counts,
                    values='Count',
                    names='Fund Type',
                    title="Distribution of Fund Types"
                )
                st.plotly_chart(fig_fund_types, use_container_width=True)
            
            with col2:
                st.write("**Legal Structure Distribution**")
                structure_counts = fund_characteristics['legal_structure'].value_counts().reset_index()
                structure_counts.columns = ['Legal Structure', 'Count']
                fig_structures = px.bar(
                    structure_counts,
                    x='Legal Structure',
                    y='Count',
                    title="Legal Structure Distribution"
                )
                st.plotly_chart(fig_structures, use_container_width=True)
            
            # AUM analysis
            st.write("**Assets Under Management Analysis**")
            fig_aum = px.scatter(
                fund_characteristics,
                x='target_aum_min',
                y='target_aum_max',
                size='aum_current_estimate',
                color='fund_type',
                hover_name='fund_name',
                title="Target AUM Range vs Current AUM Estimate",
                labels={
                    'target_aum_min': 'Target AUM Minimum',
                    'target_aum_max': 'Target AUM Maximum',
                    'aum_current_estimate': 'Current AUM Estimate'
                }
            )
            fig_aum.update_layout(height=500)
            st.plotly_chart(fig_aum, use_container_width=True)
            
            # Fund characteristics table
            st.write("**Fund Characteristics Detail**")
            chars_display = fund_characteristics.copy()
            chars_display['aum_current_estimate'] = chars_display['aum_current_estimate'].round(2)
            st.dataframe(chars_display, use_container_width=True)
            
        else:
            st.write("**Fund Characteristics Summary**")
            st.dataframe(fund_characteristics, use_container_width=True)
    
    with op_tab_dashboard:
        st.subheader("Operations Dashboard")
        
        # Key metrics summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_funds = len(fund_characteristics)
            st.metric("Total Funds", total_funds)
        
        with col2:
            total_holdings = len(custody_holdings)
            st.metric("Total Holdings", total_holdings)
        
        with col3:
            total_aum = fund_characteristics['aum_current_estimate'].sum()
            st.metric("Total AUM", f"${total_aum:,.0f}")
        
        with col4:
            total_pnl = custody_holdings['unrealized_gain_loss'].sum()
            st.metric("Total Unrealized P&L", f"${total_pnl:,.0f}")
        
        if PLOTLY_AVAILABLE:
            # Operational risk indicators
            st.write("**Operational Risk Indicators**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Expense ratio analysis
                fig_expense = px.histogram(
                    fund_characteristics,
                    x='expense_ratio_pct',
                    title="Expense Ratio Distribution",
                    nbins=10
                )
                st.plotly_chart(fig_expense, use_container_width=True)
            
            with col2:
                # Fund age analysis
                fund_characteristics['fund_age_years'] = (
                    pd.Timestamp.now() - fund_characteristics['inception_date']
                ).dt.days / 365.25
                
                fig_age = px.scatter(
                    fund_characteristics,
                    x='fund_age_years',
                    y='aum_current_estimate',
                    size='expense_ratio_pct',
                    color='fund_type',
                    title="Fund Age vs AUM",
                    labels={'fund_age_years': 'Fund Age (Years)', 'aum_current_estimate': 'Current AUM'}
                )
                st.plotly_chart(fig_age, use_container_width=True)
            
            # Custodian and safekeeping analysis
            st.write("**Custodian and Safekeeping Analysis**")
            safekeeping_summary = custody_holdings.groupby('safekeeping_location')['market_value'].sum().reset_index()
            fig_safekeeping = px.treemap(
                safekeeping_summary,
                path=['safekeeping_location'],
                values='market_value',
                title="Assets by Safekeeping Location"
            )
            st.plotly_chart(fig_safekeeping, use_container_width=True)
    
    with op_tab_workstreams:
        st.subheader("Operational Workstream Network Analysis")
        st.info("Network analysis of key fund administration workstreams and their interconnections.")
        
        # Define operational workstreams based on the loaded data
        operational_workstreams = {
            "Transfer Agent": {
                "description": "Shareholder record keeping and transaction processing",
                "complexity": 8,
                "automation": 6,
                "risk": 7,
                "impact": 8,
                "applications": ["ShareTrak", "InvestorLink", "RegCom"],
                "dependencies": ["NAV Calculation", "Accounting", "Compliance"],
                "data_sources": ["fund_characteristics", "nav_data"]
            },
            "Custody": {
                "description": "Asset safekeeping and settlement services",
                "complexity": 9,
                "automation": 7,
                "risk": 9,
                "impact": 9,
                "applications": ["CustodyPro", "SettleNet", "AssetGuard"],
                "dependencies": ["Trading", "Accounting", "Risk Management"],
                "data_sources": ["custody_holdings"]
            },
            "Accounting": {
                "description": "Financial reporting and book keeping",
                "complexity": 7,
                "automation": 8,
                "risk": 6,
                "impact": 8,
                "applications": ["FundBooks", "AcctRec", "FinReport"],
                "dependencies": ["NAV Calculation", "Custody", "Compliance"],
                "data_sources": ["nav_data", "custody_holdings", "fund_characteristics"]
            },
            "NAV Calculation": {
                "description": "Daily net asset value computation",
                "complexity": 9,
                "automation": 5,
                "risk": 8,
                "impact": 10,
                "applications": ["NAVCalc", "PriceLink", "ValEngine"],
                "dependencies": ["Pricing", "Custody", "Accounting"],
                "data_sources": ["nav_data"]
            },
            "Compliance": {
                "description": "Regulatory monitoring and reporting",
                "complexity": 8,
                "automation": 4,
                "risk": 9,
                "impact": 9,
                "applications": ["CompliTrack", "RegReport", "MonitorPro"],
                "dependencies": ["All Workstreams"],
                "data_sources": ["fund_characteristics"]
            },
            "Risk Management": {
                "description": "Portfolio risk monitoring and analysis",
                "complexity": 9,
                "automation": 6,
                "risk": 8,
                "impact": 9,
                "applications": ["RiskAnalyzer", "LimitTrack", "StressTest"],
                "dependencies": ["Custody", "Pricing", "NAV Calculation"],
                "data_sources": ["custody_holdings", "nav_data"]
            }
        }
        
        if PLOTLY_AVAILABLE and NETWORKX_AVAILABLE:
            # Create network graph
            G = nx.Graph()
            
            # Add nodes (workstreams)
            for workstream, data in operational_workstreams.items():
                G.add_node(workstream, 
                          complexity=data["complexity"],
                          automation=data["automation"],
                          risk=data["risk"],
                          impact=data["impact"],
                          type="workstream")
            
            # Add application nodes
            applications = set()
            for data in operational_workstreams.values():
                applications.update(data["applications"])
            
            for app in applications:
                G.add_node(app, type="application")
            
            # Add edges for dependencies
            for workstream, data in operational_workstreams.items():
                # Connect to applications
                for app in data["applications"]:
                    G.add_edge(workstream, app, relationship="uses")
                
                # Connect to dependencies
                for dep in data["dependencies"]:
                    if dep != "All Workstreams" and dep in operational_workstreams:
                        G.add_edge(workstream, dep, relationship="depends_on")
            
            # Create network visualization
            pos = nx.spring_layout(G, k=3, iterations=50)
            
            # Separate workstream and application nodes
            workstream_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'workstream']
            app_nodes = [node for node in G.nodes() if G.nodes[node].get('type') == 'application']
            
            # Create Plotly network graph
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=0.5, color='#888'),
                                   hoverinfo='none',
                                   mode='lines')
            
            # Workstream nodes
            workstream_x = [pos[node][0] for node in workstream_nodes]
            workstream_y = [pos[node][1] for node in workstream_nodes]
            workstream_text = [f"{node}<br>Risk: {G.nodes[node]['risk']}<br>Impact: {G.nodes[node]['impact']}" 
                              for node in workstream_nodes]
            
            workstream_trace = go.Scatter(x=workstream_x, y=workstream_y,
                                         mode='markers+text',
                                         hoverinfo='text',
                                         hovertext=workstream_text,
                                         text=workstream_nodes,
                                         textposition="middle center",
                                         marker=dict(size=20,
                                                   color=[G.nodes[node]['risk'] for node in workstream_nodes],
                                                   colorscale='Reds',
                                                   colorbar=dict(title="Risk Level"),
                                                   line=dict(width=2, color='black')))
            
            # Application nodes
            app_x = [pos[node][0] for node in app_nodes]
            app_y = [pos[node][1] for node in app_nodes]
            
            app_trace = go.Scatter(x=app_x, y=app_y,
                                  mode='markers+text',
                                  hoverinfo='text',
                                  hovertext=app_nodes,
                                  text=app_nodes,
                                  textposition="middle center",
                                  marker=dict(size=12,
                                            color='lightblue',
                                            line=dict(width=1, color='black')))
            
            fig_network = go.Figure(data=[edge_trace, workstream_trace, app_trace],
                                   layout=go.Layout(title=dict(text='Operational Workstream Network', font_size=16),
                                                   showlegend=False,
                                                   hovermode='closest',
                                                   margin=dict(b=20,l=5,r=5,t=40),
                                                   annotations=[ dict(
                                                       text="Red intensity indicates risk level. Blue nodes are applications.",
                                                       showarrow=False,
                                                       xref="paper", yref="paper",
                                                       x=0.005, y=-0.002,
                                                       xanchor="left", yanchor="bottom",
                                                       font=dict(size=12)
                                                   )],
                                                   xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                   yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                                   height=600))
            
            st.plotly_chart(fig_network, use_container_width=True)
            
            # Workstream metrics analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Workstream Risk-Complexity Matrix**")
                workstream_df = pd.DataFrame(operational_workstreams).T
                workstream_df = workstream_df.reset_index()
                workstream_df['complexity'] = pd.to_numeric(workstream_df['complexity'])
                workstream_df['risk'] = pd.to_numeric(workstream_df['risk'])
                workstream_df['impact'] = pd.to_numeric(workstream_df['impact'])
                workstream_df['automation'] = pd.to_numeric(workstream_df['automation'])
                
                fig_risk_complex = px.scatter(
                    workstream_df,
                    x='complexity',
                    y='risk',
                    size='impact',
                    color='automation',
                    hover_name='index',
                    title="Risk vs Complexity (Size = Impact, Color = Automation)",
                    labels={'index': 'Workstream', 'complexity': 'Complexity', 'risk': 'Risk Level'}
                )
                st.plotly_chart(fig_risk_complex, use_container_width=True)
            
            with col2:
                st.write("**Data Source Dependencies**")
                # Create data source analysis
                data_sources = {}
                for workstream, data in operational_workstreams.items():
                    for source in data["data_sources"]:
                        if source not in data_sources:
                            data_sources[source] = []
                        data_sources[source].append(workstream)
                
                source_counts = pd.DataFrame([(source, len(workstreams)) for source, workstreams in data_sources.items()],
                                           columns=['Data Source', 'Workstream Count'])
                
                fig_sources = px.bar(
                    source_counts,
                    x='Data Source',
                    y='Workstream Count',
                    title="Data Source Usage Across Workstreams"
                )
                st.plotly_chart(fig_sources, use_container_width=True)
            
            # Critical path analysis
            st.write("**Critical Path Analysis**")
            centrality = nx.betweenness_centrality(G)
            degree_centrality = nx.degree_centrality(G)
            
            centrality_df = pd.DataFrame([
                {'Node': node, 'Betweenness': centrality[node], 'Degree': degree_centrality[node], 
                 'Type': G.nodes[node].get('type', 'unknown')}
                for node in G.nodes()
            ]).sort_values('Betweenness', ascending=False)
            
            st.dataframe(centrality_df, use_container_width=True)
            
        else:
            st.write("**Operational Workstreams Summary**")
            workstream_summary = pd.DataFrame(operational_workstreams).T[['complexity', 'automation', 'risk', 'impact']]
            st.dataframe(workstream_summary, use_container_width=True)

# --- 3D Fund Positioning Analysis ---
if nav_data is not None and fund_characteristics is not None and custody_holdings is not None:
    st.markdown("---")
    st.header("🎯 3D Fund Positioning Analysis")
    st.info("Interactive 3D analysis of fund types with selectable funds and positioning metrics.")
    
    # Fund selection interface
    col1, col2, col3 = st.columns(3)
    
    with col1:
        available_funds = fund_characteristics['fund_id'].unique().tolist()
        selected_funds = st.multiselect(
            "Select Funds to Analyze:",
            options=available_funds,
            default=available_funds[:3] if len(available_funds) >= 3 else available_funds,
            key="fund_3d_selection"
        )
    
    with col2:
        x_metric = st.selectbox(
            "X-Axis Metric:",
            options=['aum_current_estimate', 'expense_ratio_pct', 'fund_age_years'],
            format_func=lambda x: {
                'aum_current_estimate': 'Current AUM',
                'expense_ratio_pct': 'Expense Ratio (%)',
                'fund_age_years': 'Fund Age (Years)'
            }[x],
            key="x_axis_3d"
        )
    
    with col3:
        y_metric = st.selectbox(
            "Y-Axis Metric:",
            options=['expense_ratio_pct', 'aum_current_estimate', 'fund_age_years'],
            format_func=lambda x: {
                'aum_current_estimate': 'Current AUM',
                'expense_ratio_pct': 'Expense Ratio (%)',
                'fund_age_years': 'Fund Age (Years)'
            }[x],
            index=1,
            key="y_axis_3d"
        )
    
    if selected_funds and PLOTLY_AVAILABLE:
        # Prepare fund data for 3D analysis
        fund_3d_data = fund_characteristics[fund_characteristics['fund_id'].isin(selected_funds)].copy()
        
        # Calculate fund age
        fund_3d_data['fund_age_years'] = (
            pd.Timestamp.now() - fund_3d_data['inception_date']
        ).dt.days / 365.25
        
        # Get NAV volatility for Z-axis
        nav_volatility = nav_data.groupby('fund_id')['nav_per_share'].std().reset_index()
        nav_volatility.columns = ['fund_id', 'nav_volatility']
        fund_3d_data = fund_3d_data.merge(nav_volatility, on='fund_id', how='left')
        
        # Get average holdings value for size
        holdings_avg = custody_holdings.groupby('fund_id')['market_value'].mean().reset_index()
        holdings_avg.columns = ['fund_id', 'avg_holding_value']
        fund_3d_data = fund_3d_data.merge(holdings_avg, on='fund_id', how='left')
        
        # Fill NaN values with defaults
        fund_3d_data['nav_volatility'] = fund_3d_data['nav_volatility'].fillna(0.1)
        fund_3d_data['avg_holding_value'] = fund_3d_data['avg_holding_value'].fillna(1000000)
        
        # Ensure all numeric columns are properly converted
        numeric_cols = ['aum_current_estimate', 'expense_ratio_pct', 'fund_age_years', 'nav_volatility', 'avg_holding_value']
        for col in numeric_cols:
            if col in fund_3d_data.columns:
                fund_3d_data[col] = pd.to_numeric(fund_3d_data[col], errors='coerce').fillna(0)
        
        # Create tabs for different 3D views
        tab_3d_main, tab_3d_risk, tab_3d_performance = st.tabs([
            "📊 Main 3D Analysis", "⚠️ Risk Positioning", "📈 Performance Metrics"
        ])
        
        with tab_3d_main:
            st.subheader("Interactive 3D Fund Positioning")
            
            # Main 3D scatter plot
            fig_3d_main = px.scatter_3d(
                fund_3d_data,
                x=x_metric,
                y=y_metric,
                z='nav_volatility',
                size='avg_holding_value',
                color='fund_type',
                hover_name='fund_name',
                hover_data={
                    'fund_id': True,
                    'legal_structure': True,
                    'base_currency': True,
                    'is_active': True
                },
                title=f"3D Fund Analysis: {x_metric.replace('_', ' ').title()} vs {y_metric.replace('_', ' ').title()} vs NAV Volatility",
                labels={
                    x_metric: x_metric.replace('_', ' ').title(),
                    y_metric: y_metric.replace('_', ' ').title(),
                    'nav_volatility': 'NAV Volatility',
                    'avg_holding_value': 'Avg Holding Value'
                }
            )
            fig_3d_main.update_layout(height=600)
            st.plotly_chart(fig_3d_main, use_container_width=True)
            
            # Fund comparison table
            st.write("**Selected Fund Comparison**")
            comparison_cols = ['fund_name', 'fund_type', 'legal_structure', 'aum_current_estimate', 
                             'expense_ratio_pct', 'fund_age_years', 'nav_volatility']
            comparison_df = fund_3d_data[comparison_cols].round(4)
            st.dataframe(comparison_df, use_container_width=True)
        
        with tab_3d_risk:
            st.subheader("Risk-Based 3D Positioning")
            
            # Risk-focused 3D analysis
            fig_3d_risk = px.scatter_3d(
                fund_3d_data,
                x='expense_ratio_pct',
                y='nav_volatility',
                z='aum_current_estimate',
                size='fund_age_years',
                color='legal_structure',
                hover_name='fund_name',
                title="Risk Profile Analysis: Expense Ratio vs NAV Volatility vs AUM",
                labels={
                    'expense_ratio_pct': 'Expense Ratio (%)',
                    'nav_volatility': 'NAV Volatility',
                    'aum_current_estimate': 'Current AUM',
                    'fund_age_years': 'Fund Age (Years)'
                }
            )
            fig_3d_risk.update_layout(height=600)
            st.plotly_chart(fig_3d_risk, use_container_width=True)
            
            # Risk metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Risk Profile Summary**")
                risk_summary = fund_3d_data.groupby('fund_type').agg({
                    'nav_volatility': 'mean',
                    'expense_ratio_pct': 'mean',
                    'aum_current_estimate': 'mean'
                }).round(4)
                st.dataframe(risk_summary, use_container_width=True)
            
            with col2:
                st.write("**Legal Structure Risk Distribution**")
                fig_structure_risk = px.box(
                    fund_3d_data,
                    x='legal_structure',
                    y='nav_volatility',
                    title="NAV Volatility by Legal Structure"
                )
                st.plotly_chart(fig_structure_risk, use_container_width=True)
        
        with tab_3d_performance:
            st.subheader("Performance Metrics 3D View")
            
            # Get NAV performance metrics
            nav_performance = nav_data.groupby('fund_id').agg({
                'nav_per_share': ['mean', 'min', 'max']
            }).round(4)
            nav_performance.columns = ['avg_nav', 'min_nav', 'max_nav']
            nav_performance['nav_range'] = nav_performance['max_nav'] - nav_performance['min_nav']
            nav_performance = nav_performance.reset_index()
            
            fund_perf_data = fund_3d_data.merge(nav_performance, on='fund_id', how='left')
            
            # Performance 3D scatter
            fig_3d_perf = px.scatter_3d(
                fund_perf_data,
                x='avg_nav',
                y='nav_range',
                z='aum_current_estimate',
                size='expense_ratio_pct',
                color='base_currency',
                hover_name='fund_name',
                title="Performance Analysis: Average NAV vs NAV Range vs AUM",
                labels={
                    'avg_nav': 'Average NAV',
                    'nav_range': 'NAV Range (Max - Min)',
                    'aum_current_estimate': 'Current AUM',
                    'expense_ratio_pct': 'Expense Ratio (%)'
                }
            )
            fig_3d_perf.update_layout(height=600)
            st.plotly_chart(fig_3d_perf, use_container_width=True)
            
            # Performance insights
            st.write("**Performance Insights**")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                best_performer = fund_perf_data.loc[fund_perf_data['avg_nav'].idxmax()]
                st.metric(
                    "Highest Avg NAV",
                    f"{best_performer['fund_name']}",
                    f"{best_performer['avg_nav']:.2f}"
                )
            
            with col2:
                most_stable = fund_perf_data.loc[fund_perf_data['nav_volatility'].idxmin()]
                st.metric(
                    "Most Stable (Low Volatility)",
                    f"{most_stable['fund_name']}",
                    f"{most_stable['nav_volatility']:.4f}"
                )
            
            with col3:
                largest_fund = fund_perf_data.loc[fund_perf_data['aum_current_estimate'].idxmax()]
                st.metric(
                    "Largest AUM",
                    f"{largest_fund['fund_name']}",
                    f"${largest_fund['aum_current_estimate']:,.0f}"
                )
    
    elif not selected_funds:
        st.warning("Please select at least one fund to analyze.")
    else:
        st.warning("3D analysis requires Plotly library for interactive visualizations.")

# --- AI-Powered Fund Performance Predictor ---
st.markdown("---")
st.header("🤖 AI-Powered Fund Performance Predictor")
st.info("Machine learning models for fund performance forecasting and predictive analytics.")

def prepare_prediction_features(nav_data, market_indicators=None):
    """Prepare features for ML prediction"""
    if nav_data is None or nav_data.empty:
        return None
    
    features_data = []
    
    for fund_id in nav_data['fund_id'].unique():
        fund_data = nav_data[nav_data['fund_id'] == fund_id].sort_values('nav_date').copy()
        
        if len(fund_data) < 5:  # Need minimum data points
            continue
        
        # Calculate technical indicators
        fund_data['returns'] = fund_data['nav_per_share'].pct_change()
        fund_data['volatility'] = fund_data['returns'].rolling(5).std()
        fund_data['sma_5'] = fund_data['nav_per_share'].rolling(5).mean()
        fund_data['rsi'] = calculate_rsi(fund_data['nav_per_share'], 5)
        
        # Create features for each row
        for i in range(5, len(fund_data)):
            row = fund_data.iloc[i]
            features = {
                'fund_id': fund_id,
                'nav_per_share': row['nav_per_share'],
                'returns_1d': fund_data['returns'].iloc[i],
                'returns_5d_avg': fund_data['returns'].iloc[i-4:i+1].mean(),
                'volatility': row['volatility'],
                'sma_ratio': row['nav_per_share'] / row['sma_5'] if row['sma_5'] != 0 else 1,
                'rsi': row['rsi'],
                'volume_trend': 1,  # Placeholder
                'target': fund_data['returns'].iloc[i+1] if i < len(fund_data)-1 else 0  # Next day return
            }
            features_data.append(features)
    
    return pd.DataFrame(features_data) if features_data else None

def calculate_rsi(prices, window):
    """Calculate Relative Strength Index"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def train_prediction_model(features_df):
    """Train machine learning model for NAV prediction"""
    if features_df is None or len(features_df) < 10:
        return None, None
    
    feature_cols = ['returns_1d', 'returns_5d_avg', 'volatility', 'sma_ratio', 'rsi', 'volume_trend']
    
    # Prepare data
    X = features_df[feature_cols].fillna(0)
    y = features_df['target'].fillna(0)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    model_stats = {
        'mae': mae,
        'r2': r2,
        'feature_importance': dict(zip(feature_cols, model.feature_importances_))
    }
    
    return model, model_stats

def predict_fund_performance(model, latest_features, days_ahead=5):
    """Predict fund performance for next N days"""
    if model is None or latest_features is None:
        return None
    
    predictions = []
    current_features = latest_features.copy()
    
    for day in range(days_ahead):
        # Make prediction
        pred_return = model.predict([current_features])[0]
        predictions.append({
            'day': day + 1,
            'predicted_return': pred_return,
            'confidence': min(abs(pred_return) * 100, 95)  # Simple confidence estimate
        })
        
        # Update features for next prediction (simplified)
        current_features[0] = pred_return  # Update returns_1d
        current_features[1] = (current_features[1] + pred_return) / 2  # Update avg returns
    
    return predictions

if SKLEARN_AVAILABLE and nav_data is not None:
    # Create tabs for AI predictions
    ai_tab_predictor, ai_tab_model, ai_tab_scenarios, ai_tab_insights = st.tabs([
        "🔮 Performance Predictor", "📊 Model Analytics", "🎭 Scenario Analysis", "💡 AI Insights"
    ])
    
    with ai_tab_predictor:
        st.subheader("🔮 Fund Performance Predictions")
        
        # Model training
        with st.spinner("🤖 Training AI models..."):
            features_df = prepare_prediction_features(nav_data)
            
            if features_df is not None and len(features_df) > 10:
                model, model_stats = train_prediction_model(features_df)
                
                if model is not None:
                    st.success(f"✅ AI model trained successfully! R² Score: {model_stats['r2']:.3f}")
                    
                    # Fund selection for prediction
                    available_funds = nav_data['fund_id'].unique()
                    selected_fund = st.selectbox("Select Fund for Prediction:", available_funds)
                    
                    # Prediction parameters
                    col1, col2 = st.columns(2)
                    with col1:
                        prediction_days = st.slider("Prediction Horizon (Days)", 1, 30, 7)
                    with col2:
                        confidence_level = st.selectbox("Confidence Level", [80, 90, 95], index=1)
                    
                    if st.button("🚀 Generate Predictions"):
                        # Get latest features for selected fund
                        fund_features = features_df[features_df['fund_id'] == selected_fund]
                        if not fund_features.empty:
                            latest_features = fund_features.iloc[-1][['returns_1d', 'returns_5d_avg', 'volatility', 'sma_ratio', 'rsi', 'volume_trend']].values
                            
                            # Generate predictions
                            predictions = predict_fund_performance(model, latest_features, prediction_days)
                            
                            if predictions:
                                # Display predictions
                                pred_df = pd.DataFrame(predictions)
                                
                                # Prediction chart
                                fig_pred = px.line(
                                    pred_df,
                                    x='day',
                                    y='predicted_return',
                                    title=f"{selected_fund} - {prediction_days} Day Performance Forecast",
                                    labels={'day': 'Days Ahead', 'predicted_return': 'Predicted Return (%)'}
                                )
                                fig_pred.update_traces(mode='lines+markers')
                                fig_pred.update_layout(height=400)
                                st.plotly_chart(fig_pred, use_container_width=True)
                                
                                # Prediction summary
                                total_predicted_return = sum([p['predicted_return'] for p in predictions])
                                avg_confidence = sum([p['confidence'] for p in predictions]) / len(predictions)
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Predicted Return", f"{total_predicted_return:.2%}")
                                with col2:
                                    st.metric("Average Confidence", f"{avg_confidence:.1f}%")
                                with col3:
                                    risk_level = "High" if abs(total_predicted_return) > 0.05 else "Medium" if abs(total_predicted_return) > 0.02 else "Low"
                                    st.metric("Risk Level", risk_level)
                                
                                # Detailed predictions table
                                st.write("**Detailed Predictions**")
                                pred_display = pred_df.copy()
                                pred_display['predicted_return'] = pred_display['predicted_return'].apply(lambda x: f"{x:.2%}")
                                pred_display['confidence'] = pred_display['confidence'].apply(lambda x: f"{x:.1f}%")
                                st.dataframe(pred_display, use_container_width=True)
                    
                else:
                    st.error("Failed to train prediction model. Please check data quality.")
            else:
                st.warning("Insufficient data for AI model training. Need at least 10 data points per fund.")
    
    with ai_tab_model:
        st.subheader("📊 Model Performance Analytics")
        
        if 'model_stats' in locals() and model_stats is not None:
            # Model performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Mean Absolute Error", f"{model_stats['mae']:.4f}")
                st.metric("R² Score", f"{model_stats['r2']:.3f}")
                
                # Model quality assessment
                if model_stats['r2'] > 0.7:
                    st.success("🟢 Excellent model performance")
                elif model_stats['r2'] > 0.5:
                    st.warning("🟡 Good model performance")
                else:
                    st.error("🔴 Model needs improvement")
            
            with col2:
                # Feature importance
                st.write("**Feature Importance**")
                importance_df = pd.DataFrame([
                    {'Feature': k, 'Importance': v}
                    for k, v in model_stats['feature_importance'].items()
                ]).sort_values('Importance', ascending=True)
                
                fig_importance = px.bar(
                    importance_df,
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Feature Importance in Predictions"
                )
                st.plotly_chart(fig_importance, use_container_width=True)
        else:
            st.info("Train a model in the Predictor tab to see analytics")
    
    with ai_tab_scenarios:
        st.subheader("🎭 Scenario Analysis")
        
        if 'model' in locals() and model is not None:
            st.write("**What-If Scenario Testing**")
            
            # Scenario parameters
            col1, col2 = st.columns(2)
            
            with col1:
                scenario_volatility = st.slider("Market Volatility", 0.01, 0.10, 0.02, 0.01, format="%.2f")
                scenario_trend = st.slider("Market Trend", -0.05, 0.05, 0.0, 0.01, format="%.2f")
            
            with col2:
                scenario_rsi = st.slider("RSI Level", 20, 80, 50, 5)
                scenario_momentum = st.slider("Momentum Factor", -0.03, 0.03, 0.0, 0.01, format="%.2f")
            
            if st.button("🎯 Run Scenario Analysis"):
                # Create scenario features
                scenario_features = [scenario_trend, scenario_momentum, scenario_volatility, 1.0, scenario_rsi, 1.0]
                
                # Generate scenario predictions for all funds
                scenario_results = []
                for fund_id in nav_data['fund_id'].unique():
                    pred_return = model.predict([scenario_features])[0]
                    scenario_results.append({
                        'Fund': fund_id,
                        'Predicted Return': pred_return,
                        'Risk Level': 'High' if abs(pred_return) > 0.03 else 'Medium' if abs(pred_return) > 0.01 else 'Low'
                    })
                
                scenario_df = pd.DataFrame(scenario_results)
                
                # Scenario results chart
                fig_scenario = px.bar(
                    scenario_df,
                    x='Fund',
                    y='Predicted Return',
                    color='Risk Level',
                    title="Scenario Analysis Results",
                    color_discrete_map={'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                )
                st.plotly_chart(fig_scenario, use_container_width=True)
                
                # Scenario summary
                st.write("**Scenario Summary**")
                st.dataframe(scenario_df, use_container_width=True)
        else:
            st.info("Train a model in the Predictor tab to run scenario analysis")
    
    with ai_tab_insights:
        st.subheader("💡 AI-Generated Insights")
        
        if nav_data is not None:
            # Generate insights based on data analysis
            insights = []
            
            # Fund performance insights
            fund_performance = nav_data.groupby('fund_id').agg({
                'nav_per_share': ['mean', 'std', 'min', 'max']
            }).round(4)
            fund_performance.columns = ['Avg NAV', 'Volatility', 'Min NAV', 'Max NAV']
            
            # Best performer
            best_performer = fund_performance.loc[fund_performance['Avg NAV'].idxmax()]
            insights.append(f"🏆 **Best Performer**: {best_performer.name} with average NAV of {best_performer['Avg NAV']:.4f}")
            
            # Most stable
            most_stable = fund_performance.loc[fund_performance['Volatility'].idxmin()]
            insights.append(f"🛡️ **Most Stable**: {most_stable.name} with volatility of {most_stable['Volatility']:.4f}")
            
            # Risk assessment
            high_risk_funds = fund_performance[fund_performance['Volatility'] > fund_performance['Volatility'].median()].index.tolist()
            if high_risk_funds:
                insights.append(f"⚠️ **Higher Risk Funds**: {', '.join(high_risk_funds)} show above-median volatility")
            
            # Correlation insights
            if len(nav_data['fund_id'].unique()) > 2:
                nav_pivot = nav_data.pivot(index='nav_date', columns='fund_id', values='nav_per_share')
                correlation_matrix = nav_pivot.corr()
                avg_correlation = correlation_matrix.values[correlation_matrix.values != 1].mean()
                insights.append(f"🔗 **Fund Correlation**: Average correlation of {avg_correlation:.2f} indicates {'high' if avg_correlation > 0.7 else 'moderate' if avg_correlation > 0.3 else 'low'} interconnection")
            
            # Display insights
            for insight in insights:
                st.write(insight)
            
            # Performance summary table
            st.write("**Fund Performance Summary**")
            st.dataframe(fund_performance, use_container_width=True)
            
        else:
            st.info("Load fund data to see AI-generated insights")

else:
    if not SKLEARN_AVAILABLE:
        st.warning("AI predictions require scikit-learn library. Please install: pip install scikit-learn")
    else:
        st.info("Load operational fund data to enable AI-powered predictions")

# Asset data table for reference
st.markdown("---")
st.subheader("📋 Synthetic Asset Data Reference")
st.info("This is the original synthetic periodic table data used for the main visualizations above.")

# Enhanced data table with styling
st.dataframe(
    display_df.style.background_gradient(subset=['Risk', 'Liquidity', 'OpCost', 'OpRisk'], cmap='RdYlGn_r'),
    use_container_width=True,
    height=400
)

# Category breakdown for filtered results
if selected_category != 'All' or search_term:
    st.markdown("---")
    filtered_df = df.copy()
    if selected_category != 'All':
        filtered_df = filtered_df[filtered_df['Category'] == selected_category]
    if search_term:
        search_mask = (
            filtered_df['Symbol'].str.contains(search_term, case=False, na=False) |
            filtered_df['Name'].str.contains(search_term, case=False, na=False)
        )
        filtered_df = filtered_df[search_mask]
    
    if len(filtered_df) > 0:
        st.subheader(f"📋 Filtered Results ({len(filtered_df)} assets)")
        
        # Create expandable details for filtered assets
        for _, asset in filtered_df.iterrows():
            with st.expander(f"📊 {asset['Symbol']} - {asset['Name']} ({asset['Category']})"):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("🎲 Risk", f"{asset['Risk']}/10")
                with col2:
                    st.metric("💧 Liquidity", f"{asset['Liquidity']}/10")
                with col3:
                    st.metric("💰 Op Cost", f"{asset['OpCost']}/10")
                with col4:
                    st.metric("⚠️ Op Risk", f"{asset['OpRisk']}/10")
                
                st.write(f"**Grid Position:** Row {asset['GridRow']}, Column {asset['GridCol']}")
                
                # Add market data if available
                if asset['Symbol'] in market_data:
                    data = market_data[asset['Symbol']]
                    change_color = "🟢" if data['change'] >= 0 else "🔴"
                    st.write(f"**Current Price:** ${data['price']:.2f} {change_color} {data['change']:+.1f}%")
    else:
        st.warning("No assets match your current filter criteria.")

# --- Operational Workstreams Periodic Table ---
st.markdown("---")
st.header("⚙️ Operational Workstreams - Fund Administration Periodic Table")

# Add workstream controls
col1, col2 = st.columns(2)
with col1:
    workstream_metric = st.selectbox(
        "Color Code Workstreams By:",
        options=['complexity', 'operational_risk', 'automation', 'client_impact'],
        format_func=lambda x: {
            'complexity': 'Process Complexity',
            'operational_risk': 'Operational Risk',
            'automation': 'Automation Level',
            'client_impact': 'Client Impact'
        }[x]
    )

with col2:
    show_projects = st.checkbox("Show Capital Projects", value=True)

# Helper function for workstream colors
@st.cache_data
def get_workstream_color(value, metric):
    val_norm = (value - 1) / 9.0
    if metric == 'automation':
        # Green scale for automation: low automation is reddish, high is greenish
        red = int(255 * (1 - val_norm))
        green = int(255 * val_norm)
        blue = 40
    else:
        # Red scale for complexity/risk/impact: low is greenish, high is reddish
        red = int(255 * val_norm)
        green = int(255 * (1 - val_norm))
        blue = 40
    return f"rgb({red}, {green}, {blue})"

# Display workstreams in organized layout
st.subheader("🔧 Fund Administration Value Streams")

# Group workstreams by row
max_ws_row = max(ws['row'] for ws in workstreams_data.values())
max_ws_col = max(ws['col'] for ws in workstreams_data.values())

for row in range(1, max_ws_row + 1):
    row_workstreams = {name: data for name, data in workstreams_data.items() if data['row'] == row}
    
    if row_workstreams:
        # Create columns for this row
        cols = st.columns(max_ws_col)
        
        for name, workstream in row_workstreams.items():
            color = get_workstream_color(workstream[workstream_metric], workstream_metric)
            col_idx = workstream['col'] - 1
            
            with cols[col_idx]:
                # Check if there are capital projects for this workstream (use session state)
                current_capital_projects = st.session_state.get('capital_projects', capital_projects)
                related_projects = [proj for proj, details in current_capital_projects.items() 
                                  if details['value_stream'] == name or details['value_stream'] in name]
                
                project_indicator = "$" if related_projects and show_projects else ""
                
                st.markdown(f"""
                <div style="
                    background-color: {color}; 
                    padding: 12px; 
                    border-radius: 8px; 
                    text-align: center;
                    border: 2px solid #333;
                    margin: 3px;
                    height: 120px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                ">
                    <strong style="font-size: 1.1em; margin-bottom: 4px;">{name} {project_indicator}</strong><br/>
                    <small style="font-size: 0.6em; line-height: 1.1;">{workstream_metric.replace('_', ' ').title()}: {workstream[workstream_metric]}/10</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Expandable details
                with st.expander(f"📊 {name} Details"):
                    st.write(f"**Processes ({len(workstream['processes'])}):**")
                    for process in workstream['processes']:
                        st.write(f"• {process}")
                    
                    st.write(f"**Applications ({len(workstream['applications'])}):**")
                    for app in workstream['applications']:
                        st.write(f"• {app}")
                    
                    # Metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Complexity", f"{workstream['complexity']}/10")
                        st.metric("Automation", f"{workstream['automation']}/10")
                    with col2:
                        st.metric("Op Risk", f"{workstream['operational_risk']}/10")
                        st.metric("Client Impact", f"{workstream['client_impact']}/10")
                    
                    # Show related projects
                    if related_projects:
                        st.write("**Related Capital Projects:**")
                        for proj in related_projects:
                            proj_details = current_capital_projects[proj]
                            st.write(f"• {proj} ({proj_details['classification']}) - {proj_details['budget']} Budget")
                    
                    # Show identified gaps
                    if name in identified_gaps:
                        st.write("**Identified Gaps:**")
                        for gap in identified_gaps[name]:
                            st.write(f"• {gap}")

# --- Editable Workstreams Management ---
st.markdown("---")
st.subheader("✏️ Manage Workstreams - Add/Edit/Delete")

# Initialize workstreams in session state
if 'workstreams_data' not in st.session_state:
    st.session_state.workstreams_data = workstreams_data.copy()

# Workstream management interface
col1, col2 = st.columns([2, 1])

with col1:
    st.write("**Add New Workstream**")
    
    # Form for adding new workstream
    with st.form("add_workstream"):
        new_name = st.text_input("Workstream Name")
        new_complexity = st.slider("Complexity (1-10)", 1, 10, 5)
        new_automation = st.slider("Automation Level (1-10)", 1, 10, 5)
        new_operational_risk = st.slider("Operational Risk (1-10)", 1, 10, 5)
        new_client_impact = st.slider("Client Impact (1-10)", 1, 10, 5)
        
        col_proc, col_app = st.columns(2)
        with col_proc:
            new_processes = st.text_area("Processes (one per line)", height=100)
        with col_app:
            new_applications = st.text_area("Applications (one per line)", height=100)
        
        submit_new = st.form_submit_button("➕ Add Workstream")
        
        if submit_new and new_name:
            if new_name not in st.session_state.workstreams_data:
                st.session_state.workstreams_data[new_name] = {
                    "processes": [p.strip() for p in new_processes.split('\n') if p.strip()],
                    "applications": [a.strip() for a in new_applications.split('\n') if a.strip()],
                    "complexity": new_complexity,
                    "operational_risk": new_operational_risk,
                    "automation": new_automation,
                    "client_impact": new_client_impact
                }
                st.success(f"✅ Added new workstream: {new_name}")
                st.rerun()
            else:
                st.warning("⚠️ Workstream name already exists!")

with col2:
    st.write("**Delete Workstream**")
    
    # Select workstream to delete
    workstream_to_delete = st.selectbox(
        "Select workstream to delete:",
        options=list(st.session_state.workstreams_data.keys()),
        key="delete_workstream_select"
    )
    
    if st.button("🗑️ Delete Workstream", type="secondary"):
        if workstream_to_delete:
            del st.session_state.workstreams_data[workstream_to_delete]
            st.success(f"🗑️ Deleted workstream: {workstream_to_delete}")
            st.rerun()

# Edit existing workstream
st.write("**Edit Existing Workstream**")
edit_workstream = st.selectbox(
    "Select workstream to edit:",
    options=list(st.session_state.workstreams_data.keys()),
    key="edit_workstream_select"
)

if edit_workstream:
    current_data = st.session_state.workstreams_data[edit_workstream]
    
    with st.form(f"edit_{edit_workstream}"):
        st.write(f"**Editing: {edit_workstream}**")
        
        col1, col2 = st.columns(2)
        with col1:
            edit_complexity = st.slider("Complexity", 1, 10, current_data["complexity"], key=f"edit_complex_{edit_workstream}")
            edit_automation = st.slider("Automation", 1, 10, current_data["automation"], key=f"edit_auto_{edit_workstream}")
        with col2:
            edit_operational_risk = st.slider("Operational Risk", 1, 10, current_data["operational_risk"], key=f"edit_risk_{edit_workstream}")
            edit_client_impact = st.slider("Client Impact", 1, 10, current_data["client_impact"], key=f"edit_impact_{edit_workstream}")
        
        col_proc, col_app = st.columns(2)
        with col_proc:
            edit_processes = st.text_area(
                "Processes", 
                value='\n'.join(current_data["processes"]), 
                height=100,
                key=f"edit_proc_{edit_workstream}"
            )
        with col_app:
            edit_applications = st.text_area(
                "Applications", 
                value='\n'.join(current_data["applications"]), 
                height=100,
                key=f"edit_app_{edit_workstream}"
            )
        
        submit_edit = st.form_submit_button("💾 Update Workstream")
        
        if submit_edit:
            st.session_state.workstreams_data[edit_workstream] = {
                "processes": [p.strip() for p in edit_processes.split('\n') if p.strip()],
                "applications": [a.strip() for a in edit_applications.split('\n') if a.strip()],
                "complexity": edit_complexity,
                "operational_risk": edit_operational_risk,
                "automation": edit_automation,
                "client_impact": edit_client_impact
            }
            st.success(f"💾 Updated workstream: {edit_workstream}")
            st.rerun()

# Update the workstreams_data variable to use session state
workstreams_data = st.session_state.workstreams_data

# --- Editable Capital Portfolio ---
st.markdown("---")
st.subheader("$ Editable Capital Portfolio - USD 26M (2025)")

# Initialize capital projects in session state
if 'capital_projects' not in st.session_state:
    st.session_state.capital_projects = capital_projects.copy()

# Project management interface
col1, col2 = st.columns([2, 1])

with col1:
    st.write("**Current Capital Projects:**")
    
    # Display editable projects
    projects_to_remove = []
    updated_projects = {}
    
    for project_name, details in st.session_state.capital_projects.items():
        with st.expander(f"📝 Edit: {project_name}"):
            col_edit1, col_edit2, col_remove = st.columns([2, 2, 1])
            
            with col_edit1:
                new_classification = st.selectbox(
                    "Classification:",
                    options=['Rock', 'Sand', 'Pebble'],
                    index=['Rock', 'Sand', 'Pebble'].index(details['classification']),
                    key=f"class_{project_name}"
                )
                
                new_budget = st.selectbox(
                    "Budget Level:",
                    options=['High', 'Medium', 'Low'],
                    index=['High', 'Medium', 'Low'].index(details['budget']),
                    key=f"budget_{project_name}"
                )
            
            with col_edit2:
                # Get unique value streams from workstreams_data
                value_stream_options = list(workstreams_data.keys()) + ['Multiple', 'FA Workflow', 'ETF Growth']
                current_vs = details['value_stream']
                if current_vs not in value_stream_options:
                    value_stream_options.append(current_vs)
                
                new_value_stream = st.selectbox(
                    "Value Stream:",
                    options=value_stream_options,
                    index=value_stream_options.index(current_vs),
                    key=f"vs_{project_name}"
                )
            
            with col_remove:
                st.write("")  # Spacer
                st.write("")  # Spacer
                if st.button("🗑️ Remove", key=f"remove_proj_{project_name}"):
                    projects_to_remove.append(project_name)
            
            # Update project details
            updated_projects[project_name] = {
                'classification': new_classification,
                'value_stream': new_value_stream,
                'budget': new_budget
            }
    
    # Remove projects marked for removal
    for proj in projects_to_remove:
        if proj in st.session_state.capital_projects:
            del st.session_state.capital_projects[proj]
            st.success(f"Removed project: {proj}")
            st.rerun()
    
    # Update all projects
    st.session_state.capital_projects.update(updated_projects)

with col2:
    st.write("**Add New Project:**")
    
    new_project_name = st.text_input("Project Name:", key="new_proj_name")
    new_project_class = st.selectbox("Classification:", options=['Rock', 'Sand', 'Pebble'], key="new_proj_class")
    new_project_vs = st.selectbox("Value Stream:", options=list(workstreams_data.keys()) + ['Multiple', 'FA Workflow', 'ETF Growth'], key="new_proj_vs")
    new_project_budget = st.selectbox("Budget Level:", options=['High', 'Medium', 'Low'], key="new_proj_budget")
    
    if st.button("➕ Add Project") and new_project_name.strip():
        if new_project_name not in st.session_state.capital_projects:
            st.session_state.capital_projects[new_project_name] = {
                'classification': new_project_class,
                'value_stream': new_project_vs,
                'budget': new_project_budget
            }
            st.success(f"Added project: {new_project_name}")
            st.rerun()
        else:
            st.warning("Project name already exists!")
    
    # Project management actions
    st.markdown("---")
    st.write("**Portfolio Actions:**")
    
    col_reset, col_export = st.columns(2)
    with col_reset:
        if st.button("🔄 Reset to Original", key="reset_capital_portfolio"):
            st.session_state.capital_projects = capital_projects.copy()
            st.success("Portfolio reset!")
            st.rerun()
    
    with col_export:
        # Export current portfolio
        current_projects_df = pd.DataFrame([
            {
                'Project': proj,
                'Classification': details['classification'],
                'Value Stream': details['value_stream'],
                'Budget': details['budget']
            }
            for proj, details in st.session_state.capital_projects.items()
        ])
        
        portfolio_csv = current_projects_df.to_csv(index=False)
        st.download_button(
            label="📁 Export Portfolio",
            data=portfolio_csv,
            file_name="capital_portfolio.csv",
            mime="text/csv"
        )

# Analysis of current portfolio
st.markdown("---")
st.write("**Portfolio Analysis:**")

col1, col2 = st.columns(2)

with col1:
    st.write("**Projects by Classification**")
    classification_counts = {}
    for proj, details in st.session_state.capital_projects.items():
        cls = details['classification']
        classification_counts[cls] = classification_counts.get(cls, 0) + 1
    
    if classification_counts:
        st.bar_chart(pd.Series(classification_counts))
    else:
        st.write("No projects to display")

with col2:
    st.write("**Projects by Value Stream**")
    valuestream_counts = {}
    for proj, details in st.session_state.capital_projects.items():
        vs = details['value_stream']
        valuestream_counts[vs] = valuestream_counts.get(vs, 0) + 1
    
    # Display as metrics
    for vs, count in sorted(valuestream_counts.items()):
        st.metric(vs, f"{count} project{'s' if count > 1 else ''}")

# Current portfolio summary
st.write("**Current Portfolio Summary:**")
current_projects_df = pd.DataFrame([
    {
        'Project': proj,
        'Classification': details['classification'],
        'Value Stream': details['value_stream'],
        'Budget': details['budget']
    }
    for proj, details in st.session_state.capital_projects.items()
])

if not current_projects_df.empty:
    st.dataframe(current_projects_df, use_container_width=True)
    
    # Portfolio statistics
    total_projects = len(current_projects_df)
    rock_projects = len(current_projects_df[current_projects_df['Classification'] == 'Rock'])
    sand_projects = len(current_projects_df[current_projects_df['Classification'] == 'Sand'])
    high_budget_projects = len(current_projects_df[current_projects_df['Budget'] == 'High'])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", total_projects)
    with col2:
        st.metric("Rock Projects", rock_projects)
    with col3:
        st.metric("Sand Projects", sand_projects)
    with col4:
        st.metric("High Budget Projects", high_budget_projects)
else:
    st.info("No projects in portfolio. Add some projects to see analysis.")

# --- Advanced Workstream Analytics ---
if PLOTLY_AVAILABLE or SEABORN_AVAILABLE:
    st.markdown("---")
    st.subheader("🔬 Advanced Workstream Analytics")

    # Create workstream analysis dataframe
    workstream_df = pd.DataFrame([
        {
            'Workstream': name,
            'Complexity': data['complexity'],
            'Operational_Risk': data['operational_risk'],
            'Automation': data['automation'],
            'Client_Impact': data['client_impact'],
            'Process_Count': len(data['processes']),
            'Application_Count': len(data['applications']),
            'Gap_Count': len(identified_gaps.get(name, [])),
            'Project_Count': len([p for p, details in st.session_state.get('capital_projects', capital_projects).items() 
                                 if details['value_stream'] == name])
        }
        for name, data in workstreams_data.items()
    ])

    col1, col2 = st.columns(2)

    with col1:
        if PLOTLY_AVAILABLE:
            st.write("**Workstream Complexity vs Automation Analysis**")
            
            # Create bubble chart with Plotly
            fig_bubble = px.scatter(
                workstream_df,
                x='Complexity',
                y='Automation',
                size='Process_Count',
                color='Gap_Count',
                hover_name='Workstream',
                hover_data={'Operational_Risk': True, 'Client_Impact': True, 'Project_Count': True},
                title="Workstream Complexity vs Automation Level",
                labels={
                    'Complexity': 'Process Complexity (1-10)',
                    'Automation': 'Automation Level (1-10)',
                    'Gap_Count': 'Number of Gaps'
                },
                color_continuous_scale='Reds'
            )
            
            # Add quadrant lines
            fig_bubble.add_hline(y=6.5, line_dash="dash", line_color="gray", opacity=0.5)
            fig_bubble.add_vline(x=6.5, line_dash="dash", line_color="gray", opacity=0.5)
            
            # Add quadrant annotations
            fig_bubble.add_annotation(x=3, y=8.5, text="Simple & Automated<br>(Low Complexity, High Automation)", 
                                     showarrow=False, font=dict(size=9), bgcolor="lightgreen", opacity=0.7)
            fig_bubble.add_annotation(x=8.5, y=8.5, text="Complex & Automated<br>(High Complexity, High Automation)", 
                                     showarrow=False, font=dict(size=9), bgcolor="yellow", opacity=0.7)
            fig_bubble.add_annotation(x=3, y=3, text="Simple & Manual<br>(Low Complexity, Low Automation)", 
                                     showarrow=False, font=dict(size=9), bgcolor="lightblue", opacity=0.7)
            fig_bubble.add_annotation(x=8.5, y=3, text="Complex & Manual<br>(High Complexity, Low Automation)", 
                                     showarrow=False, font=dict(size=9), bgcolor="lightcoral", opacity=0.7)
            
            fig_bubble.update_layout(height=500)
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.write("**Workstream Analysis**")
            st.bar_chart(workstream_df.set_index('Workstream')[['Complexity', 'Automation']])

    with col2:
        if SEABORN_AVAILABLE:
            st.write("**Workstream Risk-Impact Matrix**")
            
            # Create risk-impact heatmap
            fig_risk, ax = plt.subplots(figsize=(10, 8))
            
            # Create pivot table for heatmap
            risk_impact_data = workstream_df.pivot_table(
                values='Gap_Count', 
                index='Operational_Risk', 
                columns='Client_Impact', 
                aggfunc='mean',
                fill_value=0
            )
            
            sns.heatmap(risk_impact_data, annot=True, cmap='YlOrRd', ax=ax, 
                        cbar_kws={"shrink": .8}, fmt='.1f')
            ax.set_title('Risk-Impact Heatmap (Average Gap Count)')
            ax.set_xlabel('Client Impact Level')
            ax.set_ylabel('Operational Risk Level')
            
            st.pyplot(fig_risk, use_container_width=True)
        else:
            st.write("**Workstream Metrics**")
            st.bar_chart(workstream_df.set_index('Workstream')[['Operational_Risk', 'Client_Impact']])

    # Workstream Performance Radar Chart  
    if PLOTLY_AVAILABLE:
        st.write("**Workstream Performance Comparison (Radar Chart)**")

        # Select workstreams to compare
        selected_workstreams = st.multiselect(
            "Select Workstreams to Compare:",
            options=workstream_df['Workstream'].tolist(),
            default=workstream_df['Workstream'].tolist()[:3],
            key="workstream_comparison"
        )

        if selected_workstreams:
            # Create radar chart data
            metrics = ['Complexity', 'Operational_Risk', 'Automation', 'Client_Impact']
            
            fig_radar = go.Figure()
            
            for workstream in selected_workstreams:
                ws_data = workstream_df[workstream_df['Workstream'] == workstream].iloc[0]
                values = [ws_data[metric] for metric in metrics]
                values.append(values[0])  # Close the radar chart
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=values,
                    theta=metrics + [metrics[0]],
                    fill='toself',
                    name=workstream,
                    line=dict(width=2),
                    opacity=0.7
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 10]
                    )),
                showlegend=True,
                title="Workstream Performance Comparison",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)

    # Capital Portfolio vs Workstream Analysis
    st.write("**Capital Investment vs Gap Analysis**")

    col1, col2 = st.columns(2)

    with col1:
        if PLOTLY_AVAILABLE:
            # Treemap of gaps by workstream
            gap_data = []
            for workstream, gaps in identified_gaps.items():
                if len(gaps) > 0:
                    gap_data.append({
                        'Workstream': workstream,
                        'Gap_Count': len(gaps),
                        'Complexity': workstreams_data[workstream]['complexity']
                    })
            
            if gap_data:
                gap_df = pd.DataFrame(gap_data)
                
                fig_gap_tree = px.treemap(
                    gap_df,
                    path=['Workstream'],
                    values='Gap_Count',
                    color='Complexity',
                    color_continuous_scale='Reds',
                    title="Identified Gaps by Workstream"
                )
                
                fig_gap_tree.update_layout(height=400)
                st.plotly_chart(fig_gap_tree, use_container_width=True)
        else:
            st.write("**Gap Analysis**")
            gap_summary = pd.Series({ws: len(gaps) for ws, gaps in identified_gaps.items()})
            st.bar_chart(gap_summary)

    with col2:
        if PLOTLY_AVAILABLE:
            # Investment allocation vs complexity
            investment_data = []
            for proj, details in st.session_state.get('capital_projects', capital_projects).items():
                vs = details['value_stream']
                if vs in workstreams_data:
                    investment_data.append({
                        'Project': proj,
                        'Workstream': vs,
                        'Budget': details['budget'],
                        'Classification': details['classification'],
                        'Complexity': workstreams_data[vs]['complexity'],
                        'Gap_Count': len(identified_gaps.get(vs, []))
                    })
            
            if investment_data:
                invest_df = pd.DataFrame(investment_data)
                
                # Map budget levels to numeric values
                budget_map = {'High': 3, 'Medium': 2, 'Low': 1}
                invest_df['Budget_Numeric'] = invest_df['Budget'].map(budget_map)
                
                fig_invest = px.scatter(
                    invest_df,
                    x='Complexity',
                    y='Gap_Count',
                    size='Budget_Numeric',
                    color='Classification',
                    hover_name='Project',
                    hover_data={'Workstream': True, 'Budget': True},
                    title="Capital Investment vs Workstream Complexity & Gaps",
                    labels={
                        'Complexity': 'Workstream Complexity',
                        'Gap_Count': 'Number of Identified Gaps'
                    }
                )
                
                fig_invest.update_layout(height=400)
                st.plotly_chart(fig_invest, use_container_width=True)
        else:
            st.write("**Project Analysis**")
            project_summary = pd.Series({
                'High Budget': len([p for p, d in st.session_state.get('capital_projects', capital_projects).items() if d['budget'] == 'High']),
                'Medium Budget': len([p for p, d in st.session_state.get('capital_projects', capital_projects).items() if d['budget'] == 'Medium']),
                'Low Budget': len([p for p, d in st.session_state.get('capital_projects', capital_projects).items() if d['budget'] == 'Low'])
            })
            st.bar_chart(project_summary)

# --- Client Change Requests Widget ---
st.markdown("---")
st.subheader("📋 Client Change Request Distribution")

# Initialize client change data in session state
if 'client_changes' not in st.session_state:
    st.session_state.client_changes = client_change_data.copy()

st.write("**Edit the distribution percentages for client change requests:**")

# Create editable widget for client changes
col1, col2 = st.columns([2, 1])

with col1:
    total_percentage = 0
    updated_changes = {}
    
    for change_type, current_value in st.session_state.client_changes.items():
        new_value = st.number_input(
            f"{change_type} (%)",
            min_value=0.0,
            max_value=100.0,
            value=current_value,
            step=0.1,
            key=f"change_{change_type.replace(' ', '_')}"
        )
        updated_changes[change_type] = new_value
        total_percentage += new_value
    
    # Update session state
    st.session_state.client_changes = updated_changes
    
    # Validation
    if abs(total_percentage - 100.0) > 0.1:
        st.warning(f"⚠️ Total percentage: {total_percentage:.1f}%. Consider adjusting to 100%.")
    else:
        st.success("✅ Total percentage equals 100%!")

with col2:
    st.write("**Current Distribution:**")
    for change_type, value in st.session_state.client_changes.items():
        st.metric(change_type, f"{value:.1f}%")
    
    # Action buttons
    if st.button("🔄 Reset to Original", key="reset_client_changes"):
        st.session_state.client_changes = client_change_data.copy()
        st.rerun()
    
    if st.button("⚖️ Redistribute Equally", key="redistribute_client_changes"):
        equal_value = 100.0 / len(st.session_state.client_changes)
        for change_type in st.session_state.client_changes:
            st.session_state.client_changes[change_type] = equal_value
        st.rerun()

# Visualization
st.write("**Client Change Distribution Visualization:**")
changes_df = pd.DataFrame([
    {'Change Type': change_type, 'Percentage': value}
    for change_type, value in st.session_state.client_changes.items()
])

col1, col2 = st.columns(2)
with col1:
    st.bar_chart(changes_df.set_index('Change Type')['Percentage'])

with col2:
    # Create a simple pie chart using native streamlit
    st.write("**Top Change Categories:**")
    sorted_changes = sorted(st.session_state.client_changes.items(), key=lambda x: x[1], reverse=True)
    for i, (change_type, value) in enumerate(sorted_changes[:3]):
        st.metric(f"{i+1}. {change_type}", f"{value:.1f}%")

# Export client changes
st.write("**Export Client Change Data:**")
changes_csv = changes_df.to_csv(index=False)
st.download_button(
    label="📁 Export Client Changes CSV",
    data=changes_csv,
    file_name="client_change_distribution.csv",
    mime="text/csv"
)

# --- Gap Analysis Summary ---
st.markdown("---")
st.subheader("🔍 Identified Gaps Summary")

total_gaps = sum(len(gaps) for gaps in identified_gaps.values())
st.metric("Total Identified Gaps", total_gaps)

# Gaps by workstream
gaps_by_workstream = {
    workstream: len(gaps) for workstream, gaps in identified_gaps.items()
}

col1, col2 = st.columns(2)
with col1:
    st.write("**Gaps by Workstream:**")
    st.bar_chart(pd.Series(gaps_by_workstream))

with col2:
    st.write("**Priority Workstreams (Most Gaps):**")
    sorted_gaps = sorted(gaps_by_workstream.items(), key=lambda x: x[1], reverse=True)
    for workstream, gap_count in sorted_gaps[:5]:
        st.metric(workstream, f"{gap_count} gaps")

# Detailed gaps
st.write("**All Identified Gaps by Workstream:**")
for workstream, gaps in identified_gaps.items():
    with st.expander(f"{workstream} ({len(gaps)} gaps)"):
        for i, gap in enumerate(gaps, 1):
            st.write(f"{i}. {gap}")

# --- Asset Comparison Section ---
st.markdown("---")
st.header("🔍 Asset Comparison")

col1, col2 = st.columns(2)

with col1:
    asset1 = st.selectbox(
        "Select First Asset:",
        options=df['Symbol'].tolist(),
        format_func=lambda x: f"{x} - {df[df['Symbol']==x]['Name'].iloc[0]}"
    )

with col2:
    asset2 = st.selectbox(
        "Select Second Asset:",
        options=df['Symbol'].tolist(),
        format_func=lambda x: f"{x} - {df[df['Symbol']==x]['Name'].iloc[0]}",
        index=1 if len(df) > 1 else 0
    )

if asset1 and asset2:
    asset1_data = df[df['Symbol'] == asset1].iloc[0]
    asset2_data = df[df['Symbol'] == asset2].iloc[0]
    
    # Comparison data
    metrics = ['Risk', 'Liquidity', 'OpCost', 'OpRisk']
    asset1_values = [asset1_data[metric] for metric in metrics]
    asset2_values = [asset2_data[metric] for metric in metrics]
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Metric': metrics,
        f'{asset1} ({asset1_data["Name"]})': asset1_values,
        f'{asset2} ({asset2_data["Name"]})': asset2_values,
        'Difference': [asset2_values[i] - asset1_values[i] for i in range(len(metrics))]
    })
    
    # Format the difference column with colors
    def highlight_diff(val):
        if val > 0:
            return 'color: red'
        elif val < 0:
            return 'color: green'
        return ''
    
    styled_df = comparison_df.style.map(highlight_diff, subset=['Difference'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Key insights
    st.subheader("📊 Key Insights")
    insights = []
    
    if asset1_data['Risk'] > asset2_data['Risk']:
        insights.append(f"• **{asset1}** has higher market risk than **{asset2}** (+{asset1_data['Risk'] - asset2_data['Risk']} points)")
    elif asset2_data['Risk'] > asset1_data['Risk']:
        insights.append(f"• **{asset2}** has higher market risk than **{asset1}** (+{asset2_data['Risk'] - asset1_data['Risk']} points)")
    
    if asset1_data['Liquidity'] > asset2_data['Liquidity']:
        insights.append(f"• **{asset1}** is more liquid than **{asset2}** (+{asset1_data['Liquidity'] - asset2_data['Liquidity']} points)")
    elif asset2_data['Liquidity'] > asset1_data['Liquidity']:
        insights.append(f"• **{asset2}** is more liquid than **{asset1}** (+{asset2_data['Liquidity'] - asset1_data['Liquidity']} points)")
    
    for insight in insights:
        st.markdown(insight)

# --- Portfolio Builder Section ---
st.markdown("---")
st.header("🏗️ Portfolio Builder")

# Initialize portfolio in session state
if 'portfolio' not in st.session_state:
    st.session_state.portfolio = {}

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Build Your Portfolio")
    
    # Asset selector
    selected_asset = st.selectbox(
        "Add Asset to Portfolio:",
        options=df['Symbol'].tolist(),
        format_func=lambda x: f"{x} - {df[df['Symbol']==x]['Name'].iloc[0]} ({df[df['Symbol']==x]['Category'].iloc[0]})"
    )
    
    col_add, col_template = st.columns([1, 1])
    with col_add:
        if st.button("➕ Add to Portfolio"):
            if selected_asset not in st.session_state.portfolio:
                st.session_state.portfolio[selected_asset] = 10.0  # Default 10% weight
                st.success(f"Added {selected_asset} to portfolio!")
            else:
                st.warning(f"{selected_asset} is already in portfolio!")
    
    with col_template:
        template_option = st.selectbox(
            "Or try a template:",
            options=["Select Template", "Conservative", "Balanced", "Aggressive", "Liquid Assets Only"]
        )
        
        if st.button("🎯 Apply Template") and template_option != "Select Template":
            st.session_state.portfolio = {}  # Clear existing
            
            if template_option == "Conservative":
                # Low risk, high liquidity
                st.session_state.portfolio = {
                    'USD': 30.0, 'UST': 25.0, 'EUR': 20.0, 'Bund': 15.0, 'IGC': 10.0
                }
            elif template_option == "Balanced":
                # Mixed risk/liquidity profile
                st.session_state.portfolio = {
                    'UST': 20.0, 'IGC': 20.0, 'ETF': 25.0, 'HYC': 15.0, 'EMD': 10.0, 'Au': 10.0
                }
            elif template_option == "Aggressive":
                # Higher risk, potentially higher returns
                st.session_state.portfolio = {
                    'ETF': 20.0, 'HYC': 20.0, 'PE': 15.0, 'VC': 15.0, 'HF': 15.0, 'Oil': 15.0
                }
            elif template_option == "Liquid Assets Only":
                # High liquidity focus
                st.session_state.portfolio = {
                    'USD': 25.0, 'EUR': 20.0, 'UST': 20.0, 'ETF': 20.0, 'Fut': 15.0
                }
            
            st.success(f"Applied {template_option} template!")
            st.rerun()
    
    # Portfolio composition
    if st.session_state.portfolio:
        st.subheader("📈 Current Portfolio")
        
        # Create weight adjusters
        portfolio_data = []
        total_weight = 0
        
        for symbol in list(st.session_state.portfolio.keys()):
            asset_info = df[df['Symbol'] == symbol].iloc[0]
            
            col_symbol, col_weight, col_remove = st.columns([2, 2, 1])
            
            with col_symbol:
                st.write(f"**{symbol}** - {asset_info['Name']}")
                st.write(f"*{asset_info['Category']}*")
            
            with col_weight:
                weight = st.number_input(
                    f"Weight % ({symbol})",
                    min_value=0.0,
                    max_value=100.0,
                    value=st.session_state.portfolio[symbol],
                    step=1.0,
                    key=f"weight_{symbol}"
                )
                st.session_state.portfolio[symbol] = weight
                total_weight += weight
            
            with col_remove:
                st.write("")  # Spacer
                if st.button("🗑️", key=f"remove_{symbol}", help=f"Remove {symbol}"):
                    del st.session_state.portfolio[symbol]
                    st.rerun()
            
            portfolio_data.append({
                'Symbol': symbol,
                'Name': asset_info['Name'],
                'Category': asset_info['Category'],
                'Weight': weight,
                'Risk': asset_info['Risk'],
                'Liquidity': asset_info['Liquidity'],
                'OpCost': asset_info['OpCost'],
                'OpRisk': asset_info['OpRisk']
            })
        
        # Weight validation
        if abs(total_weight - 100.0) > 0.1:
            st.warning(f"⚠️ Portfolio weights sum to {total_weight:.1f}%. Consider adjusting to 100%.")
        else:
            st.success("✅ Portfolio weights sum to 100%!")

with col2:
    st.subheader("🎯 Portfolio Scoring")
    
    if st.session_state.portfolio and portfolio_data:
        # Calculate weighted portfolio scores
        def calculate_portfolio_score(portfolio_data, metric):
            total_weighted_score = 0
            total_weight = 0
            
            for asset in portfolio_data:
                weight = asset['Weight'] / 100.0  # Convert percentage to decimal
                score = asset[metric]
                total_weighted_score += weight * score
                total_weight += weight
            
            # Return weighted average, normalized by total weight
            return total_weighted_score / total_weight if total_weight > 0 else 0
        
        # Calculate scores
        portfolio_risk = calculate_portfolio_score(portfolio_data, 'Risk')
        portfolio_liquidity = calculate_portfolio_score(portfolio_data, 'Liquidity')
        portfolio_opcost = calculate_portfolio_score(portfolio_data, 'OpCost')
        portfolio_oprisk = calculate_portfolio_score(portfolio_data, 'OpRisk')
        
        # Display portfolio metrics
        st.metric("🎲 Portfolio Risk", f"{portfolio_risk:.1f}/10")
        st.metric("💧 Portfolio Liquidity", f"{portfolio_liquidity:.1f}/10")
        st.metric("💰 Portfolio Op Cost", f"{portfolio_opcost:.1f}/10")
        st.metric("⚠️ Portfolio Op Risk", f"{portfolio_oprisk:.1f}/10")
        
        # Overall portfolio score (simple average of normalized metrics)
        # Note: Liquidity is inverted for scoring (higher liquidity = better score)
        risk_score = (10 - portfolio_risk) / 10  # Lower risk = better
        liquidity_score = portfolio_liquidity / 10  # Higher liquidity = better
        opcost_score = (10 - portfolio_opcost) / 10  # Lower cost = better
        oprisk_score = (10 - portfolio_oprisk) / 10  # Lower risk = better
        
        overall_score = (risk_score + liquidity_score + opcost_score + oprisk_score) / 4 * 100
        
        st.markdown("---")
        st.metric("🏆 Overall Portfolio Score", f"{overall_score:.1f}/100")
        
        # Score interpretation
        if overall_score >= 80:
            st.success("🟢 Excellent Portfolio - Low risk, high liquidity, efficient operations")
        elif overall_score >= 60:
            st.info("🟡 Good Portfolio - Balanced risk and operational characteristics")
        elif overall_score >= 40:
            st.warning("🟠 Moderate Portfolio - Some risk or operational concerns")
        else:
            st.error("🔴 High Risk Portfolio - Consider rebalancing for better risk/liquidity profile")

# Portfolio Analysis
if st.session_state.portfolio and portfolio_data:
    st.markdown("---")
    st.subheader("📊 Portfolio Analysis")
    
    # Create portfolio DataFrame
    portfolio_df = pd.DataFrame(portfolio_data)
    
    # Advanced Portfolio Visualizations
    if PLOTLY_AVAILABLE or SEABORN_AVAILABLE:
        st.subheader("📊 Advanced Portfolio Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE:
                st.write("**Risk vs Liquidity Analysis (Interactive)**")
                
                # Create interactive scatter plot
                fig = px.scatter(
                    portfolio_df, 
                    x='Risk', 
                    y='Liquidity',
                    size='Weight',
                    color='Category',
                    hover_name='Symbol',
                    hover_data={'Name': True, 'Weight': ':.1f%'},
                    title="Portfolio Risk-Liquidity Profile",
                    labels={'Risk': 'Risk Level (1-10)', 'Liquidity': 'Liquidity Level (1-10)'}
                )
                
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02)
                )
                
                # Add quadrant lines
                fig.add_hline(y=5.5, line_dash="dash", line_color="gray", opacity=0.5)
                fig.add_vline(x=5.5, line_dash="dash", line_color="gray", opacity=0.5)
                
                # Add quadrant annotations
                fig.add_annotation(x=2.5, y=8.5, text="Safe Haven<br>(Low Risk, High Liquidity)", 
                                  showarrow=False, font=dict(size=10), bgcolor="lightgreen", opacity=0.7)
                fig.add_annotation(x=8.5, y=8.5, text="High Risk Liquid<br>(High Risk, High Liquidity)", 
                                  showarrow=False, font=dict(size=10), bgcolor="yellow", opacity=0.7)
                fig.add_annotation(x=2.5, y=2.5, text="Conservative Illiquid<br>(Low Risk, Low Liquidity)", 
                                  showarrow=False, font=dict(size=10), bgcolor="lightblue", opacity=0.7)
                fig.add_annotation(x=8.5, y=2.5, text="High Risk Illiquid<br>(High Risk, Low Liquidity)", 
                                  showarrow=False, font=dict(size=10), bgcolor="lightcoral", opacity=0.7)
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("**Portfolio Composition**")
                st.bar_chart(portfolio_df.set_index('Symbol')['Weight'])
        
        with col2:
            if SEABORN_AVAILABLE:
                st.write("**Operational Cost vs Risk Heatmap**")
                
                # Create correlation matrix for operational metrics
                metrics_df = portfolio_df[['Symbol', 'Risk', 'Liquidity', 'OpCost', 'OpRisk', 'Weight']].set_index('Symbol')
                
                # Create heatmap with Seaborn
                fig_heat, ax = plt.subplots(figsize=(8, 6))
                correlation_matrix = metrics_df[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].corr()
                
                sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, 
                           square=True, ax=ax, cbar_kws={"shrink": .8})
                ax.set_title('Portfolio Metrics Correlation Matrix')
                
                st.pyplot(fig_heat, use_container_width=True)
            else:
                st.write("**Category Breakdown**")
                category_weights = portfolio_df.groupby('Category')['Weight'].sum()
                st.bar_chart(category_weights)
    
        # Portfolio Composition Treemap
        if PLOTLY_AVAILABLE:
            st.write("**Portfolio Allocation Treemap (Interactive)**")
            
            # Create treemap with plotly
            fig_tree = px.treemap(
                portfolio_df, 
                path=['Category', 'Symbol'], 
                values='Weight',
                color='Risk',
                color_continuous_scale='RdYlBu_r',
                title="Portfolio Allocation by Category and Risk Level",
                hover_data={'Name': True, 'Liquidity': True, 'OpCost': True}
            )
            
            fig_tree.update_layout(height=500)
            st.plotly_chart(fig_tree, use_container_width=True)
        
        # Time-Series Portfolio Evolution with Altair
        if ALTAIR_AVAILABLE and NUMPY_AVAILABLE:
            st.write("**Portfolio Evolution Simulation (Time Series)**")
            
            # Generate simulated time series data
            periods = 12  # 12 months
            dates = pd.date_range('2024-01-01', periods=periods, freq='M')
            
            # Simulate portfolio evolution with some randomness
            time_series_data = []
            for i, date in enumerate(dates):
                # Add some market volatility
                market_factor = 1 + np.sin(i/2) * 0.1  # Cyclical market conditions
                noise_factor = 1 + np.random.normal(0, 0.05)  # Random market noise
                
                for _, asset in portfolio_df.iterrows():
                    # Simulate asset performance over time
                    base_return = (10 - asset['Risk']) * 0.02  # Lower risk = steadier returns
                    volatility = asset['Risk'] * 0.03  # Higher risk = more volatility
                    
                    simulated_return = base_return * market_factor * noise_factor + np.random.normal(0, volatility)
                    
                    # Calculate cumulative portfolio value
                    portfolio_value = asset['Weight'] * (1 + simulated_return * (i + 1) / 12)
                    
                    time_series_data.append({
                        'Date': date,
                        'Month': i + 1,
                        'Asset': asset['Symbol'],
                        'Category': asset['Category'],
                        'Weight': asset['Weight'],
                        'Portfolio_Value': portfolio_value,
                        'Risk_Level': asset['Risk'],
                        'Liquidity_Level': asset['Liquidity'],
                        'Return': simulated_return * 100
                    })
            
            time_series_df = pd.DataFrame(time_series_data)
            
            # Create multi-series line chart with Altair
            asset_selection = alt.selection_multi(fields=['Asset'])
            
            line_chart = alt.Chart(time_series_df).mark_line(
                point=True,
                strokeWidth=2
            ).add_selection(
                asset_selection
            ).encode(
                x=alt.X('Month:O', title='Month'),
                y=alt.Y('Portfolio_Value:Q', title='Portfolio Value (Weighted %)'),
                color=alt.Color('Asset:N', scale=alt.Scale(scheme='category20')),
                strokeDash=alt.condition(
                    alt.datum.Risk_Level > 6,
                    alt.value([5, 5]),  # Dashed line for high-risk assets
                    alt.value([1])      # Solid line for low-risk assets
                ),
                opacity=alt.condition(
                    asset_selection,
                    alt.value(1.0),
                    alt.value(0.3)
                ),
                tooltip=['Asset:N', 'Category:N', 'Month:O', 'Portfolio_Value:Q', 'Return:Q']
            ).properties(
                width=700,
                height=400,
                title="Simulated Portfolio Evolution Over Time"
            ).interactive()
            
            st.altair_chart(line_chart, use_container_width=True)
            
            # Portfolio volatility analysis with Altair
            volatility_chart = alt.Chart(time_series_df).mark_circle(
                size=100,
                opacity=0.7
            ).encode(
                x=alt.X('Risk_Level:Q', title='Risk Level'),
                y=alt.Y('mean(Return):Q', title='Average Return (%)'),
                size=alt.Size('Weight:Q', scale=alt.Scale(range=[50, 400])),
                color=alt.Color('Category:N'),
                tooltip=['Asset:N', 'Category:N', 'Risk_Level:Q', 'mean(Return):Q', 'Weight:Q']
            ).properties(
                width=700,
                height=400,
                title="Risk vs Return Analysis with Asset Weights"
            )
            
            st.altair_chart(volatility_chart, use_container_width=True)
            
            # Portfolio composition evolution (stacked area chart)
            area_chart = alt.Chart(time_series_df).mark_area(
                opacity=0.7
            ).encode(
                x=alt.X('Month:O', title='Month'),
                y=alt.Y('sum(Portfolio_Value):Q', title='Total Portfolio Value', stack='zero'),
                color=alt.Color('Category:N', scale=alt.Scale(scheme='category10')),
                tooltip=['Category:N', 'Month:O', 'sum(Portfolio_Value):Q']
            ).properties(
                width=700,
                height=400,
                title="Portfolio Composition Evolution (Stacked by Category)"
            )
            
            st.altair_chart(area_chart, use_container_width=True)
        
        # Advanced Metrics Dashboard
        col1, col2 = st.columns(2)
        
        with col1:
            if PLOTLY_AVAILABLE and NUMPY_AVAILABLE:
                st.write("**Portfolio Efficiency Frontier**")
                
                # Generate efficient frontier simulation
                n_portfolios = 50
                risk_range = np.linspace(portfolio_df['Risk'].min(), portfolio_df['Risk'].max(), n_portfolios)
                
                # Simulate portfolio combinations
                efficient_frontier = []
                for target_risk in risk_range:
                    # Simple optimization: weight inversely to distance from target risk
                    weights = 1 / (np.abs(portfolio_df['Risk'] - target_risk) + 0.1)
                    weights = weights / weights.sum() * 100
                    
                    portfolio_liquidity = np.average(portfolio_df['Liquidity'], weights=weights/100)
                    portfolio_opcost = np.average(portfolio_df['OpCost'], weights=weights/100)
                    
                    efficient_frontier.append({
                        'Risk': target_risk,
                        'Liquidity': portfolio_liquidity,
                        'OpCost': portfolio_opcost
                    })
                
                frontier_df = pd.DataFrame(efficient_frontier)
                
                fig_frontier = px.line(
                    frontier_df, 
                    x='Risk', 
                    y='Liquidity',
                    title="Efficient Frontier: Risk vs Liquidity",
                    labels={'Risk': 'Portfolio Risk', 'Liquidity': 'Portfolio Liquidity'}
                )
                
                # Add current portfolio point
                current_risk = np.average(portfolio_df['Risk'], weights=portfolio_df['Weight']/100)
                current_liquidity = np.average(portfolio_df['Liquidity'], weights=portfolio_df['Weight']/100)
                
                fig_frontier.add_scatter(
                    x=[current_risk], 
                    y=[current_liquidity], 
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Current Portfolio'
                )
                
                st.plotly_chart(fig_frontier, use_container_width=True)
            else:
                st.write("**Portfolio Risk Analysis**")
                risk_df = portfolio_df.groupby('Category')[['Risk', 'Liquidity']].mean()
                st.bar_chart(risk_df)
        
        with col2:
            if PLOTLY_AVAILABLE:
                st.write("**Asset Distribution by Category (Donut Chart)**")
                
                category_weights = portfolio_df.groupby('Category')['Weight'].sum().reset_index()
                
                fig_donut = px.pie(
                    category_weights, 
                    values='Weight', 
                    names='Category',
                    title="Portfolio Distribution by Asset Category",
                    hole=0.4
                )
                
                fig_donut.update_traces(textposition='inside', textinfo='percent+label')
                fig_donut.update_layout(height=400, showlegend=True)
                
                st.plotly_chart(fig_donut, use_container_width=True)
            else:
                st.write("**Category Distribution**")
                category_weights = portfolio_df.groupby('Category')['Weight'].sum()
                st.bar_chart(category_weights)
    
    # Detailed portfolio table
    st.write("**Detailed Portfolio Holdings**")
    display_df = portfolio_df[['Symbol', 'Name', 'Category', 'Weight', 'Risk', 'Liquidity', 'OpCost', 'OpRisk']].copy()
    st.dataframe(display_df, use_container_width=True)
    
    # Portfolio actions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Rebalance to Equal Weights"):
            equal_weight = 100.0 / len(st.session_state.portfolio)
            for symbol in st.session_state.portfolio:
                st.session_state.portfolio[symbol] = equal_weight
            st.success("Portfolio rebalanced to equal weights!")
            st.rerun()
    
    with col2:
        if st.button("🗑️ Clear Portfolio"):
            st.session_state.portfolio = {}
            st.success("Portfolio cleared!")
            st.rerun()
    
    with col3:
        # Export portfolio
        portfolio_csv = portfolio_df.to_csv(index=False)
        st.download_button(
            label="📁 Export Portfolio CSV",
            data=portfolio_csv,
            file_name="my_portfolio.csv",
            mime="text/csv"
        )
    
    # --- Portfolio Optimization Section ---
    st.markdown("---")
    st.subheader("🎯 Portfolio Optimization")
    
    if not SCIPY_AVAILABLE:
        st.warning("⚠️ Portfolio optimization requires SciPy. Install with: `pip install scipy`")
        st.info("Without optimization, you can still use the portfolio builder and manual rebalancing features.")
    elif len(portfolio_data) >= 2:
        col1, col2 = st.columns(2)
        
        with col1:
            optimization_method = st.selectbox(
                "Optimization Method:",
                options=["max_sharpe", "min_vol"],
                format_func=lambda x: {
                    "max_sharpe": "Maximize Sharpe Ratio",
                    "min_vol": "Minimize Volatility"
                }[x]
            )
            
            if st.button("🚀 Optimize Portfolio"):
                with st.spinner("Optimizing portfolio..."):
                    optimization_result = calculate_portfolio_optimization(portfolio_data, optimization_method)
                    
                    if optimization_result:
                        st.success("Portfolio optimization completed!")
                        
                        # Update portfolio weights with optimized values
                        for i, asset_symbol in enumerate(optimization_result['assets']):
                            optimal_weight = optimization_result['optimal_weights'][i] * 100
                            st.session_state.portfolio[asset_symbol] = optimal_weight
                        
                        # Display optimization results
                        st.write("**Optimization Results:**")
                        col_opt1, col_opt2, col_opt3 = st.columns(3)
                        with col_opt1:
                            st.metric("Expected Return", f"{optimization_result['expected_return']:.2%}")
                        with col_opt2:
                            st.metric("Volatility", f"{optimization_result['volatility']:.2%}")
                        with col_opt3:
                            st.metric("Sharpe Ratio", f"{optimization_result['sharpe_ratio']:.3f}")
                        
                        # Show optimal weights
                        st.write("**Optimal Asset Allocation:**")
                        opt_weights_df = pd.DataFrame({
                            'Asset': optimization_result['assets'],
                            'Optimal Weight (%)': optimization_result['optimal_weights'] * 100
                        }).sort_values('Optimal Weight (%)', ascending=False)
                        
                        st.bar_chart(opt_weights_df.set_index('Asset')['Optimal Weight (%)'])
                        st.dataframe(opt_weights_df, use_container_width=True)
                        
                        st.rerun()
                    else:
                        st.error("Portfolio optimization failed. Please check your portfolio composition.")
        
        with col2:
            st.write("**Portfolio Analysis:**")
            
            # Current portfolio metrics
            current_return = sum([(0.02 + (asset['Risk'] / 10) * 0.12) * (asset['Weight'] / 100) 
                                for asset in portfolio_data])
            current_vol = (sum([((asset['Risk'] / 10) * 0.3 * (1 - asset['Liquidity'] / 20))**2 * (asset['Weight'] / 100)**2 
                              for asset in portfolio_data]))**0.5
            current_sharpe = current_return / current_vol if current_vol > 0 else 0
            
            col_curr1, col_curr2 = st.columns(2)
            with col_curr1:
                st.metric("Current Return", f"{current_return:.2%}")
                st.metric("Current Sharpe", f"{current_sharpe:.3f}")
            with col_curr2:
                st.metric("Current Volatility", f"{current_vol:.2%}")
                
                # Risk-return scatter of current portfolio
                if PLOTLY_AVAILABLE:
                    fig_current = px.scatter(
                        x=[current_vol],
                        y=[current_return],
                        title="Current Portfolio Position",
                        labels={'x': 'Volatility', 'y': 'Expected Return'},
                        color_discrete_sequence=['red']
                    )
                    fig_current.update_traces(marker_size=15)
                    fig_current.update_layout(height=300)
                    st.plotly_chart(fig_current, use_container_width=True)
        
        # Efficient Frontier Visualization
        if PLOTLY_AVAILABLE:
            st.write("**Efficient Frontier Analysis**")
            
            frontier_data = calculate_efficient_frontier(portfolio_data)
            if frontier_data:
                frontier_df = pd.DataFrame(frontier_data)
                
                fig_frontier = px.scatter(
                    frontier_df,
                    x='volatility',
                    y='return',
                    color='sharpe',
                    title="Efficient Frontier - Risk vs Return",
                    labels={'volatility': 'Volatility (Risk)', 'return': 'Expected Return', 'sharpe': 'Sharpe Ratio'},
                    color_continuous_scale='Viridis'
                )
                
                # Add current portfolio point
                fig_frontier.add_scatter(
                    x=[current_vol],
                    y=[current_return],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Current Portfolio'
                )
                
                fig_frontier.update_layout(height=500)
                st.plotly_chart(fig_frontier, use_container_width=True)
                
                st.info("🎯 **Interpretation:** Points closer to the upper-left represent better risk-adjusted returns. " +
                       "Your current portfolio is shown as a red star.")
    else:
        st.info("Add at least 2 assets to your portfolio to enable optimization features.")

# --- Workstream Network Analysis ---
st.markdown("---")
st.subheader("🌐 Workstream Network Analysis")

if PLOTLY_AVAILABLE and NETWORKX_AVAILABLE:
    st.write("**Interactive Network Graph - Workstream Dependencies**")
    st.info("This network shows how workstreams are connected through shared applications and processes. Larger nodes indicate higher complexity workstreams.")
    
    try:
        # Build network data
        # Create graph
        G = nx.Graph()
        
        # Add workstream nodes
        for name, data in workstreams_data.items():
            G.add_node(name, 
                      node_type='workstream',
                      complexity=data['complexity'],
                      operational_risk=data['operational_risk'],
                      gap_count=len(identified_gaps.get(name, [])),
                      process_count=len(data['processes']),
                      app_count=len(data['applications']))
        
        # Add application nodes and connections
        app_connections = {}
        for ws_name, ws_data in workstreams_data.items():
            for app in ws_data['applications']:
                if app not in app_connections:
                    app_connections[app] = []
                app_connections[app].append(ws_name)
        
        # Create connections between workstreams that share applications
        for app, workstreams in app_connections.items():
            if len(workstreams) > 1:
                # Add application as a node
                G.add_node(app, node_type='application', shared_by=len(workstreams))
                
                # Connect workstreams through shared applications
                for ws in workstreams:
                    G.add_edge(ws, app, connection_type='uses_application')
        
        # Calculate layout using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Prepare data for Plotly network graph
        edge_x = []
        edge_y = []
        edge_info = []
        
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_info.append(f"{edge[0]} ↔ {edge[1]}")
        
        # Create edges trace
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1, color='#888'),
            hoverinfo='none',
            mode='lines'
        )
        
        # Prepare node data
        node_x = []
        node_y = []
        node_info = []
        node_colors = []
        node_sizes = []
        node_symbols = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
            if G.nodes[node]['node_type'] == 'workstream':
                # Workstream nodes
                complexity = G.nodes[node]['complexity']
                gap_count = G.nodes[node]['gap_count']
                process_count = G.nodes[node]['process_count']
                
                node_info.append(f"<b>{node}</b><br>" +
                               f"Complexity: {complexity}/10<br>" +
                               f"Processes: {process_count}<br>" +
                               f"Gaps: {gap_count}<br>" +
                               f"Type: Workstream")
                
                node_colors.append(complexity)
                node_sizes.append(max(20, complexity * 3))
                node_symbols.append('circle')
                
            else:
                # Application nodes
                shared_by = G.nodes[node]['shared_by']
                node_info.append(f"<b>{node}</b><br>" +
                               f"Shared by: {shared_by} workstreams<br>" +
                               f"Type: Application")
                
                node_colors.append(shared_by + 10)  # Different color scale
                node_sizes.append(max(15, shared_by * 5))
                node_symbols.append('diamond')
        
        # Create nodes trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=[node for node in G.nodes()],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=node_info,
            marker=dict(
                showscale=True,
                colorscale='Viridis',
                reversescale=True,
                color=node_colors,
                size=node_sizes,
                symbol=node_symbols,
                colorbar=dict(
                    thickness=15,
                    len=0.5,
                    x=1.05,
                    title="Complexity / Sharing Level"
                ),
                line=dict(width=2, color='white')
            )
        )
        
        # Create figure
        fig_network = go.Figure(data=[edge_trace, node_trace])
        
        fig_network.update_layout(
            title=dict(
                text="Workstream Dependencies Network<br><sub>Circles = Workstreams, Diamonds = Shared Applications</sub>",
                font=dict(size=16)
            ),
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=700
        )
        
        # Add annotation separately
        fig_network.add_annotation(
            text="Connections show shared applications between workstreams",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color='gray', size=10)
        )
        
        st.plotly_chart(fig_network, use_container_width=True)
        
        # Network analysis metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Workstreams", len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'workstream']))
            st.metric("Shared Applications", len([n for n in G.nodes() if G.nodes[n]['node_type'] == 'application']))
        
        with col2:
            # Calculate centrality metrics
            centrality = nx.degree_centrality(G)
            most_connected_ws = max([n for n in G.nodes() if G.nodes[n]['node_type'] == 'workstream'], 
                                   key=lambda x: centrality[x])
            st.metric("Most Connected Workstream", most_connected_ws)
            st.metric("Connection Score", f"{centrality[most_connected_ws]:.2f}")
        
        with col3:
            # Find most shared application
            most_shared_app = max([n for n in G.nodes() if G.nodes[n]['node_type'] == 'application'], 
                                 key=lambda x: G.nodes[x]['shared_by'])
            st.metric("Most Shared Application", most_shared_app)
            st.metric("Used by", f"{G.nodes[most_shared_app]['shared_by']} workstreams")
        
        # Detailed network analysis
        st.write("**Network Analysis Insights**")
        
        # Application sharing analysis
        st.write("**Most Critical Shared Applications:**")
        app_sharing = [(app, data['shared_by']) for app, data in G.nodes(data=True) 
                       if data['node_type'] == 'application']
        app_sharing.sort(key=lambda x: x[1], reverse=True)
        
        for app, share_count in app_sharing[:5]:
            connected_ws = [n for n in G.neighbors(app)]
            st.write(f"• **{app}**: Shared by {share_count} workstreams ({', '.join(connected_ws)})")
        
        # Workstream connectivity analysis
        st.write("**Workstream Connectivity Rankings:**")
        ws_connectivity = [(ws, centrality[ws]) for ws in G.nodes() 
                          if G.nodes[ws]['node_type'] == 'workstream']
        ws_connectivity.sort(key=lambda x: x[1], reverse=True)
        
        for ws, conn_score in ws_connectivity[:5]:
            neighbor_count = len(list(G.neighbors(ws)))
            st.write(f"• **{ws}**: Connectivity score {conn_score:.3f} ({neighbor_count} connections)")
            
    except Exception as e:
        st.error(f"Error creating network visualization: {str(e)}")
        st.info("Network analysis requires both Plotly and NetworkX libraries.")

else:
    st.warning("Network analysis requires Plotly and NetworkX libraries.")
    st.write("**Alternative: Application Sharing Summary**")
    
    # Create fallback analysis without NetworkX
    app_connections = {}
    for ws_name, ws_data in workstreams_data.items():
        for app in ws_data['applications']:
            if app not in app_connections:
                app_connections[app] = []
            app_connections[app].append(ws_name)
    
    shared_apps = {app: len(workstreams) for app, workstreams in app_connections.items() if len(workstreams) > 1}
    
    if shared_apps:
        shared_apps_df = pd.DataFrame([
            {'Application': app, 'Shared_By': count}
            for app, count in sorted(shared_apps.items(), key=lambda x: x[1], reverse=True)
        ])
        
        st.bar_chart(shared_apps_df.set_index('Application')['Shared_By'])
        
        st.write("**Most Shared Applications:**")
        for app, count in sorted(shared_apps.items(), key=lambda x: x[1], reverse=True)[:5]:
            workstreams = app_connections[app]
            st.write(f"• **{app}**: Used by {count} workstreams ({', '.join(workstreams)})")

# --- Data Export Section ---
st.markdown("---")
st.header("📤 Data Export")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Filtered Data (CSV)"):
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"asset_data_filtered_{selected_category.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("📈 Export All Data (JSON)"):
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name="asset_data_complete.json",
            mime="application/json"
        )

with col3:
    if st.button("📋 Export Summary Stats"):
        summary_stats = df.describe()
        csv_stats = summary_stats.to_csv()
        st.download_button(
            label="Download Stats CSV",
            data=csv_stats,
            file_name="asset_summary_statistics.csv",
            mime="text/csv"
        )

# Tab 3: MLOps for Fund Accounting
with main_tab3:
    st.header("🤖 MLOps for Fund Accounting")
st.markdown("""
**Machine Learning Operations (MLOps) in Financial Services**

Explore how modern MLOps tools can revolutionize fund accounting workflows, 
from automated data quality checks to predictive modeling and deployment.
""")

# Create MLOps tabs
mlops_tabs = st.tabs([
    "📋 MLOps Overview", 
    "📊 Model Tracking (MLflow)", 
    "✅ Data Quality (Great Expectations)", 
    "🔄 Workflow Management (Prefect)",
    "📦 Model Deployment",
    "📈 Production Monitoring"
])

with mlops_tabs[0]:
    st.subheader("📋 MLOps Tools for Fund Accounting")
    
    st.markdown("""
    ### Why MLOps in Fund Accounting?
    
    Modern fund accounting involves complex data pipelines, regulatory compliance,
    and the need for accurate, reproducible calculations. MLOps tools can help:
    
    - **Automate** repetitive data validation and calculation processes
    - **Ensure** data quality and regulatory compliance
    - **Track** model performance and data lineage
    - **Scale** operations for hundreds of funds efficiently
    - **Reduce** manual errors and operational risk
    """)
    
    # MLOps Tools Overview
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### 🎯 Core MLOps Tools
        
        **MLflow** - Model lifecycle management
        - Track experiments and model versions
        - Model registry for production deployment
        - Performance monitoring and comparison
        
        **DVC** - Data version control
        - Version control for large datasets
        - Reproducible data pipelines
        - Audit trails for compliance
        
        **Great Expectations** - Data quality
        - Automated data validation
        - Quality checks before calculations
        - Compliance reporting
        
        **Prefect** - Workflow orchestration
        - Robust data pipeline management
        - Automatic retry and error handling
        - Observable workflow execution
        """)
    
    with col2:
        st.markdown("""
        #### 🏦 Fund Accounting Applications
        
        **NAV Calculation Automation**
        - Automated daily NAV calculations
        - Data quality checks before processing
        - Model-driven price validation
        
        **Investor Behavior Prediction**
        - Redemption forecasting models
        - Subscription pattern analysis
        - Liquidity planning optimization
        
        **Risk Management**
        - Fraud detection in transactions
        - Portfolio risk assessment
        - Regulatory compliance monitoring
        
        **Reporting Automation**
        - Automated P&L generation
        - Regulatory report creation
        - Performance attribution analysis
        """)
    
    # Success metrics
    st.markdown("---")
    st.subheader("📈 Expected Benefits")
    
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Process Automation", "85%", "Time savings")
    with metric_col2:
        st.metric("Error Reduction", "95%", "Accuracy improvement")
    with metric_col3:
        st.metric("Compliance", "100%", "Audit trail coverage")
    with metric_col4:
        st.metric("Scalability", "10x", "Fund capacity increase")

with mlops_tabs[1]:
    st.subheader("📊 MLflow: Model Tracking for Fund Management")
    
    st.markdown("""
    ### MLflow in Fund Accounting Context
    
    MLflow helps track, manage, and deploy machine learning models used in fund operations.
    Common applications include redemption prediction, fraud detection, and portfolio optimization.
    """)
    
    # Simulated MLflow experiment tracking
    st.markdown("#### 🧪 Example: Investor Redemption Prediction Model")
    
    # Create mock experiment data
    import random
    import datetime
    
    experiment_data = []
    model_types = ["Random Forest", "XGBoost", "Neural Network", "SVM", "Logistic Regression"]
    
    for i in range(10):
        experiment_data.append({
            "Run ID": f"run_{i+1:03d}",
            "Model": random.choice(model_types),
            "Accuracy": round(random.uniform(0.75, 0.95), 3),
            "Precision": round(random.uniform(0.70, 0.90), 3),
            "Recall": round(random.uniform(0.65, 0.85), 3),
            "F1 Score": round(random.uniform(0.68, 0.87), 3),
            "Training Time": f"{random.randint(5, 45)} min",
            "Data Version": f"v1.{random.randint(1, 5)}",
            "Status": random.choice(["Completed", "Running", "Failed"]),
            "Created": (datetime.datetime.now() - datetime.timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d")
        })
    
    experiment_df = pd.DataFrame(experiment_data)
    
    # Display experiment tracking table
    st.markdown("**Experiment Tracking Dashboard**")
    st.dataframe(experiment_df, use_container_width=True)
    
    # Model performance comparison
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Model Performance Comparison**")
        if PLOTLY_AVAILABLE:
            fig_performance = px.scatter(
                experiment_df,
                x='Precision',
                y='Recall',
                size='Accuracy',
                color='Model',
                hover_data=['F1 Score', 'Run ID'],
                title="Model Performance: Precision vs Recall"
            )
            fig_performance.update_layout(height=400)
            st.plotly_chart(fig_performance, use_container_width=True, key="mlflow_performance")
        else:
            st.info("Install Plotly for interactive performance visualization")
    
    with col2:
        st.markdown("**Best Performing Models**")
        top_models = experiment_df.nlargest(3, 'F1 Score')[['Model', 'F1 Score', 'Accuracy', 'Run ID']]
        
        for idx, row in top_models.iterrows():
            st.markdown(f"""
            **#{idx+1}: {row['Model']}** (Run: {row['Run ID']})
            - F1 Score: {row['F1 Score']:.3f}
            - Accuracy: {row['Accuracy']:.3f}
            """)
    
    # MLflow features demonstration
    st.markdown("---")
    st.markdown("#### 🎯 MLflow Key Features")
    
    feature_col1, feature_col2, feature_col3 = st.columns(3)
    
    with feature_col1:
        st.markdown("""
        **Experiment Tracking**
        - Log parameters, metrics, artifacts
        - Compare model performance
        - Track data versions
        - Reproducible experiments
        """)
    
    with feature_col2:
        st.markdown("""
        **Model Registry**
        - Centralized model store
        - Version management
        - Stage transitions (Dev → Staging → Prod)
        - Model lineage tracking
        """)
    
    with feature_col3:
        st.markdown("""
        **Model Deployment**
        - REST API serving
        - Batch inference
        - Real-time predictions
        - A/B testing support
        """)
    
    # Implementation code example
    with st.expander("💻 Implementation Example"):
        st.code("""
# Example: MLflow tracking for redemption prediction model

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Start MLflow experiment
mlflow.set_experiment("investor_redemption_prediction")

with mlflow.start_run():
    # Log parameters
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("data_version", "v1.3")
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=10)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Log metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("precision", precision_score(y_test, y_pred))
    mlflow.log_metric("recall", recall_score(y_test, y_pred))
    
    # Log model
    mlflow.sklearn.log_model(model, "redemption_model")
    
    # Log artifacts (e.g., feature importance plot)
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")

print("Experiment logged to MLflow!")
        """, language="python")

with mlops_tabs[2]:
    st.subheader("✅ Great Expectations: Data Quality for Fund Operations")
    
    st.markdown("""
    ### Data Quality in Fund Accounting
    
    Great Expectations ensures data quality and integrity before critical fund calculations.
    It's essential for NAV calculations, regulatory compliance, and investor reporting.
    """)
    
    # Simulate data quality checks on NAV data
    st.markdown("#### 🔍 Example: Daily NAV Data Quality Checks")
    
    # Create mock data quality results
    expectations_results = [
        {
            "Expectation": "NAV values must be non-null",
            "Status": "✅ PASSED",
            "Details": "3,247 of 3,247 values are non-null (100%)",
            "Critical": True
        },
        {
            "Expectation": "NAV prices must be positive",
            "Status": "✅ PASSED", 
            "Details": "All 3,247 NAV values are > 0",
            "Critical": True
        },
        {
            "Expectation": "Daily NAV change must be < 10%",
            "Status": "⚠️ WARNING",
            "Details": "2 funds exceeded 10% daily change threshold",
            "Critical": True
        },
        {
            "Expectation": "Fund IDs must match registry",
            "Status": "✅ PASSED",
            "Details": "All 187 fund IDs found in master registry",
            "Critical": True
        },
        {
            "Expectation": "Trade dates must be valid business days",
            "Status": "✅ PASSED",
            "Details": "All 3,247 dates are valid business days",
            "Critical": False
        },
        {
            "Expectation": "Currency codes must be ISO 4217",
            "Status": "❌ FAILED",
            "Details": "5 invalid currency codes found (XYZ, ABC)",
            "Critical": False
        }
    ]
    
    # Display expectations results
    expectations_df = pd.DataFrame(expectations_results)
    
    # Color code the status
    def color_status(val):
        if "PASSED" in val:
            return 'background-color: #d4edda; color: #155724'
        elif "WARNING" in val:
            return 'background-color: #fff3cd; color: #856404'
        elif "FAILED" in val:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_df = expectations_df.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Data quality metrics
    st.markdown("---")
    st.markdown("#### 📊 Data Quality Summary")
    
    dq_col1, dq_col2, dq_col3, dq_col4 = st.columns(4)
    
    passed_count = len([r for r in expectations_results if "PASSED" in r["Status"]])
    warning_count = len([r for r in expectations_results if "WARNING" in r["Status"]])
    failed_count = len([r for r in expectations_results if "FAILED" in r["Status"]])
    critical_failed = len([r for r in expectations_results if "FAILED" in r["Status"] and r["Critical"]])
    
    with dq_col1:
        st.metric("Tests Passed", f"{passed_count}/6", f"{passed_count/6*100:.0f}%")
    with dq_col2:
        st.metric("Warnings", warning_count, "Non-blocking")
    with dq_col3:
        st.metric("Failed Tests", failed_count, "Needs attention")
    with dq_col4:
        st.metric("Critical Failures", critical_failed, "⚠️ Blocking" if critical_failed > 0 else "✅ Safe")
    
    # Data quality trends
    if PLOTLY_AVAILABLE:
        st.markdown("#### 📈 Data Quality Trends (Last 30 Days)")
        
        # Generate mock trend data
        import datetime
        dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(30, 0, -1)]
        
        trend_data = {
            'Date': dates,
            'Passed': [random.randint(85, 100) for _ in range(30)],
            'Warnings': [random.randint(0, 10) for _ in range(30)],
            'Failed': [random.randint(0, 5) for _ in range(30)]
        }
        
        trend_df = pd.DataFrame(trend_data)
        
        fig_quality_trend = px.line(
            trend_df, 
            x='Date', 
            y=['Passed', 'Warnings', 'Failed'],
            title="Data Quality Trends - Daily Test Results",
            labels={'value': 'Number of Tests', 'variable': 'Status'},
            color_discrete_map={'Passed': 'green', 'Warnings': 'orange', 'Failed': 'red'}
        )
        fig_quality_trend.update_layout(height=400)
        st.plotly_chart(fig_quality_trend, use_container_width=True, key="data_quality_trend")
    
    # Great Expectations features
    st.markdown("---")
    st.markdown("#### 🎯 Great Expectations Features")
    
    ge_col1, ge_col2, ge_col3 = st.columns(3)
    
    with ge_col1:
        st.markdown("""
        **Data Validation**
        - Automated quality checks
        - Custom business rules
        - Statistical validations
        - Data profiling
        """)
    
    with ge_col2:
        st.markdown("""
        **Documentation**
        - Auto-generated data docs
        - Expectation suites
        - Validation results
        - Data lineage tracking
        """)
    
    with ge_col3:
        st.markdown("""
        **Integration**
        - Pipeline integration
        - Alert notifications
        - CI/CD workflows
        - Database connectors
        """)
    
    # Implementation example
    with st.expander("💻 Great Expectations Implementation"):
        st.code("""
# Example: Data quality checks for daily NAV data

import great_expectations as ge
from great_expectations.dataset import PandasDataset

# Load NAV data with Great Expectations
nav_df = ge.read_csv("daily_nav_data.csv")

# Define expectations (data quality rules)
nav_df.expect_column_values_to_not_be_null("nav_per_share")
nav_df.expect_column_values_to_be_between("nav_per_share", min_value=0)
nav_df.expect_column_values_to_be_in_set("fund_id", value_set=valid_fund_ids)
nav_df.expect_column_values_to_match_regex("currency", "^[A-Z]{3}$")

# Custom expectation for daily NAV change
def expect_daily_nav_change_to_be_reasonable(df, column, max_change=0.10):
    df_sorted = df.sort_values(['fund_id', 'nav_date'])
    df_sorted['daily_change'] = df_sorted.groupby('fund_id')[column].pct_change()
    extreme_changes = df_sorted[abs(df_sorted['daily_change']) > max_change]
    
    return {
        "success": len(extreme_changes) == 0,
        "result": {
            "observed_value": len(extreme_changes),
            "details": extreme_changes[['fund_id', 'nav_date', 'daily_change']].to_dict('records')
        }
    }

# Run validation
validation_results = nav_df.validate()

# Generate data documentation
context = ge.DataContext()
checkpoint = context.get_checkpoint("nav_data_quality")
results = checkpoint.run()

if results["success"]:
    print("✅ All data quality checks passed!")
    # Proceed with NAV calculations
else:
    print("❌ Data quality issues found:")
    for result in results["run_results"]:
        if not result["validation_result"]["success"]:
            print(f"- {result['expectation_type']}: {result['kwargs']}")
    # Alert team and halt processing
        """, language="python")

with mlops_tabs[3]:
    st.subheader("🔄 Prefect: Workflow Management for Fund Operations")
    
    st.markdown("""
    ### Prefect for Robust Fund Accounting Workflows
    
    Prefect orchestrates complex data pipelines for fund accounting, ensuring reliable execution
    of daily NAV calculations, P&L generation, and regulatory reporting with automatic retry
    and comprehensive monitoring.
    """)
    
    # Simulate a fund accounting workflow with Prefect
    st.markdown("#### 🏗️ Example: Daily Fund Accounting Pipeline")
    
    # Create mock workflow data
    workflow_tasks = [
        {
            "Task": "Fetch Market Data",
            "Status": "✅ Completed",
            "Duration": "2m 34s",
            "Start Time": "06:00:00",
            "End Time": "06:02:34",
            "Retries": 0,
            "Dependencies": [],
            "Output": "4,247 securities updated"
        },
        {
            "Task": "Validate Trade Files",
            "Status": "✅ Completed", 
            "Duration": "1m 12s",
            "Start Time": "06:02:35",
            "End Time": "06:03:47",
            "Retries": 1,
            "Dependencies": ["Fetch Market Data"],
            "Output": "3,128 trades validated"
        },
        {
            "Task": "Calculate Portfolio Values",
            "Status": "✅ Completed",
            "Duration": "4m 56s", 
            "Start Time": "06:03:48",
            "End Time": "06:08:44",
            "Retries": 0,
            "Dependencies": ["Fetch Market Data", "Validate Trade Files"],
            "Output": "187 portfolios valued"
        },
        {
            "Task": "Compute NAV",
            "Status": "🔄 Running",
            "Duration": "2m 18s",
            "Start Time": "06:08:45",
            "End Time": "-",
            "Retries": 0,
            "Dependencies": ["Calculate Portfolio Values"],
            "Output": "124/187 funds completed"
        },
        {
            "Task": "Generate P&L Reports",
            "Status": "⏸️ Waiting",
            "Duration": "-",
            "Start Time": "-",
            "End Time": "-", 
            "Retries": 0,
            "Dependencies": ["Compute NAV"],
            "Output": "Pending NAV completion"
        },
        {
            "Task": "Regulatory Compliance Check",
            "Status": "⏸️ Waiting",
            "Duration": "-",
            "Start Time": "-", 
            "End Time": "-",
            "Retries": 0,
            "Dependencies": ["Generate P&L Reports"],
            "Output": "Awaiting P&L data"
        },
        {
            "Task": "Distribute Client Reports",
            "Status": "⏸️ Waiting",
            "Duration": "-",
            "Start Time": "-",
            "End Time": "-",
            "Retries": 0,
            "Dependencies": ["Regulatory Compliance Check"],
            "Output": "Final step pending"
        }
    ]
    
    workflow_df = pd.DataFrame(workflow_tasks)
    
    # Display workflow status
    st.markdown("**Current Workflow Execution Status**")
    
    # Color code status
    def style_status(val):
        if "Completed" in val:
            return 'background-color: #d4edda; color: #155724'
        elif "Running" in val:
            return 'background-color: #cce5ff; color: #004085'
        elif "Waiting" in val:
            return 'background-color: #f8f9fa; color: #6c757d'
        elif "Failed" in val:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_workflow = workflow_df.style.applymap(style_status, subset=['Status'])
    st.dataframe(styled_workflow, use_container_width=True)
    
    # Workflow progress metrics
    st.markdown("---")
    st.markdown("#### 📊 Pipeline Progress")
    
    progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
    
    completed_tasks = len([t for t in workflow_tasks if "Completed" in t["Status"]])
    running_tasks = len([t for t in workflow_tasks if "Running" in t["Status"]])
    waiting_tasks = len([t for t in workflow_tasks if "Waiting" in t["Status"]])
    total_tasks = len(workflow_tasks)
    
    with progress_col1:
        st.metric("Total Progress", f"{completed_tasks}/{total_tasks}", f"{completed_tasks/total_tasks*100:.0f}%")
    with progress_col2:
        st.metric("Completed Tasks", completed_tasks, "✅")
    with progress_col3:
        st.metric("Running Tasks", running_tasks, "🔄")
    with progress_col4:
        st.metric("Waiting Tasks", waiting_tasks, "⏸️")
    
    # Workflow visualization
    if PLOTLY_AVAILABLE:
        st.markdown("#### 🕒 Task Timeline Visualization")
        
        # Create Gantt-like chart for completed tasks
        timeline_data = []
        for task in workflow_tasks:
            if task["Start Time"] != "-" and task["End Time"] != "-":
                # Convert times to datetime for plotting
                start_dt = datetime.datetime.strptime(f"2024-01-01 {task['Start Time']}", "%Y-%m-%d %H:%M:%S")
                end_dt = datetime.datetime.strptime(f"2024-01-01 {task['End Time']}", "%Y-%m-%d %H:%M:%S")
                
                timeline_data.append({
                    'Task': task['Task'],
                    'Start': start_dt,
                    'Finish': end_dt,
                    'Status': task['Status'],
                    'Duration': task['Duration'],
                    'Retries': task['Retries']
                })
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            
            # Create Gantt chart
            fig_timeline = px.timeline(
                timeline_df,
                x_start="Start",
                x_end="Finish", 
                y="Task",
                color="Status",
                title="Fund Accounting Pipeline - Task Execution Timeline",
                hover_data=["Duration", "Retries"]
            )
            
            fig_timeline.update_layout(
                height=400,
                xaxis_title="Time",
                yaxis_title="Tasks"
            )
            
            st.plotly_chart(fig_timeline, use_container_width=True, key="prefect_timeline")
    
    # Workflow dependency graph visualization
    if PLOTLY_AVAILABLE and NETWORKX_AVAILABLE:
        st.markdown("#### 🔗 Task Dependency Graph")
        
        try:
            # Create dependency graph
            G_workflow = nx.DiGraph()
            
            # Add task nodes
            for task in workflow_tasks:
                status_color = {
                    "✅ Completed": "green",
                    "🔄 Running": "blue", 
                    "⏸️ Waiting": "gray",
                    "❌ Failed": "red"
                }.get(task["Status"], "gray")
                
                G_workflow.add_node(task["Task"], status=task["Status"], color=status_color)
            
            # Add dependency edges
            for task in workflow_tasks:
                for dep in task["Dependencies"]:
                    G_workflow.add_edge(dep, task["Task"])
            
            # Create layout
            pos_workflow = nx.spring_layout(G_workflow, k=2, iterations=50)
            
            # Prepare Plotly data
            edge_x, edge_y = [], []
            for edge in G_workflow.edges():
                x0, y0 = pos_workflow[edge[0]]
                x1, y1 = pos_workflow[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Create edge trace
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            # Prepare node data
            node_x, node_y, node_colors, node_text, node_hover = [], [], [], [], []
            
            for node in G_workflow.nodes():
                x, y = pos_workflow[node]
                node_x.append(x)
                node_y.append(y)
                
                status = G_workflow.nodes[node]['status']
                color = G_workflow.nodes[node]['color']
                
                node_colors.append(color)
                node_text.append(node.replace(' ', '<br>'))  # Break long names
                node_hover.append(f"<b>{node}</b><br>Status: {status}")
            
            # Create node trace
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                hovertext=node_hover,
                text=node_text,
                textposition="middle center",
                textfont=dict(size=10, color="white"),
                marker=dict(
                    size=40,
                    color=node_colors,
                    line=dict(width=2, color='white')
                )
            )
            
            # Create figure
            fig_deps = go.Figure(data=[edge_trace, node_trace])
            fig_deps.update_layout(
                title="Task Dependency Flow",
                showlegend=False,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                height=500
            )
            
            st.plotly_chart(fig_deps, use_container_width=True, key="prefect_deps")
            
        except Exception as e:
            st.warning(f"Dependency graph requires NetworkX library: {str(e)}")
    
    # Historical workflow performance
    if PLOTLY_AVAILABLE:
        st.markdown("#### 📈 Historical Pipeline Performance (Last 30 Days)")
        
        # Generate mock historical data
        hist_dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(30, 0, -1)]
        
        historical_data = {
            'Date': hist_dates,
            'Success_Rate': [random.uniform(0.92, 1.0) for _ in range(30)],
            'Avg_Duration': [random.uniform(15, 25) for _ in range(30)],  # minutes
            'Failed_Tasks': [random.randint(0, 3) for _ in range(30)],
            'Retry_Count': [random.randint(0, 8) for _ in range(30)]
        }
        
        hist_df = pd.DataFrame(historical_data)
        
        # Create performance dashboard
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            fig_success = px.line(
                hist_df,
                x='Date',
                y='Success_Rate',
                title="Pipeline Success Rate",
                labels={'Success_Rate': 'Success Rate', 'Date': 'Date'}
            )
            fig_success.update_traces(line_color='green')
            fig_success.update_layout(height=300, yaxis=dict(range=[0.9, 1.0]))
            st.plotly_chart(fig_success, use_container_width=True, key="prefect_success")
        
        with perf_col2:
            fig_duration = px.line(
                hist_df,
                x='Date', 
                y='Avg_Duration',
                title="Average Pipeline Duration",
                labels={'Avg_Duration': 'Duration (minutes)', 'Date': 'Date'}
            )
            fig_duration.update_traces(line_color='blue')
            fig_duration.update_layout(height=300)
            st.plotly_chart(fig_duration, use_container_width=True, key="prefect_duration")
    
    # Prefect features overview
    st.markdown("---")
    st.markdown("#### 🎯 Prefect Key Features")
    
    prefect_col1, prefect_col2, prefect_col3 = st.columns(3)
    
    with prefect_col1:
        st.markdown("""
        **Workflow Orchestration**
        - Task dependencies management
        - Parallel execution support
        - Dynamic workflow generation
        - Conditional logic handling
        """)
    
    with prefect_col2:
        st.markdown("""
        **Reliability & Monitoring**
        - Automatic retry mechanisms
        - Failure handling strategies
        - Real-time status monitoring
        - Comprehensive logging
        """)
    
    with prefect_col3:
        st.markdown("""
        **Scaling & Deployment**
        - Cloud and on-premise deployment
        - Kubernetes integration
        - Resource optimization
        - Multi-environment support
        """)
    
    # Alert management
    st.markdown("---")
    st.markdown("#### 🚨 Alert Management")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("**Recent Alerts (Last 7 Days)**")
        alerts_data = [
            {"Date": "2024-01-03", "Type": "WARNING", "Message": "Trade validation took 3x longer than usual", "Resolved": True},
            {"Date": "2024-01-02", "Type": "INFO", "Message": "Market data fetch completed with 1 retry", "Resolved": True},
            {"Date": "2024-01-01", "Type": "ERROR", "Message": "P&L report generation failed - missing price data", "Resolved": False}
        ]
        
        for alert in alerts_data:
            status_emoji = "✅" if alert["Resolved"] else "🔴"
            type_color = {"WARNING": "🟡", "INFO": "🔵", "ERROR": "🔴"}[alert["Type"]]
            st.markdown(f"{status_emoji} {type_color} **{alert['Date']}**: {alert['Message']}")
    
    with alert_col2:
        st.markdown("**Alert Configuration**")
        st.markdown("""
        - **Critical Task Failures**: Immediate email + Slack
        - **Performance Degradation**: Daily digest
        - **Data Quality Issues**: Real-time notifications
        - **Compliance Violations**: Escalation to managers
        """)
    
    # Implementation example
    with st.expander("💻 Prefect Implementation Example"):
        st.code("""
# Example: Daily Fund Accounting Pipeline with Prefect

from prefect import flow, task
from prefect.tasks import task_input_hash
from datetime import datetime, timedelta
import pandas as pd

@task(cache_key_fn=task_input_hash, cache_expiration=timedelta(hours=1))
def fetch_market_data(date: str) -> pd.DataFrame:
    \"\"\"Fetch market data for given date with caching\"\"\"
    # Implementation to fetch from Bloomberg, Reuters, etc.
    print(f"📊 Fetching market data for {date}")
    # ... fetch logic ...
    return market_data

@task(retries=3, retry_delay_seconds=60)
def validate_trades(trade_file: str) -> dict:
    \"\"\"Validate trade file with automatic retries\"\"\"
    try:
        # Implementation of trade validation logic
        trades_df = pd.read_csv(trade_file)
        
        # Validation checks
        assert len(trades_df) > 0, "Trade file is empty"
        assert trades_df['trade_amount'].notna().all(), "Missing trade amounts"
        
        print(f"✅ Validated {len(trades_df)} trades")
        return {"status": "success", "trade_count": len(trades_df)}
        
    except Exception as e:
        print(f"❌ Trade validation failed: {str(e)}")
        raise

@task
def calculate_portfolio_values(market_data: pd.DataFrame, trades: dict) -> pd.DataFrame:
    \"\"\"Calculate portfolio values using market data and trades\"\"\"
    # Implementation of portfolio valuation logic
    print(f"💰 Calculating values for {trades['trade_count']} trades")
    # ... calculation logic ...
    return portfolio_values

@task
def compute_nav(portfolio_values: pd.DataFrame) -> dict:
    \"\"\"Compute NAV for all funds\"\"\"
    nav_results = {}
    for fund_id in portfolio_values['fund_id'].unique():
        fund_data = portfolio_values[portfolio_values['fund_id'] == fund_id]
        total_value = fund_data['market_value'].sum()
        shares_outstanding = fund_data['shares_outstanding'].iloc[0]
        nav = total_value / shares_outstanding
        nav_results[fund_id] = nav
    
    print(f"📊 Computed NAV for {len(nav_results)} funds")
    return nav_results

@task
def generate_reports(nav_data: dict) -> str:
    \"\"\"Generate P&L and regulatory reports\"\"\"
    # Implementation of report generation
    report_path = f"reports/daily_pnl_{datetime.now().strftime('%Y%m%d')}.pdf"
    print(f"📄 Generated reports: {report_path}")
    return report_path

@flow(name="Daily Fund Accounting Pipeline")
def daily_fund_accounting_flow(processing_date: str = None):
    \"\"\"Main flow for daily fund accounting operations\"\"\"
    
    if processing_date is None:
        processing_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"🚀 Starting daily fund accounting for {processing_date}")
    
    # Parallel data fetching
    market_data = fetch_market_data(processing_date)
    trades = validate_trades(f"trades_{processing_date}.csv")
    
    # Sequential processing
    portfolio_values = calculate_portfolio_values(market_data, trades)
    nav_data = compute_nav(portfolio_values)
    report_path = generate_reports(nav_data)
    
    print(f"✅ Pipeline completed successfully!")
    return {
        "status": "completed",
        "nav_count": len(nav_data),
        "report_path": report_path
    }

# Schedule the flow
if __name__ == "__main__":
    # Run immediately
    result = daily_fund_accounting_flow()
    
    # Or deploy with scheduling
    daily_fund_accounting_flow.deploy(
        name="fund-accounting-daily",
        schedule="0 6 * * 1-5",  # 6 AM weekdays
        tags=["fund-accounting", "daily", "production"]
    )
        """, language="python")

with mlops_tabs[4]:
    st.subheader("📦 Model Deployment in Fund Operations")
    
    st.markdown("""
    ### Model Deployment for Fund Accounting
    
    Deploy machine learning models for real-time predictions in fund operations,
    from fraud detection APIs to NAV prediction services and regulatory compliance automation.
    """)
    
    # Model deployment dashboard
    st.markdown("#### 🚀 Current Model Deployments")
    
    # Mock deployed models data
    deployed_models = [
        {
            "Model Name": "Fraud Detection API",
            "Version": "v2.1.3",
            "Status": "🟢 Active",
            "Environment": "Production",
            "Requests/Day": "~12,500",
            "Latency": "45ms",
            "Accuracy": "94.2%",
            "Last Updated": "2024-01-02",
            "Endpoint": "/api/v2/fraud-detection"
        },
        {
            "Model Name": "NAV Prediction Service",
            "Version": "v1.8.1",
            "Status": "🟢 Active", 
            "Environment": "Production",
            "Requests/Day": "~850",
            "Latency": "120ms",
            "Accuracy": "89.7%",
            "Last Updated": "2023-12-28",
            "Endpoint": "/api/v1/nav-predict"
        },
        {
            "Model Name": "Redemption Forecast",
            "Version": "v3.0.0",
            "Status": "🟡 Staging",
            "Environment": "Staging",
            "Requests/Day": "~200",
            "Latency": "85ms",
            "Accuracy": "91.8%",
            "Last Updated": "2024-01-04",
            "Endpoint": "/api/v3/redemption-forecast"
        },
        {
            "Model Name": "Portfolio Risk Classifier",
            "Version": "v1.2.0",
            "Status": "🔴 Failed",
            "Environment": "Production",
            "Requests/Day": "0",
            "Latency": "N/A",
            "Accuracy": "N/A",
            "Last Updated": "2024-01-03",
            "Endpoint": "/api/v1/risk-classify"
        }
    ]
    
    models_df = pd.DataFrame(deployed_models)
    
    # Color code status column
    def style_model_status(val):
        if "Active" in val:
            return 'background-color: #d4edda; color: #155724'
        elif "Staging" in val:
            return 'background-color: #fff3cd; color: #856404'
        elif "Failed" in val:
            return 'background-color: #f8d7da; color: #721c24'
        return ''
    
    styled_models = models_df.style.applymap(style_model_status, subset=['Status'])
    st.dataframe(styled_models, use_container_width=True)
    
    # Deployment metrics
    st.markdown("---")
    st.markdown("#### 📊 Deployment Metrics")
    
    deploy_col1, deploy_col2, deploy_col3, deploy_col4 = st.columns(4)
    
    active_models = len([m for m in deployed_models if "Active" in m["Status"]])
    staging_models = len([m for m in deployed_models if "Staging" in m["Status"]])
    failed_models = len([m for m in deployed_models if "Failed" in m["Status"]])
    total_requests = sum([int(m["Requests/Day"].replace("~", "").replace(",", "")) for m in deployed_models if m["Requests/Day"] != "0"])
    
    with deploy_col1:
        st.metric("Active Models", active_models, "Production ready")
    with deploy_col2:
        st.metric("Models in Staging", staging_models, "Testing phase")
    with deploy_col3:
        st.metric("Failed Deployments", failed_models, "⚠️ Needs attention")
    with deploy_col4:
        st.metric("Daily API Calls", f"{total_requests:,}", "Across all models")
    
    # Model performance monitoring
    if PLOTLY_AVAILABLE:
        st.markdown("#### 📈 Model Performance Trends")
        
        # Generate mock performance data
        performance_dates = [datetime.date.today() - datetime.timedelta(days=i) for i in range(14, 0, -1)]
        
        perf_data = {
            'Date': performance_dates * 2,  # Two models
            'Model': ['Fraud Detection API'] * 14 + ['NAV Prediction Service'] * 14,
            'Accuracy': [random.uniform(0.92, 0.96) for _ in range(14)] + [random.uniform(0.85, 0.92) for _ in range(14)],
            'Latency': [random.uniform(35, 55) for _ in range(14)] + [random.uniform(100, 140) for _ in range(14)],
            'Requests': [random.randint(11000, 14000) for _ in range(14)] + [random.randint(700, 1000) for _ in range(14)]
        }
        
        perf_df = pd.DataFrame(perf_data)
        
        perf_viz_col1, perf_viz_col2 = st.columns(2)
        
        with perf_viz_col1:
            fig_accuracy = px.line(
                perf_df,
                x='Date',
                y='Accuracy',
                color='Model',
                title="Model Accuracy Over Time",
                labels={'Accuracy': 'Accuracy (%)', 'Date': 'Date'}
            )
            fig_accuracy.update_layout(height=350, yaxis=dict(range=[0.8, 1.0]))
            st.plotly_chart(fig_accuracy, use_container_width=True, key="model_accuracy")
        
        with perf_viz_col2:
            fig_latency = px.line(
                perf_df,
                x='Date',
                y='Latency',
                color='Model',
                title="API Response Latency",
                labels={'Latency': 'Latency (ms)', 'Date': 'Date'}
            )
            fig_latency.update_layout(height=350)
            st.plotly_chart(fig_latency, use_container_width=True, key="model_latency")
    
    # Deployment architecture
    st.markdown("---")
    st.markdown("#### 🏗️ Deployment Architecture")
    
    arch_col1, arch_col2, arch_col3 = st.columns(3)
    
    with arch_col1:
        st.markdown("""
        **Container Orchestration**
        - Docker containerization
        - Kubernetes deployment
        - Auto-scaling based on load
        - Health check monitoring
        """)
    
    with arch_col2:
        st.markdown("""
        **API Gateway**
        - Load balancing
        - Rate limiting
        - Authentication/authorization
        - Request/response logging
        """)
    
    with arch_col3:
        st.markdown("""
        **Model Serving**
        - REST API endpoints
        - Batch prediction jobs
        - Real-time inference
        - A/B testing support
        """)
    
    # Deployment strategies
    st.markdown("---")
    st.markdown("#### 🎯 Deployment Strategies")
    
    strategy_col1, strategy_col2 = st.columns(2)
    
    with strategy_col1:
        st.markdown("**Current Deployment Pipeline**")
        pipeline_steps = [
            "1. Model training & validation",
            "2. Container image building", 
            "3. Security scanning",
            "4. Staging deployment",
            "5. Integration testing",
            "6. Performance validation",
            "7. Production rollout",
            "8. Monitoring activation"
        ]
        
        for step in pipeline_steps:
            st.markdown(f"✅ {step}")
    
    with strategy_col2:
        st.markdown("**Rollout Strategies Available**")
        st.markdown("""
        - **Blue-Green Deployment**: Zero-downtime updates
        - **Canary Releases**: Gradual traffic shifting
        - **Rolling Updates**: Sequential pod replacement
        - **Feature Flags**: Runtime model switching
        """)
        
        st.markdown("**Current Strategy: Blue-Green**")
        st.progress(0.8)
        st.caption("80% traffic on new version, 20% on previous")
    
    # Model registry integration
    st.markdown("---")
    st.markdown("#### 📚 Model Registry Integration")
    
    registry_col1, registry_col2 = st.columns(2)
    
    with registry_col1:
        st.markdown("**Model Lifecycle Stages**")
        
        lifecycle_data = [
            {"Stage": "Development", "Models": 12, "Description": "In training/testing"},
            {"Stage": "Staging", "Models": 3, "Description": "Ready for validation"},
            {"Stage": "Production", "Models": 2, "Description": "Serving live traffic"},
            {"Stage": "Archived", "Models": 8, "Description": "Deprecated versions"}
        ]
        
        lifecycle_df = pd.DataFrame(lifecycle_data)
        st.dataframe(lifecycle_df, use_container_width=True)
    
    with registry_col2:
        st.markdown("**Model Promotion Workflow**")
        st.markdown("""
        **Promotion Criteria:**
        - ✅ Performance benchmarks met
        - ✅ Security scan passed
        - ✅ Integration tests passed
        - ✅ Load testing completed
        - ✅ Business approval obtained
        
        **Auto-promotion enabled for:**
        - Minor version updates
        - Bug fixes
        - Performance improvements
        """)
    
    # Implementation example
    with st.expander("💻 BentoML Deployment Example"):
        st.code("""
# Example: Deploying a fraud detection model with BentoML

import bentoml
import pandas as pd
from sklearn.ensemble import IsolationForest

# Save trained model to BentoML model store
@bentoml.runner.service_runner()
class FraudDetectionRunner:
    def __init__(self):
        self.model = bentoml.sklearn.load_runner("fraud_detection_model:latest")
    
    @bentoml.handler.dataframe()
    def predict(self, input_df: pd.DataFrame) -> pd.DataFrame:
        # Preprocessing
        features = self.preprocess_transaction_data(input_df)
        
        # Prediction
        fraud_scores = self.model.predict(features)
        anomaly_scores = self.model.decision_function(features)
        
        # Post-processing
        results = pd.DataFrame({
            'transaction_id': input_df['transaction_id'],
            'fraud_probability': fraud_scores,
            'anomaly_score': anomaly_scores,
            'risk_level': self.classify_risk_level(fraud_scores)
        })
        
        return results
    
    def preprocess_transaction_data(self, df):
        # Feature engineering for fraud detection
        features = df.copy()
        
        # Time-based features
        features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        features['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        # Amount-based features
        features['amount_log'] = np.log1p(df['amount'])
        features['amount_zscore'] = (df['amount'] - df['amount'].mean()) / df['amount'].std()
        
        # Historical features (would come from database in practice)
        features['avg_monthly_amount'] = df.groupby('account_id')['amount'].transform('mean')
        features['transaction_count_last_24h'] = df.groupby('account_id').size()
        
        return features[['hour', 'day_of_week', 'amount_log', 'amount_zscore', 
                        'avg_monthly_amount', 'transaction_count_last_24h']]
    
    def classify_risk_level(self, fraud_scores):
        return pd.cut(fraud_scores, 
                     bins=[0, 0.3, 0.7, 1.0], 
                     labels=['Low', 'Medium', 'High'])

# Create BentoML service
fraud_detection_runner = FraudDetectionRunner()
fraud_detection_service = bentoml.Service("fraud_detection_api", runners=[fraud_detection_runner])

# REST API endpoint
@fraud_detection_service.api.route("/predict", methods=["POST"])
def predict_fraud(input_data: pd.DataFrame) -> pd.DataFrame:
    # Validate input
    required_fields = ['transaction_id', 'account_id', 'amount', 'timestamp', 'merchant_id']
    missing_fields = [field for field in required_fields if field not in input_data.columns]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Make prediction
    results = fraud_detection_runner.predict(input_data)
    
    # Log for monitoring
    logger.info(f"Processed {len(input_data)} transactions, "
               f"flagged {sum(results['risk_level'] == 'High')} as high risk")
    
    return results

# Health check endpoint
@fraud_detection_service.api.route("/health", methods=["GET"])
def health_check():
    return {"status": "healthy", "model_version": "v2.1.3", "timestamp": datetime.now().isoformat()}

# Build and deploy
if __name__ == "__main__":
    # Build Bento
    bento = bentoml.build("fraud_detection_api", 
                         service=fraud_detection_service,
                         description="Real-time fraud detection for fund transactions")
    
    # Deploy to production (example with Docker)
    bentoml.container.build(bento, name="fraud-detection-api:v2.1.3")
    
    # Or deploy to Kubernetes
    bentoml.deploy("fraud_detection_k8s", 
                  bento=bento,
                  config_file="k8s_deployment.yaml")
        """, language="python")

with mlops_tabs[5]:
    st.subheader("📈 Production Monitoring & Observability")
    
    st.markdown("""
    ### Comprehensive Model Monitoring
    
    Monitor model performance, data drift, and system health in production to ensure
    reliable fund accounting operations and early detection of issues.
    """)
    
    # Real-time monitoring dashboard
    st.markdown("#### 🔄 Real-Time Model Health")
    
    # Current system status
    health_col1, health_col2, health_col3, health_col4 = st.columns(4)
    
    with health_col1:
        st.metric("System Uptime", "99.97%", "Last 30 days")
    with health_col2:
        st.metric("Active Alerts", "2", "🟡 1 warning, 🔴 1 critical")
    with health_col3:
        st.metric("Avg Response Time", "67ms", "-12ms from yesterday")
    with health_col4:
        st.metric("Error Rate", "0.12%", "+0.03% from last week")
    
    # Alert management
    st.markdown("---")
    st.markdown("#### 🚨 Active Alerts & Incidents")
    
    alerts_monitoring = [
        {
            "Timestamp": "2024-01-05 14:23:00",
            "Severity": "🔴 Critical",
            "Service": "Portfolio Risk Classifier",
            "Alert": "Model prediction accuracy dropped below 85%",
            "Status": "Active",
            "Duration": "2h 15m",
            "Assigned": "ML Engineering Team"
        },
        {
            "Timestamp": "2024-01-05 09:45:00", 
            "Severity": "🟡 Warning",
            "Service": "NAV Prediction Service",
            "Alert": "Data drift detected in input features",
            "Status": "Investigating",
            "Duration": "6h 48m",
            "Assigned": "Data Science Team"
        },
        {
            "Timestamp": "2024-01-04 16:12:00",
            "Severity": "🟢 Info",
            "Service": "Fraud Detection API",
            "Alert": "High traffic volume detected",
            "Status": "Resolved",
            "Duration": "45m",
            "Assigned": "DevOps Team"
        }
    ]
    
    alerts_df = pd.DataFrame(alerts_monitoring)
    
    def style_alert_severity(val):
        if "Critical" in val:
            return 'background-color: #f8d7da; color: #721c24'
        elif "Warning" in val:
            return 'background-color: #fff3cd; color: #856404'
        elif "Info" in val:
            return 'background-color: #d1ecf1; color: #0c5460'
        return ''
    
    styled_alerts = alerts_df.style.applymap(style_alert_severity, subset=['Severity'])
    st.dataframe(styled_alerts, use_container_width=True)
    
    # Performance monitoring charts
    if PLOTLY_AVAILABLE:
        st.markdown("---")
        st.markdown("#### 📊 Performance Monitoring")
        
        # Generate monitoring data
        monitor_dates = pd.date_range('2024-01-01', periods=120, freq='H')
        
        monitoring_data = {
            'timestamp': monitor_dates,
            'response_time': [random.uniform(40, 120) + 20*np.sin(i/24) for i in range(120)],
            'throughput': [random.randint(800, 1200) + 200*np.sin(i/12) for i in range(120)],
            'error_rate': [max(0, random.uniform(0, 0.5) + 0.3*np.sin(i/6)) for i in range(120)],
            'cpu_usage': [random.uniform(30, 80) + 15*np.sin(i/8) for i in range(120)],
            'memory_usage': [random.uniform(40, 85) + 10*np.sin(i/16) for i in range(120)]
        }
        
        monitoring_df = pd.DataFrame(monitoring_data)
        
        # Create monitoring dashboard
        monitor_col1, monitor_col2 = st.columns(2)
        
        with monitor_col1:
            # Response time and throughput
            fig_perf = px.line(
                monitoring_df,
                x='timestamp',
                y=['response_time', 'throughput'],
                title="API Performance Metrics",
                labels={'value': 'Metric Value', 'timestamp': 'Time'},
            )
            
            # Add threshold lines
            fig_perf.add_hline(y=100, line_dash="dash", line_color="red", 
                              annotation_text="Response Time SLA (100ms)")
            
            fig_perf.update_layout(height=400)
            st.plotly_chart(fig_perf, use_container_width=True, key="perf_monitoring")
        
        with monitor_col2:
            # System resources
            fig_resources = px.line(
                monitoring_df,
                x='timestamp',
                y=['cpu_usage', 'memory_usage'],
                title="System Resource Utilization",
                labels={'value': 'Usage (%)', 'timestamp': 'Time'}
            )
            
            fig_resources.add_hline(y=80, line_dash="dash", line_color="orange",
                                   annotation_text="Resource Alert Threshold")
            
            fig_resources.update_layout(height=400, yaxis=dict(range=[0, 100]))
            st.plotly_chart(fig_resources, use_container_width=True, key="resource_monitoring")
        
        # Error rate monitoring
        fig_errors = px.line(
            monitoring_df,
            x='timestamp',
            y='error_rate',
            title="Error Rate Monitoring (Last 5 Days)",
            labels={'error_rate': 'Error Rate (%)', 'timestamp': 'Time'}
        )
        
        fig_errors.add_hline(y=0.5, line_dash="dash", line_color="red",
                            annotation_text="Error Rate SLA Threshold (0.5%)")
        
        fig_errors.update_traces(line_color='red')
        fig_errors.update_layout(height=300, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig_errors, use_container_width=True, key="error_monitoring")
    
    # Data drift monitoring
    st.markdown("---")
    st.markdown("#### 📉 Data Drift Detection")
    
    drift_col1, drift_col2 = st.columns(2)
    
    with drift_col1:
        st.markdown("**Feature Drift Analysis**")
        
        # Mock drift detection results
        drift_features = [
            {"Feature": "transaction_amount", "Drift_Score": 0.15, "Status": "🟢 Normal", "Threshold": 0.3},
            {"Feature": "account_balance", "Drift_Score": 0.45, "Status": "🟡 Warning", "Threshold": 0.3},
            {"Feature": "transaction_frequency", "Drift_Score": 0.8, "Status": "🔴 Critical", "Threshold": 0.3},
            {"Feature": "merchant_category", "Drift_Score": 0.12, "Status": "🟢 Normal", "Threshold": 0.3},
            {"Feature": "time_of_day", "Drift_Score": 0.25, "Status": "🟢 Normal", "Threshold": 0.3}
        ]
        
        drift_df = pd.DataFrame(drift_features)
        
        # Style drift status
        def style_drift_status(val):
            if "Normal" in val:
                return 'background-color: #d4edda; color: #155724'
            elif "Warning" in val:
                return 'background-color: #fff3cd; color: #856404'
            elif "Critical" in val:
                return 'background-color: #f8d7da; color: #721c24'
            return ''
        
        styled_drift = drift_df.style.applymap(style_drift_status, subset=['Status'])
        st.dataframe(styled_drift, use_container_width=True)
    
    with drift_col2:
        st.markdown("**Drift Mitigation Actions**")
        
        mitigation_actions = [
            "🔄 **Auto-retrain triggered** for transaction_frequency feature",
            "📊 **Collect more recent data** for account_balance analysis", 
            "🎯 **Update feature engineering** pipeline scheduled for next week",
            "⚠️ **Alert sent to data science team** for manual review",
            "📝 **Drift report generated** and shared with stakeholders"
        ]
        
        for action in mitigation_actions:
            st.markdown(action)
        
        st.markdown("---")
        st.markdown("**Drift Detection Settings**")
        st.slider("Drift Detection Sensitivity", 0.1, 1.0, 0.3, 0.1, key="drift_sensitivity")
        st.selectbox("Retraining Frequency", ["Daily", "Weekly", "Monthly", "On-demand"], index=1, key="retrain_freq")
        
        if st.button("🔄 Trigger Manual Retraining", key="manual_retrain"):
            st.success("Manual retraining job queued successfully!")
    
    # Model performance degradation
    if PLOTLY_AVAILABLE:
        st.markdown("---")
        st.markdown("#### 📈 Model Performance Degradation Analysis")
        
        # Generate performance degradation data
        perf_dates = pd.date_range('2023-12-01', '2024-01-05', freq='D')
        
        degradation_data = {
            'date': perf_dates,
            'fraud_detection_accuracy': [0.94 - 0.001*i + random.uniform(-0.01, 0.01) for i in range(len(perf_dates))],
            'nav_prediction_mse': [0.15 + 0.002*i + random.uniform(-0.02, 0.02) for i in range(len(perf_dates))],
            'redemption_forecast_mae': [0.08 + 0.001*i + random.uniform(-0.005, 0.005) for i in range(len(perf_dates))]
        }
        
        degradation_df = pd.DataFrame(degradation_data)
        
        # Model performance over time
        fig_degradation = px.line(
            degradation_df,
            x='date',
            y=['fraud_detection_accuracy', 'nav_prediction_mse', 'redemption_forecast_mae'], 
            title="Model Performance Trends (Last 35 Days)",
            labels={'value': 'Performance Metric', 'date': 'Date'}
        )
        
        # Add performance threshold lines
        fig_degradation.add_hline(y=0.90, line_dash="dash", line_color="red",
                                 annotation_text="Minimum Accuracy Threshold")
        
        fig_degradation.update_layout(height=400)
        st.plotly_chart(fig_degradation, use_container_width=True, key="degradation_analysis")
    
    # Observability stack
    st.markdown("---")
    st.markdown("#### 🔍 Observability Stack")
    
    observability_col1, observability_col2, observability_col3 = st.columns(3)
    
    with observability_col1:
        st.markdown("""
        **Metrics Collection**
        - Prometheus for metrics
        - Custom business metrics
        - Model performance KPIs
        - Infrastructure monitoring
        """)
    
    with observability_col2:
        st.markdown("""
        **Logging & Tracing**
        - Centralized log aggregation
        - Distributed tracing
        - Error tracking
        - Audit trail maintenance
        """)
    
    with observability_col3:
        st.markdown("""
        **Alerting & Notification**
        - Smart alert routing
        - Escalation policies
        - Integration with Slack/email
        - On-call management
        """)
    
    # Compliance and audit
    st.markdown("---")
    st.markdown("#### 📋 Compliance & Audit Trail")
    
    compliance_col1, compliance_col2 = st.columns(2)
    
    with compliance_col1:
        st.markdown("**Regulatory Requirements**")
        
        compliance_items = [
            {"Requirement": "Model Risk Management", "Status": "✅ Compliant", "Last Audit": "2023-12-15"},
            {"Requirement": "Data Lineage Tracking", "Status": "✅ Compliant", "Last Audit": "2024-01-02"},
            {"Requirement": "Bias Testing", "Status": "🟡 In Progress", "Last Audit": "2023-11-20"},
            {"Requirement": "Explainability Reports", "Status": "✅ Compliant", "Last Audit": "2024-01-01"},
            {"Requirement": "Change Management", "Status": "✅ Compliant", "Last Audit": "2023-12-28"}
        ]
        
        compliance_df = pd.DataFrame(compliance_items)
        st.dataframe(compliance_df, use_container_width=True)
    
    with compliance_col2:
        st.markdown("**Audit Trail Summary**")
        st.markdown("""
        **Last 30 Days:**
        - 🔄 47 model predictions logged
        - 📊 12 retraining events recorded  
        - 🚨 8 alerts generated and resolved
        - 📝 3 compliance reports generated
        - 👥 15 user access events logged
        
        **Retention Policy:**
        - Prediction logs: 2 years
        - Training data: 7 years
        - Model artifacts: 5 years
        - Audit logs: 10 years
        """)
        
        if st.button("📋 Generate Compliance Report", key="compliance_report"):
            st.success("Compliance report generated and sent to regulatory team!")
    
    # Implementation example
    with st.expander("💻 Monitoring Implementation Example"):
        st.code("""
# Example: Production model monitoring with custom metrics

import time
import logging
from datetime import datetime
from prometheus_client import Counter, Histogram, Gauge
import numpy as np

# Prometheus metrics
prediction_counter = Counter('model_predictions_total', 'Total predictions made', ['model_name', 'version'])
prediction_latency = Histogram('model_prediction_latency_seconds', 'Prediction latency', ['model_name'])
model_accuracy = Gauge('model_accuracy_score', 'Current model accuracy', ['model_name', 'version'])
data_drift_score = Gauge('model_data_drift_score', 'Data drift detection score', ['model_name', 'feature'])

class ModelMonitor:
    def __init__(self, model_name, model_version, drift_threshold=0.3):
        self.model_name = model_name
        self.model_version = model_version
        self.drift_threshold = drift_threshold
        self.logger = logging.getLogger(f"{model_name}_monitor")
        
        # Performance tracking
        self.recent_predictions = []
        self.recent_actuals = []
        self.baseline_stats = {}
        
    def log_prediction(self, input_data, prediction, actual=None, latency=None):
        \"\"\"Log prediction for monitoring and analysis\"\"\"
        
        # Update Prometheus metrics
        prediction_counter.labels(
            model_name=self.model_name, 
            version=self.model_version
        ).inc()
        
        if latency:
            prediction_latency.labels(model_name=self.model_name).observe(latency)
        
        # Store for performance calculation
        self.recent_predictions.append(prediction)
        if actual is not None:
            self.recent_actuals.append(actual)
        
        # Log structured data for audit trail
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'model_name': self.model_name,
            'model_version': self.model_version,
            'input_features': input_data.to_dict() if hasattr(input_data, 'to_dict') else input_data,
            'prediction': prediction,
            'actual': actual,
            'latency_ms': latency * 1000 if latency else None
        }
        
        self.logger.info(f"Prediction logged: {log_entry}")
        
        # Trigger drift detection periodically
        if len(self.recent_predictions) % 100 == 0:
            self.check_data_drift(input_data)
    
    def check_data_drift(self, current_data):
        \"\"\"Detect data drift in input features\"\"\"
        
        for feature in current_data.columns:
            if feature in self.baseline_stats:
                # Calculate drift using KL divergence approximation
                current_mean = current_data[feature].mean()
                current_std = current_data[feature].std()
                
                baseline_mean = self.baseline_stats[feature]['mean']
                baseline_std = self.baseline_stats[feature]['std']
                
                # Simple drift score (normalized difference in means)
                drift_score = abs(current_mean - baseline_mean) / (baseline_std + 1e-8)
                
                # Update Prometheus metric
                data_drift_score.labels(
                    model_name=self.model_name,
                    feature=feature
                ).set(drift_score)
                
                # Alert if drift exceeds threshold
                if drift_score > self.drift_threshold:
                    self.send_drift_alert(feature, drift_score)
    
    def calculate_performance_metrics(self):
        \"\"\"Calculate and update model performance metrics\"\"\"
        
        if len(self.recent_actuals) >= 10:  # Minimum samples for reliable metrics
            if self.model_name == "fraud_detection":
                # Classification metrics
                accuracy = np.mean(np.array(self.recent_predictions) == np.array(self.recent_actuals))
                model_accuracy.labels(
                    model_name=self.model_name,
                    version=self.model_version
                ).set(accuracy)
                
                # Alert if accuracy drops below threshold
                if accuracy < 0.85:
                    self.send_performance_alert("accuracy", accuracy)
                    
            elif self.model_name == "nav_prediction":
                # Regression metrics
                mse = np.mean((np.array(self.recent_predictions) - np.array(self.recent_actuals))**2)
                
                # Log MSE (can be converted to gauge if needed)
                self.logger.info(f"Current MSE: {mse}")
                
                if mse > 0.2:  # MSE threshold
                    self.send_performance_alert("mse", mse)
    
    def send_drift_alert(self, feature, drift_score):
        \"\"\"Send alert for data drift detection\"\"\"
        
        alert_message = {
            'alert_type': 'data_drift',
            'model_name': self.model_name,
            'feature': feature,
            'drift_score': drift_score,
            'threshold': self.drift_threshold,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high' if drift_score > 0.5 else 'medium'
        }
        
        self.logger.warning(f"Data drift detected: {alert_message}")
        
        # In practice, send to alerting system (PagerDuty, Slack, etc.)
        # send_to_slack(alert_message)
        # create_pagerduty_incident(alert_message)
    
    def send_performance_alert(self, metric_name, metric_value):
        \"\"\"Send alert for model performance degradation\"\"\"
        
        alert_message = {
            'alert_type': 'performance_degradation',
            'model_name': self.model_name,
            'metric': metric_name,
            'value': metric_value,
            'timestamp': datetime.now().isoformat(),
            'severity': 'high'
        }
        
        self.logger.error(f"Performance degradation detected: {alert_message}")
        
        # Trigger automatic retraining if enabled
        if self.should_trigger_retraining(metric_name, metric_value):
            self.trigger_retraining()
    
    def should_trigger_retraining(self, metric_name, metric_value):
        \"\"\"Determine if automatic retraining should be triggered\"\"\"
        
        retraining_thresholds = {
            'accuracy': 0.80,  # Retrain if accuracy drops below 80%
            'mse': 0.25        # Retrain if MSE exceeds 0.25
        }
        
        return metric_value < retraining_thresholds.get(metric_name, float('inf'))
    
    def trigger_retraining(self):
        \"\"\"Trigger model retraining pipeline\"\"\"
        
        self.logger.info(f"Triggering automatic retraining for {self.model_name}")
        
        # In practice, trigger retraining job
        # trigger_mlflow_job(self.model_name)
        # trigger_airflow_dag(f"{self.model_name}_retrain")

# Usage example
fraud_monitor = ModelMonitor("fraud_detection", "v2.1.3")

# In prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    
    input_data = pd.DataFrame(request.json)
    prediction = model.predict(input_data)
    
    latency = time.time() - start_time
    
    # Log prediction for monitoring
    fraud_monitor.log_prediction(input_data, prediction, latency=latency)
    
    return {'prediction': prediction.tolist(), 'model_version': 'v2.1.3'}
        """, language="python")
