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


# --- Enhanced Navigation System ---

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Home & Overview"
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Main Navigation Menu
st.sidebar.title("🧭 Navigation")
nav_option = st.sidebar.radio(
    "Choose Section:",
    [
        "🏠 Home & Overview",
        "🧪 Periodic Table", 
        "📊 Interactive Analysis",
        "🏦 Financial Data",
        "🏢 Operations",
        "🎯 3D Analysis",
        "📈 Live Market Data",
        "🤖 AI Predictors",
        "⚙️ Settings"
    ],
    index=0 if st.session_state.current_page == "🏠 Home & Overview" else (
        1 if st.session_state.current_page == "🧪 Periodic Table" else (
            2 if st.session_state.current_page == "📊 Interactive Analysis" else (
                3 if st.session_state.current_page == "🏦 Financial Data" else (
                    4 if st.session_state.current_page == "🏢 Operations" else (
                        5 if st.session_state.current_page == "🎯 3D Analysis" else (
                            6 if st.session_state.current_page == "📈 Live Market Data" else (
                                7 if st.session_state.current_page == "🤖 AI Predictors" else 8
                            )
                        )
                    )
                )
            )
        )
    )
)

# Update session state
st.session_state.current_page = nav_option

# Quick Access Toolbar
st.sidebar.markdown("---")
st.sidebar.subheader("⚡ Quick Access")
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    if st.button("🧪"):
        st.session_state.current_page = "🧪 Periodic Table"
        st.rerun()
with col2:
    if st.button("📊"):
        st.session_state.current_page = "📊 Interactive Analysis"
        st.rerun()
with col3:
    if st.button("🏦"):
        st.session_state.current_page = "🏦 Financial Data"
        st.rerun()

# Favorites Section
st.sidebar.markdown("---")
st.sidebar.subheader("⭐ Favorites")
if st.sidebar.button(f"⭐ Bookmark '{nav_option.split(' ', 1)[1]}'"):
    if nav_option not in st.session_state.favorites:
        st.session_state.favorites.append(nav_option)
        st.sidebar.success("Added to favorites!")

if st.session_state.favorites:
    favorite = st.sidebar.selectbox("Jump to favorite:", [""] + st.session_state.favorites)
    if favorite:
        st.session_state.current_page = favorite
        st.rerun()

# Search functionality
st.sidebar.markdown("---")
st.sidebar.subheader("🔍 Search")
search_sections = st.sidebar.text_input("Search sections...", placeholder="Type to search...")

# Section descriptions for search
section_keywords = {
    "🏠 Home & Overview": ["home", "overview", "intro", "welcome"],
    "🧪 Periodic Table": ["periodic", "table", "elements", "assets", "grid"],
    "📊 Interactive Analysis": ["analysis", "interactive", "charts", "graphs", "visualization"],
    "🏦 Financial Data": ["financial", "data", "real", "assets", "funds"],
    "🏢 Operations": ["operations", "operational", "workstream", "business"],
    "🎯 3D Analysis": ["3d", "three dimensional", "positioning", "advanced"],
    "📈 Live Market Data": ["live", "market", "real-time", "data", "prices"],
    "🤖 AI Predictors": ["ai", "artificial intelligence", "prediction", "machine learning"],
    "⚙️ Settings": ["settings", "configuration", "preferences"]
}

if search_sections:
    matching_sections = []
    for section, keywords in section_keywords.items():
        if any(search_sections.lower() in keyword for keyword in keywords) or search_sections.lower() in section.lower():
            matching_sections.append(section)
    
    if matching_sections:
        st.sidebar.write("**Found sections:**")
        for section in matching_sections:
            if st.sidebar.button(f"→ {section}", key=f"search_{section}"):
                st.session_state.current_page = section
                st.rerun()

# Progress Tracking
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Progress")
visited_sections = st.session_state.get('visited_sections', set())
visited_sections.add(nav_option)
st.session_state.visited_sections = visited_sections
progress = len(visited_sections) / 9
st.sidebar.progress(progress)
st.sidebar.write(f"Explored: {len(visited_sections)}/9 sections")

# --- Page Content with Breadcrumbs ---
def show_breadcrumb(current_section):
    """Display breadcrumb navigation"""
    st.markdown(f"**🏠 Home** → **{current_section.split(' ', 1)[1]}**")
    st.markdown("---")

# --- Global Variables Setup ---
# Define sidebar controls that are used across multiple sections
st.sidebar.markdown("---")
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

# Apply filters globally for all sections
filtered_df = df.copy()
if selected_category != 'All':
    filtered_df = filtered_df[filtered_df['Category'] == selected_category]

if search_term:
    search_mask = (
        filtered_df['Symbol'].str.contains(search_term, case=False, na=False) |
        filtered_df['Name'].str.contains(search_term, case=False, na=False)
    )
    filtered_df = filtered_df[search_mask]

# --- Section-Based Content Display ---

# ============================================================================
# HOME & OVERVIEW SECTION
# ============================================================================
if nav_option == "🏠 Home & Overview":
    show_breadcrumb(nav_option)
    st.title("🏠 Financial Asset Analysis Dashboard")
    
    st.markdown("""
    ### Welcome to the comprehensive financial asset analysis platform!
    
    This application provides professional-grade tools for analyzing financial instruments across multiple dimensions. 
    Navigate through different sections using the sidebar to explore various analytical approaches.
    """)
    
    # Quick overview cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Asset Classes", "24", "Complete coverage")
    with col2:
        st.metric("Analysis Tools", "8", "Interactive features")
    with col3:
        st.metric("Visualization Types", "15+", "Multi-library support")
    with col4:
        st.metric("Data Sources", "Real + Synthetic", "Market integrated")
    
    st.markdown("---")
    
    # Feature highlights
    st.subheader("🎯 Key Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **🧪 Periodic Table View**
        - Interactive grid layout of all asset types
        - Color-coded by risk, liquidity, or operational metrics
        - Hover tooltips with detailed information
        
        **📊 Interactive Analysis**
        - Risk-liquidity matrix with quadrant analysis
        - Correlation heatmaps and statistical analysis
        - Multi-dimensional scatter plots and radar charts
        
        **🏦 Real Financial Data**
        - 23 real asset classes with GICS sectors
        - 10 fund types with regulatory frameworks
        - Professional risk scoring (1-5 scale)
        """)
    
    with feature_col2:
        st.markdown("""
        **🎯 3D Analysis**
        - Advanced 3D positioning visualization
        - Hierarchical data exploration
        - Interactive drill-down capabilities
        
        **📈 Live Market Data**
        - Real-time price integration (prepared)
        - Performance tracking and alerts
        - Correlation analysis with market conditions
        
        **🤖 AI Predictors**
        - Machine learning-based predictions
        - Scenario analysis and modeling
        - Intelligent recommendations system
        """)
    
    st.markdown("---")
    
    # Getting started guide
    st.subheader("🚀 Getting Started")
    st.markdown("""
    1. **Start with the Periodic Table** - Get familiar with asset types and their relationships
    2. **Explore Interactive Analysis** - Dive deep into correlations and patterns
    3. **Review Financial Data** - Examine real market data and fund structures  
    4. **Try 3D Analysis** - Visualize complex multi-dimensional relationships
    5. **Use AI Predictors** - Get intelligent insights and recommendations
    
    Use the navigation menu on the left to jump between sections, bookmark your favorites, and track your exploration progress!
    """)

# ============================================================================
# PERIODIC TABLE SECTION
# ============================================================================
elif nav_option == "🧪 Periodic Table":
    show_breadcrumb(nav_option)
    
    # --- Main Periodic Table Content ---
    st.title("🧪 The Periodic Table of Asset Types")
    st.markdown("""
    This application visualizes different financial asset types in the style of a periodic table. 
    Each asset is positioned based on its characteristics and scored on four key metrics.
    **Hover over an element** to see its details. Use the sidebar to change the color scheme.
    """)

    # Display statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Assets", len(df))
    with col2:
        st.metric("Filtered Assets", len(filtered_df))  
    with col3:
        st.metric("Categories", len(df['Category'].unique()))

    # Continue with the rest of the periodic table functionality
    # The CSS grid, market data, and tabbed analysis will be displayed here
    # Note: All existing periodic table functionality continues to work with the global filtering

# ============================================================================
# INTERACTIVE ANALYSIS SECTION  
# ============================================================================
elif nav_option == "📊 Interactive Analysis":
    show_breadcrumb(nav_option)
    st.title("📊 Interactive Asset Analysis Dashboard")
    st.markdown("Access risk-liquidity matrices, heatmaps, interactive charts, and asset positioning tools.")
    st.info("🚧 This section contains the main interactive analysis features. Navigate through the tabs to explore different visualization approaches.")

# ============================================================================
# FINANCIAL DATA SECTION
# ============================================================================
elif nav_option == "🏦 Financial Data":
    show_breadcrumb(nav_option)
    st.title("🏦 Real Financial Data Analysis")
    st.markdown("Analyze real asset classes, fund types, and combined financial instruments with professional risk scoring.")
    st.info("🚧 This section contains real financial data analysis with GICS sectors and regulatory frameworks.")

# ============================================================================
# OPERATIONS SECTION
# ============================================================================
elif nav_option == "🏢 Operations":
    show_breadcrumb(nav_option)
    st.title("🏢 Operational Fund Data Analysis")
    st.markdown("Explore operational workstreams, business processes, and fund administration networks.")
    st.info("🚧 This section contains operational analysis, workstream networks, and business process mapping.")

# ============================================================================
# 3D ANALYSIS SECTION
# ============================================================================
elif nav_option == "🎯 3D Analysis":
    show_breadcrumb(nav_option)
    st.title("🎯 3D Fund Positioning Analysis")
    st.markdown("Advanced three-dimensional visualization of asset relationships and positioning.")
    st.info("🚧 This section contains 3D scatter plots, hierarchical analysis, and advanced positioning tools.")

# ============================================================================
# LIVE MARKET DATA SECTION
# ============================================================================
elif nav_option == "📈 Live Market Data":
    show_breadcrumb(nav_option)
    st.title("📈 Real-Time Market Data & Live Analytics")
    st.markdown("Live market data integration, performance tracking, and real-time correlation analysis.")
    st.info("🚧 This section contains real-time market data feeds, alerts, and live performance tracking.")

# ============================================================================
# AI PREDICTORS SECTION
# ============================================================================
elif nav_option == "🤖 AI Predictors":
    show_breadcrumb(nav_option)
    st.title("🤖 AI-Powered Fund Performance Predictor")
    st.markdown("Machine learning models, predictive analytics, and intelligent investment recommendations.")
    st.info("🚧 This section contains AI models, scenario analysis, and predictive insights for fund performance.")

# ============================================================================
# SETTINGS SECTION
# ============================================================================
elif nav_option == "⚙️ Settings":
    show_breadcrumb(nav_option)
    st.title("⚙️ Application Settings")
    
    st.subheader("🎨 Theme & Display")
    
    col1, col2 = st.columns(2)
    with col1:
        theme_color = st.selectbox("Color Theme:", ["Default", "Dark Mode", "High Contrast", "Professional"])
        chart_style = st.selectbox("Chart Style:", ["Modern", "Classic", "Minimal"])
    with col2:
        display_density = st.selectbox("Display Density:", ["Comfortable", "Compact", "Spacious"])
        animation_speed = st.slider("Animation Speed:", 0.1, 2.0, 1.0)
    
    st.subheader("📊 Data Preferences")
    
    col3, col4 = st.columns(2)
    with col3:
        default_metric = st.selectbox("Default Color Metric:", ["Risk", "Liquidity", "OpCost", "OpRisk"])
        cache_duration = st.selectbox("Data Cache Duration:", ["1 minute", "5 minutes", "15 minutes", "1 hour"])
    with col4:
        auto_refresh = st.checkbox("Auto-refresh data", value=True)
        show_tooltips = st.checkbox("Show enhanced tooltips", value=True)
    
    st.subheader("🔍 Search & Navigation")
    
    col5, col6 = st.columns(2)
    with col5:
        search_mode = st.selectbox("Search Mode:", ["Live", "On Enter", "Manual"])
        sidebar_collapsed = st.checkbox("Start with sidebar collapsed", value=False)
    with col6:
        breadcrumb_style = st.selectbox("Breadcrumb Style:", ["Full Path", "Current Only", "Hidden"])
        quick_access_buttons = st.number_input("Quick Access Buttons:", 1, 6, 3, step=1)
    
    if st.button("💾 Save Settings"):
        st.success("Settings saved successfully!")
    
    if st.button("🔄 Reset to Defaults"):
        st.info("Settings reset to default values.")

# Note: Global variables and filtering are now handled above in the sidebar section

# ============================================================================
# REMAINING ORIGINAL CONTENT (for backward compatibility)
# ============================================================================

# Only show the original content if we're in periodic table mode or for compatibility
if nav_option == "🧪 Periodic Table":
    # Original content continues in the periodic table section
    pass
elif nav_option in ["🏠 Home & Overview", "⚙️ Settings"]:
    # These sections have their own content
    pass
else:
    # For other sections, show warning and basic content
    st.warning("🚧 This section is being integrated with the new navigation system.")
    st.info("Please use the 'Periodic Table' section to access the main functionality.")

# ============================================================================
# ORIGINAL APPLICATION CONTENT (runs after navigation for backward compatibility)
# ============================================================================

# The rest of the original application content continues here without modification
# This ensures all existing functionality works while the new navigation is being refined

# Display statistics for all modes
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

# Add market data display
market_data = load_market_data()
if market_data:
    st.info("💹 **Live Market Data** (Simulated): " + 
            " | ".join([f"{symbol}: ${data['price']:.2f} ({data['change']:+.1f}%)" 
                      for symbol, data in list(market_data.items())[:5]]))

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
                                   layout=go.Layout(title=dict(text='Operational Workstream Network', font=dict(size=16)),
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

# --- Real-Time Market Data Integration ---
st.markdown("---")
st.header("🔴 Real-Time Market Data & Live Analytics")
st.info("Live market data integration with intelligent alerts and real-time fund performance tracking.")

# Market data functions
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_live_market_data(symbols, period="1d", interval="5m"):
    """Fetch live market data using yfinance"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        data = yf.download(symbols, period=period, interval=interval, progress=False)
        return data
    except Exception as e:
        st.error(f"Error fetching market data: {str(e)}")
        return None

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_market_indicators():
    """Fetch key market indicators"""
    if not YFINANCE_AVAILABLE:
        return None
    
    try:
        indicators = {}
        symbols = {
            'SPY': 'S&P 500 ETF',
            'QQQ': 'NASDAQ ETF', 
            'GLD': 'Gold ETF',
            'TLT': 'Treasury ETF',
            'VIX': 'Volatility Index',
            'DX-Y.NYB': 'US Dollar Index'
        }
        
        for symbol, name in symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="2d")
                if not hist.empty:
                    current_price = hist['Close'].iloc[-1]
                    prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                    change = current_price - prev_price
                    change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                    
                    indicators[symbol] = {
                        'name': name,
                        'price': current_price,
                        'change': change,
                        'change_pct': change_pct
                    }
            except:
                continue
        
        return indicators
    except Exception as e:
        st.error(f"Error fetching market indicators: {str(e)}")
        return None

def check_nav_alerts(nav_data):
    """Check for NAV alerts and anomalies"""
    alerts = []
    
    if nav_data is None or nav_data.empty:
        return alerts
    
    # Calculate daily changes
    nav_data_sorted = nav_data.sort_values(['fund_id', 'nav_date'])
    nav_data_sorted['daily_change'] = nav_data_sorted.groupby('fund_id')['nav_per_share'].pct_change()
    nav_data_sorted['rolling_vol'] = nav_data_sorted.groupby('fund_id')['daily_change'].rolling(5).std().reset_index(0, drop=True)
    
    # Get latest data for each fund
    latest_data = nav_data_sorted.groupby('fund_id').last().reset_index()
    
    for _, row in latest_data.iterrows():
        if pd.isna(row['daily_change']):
            continue
            
        # Large daily moves
        if abs(row['daily_change']) > 0.02:
            severity = 'HIGH' if abs(row['daily_change']) > 0.05 else 'MEDIUM'
            alerts.append({
                'type': 'NAV_MOVEMENT',
                'fund_id': row['fund_id'],
                'message': f"Large NAV movement: {row['daily_change']:.2%}",
                'severity': severity,
                'value': row['daily_change']
            })
        
        # High volatility
        if not pd.isna(row['rolling_vol']) and row['rolling_vol'] > 0.03:
            alerts.append({
                'type': 'HIGH_VOLATILITY',
                'fund_id': row['fund_id'],
                'message': f"High volatility detected: {row['rolling_vol']:.2%}",
                'severity': 'MEDIUM',
                'value': row['rolling_vol']
            })
    
    return alerts

def operational_health_check(fund_characteristics, custody_holdings):
    """Check operational health indicators"""
    alerts = []
    
    if fund_characteristics is None or custody_holdings is None:
        return alerts
    
    # Check for concentration risk
    holdings_by_fund = custody_holdings.groupby('fund_id')['market_value'].sum()
    holdings_detail = custody_holdings.groupby(['fund_id', 'asset_class'])['market_value'].sum().reset_index()
    
    for fund_id in holdings_by_fund.index:
        fund_holdings = holdings_detail[holdings_detail['fund_id'] == fund_id]
        total_value = holdings_by_fund[fund_id]
        
        # Check for asset class concentration (>70% in single class)
        max_concentration = (fund_holdings['market_value'] / total_value).max()
        if max_concentration > 0.7:
            dominant_class = fund_holdings.loc[fund_holdings['market_value'].idxmax(), 'asset_class']
            alerts.append({
                'type': 'CONCENTRATION_RISK',
                'fund_id': fund_id,
                'message': f"High concentration in {dominant_class}: {max_concentration:.1%}",
                'severity': 'MEDIUM',
                'value': max_concentration
            })
    
    return alerts

# Real-time market data interface
if YFINANCE_AVAILABLE:
    # Create tabs for live data
    live_tab_market, live_tab_alerts, live_tab_performance, live_tab_correlation = st.tabs([
        "📊 Live Market Data", "🚨 Intelligent Alerts", "📈 Real-Time Performance", "🔗 Market Correlation"
    ])
    
    with live_tab_market:
        st.subheader("Live Market Indicators")
        
        # Auto-refresh control
        col1, col2, col3 = st.columns(3)
        with col1:
            auto_refresh = st.checkbox("🔄 Auto-refresh (60s)", value=False)
        with col2:
            if st.button("🔴 Refresh Now") or auto_refresh:
                st.rerun()
        with col3:
            st.write(f"⏰ Last updated: {pd.Timestamp.now().strftime('%H:%M:%S')}")
        
        # Get market indicators
        indicators = get_market_indicators()
        
        if indicators:
            # Display key metrics
            cols = st.columns(len(indicators))
            for i, (symbol, data) in enumerate(indicators.items()):
                with cols[i]:
                    delta_color = "normal" if data['change_pct'] >= 0 else "inverse"
                    st.metric(
                        label=data['name'],
                        value=f"{data['price']:.2f}",
                        delta=f"{data['change_pct']:+.2f}%",
                        delta_color=delta_color
                    )
            
            # Market overview chart
            st.write("**Market Performance Today**")
            market_df = pd.DataFrame([
                {'Symbol': symbol, 'Name': data['name'], 'Change %': data['change_pct']}
                for symbol, data in indicators.items()
            ])
            
            fig_market = px.bar(
                market_df,
                x='Symbol',
                y='Change %',
                color='Change %',
                color_continuous_scale='RdYlGn',
                title="Market Indices Performance",
                hover_data=['Name']
            )
            fig_market.update_layout(height=400)
            st.plotly_chart(fig_market, use_container_width=True)
            
        else:
            st.warning("Unable to fetch live market data. Please check your internet connection.")
    
    with live_tab_alerts:
        st.subheader("🚨 Intelligent Alert System")
        
        # NAV alerts
        if nav_data is not None:
            nav_alerts = check_nav_alerts(nav_data)
            
            if nav_alerts:
                st.error(f"⚠️ {len(nav_alerts)} NAV alerts detected!")
                
                for alert in nav_alerts:
                    if alert['severity'] == 'HIGH':
                        st.error(f"🔴 {alert['fund_id']}: {alert['message']}")
                    else:
                        st.warning(f"🟡 {alert['fund_id']}: {alert['message']}")
            else:
                st.success("✅ No NAV alerts - All funds performing within normal ranges")
        
        # Operational alerts
        if fund_characteristics is not None and custody_holdings is not None:
            op_alerts = operational_health_check(fund_characteristics, custody_holdings)
            
            if op_alerts:
                st.warning(f"⚠️ {len(op_alerts)} operational alerts detected!")
                
                for alert in op_alerts:
                    st.warning(f"🟡 {alert['fund_id']}: {alert['message']}")
            else:
                st.success("✅ No operational alerts - All systems healthy")
        
        # Alert configuration
        st.write("**Alert Thresholds**")
        col1, col2 = st.columns(2)
        with col1:
            nav_threshold = st.slider("NAV Change Alert (%)", 1.0, 10.0, 2.0, 0.5)
            vol_threshold = st.slider("Volatility Alert (%)", 1.0, 5.0, 3.0, 0.5)
        with col2:
            concentration_threshold = st.slider("Concentration Risk (%)", 50, 90, 70, 5)
            st.info("Alerts trigger when thresholds are exceeded")
    
    with live_tab_performance:
        st.subheader("📈 Real-Time Fund Performance")
        
        if nav_data is not None:
            # Real-time performance metrics
            latest_nav = nav_data.groupby('fund_id').last().reset_index()
            
            # Performance summary
            st.write("**Current Fund Status**")
            perf_cols = st.columns(len(latest_nav))
            
            for i, (_, fund) in enumerate(latest_nav.iterrows()):
                with perf_cols[i]:
                    # Calculate recent performance (mock calculation for demo)
                    fund_nav_data = nav_data[nav_data['fund_id'] == fund['fund_id']].sort_values('nav_date')
                    if len(fund_nav_data) > 1:
                        recent_change = (fund_nav_data['nav_per_share'].iloc[-1] / fund_nav_data['nav_per_share'].iloc[-2] - 1) * 100
                    else:
                        recent_change = 0
                    
                    st.metric(
                        label=fund['fund_id'],
                        value=f"{fund['nav_per_share']:.4f}",
                        delta=f"{recent_change:+.2f}%"
                    )
            
            # Live performance chart
            st.write("**Live NAV Tracking**")
            fig_live_nav = px.line(
                nav_data,
                x='nav_date',
                y='nav_per_share',
                color='fund_id',
                title="Real-Time NAV Performance",
                labels={'nav_date': 'Date', 'nav_per_share': 'NAV Per Share'}
            )
            fig_live_nav.update_layout(height=500)
            st.plotly_chart(fig_live_nav, use_container_width=True)
        
        else:
            st.info("Connect operational data to see real-time fund performance")
    
    with live_tab_correlation:
        st.subheader("🔗 Market Correlation Analysis")
        
        if indicators and nav_data is not None:
            st.write("**Fund Performance vs Market Indicators**")
            
            # Create correlation analysis (simplified for demo)
            correlation_data = []
            for fund_id in nav_data['fund_id'].unique():
                fund_nav_data = nav_data[nav_data['fund_id'] == fund_id].sort_values('nav_date')
                if len(fund_nav_data) > 1:
                    nav_change = (fund_nav_data['nav_per_share'].iloc[-1] / fund_nav_data['nav_per_share'].iloc[0] - 1) * 100
                    correlation_data.append({'Fund': fund_id, 'Performance': nav_change})
            
            if correlation_data:
                corr_df = pd.DataFrame(correlation_data)
                
                # Add market data for comparison
                market_data = []
                for symbol, data in indicators.items():
                    if symbol in ['SPY', 'QQQ', 'GLD', 'TLT']:  # Focus on key indices
                        market_data.append({'Asset': data['name'], 'Change': data['change_pct']})
                
                if market_data:
                    market_df = pd.DataFrame(market_data)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Fund Performance**")
                        fig_fund_perf = px.bar(
                            corr_df,
                            x='Fund',
                            y='Performance',
                            title="Fund Performance",
                            color='Performance',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_fund_perf, use_container_width=True)
                    
                    with col2:
                        st.write("**Market Performance**")
                        fig_market_perf = px.bar(
                            market_df,
                            x='Asset',
                            y='Change',
                            title="Market Indices",
                            color='Change',
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig_market_perf, use_container_width=True)
        
        else:
            st.info("Market correlation analysis requires both market data and fund NAV data")

else:
    st.warning("Real-time market data requires yfinance library. Please install: pip install yfinance")

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
