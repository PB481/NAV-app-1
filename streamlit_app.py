import streamlit as st
import pandas as pd
import math
import json

# Conditional imports for visualization libraries
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

# --- Page Configuration ---
st.set_page_config(
    page_title="Periodic Table of Asset Types",
    page_icon="📊",
    layout="wide", # Use the full screen width
)


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


# --- App UI ---

st.title("🧪 The Periodic Table of Asset Types")
st.markdown("""
This application visualizes different financial asset types in the style of a periodic table. 
Each asset is positioned based on its characteristics and scored on four key metrics.
**Hover over an element** to see its details. Use the sidebar to change the color scheme.
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

# --- Generate the Periodic Table using Native Streamlit Components ---

# Determine the max number of rows and columns needed for the grid
max_col = df['GridCol'].max()
max_row = df['GridRow'].max()

# Create the periodic table using Streamlit columns and containers
st.subheader("🧪 The Periodic Table of Asset Types")

# Display assets in a simplified grid layout to avoid st.columns() issues
# Group assets by category for reliable display
display_assets = df.copy()

# Apply filters
if selected_category != 'All':
    display_assets = display_assets[display_assets['Category'] == selected_category]
if search_term:
    search_mask = (
        display_assets['Symbol'].str.contains(search_term, case=False, na=False) |
        display_assets['Name'].str.contains(search_term, case=False, na=False)
    )
    display_assets = display_assets[search_mask]

# Group by category and display in manageable chunks
categories = sorted(display_assets['Category'].unique())

for category in categories:
    category_assets = display_assets[display_assets['Category'] == category].sort_values(['GridRow', 'GridCol'])
    
    if len(category_assets) > 0:
        st.markdown(f"### {category}")
        
        # Display assets in rows of 5 (safe column count)
        assets_list = category_assets.to_dict('records')
        
        for i in range(0, len(assets_list), 5):
            row_assets = assets_list[i:i+5]
            if len(row_assets) > 0:  # Additional safety check
                cols = st.columns(len(row_assets))
                
                for idx, asset in enumerate(row_assets):
                    color = get_color_for_value(asset[color_metric], color_metric)
                    
                    # Check if this asset should be highlighted or dimmed based on filters
                    is_filtered_out = (
                        (selected_category != 'All' and asset['Category'] != selected_category) or
                        (search_term and not (
                            search_term.lower() in asset['Symbol'].lower() or 
                            search_term.lower() in asset['Name'].lower()
                        ))
                    )
                    
                    with cols[idx]:
                        # Create the asset element with styling matching workstreams
                        opacity_style = "opacity: 0.3;" if is_filtered_out else "opacity: 1.0;"
                        
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
                            {opacity_style}
                        ">
                            <strong style="font-size: 1.4em; margin-bottom: 4px;">{asset['Symbol']}</strong><br/>
                            <small style="font-size: 0.6em; line-height: 1.1;">{color_metric}: {asset[color_metric]}/10</small>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show detailed information in an expander
                        with st.expander(f"📊 {asset['Symbol']} Details"):
                            st.write(f"**Name:** {asset['Name']}")
                            st.write(f"**Category:** {asset['Category']}")
                            st.write(f"**Position:** Row {asset['GridRow']}, Col {asset['GridCol']}")
                            
                            # Metrics with visual indicators  
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Risk", f"{asset['Risk']}/10", help="Market/Credit Risk")
                                st.metric("Op Cost", f"{asset['OpCost']}/10", help="Operational Cost")
                            with col2:
                                st.metric("Liquidity", f"{asset['Liquidity']}/10", help="Liquidity Level")
                                st.metric("Op Risk", f"{asset['OpRisk']}/10", help="Operational Risk")
        
        st.markdown("---")  # Separator between categories

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
                
                project_indicator = "🏗️" if related_projects and show_projects else ""
                
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

# --- Editable Capital Portfolio ---
st.markdown("---")
st.subheader("💰 Editable Capital Portfolio - USD 26M (2025)")

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
            title="Workstream Dependencies Network<br><sub>Circles = Workstreams, Diamonds = Shared Applications</sub>",
            titlefont_size=16,
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
