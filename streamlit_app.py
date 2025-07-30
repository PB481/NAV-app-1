import streamlit as st
import pandas as pd
import math
import json

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
        if st.button("🔄 Reset to Original"):
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
    if st.button("🔄 Reset to Original"):
        st.session_state.client_changes = client_change_data.copy()
        st.rerun()
    
    if st.button("⚖️ Redistribute Equally"):
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
    
    # Portfolio composition chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Portfolio Composition by Weight**")
        st.bar_chart(portfolio_df.set_index('Symbol')['Weight'])
    
    with col2:
        st.write("**Category Breakdown**")
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
