import streamlit as st
import pandas as pd
import math

# --- Page Configuration ---
st.set_page_config(
    page_title="Periodic Table of Asset Types",
    page_icon="📊",
    layout="wide", # Use the full screen width
)

# Add debug information for troubleshooting
st.sidebar.markdown("---")
st.sidebar.header("🔧 Debug Info")
st.sidebar.write(f"Streamlit version: {st.__version__}")
st.sidebar.write(f"Pandas version: {pd.__version__}")

# Add option to force fallback mode
force_fallback = st.sidebar.checkbox("Force Fallback Mode", help="Use native Streamlit components instead of HTML")

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

# --- Generate the Periodic Table using HTML and CSS ---

# We use st.markdown with unsafe_allow_html=True to render custom HTML.
# This gives us full control over the layout and styling.

# Determine the max number of rows and columns needed for the grid
max_col = df['GridCol'].max()
max_row = df['GridRow'].max()

# Try HTML rendering first, with fallback to Streamlit native components
if not force_fallback:
    try:
        # CSS for the grid container and the individual elements
        # This is where the magic happens for the layout and hover effects.
        html_string = f"""
    <style>
        .periodic-table-container {{
            display: grid;
            grid-template-columns: repeat({max_col}, minmax(80px, 1fr));
            grid-template-rows: repeat({max_row}, auto);
            gap: 8px;
            width: 100%;
            margin: 20px 0;
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 10px;
        }}
        
        .asset-element {{
            border: 2px solid #333;
            border-radius: 8px;
            padding: 8px;
            text-align: center;
            position: relative;
            cursor: pointer;
            transition: all 0.3s ease;
            height: 90px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: 'Arial', sans-serif;
        }}
        
        .asset-element.dimmed {{
            opacity: 0.3;
            transform: scale(0.95);
        }}
        
        .asset-element:hover {{
            transform: scale(1.05);
            box-shadow: 0 6px 25px rgba(0,0,0,0.3);
            z-index: 100;
            border-color: #fff;
        }}
        
        .asset-symbol {{
            font-size: 1.4em;
            font-weight: bold;
            color: #000;
            margin-bottom: 2px;
        }}
        
        .asset-name {{
            font-size: 0.65em;
            color: #333;
            line-height: 1.2;
        }}
        
        .tooltip-content {{
            visibility: hidden;
            position: absolute;
            bottom: 110%;
            left: 50%;
            transform: translateX(-50%);
            background-color: rgba(0,0,0,0.9);
            color: white;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.8em;
            white-space: nowrap;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.3s;
            min-width: 200px;
        }}
        
        .asset-element:hover .tooltip-content {{
            visibility: visible;
            opacity: 1;
        }}
        
        .tooltip-content::after {{
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: rgba(0,0,0,0.9) transparent transparent transparent;
        }}
        
        /* Responsive adjustments */
        @media (max-width: 1200px) {{
            .periodic-table-container {{
                grid-template-columns: repeat({min(max_col//2, 8)}, minmax(70px, 1fr));
                gap: 6px;
            }}
            .asset-element {{
                height: 75px;
                padding: 6px;
            }}
        }}
        
        @media (max-width: 768px) {{
            .periodic-table-container {{
                grid-template-columns: repeat({min(max_col//3, 6)}, minmax(60px, 1fr));
                gap: 4px;
            }}
            .asset-element {{
                height: 60px;
                padding: 4px;
            }}
            .asset-symbol {{
                font-size: 1.1em;
            }}
            .asset-name {{
                font-size: 0.55em;
            }}
        }}
    </style>
    
    <div class="periodic-table-container">
    """

    # Loop through the DataFrame and create an HTML element for each asset
    for _, asset in df.iterrows():
        color = get_color_for_value(asset[color_metric], color_metric)
        
        # Check if this asset should be highlighted or dimmed based on filters
        is_filtered_out = (
            (selected_category != 'All' and asset['Category'] != selected_category) or
            (search_term and not (
                search_term.lower() in asset['Symbol'].lower() or 
                search_term.lower() in asset['Name'].lower()
            ))
        )
        
        css_class = "asset-element dimmed" if is_filtered_out else "asset-element"
        
        # Create tooltip content with better formatting
        tooltip_content = f"""
        <div class='tooltip-content'>
            <strong>{asset['Name']}</strong><br/>
            Category: {asset['Category']}<br/>
            Risk: {asset['Risk']}/10 | Liquidity: {asset['Liquidity']}/10<br/>
            Op Cost: {asset['OpCost']}/10 | Op Risk: {asset['OpRisk']}/10
        </div>
        """

        html_string += f"""
        <div class="{css_class}" style="grid-column: {asset['GridCol']}; grid-row: {asset['GridRow']}; background-color: {color};">
            <div class="asset-symbol">{asset['Symbol']}</div>
            <div class="asset-name">{asset['Name']}</div>
            {tooltip_content}
        </div>
        """

    html_string += "</div>"

    # Render the final HTML in Streamlit
    st.markdown(html_string, unsafe_allow_html=True)
    
    except Exception as e:
        # Fallback: Use Streamlit native components if HTML fails
        st.error(f"HTML rendering failed: {str(e)}")
        st.warning("Falling back to native Streamlit components...")
        force_fallback = True

if force_fallback:
    # Create fallback using Streamlit columns and containers
    st.subheader("🧪 Asset Overview (Fallback Mode)")
    
    # Group assets by row for better display
    for row in range(1, max_row + 1):
        row_assets = df[df['GridRow'] == row].sort_values('GridCol')
        if len(row_assets) > 0:
            cols = st.columns(len(row_assets))
            for idx, (_, asset) in enumerate(row_assets.iterrows()):
                color = get_color_for_value(asset[color_metric], color_metric)
                
                # Check if this asset should be highlighted or dimmed based on filters
                is_filtered_out = (
                    (selected_category != 'All' and asset['Category'] != selected_category) or
                    (search_term and not (
                        search_term.lower() in asset['Symbol'].lower() or 
                        search_term.lower() in asset['Name'].lower()
                    ))
                )
                
                opacity = "0.3" if is_filtered_out else "1.0"
                
                with cols[idx]:
                    st.markdown(f"""
                    <div style="
                        background-color: {color}; 
                        padding: 10px; 
                        border-radius: 5px; 
                        text-align: center;
                        border: 1px solid #333;
                        margin: 2px;
                        opacity: {opacity};
                        transition: opacity 0.3s ease;
                    ">
                        <strong style="font-size: 1.2em;">{asset['Symbol']}</strong><br/>
                        <small>{asset['Name']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show details on hover (using expander as fallback)
                    with st.expander(f"Details: {asset['Symbol']}"):
                        st.write(f"**Name:** {asset['Name']}")
                        st.write(f"**Category:** {asset['Category']}")
                        st.write(f"**Risk:** {asset['Risk']}/10")
                        st.write(f"**Liquidity:** {asset['Liquidity']}/10")
                        st.write(f"**Op Cost:** {asset['OpCost']}/10")
                        st.write(f"**Op Risk:** {asset['OpRisk']}/10")

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
    
    # Create comparison chart
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Radar chart data
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
    
    styled_df = comparison_df.style.applymap(highlight_diff, subset=['Difference'])
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
