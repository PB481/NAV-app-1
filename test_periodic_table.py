import streamlit as st
import pandas as pd

# Simple test for the periodic table
st.title("Periodic Table Test")

# Sample data
asset_data = [
    {'Symbol': 'USD', 'Name': 'US Dollar', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'EUR', 'Name': 'Euro', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 2, 'Risk': 2, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'ETF', 'Name': 'Equity ETF', 'Category': 'Fund', 'GridRow': 2, 'GridCol': 6, 'Risk': 5, 'Liquidity': 9, 'OpCost': 1, 'OpRisk': 2},
    {'Symbol': 'Au', 'Name': 'Gold', 'Category': 'Commodity', 'GridRow': 7, 'GridCol': 7, 'Risk': 5, 'Liquidity': 7, 'OpCost': 5, 'OpRisk': 6},
]

df = pd.DataFrame(asset_data)

def get_color_for_value(value, metric):
    if pd.isna(value):
        return "#f0f2f6"
    
    val_norm = (value - 1) / 9.0
    
    if metric == 'Liquidity':
        red = int(255 * (1 - val_norm))
        green = int(255 * val_norm)
        blue = 40
    else:
        red = int(255 * val_norm)
        green = int(255 * (1 - val_norm))
        blue = 40
        
    return f"rgb({red}, {green}, {blue})"

# Test the simple grid
color_metric = st.selectbox("Color by:", ['Risk', 'Liquidity', 'OpCost', 'OpRisk'])

st.write("### Method 1: Simple HTML Grid")

max_row, max_col = int(df['GridRow'].max()), int(df['GridCol'].max())

html = f"""
<div style="width: 100%; overflow-x: auto; padding: 10px;">
    <div style="display: grid; 
                grid-template-columns: repeat({max_col}, 80px); 
                grid-template-rows: repeat({max_row}, 80px); 
                gap: 3px; 
                justify-content: center;
                background: #f5f5f5;
                padding: 20px;
                border-radius: 10px;">
"""

for _, asset in df.iterrows():
    color = get_color_for_value(asset[color_metric], color_metric)
    
    html += f'''
    <div style="grid-row: {asset['GridRow']}; 
                grid-column: {asset['GridCol']}; 
                background-color: {color}; 
                border: 2px solid #333; 
                border-radius: 8px; 
                display: flex; 
                flex-direction: column; 
                justify-content: center; 
                align-items: center; 
                text-align: center; 
                cursor: pointer;
                font-family: Arial, sans-serif;
                transition: transform 0.2s;"
         title="{asset['Name']} - {asset['Category']}">
        <div style="font-size: 16px; font-weight: bold; color: white; text-shadow: 1px 1px 2px black;">
            {asset['Symbol']}
        </div>
        <div style="font-size: 12px; color: white; text-shadow: 1px 1px 2px black;">
            {asset[color_metric]}/10
        </div>
    </div>
    '''

html += """
    </div>
</div>
"""

st.markdown(html, unsafe_allow_html=True)

st.write("### Method 2: Using Plotly (Alternative)")

try:
    import plotly.graph_objects as go
    
    fig = go.Figure()
    
    for _, asset in df.iterrows():
        color_val = asset[color_metric]
        if color_metric == 'Liquidity':
            color = f'rgb({int(255 * (1 - (color_val-1)/9))}, {int(255 * (color_val-1)/9)}, 40)'
        else:
            color = f'rgb({int(255 * (color_val-1)/9)}, {int(255 * (1 - (color_val-1)/9))}, 40)'
        
        fig.add_trace(go.Scatter(
            x=[asset['GridCol']],
            y=[asset['GridRow']],
            mode='markers+text',
            marker=dict(
                size=80,
                color=color,
                line=dict(color='black', width=2)
            ),
            text=asset['Symbol'],
            textfont=dict(size=14, color='white'),
            hovertemplate=f"<b>{asset['Name']}</b><br>" +
                         f"Category: {asset['Category']}<br>" +
                         f"{color_metric}: {asset[color_metric]}/10<extra></extra>",
            showlegend=False,
            name=asset['Symbol']
        ))
    
    fig.update_layout(
        title=f"Periodic Table - Colored by {color_metric}",
        xaxis=dict(showgrid=True, range=[0, max_col + 1]),
        yaxis=dict(showgrid=True, range=[0, max_row + 1], autorange='reversed'),
        height=500,
        plot_bgcolor='white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
except ImportError:
    st.error("Plotly not available")

st.write("### Raw Data for Reference")
st.dataframe(df)