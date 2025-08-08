import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import json
from datetime import datetime, timedelta

# Page Configuration
st.set_page_config(
    page_title="Operational Workstreams - Fund Administration",
    page_icon="🏗️",
    layout="wide"
)

# Initialize session state for workstream data
if 'workstream_data' not in st.session_state:
    st.session_state.workstream_data = [
        # NAV Calculation Workstreams
        {'id': 'nav_001', 'name': 'Capstock Processing', 'category': 'NAV Calculation', 'complexity': 6, 'automation': 4, 'risk': 5, 'investment': 2.5, 'completion': 65, 'priority': 'Medium', 'description': 'Processing of capital stock transactions and subscriptions/redemptions'},
        {'id': 'nav_002', 'name': 'NAV Calculation & Publication', 'category': 'NAV Calculation', 'complexity': 8, 'automation': 7, 'risk': 8, 'investment': 4.2, 'completion': 80, 'priority': 'High', 'description': 'Core NAV calculation engine and publication workflow'},
        {'id': 'nav_003', 'name': 'Income Equalisation', 'category': 'NAV Calculation', 'complexity': 7, 'automation': 3, 'risk': 6, 'investment': 1.8, 'completion': 45, 'priority': 'Medium', 'description': 'Income equalisation calculations for unit pricing'},
        {'id': 'nav_004', 'name': 'Swing Pricing', 'category': 'NAV Calculation', 'complexity': 9, 'automation': 2, 'risk': 7, 'investment': 3.1, 'completion': 25, 'priority': 'High', 'description': 'Dynamic swing pricing mechanism implementation'},
        
        # Portfolio Valuation Workstreams
        {'id': 'val_001', 'name': 'Exchange Listed Securities', 'category': 'Portfolio Valuation', 'complexity': 5, 'automation': 8, 'risk': 3, 'investment': 1.5, 'completion': 90, 'priority': 'Low', 'description': 'Automated valuation of exchange-traded securities'},
        {'id': 'val_002', 'name': 'OTC Securities Valuation', 'category': 'Portfolio Valuation', 'complexity': 8, 'automation': 4, 'risk': 8, 'investment': 3.8, 'completion': 55, 'priority': 'High', 'description': 'Over-the-counter securities pricing and valuation'},
        {'id': 'val_003', 'name': 'FX Rates Processing', 'category': 'Portfolio Valuation', 'complexity': 4, 'automation': 9, 'risk': 2, 'investment': 0.8, 'completion': 95, 'priority': 'Low', 'description': 'Foreign exchange rate feeds and processing'},
        
        # Trade Capture Workstreams
        {'id': 'trade_001', 'name': 'Cash Trade Processing', 'category': 'Trade Capture', 'complexity': 5, 'automation': 7, 'risk': 4, 'investment': 2.0, 'completion': 75, 'priority': 'Medium', 'description': 'Cash transaction capture and processing'},
        {'id': 'trade_002', 'name': 'Derivative Trade Capture', 'category': 'Trade Capture', 'complexity': 9, 'automation': 5, 'risk': 9, 'investment': 4.5, 'completion': 40, 'priority': 'High', 'description': 'Complex derivative instrument trade processing'},
        {'id': 'trade_003', 'name': 'FX Hedging Operations', 'category': 'Trade Capture', 'complexity': 7, 'automation': 6, 'risk': 6, 'investment': 2.8, 'completion': 60, 'priority': 'Medium', 'description': 'Foreign exchange hedging trade management'},
        
        # Reconciliation Workstreams
        {'id': 'recon_001', 'name': 'Stock Reconciliation', 'category': 'Reconciliation', 'complexity': 6, 'automation': 5, 'risk': 6, 'investment': 2.2, 'completion': 70, 'priority': 'Medium', 'description': 'Stock position reconciliation with custodians'},
        {'id': 'recon_002', 'name': 'Cash Reconciliation', 'category': 'Reconciliation', 'complexity': 5, 'automation': 6, 'risk': 5, 'investment': 1.9, 'completion': 80, 'priority': 'Medium', 'description': 'Cash balance reconciliation across accounts'},
        {'id': 'recon_003', 'name': 'Fund Reconciliation', 'category': 'Reconciliation', 'complexity': 8, 'automation': 3, 'risk': 8, 'investment': 3.5, 'completion': 35, 'priority': 'High', 'description': 'Fund-level reconciliation and break analysis'},
        
        # Corporate Actions
        {'id': 'corp_001', 'name': 'Mandatory Corp Actions', 'category': 'Corporate Actions', 'complexity': 6, 'automation': 7, 'risk': 5, 'investment': 2.1, 'completion': 85, 'priority': 'Low', 'description': 'Automatic processing of mandatory corporate actions'},
        {'id': 'corp_002', 'name': 'Voluntary Corp Actions', 'category': 'Corporate Actions', 'complexity': 8, 'automation': 4, 'risk': 7, 'investment': 3.2, 'completion': 50, 'priority': 'High', 'description': 'Complex voluntary corporate action elections'},
        
        # Expense Management
        {'id': 'exp_001', 'name': 'Performance Fees', 'category': 'Expense Management', 'complexity': 9, 'automation': 3, 'risk': 8, 'investment': 4.0, 'completion': 30, 'priority': 'High', 'description': 'Performance fee calculations and accruals'},
        {'id': 'exp_002', 'name': 'Invoice Management', 'category': 'Expense Management', 'complexity': 4, 'automation': 8, 'risk': 3, 'investment': 1.2, 'completion': 90, 'priority': 'Low', 'description': 'Automated invoice processing and approval'},
        
        # Reporting
        {'id': 'rep_001', 'name': 'Regulatory Reporting', 'category': 'Reporting', 'complexity': 8, 'automation': 5, 'risk': 9, 'investment': 3.8, 'completion': 55, 'priority': 'High', 'description': 'Automated regulatory report generation'},
        {'id': 'rep_002', 'name': 'Client Reporting', 'category': 'Reporting', 'complexity': 6, 'automation': 6, 'risk': 4, 'investment': 2.3, 'completion': 75, 'priority': 'Medium', 'description': 'Customized client report generation'},
    ]

def get_category_color(category):
    """Return color for each workstream category"""
    color_map = {
        'NAV Calculation': '#FF6B6B',
        'Portfolio Valuation': '#4ECDC4', 
        'Trade Capture': '#45B7D1',
        'Reconciliation': '#96CEB4',
        'Corporate Actions': '#FFEAA7',
        'Expense Management': '#DDA0DD',
        'Reporting': '#98D8C8'
    }
    return color_map.get(category, '#B0B0B0')

def create_workstream_periodic_table():
    """Create a periodic table style visualization of workstreams"""
    df = pd.DataFrame(st.session_state.workstream_data)
    
    # Create grid positions based on category and complexity
    category_rows = {
        'NAV Calculation': 1,
        'Portfolio Valuation': 2, 
        'Trade Capture': 3,
        'Reconciliation': 4,
        'Corporate Actions': 5,
        'Expense Management': 6,
        'Reporting': 7
    }
    
    html_elements = []
    
    for _, row in df.iterrows():
        grid_row = category_rows.get(row['category'], 1)
        grid_col = int(row['complexity']) 
        
        color = get_category_color(row['category'])
        
        # Risk-based opacity
        opacity = 0.3 + (row['risk'] / 10) * 0.7
        
        # Priority border
        border_color = {'High': '#FF0000', 'Medium': '#FFA500', 'Low': '#008000'}.get(row['priority'], '#000000')
        
        element_html = f"""
        <div class="workstream-element" style="
            grid-row: {grid_row}; 
            grid-column: {grid_col};
            background-color: {color};
            opacity: {opacity};
            border: 3px solid {border_color};
            border-radius: 8px;
            padding: 10px;
            margin: 2px;
            text-align: center;
            position: relative;
            min-height: 80px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <div style="font-weight: bold; font-size: 12px; color: #333;">
                {row['name'][:20]}{'...' if len(row['name']) > 20 else ''}
            </div>
            <div style="font-size: 10px; color: #666; margin-top: 5px;">
                Automation: {row['automation']}/10
            </div>
            <div style="font-size: 10px; color: #666;">
                Complete: {row['completion']}%
            </div>
            
            <div class="tooltip" style="
                visibility: hidden;
                width: 300px;
                background-color: #555;
                color: white;
                text-align: left;
                border-radius: 6px;
                padding: 10px;
                position: absolute;
                z-index: 1;
                bottom: 125%;
                left: 50%;
                margin-left: -150px;
                opacity: 0;
                transition: opacity 0.3s;
                font-size: 12px;
            ">
                <strong>{row['name']}</strong><br/>
                Category: {row['category']}<br/>
                Complexity: {row['complexity']}/10<br/>
                Automation: {row['automation']}/10<br/>
                Risk: {row['risk']}/10<br/>
                Investment: ${row['investment']:.1f}M<br/>
                Completion: {row['completion']}%<br/>
                Priority: {row['priority']}<br/><br/>
                {row['description']}
            </div>
        </div>
        """
        html_elements.append(element_html)
    
    # Complete HTML with CSS
    full_html = f"""
    <style>
        .periodic-table {{
            display: grid;
            grid-template-columns: repeat(10, 1fr);
            grid-template-rows: repeat(7, 100px);
            gap: 5px;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            margin: 20px 0;
        }}
        
        .workstream-element:hover .tooltip {{
            visibility: visible;
            opacity: 1;
        }}
        
        .workstream-element:hover {{
            transform: scale(1.05);
            transition: transform 0.2s;
            z-index: 10;
        }}
    </style>
    
    <div class="periodic-table">
        {"".join(html_elements)}
    </div>
    
    <div style="margin-top: 20px; padding: 15px; background-color: #f0f0f0; border-radius: 10px;">
        <h4>Legend:</h4>
        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
            <div><span style="color: #FF0000;">⬛</span> High Priority</div>
            <div><span style="color: #FFA500;">⬛</span> Medium Priority</div>
            <div><span style="color: #008000;">⬛</span> Low Priority</div>
        </div>
        <br/>
        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
            <div><span style="background-color: #FF6B6B; padding: 2px 8px; border-radius: 3px;">NAV Calculation</span></div>
            <div><span style="background-color: #4ECDC4; padding: 2px 8px; border-radius: 3px;">Portfolio Valuation</span></div>
            <div><span style="background-color: #45B7D1; padding: 2px 8px; border-radius: 3px;">Trade Capture</span></div>
            <div><span style="background-color: #96CEB4; padding: 2px 8px; border-radius: 3px;">Reconciliation</span></div>
            <div><span style="background-color: #FFEAA7; padding: 2px 8px; border-radius: 3px;">Corporate Actions</span></div>
            <div><span style="background-color: #DDA0DD; padding: 2px 8px; border-radius: 3px;">Expense Management</span></div>
            <div><span style="background-color: #98D8C8; padding: 2px 8px; border-radius: 3px;">Reporting</span></div>
        </div>
    </div>
    """
    
    return full_html

def create_3d_analysis():
    """Create 3D analysis visualization"""
    df = pd.DataFrame(st.session_state.workstream_data)
    
    # Create 3D scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=df['complexity'],
        y=df['automation'],
        z=df['risk'],
        mode='markers+text',
        marker=dict(
            size=df['investment'] * 3,
            color=[get_category_color(cat) for cat in df['category']],
            opacity=0.8,
            line=dict(width=2, color='DarkSlateGray')
        ),
        text=df['name'],
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>' +
                      'Complexity: %{x}<br>' +
                      'Automation: %{y}<br>' +
                      'Risk: %{z}<br>' +
                      'Investment: $%{marker.size:.1f}M<br>' +
                      '<extra></extra>'
    ))
    
    fig.update_layout(
        title="3D Workstream Analysis: Complexity vs Automation vs Risk",
        scene=dict(
            xaxis_title="Complexity Level (1-10)",
            yaxis_title="Automation Level (1-10)", 
            zaxis_title="Risk Level (1-10)",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=800,
        height=600,
        margin=dict(r=20, b=10, l=10, t=40)
    )
    
    return fig

def workstream_management_interface():
    """Create interface for managing workstreams"""
    st.subheader("🛠️ Manage Workstreams - Add/Edit/Delete")
    
    tab1, tab2, tab3 = st.tabs(["➕ Add New", "✏️ Edit Existing", "🗑️ Delete"])
    
    with tab1:
        st.markdown("### Add New Workstream")
        
        col1, col2 = st.columns(2)
        
        with col1:
            new_name = st.text_input("Workstream Name")
            new_category = st.selectbox("Category", [
                'NAV Calculation', 'Portfolio Valuation', 'Trade Capture',
                'Reconciliation', 'Corporate Actions', 'Expense Management', 'Reporting'
            ])
            new_complexity = st.slider("Complexity Level", 1, 10, 5)
            new_automation = st.slider("Automation Level", 1, 10, 5)
            new_risk = st.slider("Risk Level", 1, 10, 5)
        
        with col2:
            new_investment = st.number_input("Investment ($M)", 0.0, 50.0, 1.0, 0.1)
            new_completion = st.slider("Completion %", 0, 100, 50)
            new_priority = st.selectbox("Priority", ['High', 'Medium', 'Low'])
            new_description = st.text_area("Description")
        
        if st.button("Add Workstream", type="primary"):
            if new_name:
                new_id = f"custom_{len(st.session_state.workstream_data):03d}"
                new_workstream = {
                    'id': new_id,
                    'name': new_name,
                    'category': new_category,
                    'complexity': new_complexity,
                    'automation': new_automation,
                    'risk': new_risk,
                    'investment': new_investment,
                    'completion': new_completion,
                    'priority': new_priority,
                    'description': new_description
                }
                st.session_state.workstream_data.append(new_workstream)
                st.success(f"Added workstream: {new_name}")
                st.rerun()
            else:
                st.error("Please provide a workstream name")
    
    with tab2:
        st.markdown("### Edit Existing Workstream")
        
        df = pd.DataFrame(st.session_state.workstream_data)
        workstream_names = [f"{row['name']} ({row['category']})" for _, row in df.iterrows()]
        
        selected_workstream = st.selectbox("Select Workstream to Edit", workstream_names)
        
        if selected_workstream:
            selected_idx = workstream_names.index(selected_workstream)
            workstream = st.session_state.workstream_data[selected_idx]
            
            col1, col2 = st.columns(2)
            
            with col1:
                edit_name = st.text_input("Name", value=workstream['name'])
                edit_category = st.selectbox("Category", [
                    'NAV Calculation', 'Portfolio Valuation', 'Trade Capture',
                    'Reconciliation', 'Corporate Actions', 'Expense Management', 'Reporting'
                ], index=['NAV Calculation', 'Portfolio Valuation', 'Trade Capture',
                         'Reconciliation', 'Corporate Actions', 'Expense Management', 'Reporting'].index(workstream['category']))
                edit_complexity = st.slider("Complexity", 1, 10, workstream['complexity'])
                edit_automation = st.slider("Automation", 1, 10, workstream['automation'])
                edit_risk = st.slider("Risk", 1, 10, workstream['risk'])
            
            with col2:
                edit_investment = st.number_input("Investment ($M)", 0.0, 50.0, workstream['investment'], 0.1)
                edit_completion = st.slider("Completion %", 0, 100, workstream['completion'])
                edit_priority = st.selectbox("Priority", ['High', 'Medium', 'Low'], 
                                           index=['High', 'Medium', 'Low'].index(workstream['priority']))
                edit_description = st.text_area("Description", value=workstream['description'])
            
            if st.button("Update Workstream", type="primary"):
                st.session_state.workstream_data[selected_idx].update({
                    'name': edit_name,
                    'category': edit_category,
                    'complexity': edit_complexity,
                    'automation': edit_automation,
                    'risk': edit_risk,
                    'investment': edit_investment,
                    'completion': edit_completion,
                    'priority': edit_priority,
                    'description': edit_description
                })
                st.success("Workstream updated successfully!")
                st.rerun()
    
    with tab3:
        st.markdown("### Delete Workstream")
        
        df = pd.DataFrame(st.session_state.workstream_data)
        workstream_names = [f"{row['name']} ({row['category']})" for _, row in df.iterrows()]
        
        delete_workstream = st.selectbox("Select Workstream to Delete", workstream_names)
        
        if delete_workstream:
            selected_idx = workstream_names.index(delete_workstream)
            workstream = st.session_state.workstream_data[selected_idx]
            
            st.warning(f"Are you sure you want to delete: **{workstream['name']}**?")
            st.info(f"Category: {workstream['category']} | Investment: ${workstream['investment']}M")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Confirm Delete", type="secondary"):
                    del st.session_state.workstream_data[selected_idx]
                    st.success(f"Deleted workstream: {workstream['name']}")
                    st.rerun()
            with col2:
                if st.button("❌ Cancel"):
                    st.info("Delete cancelled")

# Main App Layout
st.title("🏗️ Operational Workstreams - Fund Administration Periodic Table")

st.markdown("""
### Interactive Workstream Management & 3D Analysis Platform

This application provides a comprehensive view of fund administration operational workstreams with interactive management capabilities and real-time 3D analysis.
""")

# Create tabs for different sections
main_tab1, main_tab2, main_tab3 = st.tabs([
    "🧪 Periodic Table View",
    "📊 3D Analysis", 
    "⚙️ Manage Workstreams"
])

with main_tab1:
    st.markdown("### Fund Administration Workstreams - Periodic Table Layout")
    st.markdown("*Hover over elements to see detailed information. Border colors indicate priority levels.*")
    
    # Display the periodic table
    periodic_table_html = create_workstream_periodic_table()
    st.markdown(periodic_table_html, unsafe_allow_html=True)
    
    # Summary metrics
    df = pd.DataFrame(st.session_state.workstream_data)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Workstreams", len(df))
    
    with col2:
        avg_completion = df['completion'].mean()
        st.metric("Avg Completion", f"{avg_completion:.1f}%")
    
    with col3:
        total_investment = df['investment'].sum()
        st.metric("Total Investment", f"${total_investment:.1f}M")
    
    with col4:
        high_priority = len(df[df['priority'] == 'High'])
        st.metric("High Priority", f"{high_priority} workstreams")

with main_tab2:
    st.markdown("### 3D Workstream Analysis")
    st.markdown("*Interactive 3D visualization showing the relationship between complexity, automation level, and risk. Marker size represents investment amount.*")
    
    # Create and display 3D plot
    fig_3d = create_3d_analysis()
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Analysis insights
    df = pd.DataFrame(st.session_state.workstream_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 Key Insights")
        
        # High complexity, low automation workstreams
        high_complex_low_auto = df[(df['complexity'] >= 7) & (df['automation'] <= 4)]
        if not high_complex_low_auto.empty:
            st.warning("**High Complexity, Low Automation Workstreams:**")
            for _, row in high_complex_low_auto.iterrows():
                st.write(f"• {row['name']} - Investment: ${row['investment']}M")
        
        # High risk workstreams
        high_risk = df[df['risk'] >= 7]
        if not high_risk.empty:
            st.error("**High Risk Workstreams Requiring Attention:**")
            for _, row in high_risk.iterrows():
                st.write(f"• {row['name']} - Risk Level: {row['risk']}/10")
    
    with col2:
        st.markdown("#### 🎯 Recommendations")
        
        # Investment recommendations
        low_investment_high_risk = df[(df['investment'] <= 2.0) & (df['risk'] >= 6)]
        if not low_investment_high_risk.empty:
            st.info("**Consider Increased Investment:**")
            for _, row in low_investment_high_risk.iterrows():
                st.write(f"• {row['name']} - Current: ${row['investment']}M")
        
        # Automation opportunities
        low_automation = df[df['automation'] <= 4].sort_values('complexity', ascending=False)
        if not low_automation.empty:
            st.success("**Automation Opportunities:**")
            for _, row in low_automation.head(3).iterrows():
                st.write(f"• {row['name']} - Automation: {row['automation']}/10")

with main_tab3:
    workstream_management_interface()
    
    # Real-time data export
    st.markdown("---")
    st.subheader("📤 Export Data")
    
    df = pd.DataFrame(st.session_state.workstream_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download as CSV",
            data=csv_data,
            file_name=f"workstreams_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    with col2:
        json_data = df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download as JSON", 
            data=json_data,
            file_name=f"workstreams_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

# Footer
st.markdown("---")
st.markdown("**🚀 Iluvalcar 2.0** - Fund Administration Workstream Management Platform")