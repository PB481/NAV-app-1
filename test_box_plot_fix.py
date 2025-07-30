import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Box Plot Fix Test")

# Sample data
data = {
    'Category': ['A', 'A', 'B', 'B', 'C', 'C'] * 10,
    'Value': [1, 2, 3, 4, 5, 6] * 10
}

df = pd.DataFrame(data)

st.write("Testing box plot with fixed axis rotation:")

try:
    fig_box = px.box(
        df, 
        x='Category', 
        y='Value',
        points="all",
        title="Test Box Plot"
    )
    # Fixed method - use update_layout instead of update_xaxis
    fig_box.update_layout(xaxis_tickangle=45)
    st.plotly_chart(fig_box, use_container_width=True)
    st.success("✅ Box plot rendered successfully!")
    
except Exception as e:
    st.error(f"❌ Error: {str(e)}")

st.write("Raw data:")
st.dataframe(df.head(10))