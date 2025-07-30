# 📊 NAV-app-1: Interactive Asset Analysis Dashboard

A comprehensive financial data visualization platform that transforms asset analysis through interactive dashboards, replacing traditional periodic table layouts with modern, professional visualizations.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Features

### **Interactive Asset Analysis Dashboard**
- **Risk-Liquidity Matrix**: Quadrant analysis with interactive bubble charts
- **Correlation Heatmaps**: Statistical analysis of asset metrics relationships
- **Multi-dimensional Charts**: Scatter matrix, parallel coordinates, radar charts, box plots
- **3D Asset Positioning**: Advanced visualization with hierarchical breakdowns
- **Smart Recommendations**: Algorithm-based asset scoring and ranking

### **Professional Financial Visualizations**
- **Quadrant Classification System**:
  - 💚 **Safe Haven**: Low risk, high liquidity assets (USD, EUR, UST)
  - 🟡 **High Risk Liquid**: Active trading assets (ETF, Futures, Options)
  - 🔵 **Conservative Illiquid**: Long-term stable investments (Infrastructure, Real Estate)
  - 🔴 **High Risk Illiquid**: Speculative alternatives (VC, PE, Art, Wine)

### **Advanced Analytics**
- **Portfolio Optimization**: Modern Portfolio Theory with Sharpe ratio maximization
- **Efficient Frontier**: Risk-return analysis with interactive visualizations
- **Market Data Integration**: Live pricing simulation with real API structure
- **Performance Caching**: Optimized data processing and visualization rendering

## 🛠️ Technology Stack

### **Core Framework**
- **[Streamlit](https://streamlit.io/)** `>=1.28.0` - Web application framework
- **[Pandas](https://pandas.pydata.org/)** `>=1.5.0` - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** `>=1.21.0` - Numerical computing

### **Visualization Libraries**
- **[Plotly](https://plotly.com/python/)** `>=5.0.0` - Interactive charts and 3D visualizations
- **[Altair](https://altair-viz.github.io/)** `>=4.2.0` - Grammar of graphics with brush selection
- **[Seaborn](https://seaborn.pydata.org/)** `>=0.11.0` - Statistical data visualization
- **[Matplotlib](https://matplotlib.org/)** `>=3.5.0` - Plotting library and fallback visualizations
- **[Bokeh](https://bokeh.org/)** `>=2.4.0` - Interactive visualization library

### **Financial Analytics**
- **[SciPy](https://scipy.org/)** `>=1.9.0` - Portfolio optimization algorithms
- **[yFinance](https://pypi.org/project/yfinance/)** `>=0.2.18` - Market data integration (prepared)
- **[NetworkX](https://networkx.org/)** `>=2.8.0` - Network analysis for workstream dependencies

## 📋 Installation

### **Prerequisites**
- Python 3.8 or higher
- pip package manager

### **Quick Setup**
```bash
# Clone the repository
git clone <repository-url>
cd NAV-app-1

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run streamlit_app.py
```

### **Manual Installation**
```bash
pip install streamlit>=1.28.0 pandas>=1.5.0 plotly>=5.0.0 altair>=4.2.0 seaborn>=0.11.0 matplotlib>=3.5.0 numpy>=1.21.0 networkx>=2.8.0 scipy>=1.9.0 yfinance>=0.2.18
```

## 🏗️ Architecture

### **Project Structure**
```
NAV-app-1/
├── streamlit_app.py              # Main application
├── data_config.py                # Data configuration module
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── IMPROVEMENTS_SUMMARY.md       # Implementation details
├── VISUALIZATION_UPGRADE_SUMMARY.md  # Visualization features
├── ERROR_FIX_SUMMARY.md          # Technical fixes
├── claude.md                     # Claude Code configuration
├── test_periodic_table.py        # Testing utilities
├── test_box_plot_fix.py          # Validation scripts
└── Workstreams.txt               # Business requirements
```

### **Code Organization**

#### **Main Application (`streamlit_app.py`)**
- **Configuration Section**: Page setup and library imports with graceful fallbacks
- **Data Processing**: Asset data curation and market data integration
- **Helper Functions**: Color calculations, portfolio optimization, visualization creation
- **UI Components**: Sidebar controls, tabbed interface, interactive widgets
- **Visualization Engine**: Multi-library chart rendering with error handling

#### **Data Configuration (`data_config.py`)**
```python
# Asset data structure
ASSET_DATA = [
    {
        'Symbol': 'USD', 'Name': 'US Dollar', 'Category': 'Currency',
        'GridRow': 1, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 
        'OpCost': 1, 'OpRisk': 1
    },
    # ... 24 total assets
]

# Portfolio templates
PORTFOLIO_TEMPLATES = {
    "Conservative": {'USD': 30.0, 'UST': 25.0, ...},
    "Balanced": {'UST': 20.0, 'IGC': 20.0, ...},
    "Aggressive": {'ETF': 20.0, 'HYC': 20.0, ...}
}
```

## 📊 Data Model

### **Asset Attributes**
| Field | Type | Range | Description |
|-------|------|-------|-------------|
| `Symbol` | String | 2-4 chars | Asset identifier (e.g., 'USD', 'ETF') |
| `Name` | String | - | Full asset name |
| `Category` | String | - | Asset classification |
| `GridRow` | Integer | 1-7 | Periodic table row position |
| `GridCol` | Integer | 1-17 | Periodic table column position |
| `Risk` | Integer | 1-10 | Market/Credit risk level |
| `Liquidity` | Integer | 1-10 | Liquidity level |
| `OpCost` | Integer | 1-10 | Operational cost |
| `OpRisk` | Integer | 1-10 | Operational risk |

### **Asset Categories**
- **Currency** (4 assets): USD, EUR, EMC
- **Fixed Income** (5 assets): UST, Bund, IGC, HYC, EMD
- **Fund** (3 assets): ETF, MFt, HF
- **Derivative** (4 assets): Fut, Opt, Sw, CDS
- **Private Equity** (2 assets): PE, VC
- **Real Estate** (1 asset): CRE
- **Infrastructure** (1 asset): Inf
- **Commodity** (2 assets): Au, Oil
- **Collectable** (2 assets): Art, Wn
- **Structured Product** (1 asset): SP

## 🎨 Visualization Features

### **Tab 1: Risk-Liquidity Matrix** 🔬
```python
# Interactive bubble chart with quadrant analysis
fig_bubble = px.scatter(
    data, x='Risk', y='Liquidity', size='OpCost', color=color_metric,
    hover_data=['Name', 'Category', 'OpRisk']
)
# Quadrant lines and annotations automatically added
```

**Features:**
- Interactive bubble chart with hover tooltips
- Automatic quadrant classification and analysis
- Asset positioning insights and recommendations
- Real-time filtering and search integration

### **Tab 2: Heatmaps** 🌡️
```python
# Correlation matrix
corr_matrix = data[['Risk', 'Liquidity', 'OpCost', 'OpRisk']].corr()
fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale='RdBu_r')

# Category analysis
category_metrics = data.groupby('Category')[metrics].mean()
fig_cat = px.imshow(category_metrics.T, text_auto=True, color_continuous_scale='YlOrRd')
```

**Features:**
- Metrics correlation analysis
- Category-wise performance breakdown
- Statistical pattern identification
- Dual implementation (Plotly + Seaborn)

### **Tab 3: Interactive Charts** 📈
**Available Chart Types:**
- **Scatter Matrix**: Multi-dimensional analysis
- **Parallel Coordinates**: Asset profile comparison
- **Radar Charts**: Direct asset comparison (up to 5 assets)
- **Box Plots**: Distribution analysis by category

### **Tab 4: Asset Positioning** 🎯
```python
# 3D scatter plot
fig_3d = px.scatter_3d(
    data, x='Risk', y='Liquidity', z='OpCost', 
    color=color_metric, size='OpRisk'
)

# Sunburst hierarchy
fig_sunburst = px.sunburst(
    data, path=['Category', 'Symbol'], 
    values='Risk', color=color_metric
)
```

### **Alternative Visualization (Altair)** 🎨
```python
# Interactive brush selection
brush = alt.selection_interval()
scatter = alt.Chart(data).add_selection(brush).mark_circle().encode(
    x='Risk:Q', y='Liquidity:Q', color=f'{color_metric}:Q'
)
# Linked category breakdown chart
bars = alt.Chart(data).mark_bar().encode(
    x='count():Q', y='Category:N'
).transform_filter(brush)
```

## 🧮 Financial Analytics

### **Portfolio Optimization**
```python
def calculate_portfolio_optimization(portfolio_data, method="max_sharpe"):
    # Modern Portfolio Theory implementation
    # Constraints: sum(weights) = 1, 0.05 <= weight <= 0.5
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return {
        'optimal_weights': result.x,
        'expected_return': portfolio_return,
        'volatility': portfolio_vol,
        'sharpe_ratio': sharpe_ratio
    }
```

**Methods:**
- **Sharpe Ratio Maximization**: Risk-adjusted return optimization
- **Volatility Minimization**: Risk reduction focus
- **Efficient Frontier**: Risk-return trade-off analysis

### **Smart Scoring Algorithm**
```python
# Composite scoring system
overall_score = (
    liquidity_score * 0.3 +      # 30% weight
    risk_score * 0.3 +           # 30% weight  
    opcost_score * 0.2 +         # 20% weight
    oprisk_score * 0.2           # 20% weight
)
```

### **Market Data Integration**
```python
@st.cache_data(ttl=300)  # 5-minute cache
def load_market_data():
    # Prepared for real API integration
    # tickers = ['SPY', 'GLD', 'TLT', 'DX-Y.NYB', 'CL=F']
    # data = yf.download(tickers, period='1d')
    return simulated_market_data
```

## ⚙️ Configuration

### **Sidebar Controls**
- **Color Metric Selector**: Risk, Liquidity, OpCost, OpRisk
- **Category Filter**: All categories + individual selection
- **Search Functionality**: Real-time asset filtering
- **Color Scale Legend**: Visual metric interpretation guide

### **Caching Configuration**
```python
@st.cache_data(ttl=300)  # Cache for 5 minutes
def expensive_function():
    # Cached operations for performance
    pass
```

### **App Configuration**
```python
st.set_page_config(
    page_title="Periodic Table of Asset Types",
    page_icon="📊", 
    layout="wide"
)
```

## 🔧 Error Handling

### **Graceful Library Fallbacks**
```python
try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.warning("Plotly not available - some visualizations will be disabled")
```

### **Visualization Error Recovery**
```python
try:
    # Primary visualization method
    advanced_chart()
except Exception as e:
    st.error(f"Advanced visualization failed: {str(e)}")
    # Fallback to basic visualization
    basic_chart()
```

## 🚀 Performance Optimizations

### **Caching Strategy**
- **Data Processing**: `@st.cache_data` on expensive calculations
- **Market Data**: TTL-based caching (5 minutes)
- **Color Calculations**: Cached for repeated use
- **Portfolio Optimization**: Cached mathematical operations

### **Responsive Design**
```css
/* Mobile responsiveness built-in */
@media (max-width: 768px) {
    .visualization-container {
        grid-template-columns: 1fr;
        gap: 10px;
    }
}
```

## 📈 Usage Examples

### **Basic Asset Analysis**
```python
# 1. Select color metric in sidebar (e.g., "Risk")
# 2. Filter by category (e.g., "Currency") 
# 3. Search for specific assets (e.g., "USD")
# 4. Navigate through visualization tabs
# 5. Hover over charts for detailed tooltips
```

### **Portfolio Building**
```python
# 1. Go to Portfolio Builder section
# 2. Add assets using dropdown selector
# 3. Adjust weights with sliders
# 4. Apply portfolio templates (Conservative, Balanced, Aggressive)
# 5. Use optimization algorithms for optimal allocation
# 6. View efficient frontier analysis
```

### **Interactive Analysis**
```python
# 1. Use Altair visualization tab
# 2. Select/brush areas in scatter plot
# 3. View filtered category breakdown
# 4. Compare assets using radar charts
# 5. Analyze correlations in heatmaps
```

## 🔍 Advanced Features

### **Workstream Analysis**
- **Network Dependencies**: Visualize operational workstream connections
- **Capital Portfolio Management**: Interactive project tracking ($26M budget)
- **Gap Analysis**: Identified operational improvements
- **Client Change Distribution**: Request type breakdown and analysis

### **Interactive Network Graph**
```python
# NetworkX-powered workstream visualization
G = nx.Graph()
# Nodes: workstreams + shared applications
# Edges: dependencies and connections
# Metrics: centrality, clustering, path analysis
```

### **Editable Components**
- **Capital Projects**: Add, edit, remove projects with budget tracking
- **Client Changes**: Adjustable percentage distributions
- **Portfolio Weights**: Real-time rebalancing tools

## 🧪 Testing

### **Test Files**
- `test_periodic_table.py`: Visualization validation
- `test_box_plot_fix.py`: Error fix verification

### **Running Tests**
```bash
# Test basic functionality
streamlit run test_periodic_table.py --server.port 8503

# Validate error fixes  
streamlit run test_box_plot_fix.py --server.port 8506
```

## 🐛 Troubleshooting

### **Common Issues**

**Problem**: AttributeError with box plots
```python
# ❌ Incorrect
fig.update_xaxis(tickangle=45)

# ✅ Correct  
fig.update_layout(xaxis_tickangle=45)
```

**Problem**: Missing visualization libraries
```python
# The app gracefully handles missing libraries
if not PLOTLY_AVAILABLE:
    st.warning("Plotly not available")
    # Falls back to Seaborn/Matplotlib
```

**Problem**: Performance issues
```python
# Ensure caching is working
@st.cache_data
def expensive_operation():
    # This should only run once per cache period
    pass
```

### **Library Dependencies**
If you encounter import errors, install missing packages:
```bash
pip install plotly altair seaborn matplotlib scipy networkx yfinance
```

## 📊 Data Sources

### **Asset Data**
- **24 Financial Assets** across 10 categories
- **Synthetic Metrics** based on real-world asset characteristics
- **Grid Positioning** designed for periodic table layout
- **Market Data Structure** prepared for live API integration

### **Workstream Data**
- **12 Operational Workstreams** for fund administration
- **100+ Business Processes** mapped to applications
- **Capital Projects** with budget and classification tracking
- **Identified Gaps** for operational improvements

## 🔮 Future Enhancements

### **Ready for Implementation**
1. **Real Market Data**: Replace simulated data with yfinance API calls
2. **User Authentication**: Portfolio saving/loading per user
3. **Advanced Analytics**: Monte Carlo simulations, VaR calculations  
4. **Export Features**: PDF reports, Excel workbooks
5. **Real-time Updates**: WebSocket integration for live data
6. **Mobile App**: PWA conversion for mobile experience

### **Prepared Architecture**
- **Modular Design**: Easy feature additions
- **Scalable Caching**: Ready for production deployment
- **Error Handling**: Robust production-ready code
- **API Structure**: Prepared for external data integration

## 🤝 Contributing

### **Development Setup**
```bash
git clone <repository-url>
cd NAV-app-1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

### **Code Style**
- **PEP 8** compliance for Python code
- **Descriptive naming** for functions and variables
- **Comprehensive error handling** with try-catch blocks
- **Performance considerations** with caching decorators

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit Team** for the excellent web app framework
- **Plotly** for interactive visualization capabilities  
- **Financial Industry** standards for visualization patterns
- **Modern Portfolio Theory** for optimization algorithms

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review error logs in Streamlit Cloud (if deployed)
3. Test with the provided validation scripts
4. Ensure all dependencies are properly installed

---

**Built with ❤️ using Python, Streamlit, and modern data visualization libraries.**

*Transform your financial data analysis with professional-grade interactive dashboards.*