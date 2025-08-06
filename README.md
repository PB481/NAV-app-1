# 📊 NAV-app-1: Advanced Fund Accounting & MLOps Platform

A comprehensive financial data visualization and machine learning operations platform that revolutionizes fund accounting through interactive dashboards, advanced analytics, and production-ready MLOps workflows.

![Dashboard Preview](https://img.shields.io/badge/Status-Active-brightgreen) ![Python](https://img.shields.io/badge/Python-3.8+-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red) ![MLOps](https://img.shields.io/badge/MLOps-Enabled-purple) ![License](https://img.shields.io/badge/License-MIT-yellow)

## 🚀 Core Features

### **🤖 MLOps for Fund Accounting** 🆕
Complete machine learning operations platform for financial services:
- **MLflow Model Tracking**: Experiment management and model versioning for fund analytics
- **Great Expectations**: Automated data quality validation for NAV calculations
- **Prefect Workflows**: Production-ready pipeline orchestration for daily operations
- **Model Deployment**: BentoML-powered API serving for fraud detection and predictions
- **Production Monitoring**: Comprehensive observability with drift detection and alerting
- **Regulatory Compliance**: Full audit trail and compliance reporting for financial ML

### **Real Financial Data Analysis** 
- **23 Real Asset Classes**: Professional analysis of actual financial instruments
- **10 Fund Types**: Regulatory framework and structure analysis (UCITS, AIFMD, ELTIF 2.0)
- **Combined Universe**: Integrated view of assets and fund vehicles
- **GICS Sector Analysis**: Industry-specific risk profiling
- **Regulatory Comparison**: Framework-based cost and complexity analysis

### **Interactive Asset Analysis Dashboard**
- **Risk-Liquidity Matrix**: Quadrant analysis with interactive bubble charts
- **Correlation Heatmaps**: Statistical analysis of asset metrics relationships
- **Multi-dimensional Charts**: Scatter matrix, parallel coordinates, radar charts, box plots
- **3D Asset Positioning**: Advanced visualization with hierarchical breakdowns
- **Smart Recommendations**: Algorithm-based asset scoring and ranking

### **Professional Fund Operations**
- **Operational Workstreams**: Interactive periodic table of fund administration processes
- **Portfolio Builder**: Advanced portfolio construction with optimization algorithms
- **Network Analysis**: Workstream dependency mapping with NetworkX visualization
- **Capital Project Management**: Interactive $26M portfolio tracking and analysis
- **AI-Powered Insights**: Machine learning predictions for operational efficiency

### **Advanced Analytics & AI**
- **Portfolio Optimization**: Modern Portfolio Theory with Sharpe ratio maximization
- **Efficient Frontier**: Risk-return analysis with interactive visualizations
- **Predictive Models**: Investor behavior, NAV forecasting, fraud detection
- **Market Data Integration**: Live pricing simulation with real API structure
- **Performance Caching**: Optimized data processing and visualization rendering

## 🛠️ Technology Stack

### **Core Framework**
- **[Streamlit](https://streamlit.io/)** `>=1.28.0` - Web application framework
- **[Pandas](https://pandas.pydata.org/)** `>=1.5.0` - Data manipulation and analysis
- **[NumPy](https://numpy.org/)** `>=1.21.0` - Numerical computing

### **MLOps & Machine Learning** 🆕
- **[MLflow](https://mlflow.org/)** `>=2.0.0` - Model lifecycle management and experiment tracking
- **[Great Expectations](https://greatexpectations.io/)** `>=0.18.0` - Data quality validation and testing
- **[Prefect](https://prefect.io/)** `>=2.14.0` - Workflow orchestration and pipeline automation
- **[BentoML](https://bentoml.org/)** `>=1.1.0` - Model deployment and serving platform
- **[Prometheus Client](https://prometheus.io/)** `>=0.19.0` - Metrics collection and monitoring
- **[scikit-learn](https://scikit-learn.org/)** `>=1.3.0` - Machine learning algorithms and utilities

### **Visualization Libraries**
- **[Plotly](https://plotly.com/python/)** `>=5.0.0` - Interactive charts and 3D visualizations
- **[Altair](https://altair-viz.github.io/)** `>=4.2.0` - Grammar of graphics with brush selection
- **[Seaborn](https://seaborn.pydata.org/)** `>=0.11.0` - Statistical data visualization
- **[Matplotlib](https://matplotlib.org/)** `>=3.5.0` - Plotting library and fallback visualizations
- **[Bokeh](https://bokeh.org/)** `>=2.4.0` - Interactive visualization library

### **Financial Analytics & Operations**
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
# Core dependencies
pip install streamlit>=1.28.0 pandas>=1.5.0 numpy>=1.21.0

# Visualization libraries
pip install plotly>=5.0.0 altair>=4.2.0 seaborn>=0.11.0 matplotlib>=3.5.0

# MLOps and machine learning
pip install mlflow>=2.0.0 great-expectations>=0.18.0 prefect>=2.14.0 bentoml>=1.1.0 prometheus-client>=0.19.0 scikit-learn>=1.3.0

# Financial analytics
pip install scipy>=1.9.0 networkx>=2.8.0 yfinance>=0.2.18
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

## 🤖 MLOps Platform Features

### **MLflow Model Tracking** 📊
Comprehensive experiment management for fund accounting models:
```python
# Example: Track redemption prediction experiments
with mlflow.start_run():
    mlflow.log_param("model_type", "RandomForest")
    mlflow.log_param("data_version", "v1.3")
    mlflow.log_metric("accuracy", 0.94)
    mlflow.sklearn.log_model(model, "redemption_model")
```

**Features:**
- Interactive experiment dashboard with 10+ model runs
- Performance comparison across different algorithms
- Model registry with staging/production promotion
- Complete implementation examples for fund analytics

### **Great Expectations Data Quality** ✅
Automated validation for critical fund operations:
```python
# NAV data quality checks
nav_df.expect_column_values_to_not_be_null("nav_per_share")
nav_df.expect_column_values_to_be_between("nav_per_share", min_value=0)
nav_df.expect_daily_nav_change_to_be_reasonable(max_change=0.10)
```

**Features:**
- Real-time data quality dashboard
- 6 critical validation rules for NAV calculations
- 30-day quality trend analysis
- Automated alert generation for compliance

### **Prefect Workflow Orchestration** 🔄
Production-ready pipeline management:
```python
@flow(name="Daily Fund Accounting Pipeline")
def daily_fund_accounting_flow():
    market_data = fetch_market_data()
    trades = validate_trades()
    nav_data = compute_nav(market_data, trades)
    return generate_reports(nav_data)
```

**Features:**
- Live 7-step fund accounting pipeline visualization
- Interactive Gantt chart timeline
- Task dependency graph with NetworkX
- Historical performance monitoring
- Comprehensive alert management

### **Model Deployment & Serving** 📦
BentoML-powered API deployment:
```python
@bentoml.service
class FraudDetectionService:
    @bentoml.api
    def predict(self, transactions: pd.DataFrame) -> pd.DataFrame:
        return self.model.predict(transactions)
```

**Features:**
- 4 production-deployed models (fraud detection, NAV prediction, etc.)
- Performance monitoring dashboard
- Blue-green deployment strategy visualization
- Model registry integration
- Complete Kubernetes deployment examples

### **Production Monitoring** 📈
Comprehensive observability platform:
```python
# Custom monitoring metrics
prediction_counter = Counter('model_predictions_total')
data_drift_score = Gauge('model_data_drift_score')

class ModelMonitor:
    def check_data_drift(self, current_data):
        # Automated drift detection with alerting
        pass
```

**Features:**
- Real-time system health dashboard (99.97% uptime)
- Data drift detection with feature-level analysis
- Performance degradation monitoring
- Regulatory compliance reporting
- Full audit trail maintenance

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

### **Real Financial Data** 🆕
- **23 Real Asset Classes** from comprehensive CSV dataset
  - GICS Sectors: Energy, Materials, Industrials, Info Tech, Financials, Real Estate
  - Asset Classes: Cash & Equivalents, Fixed Income, Equities, Derivatives, Private Credit, Private Equity, Real Estate, Infrastructure, Commodities, Digital Assets, Carbon Credits, Litigation Finance
  - Professional risk scoring (1-5 scale): Risk, Liquidity, Ops Risk, Cost
- **10 Real Fund Types** with regulatory analysis
  - Frameworks: UCITS, AIFMD, ELTIF 2.0
  - Structures: Irish ICAV, Lux SICAV, SCSp/SCS, ILP, QIAIF, SIF, RAIF
  - Fund strategies: ETFs, Money Market, Multi-Asset, Hedge Funds, Private Equity, Real Estate, Infrastructure

### **Synthetic Asset Data**  
- **24 Financial Assets** across 10 categories for periodic table visualization
- **Synthetic Metrics** based on real-world asset characteristics (1-10 scale)
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
2. **User Authentication**: Portfolio saving/loading per user with MLOps tracking
3. **Advanced Analytics**: Monte Carlo simulations, VaR calculations with ML models
4. **Export Features**: PDF reports, Excel workbooks with model insights
5. **Real-time Updates**: WebSocket integration for live data and model predictions
6. **Mobile App**: PWA conversion for mobile MLOps dashboard

### **MLOps Production Scaling** 🚀
1. **Model Training Automation**: Scheduled retraining pipelines with Prefect
2. **A/B Testing Framework**: Model performance comparison in production
3. **Advanced Monitoring**: Custom business metrics and KPI tracking
4. **Multi-tenant Architecture**: Client-specific model deployments
5. **Edge Computing**: Local model serving for latency-sensitive operations
6. **Regulatory Automation**: AI-powered compliance checking and reporting

### **Production-Ready Architecture**
- **Modular MLOps Design**: Easy model and pipeline additions
- **Scalable Monitoring**: Prometheus + Grafana integration ready
- **Cloud-Native Deployment**: Kubernetes and Docker configurations
- **Enterprise Security**: Authentication, authorization, and audit trails
- **API-First Architecture**: External system integration capabilities

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
- **MLOps Community** for advancing machine learning operations practices
- **MLflow Team** for comprehensive model lifecycle management
- **Great Expectations** for data quality validation frameworks
- **Prefect Team** for modern workflow orchestration
- **BentoML** for seamless model deployment solutions
- **Plotly** for interactive visualization capabilities  
- **Financial Industry** standards for visualization and compliance patterns
- **Modern Portfolio Theory** for optimization algorithms

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section above
2. Review error logs in Streamlit Cloud (if deployed)
3. Test with the provided validation scripts
4. Ensure all dependencies are properly installed

---

**Built with ❤️ using Python, Streamlit, and cutting-edge MLOps technologies.**

*Revolutionize fund accounting operations with professional-grade interactive dashboards and production-ready machine learning workflows.*

### 🎯 Perfect For:
- **Fund Managers**: Advanced portfolio analytics and risk management
- **Data Scientists**: Complete MLOps platform for financial modeling
- **Operations Teams**: Automated workflow orchestration and monitoring  
- **Compliance Officers**: Regulatory reporting and audit trail management
- **Technology Leaders**: Modern, scalable financial technology solutions