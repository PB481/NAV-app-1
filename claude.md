**Your Task:** Create a complete, single-file Python script for a Streamlit web application called "The Periodic Table of Asset Types".

**Your Persona:** You are an expert in asset management, fund operations, and data science, with advanced proficiency in Python and the Streamlit framework.

**Core Concept:**
The application should visualize a diverse range of financial asset types in a grid that mimics the layout of the chemical Periodic Table. Each "element" in the table will represent a specific asset type. The visualization must be interactive and data-driven, allowing users to understand the characteristics of each asset at a glance.

**1. Data Curation and Structure:**
First, create the dataset that will power the application. This data should be structured as a Python list of dictionaries or a Pandas DataFrame. Each asset must have the following attributes:

  * `Symbol`: A short, 1-3 letter symbol (e.g., 'UST', 'PE').
  * `Name`: The full name of the asset (e.g., 'US Treasury Bill', 'Private Equity').
  * `Category`: The asset class category (e.g., 'Fixed Income', 'Private Equity', 'Currency').
  * `GridRow` and `GridCol`: Integer values that explicitly define the asset's position in the CSS grid to create the periodic table shape.
  * `Risk`: A score from 1-10 representing market/credit risk (1=Low, 10=High).
  * `Liquidity`: A score from 1-10 representing liquidity (1=Low, 10=High).
  * `OpCost`: A score from 1-10 for the operational cost to manage the asset (1=Low, 10=High).
  * `OpRisk`: A score from 1-10 for the operational risk associated with the asset (1=Low, 10=High).

**Please use this exact data for the application:**

```python
asset_data = [
    {'Symbol': 'USD', 'Name': 'US Dollar', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'UST', 'Name': 'US Treasury Bill', 'Category': 'Fixed Income', 'GridRow': 2, 'GridCol': 1, 'Risk': 1, 'Liquidity': 10, 'OpCost': 2, 'OpRisk': 2},
    {'Symbol': 'EUR', 'Name': 'Euro', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 2, 'Risk': 2, 'Liquidity': 10, 'OpCost': 1, 'OpRisk': 1},
    {'Symbol': 'Bund', 'Name': 'German Bund', 'Category': 'Fixed Income', 'GridRow': 2, 'GridCol': 2, 'Risk': 2, 'Liquidity': 9, 'OpCost': 2, 'OpRisk': 2},
    {'Symbol': 'IGC', 'Name': 'Investment Grade Corp Bond', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 4, 'Risk': 4, 'Liquidity': 7, 'OpCost': 3, 'OpRisk': 3},
    {'Symbol': 'HYC', 'Name': 'High-Yield Corp Bond', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 5, 'Risk': 6, 'Liquidity': 6, 'OpCost': 4, 'OpRisk': 4},
    {'Symbol': 'ETF', 'Name': 'Equity ETF (e.g., SPY)', 'Category': 'Fund', 'GridRow': 2, 'GridCol': 6, 'Risk': 5, 'Liquidity': 9, 'OpCost': 1, 'OpRisk': 2},
    {'Symbol': 'MFt', 'Name': 'Active Mutual Fund', 'Category': 'Fund', 'GridRow': 2, 'GridCol': 7, 'Risk': 6, 'Liquidity': 8, 'OpCost': 3, 'OpRisk': 3},
    {'Symbol': 'EMD', 'Name': 'Emerging Market Debt', 'Category': 'Fixed Income', 'GridRow': 3, 'GridCol': 8, 'Risk': 7, 'Liquidity': 5, 'OpCost': 5, 'OpRisk': 6},
    {'Symbol': 'EMC', 'Name': 'Emerging Market Currency', 'Category': 'Currency', 'GridRow': 1, 'GridCol': 9, 'Risk': 8, 'Liquidity': 6, 'OpCost': 4, 'OpRisk': 5},
    {'Symbol': 'Fut', 'Name': 'Futures (Listed)', 'Category': 'Derivative', 'GridRow': 2, 'GridCol': 13, 'Risk': 7, 'Liquidity': 9, 'OpCost': 3, 'OpRisk': 4},
    {'Symbol': 'Opt', 'Name': 'Options (Listed)', 'Category': 'Derivative', 'GridRow': 2, 'GridCol': 14, 'Risk': 8, 'Liquidity': 8, 'OpCost': 4, 'OpRisk': 5},
    {'Symbol': 'Sw', 'Name': 'OTC Interest Rate Swap', 'Category': 'Derivative', 'GridRow': 3, 'GridCol': 15, 'Risk': 6, 'Liquidity': 5, 'OpCost': 8, 'OpRisk': 8},
    {'Symbol': 'CDS', 'Name': 'Credit Default Swap', 'Category': 'Derivative', 'GridRow': 3, 'GridCol': 16, 'Risk': 8, 'Liquidity': 4, 'OpCost': 9, 'OpRisk': 9},
    {'Symbol': 'SP', 'Name': 'Structured Product (CLO)', 'Category': 'Structured Product', 'GridRow': 4, 'GridCol': 17, 'Risk': 9, 'Liquidity': 3, 'OpCost': 9, 'OpRisk': 9},
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
```

**2. Application Layout and Features:**

  * **Page Configuration:** Set the page layout to "wide" using `st.set_page_config()`.
  * **Sidebar Controls:**
      * Create a sidebar with a `st.selectbox` that allows the user to choose which metric to use for color-coding the table. The options are 'Risk', 'Liquidity', 'OpCost', and 'OpRisk'.
      * Include a section in the sidebar that provides clear definitions for each of these four metrics.
  * **Main View (The Periodic Table):**
      * This is the most critical part. The table **must** be rendered using custom HTML and CSS inside an `st.markdown(html_string, unsafe_allow_html=True)` call.
      * Use **CSS Grid** for the layout, dynamically setting the `grid-column` and `grid-row` for each asset based on the data provided.
      * Each grid item (asset) should display the `Symbol` in a large font and the `Name` in a smaller font below it.
      * The background color of each item must dynamically change based on the metric selected in the sidebar. Create a color-scaling function that maps scores (1-10) to a color (e.g., green-to-red). Note that the color scale for 'Liquidity' should be inverted (high score = green).
      * **Hover Tooltip:** Implement a pure CSS hover effect. When the user's mouse hovers over an asset, a tooltip box must appear. This tooltip should display the asset's full `Name`, `Category`, and all four metric scores with a simple visual representation (e.g., using emojis or star ratings).

**3. Technical Requirements:**

  * The final output must be a single, complete, and runnable Python script.
  * The only external libraries required should be `streamlit` and `pandas`.
  * The code must be well-commented to explain the data structure, the color-scaling function, and the CSS/HTML generation logic.
  * Ensure the use of `unsafe_allow_html=True` is correctly implemented for the final `st.markdown` call to render the table.

Please build the complete application script according to these detailed specifications.