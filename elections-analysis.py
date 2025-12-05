import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return


@app.cell
def _():
    import json
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.express as px
    import plotly.graph_objects as go

    # Read JSON files from data directory
    with open('data/2024.json', 'r') as f:
        data_2024 = json.load(f)

    with open('data/2025.json', 'r') as f:
        data_2025 = json.load(f)

    # Convert to DataFrames for easier manipulation
    df_2024 = pd.DataFrame(data_2024) if isinstance(data_2024, list) else pd.json_normalize(data_2024)
    df_2025 = pd.DataFrame(data_2025) if isinstance(data_2025, list) else pd.json_normalize(data_2025)

    # Add year column to distinguish the datasets
    df_2024['year'] = 2024
    df_2025['year'] = 2025

    # Display basic info about the datasets
    print("2024 Data shape:", df_2024.shape)
    print("2024 Data columns:", df_2024.columns.tolist())
    print("\n2025 Data shape:", df_2025.shape)
    print("2025 Data columns:", df_2025.columns.tolist())

    # Show first few rows
    print("\n2024 Data sample:")
    print(df_2024.head())
    print("\n2025 Data sample:")
    print(df_2025.head())
    return df_2024, df_2025, go, pd, px


@app.cell
def _(df_2024, df_2025, pd, px):
    # Combine both datasets for comparison
    combined_df = pd.concat([df_2024, df_2025], ignore_index=True)

    # Filter for PS party data (assuming there's a party column)
    # Try different possible column names for party identification
    party_col = None
    percent_col = None

    for col in combined_df.columns:
        if 'party' in col.lower() or 'parti' in col.lower():
            party_col = col
        if 'percent' in col.lower() or '%' in col or 'pourcent' in col.lower():
            percent_col = col

    # If no specific party column, look for PS in values
    if party_col is None:
        # Look for columns that might contain party names
        for col in combined_df.columns:
            if combined_df[col].dtype == 'object':
                if any('PS' in str(val) for val in combined_df[col].dropna()):
                    party_col = col
                    break

    # Filter for PS party data
    if party_col and percent_col:
        ps_data = combined_df[combined_df[party_col].str.contains('PS', case=False, na=False)]
    
        fig = px.bar(ps_data, 
                     x='year', 
                     y=percent_col,
                     title='PS Party Percentage by Year',
                     labels={percent_col: 'Percentage (%)', 'year': 'Year'},
                     color='year',
                     text=percent_col)
    
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig.update_layout(showlegend=False)
    
        fig
    else:
        # If columns not found, show available columns for debugging
        print("Available columns:", combined_df.columns.tolist())
        print("Sample data:")
        print(combined_df.head())
    return


@app.cell
def _(df_2024, df_2025, go):
    # Let's explore and visualize the actual data we have
    print("Exploring 2024 data structure:")
    print(df_2024.info())
    print("\n2024 data sample:")
    print(df_2024.head())

    print("\n" + "="*50)

    print("Exploring 2025 data structure:")
    print(df_2025.info())
    print("\n2025 data sample:")
    print(df_2025.head())

    # Create a simple visualization of the data structure
    if not df_2024.empty and not df_2025.empty:
        # Get numeric columns for plotting
        numeric_cols_2024 = df_2024.select_dtypes(include=['number']).columns.tolist()
        numeric_cols_2025 = df_2025.select_dtypes(include=['number']).columns.tolist()
    
        print(f"\nNumeric columns in 2024: {numeric_cols_2024}")
        print(f"Numeric columns in 2025: {numeric_cols_2025}")
    
        # If there are numeric columns, create a simple comparison
        if numeric_cols_2024 and numeric_cols_2025:
            # Find common numeric columns
            common_cols = list(set(numeric_cols_2024) & set(numeric_cols_2025))
            if common_cols:
                first_col = common_cols[0]
            
                # Create a simple bar chart comparing the years
                fig = go.Figure()
            
                if hasattr(df_2024[first_col], 'sum'):
                    fig.add_bar(name='2024', x=['2024'], y=[df_2024[first_col].sum()])
                    fig.add_bar(name='2025', x=['2025'], y=[df_2025[first_col].sum()])
                else:
                    fig.add_bar(name='2024', x=['2024'], y=[df_2024[first_col].iloc[0] if len(df_2024) > 0 else 0])
                    fig.add_bar(name='2025', x=['2025'], y=[df_2025[first_col].iloc[0] if len(df_2025) > 0 else 0])
            
                fig.update_layout(
                    title=f'Comparison of {first_col} between 2024 and 2025',
                    xaxis_title='Year',
                    yaxis_title=first_col
                )
            
                fig
    return


if __name__ == "__main__":
    app.run()
