import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from datetime import date
from typing import Tuple

# --- Constants for File Access ---
# CHANGE: Use the standard file name for local execution.
SALES_FILE_PATH = "Sample - Superstore.csv"

# --- Main Forecasting Script ---
# CHANGE: The return signature now includes an optional flag to simplify error handling
def forecast_sales() -> Tuple[pd.DataFrame | None, Prophet | None, pd.DataFrame | None]:
    """
    Loads, cleans, aggregates, trains the Prophet model, and generates a 1-year sales forecast
    using the Superstore dataset.
    """
    # 1. Data Loading and Aggregation
    print("1. Loading and aggregating Superstore data...")
    try:
        # FIX: Add encoding='latin-1' to handle non-UTF-8 characters in the CSV,
        # resolving the 'invalid start byte' error.
        df = pd.read_csv(SALES_FILE_PATH, encoding='latin-1')
    except FileNotFoundError:
        # FIX: Ensure all three expected return values are None
        print(f"Error: Could not find file named '{SALES_FILE_PATH}'. Please ensure the file is in the same directory as the script.")
        return None, None, None
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return None, None, None

    # Convert 'Order Date' to datetime objects
    # Note: Using infer_datetime_format=True for robustness, though the explicit format is good.
    df['Order Date'] = pd.to_datetime(df['Order Date'])

    # Aggregate total Sales by 'Order Date' to create a daily time series
    daily_sales_df = df.groupby('Order Date')['Sales'].sum().reset_index()

    # Prophet requires 'ds' (date stamp) and 'y' (value)
    prophet_df = daily_sales_df.rename(columns={'Order Date': 'ds', 'Sales': 'y'})

    # 2. Feature Engineering (Defining Holidays/Special Events)
    print("2. Defining Holiday Regressors...")

    # Define a simple list of observed U.S. holidays for the model to learn from
    holidays = pd.DataFrame({
        'holiday': 'Major_US_Holiday',
        'ds': pd.to_datetime([
            '2014-12-25', '2015-12-25', '2016-12-25', '2017-12-25', # Christmas
            '2014-01-01', '2015-01-01', '2016-01-01', '2017-01-01', # New Year's Day
            '2014-11-27', '2015-11-26', '2016-11-24', '2017-11-23'  # Thanksgiving (approx 4th Thurs in Nov)
        ]),
        'lower_window': -3, # Start effect 3 days before
        'upper_window': 1   # End effect 1 day after
    })
    
    # 3. Model Training (Prophet)
    print("3. Training Prophet model...")
    model = Prophet(
        holidays=holidays,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive' # Additive mode often better for raw aggregated sales
    )
    # Fit the model to the historical data
    # Suppress a known FutureWarning from Prophet during fitting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        model.fit(prophet_df)

    # 4. Creating the Future Dataframe
    # We want to forecast 365 days (1 year) into the future
    future = model.make_future_dataframe(periods=365, freq='D')

    # 5. Generating the Forecast
    print("4. Generating 1-year forecast...")
    forecast = model.predict(future)

    # Store the historical sales data separately for merging later
    historical_data = prophet_df[['ds', 'y']].rename(columns={'y': 'Actual_Sales'})

    return forecast, model, historical_data

# --- Execution Block ---
if __name__ == "__main__":
    import warnings # Import warnings here for use in the __main__ block

    # The function call now returns three Nones if it fails, which the if-statement handles.
    forecast_df, prophet_model, historical_data = forecast_sales()

    if forecast_df is None:
        print("Forecasting process halted due to data loading error.")
    else:
        # C. Visualize Results (Matplotlib requirement)
        print("5. Generating visualizations and saving plots to files...")

        # Plot 1: Trend, Seasonality, and Forecast
        fig1 = prophet_model.plot(forecast_df, xlabel='Date', ylabel='Sales ($)')
        plt.title('Superstore Daily Sales Forecast (Prophet Model)')
        # FIX: Save plot instead of showing it
        forecast_plot_path = 'sales_forecast_plot.png'
        fig1.savefig(forecast_plot_path)
        plt.close(fig1) # Close the figure to free memory

        # Plot 2: Decomposed Components (Trend and Seasonality)
        fig2 = prophet_model.plot_components(forecast_df)
        plt.suptitle('Prophet Decomposed Components: Trend and Seasonality', y=1.02)
        # FIX: Save plot instead of showing it
        components_plot_path = 'sales_components_plot.png'
        fig2.savefig(components_plot_path)
        plt.close(fig2) # Close the figure to free memory

        print(f"   -> Saved Plot 1 (Forecast) to: {forecast_plot_path}")
        print(f"   -> Saved Plot 2 (Components) to: {components_plot_path}")


        # D. Prepare Deliverable (Forecast for Power BI)
        # Select key columns: Date (ds), Predicted Sales (yhat), and Confidence Interval
        final_forecast_output = forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

        # Join the historical sales data to the final output for complete data analysis
        combined_df = final_forecast_output.merge(
            historical_data,
            on='ds',
            how='left'
        )

        # Clean up column names for dashboard presentation
        combined_df.columns = ['Date', 'Predicted_Sales', 'Lower_Bound', 'Upper_Bound', 'Actual_Sales']

        # Save to CSV for Power BI input
        output_path = 'retail_sales_forecast_for_powerbi.csv'
        combined_df.to_csv(output_path, index=False)
        print(f"\n6. Successfully saved forecast data to: {output_path}")

        # Display a summary of the forecast
        last_historical_date = historical_data['ds'].max()
        first_forecast_date = combined_df[combined_df['Actual_Sales'].isnull()]['Date'].min()
        print("\n--- Summary ---")
        print(f"Historical Data Range: {historical_data['ds'].min().date()} to {last_historical_date.date()}")
        print(f"Forecast Range: {first_forecast_date.date()} to {combined_df['Date'].max().date()}")
        print("The two required Matplotlib plots (Forecast and Components) have been saved as PNG files.")
        print(f"The combined data is ready for your Power BI dashboard in: {output_path}")
