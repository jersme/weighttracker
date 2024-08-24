import streamlit as st
import psycopg2
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import numpy as np

# Constants
MIN_REQUIRED_POINTS = 5  # Minimum data points required for linear regression to make predictions
CALORIES_PER_KG = 7000  # Caloric equivalent of 1 kg of weight loss
VERSION = "1.1.12"  # Current version of the application

def connect_to_db():
    """
    Establish a connection to the PostgreSQL database using SSL.

    Returns:
        conn (psycopg2.connection): Connection object if successful, None if an error occurs.
    """
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            sslmode='require'  # Enforce SSL for secure connection
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_weightracker_data(height_m, target_weight):
    """
    Fetch data from the 'weightracker' table and apply necessary transformations.

    Args:
        height_m (float): User's height in meters.
        target_weight (float): User's target weight in kilograms.

    Returns:
        df (pd.DataFrame): Transformed DataFrame with additional calculated columns.
    """
    conn = connect_to_db()
    if conn is not None:
        try:
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
            df = df.sort_values(by='date')

            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']
            df['BMI'] = df['weight'] / (height_m ** 2)
            df["kgs_to_target"] = df["weight"] - target_weight
            df["calories_to_save"] = df["kgs_to_target"] * CALORIES_PER_KG
            df['cumulative_calories_saved'] = -df['calorie_delta'].cumsum()
            df['theoretical_kgs_saved'] = df['cumulative_calories_saved'] / CALORIES_PER_KG

            initial_weight = df['weight'].iloc[0]
            df['actual_kgs_saved'] = initial_weight - df['weight']

            df['theoretical_kgs_saved'] = df['theoretical_kgs_saved'].abs()
            df['cumulative_calories_saved'] = df['cumulative_calories_saved'].abs()
            df['weight_delta'] = df['weight'].diff()

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()
        finally:
            conn.close()

def calculate_rolling_average(df, window_size):
    """
    Calculate a rolling average for weight_delta and calories_consumed based on the given window size.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.
        window_size (int): Number of days to calculate the rolling average over.

    Returns:
        df_rolling_avg (pd.DataFrame): DataFrame with the rolling average of weight_delta and calories_consumed.
    """
    df_rolling_avg = df[['weight_delta', 'calories_consumed']].rolling(window=window_size).mean()
    df_rolling_avg = df_rolling_avg.dropna()
    return df_rolling_avg

def calculate_linear_regression_rolling_avg(df_rolling_avg):
    """
    Fit a linear regression model on the rolling average data.

    Args:
        df_rolling_avg (pd.DataFrame): DataFrame containing the rolling average of weight and calorie data.

    Returns:
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.
    """
    X = df_rolling_avg['calories_consumed'].values.reshape(-1, 1)
    y = df_rolling_avg['weight_delta'].values.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return model

def plot_weight_vs_calorie_scatter_with_regression(df, model):
    """
    Create a scatter plot of weight delta versus calories consumed with the linear regression line.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the scatter plot and regression line.
    """
    fig = px.scatter(df, x='calories_consumed', y='weight_delta', 
                     title=f'Weight Delta vs Calories Consumed (Rolling Average)',
                     labels={'calories_consumed': 'Calories Consumed', 'weight_delta': 'Delta Weight (kg)'})

    x_min = df['calories_consumed'].min()
    x_max = df['calories_consumed'].max()
    X_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    fig.add_trace(
        go.Scatter(
            x=X_plot.flatten(),
            y=y_plot.flatten(),
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        )
    )

    fig.update_layout(xaxis_title='Calories Consumed', 
                      yaxis_title='Delta Weight (kg)',
                      yaxis=dict(range=[-1, 1]))
    return fig

def calculate_personalized_calories_per_kg_regression_rolling(df, window_size=7):
    """
    Calculate personalized calories required to lose 1 kg of fat using a regression model with rolling averages.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.
        window_size (int): Number of days to calculate the rolling average over.

    Returns:
        calories_per_kg (float): Personalized calories required to lose 1 kg of fat based on regression model.
        model (sklearn.linear_model.LinearRegression): The fitted regression model.
    """
    df_rolling_avg = df[['weight_delta', 'calorie_delta']].rolling(window=window_size).mean()
    df_rolling_avg = df_rolling_avg.dropna()

    if df_rolling_avg.empty:
        st.error("Insufficient data after applying rolling average. Ensure that weight and calorie data are present.")
        return None, None

    X = df_rolling_avg['calorie_delta'].values.reshape(-1, 1)
    y = df_rolling_avg['weight_delta'].values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(X, y)

    slope = model.coef_[0][0]

    if slope >= 0:
        st.error("Unexpected result: the regression suggests a positive or zero calorie burn rate, which is incorrect. Please check your data.")
        return None, None

    calories_per_kg = 1 / slope
    return calories_per_kg, model

def plot_calories_per_kg_regression(df_rolling_avg, model):
    """
    Plot the regression line for the rolling average data and display the personalized calories per kg.

    Args:
        df_rolling_avg (pd.DataFrame): DataFrame containing the rolling average of weight and calorie data.
        model (sklearn.linear_model.LinearRegression): The fitted regression model.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the scatter plot and regression line.
    """
    fig = px.scatter(df_rolling_avg, x='calorie_delta', y='weight_delta',
                     title='Weight Delta vs Calories Saved (Rolling Average)',
                     labels={'calorie_delta': 'Calories Saved', 'weight_delta': 'Delta Weight (kg)'})

    x_min = df_rolling_avg['calorie_delta'].min()
    x_max = df_rolling_avg['calorie_delta'].max()
    X_plot = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    fig.add_trace(
        go.Scatter(
            x=X_plot.flatten(),
            y=y_plot.flatten(),
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        )
    )

    fig.update_layout(xaxis_title='Calories Saved', yaxis_title='Delta Weight (kg)')
    return fig

def display_metrics(df, target_weight):
    """
    Display key metrics based on the weight tracker data.

    Args:
        df (pd.DataFrame): DataFrame containing weight tracker data.
        target_weight (float): User's target weight in kilograms.
    """
    # Calculate metrics based on the current data
    initial_weight = df['weight'].iloc[0]
    kgs_lost_since_start = df['actual_kgs_saved'].iloc[-1]
    kgs_to_go = df['kgs_to_target'].iloc[-1]
    calories_saved = df['cumulative_calories_saved'].iloc[-1]
    calories_to_go = df['calories_to_save'].iloc[-1]

    # Calculate the deltas using the rolling delta function
    kgs_lost_delta = calculate_rolling_delta(df, 'actual_kgs_saved')
    kgs_to_go_delta = calculate_rolling_delta(df, 'kgs_to_target')
    calories_saved_delta = calculate_rolling_delta(df, 'cumulative_calories_saved')
    calories_to_go_delta = calculate_rolling_delta(df, 'calories_to_save')

    # Calculate the total weight to lose and progress percentage
    total_weight_to_lose = initial_weight - target_weight
    progress_percentage = (kgs_lost_since_start / total_weight_to_lose) * 100

    # Display the progress bar in the sidebar
    st.sidebar.write(f"**Progress**: {progress_percentage:.2f}%")
    st.sidebar.progress(progress_percentage / 100)

    # Display the metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kg's Lost Since Start", f"{kgs_lost_since_start:.2f} kg", delta=f"{kgs_lost_delta:.2f} kg")
    with col2:
        st.metric("Kg's to Go", f"{kgs_to_go:.2f} kg", delta=f"{kgs_to_go_delta:.2f} kg")
    with col3:
        st.metric("Calories Saved", f"{calories_saved:.0f}", delta=f"{calories_saved_delta:.0f}")
    with col4:
        st.metric("Calories to Go", f"{calories_to_go:.0f}", delta=f"{calories_to_go_delta:.0f}")

def display_tabs(df, target_weight, height_m):
    """
    Display tabs for analysis, data, and predictions.

    Args:
        df (Pd.DataFrame): DataFrame containing weight tracker data.
        target_weight (float): User's target weight in kilograms.
        height_m (float): User's height in meters.
    """
    tab1, tab2, tab3 = st.tabs(["Analysis", "Data", "Predictions"])

    with tab1:
        st.header("Analysis")
        if not df.empty:
            st.subheader("Summary Statistics")
            summary_df = df.describe().reset_index()
            gb = GridOptionsBuilder.from_dataframe(summary_df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()
            AgGrid(summary_df, gridOptions=grid_options, height=300, width='100%')

            st.subheader("Visualizations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(plot_weight_over_time(df, target_weight), use_container_width=True)
            with col2:
                st.plotly_chart(plot_calorie_delta_over_time(df), use_container_width=True)
            with col3:
                st.plotly_chart(plot_bmi_over_time(df, height_m, target_weight), use_container_width=True)

            st.plotly_chart(plot_kgs_saved(df), use_container_width=True)
            st.plotly_chart(plot_weight_vs_calorie_scatter(df), use_container_width=True)
            
            correlation = calculate_correlation(df)
            st.subheader(f"Correlation between daily weight change and calories saved: {correlation:.2f}")
        else:
            st.write("No data available for analysis.")

    with tab2:
        st.header("Data")
        if not df.empty:
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()
            AgGrid(df, gridOptions=grid_options, height=400, width='100%')
        else:
            st.write("No data available.")

    with tab3:
        st.header("Predictions")
        if not df.empty:
            predicted_date, model = predict_target_reach(df, target_weight)
            if predicted_date and model:
                st.write(f"**Predicted date to reach target weight:** {predicted_date.date()}")
                prediction_plot = plot_prediction(df, model, target_weight)
                if prediction_plot:
                    st.plotly_chart(prediction_plot, use_container_width=True)
                else:
                    st.write("Prediction plot could not be created.")
            else:
                st.write("Prediction model could not be created.")
            
            calorie_burn_rate, maintenance_calories = calculate_calorie_burn_rate_and_maintenance(df)
            if calorie_burn_rate is not None and maintenance_calories is not None:
                st.subheader(f"Calorie Burn Rate: {calorie_burn_rate:.6f} kg/cal")
                st.subheader(f"Daily Calories to Maintain Weight: {maintenance_calories:.0f} cal")

            maintenance_calories_ratio = calculate_calorie_maintenance_using_ratios(df)
            if maintenance_calories_ratio is not None:
                st.subheader(f"Daily Calories to Maintain Weight (Ratio Method): {maintenance_calories_ratio:.0f} cal")

            window_size = st.slider("Select number of days for rolling average", min_value=2, max_value=30, value=7, step=1)

            st.subheader("Personalized Calories per Kg (Regression-Based with Rolling Average)")
            personal_calories_per_kg, model = calculate_personalized_calories_per_kg_regression_rolling(df, window_size)
            if personal_calories_per_kg is not None:
                st.metric("Personalized Calories per Kg", f"{personal_calories_per_kg:.2f} cal/kg")
                
                df_rolling_avg = df[['weight_delta', 'calorie_delta']].rolling(window=window_size).mean().dropna()
                regression_plot = plot_calories_per_kg_regression(df_rolling_avg, model)
                st.plotly_chart(regression_plot, use_container_width=True)
            else:
                st.write("Unable to calculate personalized calories per kg.")
        else:
            st.write("No data available for predictions.")

def main():
    """
    Main function to run the Streamlit app.

    This function initializes the app, fetches and processes data, 
    and displays various analyses and predictions related to weight tracking.
    """
    st.set_page_config(layout="wide")
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    target_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    height_m = st.sidebar.number_input("Height (m)", min_value=1.0, value=1.83, step=0.01)

    # Display version information in the sidebar
    st.sidebar.write(f"Version: {VERSION}")

    # Button to refresh data
    if st.sidebar.button("Refresh Data"):
        st.session_state.refresh = True

    # Initialize session state for refreshing data
    if 'refresh' not in st.session_state:
        st.session_state.refresh = True

    # Fetch and process data if the refresh flag is set
    if st.session_state.refresh:
        df = get_weightracker_data(height_m, target_weight)
        st.session_state.df = df
        st.session_state.refresh = False
    else:
        df = st.session_state.df

    # Display metrics, analysis, data, and predictions based on the fetched data
    if not df.empty:
        display_metrics(df, target_weight)
        display_tabs(df, target_weight, height_m)
    else:
        st.write("No data available.")

if __name__ == "__main__":
    main()
