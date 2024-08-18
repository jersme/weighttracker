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
VERSION = "1.1.1"  # Current version of the application

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
            # Fetch all data from the weightracker table
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            # Convert the 'date' column to datetime format for time-based calculations
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # Sort the dataframe by date to ensure correct order for cumulative calculations
            df = df.sort_values(by='date')

            # Calculate daily calorie delta (consumed - burned)
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            # Calculate BMI based on user's height
            df['BMI'] = df['weight'] / (height_m ** 2)

            # Calculate kilograms remaining to reach the target weight
            df["kgs_to_target"] = df["weight"] - target_weight

            # Calculate calories needed to save to reach target weight
            df["calories_to_save"] = df["kgs_to_target"] * CALORIES_PER_KG

            # Calculate cumulative calories saved over time
            df['cumulative_calories_saved'] = -df['calorie_delta'].cumsum()  # Inverted to show savings as positive

            # Calculate theoretical kilograms saved based on cumulative calorie savings
            df['theoretical_kgs_saved'] = df['cumulative_calories_saved'] / CALORIES_PER_KG

            # Calculate actual kilograms saved compared to initial weight
            initial_weight = df['weight'].iloc[0]
            df['actual_kgs_saved'] = initial_weight - df['weight']

            # Ensure positive values for theoretical calculations
            df['theoretical_kgs_saved'] = df['theoretical_kgs_saved'].abs()
            df['cumulative_calories_saved'] = df['cumulative_calories_saved'].abs()

            # Calculate daily weight change (delta)
            df['weight_delta'] = df['weight'].diff()

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of an error
        finally:
            conn.close()

def calculate_rolling_average(df):
    """
    Calculate a 3-day rolling average for weight_delta and calorie_delta.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.

    Returns:
        df_rolling_avg (pd.DataFrame): DataFrame with the 3-day rolling average of weight_delta and calorie_delta.
    """
    df_rolling_avg = df[['weight_delta', 'calorie_delta']].rolling(window=3).mean()
    df_rolling_avg = df_rolling_avg.dropna()  # Drop rows with NaN values resulting from the rolling average calculation
    return df_rolling_avg

def calculate_linear_regression_rolling_avg(df_rolling_avg):
    """
    Fit a linear regression model on the 3-day rolling average data.

    Args:
        df_rolling_avg (pd.DataFrame): DataFrame containing the 3-day rolling average of weight and calorie data.

    Returns:
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.
    """
    X = df_rolling_avg['weight_delta'].values.reshape(-1, 1)
    y = df_rolling_avg['calorie_delta'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    return model

def plot_weight_vs_calorie_scatter_with_regression(df, model):
    """
    Create a scatter plot of weight delta versus calorie delta with the linear regression line.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the scatter plot and regression line.
    """
    # Create the scatter plot
    fig = px.scatter(df, x='weight_delta', y='calorie_delta', 
                     title='Weight Delta vs Calorie Saved (3-Day Rolling Average)',
                     labels={'weight_delta': 'Delta Weight (kg)', 'calorie_delta': 'Calories Saved'})

    # Generate regression line points
    X_plot = np.linspace(df['weight_delta'].min(), df['weight_delta'].max(), 100).reshape(-1, 1)
    y_plot = model.predict(X_plot)

    # Add the regression line to the plot
    fig.add_trace(
        go.Scatter(
            x=X_plot.flatten(),
            y=y_plot.flatten(),
            mode='lines',
            name='Regression Line',
            line=dict(color='red')
        )
    )
    
    fig.update_layout(xaxis_title='Delta Weight (kg)', yaxis_title='Calories Saved')
    return fig

def plot_weight_over_time(df, target_weight):
    """
    Create a line plot of weight changes over time.

    Args:
        df (pd.DataFrame): DataFrame containing weight data.
        target_weight (float): User's target weight in kilograms.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the weight over time plot.
    """
    # Create a line plot with date on the x-axis and weight on the y-axis
    fig = px.line(df, x='date', y='weight', title='Weight Over Time', markers=True)
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Weight (kg)',
        yaxis=dict(range=[0, df['weight'].max() * 1.1])  # Set y-axis range to start from 0 and extend slightly above max weight
    )
    
    # Add a horizontal line indicating the target weight
    fig.add_hline(y=target_weight, line_dash="dash", line_color="green", annotation_text="Target Weight", annotation_position="bottom right")
    
    return fig

def plot_calorie_delta_over_time(df):
    """
    Create a bar plot of daily calorie delta over time.

    Args:
        df (pd.DataFrame): DataFrame containing calorie data.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the calorie delta over time plot.
    """
    # Create a bar plot with date on the x-axis and calorie delta on the y-axis
    fig = px.bar(df, x='date', y='calorie_delta', title='Calorie Delta Over Time', color='calorie_delta', color_continuous_scale='RdBu')
    fig.update_layout(xaxis_title='Date', yaxis_title='Calorie Delta', showlegend=False)
    
    # Add a horizontal line at y=0 to indicate the balance point
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

def plot_bmi_over_time(df, height_m, target_weight):
    """
    Create a line plot of BMI changes over time.

    Args:
        df (pd.DataFrame): DataFrame containing BMI data.
        height_m (float): User's height in meters.
        target_weight (float): User's target weight in kilograms.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the BMI over time plot.
    """
    # Calculate the target BMI based on user's height and target weight
    target_bmi = target_weight / (height_m ** 2)

    # Create a color column to differentiate BMI values above and below 25
    df['bmi_color'] = np.where(df['BMI'] > 25, 'red', 'blue')

    # Create a line plot with date on the x-axis and BMI on the y-axis
    fig = px.line(df, x='date', y='BMI', title='BMI Over Time', markers=True, line_shape='linear', color='bmi_color', color_discrete_map={"red": "red", "blue": "blue"})
    fig.update_layout(xaxis_title='Date', yaxis_title='BMI')
    
    # Add a horizontal line indicating the target BMI
    fig.add_hline(y=target_bmi, line_dash="dash", line_color="green", annotation_text="Target BMI", annotation_position="bottom right")

    return fig

def plot_kgs_saved(df):
    """
    Create a line plot comparing theoretical versus actual kilograms saved over time.

    Args:
        df (pd.DataFrame): DataFrame containing weight loss data.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the theoretical vs actual kgs saved plot.
    """
    # Create a line plot comparing theoretical and actual kilograms saved
    fig = px.line(df, x='date', y=['theoretical_kgs_saved', 'actual_kgs_saved'], 
                  title='Theoretical vs Actual Kgs Saved Over Time', 
                  markers=True, line_shape='linear')
    fig.update_layout(xaxis_title='Date', yaxis_title='Kilograms Saved',
                      legend_title_text='Type of Savings')
    return fig

def plot_weight_vs_calorie_scatter(df):
    """
    Create a scatter plot of weight delta versus calorie delta.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the scatter plot of weight delta vs calorie delta.
    """
    # Create a scatter plot with weight delta on the x-axis and calorie delta on the y-axis
    fig = px.scatter(df, x='weight_delta', y='calorie_delta', 
                     title='Weight Delta vs Calorie Saved',
                     labels={'weight_delta': 'Delta Weight (kg)', 'calorie_delta': 'Calorie Saved'})
    fig.update_layout(xaxis_title='Delta Weight (kg)', yaxis_title='Calories Saved')
    return fig

def calculate_correlation(df):
    """
    Calculate the correlation between daily weight change and calorie delta.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.

    Returns:
        correlation (float): Correlation coefficient between weight delta and calorie delta.
    """
    correlation = df[['weight_delta', 'calorie_delta']].corr().iloc[0, 1]
    return correlation

def predict_target_reach(df, target_weight):
    """
    Predict the date when the target weight will be reached using linear regression.

    Args:
        df (pd.DataFrame): DataFrame containing weight loss data.
        target_weight (float): User's target weight in kilograms.

    Returns:
        predicted_date (datetime.datetime): Predicted date when target weight will be reached.
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.
    """
    if df.empty:
        st.error("No data available to perform prediction.")
        return None, None

    # Calculate actual kilograms saved from initial weight
    initial_weight = df['weight'].iloc[0]
    df['actual_kgs_saved'] = initial_weight - df['weight']

    # Calculate the number of days since the start
    df['days'] = (df['date'] - df['date'].min()).dt.days

    if df['days'].empty or df['actual_kgs_saved'].empty:
        st.error("Insufficient data for prediction.")
        return None, None

    # Prepare the data for linear regression
    X = df['days'].values.reshape(-1, 1)
    y = df['actual_kgs_saved'].values.reshape(-1, 1)

    # Ensure there are enough data points to perform regression
    if X.shape[0] < MIN_REQUIRED_POINTS or y.shape[0] < MIN_REQUIRED_POINTS:
        st.error(f"Insufficient data points for regression. At least {MIN_REQUIRED_POINTS} points are required. "
                 f"Current points: {X.shape[0]}. Additional points needed: {MIN_REQUIRED_POINTS - X.shape[0]}")
        return None, None

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate the day at which the target weight will be reached
    target_kgs_saved = initial_weight - target_weight
    predicted_days = (target_kgs_saved - model.intercept_[0]) / model.coef_[0][0]
    
    if predicted_days < 0:
        st.error("Model predicts that the target weight has already been achieved.")
        return None, None

    # Convert the predicted days into a date
    predicted_date = df['date'].min() + pd.DateOffset(days=int(predicted_days))

    return predicted_date, model

def calculate_calorie_burn_rate_and_maintenance(df):
    """
    Calculate the calorie burn rate and daily calorie intake needed to maintain weight.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.

    Returns:
        calorie_burn_rate (float): The slope of the regression line, indicating the rate of calorie burn.
        maintenance_calories (float): The daily calorie intake needed to maintain current weight.
    """
    if df.empty or df['weight_delta'].isnull().all() or df['calorie_delta'].isnull().all():
        st.error("Insufficient data to calculate calorie burn rate and maintenance calories.")
        return None, None

    # Drop rows with NaN values in weight_delta or calorie_delta
    df = df.dropna(subset=['weight_delta', 'calorie_delta'])

    # Prepare the data for regression: calorie_delta vs. weight_delta
    X = df['calorie_delta'].values.reshape(-1, 1)
    y = df['weight_delta'].values.reshape(-1, 1)

    # Fit the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Calculate the slope (calorie_burn_rate) and intercept of the regression line
    calorie_burn_rate = model.coef_[0][0]
    intercept = model.intercept_[0]

    # Ensure that the slope is negative, indicating that a calorie surplus leads to weight gain
    if calorie_burn_rate >= 0:
        st.error("Unexpected result: the regression suggests a positive or zero calorie burn rate, which is incorrect. Please check your data.")
        return None, None

    # Calculate the calorie delta needed to maintain weight (weight_delta = 0)
    calorie_delta_to_maintain_weight = -intercept / calorie_burn_rate

    # Calculate the daily calories needed to maintain weight
    avg_calories_burned = df['calories_burned'].mean()
    maintenance_calories = avg_calories_burned + calorie_delta_to_maintain_weight

    return calorie_burn_rate, maintenance_calories

def calculate_calorie_maintenance_using_ratios(df):
    """
    Calculate the daily calorie intake needed to maintain weight using the ratio method.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.

    Returns:
        C_maintain (float): The daily calorie intake needed to maintain current weight.
    """
    if df.empty or df['weight_delta'].isnull().all() or df['calories_consumed'].isnull().all():
        st.error("Insufficient data to calculate maintenance calories using the ratio method.")
        return None

    # Drop rows with NaN values in weight_delta or calories_consumed
    df = df.dropna(subset=['weight_delta', 'calories_consumed'])

    # Calculate the average weight change and average calorie consumption
    avg_weight_change = df['weight_delta'].mean()  # Average weight change in kg
    avg_calories_consumed = df['calories_consumed'].mean()  # Average calories consumed

    # If the average weight change is zero, use the average calorie consumption as the maintenance level
    if avg_weight_change == 0:
        st.write("No average weight change observed. Using average calorie consumption as maintenance.")
        return avg_calories_consumed

    # Calculate the ratio of calories to weight change
    calories_per_kg = avg_calories_consumed / avg_weight_change

    # To maintain weight (i.e., no weight change), the required caloric intake
    C_maintain = calories_per_kg * avg_weight_change

    return C_maintain

def plot_prediction(df, model, target_weight):
    """
    Plot the prediction line along with actual kilograms lost over time.

    Args:
        df (pd.DataFrame): DataFrame containing weight loss data.
        model (sklearn.linear_model.LinearRegression): Fitted linear regression model.
        target_weight (float): User's target weight in kilograms.

    Returns:
        fig (plotly.graph_objs._figure.Figure): Plotly figure object with the prediction plot.
    """
    if model is None:
        return

    # Calculate initial weight and predicted date
    initial_weight = df['weight'].iloc[0]
    predicted_date, _ = predict_target_reach(df, target_weight)
    if predicted_date is None:
        return

    # Convert predicted_date to a string in 'YYYY-MM-DD' format for Plotly
    predicted_date_str = predicted_date.strftime('%Y-%m-%d')

    # Ensure `max_days` is an integer and not directly added to a datetime object
    max_days = (predicted_date - df['date'].min()).days
    if max_days <= 0:
        st.error("Prediction date is in the past. No future data to plot.")
        return

    # Calculate future dates using Timedelta
    future_days = np.arange(0, max_days + 1)
    future_dates = [df['date'].min() + pd.Timedelta(days=int(day)) for day in future_days]

    # Predict kilograms saved for future dates
    predicted_kgs_saved = model.predict(future_days.reshape(-1, 1))

    # Create a DataFrame for plotting the prediction
    prediction_df = pd.DataFrame({
        'date': future_dates,
        'predicted_kgs_saved': predicted_kgs_saved.flatten()
    })

    # Create a line plot with the prediction and actual data
    fig = px.line(prediction_df, x='date', y='predicted_kgs_saved', 
                  title='Prediction of Weight Loss Over Time', 
                  labels={'predicted_kgs_saved': 'Kilograms Saved'},
                  line_dash_sequence=['dash'],
                  color_discrete_sequence=['green'],
                  )
    fig.add_scatter(x=df['date'], y=df['actual_kgs_saved'], mode='markers', name="Actual Kg's Saved", marker=dict(color='blue'))
    
    # Add a vertical line indicating the predicted target date
    fig.add_shape(
        type="line",
        x0=predicted_date_str,
        x1=predicted_date_str,
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        line=dict(color="red", width=2, dash="dot"),
    )

    # Add an annotation for the target date
    fig.add_annotation(
        x=predicted_date_str,
        y=max(prediction_df['predicted_kgs_saved']),
        text="Target Date",
        showarrow=True,
        arrowhead=1,
        ax=0,
        ay=-40
    )

    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=12))

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

    # Calculate the total weight to lose and progress percentage
    total_weight_to_lose = initial_weight - target_weight
    progress_percentage = (kgs_lost_since_start / total_weight_to_lose) * 100

    # Display the progress bar in the sidebar
    st.sidebar.write(f"**Progress**: {progress_percentage:.2f}%")
    st.sidebar.progress(progress_percentage / 100)

    # Filter for the current week's data
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    current_week_data = df[df['date'].dt.date >= start_of_week]

    # Calculate average values for the current week
    avg_kgs_lost_this_week = current_week_data['actual_kgs_saved'].mean() if not current_week_data.empty else 0
    avg_kgs_to_go_this_week = current_week_data['kgs_to_target'].mean() if not current_week_data.empty else 0
    avg_calories_saved_this_week = current_week_data['cumulative_calories_saved'].mean() if not current_week_data.empty else 0
    avg_calories_to_go_this_week = current_week_data['calories_to_save'].mean() if not current_week_data.empty else 0

    # Display the metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kg's Lost Since Start", f"{kgs_lost_since_start:.2f} kg", delta=f"{avg_kgs_lost_this_week:.2f} kg")
    with col2:
        st.metric("Kg's to Go", f"{kgs_to_go:.2f} kg", delta=f"{avg_kgs_to_go_this_week:.2f} kg")
    with col3:
        st.metric("Calories Saved", f"{calories_saved:.0f}", delta=f"{avg_calories_saved_this_week:.0f}")
    with col4:
        st.metric("Calories to Go", f"{calories_to_go:.0f}", delta=f"{avg_calories_to_go_this_week:.0f}")

def display_prediction_tab_with_rolling_avg(df, target_weight):
    """
    Display the prediction tab, including a scatter plot and regression line based on 3-day rolling average data.

    Args:
        df (pd.DataFrame): DataFrame containing weight and calorie data.
        target_weight (float): User's target weight in kilograms.
    """
    # Calculate the rolling average
    df_rolling_avg = calculate_rolling_average(df)

    if not df_rolling_avg.empty:
        # Fit the linear regression model on the rolling average data
        model = calculate_linear_regression_rolling_avg(df_rolling_avg)
        
        # Create and display the scatter plot with the regression line
        scatter_plot_with_regression = plot_weight_vs_calorie_scatter_with_regression(df_rolling_avg, model)
        st.plotly_chart(scatter_plot_with_regression, use_container_width=True)

        # Display the predicted target date using the original prediction function
        predicted_date, _ = predict_target_reach(df, target_weight)
        if predicted_date:
            st.write(f"**Predicted date to reach target weight:** {predicted_date.date()}")
        else:
            st.write("Prediction model could not be created.")
    else:
        st.write("No sufficient data to perform rolling average calculation.")

def display_tabs(df, target_weight, height_m):
    """
    Display tabs for analysis, data, and predictions.

    Args:
        df (pd.DataFrame): DataFrame containing weight tracker data.
        target_weight (float): User's target weight in kilograms.
        height_m (float): User's height in meters.
    """
    # Create tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Analysis", "Data", "Predictions"])

    with tab1:
        st.header("Analysis")
        if not df.empty:
            st.subheader("Summary Statistics")
            # Display summary statistics in an interactive grid
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
            st.plotly_chart(plot_weight_vs_calorie_scatter(df), use_container_width=True)  # Original scatter plot
            
            # Calculate and display the correlation between weight delta and calorie delta
            correlation = calculate_correlation(df)
            st.subheader(f"Correlation between daily weight change and calories saved: {correlation:.2f}")
        else:
            st.write("No data available for analysis.")

    with tab2:
        st.header("Data")
        if not df.empty:
            # Display the full dataset in an interactive grid
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
            display_prediction_tab_with_rolling_avg(df, target_weight)
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
