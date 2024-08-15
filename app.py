import streamlit as st
import psycopg2
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

# Constants
MIN_REQUIRED_POINTS = 5
CALORIES_PER_KG = 7000
VERSION = "1.0.4"  # Updated version number

def connect_to_db():
    """Establish a connection to the PostgreSQL database with SSL."""
    try:
        conn = psycopg2.connect(
            host=st.secrets["DB_HOST"],
            port=st.secrets["DB_PORT"],
            dbname=st.secrets["DB_NAME"],
            user=st.secrets["DB_USER"],
            password=st.secrets["DB_PASS"],
            sslmode='require'  # Ensure SSL is used for the connection
        )
        return conn
    except Exception as e:
        st.error(f"Error connecting to database: {e}")
        return None

def get_weightracker_data(height_m, target_weight):
    """Fetch data from the weightracker table and transform it."""
    conn = connect_to_db()
    if conn is not None:
        try:
            query = "SELECT * FROM weightracker;"
            df = pd.read_sql(query, conn)
            
            # Transform the 'date' column to a datetime format
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')

            # Sort the dataframe by date to correctly calculate cumulative values
            df is df.sort_values(by='date')

            # Calculate the daily calorie delta
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            # Calculate BMI using the user's height
            df['BMI'] = df['weight'] / (height_m ** 2)

            # Calculate KGs to go to target weight
            df["kgs_to_target"] = df["weight"] - target_weight

            # Calculate calories to save to reach target weight
            df["calories_to_save"] = df["kgs_to_target"] * CALORIES_PER_KG

            # Calculate cumulative calories saved
            df['cumulative_calories_saved'] = -df['calorie_delta'].cumsum()  # Inverted to reflect savings positively

            # Calculate theoretical kilograms saved
            df['theoretical_kgs_saved'] = df['cumulative_calories_saved'] / CALORIES_PER_KG

            # Calculate actual kilograms saved
            initial_weight = df['weight'].iloc[0]
            df['actual_kgs_saved'] = initial_weight - df['weight']

            # Ensure positive values for theoretical calculations
            df['theoretical_kgs_saved'] = df['theoretical_kgs_saved'].abs()
            df['cumulative_calories_saved'] = df['cumulative_calories_saved'].abs()

            return df
        except Exception as e:
            st.error(f"Error fetching or transforming data: {e}")
            return pd.DataFrame()  # Return empty DataFrame in case of error
        finally:
            conn.close()

def plot_weight_over_time(df, target_weight):
    """Plot weight changes over time using Plotly."""
    fig = px.line(df, x='date', y='weight', title='Weight Over Time', markers=True)
    fig.update_layout(
        xaxis_title='Date', 
        yaxis_title='Weight (kg)',
        yaxis=dict(range=[0, df['weight'].max() * 1.1])  # Set the y-axis to start at 0 and go slightly above the max weight
    )
    
    # Add a horizontal line for the target weight
    fig.add_hline(y=target_weight, line_dash="dash", line_color="green", annotation_text="Target Weight", annotation_position="bottom right")
    
    return fig


def plot_calorie_delta_over_time(df):
    """Plot calorie delta over time using Plotly."""
    fig = px.bar(df, x='date', y='calorie_delta', title='Calorie Delta Over Time', color='calorie_delta', color_continuous_scale='RdBu')
    fig.update_layout(xaxis_title='Date', yaxis_title='Calorie Delta', showlegend=False)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

def plot_bmi_over_time(df, height_m, target_weight):
    """Plot BMI changes over time using Plotly."""
    target_bmi = target_weight / (height_m ** 2)

    # Create a color column to change the color of the line based on BMI value
    df['bmi_color'] = np.where(df['BMI'] > 25, 'red', 'blue')

    fig = px.line(df, x='date', y='BMI', title='BMI Over Time', markers=True, line_shape='linear', color='bmi_color', color_discrete_map={"red": "red", "blue": "blue"})
    fig.update_layout(xaxis_title='Date', yaxis_title='BMI')
    
    # Add a horizontal line for the target BMI
    fig.add_hline(y=target_bmi, line_dash="dash", line_color="green", annotation_text="Target BMI", annotation_position="bottom right")

    return fig

def plot_kgs_saved(df):
    """Plot theoretical vs actual kilograms saved over time using Plotly."""
    fig = px.line(df, x='date', y=['theoretical_kgs_saved', 'actual_kgs_saved'], 
                  title='Theoretical vs Actual Kgs Saved Over Time', 
                  markers=True, line_shape='linear')
    fig.update_layout(xaxis_title='Date', yaxis_title='Kilograms Saved',
                      legend_title_text='Type of Savings')
    return fig

def predict_target_reach(df, target_weight):
    """Predict the date when the target weight will be reached using linear regression."""
    if df.empty:
        st.error("No data available to perform prediction.")
        return None, None

    initial_weight = df['weight'].iloc[0]
    df['actual_kgs_saved'] = initial_weight - df['weight']
    df['days'] = (df['date'] - df['date'].min()).dt.days

    if df['days'].empty or df['actual_kgs_saved'].empty:
        st.error("Insufficient data for prediction.")
        return None, None

    # Prepare the data for regression
    X = df['days'].values.reshape(-1, 1)
    y = df['actual_kgs_saved'].values.reshape(-1, 1)

    if X.shape[0] < MIN_REQUIRED_POINTS or y.shape[0] < MIN_REQUIRED_POINTS:
        st.error(f"Insufficient data points for regression. At least {MIN_REQUIRED_POINTS} points are required. "
                 f"Current points: {X.shape[0]}. Additional points needed: {MIN_REQUIRED_POINTS - X.shape[0]}")
        return None, None

    # Perform linear regression
    model = LinearRegression()
    model.fit(X, y)

    # Predict the day at which the target weight will be reached
    target_kgs_saved = initial_weight - target_weight
    predicted_days = (target_kgs_saved - model.intercept_[0]) / model.coef_[0][0]
    
    if predicted_days < 0:
        st.error("Model predicts that the target weight has already been achieved.")
        return None, None

    # Convert predicted days into a DateOffset and add to the min date
    predicted_date = df['date'].min() + pd.DateOffset(days=int(predicted_days))

    return predicted_date, model

def plot_prediction(df, model, target_weight):
    """Plot the prediction line along with actual kilograms lost."""
    if model is None:
        return

    initial_weight = df['weight'].iloc[0]
    predicted_date, _ = predict_target_reach(df, target_weight)
    if predicted_date is None:
        return

    max_days = (predicted_date - df['date'].min()).days
    if max_days <= 0:
        st.error("Prediction date is in the past. No future data to plot.")
        return

    future_days = np.arange(0, max_days + 1).reshape(-1, 1)
    predicted_kgs_saved = model.predict(future_days)

    future_dates = df['date'].min() + pd.to_timedelta(future_days.flatten(), unit='D')

    prediction_df = pd.DataFrame({
        'date': future_dates,
        'predicted_kgs_saved': predicted_kgs_saved.flatten()
    })

    fig = px.line(prediction_df, x='date', y='predicted_kgs_saved', 
                  title='Prediction of Weight Loss Over Time', 
                  labels={'predicted_kgs_saved': 'Kilograms Saved'},
                  line_dash_sequence=['dash'],
                  color_discrete_sequence=['green'],
                  )
    fig.add_scatter(x=df['date'], y=df['actual_kgs_saved'], mode='markers', name="Actual Kg's Saved", marker=dict(color='blue'))
    
    # Add vertical line for the predicted date
    fig.add_vline(x=predicted_date, line_dash="dot", line_color="red", 
                  annotation_text="Target Date", annotation_position="top right")

    fig.update_layout(legend=dict(y=0.5, traceorder='reversed', font_size=12))

    return fig

def display_metrics(df, target_weight):
    """Display key metrics based on the weight tracker data."""
    # Calculate metrics
    kgs_lost_since_start = df['actual_kgs_saved'].iloc[-1]
    kgs_to_go = df['kgs_to_target'].iloc[-1]
    calories_saved = df['cumulative_calories_saved'].iloc[-1]
    calories_to_go = df['calories_to_save'].iloc[-1]

    # Filter for the current week's data
    today = datetime.date.today()
    start_of_week = today - datetime.timedelta(days=today.weekday())
    current_week_data = df[df['date'].dt.date >= start_of_week]

    # Average values for the current week
    avg_kgs_lost_this_week = current_week_data['actual_kgs_saved'].mean() if not current_week_data.empty else 0
    avg_kgs_to_go_this_week = current_week_data['kgs_to_target'].mean() if not current_week_data.empty else 0
    avg_calories_saved_this_week = current_week_data['cumulative_calories_saved'].mean() if not current_week_data.empty else 0
    avg_calories_to_go_this_week = current_week_data['calories_to_save'].mean() if not current_week_data.empty else 0

    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Kg's Lost Since Start", f"{kgs_lost_since_start:.2f} kg", delta=f"{avg_kgs_lost_this_week:.2f} kg")
    with col2:
        st.metric("Kg's to Go", f"{kgs_to_go:.2f} kg", delta=f"{avg_kgs_to_go_this_week:.2f} kg")
    with col3:
        st.metric("Calories Saved", f"{calories_saved:.0f}", delta=f"{avg_calories_saved_this_week:.0f}")
    with col4:
        st.metric("Calories to Go", f"{calories_to_go:.0f}", delta=f"{avg_calories_to_go_this_week:.0f}")

def display_tabs(df, target_weight, height_m):
    """Display tabs for analysis, data, and predictions."""
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
        else:
            st.write("No data available for predictions.")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    target_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    height_m = st.sidebar.number_input("Height (m)", min_value=1.0, value=1.83, step=0.01)

    # Display version information
    st.sidebar.write(f"Version: {VERSION}")

    # Button to refresh data
    if st.sidebar.button("Refresh Data"):
        st.session_state.refresh = True

    # Initialize session state for refreshing
    if 'refresh' not in st.session_state:
        st.session_state.refresh = True

    if st.session_state.refresh:
        df = get_weightracker_data(height_m, target_weight)
        st.session_state.df = df
        st.session_state.refresh = False
    else:
        df = st.session_state.df

    if not df.empty:
        display_metrics(df, target_weight)
        display_tabs(df, target_weight, height_m)
    else:
        st.write("No data available.")

if __name__ == "__main__":
    main()
