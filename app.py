import streamlit as st
import psycopg2
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import datetime
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

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
            df = df.sort_values(by='date')

            # Calculate the daily calorie delta
            df['calorie_delta'] = df['calories_consumed'] - df['calories_burned']

            # Calculate BMI using the user's height
            df['BMI'] = df['weight'] / (height_m ** 2)

            # Calculate KGs to go to target weight
            df["kgs_to_target"] = df["weight"] - target_weight

            # Calculate calories to save to reach target weight
            df["calories_to_save"] = df["kgs_to_target"] * 7000

            # Calculate cumulative calories saved
            df['cumulative_calories_saved'] = -df['calorie_delta'].cumsum()  # Inverted to reflect savings positively

            # Calculate theoretical kilograms saved
            df['theoretical_kgs_saved'] = df['cumulative_calories_saved'] / 7000

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

def plot_weight_over_time(df):
    """Plot weight changes over time using Plotly."""
    fig = px.line(df, x='date', y='weight', title='Weight Over Time', markers=True)
    fig.update_layout(xaxis_title='Date', yaxis_title='Weight (kg)')
    return fig

def plot_calorie_delta_over_time(df):
    """Plot calorie delta over time using Plotly."""
    fig = px.bar(df, x='date', y='calorie_delta', title='Calorie Delta Over Time', color='calorie_delta')
    fig.update_layout(xaxis_title='Date', yaxis_title='Calorie Delta', showlegend=False)
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    return fig

def plot_bmi_over_time(df):
    """Plot BMI changes over time using Plotly."""
    fig = px.line(df, x='date', y='BMI', title='BMI Over Time', markers=True, line_shape='linear')
    fig.update_layout(xaxis_title='Date', yaxis_title='BMI')
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
    """Predict the date when the target weight will be reached."""
    if df.empty:
        st.error("No data available to perform prediction.")
        return None, None, None

    initial_weight = df['weight'].iloc[0]  # Ensure initial weight is obtained here
    df['actual_kgs_saved'] = initial_weight - df['weight']
    df['days'] = (df['date'] - df['date'].min()).dt.days

    if df['days'].empty or df['actual_kgs_saved'].empty:
        st.error("Insufficient data for prediction.")
        return None, None, None

    # Prepare the data for regression
    X = df['days'].values.reshape(-1, 1)
    y = df['actual_kgs_saved'].values.reshape(-1, 1)

    if X.shape[0] == 0 or y.shape[0] == 0:
        st.error("Insufficient data points for regression.")
        return None, None, None

    # Perform polynomial regression
    poly = PolynomialFeatures(degree=2)  # Adjust degree as needed
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict future days needed to reach the target weight
    future_days = np.arange(0, 2 * df['days'].max()).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    predicted_kgs_saved = model.predict(future_days_poly)

    # Determine the day at which target weight is reached
    target_kgs_saved = initial_weight - target_weight
    target_day_index = np.argmax(predicted_kgs_saved >= target_kgs_saved)
    if target_day_index == 0 and predicted_kgs_saved[0] < target_kgs_saved:
        st.error("Model predicts target weight is not achievable with current trend.")
        return None, None, None

    target_day = future_days[target_day_index]
    predicted_date = df['date'].min() + datetime.timedelta(days=int(target_day))

    return predicted_date, model, poly

def plot_prediction(df, model, poly, target_weight):
    """Plot the prediction line along with actual kilograms lost."""
    if model is None or poly is None:
        return

    initial_weight = df['weight'].iloc[0]
    max_days = (predict_target_reach(df, target_weight)[0] - df['date'].min()).days

    future_days = np.arange(0, max_days).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    predicted_kgs_saved = model.predict(future_days_poly)

    future_dates = [df['date'].min() + datetime.timedelta(days=int(day)) for day in future_days]

    prediction_df = pd.DataFrame({
        'date': future_dates,
        'predicted_kgs_saved': predicted_kgs_saved.flatten()
    })

    fig = px.line(prediction_df, x='date', y='predicted_kgs_saved', title='Prediction of Weight Loss Over Time', labels={'predicted_kgs_saved': 'Kilograms Saved'})
    fig.add_scatter(x=df['date'], y=df['actual_kgs_saved'], mode='markers', name='Actual Kg\'s Saved')

    return fig

def main():
    """Main function to run the Streamlit app."""
    # Use Streamlit's wide mode to use more horizontal space
    st.set_page_config(layout="wide")
    st.title("Weight Tracker")

    # Sidebar for additional inputs and information
    st.sidebar.header("Settings")
    target_weight = st.sidebar.number_input("Goal Weight (kg)", min_value=0.0, value=75.0, step=0.1)
    height_m = st.sidebar.number_input("Height (m)", min_value=1.0, value=1.83, step=0.01)  # Allow decimal values for height

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

    # Calculate metrics for display
    if not df.empty:
        # Total kilograms lost since start
        kgs_lost_since_start = df['actual_kgs_saved'].iloc[-1]

        # Kilograms to go to reach target weight
        kgs_to_go = df['kgs_to_target'].iloc[-1]

        # Total calories saved
        calories_saved = df['cumulative_calories_saved'].iloc[-1]

        # Total calories to save to reach target weight
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

    # Create three tabs for different sections of the app
    tab1, tab2, tab3 = st.tabs(["Analysis", "Data", "Predictions"])

    with tab1:
        st.header("Analysis")
        if not df.empty:
            st.subheader("Summary Statistics")
            # Use AG Grid for displaying summary statistics
            summary_df = df.describe().reset_index()  # Reset index for better display
            gb = GridOptionsBuilder.from_dataframe(summary_df)
            gb.configure_pagination(paginationAutoPageSize=True)  # Pagination
            gb.configure_side_bar()  # Enable side bar for filtering and more
            grid_options = gb.build()

            # Use AG Grid with full width
            AgGrid(summary_df, gridOptions=grid_options, height=300, width='100%')
            # Add more analysis or visualizations as needed here

            # Add plots using Plotly
            st.subheader("Visualizations")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.plotly_chart(plot_weight_over_time(df), use_container_width=True)
            with col2:
                st.plotly_chart(plot_calorie_delta_over_time(df), use_container_width=True)
            with col3:
                st.plotly_chart(plot_bmi_over_time(df), use_container_width=True)

            # Full-width plot for theoretical vs actual kilograms saved
            st.plotly_chart(plot_kgs_saved(df), use_container_width=True)

        else:
            st.write("No data available for analysis.")

    with tab2:
        st.header("Data")
        if not df.empty:
            # Use AG Grid for displaying the main data
            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_pagination(paginationAutoPageSize=True)
            gb.configure_side_bar()
            grid_options = gb.build()

            # Use AG Grid with full width
            AgGrid(df, gridOptions=grid_options, height=400, width='100%')
        else:
            st.write("No data available.")

    with tab3:
        st.header("Predictions")
        if not df.empty:
            predicted_date, model, poly = predict_target_reach(df, target_weight)
            if predicted_date and model and poly:
                st.write(f"Predicted date to reach target weight: {predicted_date.date()}")
                st.plotly_chart(plot_prediction(df, model, poly, target_weight), use_container_width=True)
            else:
                st.write("Prediction model could not be created.")
        else:
            st.write("No data available for predictions.")

if __name__ == "__main__":
    main()
