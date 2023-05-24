import streamlit as st
import pandas as pd
import plotly.express as px

def main():
    st.title("Data Visualization with Streamlit and Plotly")

    # Create a sample DataFrame
    data = {
        'Year': [2010, 2011, 2012, 2013, 2014, 2015],
        'Sales': [50, 70, 80, 65, 90, 120]
    }
    df = pd.DataFrame(data)

    # Display the DataFrame
    st.subheader("Data")
    st.dataframe(df)

    # Create a line chart using Plotly
    fig_line = px.line(df, x='Year', y='Sales')
    st.subheader("Line Chart")
    st.plotly_chart(fig_line)

    # Create a bar chart using Plotly
    fig_bar = px.bar(df, x='Year', y='Sales')
    st.subheader("Bar Chart")
    st.plotly_chart(fig_bar)

if __name__ == "__main__":
    main()