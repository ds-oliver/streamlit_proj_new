import streamlit as st

from functions import add_construction

st.set_page_config(
    layout="wide"
)

def main():
    
    add_construction()

    st.title('Fantasy Soccer Data Science Hub')
    st.write("""
    Welcome to the Fantasy Soccer Data Science Hub! Explore, analyze, and gain insights from the world of soccer with our suite of tools, data visualizations, and predictive models.
    """)

    st.header('Features')
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader('Player Analysis')
        st.write('Analyze player performance, compare statistics, visualize trends, and discover hidden gems.')
        st.button('Explore Players')

    with col2:
        st.subheader('Team Insights')
        st.write('Dive into team-level metrics, understand playing styles, strengths, weaknesses, and more.')
        st.button('Explore Teams')

    with col3:
        st.subheader('Fantasy Tools')
        st.write('Use our predictive models to build optimal line-ups, simulate matches, and make informed decisions.')
        st.button('Explore Fantasy Tools')

    st.header('Datasets')
    st.write("""
    Our platform provides access to comprehensive datasets including player statistics, match results, historical performance, and more. Analyze data from top leagues around the world or dive into specific player performance metrics.
    """)

    st.header('Get Started')
    st.write("""
    Navigate through our platform using the buttons above or the sidebar menu. Whether you're a soccer fan, fantasy manager, or data enthusiast, we have something for you!
    """)

if __name__ == "__main__":
    main()
