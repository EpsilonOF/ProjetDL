import streamlit as st
from presentation import show_presentation
from nice.nice import show_nice
from realnvp.real_nvp import show_realnvp
from glow.glow import show_glow


def main():
    """
    Main application function for the Normalizing Flows demonstration app.

    This function configures the Streamlit page, sets up the navigation sidebar,
    and displays the appropriate content based on user selection.
    """
    # Configuration de la page
    st.set_page_config(page_title="Normalizing Flows", page_icon="🔄", layout="wide")

    # Sidebar pour la navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Aller à:", ["Présentation", "NICE", "RealNVP", "Glow"])

    # Affichage des pages en fonction de la sélection
    if page == "Présentation":
        show_presentation()
    elif page == "NICE":
        show_nice()
    elif page == "RealNVP":
        show_realnvp()
    elif page == "Glow":
        show_glow()


if __name__ == "__main__":
    main()
