import streamlit as st


def show_presentation():
    """
    Display the main presentation page for Normalizing Flows.

    This function creates the content for the introduction page of the application,
    including explanations about the concept of normalizing flows, their mathematical
    foundations, and an overview of the models presented in the application.
    """
    st.title("Les Normalizing Flows")

    st.header("Introduction")
    st.write(
        """
     Les normalizing flows sont une famille de modèles génératifs qui permettent
     de transformer une distribution simple (comme une gaussienne) en une distribution
     complexe via une série de transformations inversibles.
     """
    )

    st.header("Principe")
    st.write(
        """
     Le concept principal des normalizing flows repose sur le théorème de changement
     de variable. Étant donné une variable aléatoire $z$ avec une densité de probabilité
     connue $p(z)$ et une fonction bijective $f$, la densité de la variable transformée
     $x = f(z)$ peut être calculée comme:

     $p(x) = p(z) | \\det(\\frac{\\partial f}{\\partial z})^{-1}|$

     où $\\frac{\\partial f}{\\partial z}$ est la jacobienne de la transformation $f$.
     """
    )

    st.header("Caractéristiques principales")
    st.write(
        """
     - **Inversibilité**: les transformations doivent être bijectives
     - **Calcul efficace**: le déterminant de la jacobienne doit être calculable efficacement
     - **Expressivité**: capacité à représenter des distributions complexes
     """
    )

    st.header("Modèles présentés")
    st.write(
        """
     Dans cette application, nous allons explorer trois architectures importantes
     de normalizing flows:
     - **NICE**: Non-linear Independent Components Estimation
     - **RealNVP**: Real-valued Non-Volume Preserving transformations
     - **Glow**: Generative Flow with Invertible 1x1 Convolutions

     Utilisez la barre latérale pour naviguer entre ces différents modèles.
     """
    )
