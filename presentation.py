import streamlit as st

def show_presentation():
    st.title("Les Normalizing Flows")
    
    st.header("Introduction")
    st.write("""
    Les normalizing flows sont une famille de modèles génératifs qui permettent 
    de transformer une distribution simple (comme une gaussienne) en une distribution 
    complexe via une série de transformations inversibles.
    """)
    
    st.header("Principe")
    st.write("""
    Le concept principal des normalizing flows repose sur le théorème de changement 
    de variable. Étant donné une variable aléatoire z avec une densité de probabilité 
    connue p(z) et une fonction bijective f, la densité de la variable transformée 
    x = f(z) peut être calculée comme:
    """)
    st.latex(r'''
    p(x) = p(z) | det(\frac{\partial f}{\partial z})^{-1} |
    ''')
    st.write("""
    où ∂f/∂z est la jacobienne de la transformation f.
    """)
    
    st.header("Caractéristiques principales")
    st.write("""
    - **Inversibilité**: les transformations doivent être bijectives
    - **Calcul efficace**: le déterminant de la jacobienne doit être calculable efficacement
    - **Expressivité**: capacité à représenter des distributions complexes
    """)
    
    st.header("Modèles présentés")
    st.write("""
    Dans cette application, nous allons explorer trois architectures importantes 
    de normalizing flows:
    - **NICE**: Non-linear Independent Components Estimation
    - **RealNVP**: Real-valued Non-Volume Preserving transformations
    - **Glow**: Generative Flow with Invertible 1x1 Convolutions
    
    Utilisez la barre latérale pour naviguer entre ces différents modèles.
    """)