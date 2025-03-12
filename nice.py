import streamlit as st

def show_nice():
    st.title("NICE: Non-linear Independent Components Estimation")
    st.write("""
    Le modèle NICE (Non-linear Independent Components Estimation) est l'un des premiers 
    normalizing flows proposés par Dinh et al. en 2015.
    """)
    
    st.header("Principe")
    st.write("""
    NICE propose des transformations additives par blocs qui sont facilement inversibles.
    L'idée centrale est d'utiliser des couches de couplage additives qui séparent les 
    variables d'entrée en deux parties : l'une reste inchangée tandis que l'autre subit 
    une transformation non-linéaire conditionnée par la première.
    """)
    
    st.header("Architecture")
    st.write("""
    Dans NICE, la transformation f est composée de plusieurs couches de couplage additives.
    Pour chaque couche:
    1. L'entrée x est divisée en deux parties (x₁, x₂)
    2. La première partie x₁ reste inchangée
    3. La seconde partie est transformée: x₂' = x₂ + m(x₁)
    4. Où m est une fonction arbitraire (généralement un réseau de neurones)
    5. La sortie de la couche est (x₁, x₂')
    """)
    
    st.latex(r'''
    \begin{align}
    y_1 &= x_1 \\
    y_2 &= x_2 + m(x_1)
    \end{align}
    ''')
    
    st.header("Jacobien")
    st.write("""
    L'un des avantages majeurs de cette approche est la simplicité du calcul du Jacobien, 
    qui est triangulaire, rendant son déterminant trivialement égal à 1. Cette caractéristique 
    facilite grandement l'optimisation directe par maximisation de la log-vraisemblance.
    
    La matrice jacobienne de cette transformation est:
    """)
    
    st.latex(r'''
    \frac{\partial y}{\partial x} = 
    \begin{pmatrix} 
    I_d & 0 \\
    \frac{\partial m(x_1)}{\partial x_1} & I_d
    \end{pmatrix}
    ''')
    
    st.write("""
    Le déterminant de cette matrice est 1, ce qui simplifie considérablement le calcul 
    de la log-vraisemblance.
    """)
    
    st.header("Forces et limitations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forces")
        st.write("""
        - Simplicité conceptuelle
        - Calcul du Jacobien trivial (det = 1)
        - Transformations facilement inversibles
        - Optimisation directe par maximisation de la log-vraisemblance
        """)
    
    with col2:
        st.subheader("Limitations")
        st.write("""
        - Transformations additives limitées par leur préservation du volume
        - Expressivité réduite comparée aux modèles ultérieurs
        - Nécessite de nombreuses couches pour modéliser des distributions complexes
        """)
    
    st.header("Impact")
    st.write("""
    Malgré ses limitations, NICE a posé les bases fondamentales des Normalizing Flows modernes,
    en démontrant qu'il était possible de transformer efficacement une distribution simple en
    une distribution complexe via des transformations inversibles dont le Jacobien est calculable
    efficacement. Ces principes fondamentaux ont inspiré directement RealNVP et par extension Glow.
    """)
    
    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur NICE:
    
    - [Article original: "NICE: Non-linear Independent Components Estimation"](https://arxiv.org/abs/1410.8516)
    """)