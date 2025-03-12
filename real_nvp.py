import streamlit as st

def show_realnvp():
    st.title("RealNVP: Real-valued Non-Volume Preserving")
    st.write("""
    RealNVP (Real-valued Non-Volume Preserving) est une extension de NICE 
    proposée par Dinh et al. en 2017, apportant des améliorations significatives
    en termes d'expressivité et de qualité des échantillons générés.
    """)
    
    st.header("Principe")
    st.write("""
    RealNVP introduit des transformations affines par blocs (multiplication et addition)
    ce qui rend le modèle plus expressif que NICE. La principale innovation est le passage
    d'une transformation purement additive à une transformation affine complète, permettant
    non seulement de translater mais aussi de rééchelonner les données.
    """)
    
    st.header("Architecture")
    st.write("""
    Dans RealNVP, la transformation affine est définie comme suit:
    1. L'entrée x est divisée en deux parties (x₁, x₂)
    2. La première partie x₁ reste inchangée
    3. La seconde partie subit une transformation affine:
       x₂' = x₂ ⊙ exp(s(x₁)) + t(x₁)
    4. Où s et t sont des fonctions arbitraires (généralement des réseaux de neurones)
    5. La sortie de la couche est (x₁, x₂')
    """)
    
    st.latex(r'''
    \begin{align}
    y_1 &= x_1 \\
    y_2 &= x_2 \odot \exp(s(x_1)) + t(x_1)
    \end{align}
    ''')
    
    st.header("Innovations architecturales")
    st.write("""
    RealNVP apporte plusieurs innovations architecturales notables:
    
    1. **Transformations affines**: Contrairement aux transformations additives de NICE,
       les transformations affines permettent des variations locales du volume, améliorant
       nettement la capacité expressive du modèle.
       
    2. **Masques alternés**: Introduction de masques en damier (checkerboard masks) pour 
       les premières couches et des masques par canaux (channel-wise) aux étapes suivantes,
       permettant une meilleure alternance et propagation de l'information.
       
    3. **Architecture multi-échelle**: RealNVP adopte une architecture multi-échelle qui
       divise le traitement en plusieurs niveaux, permettant de capturer des caractéristiques
       à différentes échelles spatiales.
    """)
    
    st.header("Calcul du Jacobien")
    st.write("""
    Comme pour NICE, la matrice jacobienne de RealNVP est triangulaire, mais le déterminant
    n'est plus égal à 1 en raison de la composante multiplicative:
    """)
    
    st.latex(r'''
    \log \left| \det \frac{\partial y}{\partial x} \right| = \sum_i s_i(x_1)
    ''')
    
    st.write("""
    Cette formulation permet toujours un calcul efficace du déterminant, nécessaire pour
    l'optimisation de la log-vraisemblance.
    """)
    
    st.header("Application aux images")
    st.write("""
    RealNVP est particulièrement efficace pour la modélisation d'images grâce à:
    
    1. **Masques en damier**: Ces masques divisent l'image en motif de damier, permettant
       d'exploiter les corrélations spatiales locales.
       
    2. **Masques par canaux**: Après plusieurs couches, RealNVP utilise des masques qui
       séparent les canaux de l'image, permettant de modéliser les dépendances inter-canaux.
       
    3. **Squeezing**: Technique qui réorganise les pixels spatialement pour augmenter
       le nombre de canaux, facilitant les transformations affines par canaux.
    """)
    
    # Illustration des masques
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Masque en damier")
        st.image("images/masque_damier.png", 
                caption="Illustration d'un masque en damier",
                width=200)
    
    with col2:
        st.subheader("Masque par canaux")
        st.image("images/masque_canaux.png", 
                caption="Illustration d'un masque par canaux",
                width=200)
    
    st.header("Forces et limitations")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Forces")
        st.write("""
        - Transformations affines plus expressives que NICE
        - Modélisation efficace des variations locales de volume
        - Architecture multi-échelle adaptée aux images
        - Meilleure qualité d'échantillons générés
        - Préservation d'un calcul efficace du Jacobien
        """)
    
    with col2:
        st.subheader("Limitations")
        st.write("""
        - Moins expressif que les modèles ultérieurs comme Glow
        - Permutations fixes limitant la flexibilité
        - Nécessite encore un grand nombre de couches pour les distributions très complexes
        - Performance limitée sur les images haute résolution
        """)
    
    st.header("Impact")
    st.write("""
    RealNVP a considérablement amélioré les performances des Normalizing Flows et a
    étendu leur domaine d'application. Ce modèle a démontré que les Normalizing Flows
    pouvaient générer des échantillons de haute qualité pour des données complexes
    comme les images naturelles. Les principes architecturaux introduits par RealNVP
    ont directement influencé les modèles ultérieurs, notamment Glow qui a poussé
    encore plus loin ces innovations.
    """)
    
    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur RealNVP:
    
    - [Article original: "Density estimation using Real NVP"](https://arxiv.org/abs/1605.08803)
    - [Implémentation de référence en TensorFlow](https://github.com/tensorflow/models/tree/master/research/real_nvp)
    """)