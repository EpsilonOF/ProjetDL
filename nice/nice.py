import streamlit as st
import os
from nice.train_nice import execute_nice


def show_nice():
    """
    Display the NICE model explanation page in the Streamlit application.

    This function creates a comprehensive presentation of the NICE (Non-linear
    Independent Components Estimation) model, including its principles, architecture,
    mathematical properties, strengths, limitations, and historical impact.

    The page also includes an interactive component allowing users to experiment
    with the NICE model by adjusting hyperparameters and visualizing results on
    different datasets.
    """
    st.title("NICE: Non-linear Independent Components Estimation")
    st.write(
        """
    NICE (Non-linear Independent Components Estimation) est un des premiers modèles
    de Normalizing Flow proposé par Dinh et al. en 2014. Il introduit le concept de
    couplage additif pour permettre des transformations inversibles avec un jacobien
    facilement calculable.
    """
    )

    st.header("Principe")
    st.write(
        """
    NICE transforme des données complexes en un espace latent avec une distribution simple
    via une série de transformations inversibles. La particularité de NICE est d'utiliser
    des transformations à jacobien triangulaire, permettant un calcul efficace du déterminant.
    """
    )

    st.header("Architecture")
    st.write(
        """
    Dans NICE, la transformation est définie comme suit:
    1. L'entrée x est divisée en deux parties (x₁, x₂)
    2. La première partie x₁ reste inchangée
    3. La seconde partie est transformée: x₂' = x₂ + m(x₁)
    4. Où m est une fonction arbitraire (généralement un réseau de neurones)
    5. La sortie de la couche est (x₁, x₂')
    """
    )

    st.latex(
        r"""
    \begin{align}
    y_1 &= x_1 \\
    y_2 &= x_2 + m(x_1)
    \end{align}
    """
    )

    st.header("Propriétés mathématiques")
    st.write(
        """
    NICE présente plusieurs propriétés mathématiques importantes:

    1. **Inversibilité**: La transformation est facilement inversible:
    """
    )

    st.latex(
        r"""
    \begin{align}
    x_1 &= y_1 \\
    x_2 &= y_2 - m(y_1)
    \end{align}
    """
    )

    st.write(
        """
    2. **Déterminant du Jacobien**: La matrice jacobienne de cette transformation
    est triangulaire, ce qui implique que son déterminant est simplement le produit
    des éléments diagonaux, qui sont tous égaux à 1.
    """
    )

    st.latex(
        r"""
    \det \left( \frac{\partial y}{\partial x} \right) = 1
    """
    )

    st.write(
        """
    3. **Conservation du volume**: NICE est une transformation préservant le volume,
       ce qui simplifie le calcul de la vraisemblance.
    """
    )

    st.header("Couches de scaling")
    st.write(
        """
    Après plusieurs couches de couplage additif, NICE utilise une transformation de mise à l'échelle
    (scaling) pour permettre une expression plus riche des distributions:
    """
    )

    st.latex(
        r"""
    y = \exp(s) \odot x
    """
    )

    st.write(
        """
    où s est un vecteur de paramètres apprenables et ⊙ est un produit élément par élément.
    """
    )

    st.header("Forces et limitations")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Forces")
        st.write(
            """
        - Inversibilité exacte et efficace
        - Calcul simple du jacobien (déterminant = 1)
        - Efficacité computationnelle
        - Apprentissage stable
        - Premier modèle démontrant la viabilité des Normalizing Flows
        """
        )

    with col2:
        st.subheader("Limitations")
        st.write(
            """
        - Expressivité limitée par les transformations additives
        - Nécessite beaucoup de couches pour modéliser des distributions complexes
        - Pas d'adaptation du volume local (contrairement à RealNVP)
        - Performance limitée sur les données de haute dimension
        """
        )

    st.header("Impact historique")
    st.write(
        """
    NICE a posé les fondations théoriques et pratiques des Normalizing Flows modernes.
    Son approche de couplage a inspiré de nombreuses améliorations, notamment RealNVP
    et Glow, qui ont étendu ses principes pour créer des modèles plus expressifs.

    Ce modèle a ouvert la voie à une nouvelle classe de modèles génératifs avec une
    vraisemblance exactement calculable, contrairement aux VAE et aux GAN.
    """
    )

    st.header("Tester le modèle NICE")
    st.write(
        """
    Expérimentez avec le modèle NICE en ajustant différents hyperparamètres et en choisissant
    différents jeux de données:
    """
    )

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Hyperparamètres")
        n_layers = st.slider(
            "Nombre de couches de couplage",
            min_value=2,
            max_value=12,
            value=4,
            step=1,
        )
        hidden_dim = st.slider(
            "Nombre de neurones cachés",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
        )

        generate_button = st.button("Générer", key="nice_generate")

    with col2:
        st.subheader("Paramètres d'entraînement")
        max_iter = st.slider(
            "Itérations d'entraînement",
            min_value=1000,
            max_value=5000,
            value=2000,
            step=500,
        )

    # Section pour afficher les résultats
    if generate_button:
        with st.spinner("Génération en cours..."):
            image_path = os.path.join(
                os.path.dirname(__file__), "images", "twomoons.png"
            )
            st.image(image_path, width=200)
            execute_nice(n_layers, hidden_dim, max_iter)
            st.write(
                f"""
            Modèle NICE avec {n_layers} couches de couplage,
            {hidden_dim} neurones cachés, et {max_iter} itérations d'entraînement
            """
            )

    st.header("Application à des données complexes")
    st.write(
        """
    NICE peut être appliqué à des problèmes plus complexes que les distributions 2D,
    comme la génération d'images. Pour les données de haute dimension comme MNIST
    (28×28 = 784 dimensions), NICE utilise la même architecture avec des couches
    de couplage additives, mais avec un réseau plus large et plus profond pour
    capturer les dépendances complexes.

    Bien que NICE fonctionne sur MNIST, sa performance est limitée par sa nature
    transformative additive. Les modèles plus récents comme RealNVP et Glow offrent une
    meilleure expressivité pour modéliser des distributions plus complexes grâce à leurs
    transformations affines et convolutives.
    """
    )

    # Espace réservé pour afficher d'autres applications dans le futur

    st.header("Ressources")
    st.write(
        """
    Pour approfondir vos connaissances sur NICE:

    - [Article original: "NICE: Non-linear Independent
    Components Estimation"](https://arxiv.org/abs/1410.8516)
    """
    )
