import streamlit as st
import os
from glow.train_glow import execute_glow


def show_glow():
    """
    Display the Glow model explanation page in the Streamlit application.

    This function creates a comprehensive presentation of the Glow model,
    including its principles, key components (Actnorm, Invertible 1x1 Convolutions,
    and Affine Coupling), multi-scale architecture, applications, advantages,
    limitations, and mathematical formulation.

    The page includes visualizations of the architecture and explanations of the
    mathematical concepts behind Glow.
    """
    st.title("Glow: Generative Flow with Invertible 1x1 Convolutions")
    st.write(
        """
    Glow étend les concepts de RealNVP avec des convolutions inversibles
    et a été proposé par Kingma et Dhariwal en 2018.
    """
    )

    st.header("Principe")
    st.write(
        """
    Glow ajoute des convolutions 1x1 inversibles pour permettre une meilleure
    permutation des dimensions, ce qui est particulièrement utile pour les données d'images.

    L'architecture Glow est composée de plusieurs blocs ayant chacun trois composantes principales:
    1. **Actnorm** - Normalisation par activation (généralisation de batch normalization)
    2. **Convolution 1x1 inversible** - Remplace l'étape de permutation utilisée dans RealNVP
    3. **Couplage affine** - Similaire à RealNVP mais avec une architecture améliorée
    """
    )

    # Affichage de l'architecture Glow
    image_path = os.path.join(os.path.dirname(__file__), "images", "architecture.png")
    st.image(image_path, caption="Architecture du modèle Glow", width=600)

    st.header("Composantes clés")

    st.subheader("1. Actnorm")
    st.write(
        """
    L'Actnorm (Activation Normalization) est une couche de normalisation par canaux similaire à
    la batch normalization, mais avec des paramètres d'échelle et de décalage appris par canal.

    Pour une entrée h, l'opération est:
    """
    )
    st.latex(
        r"""
    y = s \odot (h + b)
    """
    )
    st.write(
        """
    où s et b sont des paramètres appris et ⊙ représente la multiplication élément par élément.

    Cette couche est initialisée pour que la sortie ait une moyenne nulle et une variance unitaire,
    ce qui aide à stabiliser l'entraînement des réseaux profonds.
    """
    )

    st.subheader("2. Convolution 1x1 inversible")
    st.write(
        """
    Les convolutions 1x1 inversibles remplacent les permutations fixes utilisées dans RealNVP
    par une opération apprise. Cette opération peut être représentée par une matrice de poids W
    dont la taille est c × c (où c est le nombre de canaux).

    Pour une entrée h, l'opération est:
    """
    )
    st.latex(
        r"""
    y = W \cdot h
    """
    )
    st.write(
        """

    Le logarithme du déterminant de la jacobienne est simplement:
    """
    )
    st.latex(
        r"""
    \log|\det(\frac{\partial y}{\partial h})| = \log|\det(W)|
    """
    )
    st.write(
        """
    Le calcul naïf du déterminant aurait une complexité O(c³), mais Glow utilise une
    décomposition LU pour réduire ce coût à O(c²), rendant l'opération efficace même pour
    un grand nombre de canaux.
    """
    )

    st.subheader("3. Couplage affine")
    st.write(
        """
    La couche de couplage affine dans Glow est similaire à celle de RealNVP. Elle divise l'entrée
    en deux parties, applique une transformation à une partie conditionnée sur l'autre:
    """
    )
    st.latex(
        r"""
    h_a, h_b = \text{split}(h)
    \newline
    s, t = \text{NN}(h_a)
    \newline
    h_b' = s \odot h_b + t
    \newline
    y = \text{concat}(h_a, h_b')
    """
    )
    st.write(
        """
    où NN est un réseau de neurones. Contrairement à RealNVP, Glow utilise des réseaux plus
    sophistiqués avec des convolutions, des normalisations et des activations non-linéaires.
    """
    )

    st.header("Architecture multi-échelle")
    st.write(
        """
    Glow utilise une architecture multi-échelle similaire à RealNVP, où:

    1. L'image est traitée à travers plusieurs étapes (steps of flow)
    2. À certains points, la représentation est divisée en deux, avec une partie envoyée
    directement à la sortie
    3. Ce processus permet de capturer les dépendances à différentes échelles spatiales

    Cette conception hierarchique permet à Glow de modéliser efficacement des images de
    haute résolution.
    """
    )

    st.header("Applications")
    st.write(
        """
    Glow a démontré d'excellentes performances dans plusieurs tâches:

    1. **Génération d'images** - Création d'images photoréalistes
    2. **Manipulation sémantique** - Modification d'attributs spécifiques (sourire,
    lunettes, âge, etc.)
    3. **Interpolation** - Transition fluide entre différentes images
    4. **Inpainting** - Complétion de parties manquantes d'images
    """
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avantages")
        st.write(
            """
        - Estimation exacte de la log-vraisemblance
        - Échantillonnage direct et rapide
        - Architecture inversible efficace
        - Manipulation précise d'attributs sémantiques
        - Meilleure qualité d'images que RealNVP
        """
        )

    with col2:
        st.subheader("Limitations")
        st.write(
            """
        - Complexité computationnelle élevée
        - Nécessite beaucoup de mémoire
        - Entraînement plus long que VAEs/GANs
        - Difficulté à modéliser certaines distributions très complexes
        - Résolution d'images limitée comparée aux GANs récents
        """
        )

    st.header("Formulation mathématique")
    st.latex(
        r"""
    \log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}
    {\partial x} \right|
    """
    )

    st.write(
        """
    où:
    - $p_X(x)$ est la densité de probabilité des données réelles
    - $p_Z(z)$ est la densité de probabilité de la distribution latente
    (généralement une gaussienne)
    - $f$ est la transformation inversible apprise
    - Le second terme est le logarithme du déterminant de la jacobienne
    de la transformation
    """
    )

    # Explication supplémentaire sur le fonctionnement de la manipulation
    st.header("Manipulation d'attributs sémantiques")
    st.write(
        """
    La manipulation d'attributs dans Glow fonctionne grâce à l'espace latent structuré:

    1. Le modèle apprend à transformer des données complexes en distribution gaussienne
    2. Dans cet espace latent, certaines directions correspondent à des attributs sémantiques
    3. En modifiant les vecteurs dans l'espace latent selon ces directions, on peut
    ajuster des caractéristiques spécifiques
    4. La transformation inverse permet de générer de nouvelles images avec les attributs modifiés

    Cette approche permet une édition précise et contrôlée, contrairement aux GANs où
    le contrôle précis est souvent plus difficile à obtenir.
    """
    )

    st.write(
        """
    Par exemple, sur le jeu de données CelebA (visages de célébrités), on peut
    modifier des attributs comme le sourire, l'âge, les lunettes ou le genre en
    ajoutant un vecteur de direction appris dans l'espace latent.
    """
    )

    st.header("Tester le modèle Glow")
    st.write(
        """
    Expérimentez avec le modèle Glow en ajustant différents hyperparamètres:
    """
    )

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Hyperparamètres")
        L = st.slider(
            "Nombre de niveaux multi-échelles (L)",
            min_value=1,
            max_value=4,
            value=2,
            step=1,
        )
        K = st.slider(
            "Nombre de blocs Glow par niveau (K)",
            min_value=4,
            max_value=32,
            value=16,
            step=4,
        )

        generate_button = st.button("Générer", key="glow_generate")

    with col2:
        st.subheader("Paramètres d'entraînement")

        batch_size = st.slider(
            "Taille de batch",
            min_value=32,
            max_value=256,
            value=128,
            step=32,
        )

        max_iter = st.slider(
            "Itérations d'entraînement",
            min_value=1000,
            max_value=20000,
            value=5000,
            step=1000,
        )
    st.write(
        """
    Attention : Glow est un modèle destiné à traiter des images avec une distribution complexe.
    Ainsi, le tester sur TwoMoons comme les autres modèles n'aurait pas d'intérêt.
    L'éxécution peut donc être très longue sans une grosse puissance de calcul (plusieurs heures).
    """
    )
    # Section pour afficher les résultats
    if generate_button:
        with st.spinner("Entraînement en cours..."):
            execute_glow(
                L=L,
                K=K,
                batch_size=batch_size,
                max_iter=max_iter,
                save_path="modele.pth",
            )

    st.header("Ressources")
    st.write(
        """
    Pour approfondir vos connaissances sur Glow:

    - [Article original: "Glow: Generative Flow with Invertible 1x1
    Convolutions"](https://arxiv.org/abs/1807.03039)
    - [Code source officiel d'OpenAI](https://github.com/openai/glow)
    """
    )
