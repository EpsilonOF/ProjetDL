import streamlit as st
import torch
import matplotlib.pyplot as plt
import numpy as np
from train_glow import setup_model, prepare_training_data, generate_samples, get_bits_per_dim, execute_glow

def show_glow():
    st.title("Glow: Generative Flow with Invertible 1x1 Convolutions")
    st.write("""
    Glow étend les concepts de RealNVP avec des convolutions inversibles
    et a été proposé par Kingma et Dhariwal en 2018.
    """)
    
    st.header("Principe")
    st.write("""
    Glow ajoute des convolutions 1x1 inversibles pour permettre une meilleure 
    permutation des dimensions, ce qui est particulièrement utile pour les données d'images.
    
    L'architecture Glow est composée de plusieurs blocs ayant chacun trois composantes principales:
    1. **Actnorm** - Normalisation par activation (généralisation de batch normalization)
    2. **Convolution 1x1 inversible** - Remplace l'étape de permutation utilisée dans RealNVP
    3. **Couplage affine** - Similaire à RealNVP mais avec une architecture améliorée
    """)
    
    # Affichage de l'architecture Glow
    st.image("images/glow/architecture.png", 
             caption="Architecture du modèle Glow",
             width=600)
             
    st.header("Composantes clés")
    
    st.subheader("1. Actnorm")
    st.write("""
    L'Actnorm (Activation Normalization) est une couche de normalisation par canaux similaire à 
    la batch normalization, mais avec des paramètres d'échelle et de décalage appris par canal.
    
    Pour une entrée h, l'opération est:
    """)
    st.latex(r'''
    y = s \odot (h + b)
    ''')
    st.write("""
    où s et b sont des paramètres appris et ⊙ représente la multiplication élément par élément.
    
    Cette couche est initialisée pour que la sortie ait une moyenne nulle et une variance unitaire,
    ce qui aide à stabiliser l'entraînement des réseaux profonds.
    """)
    
    st.subheader("2. Convolution 1x1 inversible")
    st.write("""
    Les convolutions 1x1 inversibles remplacent les permutations fixes utilisées dans RealNVP
    par une opération apprise. Cette opération peut être représentée par une matrice de poids W
    dont la taille est c × c (où c est le nombre de canaux).
    
    Pour une entrée h, l'opération est:
    """)
    st.latex(r'''
    y = W \cdot h
    ''')
    st.write("""
    
    Le logarithme du déterminant de la jacobienne est simplement:
    """)
    st.latex(r'''
    \log|\det(\frac{\partial y}{\partial h})| = \log|\det(W)|
    ''')
    st.write("""
    Le calcul naïf du déterminant aurait une complexité O(c³), mais Glow utilise une 
    décomposition LU pour réduire ce coût à O(c²), rendant l'opération efficace même pour 
    un grand nombre de canaux.
    """)
    
    st.subheader("3. Couplage affine")
    st.write("""
    La couche de couplage affine dans Glow est similaire à celle de RealNVP. Elle divise l'entrée
    en deux parties, applique une transformation à une partie conditionnée sur l'autre:
    """)
    st.latex(r'''
    h_a, h_b = \text{split}(h)
    \newline
    s, t = \text{NN}(h_a)
    \newline
    h_b' = s \odot h_b + t
    \newline
    y = \text{concat}(h_a, h_b')
    ''')
    st.write("""    
    où NN est un réseau de neurones. Contrairement à RealNVP, Glow utilise des réseaux plus 
    sophistiqués avec des convolutions, des normalisations et des activations non-linéaires.
    """)
    
    st.header("Architecture multi-échelle")
    st.write("""
    Glow utilise une architecture multi-échelle similaire à RealNVP, où:
    
    1. L'image est traitée à travers plusieurs étapes (steps of flow)
    2. À certains points, la représentation est divisée en deux, avec une partie envoyée directement à la sortie
    3. Ce processus permet de capturer les dépendances à différentes échelles spatiales
    
    Cette conception hierarchique permet à Glow de modéliser efficacement des images de haute résolution.
    """)
    
    st.header("Applications")
    st.write("""
    Glow a démontré d'excellentes performances dans plusieurs tâches:
    
    1. **Génération d'images** - Création d'images photoréalistes
    2. **Manipulation sémantique** - Modification d'attributs spécifiques (sourire, lunettes, âge, etc.)
    3. **Interpolation** - Transition fluide entre différentes images
    4. **Inpainting** - Complétion de parties manquantes d'images
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Avantages")
        st.write("""
        - Estimation exacte de la log-vraisemblance
        - Échantillonnage direct et rapide
        - Architecture inversible efficace
        - Manipulation précise d'attributs sémantiques
        - Meilleure qualité d'images que RealNVP
        """)
    
    with col2:
        st.subheader("Limitations")
        st.write("""
        - Complexité computationnelle élevée
        - Nécessite beaucoup de mémoire
        - Entraînement plus long que VAEs/GANs
        - Difficulté à modéliser certaines distributions très complexes
        - Résolution d'images limitée comparée aux GANs récents
        """)
    
    st.header("Formulation mathématique")
    st.latex(r'''
    \log p_X(x) = \log p_Z(f(x)) + \log \left| \det \frac{\partial f(x)}{\partial x} \right|
    ''')
    
    st.write("""
    où:
    - $p_X(x)$ est la densité de probabilité des données réelles
    - $p_Z(z)$ est la densité de probabilité de la distribution latente (généralement une gaussienne)
    - $f$ est la transformation inversible apprise
    - Le second terme est le logarithme du déterminant de la jacobienne de la transformation
    """)
    
    st.header("Démo interactive")
    st.write("""
    Expérimentez avec le modèle Glow en ajustant les paramètres ci-dessous. Le modèle a été pré-entraîné
    sur le jeu de données MNIST pour générer des chiffres manuscrits.
    """)

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Paramètres du modèle")
        n_flow = st.slider("Nombre de couches de flux par niveau (K)", min_value=1, max_value=32, value=12, step=1)
        n_blocks = st.slider("Nombre de niveaux (L)", min_value=1, max_value=8, value=4, step=1)
        hidden_channels = st.slider("Nombre de canaux cachés", min_value=64, max_value=512, value=256, step=64)
        
        dataset = st.selectbox(
            "Jeu de données",
            ["MNIST", "CIFAR10"]
        )
        
        use_lu = not st.checkbox("Désactiver la décomposition LU (plus lent mais potentiellement plus expressif)", value=False)
        
    with col2:
        st.subheader("Paramètres de génération")
        temperature = st.slider("Température (contrôle la diversité)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        num_samples = st.slider("Nombre d'échantillons par classe", min_value=1, max_value=10, value=4, step=1)
        
        # Ajouter des options de training si souhaité
        training_options = st.expander("Options d'entraînement", expanded=False)
        with training_options:
            train = st.checkbox("Réentraîner le modèle", value=False)
            if train:
                max_iter = st.slider("Nombre d'itérations d'entraînement", min_value=1000, max_value=20000, value=5000, step=1000)
            else:
                max_iter = 5000  # Valeur par défaut si pas d'entraînement

    # Section pour le bouton de génération
    generate_button = st.button("Générer des images", key="glow_generate")

    # Section pour afficher les résultats
    if generate_button:
        with st.spinner(f"Traitement en cours... {'Entraînement et ' if train else ''}Génération d'images avec Glow"):
            # Appel à la fonction execute_glow avec les paramètres sélectionnés
            samples, bpd = execute_glow(
                n_flow=n_flow,
                n_blocks=n_blocks,
                hidden_channels=hidden_channels,
                temperature=temperature,
                num_samples=num_samples,
                train=train,
                max_iter=max_iter,
                dataset=dataset
            )
            
            # Affichage des résultats
            if samples is not None:
                st.subheader("Images générées")
                st.image("images/glow/mnist_generated.png", caption=f"Images générées par Glow (température={temperature})")
                
                if bpd is not None:
                    st.write(f"Bits par dimension (mesure de compression): {bpd:.4f}")
                    st.write("""Plus cette valeur est basse, meilleure est la qualité du modèle génératif 
                             (meilleure modélisation de la distribution des données).""")
            else:
                st.error("Erreur lors de la génération des images. Vérifiez que le modèle pré-entraîné existe ou essayez de l'entraîner.")

    # Explication supplémentaire sur le fonctionnement de la manipulation
    st.header("Manipulation d'attributs sémantiques")
    st.write("""
    La manipulation d'attributs dans Glow fonctionne grâce à l'espace latent structuré:

    1. Le modèle apprend à transformer des données complexes en distribution gaussienne
    2. Dans cet espace latent, certaines directions correspondent à des attributs sémantiques
    3. En modifiant les vecteurs dans l'espace latent selon ces directions, on peut 
    ajuster des caractéristiques spécifiques
    4. La transformation inverse permet de générer de nouvelles images avec les attributs modifiés

    Cette approche permet une édition précise et contrôlée, contrairement aux GANs où 
    le contrôle précis est souvent plus difficile à obtenir.
    """)
    
    st.write("""
    Par exemple, sur le jeu de données CelebA (visages de célébrités), on peut modifier des attributs
    comme le sourire, l'âge, les lunettes ou le genre en ajoutant un vecteur de direction appris
    dans l'espace latent.
    """)

    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur Glow:
    
    - [Article original: "Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)
    - [Code source officiel d'OpenAI](https://github.com/openai/glow)
    - [Implémentation PyTorch de Glow](https://github.com/rosinality/glow-pytorch)
    - [Tutoriel sur les Normalizing Flows](https://lilianweng.github.io/posts/2018-10-13-flow-models/)
    """)