import streamlit as st
from train_glow import execute_glow

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
    
    st.header("Composantes clés")
    
    st.subheader("1. Actnorm")
    st.write("""
    L'Actnorm (Activation Normalization) est une couche de normalisation par canaux similaire à 
    la batch normalization, mais avec des paramètres d'échelle et de décalage appris par canal.
    
    Pour une entrée h, l'opération est:
    """)
    st.latex(r'''
    y = s ⊙ (h + b)
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
    dont la taille est c * c (où c est le nombre de canaux).
    
    Pour une entrée h, l'opération est:
    """)
    st.latex(r'''
    y = W \cdot h
    ''')
    st.write("""
    
    Le logarithme du déterminant de la jacobienne est simplement:
    """)
    st.latex(r'''
    log|det(\frac{\partial y}{\partial h})| = log|det(W)|
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
    h_a, h_b = split(h)
    \newline
    s, t = NN(h_a)
    \newline
    h_b' = s ⊙ h_b + t
    \newline
    y = concat(h_a, h_b')
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
        """)
    
    with col2:
        st.subheader("Limitations")
        st.write("""
        - Complexité computationnelle élevée
        - Nécessite beaucoup de mémoire
        - Architecture moins flexible que les VAEs ou GANs
        - Difficulté à modéliser certaines distributions très complexes
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
    
    st.header("Implémentation et démo interactive")
    st.write("""
    Vous pouvez expérimenter avec le modèle Glow ci-dessous en ajustant différents paramètres:
    """)

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Paramètres du modèle")
        n_flow = st.slider("Nombre de couches de flux", min_value=1, max_value=32, value=12, step=1)
        n_blocks = st.slider("Nombre de blocs", min_value=1, max_value=8, value=4, step=1)
        no_lu = st.checkbox("Désactiver la décomposition LU (plus lent mais potentiellement plus précis)", value=False)
        
        st.subheader("Paramètres de génération")
        temperature = st.slider("Température (contrôle la variabilité)", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        batch_size = st.slider("Nombre d'images à générer", min_value=1, max_value=16, value=4, step=1)
        
        seed = st.number_input("Seed pour la génération aléatoire", min_value=0, max_value=9999, value=42)
        
        generate_button = st.button("Générer des images")

    with col2:
        st.subheader("Manipulation d'attributs")
        st.write("Ajustez les attributs sémantiques pour modifier les images générées:")
        
        sourire = st.slider("Sourire", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        age = st.slider("Âge", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        lunettes = st.slider("Lunettes", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        genre = st.slider("Genre", min_value=-2.0, max_value=2.0, value=0.0, step=0.1)
        
        attributs = {
            "Sourire": sourire,
            "Âge": age,
            "Lunettes": lunettes,
            "Genre": genre
        }

    # Section pour afficher les résultats
    st.subheader("Résultats de la génération")

    if generate_button:
        with st.spinner("Génération d'images en cours..."):
            execute_glow()
            
            # Affichage des résultats (simulations avec des placeholders)
            # st.write(f"Images générées avec température={temperature}, attributs appliqués:")
            
            # # Créer une grille d'images générées
            # image_cols = st.columns(min(4, batch_size))
            # for i in range(batch_size):
            #     col_idx = i % 4
            #     with image_cols[col_idx]:
            #         # Dans une implémentation réelle, vous afficheriez les vraies images générées
            #         # Pour l'instant, utilisons des placeholders
            #         st.image(f"https://via.placeholder.com/150?text=Glow+Image+{i+1}", 
            #                 caption=f"Image {i+1}")
                    
            #         st.write(f"Paramètres appliqués:")
            #         for attr, val in attributs.items():
            #             if abs(val) > 0.1:  # N'afficher que les attributs significativement modifiés
            #                 st.write(f"- {attr}: {val:+.1f}")


    # Explication supplémentaire sur le fonctionnement de la manipulation
    st.header("Comment fonctionne la manipulation d'attributs")
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
    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur Glow:
    
    - [Article original: "Glow: Generative Flow with Invertible 1x1 Convolutions"](https://arxiv.org/abs/1807.03039)
    - [Code source officiel d'OpenAI](https://github.com/openai/glow)
    - [Tutoriel PyTorch sur l'implémentation de Glow](https://github.com/rosinality/glow-pytorch)
    """)