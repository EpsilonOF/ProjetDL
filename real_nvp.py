import streamlit as st
import normflows as nf
import torch
import matplotlib.pyplot as plt
from train_realnvp import setup_model, target_distribution, train_model

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
        st.image("images/realnvp/masque_damier.png", 
                caption="Illustration d'un masque en damier",
                width=200)
    
    with col2:
        st.subheader("Masque par canaux")
        st.image("images/realnvp/masque_canaux.png", 
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

    st.header("Tester le modèle RealNVP")
    st.write("""
    Expérimentez avec le modèle RealNVP en ajustant différents hyperparamètres:
    """)

    # Disposition en colonnes pour les paramètres et résultats
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Hyperparamètres")
        n_blocks = st.slider("Nombre de blocs de couplage", min_value=2, max_value=12, value=6, step=1)
        hidden_channels = st.slider("Nombre de canaux cachés", min_value=32, max_value=256, value=128, step=32)
        
        st.subheader("Architecture")
        mask_type = st.radio(
            "Type de masque",
            ["Damier", "Par canaux", "Mixte"]
        )
        
        multi_scale = st.checkbox("Architecture multi-échelle", value=True)
        batch_norm = st.checkbox("Utiliser BatchNorm", value=True)
        
        st.subheader("Données")
        dataset = st.selectbox(
            "Jeu de données",
            ["MNIST", "CIFAR-10", "CelebA", "Distribution 2D", "Image personnalisée"]
        )
        if dataset=="Image personnalisée":
            uploaded_file = st.file_uploader("Télécharger une image", type=["jpg", "jpeg", "png"])
        
        generate_button = st.button("Générer", key="realnvp_generate")

    with col2:
        st.subheader("Paramètres d'échantillonnage")
        temperature = st.slider("Température", min_value=0.1, max_value=1.0, value=0.7, step=0.1)
        n_samples = st.slider("Nombre d'échantillons", min_value=1, max_value=16, value=4, step=1)
        max_iter = st.slider("Itérations d'entraînement", min_value=1000, max_value=5000, value=2000, step=500)
        if dataset == "Distribution 2D":
            distribution = st.selectbox(
                "Type de distribution 2D",
                ["Gaussienne 2D", "Mélange de gaussiennes", "Spirale", "Anneaux", "Damier"]
            )

    # Section pour afficher les résultats
    if generate_button:
        with st.spinner("Génération en cours..."):
            image_path = None
            if uploaded_file is not None:
                with open("images/realnvp/uploaded_image.png", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                    image_path = "images/realnvp/uploaded_image.png"
                
                # Afficher l'image
                st.image(uploaded_file, caption="Image téléchargée", width=200)
            # Configuration des paramètres utilisateur
            num_layers = n_blocks * 2  # Chaque bloc de couplage est souvent pair
            show_iter = int(max_iter/10)   # Affichage intermédiaire

            # Initialisation du modèle avec les hyperparamètres de l'utilisateur
            model, device, target = setup_model(image_path, num_layers=num_layers)  # image_path utilisé si un dataset est image-based
            xx, yy, zz = target_distribution(target, device)

            loss_hist = []
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

            cols = st.columns(int((max_iter // show_iter)/2))  # Colonnes pour affichage des miniatures
            cols2 = cols
            for it in range(max_iter):
                optimizer.zero_grad()
                x = target.sample(512).to(device)
                loss = model.forward_kld(x)

                if not (torch.isnan(loss) or torch.isinf(loss)):
                    loss.backward()
                    optimizer.step()

                loss_hist.append(loss.item())

                if (it + 1) % show_iter == 0:
                    model.eval()
                    with torch.no_grad():
                        log_prob = model.log_prob(zz)
                    prob = torch.exp(log_prob.to('cpu').view(*xx.shape))
                    prob[torch.isnan(prob)] = 0

                    fig, ax = plt.subplots(figsize=(3, 3))
                    ax.pcolormesh(xx, yy, prob.numpy(), cmap='coolwarm')
                    ax.set_aspect('equal', 'box')
                    ax.axis('off')
                    ax.set_title(f"Itération {it+1}", fontsize=10)
                    if it//show_iter>=len(cols):
                        cols2[(it // show_iter)%len(cols)].pyplot(fig)
                    else:
                        cols[it // show_iter].pyplot(fig)
                    plt.close(fig)  # Libération mémoire

                    model.train()

            # Affichage final de la courbe de perte
            fig_loss, ax_loss = plt.subplots()
            ax_loss.plot(loss_hist, label="Perte")
            ax_loss.set_xlabel("Itération")
            ax_loss.set_ylabel("Perte")
            ax_loss.set_title("Historique de la perte")
            ax_loss.legend()
            st.pyplot(fig_loss)

            st.write(f"Modèle RealNVP avec {n_blocks} blocs, masque '{mask_type}', et jeu de données '{dataset}'")
            
    
    st.header("Ressources")
    st.write("""
    Pour approfondir vos connaissances sur RealNVP:
    
    - [Article original: "Density estimation using Real NVP"](https://arxiv.org/abs/1605.08803)
    - [Implémentation de référence en TensorFlow](https://github.com/tensorflow/models/tree/master/research/real_nvp)
    """)