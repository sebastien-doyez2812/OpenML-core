# Bienvenue sur OpenML 🚀

**OpenML** est une bibliothèque Python open-source modulaire et extensible dédiée au Deep Learning et à la vision par ordinateur, construite par-dessus PyTorch. 

Elle a pour objectif de standardiser l'entraînement des modèles (comme les architectures U-Net), de simplifier la gestion des pertes personnalisées et de faciliter l'expérimentation en recherche et en milieu industriel.

---

## 🛠️ Fonctionnalités principales

* **Architecture modulaire :** Une classe `BaseModel` unifiée pour gérer l'entraînement, la sauvegarde et l'intégration avec TensorBoard.
* **Gestion avancée des pertes :** Une collection de pertes prêtes à l'emploi (Dice, Focal, Huber, KL Divergence, etc.) et une `LossFactory` pour les instancier dynamiquement.
* **Pertes combinées :** Possibilité de créer des pertes composites pondérées (`CustomLoss`) pour des cas d'usage complexes en segmentation.
* **Extensibilité :** Conçue pour être facilement enrichie avec de nouveaux modèles de vision par ordinateur.

---

## 📦 Installation

Clone le dépôt et installe la bibliothèque en mode éditable :

```bash
git clone [https://github.com/sebastien-doyez2812/OpenML.git](https://github.com/sebastien-doyez2812/OpenML.git)
cd OpenML
pip install -e .