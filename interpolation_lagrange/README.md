# Numerical Methods with PyTorch
Implémentation des algorithmes classiques d'Analyse Numérique assistée par GPU.

# À propos du projet
Ce projet vise à revisiter les fondamentaux mathématiques du programme de Licence en Informatique (Algèbre Linéaire et Analyse Numérique), en les modernisant avec les outils de l'Intelligence Artificielle.
Au lieu d'utiliser des implémentations "boîte noire" (comme scipy.interpolate), j'ai réécrit ces algorithmes from scratch en utilisant PyTorch. Cela permet de :
1. Comprendre la mécanique interne des mathématiques.
2. Exploiter la puissance du calcul parallèle sur GPU (CUDA).
3. Manipuler les Tenseurs et le Broadcasting.

# Fonctionnalités
- Interpolation de Lagrange:
    - Implémentation orientée objet (LagrangeInterpolator).
    - Support natif des Tenseurs PyTorch et des listes Python standard.
    - Gestion automatique du device (CPU / GPU).
    - Code documenté et typé (Type Hints).

#  Théorie Mathématique
L'algorithme implémenté repose sur la formule du polynôme d'interpolation de Lagrange. Pour un ensemble de points $(x_j, y_j)$, le polynôme est défini par :
$$P(x) = \sum_{i=0}^{n} y_i L_i(x)$$

Avec les polynômes de base $L_i(x)$ définis par :
$$L_i(x) = \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}$$
L'implémentation conserve cette structure logique pour maximiser la lisibilité pédagogique tout en utilisant les opérations tensorielles.
