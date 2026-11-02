# Neural Network — réseau de neurones en Python avec NumPy

Un petit réseau de neurones **entièrement implémenté en Python avec NumPy**, sans TensorFlow, PyTorch ni Keras.

Ce projet a pour objectif de comprendre et d'expérimenter les mécanismes d'un réseau de neurones *feedforward* : propagation avant, fonction de coût, rétropropagation et mise à jour des poids et des biais. Le cœur de la librairie se trouve dans **[`ia.py`](ia.py)**. Les autres scripts à la racine servent principalement aux expérimentations ; le dossier [`examples/`](examples/) contient des exemples complémentaires.

> **État du projet :** librairie personnelle en développement, principalement destinée à l'apprentissage et à l'expérimentation. Ce n'est pas, à ce stade, un paquet publié sur PyPI.

## Fonctionnalités

- Architecture configurable : nombre de couches et de neurones défini par l'utilisateur.
- Initialisation aléatoire des poids (limites de type Xavier/Glorot) et biais initialisés à zéro.
- Activation **sigmoïde** dans les couches cachées ; **sortie linéaire** (valeurs non limitées à l'intervalle `[0, 1]`).
- Propagation avant et rétropropagation implémentées avec NumPy.
- Fonction de coût fondée sur l'erreur quadratique.
- Entraînement sur l'ensemble des données ou sur des mini-lots séquentiels.
- Ajustement temporaire du pas d'apprentissage par *backtracking line search* (condition d'Armijo) lors des mises à jour.
- Sauvegarde et chargement des réseaux au format JSON.

## Installation

**Prérequis :** Python **3.10 ou plus récent** et NumPy.

Clonez le projet, puis installez NumPy :

```bash
git clone https://github.com/victordbss/Neural-Network-.git
cd Neural-Network-
python -m pip install numpy
```

Vous pouvez maintenant importer `Network` depuis `ia.py` dans un script Python placé à la racine du dépôt :

```python
from ia import Network
```

Pour utiliser le moteur dans un autre projet sans installer ce dépôt, vous pouvez également y copier le fichier `ia.py` et installer NumPy.

## Démarrage rapide

Exemple : apprendre la relation simple **`y = x1 + x2`**.

```python
import numpy as np
from ia import Network

# Chaque ligne est un exemple ; chaque colonne est une entrée.
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])

# Une sortie par exemple : forme (4, 1), et non (4,).
Y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [2.0],
])

# 2 neurones d'entrée -> 1 neurone de sortie linéaire.
network = Network(layers_sizes=[2, 1], learning_rate=0.1)

# Entraîner le réseau.
network.training(
    inputs=X,
    outputs=Y,
    epoch=1000,
    notify_rate=100,
    learning_rate_ratio=1.0,
)

# Prédire sur de nouvelles données (toujours en 2D).
X_test = np.array([[0.2, 0.7], [0.8, 0.3]])
predictions = network.forward(X_test)
print(predictions)  # Valeurs attendues approximativement : 0.9 et 1.1

# Évaluer, sauvegarder et recharger le réseau.
print("Coût :", network.cost_function(X, Y))
network.save("modele_addition.json")

restored = Network(layers_sizes=[2, 1])
restored.load("modele_addition.json")
print(restored.forward(X_test))
```

Les valeurs initiales des poids étant aléatoires, les résultats numériques précis peuvent varier d'une exécution à l'autre.

## Définir l'architecture

La classe **`Network`** est le point d'entrée de la librairie :

```python
network = Network(layers_sizes=[2, 5, 5, 1], learning_rate=0.1)
```

Chaque entier de `layers_sizes` représente le nombre de neurones d'une couche, **de l'entrée vers la sortie** :

```text
2 entrées  →  5 neurones cachés  →  5 neurones cachés  →  1 sortie
```

- `layers_sizes[0]` : nombre de caractéristiques fournies à chaque exemple.
- Valeurs intermédiaires : tailles des couches cachées (on peut en mettre zéro, une ou plusieurs).
- `layers_sizes[-1]` : nombre de valeurs prédites par exemple.
- `learning_rate` : pas initial utilisé pour la descente de gradient (valeur par défaut : `0.1`).

Les couches d'entrée et les couches intermédiaires sont assemblées automatiquement ; il n'est normalement pas nécessaire d'instancier directement `InputLayer` ou `Layer`.

## Format des données

L'API fonctionne avec des tableaux NumPy **à deux dimensions** :

| Donnée | Forme attendue |
| --- | --- |
| `inputs` | `(nombre_exemples, nombre_entrées)` |
| `outputs` | `(nombre_exemples, nombre_sorties)` |
| résultat de `forward(inputs)` | `(nombre_exemples, nombre_sorties)` |

Par exemple, pour **3 exemples, 2 entrées et 1 sortie** :

```python
X = np.array([[1., 2.], [3., 4.], [5., 6.]])  # (3, 2)
Y = np.array([[3.], [7.], [11.]])              # (3, 1)
```

**Attention :** pour une seule sortie, `np.array([3., 7., 11.])` a la forme `(3,)` et peut déclencher un *broadcasting* involontaire dans le calcul du coût. Utilisez `[[3.], [7.], [11.]]`. Pour une seule prédiction, appelez `network.forward(np.array([[1., 2.]]))`, et non `forward(np.array([1., 2.]))`.

## Entraîner un réseau

La méthode d'entraînement expose les paramètres suivants :

```python
network.training(
    inputs=X,
    outputs=Y,
    epoch=1000,
    notify_rate=100,
    learning_rate_ratio=1.001,
    package_size=None,
)
```

| Paramètre | Description | Valeur par défaut |
| --- | --- | --- |
| `inputs` | Exemples d'entraînement (matrice 2D) | Obligatoire |
| `outputs` | Résultats attendus (matrice 2D) | Obligatoire |
| `epoch` | Nombre d'**itérations de mise à jour** | `100` |
| `notify_rate` | Intervalle d'affichage du coût, en itérations | `100` |
| `learning_rate_ratio` | Diviseur du pas d'apprentissage lors d'un affichage | `1.001` |
| `package_size` | Nombre d'exemples traités par itération | `None` (toutes les données) |

### Entraînement avec mini-lots

```python
network.training(X, Y, epoch=2000, notify_rate=100, package_size=32)
```

Si `package_size` est défini, l'implémentation actuelle traite **un seul lot par itération**, puis passe au suivant, dans l'ordre, avant de recommencer. Les données ne sont **pas mélangées** automatiquement. Si leur nombre n'est pas divisible par `package_size`, le dernier lot incomplet n'est pas parcouru par la logique actuelle. Une `epoch` ne signifie donc pas nécessairement un passage complet sur toutes les données.

Pour un premier essai, laissez `package_size=None` : chaque itération utilise alors l'ensemble des exemples.

### Taux d'apprentissage

`learning_rate` définit le pas de départ. Pendant chaque mise à jour, le programme peut tester des pas plus petits si le nouveau coût ne satisfait pas la condition de diminution utilisée. De plus, `training()` divise **`network.learning_rate`** par `learning_rate_ratio` **aux itérations où le coût est affiché** (`i % notify_rate == 0`, y compris `i = 0`).

Pour garder le taux de base constant entre les notifications :

```python
network.training(X, Y, epoch=1000, notify_rate=100, learning_rate_ratio=1.0)
```

## Faire des prédictions et mesurer l'erreur

```python
predictions = network.forward(X_test)     # Tableau NumPy 2D
cost = network.cost_function(X, Y)        # Nombre flottant
network.shape()                           # Affiche la structure, les poids et les biais
```

La méthode `shape()` **affiche** des informations dans la console ; elle ne renvoie pas un objet décrivant l'architecture.

La fonction de coût est :

$$
J = \frac{1}{2m}\sum_{i=1}^{m}\left\|\hat{y}^{(i)}-y^{(i)}\right\|_2^2
$$

avec `m` le nombre d'exemples. Il s'agit de la moitié de la somme des erreurs quadratiques moyennée sur les exemples ; avec plusieurs sorties, le code ne divise **pas** par le nombre de sorties.

## Sauvegarder et charger un modèle

```python
network.save("mon_modele.json")
```

Cette commande crée automatiquement le dossier `data/`, si nécessaire, puis y écrit **`data/mon_modele.json`**. Le nom donné n'est pas complété automatiquement par `.json` : indiquez l'extension si vous la souhaitez.

Pour recharger les poids et les biais enregistrés :

```python
loaded_network = Network(layers_sizes=[1, 1])
loaded_network.load("mon_modele.json")
```

`load()` remplace l'architecture et les paramètres du réseau existant par ceux du fichier JSON. Le `learning_rate` n'est **pas sauvegardé** dans le format actuel : spécifiez-le lors de la construction si vous voulez reprendre l'entraînement avec un pas particulier.

Le chemin `data/` est interprété **par rapport au répertoire de travail courant**, pas nécessairement par rapport à l'emplacement de `ia.py`.

## Fonctionnement interne

Chaque couche dense calcule :

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

Pour les **couches cachées**, une sigmoïde transforme ensuite le résultat :

$$
a^{(l)} = \sigma(z^{(l)}) = \frac{1}{1+e^{-z^{(l)}}}
$$

La **dernière couche n'applique pas de fonction d'activation** : ses sorties peuvent être négatives ou supérieures à 1, ce qui est notamment utile pour des exercices de régression.

Pendant l'entraînement, la rétropropagation applique la règle de dérivation en chaîne pour obtenir les gradients par rapport aux poids et aux biais. Une descente de gradient les met à jour, avec une recherche de pas décroissants lorsque nécessaire. Les calculs matriciels s'appuient sur NumPy ; la propagation avant parcourt actuellement les exemples un par un.

## Référence de l'API (`ia.py`)

| Élément | Rôle |
| --- | --- |
| `Network(layers_sizes, learning_rate=0.1)` | Crée et initialise un réseau |
| `network.forward(inputs_list)` | Calcule les prédictions |
| `network.cost_function(inputs, outputs)` | Calcule le coût quadratique |
| `network.training(inputs, outputs, epoch=100, notify_rate=100, learning_rate_ratio=1.001, package_size=None)` | Entraîne le réseau |
| `network.shape()` | Affiche l'architecture et les paramètres |
| `network.save(file_name="new_data")` | Écrit l'état du réseau dans `data/` |
| `network.load(filename="new_data")` | Recharge un état depuis `data/` |

Les méthodes `propagation()`, `back_propagation()`, `init_layers()` et celles des classes `Layer` / `InputLayer` servent surtout au fonctionnement interne. Pour un usage normal, utilisez la classe `Network`.

## Organisation du dépôt

```text
Neural-Network-/
├── ia.py          # Librairie : couches, réseau, entraînement, sauvegarde
├── examples/      # Exemples d'utilisation et d'expérimentation
├── main.py        # Script d'expérimentation (temporaire)
├── data.py        # Données d'expérimentation (temporaires)
└── README.md      # Documentation
```

Le répertoire `data/` peut être créé automatiquement lors d'une sauvegarde ; il contient alors les fichiers JSON générés.

## Limites actuelles et pistes d'évolution

Cette librairie est volontairement simple et encore en développement :

- L'activation des couches cachées est fixée à la sigmoïde ; la couche de sortie est linéaire.
- La fonction de coût disponible est l'erreur quadratique ; il n'y a pas encore d'API dédiée aux métriques de classification.
- L'entraînement ne mélange pas les mini-lots et ne gère pas les lots incomplets.
- La propagation avant n'est pas encore entièrement vectorisée sur les exemples.
- Le taux d'apprentissage sauvegardé en mémoire n'est pas inclus dans les fichiers JSON.
- Le code ne valide pas systématiquement toutes les formes de tableaux reçues.
- Le projet n'est pas encore distribué comme paquet Python installable avec `pip`.

Les prochaines étapes possibles sont : la validation des données, le mélange des mini-lots, d'autres activations et fonctions de coût, des tests automatisés et un véritable empaquetage Python.

## Auteur

Projet développé par **[@victordbss](https://github.com/victordbss)**.

Code source : [github.com/victordbss/Neural-Network-](https://github.com/victordbss/Neural-Network-)
