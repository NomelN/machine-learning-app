# importation des modules
import tensorflow as tf
import numpy as np

"""
description du modele
"""

# création d'un réseau multicouche
MonReseau = tf.keras.Sequential()

# C1: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    input_shape=(28, 28, 1),
    padding='same'
))

# C2: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=32,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    padding='same'
))

# P1: Description de la couche de pooling (Max)
MonReseau.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'))

# C3: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    padding='same'))

# C4: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    padding='same'))

# P2: Description de la couche de pooling (Max)
MonReseau.add(tf.keras.layers.MaxPool2D(
    pool_size=(2, 2),
    strides=(2, 2),
    padding='valid'))

# C5: description de la couche de convolution
MonReseau.add(tf.keras.layers.Conv2D(
    filters=128,
    kernel_size=(3, 3),
    strides=(1, 1),
    activation='relu',
    padding='same'))

# P3: Description de la couche de pooling (Max)
MonReseau.add(tf.keras.layers.MaxPool2D(
    pool_size=(3, 3),
    strides=(2, 2),
    padding='valid'))

MonReseau.add(tf.keras.layers.Flatten())

# Création d'une couche de 128 neurones avec fonction d'activation Relu
MonReseau.add(tf.keras.layers.Dense(128, activation='relu'))

# FC6: connexion totale avec couche de 200 neurones avec fct d'activation Relu
MonReseau.add(tf.keras.layers.Dense(200, activation='relu'))

# Sortie: 10 neurones avec fct d'activation Softmax
MonReseau.add(tf.keras.layers.Dense(10, activation='softmax'))

# Affichage du descriptif du réseau
print(MonReseau.summary())


# ----------------------------------------------------------------------------
# Chargement des données d'apprentissage et de tests
# ----------------------------------------------------------------------------
# Chargement en mémoire de la base de données des caractères MNIST
#  => tableaux de type ndarray (Numpy) avec des valeur entières
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# ----------------------------------------------------------------------------
# Changements de format pour exploitation
# ----------------------------------------------------------------------------
# les valeurs associées aux pixels sont des entiers entre 0 et 255
#  => transformation en valeurs réelles entre 0.0 et 1.0
x_train, x_test = x_train / 255.0, x_test / 255.0
# Les données en entrée sont des matrices de pixels 28x28
#  => transformation en matrices 28x28 sur 1 plan en profondeur
#     (format en 4D nécessaire pour pouvoir réaliser des convolutions (conv2D))
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)
# Les données de sortie sont des entiers associés aux chiffres à identifier
#  => transformation en vecteurs booléens pour une classification en 10 valeurs
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
# ----------------------------------------------------------------------------
# COMPILATION du réseau
#  => configuration de la procédure pour l'apprentissage
# ----------------------------------------------------------------------------
MonReseau.compile(optimizer='adam',                # algo d'apprentissage
                  loss='categorical_crossentropy',  # mesure de l'erreur
                  metrics=['accuracy'])            # mesure du taux de succès
# ----------------------------------------------------------------------------
# APPRENTISSAGE du réseau
#  => calcul des paramètres du réseau à partir des exemples
# ----------------------------------------------------------------------------
hist = MonReseau.fit(
    x=x_train,  # données d'entrée pour l'apprentissage
    y=y_train,  # sorties désirées associées aux données d'entrée
    epochs=10,  # nombre de cycles d'apprentissage
    batch_size=128,  # taille des lots pour l'apprentissage
    validation_data=(x_test, y_test)  # données de test
)

# Sauvegarde du modèle
MonReseau.save('convolution_model.h5')

# ----------------------------------------------------------------------------
# EVALUATION de la capacité à généraliser du réseau
#  => test du réseau sur des exemples non utilisés pour l'apprentissage
# ----------------------------------------------------------------------------
print()
perf = MonReseau.evaluate(
    x=x_test,  # données d'entrée pour le test
    y=y_test  # sorties désirées pour le test
)
print("Taux d'exactitude sur le jeu de test: {:.2f}%".format(perf[1]*100))
