# **************************************************************************
# INF7370-Hiver 2023
# Travail pratique 1
# ===========================================================================
# ===========================================================================
# ===========================================================================

# ===========================================================================
# Le but de ce travail est de classifier les restaurants en 2 états (Fermeture définitive / Ouvert)
#
# Ce fichier consiste la deuxième étape du travail -> pré-traitement du dataset issu de la première tache.
# Dans ce fichier code vous devez  traiter l’ensemble de données préparées dans la  première étape afin de
# les rendre prêtes pour la consommation par les modèles d’apprentissage dans l'étape suivante.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres librairies dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque librairie
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# ==========================================
# ===============VARIABLES==================
# ==========================================

# l'emplacement des données sur le disque
# Note: Il faut placer le dossier "donnees"  contenant les 8 fichiers .csv dans le même endroit que les fichiers de code
data_path = "donnees/"

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres variables globales dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque variable
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# votre code ici:

# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# charger en mémoire les features préparées dans la première étape
features = pd.read_csv(data_path + "features.csv")


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Remplacez les valeurs manquantes par de propres valeurs
#
# Vous devez identifier tous les features qui manquent de valeurs ou
# qui ont des valeurs erronées dans le fichier "features.csv" préparé dans la première etape,
# puis vous devez remplacez ces valeurs manquantes ou erronées par de propres valeurs.
# Les valeurs manquantes peuvent être remplacées par des 0, ou remplacées par la moyenne ou le mode.
# La méthode choisie doit dépendre de la nature de chaque feature.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("-----------------------------------------------------")
print("Vérifier les colonnes qui ont des valeurs manquantes")
print(features.isna().sum())

#Remplacement des valeurs manquantes par 0 pour les variables indiquant un nombre d'occurence. 
#Ceci, parce que ces valeurs résultent d'un calcul d'effectif.

lst_cols = ["nb_restaurants_zone", "zone_categories_intersection" , "ville_categories_intersection", "nb_avis_favorables",\
        "nb_avis_defavorables", "nb_avis_favorables_mention", "nb_avis_defavorables_mention", "nb_avis_favorables_elites",\
        "nb_avis_defavorables_elites", "nb_conseils", "nb_conseils_compliment" , "nb_conseils_elites", "nb_checkin"]
features[lst_cols] = features[lst_cols].fillna(0,axis=1)


#Faire une description des colonnes de type float (résultant d'un calcul decimal) et choisir la méthode de remplacement
lst_cols = ["ecart_type_etoiles", "ratio_avis_favorables", "ratio_avis_defavorables", "moyenne_checkin","ecart_type_checkin"]              
print(features[lst_cols].describe())

#Nous remarquons qu'on peut remplacer les trois prémières colonnes par la moyenne vue les valeurs sont presque distribuées normalement
features["ecart_type_etoiles"] = features["ecart_type_etoiles"].fillna(features["ecart_type_etoiles"].mean())
features["ratio_avis_favorables"] = features["ratio_avis_favorables"].fillna(features["ratio_avis_favorables"].mean())
features["ratio_avis_defavorables"] = features["ratio_avis_defavorables"].fillna(features["ratio_avis_defavorables"].mean())

#Nous remarquons qu'on peut remplacer les deux dernières colonnes par la médiane vue les valeurs ne sont pas distribuées normalement
features["moyenne_checkin"] = features["moyenne_checkin"].fillna(features["moyenne_checkin"].median())
features["ecart_type_checkin"] = features["ecart_type_checkin"].fillna(features["ecart_type_checkin"].median())

#Nous allons remplacer les valeurs manquantes de la colonne zone par le mode vue ses valeurs sont catégorielles.
features["zone"] = features["zone"].fillna(features["zone"].mode()[0])

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 2
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Catégorisation des features: ville et zone
#
# Pour les deux attributs "ville" et "zone" avec des valeurs symboliques,
# il faut effectuer une transformation de ces symboles.
# Vous pouvez utiliser la fonction Categorical (de la librairie Pandas).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

features.ville = pd.Categorical(features.ville)
features.ville = features.ville.cat.codes

features.zone = pd.Categorical(features.zone)
features.zone = features.zone.cat.codes


# -----------------------------------------------------------
# Elimination de la colonne identifiante (ID): restaurant_id
print("------------------------")
print("Elimination de la colonne restaurant_id")
features = features.drop('restaurant_id', axis=1)
print("")

# -----------------------------------------------------------
# Sauvegarder l'ensemble de données pré-traitées dans un fichier csv afin d'être utilisées dans l'étape suivante
features.to_csv("donnees/features_finaux.csv", index=False)
