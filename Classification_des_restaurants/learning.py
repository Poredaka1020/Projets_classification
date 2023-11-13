# **************************************************************************
# INF7370-Hiver 2023
# Travail pratique 1
# ===========================================================================
# ===========================================================================
# ===========================================================================
# ===========================================================================

# ===========================================================================
# Le but de ce travail est de classifier les restaurants en 2 états (Fermeture définitive / Ouvert)
#
# Ce fichier consiste la troisième étape du travail -> entrainement des modèles de classification
# Dans ce fichier code, vous devez entrainer 5 modèles de classification sur les données préparées dans l'étape précédente.
# ===========================================================================

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd

# la librairie pour normalizer les données par Z-Score
from sklearn.preprocessing import StandardScaler

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres librairies dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque librairie
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# Pour afficher la courbe ROC
import matplotlib.pyplot as plt

#La librairie pour partitionner les données (entrainement et test)
from sklearn.model_selection import train_test_split

#Les librairies pour la classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier

#Import des métriques d'évaluation de performance des modèles de classification
from sklearn.metrics import confusion_matrix, classification_report, RocCurveDisplay, f1_score

#Librairie pour la sélection des top 10 meilleurs features
from sklearn.feature_selection import mutual_info_classif
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

# Votre code ici:


# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# Charger en mémoire les features préparées dans la deuxième étape (pré-traités)
features = pd.read_csv(data_path + "features_finaux.csv")

# ==========================================
# INITIALIZATION DES DONNÉES ET DES ÉTIQUETTES
# ==========================================

# Initialisation des données et des étiquettes
x = features.copy() # "x" contient l'ensemble des données d'entrainement
y = x["ferme"]      # "y" contient les étiquettes des enregistrements dans "x"

# Elimination de la colonne classe (ferme) des features
x = x.drop('ferme', axis=1)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  - Normaliser les données en utilisant Z-score (StandardScaler dans Scikit-learn)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 2
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Divisez les données en deux lots (entrainement et test)
# (indiquer dans votre rapport le pourcentage des données de test que vous avez utilisé)
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=7)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 3
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Entrainez 5 modèles de classification sur l'ensemble de données normalisées (avec tous les features)
#   1 - Arbre de decision
#   2 - Forêt d’arbres décisionnels (Random Forest)
#   3 - Classification bayésienne naïve
#   4 - Bagging
#   5 - AdaBoost
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

#   1 - Arbre de decision
dt_classifier = DecisionTreeClassifier()
dt_model = dt_classifier.fit(X_train,y_train)

dt_classifier_2 = DecisionTreeClassifier(max_depth=10)
dt_model = dt_classifier_2.fit(X_train,y_train)

#   2 - Forêt d’arbres décisionnels (Random Forest)
rf_classifier = RandomForestClassifier(max_depth=16)
rf_model = rf_classifier.fit(X_train,y_train)
    
#   3 - Classification bayésienne naïve
nb_classifier = GaussianNB()
nb_model = nb_classifier.fit(X_train,y_train)

#   4 - Bagging
bc_classifier = BaggingClassifier()
bc_model = bc_classifier.fit(X_train,y_train)

#   5 - AdaBoost
ab_classifier = AdaBoostClassifier()
ab_model = ab_classifier.fit(X_train,y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 4
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Afficher les resultats sur les données test de chaque algorithm entrainé avec tous les features
#   1- Le taux des vrais positifs (TP Rate) – de la classe Restaurants fermés définitivement.
#   2- Le taux des faux positifs (FP Rate) – de la classe Restaurants fermés définitivement.
#   3- F-measure de la classe Restaurants fermés définitivement.
#   4- La surface sous la courbe ROC (AUC).
#   5- La matrice de confusion.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
print("****************************************************")
print("****************Decision Tree**********************")
print("****************************************************")
dt_predicted = dt_model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, dt_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,dt_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(dt_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,dt_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,dt_predicted)))

print("****************************************************")
print("****************Random Forest**********************")
print("****************************************************")
rf_predicted = rf_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, rf_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,rf_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(rf_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,rf_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,rf_predicted)))

print("****************************************************")
print("****************Naives Bayes**********************")
print("****************************************************")
nb_predicted = nb_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, nb_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,nb_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(nb_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,nb_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,nb_predicted)))

print("****************************************************")
print("****************Bagging**********************")
print("****************************************************")

bc_predicted = bc_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, bc_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,bc_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(bc_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,bc_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,bc_predicted)))

print("****************************************************")
print("****************AdaBoosting**********************")
print("****************************************************")
ab_predicted = ab_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, ab_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,ab_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(ab_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,ab_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,ab_predicted)))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 5
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Selectionnez les tops 10 features
#
# Vous devez identifier les 10 meilleurs features en utilisant la mesure du Gain d’information (Mutual Info dans scikit-learn).
# Afficher les 10 meilleurs features dans un tableau (par ordre croissant selon le score obtenu par le Gain d'information).
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

features_selection = mutual_info_classif(x_scaled,y)
selected_feature = pd.DataFrame({"features":x.columns.tolist(),"rank":features_selection.tolist()}).sort_values(by='rank',ascending=False).head(10)
print("Les 10 meilleurs features sont:")
print(selected_feature)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 6
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Entrainez 5 modèles de classification sur l'ensemble de données normalisées avec seulement les top 10 features selectionnés.
#   1 - Arbre de decision
#   2 - Forêt d’arbres décisionnels (Random Forest)
#   3 - Classification bayésienne naïve
#   4 - Bagging
#   5 - AdaBoost
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

x_scaled = scaler.fit_transform(x[selected_feature['features'].tolist()])

X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)


#   1 - Arbre de decision
dt_classifier = DecisionTreeClassifier()
dt_model = dt_classifier.fit(X_train,y_train)

dt_classifier_2 = DecisionTreeClassifier(max_depth=10)
dt_model = dt_classifier_2.fit(X_train,y_train)

#   2 - Forêt d’arbres décisionnels (Random Forest)
rf_classifier = RandomForestClassifier(max_depth=16)
rf_model = rf_classifier.fit(X_train,y_train)
    
#   3 - Classification bayésienne naïve
nb_classifier = GaussianNB()
nb_model = nb_classifier.fit(X_train,y_train)

#   4 - Bagging
bc_classifier = BaggingClassifier()
bc_model = bc_classifier.fit(X_train,y_train)

#   5 - AdaBoost
ab_classifier = AdaBoostClassifier()
ab_model = ab_classifier.fit(X_train,y_train)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 7
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Afficher les resultats sur les données test de chaque algorithm entrainé avec les top 10 features
#   1- Le taux des vrais positifs (TP Rate) – de la classe Restaurants fermés définitivement.
#   2- Le taux des faux positifs (FP Rate) – de la classe Restaurants fermés définitivement.
#   3- F-measure de la classe Restaurants fermés définitivement.
#   4- La surface sous la courbe ROC (AUC).
#   5- La matrice de confusion.
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


print("****************************************************")
print("****************Decision Tree**********************")
print("****************************************************")
dt_predicted = dt_model.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, dt_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,dt_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(dt_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,dt_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,dt_predicted)))

print("****************************************************")
print("****************Random Forest**********************")
print("****************************************************")
rf_predicted = rf_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, rf_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,rf_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(rf_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,rf_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,rf_predicted)))

print("****************************************************")
print("****************Naives Bayes**********************")
print("****************************************************")
nb_predicted = nb_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, nb_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,nb_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(nb_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,nb_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,nb_predicted)))

print("****************************************************")
print("****************Bagging**********************")
print("****************************************************")

bc_predicted = bc_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, bc_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,bc_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(bc_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,bc_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,bc_predicted)))

print("****************************************************")
print("****************AdaBoosting**********************")
print("****************************************************")
ab_predicted = ab_model.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, ab_predicted).ravel()

print("1- Le taux des vrais positifs (TP Rate) est: {}".format((tp)/(tp + fn)))
print("2- Le taux des faux positifs (FP Rate) est: {}".format(1 - (tn/(tn + fp))))
print("3- F-measure de la classe Restaurants fermés définitivement est: {}".format(f1_score(y_test,ab_predicted)))
print("4- Courbe Roc")
RocCurveDisplay.from_estimator(ab_classifier, X_test, y_test)
plt.show()
print("5- Matrice de confusion: {}".format(confusion_matrix(y_test,ab_predicted)))
print("Rapport de classification: \n{}".format(classification_report(y_test,ab_predicted)))