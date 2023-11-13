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
# Ce fichier consiste la première étape du travail -> Calcul des attributs (features engineering)
# Dans ce fichier code vous devez bâtir l'ensemble de données  avec les features nécessaires pour l'entrainement des modèles de classification.
#
# Les données (8 fichiers csv):
# ------------------------------------------------
#  -utilisateurs  : Contient la liste des utilisateurs avec quelques détails.
#  -avis          : Contient les informations qui se rattachent aux avis rédigés pour chaque restaurant avec le nombre d'étoiles accordé.
#  -conseils      : Contient les informations qui se rattachent aux conseils (tips) rédigés par les utilisateurs sur les restaurants.
#  -checkin       : Contient les dates et heures de visite par restaurant.
#  -restaurants   : Contient la liste des restaurants avec quelques détails.
#  -horaires      : Contiens les heures d'ouverture de chaque restaurant pour les sept jours de la semaine.
#  -services      : Contiens la liste des services offerts par chaque restaurant.
#  -categories    : Contiens la liste des catégories des restaurants.
#
# Source des données: https://www.yelp.com/dataset
# Note: Les données étaient pré-traitées pour inclure seulement les informations nécessaires pour ce TP.
# ------------------------------------------------

# ==========================================
# ======CHARGEMENT DES LIBRAIRIES===========
# ==========================================

# la librairie principale pour la gestion des données
import pandas as pd

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# - Inclure ici toutes les autres librairies dont vous aurez besoin
# - Écrivez en commentaire le rôle de chaque librairie
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# votre code ici:

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

#Pour eviter de créer plusieurs espaces mémoires, nous utiliserons les variables globales ci-dessous pour sauvergarder nos résultats intermédiaires
temporary_df_one = pd.DataFrame()
temporary_df_two = pd.DataFrame()
temporary_df_three = pd.DataFrame()

# ==========================================
# ====CHARGEMENT DES DONNÉES EN MÉMOIRE=====
# ==========================================

# Lire les 8 tables csv et les remplir dans 8 objets de type Dataframe
# Ces 8 Dataframes doivent être utilisés pour calculer les features
utilisateurs = pd.read_csv(data_path + "utilisateurs.csv", skipinitialspace=True)
avis = pd.read_csv(data_path + "avis.csv", skipinitialspace=True)
conseils = pd.read_csv(data_path + "conseils.csv", skipinitialspace=True)
checkin = pd.read_csv(data_path + "checkin.csv", skipinitialspace=True)
restaurants = pd.read_csv(data_path + "restaurants.csv", skipinitialspace=True)
horaires = pd.read_csv(data_path + "horaires.csv", skipinitialspace=True)
services = pd.read_csv(data_path + "services.csv", skipinitialspace=True)
categories = pd.read_csv(data_path + "categories.csv", skipinitialspace=True)

# Imprimer la taille de chaque table de données
print("Taille des données:")
print("------------------")
print("utilisateurs:\t", len(utilisateurs))
print("avis:\t\t\t", len(avis))
print("conseils:\t\t", len(conseils))
print("checkin:\t\t", len(checkin))
print("restaurants:\t", len(restaurants))
print("horaires:\t\t", len(horaires))
print("services:\t\t", len(services))
print("categories:\t\t", len(categories))
print("------------------")

# ==========================================
# ==========CALCUL DES FEATURES=============
# ==========================================

# Initialisation du Dataframe "features" qui va contenir l'ensemble de données d'entrainement
# -----------------------------------------------------
# restaurant_id:  C'est l'identifiant (ID) du restaurant
# ferme: C'est la classe (Fermeture définitive : 1 , ouvert: 0)
# Les 3 premiers features sont déjà chargés: moyenne_etoiles, ville et zone
features = restaurants[['restaurant_id', 'moyenne_etoiles', 'ville', 'zone', 'ferme']].copy()

# Extraire l'année à partir des dates
# La colonne "annee" sera utilisée dans le calcul de certain features
# -----------------------------------------------------
checkin["date"] = pd.to_datetime(checkin["date"], format='%Y-%m-%d')
checkin['annee'] = checkin.date.dt.year

avis["date"] = pd.to_datetime(avis["date"], format='%Y-%m-%d')
avis['annee'] = avis.date.dt.year

conseils["date"] = pd.to_datetime(conseils["date"], format='%Y-%m-%d')
conseils['annee'] = conseils.date.dt.year

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#                      QUESTION 1
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Calculez chacun des 34 features suivants.
# Suivez la description de chaque feature afin de bien l'estimer
# Les features doivent être ajoutés au dataframe "features"
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# -----------------------------------------------------------
# 4) nb_restaurants_zone
# Le nombre de restaurants dans la zone associée au restaurant en question.
# -----------------------------------------------------------

temporary_df_one = restaurants.groupby('zone')[['restaurant_id']].count().rename(columns={"restaurant_id":"nb_restaurants_zone"})
features = pd.merge(features,temporary_df_one, how="left", on="zone")

# -----------------------------------------------------------
# 5) zone_categories_intersection
# Le nombre de restaurants dans la même zone qui partagent au moins une catégorie avec le restaurant en question.
# -----------------------------------------------------------

# D'abord, on associe la zone et la catégorie de chaque restaurant en faisant une jointure
temporary_df_one = pd.merge(categories,restaurants[['restaurant_id','zone']],how="left", on="restaurant_id")

# Determiner le nombre de catégorie se trouvant dans la même zone
temporary_df_one = temporary_df_one.groupby(['zone','categorie']).count().reset_index().rename(columns={"restaurant_id":"zone_categories_intersection"}).drop('categorie',axis=1)
temporary_df_one = temporary_df_one[temporary_df_one['zone_categories_intersection'] > 1].groupby('zone').count()

features = pd.merge(features,temporary_df_one, how="left", on="zone")

# -----------------------------------------------------------
# 6) ville_categories_intersection
# Le nombre de restaurants dans la même ville qui partagent au moins une catégorie avec le restaurant en question.
# -----------------------------------------------------------

# D'abord, on associe la ville et la catégorie de chaque restaurant en faisant une jointure
temporary_df_one = pd.merge(categories,restaurants[['restaurant_id','ville']],how="left", on="restaurant_id")

# Determiner le nombre de catégorie se trouvant dans la même ville
temporary_df_one = temporary_df_one.groupby(['ville','categorie']).count().reset_index().rename(columns={"restaurant_id":"ville_categories_intersection"}).drop('categorie',axis=1)
temporary_df_one = temporary_df_one[temporary_df_one['ville_categories_intersection'] > 1].groupby('ville').count()

features = pd.merge(features,temporary_df_one, how="left", on="ville")

# -----------------------------------------------------------
# 7) nb_restaurant_meme_annee
# Le nombre de restaurants qui sont ouverts leurs portes dans la même année que le restaurant en question.
# Ici, on considère que la première année d'un restaurant correspond à l'année de la première publication d'un avis sur ce restaurant sur Yelp.
# -----------------------------------------------------------

# On récupère le prémier avis publié pour chaque restaurant
temporary_df_one = avis[['restaurant_id','annee']].sort_values(by='annee').drop_duplicates(keep="first").drop_duplicates(subset=["restaurant_id"])

# Fonction pour returner le nombre d'avis publié à la même année que l'avis du restaurant en question 
def get_nb_restaurant_meme_annee(x):
    return len(temporary_df_one[temporary_df_one["annee"] == x])

temporary_df_one['nb_restaurant_meme_annee'] = temporary_df_one['annee'].apply(lambda x: get_nb_restaurant_meme_annee(x))

features = pd.merge(features,temporary_df_one[['restaurant_id','nb_restaurant_meme_annee']],how='left',on='restaurant_id')

# -----------------------------------------------------------
# 8) ecart_type_etoiles
# L'écart type de la moyenne des étoiles par année.
# Il faut estimer la moyenne des étoiles par années. Puis, calculer l'écart type sur ces valeurs.
# -----------------------------------------------------------

# Calcul de la moyenne des étoiles par année pour chaque restaurant
temporary_df_one = avis.groupby(['restaurant_id','annee'])[['etoiles']].mean()[['etoiles']].reset_index().rename(columns={"etoiles": "ecart_type_etoiles"})
# Calcul de l'écart type des étoiles pour chaque restaurant
temporary_df_one = temporary_df_one.groupby('restaurant_id')[['ecart_type_etoiles']].std().reset_index()

features = pd.merge(features,temporary_df_one[['restaurant_id','ecart_type_etoiles']],how='left',on='restaurant_id')


# -----------------------------------------------------------
# 9) tendance_etoiles
# La différence entre la moyenne des étoiles de la dernière année et la moyenne des étoiles de la première année d'un restaurant.
# Ici, on considère la première année d'un restaurant correspond à l'année de la première publication d'un avis sur ce restaurant sur Yelp.
# -----------------------------------------------------------
# On commence par concatener la colonne année et le restaurant_id pour chaque restaurant puis le sauvegarder dans une colonne "ref"
avis['annee'] = avis['annee'].astype('str')
avis['ref'] = avis[['restaurant_id','annee']].agg('-'.join, axis=1)

# Les étoiles de la première et dernière année de chaque restaurant
temporary_df_one = avis[['restaurant_id','annee']].sort_values(by='annee', ascending=True).drop_duplicates(keep="first").drop_duplicates(subset=["restaurant_id"]).rename(columns={"annee":"annee_f"}).agg('-'.join, axis=1)
temporary_df_two = avis[['restaurant_id','annee']].sort_values(by='annee', ascending=False).drop_duplicates(keep="first").drop_duplicates(subset=["restaurant_id"]).rename(columns={"annee":"annee_l"}).agg('-'.join, axis=1)

# La moyenne des étoiles de la première et dernière année de chaque restaurant
temporary_df_one = avis[['restaurant_id','etoiles']][(avis['ref'].isin(temporary_df_one.tolist()))].groupby('restaurant_id').mean().reset_index().rename(columns={'etoiles': "etoiles_fy"})
temporary_df_two = avis[['restaurant_id','etoiles']][(avis['ref'].isin(temporary_df_two.tolist()))].groupby('restaurant_id').mean().reset_index().rename(columns={'etoiles': 'etoiles_ly'})

# Associer les dataframes de la moyenne des étoiles de la première et dernière année de chaque restaurant
temporary_df_three = pd.merge(temporary_df_one,temporary_df_two,how='left',on='restaurant_id')

# Différence de la moyenne des étoiles
temporary_df_three['tendance_etoiles'] = temporary_df_three['etoiles_ly'] - temporary_df_three['etoiles_fy']

features = pd.merge(features,temporary_df_three[['restaurant_id','tendance_etoiles']],how='left',on='restaurant_id')


# -----------------------------------------------------------
# 10) nb_avis
# Le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
temporary_df_one = avis.groupby('restaurant_id')['avis_id'].count().reset_index(name="nb_avis")
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 11) nb_avis_favorables
# Le nombre total d'avis favorables et positifs pour ce restaurant.
# On considère un avis "favorable" si son nombre d'étoiles est >=3.
# -----------------------------------------------------------
temporary_df_one = avis[avis['etoiles'] >= 3]
temporary_df_one = temporary_df_one.groupby('restaurant_id')['avis_id'].count().reset_index(name='nb_avis_favorables')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')


# -----------------------------------------------------------
# 12) nb_avis_defavorables
# Le nombre total d'avis défavorables pour ce restaurant.
# On considère un avis comme "défavorable" si son nombre d'étoiles est  < 3.
# -----------------------------------------------------------
temporary_df_one = avis[avis['etoiles'] < 3]
temporary_df_one = temporary_df_one.groupby('restaurant_id')['avis_id'].count().reset_index(name='nb_avis_defavorables')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')


# -----------------------------------------------------------
# 13) ratio_avis_favorables
#  Le nombre d'avis favorables et positifs sur le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
features['ratio_avis_favorables'] = features['nb_avis_favorables'] / features['nb_avis']


# -----------------------------------------------------------
# 14) ratio_avis_defavorables
#  Le nombre d'avis défavorables sur le nombre total d'avis pour ce restaurant.
# -----------------------------------------------------------
features['ratio_avis_defavorables'] = features['nb_avis_defavorables'] / features['nb_avis']


# -----------------------------------------------------------
# 15) nb_avis_favorables_mention
# Le nombre total d'avis qui ont reçu au moins une mention "useful" ou "funny" ou "cool" ET le nombre d'étoiles de l'avis est >=3.
# -----------------------------------------------------------

temporary_df_one = avis[((avis['useful'] !=0) | (avis['funny'] !=0) | (avis['cool'] !=0)) & (avis['etoiles'] >= 3)]
temporary_df_one = temporary_df_one.groupby('restaurant_id')['etoiles'].count().reset_index(name='nb_avis_favorables_mention')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 16) nb_avis_defavorables_mention
# Le nombre total d'avis qui ont reçu au moins une mention "useful" ou "funny" ou "cool" ET le nombre d'étoiles de l'avis est <3.
# -----------------------------------------------------------

temporary_df_one = avis[((avis['useful'] !=0) | (avis['funny'] !=0) | (avis['cool'] !=0)) & (avis['etoiles'] < 3)]
temporary_df_one = temporary_df_one.groupby('restaurant_id')['etoiles'].count().reset_index(name='nb_avis_defavorables_mention')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 17) nb_avis_favorables_elites
# Le nombre total d'avis favorables pour un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------

# Recuperer les utilisateurs elites désignés et les avis favorables
temporary_df_one = pd.merge(avis,utilisateurs, how='left', on = 'utilisateur_id')
temporary_df_one = temporary_df_one[(temporary_df_one['elite'] == 1) & (temporary_df_one['nb_avis'] >=100) & (temporary_df_one['nb_avis_mention'] >= 100) & (temporary_df_one['etoiles'] >= 3)]

# Nombre d'avis favorables des utilisateurs désignés
temporary_df_one = temporary_df_one.groupby('restaurant_id')['elite'].count().reset_index(name='nb_avis_favorables_elites')

features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 18) nb_avis_defavorables_elites
# Le nombre total d'avis défavorables pour un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------

# Recuperer les utilisateurs elites désignés et les avis defavorables
temporary_df_one = pd.merge(avis,utilisateurs, how='left', on = 'utilisateur_id')
temporary_df_one = temporary_df_one[(temporary_df_one['elite'] == 1) & (temporary_df_one['nb_avis'] >=100) & (temporary_df_one['nb_avis_mention'] >= 100) & (temporary_df_one['etoiles'] < 3)]

# Nombre d'avis defavorables des utilisateurs désignés
temporary_df_one = temporary_df_one.groupby('restaurant_id')['elite'].count().reset_index(name='nb_avis_defavorables_elites')

features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 19) nb_conseils
#  Le nombre total de conseils (tips) associés à un restaurant.
# -----------------------------------------------------------
temporary_df_one = conseils.groupby('restaurant_id')['restaurant_id'].count().reset_index(name='nb_conseils')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 20) nb_conseils_compliment
# Le nombre total de conseils qui ont reçu au moins un compliment (voir Table Conseils).
# -----------------------------------------------------------
temporary_df_one = conseils[conseils['nb_compliments'] > 0]
temporary_df_one = temporary_df_one.groupby('restaurant_id')['nb_compliments'].count().reset_index(name='nb_conseils_compliment')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 21) nb_conseils_elites
# Le nombre total de conseils sur un restaurant qui sont rédigés par des utilisateurs élites.
# Dans ce travail, on considère un utilisateur élite si son statut est "élite" (élite = 1 dans la table Utilisateurs)
# ET il a rédigé au moins 100 avis au total ET il a au moins 100 avis avec mention.
# -----------------------------------------------------------

# Recuperer les conseils des utlisateurs elites et leur avis
temporary_df_one = pd.merge(conseils,utilisateurs, how='left', on = 'utilisateur_id')
temporary_df_one = temporary_df_one[(temporary_df_one['elite'] == 1) & (temporary_df_one['nb_avis'] >=100) & (temporary_df_one['nb_avis_mention'] >= 100)]

# Le nombre de conseil des utilisateurs elites par restaurant
temporary_df_one = temporary_df_one.groupby('restaurant_id')['elite'].count().reset_index(name='nb_conseils_elites')

features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 22) nb_checkin
# Le nombre total de visites.
# -----------------------------------------------------------

temporary_df_one = checkin.groupby('restaurant_id')['restaurant_id'].count().reset_index(name='nb_checkin')
features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 23) moyenne_checkin
# La moyenne de visites par année.
# -----------------------------------------------------------

# Le nombre de visite par année de chaque restaurant d'abord
temporary_df_one = checkin.groupby(['restaurant_id','annee']).count().reset_index().rename(columns={'date':'moyenne_checkin'}).drop('annee', axis=1)

# La moyenne du nombre de visite par année de chaque restaurant 
temporary_df_one = temporary_df_one.groupby(['restaurant_id']).mean()

features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 24) ecart_type_checkin
# L'écart type de visites par année.
# Ici, on calcul l'écart type pour le total des visites par année.
# -----------------------------------------------------------

# Le nombre de visite par année de chaque restaurant d'abord
temporary_df_one = checkin.groupby(['restaurant_id','annee']).count().reset_index().rename(columns={'date':'ecart_type_checkin'}).drop('annee', axis=1)

# La moyenne du nombre de visite par année de chaque restaurant 
temporary_df_one = temporary_df_one.groupby(['restaurant_id']).std()

features = pd.merge(features,temporary_df_one, how='left',on='restaurant_id')

# -----------------------------------------------------------
# 25) chaine
#  Prend 0 ou 1. La valeur 1 indique que le restaurant fait parti d'une chaîne (p. ex. McDonald).
#  On considère un restaurant comme il fait partie d'une chaîne, s’il existe un autre restaurant dans la base de données qui a le même nom.
# -----------------------------------------------------------

# On récupère les restaurant qui ont des noms en communs
temporary_df_one = restaurants['nom'].value_counts().reset_index(name='count')
temporary_df_one = temporary_df_one[temporary_df_one['count']>1] 

# Dire le restaurant appartient à une chaine
def set_chaine(x):
    if x in temporary_df_one['index'].tolist():
        return 1
    else: 
        return 0
    
restaurants['chaine'] = restaurants['nom'].apply(lambda x: set_chaine(x))
features = pd.merge(features,restaurants[['restaurant_id','chaine']], how='left',on='restaurant_id')


# -----------------------------------------------------------
# 26) nb_heures_ouverture_semaine
# Le nombre total d'heures d'ouverture du restaurant par semaine.
# -----------------------------------------------------------

# Calcul de la différence entre l'heure de fermeture et l'heure d'ouverture
def set_hour(x):
    try:
        heure = x.strip().split('-')
        ouv = heure[0].split(':')
        fer = heure[1].split(':')
        
        #Calul de la différence des heures
        fer_ = int(fer[0])* 60 + int(fer[1])
        ouv_ = int(ouv[0]) * 60 + int(ouv[1])
        total =  (fer_ - ouv_)/60
        
        if total < 0:
            if int(fer[0]) > 0:
                # On ajoute +12 pour prendre le format 24h
                total = total + 12
            else:
                # Nous avons remarqué quelques lignes au format 24h initialement, donc on reste dans le même format
                total = total + 24
        return total
    except AttributeError:
        return None

cols = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
for val in cols:
    col = val + "_diff"
    horaires[col] =  horaires[val].apply(lambda x: set_hour(x))

# Calcul du nombre d'heures d'ouverture par semaine pour chaque restaurant
horaires['nb_heures_ouverture_semaine'] = horaires.select_dtypes(include = ['float64']).sum(axis=1)
features = pd.merge(features,horaires[['restaurant_id','nb_heures_ouverture_semaine']], how='left',on='restaurant_id')

# -----------------------------------------------------------
# 27) ouvert_samedi
# Si le restaurant est ouvert les samedis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

horaires['ouvert_samedi'] = horaires['samedi_diff'].apply(lambda x: 1 if x >= 1 else 0 )
features = pd.merge(features,horaires[['restaurant_id','ouvert_samedi']], how='left',on='restaurant_id')


# -----------------------------------------------------------
# 28) ouvert_dimanche
# Si le restaurant est ouvert les dimanches (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

horaires['ouvert_dimanche'] = horaires['dimanche_diff'].apply(lambda x: 1 if x >= 1 else 0 )
features = pd.merge(features,horaires[['restaurant_id','ouvert_dimanche']], how='left',on='restaurant_id')


# -----------------------------------------------------------
# 29) ouvert_lundi
# Si le restaurant est ouvert les lundis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

horaires['ouvert_lundi'] = horaires['lundi_diff'].apply(lambda x: 1 if x >= 1 else 0 )
features = pd.merge(features,horaires[['restaurant_id','ouvert_lundi']], how='left',on='restaurant_id')

# -----------------------------------------------------------
# 30) ouvert_vendredi
# Si le restaurant est ouvert les vendredis (valeur booléenne : 0 ou 1).
# -----------------------------------------------------------

horaires['ouvert_vendredi'] = horaires['vendredi_diff'].apply(lambda x: 1 if x >= 1 else 0 )
features = pd.merge(features,horaires[['restaurant_id','ouvert_vendredi']], how='left',on='restaurant_id')

# -----------------------------------------------------------
# 31) emporter
features = pd.merge(features,services[['restaurant_id','emporter']], how='left',on='restaurant_id')

# 32) livraison
features = pd.merge(features,services[['restaurant_id','livraison']], how='left',on='restaurant_id')

# 33) bon_pour_groupes
features = pd.merge(features,services[['restaurant_id','bon_pour_groupes']], how='left',on='restaurant_id')

# 34) bon_pour_enfants
features = pd.merge(features,services[['restaurant_id','bon_pour_enfants']], how='left',on='restaurant_id')

# 35) reservation
features = pd.merge(features,services[['restaurant_id','reservation']], how='left',on='restaurant_id')

# 36) prix
features = pd.merge(features,services[['restaurant_id','prix']], how='left',on='restaurant_id')

# 37) terrasse
features = pd.merge(features,services[['restaurant_id','terrasse']], how='left',on='restaurant_id')


# -----------------------------------------------------------
# Sauvegarder l'ensemble de données dans un fichier csv afin d'être utilisé dans l'étape suivante
features.to_csv("donnees/features.csv", index=False)
