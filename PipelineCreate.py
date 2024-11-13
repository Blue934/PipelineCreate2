from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from pandas.plotting import scatter_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os


def reset () :
    print("Avez-vous fini ?")
    reponse1 = input("o/n\n")
    terminer = False
    if reponse1.lower() == "o":
        terminer = True
        return terminer
    else:
        print("Que voulez-vous faire d'autre ?")
        reponse = input()
        return reponse

def reset2 () :
    print("Avez-vous fini ?")
    reponse1 = input("o/n\n")
    terminer2 = False
    if reponse1.lower() == "o":
        terminer2 = True
        return terminer2
    
    else:
        print("Que voulez-vous faire d'autre ?")
        reponseA = input()
        return reponseA

def pipeline_no_num(Data):
    print("Vous etes entré dans l'interface de creation de pipeline2!")
    cat_col = Data.select_dtypes(exclude=[np.number])
    print(cat_col.shape)
    #Variables qualitatives
    print("Que voulez-vous faire de vos variables qualitatives ?")
    reponseA = input("Supprimer ou Remplacer:\n")
    terminer2 = False
    while terminer2 != True:
        if reponseA.lower() == "supprimer":
            print("Voulez-vous supprimer les valeurs nulles (nan) ou une colonne entiere ?")
            reponseB = input("Colonne ou Valeur\n")
            if reponseB.lower() == 'colonne':
                print(cat_col.columns)
                print("Quelle colonne voulez-vous supprimer ?")
                col = input()
                cat_col.drop(col, axis=1, inplace = True)
                print(f"La colonne {col} a été supprimée !")
                reset2()
            elif reponseB.lower() == 'valeur':
                print(cat_col.columns)
                print("Sur quelle colonne souhaitez-vous supprimer ces variables ?")
                col = input()
                cat_col.dropna(subset=[col], axis=1, inplace = True)
                print(f"Les variables manquantes de la colonne {col} ont été supprimés !")
                reset2()
            else:
                print("Je n'est pas compris !")
                reset2()
        elif reponseA.lower() == 'remplacer':
            print("Par quelle moyen souhaitez-vous remplacer les données qualitatives manquantes ?")
            reponseC = input("knn, simple, iterative:\n")
            if reponseC.lower() == 'knn':
                print("Selon les k plus proches voisin de combien ?")
                nb_voisin = int(input())
                
                terminer2 = True
            elif reponseC.lower() == 'simple':
                print("Quelle est la valeur du parametre strategy ?")#
                reponseD = input("mean, median, constant, most_frequent:\n")
                if reponseD.lower() == 'constant':
                    print("Quelle est la valeur constante que vous souhaitez utiliser ?")
                    valeur2 = input()
                    terminer2 = True
                else:
                    terminer2 = True
            elif reponseC.lower() == 'iterative':
                print("Quelle est la valeur du parametre max_iter ?")#
                nb_iter = int(input())
                print("Quelle est la valeur du parametre strategy ?")#
                reponseD = input("mean, median, constant, most_frequent:\n")
                terminer2 = True
            else:
                print("Je n'est pas compris !")
                reset2()
    print("Voulez-vous attriber la valeur ignore au parametre handle_unknown du OneHotEncoder ?")
    reponseE = input("oui/non\n")
    print("Toutes les caracteristique du cat_pipeline sont prete") 
    #if reponseC = simple
    if reponseC.lower() == 'simple' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'constant' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value= valeur2), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'constant' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value=valeur2), OneHotEncoder())
    #if reponseC = iterative
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), OneHotEncoder())
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'constant' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant', fill_value= valeur2), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'constant' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant', fill_value=valeur2), OneHotEncoder())
    #if reponse 3 = knn
    elif reponseC.lower() == 'knn' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'knn' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin), OneHotEncoder())

    if cat_pipeline is not None:
        cat_attributes = cat_col.columns
        print("categorie; \n", cat_attributes)
        print("Dimension de cat_attributes", cat_attributes.shape)
        preprocessing = ColumnTransformer([("cat", cat_pipeline, cat_attributes)], remainder='passthrough')
        Data_prepared = preprocessing.fit_transform(Data)
        print("Dimensions après imputation :", Data_prepared.shape)
        print("Voici les elements du cat_pipeline\n", cat_pipeline.steps)
        print("Vos données sont prêtes !")
        #print("Get_feature...:", preprocessing.get_feature_names_out())
        Data_prepared_dense = Data_prepared.toarray()
        # Afficher les colonnes avant et après 
        print("Index de Data :", Data.index)
        print("Colonnes de Data originale :", Data.columns) 
        print("Colonnes de Data préparée :", preprocessing.get_feature_names_out())
        Data_prepared_df = pd.DataFrame(Data_prepared_dense, columns=preprocessing.get_feature_names_out(), index=Data.index)
        print(Data_prepared_df.columns)
        print(Data.columns)
        print(Data_prepared_df.head())
        #print("Voici la taille de Data_prepared_df;\n", len(Data_prepared_df))
        return Data_prepared_df, preprocessing
    else:
        print("Le pipeline n'a pas été créé correctement.")
        
def Pipeline_creator (Data):
    print("Vous etes entré dans l'interface de creation de pipeline !")
    num_col = Data.select_dtypes(include=[np.number])
    cat_col = Data.select_dtypes(exclude=[np.number])
    print("Que voulez- vous faire de vos variables numeriques ?")
    reponse = input("Les Supprimer ou Remplacer:\n")
    terminer = False
    while terminer != True:
        if reponse.lower() == "supprimer":
            print("Voulez-vous supprimer les valeurs nulles (nan) ou une colonne entiere ?")
            reponse2 = input("Colonne ou Valeur :\n")
            if reponse2.lower() == 'colonne':
                print(num_col.columns)
                print("Quelle colonne voulez-vous supprimer ?")
                col = input()
                num_col.drop(col, axis=1, inplace = True)
                print(f"La colonne {col} a été supprimée !")
                reset()
            elif reponse2.lower() == 'valeur':
                print(num_col.columns)
                print("Sur quelle colonne souhaitez-vous supprimer ces variables ?")
                col = input()
                num_col.dropna(subset=[col], axis=1, inplace = True)
                print(f"Les variables manquantes de la colonne {col} ont été supprimés !")
                reset()
            else:
                print("Je n'est pas compris !")
                reset()
        elif reponse.lower() == 'remplacer':
            print("Par quelle moyen souhaitez-vous remplacer les données numeriques manquantes ?")
            reponse3 = input("knn, simple, iterative:\n")
            
            if reponse3.lower() == 'knn':
                print("Selon les k plus proches voisin de combien ?")
                nb_voisin = int(input())
                terminer = True
                
        
            elif reponse3.lower() == 'simple':
                print("Quelle est la valeur du parametre strategy ?")#
                reponse4 = input("mean, median, constant, most_frequent\n")
                
                if reponse4.lower() == 'constant':
                    print("Quelle est la valeur constante que vous souhaitez utliser ?")
                    valeur = int(input())
                terminer = True
                    
            elif reponse3.lower() == 'iterative':
                print("Quelle est la valeur du parametre max_iter ?")#
                nb_iter = int(input())
                print("Quelle est la valeur du parametre strategy ?")#
                reponse4 = input("mean, median, constant, most_frequent:\n")
                terminer = True
            else:
                print("Je n'est pas compris !")
                reset()
                                                            #Fin de la 1er boucle#
    
    ax = num_col.hist(bins=50, figsize=(10, 6))
    for axes in ax.flatten():
        axes.set_title(axes.get_title(), fontsize=5)  # Ajustez la taille de police ici
        axes.set_xlabel(axes.get_xlabel(), fontsize=5)
        axes.set_ylabel(axes.get_ylabel(), fontsize=5)
        plt.savefig('fig/ma_figure7.png')
    plt.tight_layout()
    plt.show()
    print("Souhaitez-vous transformer certaines variables à longue traine ?")
    reponse5 = input("oui/non\n")
    if reponse5.lower() == 'oui':
        print("Combien de variable voulez-vous transformer ?")
        nb_var = int(input())
        for i in range(nb_var):
            print(num_col.columns)
            print(F"Quelle est le nom de la {i+1}e variable ?")
            var = input()
            print(f"Quelle transformation souhaitez-vous effectuer sur la variable {var} ?")
            reponse6 = input("log, racine carree:\n")
            if reponse6.lower() == 'log':
                Log_tr = FunctionTransformer(np.log, inverse_func=np.exp, validate=True)
                # Filtrer les valeurs <= 0
                num_col[var] = num_col[var].replace(0, 1e-100)
                num_col[var] = num_col[var].dropna()
                #creer l'histogramme de la variable avant transformation
                plt.figure(1)
                plt.hist(num_col[var], bins=50, alpha=0.5, label=f'{var} avant transformation (log)')
                plt.savefig(f'fig/{var}_avtr.png')
                Log_var = Log_tr.transform(num_col[[var]])
                Log_var.flatten()
                plt.figure(2)
                plt.hist(Log_var, bins=50, alpha=0.5, color='green', label=f'{var} après transformation (log)')
                plt.savefig(f'fig/{var}_aptr.png')

                #visualisation avant apres
                plt.title(f"Histogramme de {var} avant et après transformation (log)")
                plt.show()
                print(f"La variable {var} à été transformé !")
                num_col.drop(var, axis=1, inplace=True)
                num_col[f"{var}"] = pd.DataFrame(Log_var, columns= [var], index= num_col.index)
                print(num_col.columns)
            elif reponse6.lower() == 'racine carree':
                # Transformation racine carrée
                sqrt_tr = FunctionTransformer(np.sqrt, inverse_func=np.square, validate=True)
                plt.figure(1)
                plt.hist(num_col[var], bins=50, alpha=0.5, label=f'{var} avant transformation (sqrt)')
                plt.savefig(f'fig/ma_figure_avtr{i+1}.png')
                sqrt_var = sqrt_tr.transform(num_col[var])
                plt.figure(2)
                plt.hist(sqrt_var, bins=50, alpha=0.5, label=f'{var} après transformation (sqrt)')
                plt.savefig(f'fig/ma_figure_avtr{i+1}.png')
                plt.show() 
                print(f"La variable {var} à été transformé")
                num_col.drop(var, axis=1, inplace=True)
            #elif autres transformation possible
            else:
                print("Je n'est pas compris")
                reponse6 = input()
        print("Toutes les variables transformé ont été ajouté au DataFrame !")
        print(num_col.columns)
    else:
        print("Vous avez fait le choix de ne transfromer aucune variable !")
    print("Par quelle moyen souhaitez-vous recaliber vos données ?")
    reponse7 = input("MinMax / StandardScaler ?\n")
    if reponse7.lower() == 'minmax':
        print("Quelle est la valeur de l'indince de depart ?")
        indx = float(input())
        print("Quelle est la valeur de l'indice de fin ?")
        indx2 = float(input())
    #Toutes les caracteristique sont prete pour le num_pipeline
    
    #if reponse3 = simple
    if reponse3.lower() == 'simple' and reponse4.lower() == 'mean' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'median' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(SimpleImputer(strategy='median'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'most_frequent' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'constant' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value=valeur), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'mean' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'median' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(SimpleImputer(strategy='median'), StandardScaler())
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'most_frequent' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), StandardScaler())
    elif reponse3.lower() == 'simple' and reponse4.lower() == 'constant' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(SimpleImputer(strategy='constant',fill_value=valeur ), StandardScaler())
    #if reponse3 = iterative
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'mean' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='mean'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'median' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='median'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'most_frequent' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), MinMaxScaler(feature_range=(indx, indx2)))
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'constant' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant',fill_value=valeur ), MinMaxScaler(feature_range=(indx, indx2)))#
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'mean' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='mean'), StandardScaler())
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'median' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='median'), StandardScaler())
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'most_frequent' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), StandardScaler())
    elif reponse3.lower() == 'iterative' and reponse4.lower() == 'constant' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant', fill_value=valeur ), StandardScaler())
    #if reponse 3 = knn
    elif reponse3.lower() == 'knn' and reponse7.lower() == 'minmax':
        num_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin)), MinMaxScaler(feature_range=(indx, indx2))
    elif reponse3.lower() == 'knn' and reponse7.lower() == 'standardscaler':
        num_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin), StandardScaler())
    
    
    #Variables qualitatives
    print("Que voulez-vous faire de vos variables qualitatives ?")
    reponseA = input("Supprimer ou Remplacer:\n")
    terminer2 = False
    while terminer2 != True:
        if reponseA.lower() == "supprimer":
            print("Voulez-vous supprimer les valeurs nulles (nan) ou une colonne entiere ?")
            reponseB = input("Colonne ou Valeur\n")
            if reponseB.lower() == 'colonne':
                print(cat_col.columns)
                print("Quelle colonne voulez-vous supprimer ?")
                col = input()
                cat_col.drop(col, axis=1, inplace = True)
                print(f"La colonne {col} a été supprimée !")
                reset2()
            elif reponse2.lower() == 'valeur':
                print(cat_col.columns)
                print("Sur quelle colonne souhaitez-vous supprimer ces variables ?")
                col = input()
                cat_col.dropna(subset=[col], axis=1, inplace = True)
                print(f"Les variables manquantes de la colonne {col} ont été supprimés !")
                reset2()
            else:
                print("Je n'est pas compris !")
                reset2()
        elif reponseA.lower() == 'remplacer':
            print("Par quelle moyen souhaitez-vous remplacer les données qualitatives manquantes ?")
            reponseC = input("knn, simple, iterative:\n")
            if reponseC.lower() == 'knn':
                print("Selon les k plus proches voisin de combien ?")
                nb_voisin = int(input())
                
                terminer2 = True
            elif reponseC.lower() == 'simple':
                print("Quelle est la valeur du parametre strategy ?")#
                reponseD = input("mean, median, constant, most_frequent:\n")
                if reponseD.lower() == 'constant':
                    print("Quelle est la valeur constante que vous souhaitez utiliser ?")
                    valeur2 = input()
                    terminer2 = True
                else:
                    terminer2 = True
            elif reponseC.lower() == 'iterative':
                print("Quelle est la valeur du parametre max_iter ?")#
                nb_iter = int(input())
                print("Quelle est la valeur du parametre strategy ?")#
                reponseD = input("mean, median, constant, most_frequent:\n")
                terminer2 = True
            else:
                print("Je n'est pas compris !")
                reset2()
    print("Voulez-vous attriber la valeur ignore au parametre handle_unknown du OneHotEncoder ?")
    reponseE = input("oui/non\n")
    #Toutes les caracteristique du cat_pipeline sont prete 
    #if reponseC = simple
    if reponseC.lower() == 'simple' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'), OneHotEncoder())
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'constant' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value= valeur2), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'simple' and reponseD.lower() == 'constant' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(SimpleImputer(strategy='constant', fill_value=valeur2), OneHotEncoder())
    #if reponseC = iterative
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'most_frequent' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='most_frequent'), OneHotEncoder())
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'constant' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant', fill_value= valeur2), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'iterative' and reponseD.lower() == 'constant' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(IterativeImputer(max_iter=nb_iter, initial_strategy='constant', fill_value=valeur2), OneHotEncoder())
    #if reponse 3 = knn
    elif reponseC.lower() == 'knn' and reponseE.lower() == 'oui':
        cat_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin), OneHotEncoder(handle_unknown='ignore'))
    elif reponseC.lower() == 'knn' and reponseE.lower() == 'non':
        cat_pipeline = make_pipeline(KNNImputer(n_neighbors=nb_voisin), OneHotEncoder())
    
    if cat_pipeline is not None:
        num_attributs = num_col.columns
        cat_attributes = cat_col.columns
        print("categorie; \n", cat_attributes)
        print("numerique;\n", num_attributs)
        preprocessing = ColumnTransformer([
            ("num", num_pipeline, num_attributs),
            ("cat", cat_pipeline, cat_attributes)
        ], remainder='drop')
        Data_prepared = preprocessing.fit_transform(Data)
        print(Data_prepared.shape)
        print(Data.shape)
        print("Voici les elements de num_pipeline\n",num_pipeline.steps)
        print("Voici les elements du cat_pipeline\n", cat_pipeline.steps)
        print("Vos données sont prêtes !")
        Data_prepared_df = pd.DataFrame(Data_prepared, columns=preprocessing.get_feature_names_out(), index=Data.index)
        print(Data_prepared_df.head())
        print("Voici la taille de Data_prepared_df;\n", len(Data_prepared_df))
        
        print ("Une visualisation graphique des données a ce stade est deconseillé car il pourrait introduire un biais*")
        biais1 = input("visualisation or not ?\n")
        if biais1.lower() == "visualisation":
            global biais_T 
            biais1 = 1
            showme(Data_prepared_df)
            print("Vous avez mal agi")
            return Data_prepared_df, preprocessing
        else:
            print("Vous avez bien fait")
            return Data_prepared_df, preprocessing
    else:
        print("Le pipeline n'a pas été créé correctement.")

def showme (Data):
    print("Voici les 5 premières lignes du dataframe:\n", Data.head())
    print("Voici les 5 dernières lignes:\n", Data.tail())
    print("Premières informations ; nb de lignes , type, compte des valeurs non-null")
    print(Data.info())
    print("Voulez-vous explorer une ou plusieurs colonne en particulier?")
    reponse = input("oui ou non :value_counts()\n")
    if reponse.lower() == 'oui':
        print("Combien de colonnes voulez-vous observer?")
        nb = int(input("Entrer un nombre\n"))
        if nb > 0:
            for i in range(nb):
                list_col = []
                df_col = Data.columns
                print (df_col)
                print(f"Enter le nom de la {1+i}e colonne")
                nom_col = input()
                if nom_col in df_col:
                    colone = Data[nom_col] 
                    compte = colone.value_counts()
                    print(compte)
                    list_col.append(nom_col)
                else:
                    print(f"La colonne '{nom_col}' n'existe pas dans le DataFrame.")
                    
                #2e condition de la boucle
                if i == nb - 1:
                    print("Vous etes arriver au bout du nombre souhaité")
                    print("Si une ou plusieurs colonnes comportent des variables non-numeriques, vous devriez trouver un moyen des convertirs grace à ces informations")
                    
                    print("Voulez vous explorer davantage ces données?")
                    reponse = input("oui ou non: .describe()\n")
                    if reponse.lower() == "oui":
                        print("RECAPITULATIF DES VARIABLES")
                        print(Data.describe())
                        ax = Data.hist(bins= 50, figsize=(12, 8 ))
                        for axes in ax.flatten():
                            axes.set_title(axes.get_title(), fontsize=6)  # Ajustez la taille de police ici
                            axes.set_xlabel(axes.get_xlabel(), fontsize=6)
                            axes.set_ylabel(axes.get_ylabel(), fontsize=6)
                        # Vérifier et créer le répertoire 'fig' s'il n'existe pas 
                        if not os.path.exists('fig'): 
                            os.makedirs('fig')
                        plt.savefig('fig/ma_figure1.png')
                        plt.tight_layout()
                        plt.show()
                        print("Je vous sugère de passer au JeSetCreate()")
                        break
                    else:
                        print("Dans se cas je vous sugère de passer au JeSetCreate()")
                        break
            
        else:
            print("Voulez vous explorer davantage ces données?")
            reponse = input("oui ou non: .describe() + data.hist() \n")
            if reponse.lower() == "oui":
                print("RECAPITULATIF DES VARIABLES")
                print(Data.describe())
                ax = Data.hist(bins= 100, figsize=(12, 8 ))
                for axes in ax.flatten():
                    axes.set_title(axes.get_title(), fontsize=7.5)  # Ajustez la taille de police ici
                    axes.set_xlabel(axes.get_xlabel(), fontsize=9)
                    axes.set_ylabel(axes.get_ylabel(), fontsize=9)
                if not os.path.exists('fig'): 
                    os.makedirs('fig')
                plt.savefig('fig/ma_figure2.png')
                plt.tight_layout()
                plt.show()
                print("Il et temps de passer au JeSetCreate")
            else:
                print("Dans se cas je vous sugère de passer au JeSetCreate()")
    else:
        print("Vous etes arriver au bout du nombre souhaité")
        print('''Si une ou plusieurs colonnes comportent des variables non-numeriques, vous devriez trouver un moyen des convertirs grace à ces informations''')
        print("Voulez vous explorer davantage ces données?")
        reponse = input("oui ou non: \n")
        if reponse.lower() == "oui":
            print("RECAPITULATIF DES VARIABLES")
            print(Data.describe())
            ax = Data.hist(bins= 100, figsize=(12, 8 ))
            for axes in ax.flatten():
                axes.set_title(axes.get_title(), fontsize=7.5)  # Ajustez la taille de police ici
                axes.set_xlabel(axes.get_xlabel(), fontsize=9)
                axes.set_ylabel(axes.get_ylabel(), fontsize=9)
            if not os.path.exists('fig'): 
                os.makedirs('fig')
            plt.savefig('fig/ma_figure3.png')
            plt.tight_layout()
            plt.show()
            print("Je vous sugère de passer au JeSetCreate()")
        else:
            print("Dans se cas je vous sugère de passer au JeSetCreate()")

def JeSetCreate(Data):
    print("Voulez-vous un decoupage par defaut ?")
    reponse = input('oui ou non\n')
    if reponse.lower() == 'oui':
        X_train, X_test = train_test_split(Data, test_size=0.2, random_state=42)
        Data_Set_Game = X_train
        Data_Set_Test = X_test
        print("X_Data_Set_Game:", len(Data_Set_Game), "X_Data_Set_Test:", len(Data_Set_Test))
        print("Vos données sont decoupées dans les varibles X Data_Set_Game et Data_Set_Test")
        return Data_Set_Game, Data_Set_Test
    else:
        print("Souhaitez-vous ajuster le ratio?")
        reponse2 = input("oui ou non\n")
        if reponse2.lower() == 'oui':
            print("Quel est le ratio")
            ratio = float(input('0.2 par defaut\n'))
            
            X_train, X_test = train_test_split(Data, test_size=ratio, random_state=42)
            Data_Set_Game = X_train
            Data_Set_Test = X_test
            print("X_Data_Set_Game:", len(Data_Set_Game), "X_Data_Set_Test:", len(Data_Set_Test))
            print(f"Vos données sont decoupées selon votre ratio de {ratio} dans les varibles X Data_Set_Game et Data_Set_Test")
            return Data_Set_Game, Data_Set_Test
    print("L'opreration est terminée !")

def SnakeEye(Data):
    Correlation = Data.corr()
    print("Souhaitez-vous une recherche par corrélation ?")
    reponse = input("oui ou non?\n")
    if reponse.lower() == "oui":
        print("Combien de corrélations souhaitez-vous effectuer ?")
        nb = int(input())
        list_cols = []
        for i in range(nb):
            print(Data.columns)
            print(f"Entrez le nom de la {i+1}e colonne")
            nom_col = input()
            if nom_col in Data.columns:
                list_cols.append(nom_col)
                print(Correlation[nom_col].sort_values(ascending=False))
            else:
                print(f"La colonne '{nom_col}' n'existe pas dans le DataFrame.")
        print("Vous etes arrivés au bout du nombre souhaitées")
        print("La valeur d'un coefficient de correlation est toujours comprise entre -1 et 1")
        print("+1 veut dire qu'il existe une forte correlation positive entre ces deux variables")
        print("-1 veut dire qu'il extise une forte correlation negtive entres ces deux variables")
    else:
        print("Aucune recherche par corrélation n'a été effectuée.")
    print("Souhaitez vous appliquer un scatter-matrix")
    reponse2 = input("All pour toutes ; Oui pour quelques une ; Non pour aucune \n ")
    if reponse2.lower() == 'oui':
        session = 'on' 
        while session == 'on':
            print("Quelle est le nombre de variables que vous voulez SccatterMatrix ?")
            nb = int(input())
            list_col = []
            for i in range(nb):
                print(Data.columns)
                print(f"Entrer le nom de la {i+1}e colonne")
                nom_col = input()
                if nom_col.lower() in Data.columns:
                    list_col.append(nom_col)
                
            ax = scatter_matrix(Data[list_col], alpha= 0.9, figsize=(12, 8), diagonal='hist' )
            for axes in ax.flatten():
                axes.set_title(axes.get_title(), fontsize=5)  # Ajustez la taille de police ici
                axes.set_xlabel(axes.get_xlabel(), fontsize=5)
                axes.set_ylabel(axes.get_ylabel(), fontsize=5)
            if not os.path.exists('fig'): 
                os.makedirs('fig')
            plt.savefig(f'fig/scatter1.{i+1}.png')
            plt.tight_layout()
            plt.show()
            print(f"Le Scatter_Matrix a été appliqué aux variables : {list_col}")
            print('Voulez vous faire un autre tableau ?')
            reponse3 = input('oui ou non;\n')
            if reponse3.lower() == 'oui':
                session = 'on' 
            else:
                session = 'off'
    elif(reponse2.lower() == 'all'):
        ax =scatter_matrix(Data, alpha=0.9, figsize=(12, 8) , diagonal='hist')
        for axes in ax.flatten():
                axes.set_title(axes.get_title(), fontsize=5)  # Ajustez la taille de police ici
                axes.set_xlabel(axes.get_xlabel(), fontsize=5)
                axes.set_ylabel(axes.get_ylabel(), fontsize=5)
        if not os.path.exists('fig'): 
            os.makedirs('fig')
        plt.savefig('fig/scatter2.png')
        plt.tight_layout()
        plt.show()
        print("Le Scatter_Matrix a été appliqué sur toutes les variables")
    
    print("Vous devriez prendre le temps d'essayer des exprimentation de combinason de variables avant de passer a l'etape suivante")

def Etic_or_not_Etic(Data):
    print("Voulez-vous separer les etiquette du jeu ?")#ou creer une colonne index
    reponse = input("oui ou non?:\n")
    if reponse.lower() == "oui":# == 'separer'
        print("Quelle est le nom de la colonne contenant les etiquettes ?")
        print(Data.columns)
        nom_col = input()
        Data_etiquettes = Data[nom_col].copy()
        Data_Set = Data.drop(nom_col, axis=1)
        print("Les etiquettes sont dans la variable Data_etiquettes et ont été supprimées du jeu dans la variable Data_Set")
        print("Voici le Data_Set :\n" ,Data_Set,)
        print("Voici le Data_etiquettes :\n" ,Data_etiquettes)
        return Data_Set, Data_etiquettes
    #elif reponse.lower() == "creer":
        Data_with__id = Data.reset_index()
        Data_Set_Game, Data_Set_Test = split_data_with_id_hash(Data_with__id, 0.2, index)
        print("Si vous utilisez l'indice de ligne comme identificateur unique, vous devez vous assurer que les nouvelles données seront ajoutées a la fin djeu de données")
        print("*Et qu'aucunes lignes ne sera JAMAIS SUPPRIMEES !")
        return Data_Set_Game, Data_Set_Test
    else:
        print("Si les etiquette de votre jeu ne sont pas déjà separées, je vous conseille de le faire !")
        
def sauvegarder_variables(variables, fichier='variables.json'):
    with open(fichier, 'w') as f:
        json.dump(variables, f)

def charger_variables(fichier='variables.json'):
    with open(fichier, 'r') as f:
        variables = json.load(f)
    return {key: pd.DataFrame.from_dict(value, orient='index') for key, value in variables.items()}

# Charger les variables sauvegardées
#variables_chargees = charger_variables()
#Data_Set_Game = variables_chargees["Data_Set_Game"]
#Data_Set_Test = variables_chargees["Data_Set_Test"]
#Data_Set_Game_num = variables_chargees["Data_Set_Game_num"]
#Data_Set_Game_cat = variables_chargees["Data_Set_Game_cat"]
#Data_Set_prepared = variables_chargees["Data_Set_prepared"]
#Data_Set = variables_chargees["Data_Set"]
#Data_Etiquette = variables_chargees["Data_Etiquette"]


print('Entrer le nom du fichier.csv')
fichier = input("")
Data = pd.read_csv(fichier, sep=',', on_bad_lines='warn' )

Data_copy = Data.copy()

while not isinstance(Data_copy, pd.DataFrame):
    print("Nous n'avons pas reussi a lire le fichier comment un dataframe")
else:
    biais_T = 0
    showme(Data_copy)
    
    Data_Set_Game, Data_Set_Test = JeSetCreate(Data_copy)
    
    Data_Set_Game_num = Data_Set_Game.select_dtypes(include=[np.number]) 
    Data_Set_Game_cat = Data_Set_Game.select_dtypes(exclude=[np.number])
    #Combinaison_var() ####### a ajouter !
    print('Votre Datasets est t-il composé de variables numeriques ?')
    num_v = input("Oui ou Non\n")
    if num_v.lower() == "oui":
        #Recherche de correllation
        SnakeEye(Data_Set_Game_num) #Nettoyage des données par les pipelines
        Data_Set_prepared, preprocessing = Pipeline_creator(Data_Set_Game)
        print(Data_Set_prepared)
    else:
        print("Une recherche par correlation est impossible car votre dataset ne comporte aucune données numeriques")
        Data_Set_prepared, preprocessing = pipeline_no_num(Data_Set_Game)
        print(Data_Set_prepared)
    
    
    
    
    print("Voulez-vous un jeu de validation ?")
    reponse = input("oui ou non?\n")
    if reponse.lower() == "oui":
        print("Quelle est la taille du jeu de validation ?")
        taille = float(input("0.2 par defaut\n"))
        Data_Set_prepared, Data_Set_v = train_test_split(Data_Set_prepared, test_size=taille, random_state=42)
        print(len(Data_Set_prepared), len(Data_Set_v))
        print(f"Data_Set_prepared c'est vu emputé de {taille} de ses données qui sont a present dans Data_Set_v")
        
    #Mise a l'ecart des etiquettes (cible)
    Data_Set, Data_Etiquette = Etic_or_not_Etic(Data_Set_prepared)
    
    variables = {
    "Data_Set_Game": Data_Set_Game.to_dict(),
    "Data_Set_Test": Data_Set_Test.to_dict(),
    "Data_Set_Game_num": Data_Set_Game_num.to_dict(),
    "Data_Set_Game_cat": Data_Set_Game_cat.to_dict(),
    "Data_Set_prepared": Data_Set_prepared.to_dict(),
    "Data_Set": Data_Set.to_dict(),
    "Data_Etiquette": Data_Etiquette.to_dict()
    }
    sauvegarder_variables(variables)
    print("Les données ont été sauvegardé")
    if biais_T > 0:
        print(f"Les données sont peut etre  biaisé a cause de {biais_T} action")
    
    
    from sklearn.linear_model import ElasticNet, LinearRegression, Lasso, Ridge, LogisticRegression
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
    from sklearn.svm import SVR, SVC
    from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.neural_network import MLPRegressor, MLPClassifier
    from sklearn.metrics import mean_squared_error
    from sklearn.experimental import enable_halving_search_cv  # noqa
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, HalvingRandomSearchCV, HalvingGridSearchCV
    from scipy.stats import uniform
    
    model_names = [
            "ElasticNet",
            "LinearRegression",
            "Lasso",
            "Ridge",
            "LogisticRegression",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
            "AdaBoostRegressor",
            "BaggingRegressor",
            "SVR",
            "SVC",
            "KNeighborsRegressor",
            "KNeighborsClassifier",
            "DecisionTreeRegressor",
            "DecisionTreeClassifier",
            "GaussianNB",
            "MultinomialNB",
            "BernoulliNB",
            "LinearDiscriminantAnalysis",
            "QuadraticDiscriminantAnalysis",
            "MLPRegressor",
            "MLPClassifier"
        ]
    
    def model_instance(model_name):
        # Vérifier si le modèle est dans le dictionnaire
        if model_name == "ElasticNet":
            model = ElasticNet()
        elif model_name == "LinearRegression":
            model = LinearRegression()
        elif model_name == "Lasso":
            model = Lasso()
        elif model_name == "Ridge":
            model = Ridge()
        elif model_name == "LogisticRegression":
            model = LogisticRegression()
        elif model_name == "RandomForestRegressor":
            model = RandomForestRegressor()
        elif model_name == "GradientBoostingRegressor":
            model = GradientBoostingRegressor()
        elif model_name == "AdaBoostRegressor":
            model = AdaBoostRegressor()
        elif model_name == "BaggingRegressor":
            model = BaggingRegressor()
        elif model_name == "SVR":
            model = SVR()
        elif model_name == "SVC":
            model = SVC()
        elif model_name == "KNeighborsRegressor":
            model = KNeighborsRegressor()
        elif model_name == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        elif model_name == "DecisionTreeRegressor":
            model = DecisionTreeRegressor()
        elif model_name == "DecisionTreeClassifier":
            model = DecisionTreeClassifier()
        elif model_name == "GaussianNB":
            model = GaussianNB()
        elif model_name == "MultinomialNB":
            model = MultinomialNB()
        elif model_name == "BernoulliNB":
            model = BernoulliNB()
        elif model_name == "LinearDiscriminantAnalysis":
            model = LinearDiscriminantAnalysis()
        elif model_name == "QuadraticDiscriminantAnalysis":
            model = QuadraticDiscriminantAnalysis()
        elif model_name == "MLPRegressor":
            model = MLPRegressor()
        elif model_name == "MLPClassifier":
            model = MLPClassifier()
        else:
            raise ValueError(f"Modèle non reconnu : {model_name}")
        return model
    
    def display_hyperparameters(model_name):
        model_hyperparameters = {
            "ElasticNet": {        
        "alpha": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de régularisation."
        },
        "l1_ratio": {
            "values": [0.1, 0.5, 0.9],
            "description": "Ratio entre la régularisation L1 (Lasso) et L2 (Ridge)."
        },
        "max_iter": {
            "values": [1000, 5000, 10000],
            "description": "Nombre maximum d'itérations."
        },
        "tol": {
            "values": [1e-4, 1e-3, 1e-2],
            "description": "Tolérance pour les critères d'arrêt."
        }
        },
            "LinearRegression": {        
        "fit_intercept": {
            "values": [True, False],
            "description": "Si True, ajoute une constante au modèle."
        },
        "normalize": {
            "values": [True, False],
            "description": "Si True, normalise les variables avant la régression."
        },
        "copy_X": {
            "values": [True, False],
            "description": "Si True, copie les données X ; sinon, peut les écraser."
        },
        "n_jobs": {
            "values": [-1, 1, 2],
            "description": "Nombre de jobs à utiliser pour le calcul. -1 signifie utiliser tous les processeurs."
        }
        },
            "Lasso": {        
        "alpha": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de régularisation L1."
        },
        "max_iter": {
            "values": [1000, 5000, 10000],
            "description": "Nombre maximum d'itérations."
        },
        "tol": {
            "values": [1e-4, 1e-3, 1e-2],
            "description": "Tolérance pour les critères d'arrêt."
        },
        "selection": {
            "values": ["cyclic", "random"],
            "description": "Méthode de sélection des mises à jour du coefficient."
        }
        },
            "Ridge": {        
        "alpha": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de régularisation L2."
        },
        "solver": {
            "values": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
            "description": "Algorithme utilisé pour l'optimisation."
        },
        "tol": {
            "values": [1e-4, 1e-3, 1e-2],
            "description": "Tolérance pour les critères d'arrêt."
        }
        },
            "LogisticRegression": {        
        "C": {
            "values": [0.1, 1.0, 10.0],
            "description": "Inverse de la force de régularisation."
        },
        "penalty": {
            "values": ["l1", "l2", "elasticnet", "none"],
            "description": "Type de régularisation."
        },
        "solver": {
            "values": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
            "description": "Algorithme utilisé pour l'optimisation."
        },
        "max_iter": {
            "values": [100, 200, 500],
            "description": "Nombre maximum d'itérations."
        }
        },
            "RandomForestRegressor": {        
        "n_estimators": {
            "values": [100, 200, 500],
            "description": "Nombre d'arbres dans la forêt."
        },
        "max_depth": {
            "values": [None, 10, 20],
            "description": "Profondeur maximale des arbres."
        },
        "min_samples_split": {
            "values": [2, 5, 10],
            "description": "Nombre minimum d'échantillons pour diviser un nœud."
        },
        "min_samples_leaf": {
            "values": [1, 2, 4],
            "description": "Nombre minimum d'échantillons dans une feuille."
        },
        "bootstrap": {
            "values": [True, False],
            "description": "Si True, utilise des échantillons bootstrap lors de la construction des arbres."
        }
        },
            "GradientBoostingRegressor": {        
        "learning_rate": {
            "values": [0.01, 0.1, 0.2],
            "description": "Taux d'apprentissage."
        },
        "n_estimators": {
            "values": [100, 200, 500],
            "description": "Nombre d'arbres."
        },
        "max_depth": {
            "values": [3, 5, 7],
            "description": "Profondeur maximale des arbres."
        },
        "min_samples_split": {
            "values": [2, 5, 10],
            "description": "Nombre minimum d'échantillons pour diviser un nœud."
        },
        "min_samples_leaf": {
            "values": [1, 2, 4],
            "description": "Nombre minimum d'échantillons dans une feuille."
        }
        },
            "AdaBoostRegressor": {        
        "n_estimators": {
            "values": [50, 100, 200],
            "description": "Nombre d'arbres."
        },
        "learning_rate": {
            "values": [0.01, 0.1, 1.0],
            "description": "Taux d'apprentissage."
        },
        "loss": {
            "values": ["linear", "square", "exponential"],
            "description": "Fonction de perte à utiliser."
        }
        },
            "BaggingRegressor": {        
        "n_estimators": {
            "values": [10, 50, 100],
            "description": "Nombre d'estimateurs de base."
        },
        "max_samples": {
            "values": [0.5, 0.7, 1.0],
            "description": "Fraction des échantillons à utiliser."
        },
        "max_features": {
            "values": [0.5, 0.7, 1.0],
            "description": "Fraction des caractéristiques à utiliser."
        },
        "bootstrap": {
            "values": [True, False],
            "description": "Si True, utilise des échantillons bootstrap."
        },
        "bootstrap_features": {
            "values": [True, False],
            "description": "Si True, utilise des échantillons bootstrap pour les caractéristiques."
        }
        },
            "SVR": {        
        "C": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de régularisation."
        },
        "kernel": {
            "values": ["linear", "poly", "rbf", "sigmoid"],
            "description": "Type de noyau."
        },
        "degree": {
            "values": [3, 4, 5],
            "description": "Degré du noyau polynomial."
        },
        "gamma": {
            "values": ["scale", "auto"],
            "description": "Coefficient du noyau."
        },
        "coef0": {
            "values": [0.0, 0.1, 0.5],
            "description": "Terme indépendant dans le noyau."
        }
        },
            "SVC": {        
        "C": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de régularisation."
        },
        "kernel": {
            "values": ["linear", "poly", "rbf", "sigmoid"],
            "description": "Type de noyau."
        },
        "degree": {
            "values": [3, 4, 5],
            "description": "Degré du noyau polynomial."
        },
        "gamma": {
            "values": ["scale", "auto"],
            "description": "Coefficient du noyau."
        },
        "coef0": {
            "values": [0.0, 0.1, 0.5],
            "description": "Terme indépendant dans le noyau."
        }
        },
            "KNeighborsRegressor": {        
        "n_neighbors": {
            "values": [3, 5, 10],
            "description": "Nombre de voisins à utiliser."
        },
        "weights": {
            "values": ["uniform", "distance"],
            "description": "Fonction de pondération."
        },
        "algorithm": {
            "values": ["auto", "ball_tree", "kd_tree", "brute"],
            "description": "Algorithme utilisé pour la recherche des voisins."
        },
        "leaf_size": {
            "values": [30, 50, 100],
            "description": "Taille des feuilles pour l'arbre."
        },
        
        "p": {
            "values": [1, 2],
            "description": "Paramètre de la distance de Minkowski."
        }
        },
            "KNeighborsClassifier": {        
        "n_neighbors": {
            "values": [3, 5, 10],
            "description": "Nombre de voisins à utiliser."
        },
        "weights": {
            "values": ["uniform", "distance"],
            "description": "Fonction de pondération."
        },
        "algorithm": {
            "values": ["auto", "ball_tree", "kd_tree", "brute"],
            "description": "Algorithme utilisé pour la recherche des voisins."
        },
        "leaf_size": {
            "values": [30, 50, 100],
            "description": "Taille des feuilles pour l'arbre."
        },
        "p": {
            "values": [1, 2],
            "description": "Paramètre de la distance de Minkowski."
        }
        },
            "DecisionTreeRegressor": {        
        "max_depth": {
            "values": [None, 10, 20],
            "description": "Profondeur maximale de l'arbre."
        },
        "min_samples_split": {
            "values": [2, 10, 20],
            "description": "Nombre minimum d'échantillons pour diviser un nœud."
        },
        "min_samples_leaf": {
            "values": [1, 2, 4],
            "description": "Nombre minimum d'échantillons dans une feuille."
        },
        "max_features": {
            "values": ["auto", "sqrt", "log2"],
            "description": "Nombre de caractéristiques à considérer pour trouver la meilleure division."
        }
        },
            "DecisionTreeClassifier": {        
        "max_depth": {
            "values": [None, 10, 20],
            "description": "Profondeur maximale de l'arbre."
        },
        "min_samples_split": {
            "values": [2, 10, 20],
            "description": "Nombre minimum d'échantillons pour diviser un nœud."
        },
        "min_samples_leaf": {
            "values": [1, 2, 4],
            "description": "Nombre minimum d'échantillons dans une feuille."
        },
        "max_features": {
            "values": ["auto", "sqrt", "log2"],
            "description": "Nombre de caractéristiques à considérer pour trouver la meilleure division."
        }
        },
        "GaussianNB": {        
        "var_smoothing": {
            "values": [1e-9, 1e-8, 1e-7],
            "description": "Portion de la plus grande variance des caractéristiques ajoutée à la variance pour la stabilité numérique."
        }
        },
        "MultinomialNB": {        
        "alpha": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de lissage."
        },
        "fit_prior": {
            "values": [True, False],
            "description": "Si True, apprend les probabilités a priori des classes."
        }
        },
            "BernoulliNB": {        
        "alpha": {
            "values": [0.1, 1.0, 10.0],
            "description": "Paramètre de lissage."
        },
        "binarize": {
            "values": [0.0, 0.5, 1.0],
            "description": "Seuil pour binariser les données."
        },
        "fit_prior": {
            "values": [True, False],
            "description": "Si True, apprend les probabilités a priori des classes."
        }
        },
        "LinearDiscriminantAnalysis": {        
        "solver": {
            "values": ["svd", "lsqr", "eigen"],
            "description": "Algorithme utilisé pour l'optimisation."
        },
        "shrinkage": {
            "values": [None, "auto", 0.1, 0.5, 1.0],
            "description": "Paramètre de régularisation."
        }
        },
        "QuadraticDiscriminantAnalysis": {        
        "reg_param": {
            "values": [0.0, 0.1, 0.5],
            "description": "Paramètre de régularisation."
        },
        "store_covariance": {
            "values": [True, False],
            "description": "Si True, stocke les matrices de covariance."
        },
        "tol": {
            "values": [1e-4, 1e-3, 1e-2],
            "description": "Tolérance pour les critères d'arrêt."
        }
        },
        "MLPRegressor": {
        "hidden_layer_sizes": {
            "values": [(100,), (50, 50), (100, 100, 100)],
            "description": "Taille des couches cachées."
        },
        "activation": {
            "values": ["relu", "tanh", "logistic"],
            "description": "Fonction d'activation."
        },
        "solver": {
            "values": ["lbfgs", "sgd", "adam"],
            "description": "Algorithme utilisé pour l'optimisation."
        },
        "alpha": {
            "values": [0.0001, 0.001, 0.01],
            "description": "Paramètre de régularisation L2."
        },
        "learning_rate": {
            "values": ["constant", "invscaling", "adaptive"],
            "description": "Horaire de mise à jour du taux d'apprentissage."
        }
        },
        "MLPClassifier": {
        "hidden_layer_sizes": {
            "values": [(100,), (50, 50), (100, 100, 100)],
            "description": "Taille des couches cachées."
        },
        "activation": {
            "values": ["relu", "tanh", "logistic"],
            "description": "Fonction d'activation."
        },
        "solver": {
            "values": ["lbfgs", "sgd", "adam"],
            "description": "Algorithme utilisé pour l'optimisation."
        },
        "alpha": {
            "values": [0.0001, 0.001, 0.01],
            "description": "Paramètre de régularisation L2."
        },
        "learning_rate": {
            "values": ["constant", "invscaling", "adaptive"],
            "description": "Horaire de mise à jour du taux d'apprentissage."
        }
        }
        }
        if model_name in model_hyperparameters:
            print(f"Les hyperparamètres pour {model_name} sont : {model_hyperparameters[model_name]}")
        else:
            print(f"Modèle non reconnu : {model_name}")
    def SupGridSearch(Data, Data_Etiquette):
        print("Combien de modèles souhaitez-vous régler avec précision avec GridSearch ?")
        nb = int(input())
        models = []
        #
        for i in range(nb):
            print(f"Modèle {i+1}:") 
            print(model_names)
            model_name = input("Nom du modèle : ")
            display_hyperparameters(model_name)
            para = eval(input("Dictionnaire des paramètres : "))
            cv = int(input("Nombre de validations croisées (cv) : "))
            scoring = input("Métrique de scoring : ")
            
            model = model_instance(model_name)

            grid = GridSearchCV(estimator=model, param_grid=para, cv=cv, scoring=scoring)
            grid.fit(Data, Data_Etiquette)
    
            print(f"Les meilleurs paramètres du modèle {model_name} sont :\n", grid.best_params_)
            print(f"Le meilleur score du modèle {model_name} est :\n", grid.best_score_)
            print("Les résultats :\n", pd.DataFrame(grid.cv_results_))

            best_model = grid.best_estimator_
            models.append(best_model)
        print("Les meilleurs modèles :\n", models)
        return models
    def SupRandomSearch(Data, Data_Etiquette):
        print("Combien de modèles souhaitez-vous régler avec précision avec RandomSearch ?")
        nb = int(input())
        models = []
        #
        for i in range(nb):
            print(f"Modèle {i+1}:") 
            print(model_names)
            model_name = input("Nom du modèle : ")
            display_hyperparameters(model_name)
            para = eval(input("Dictionnaire des paramètres : ex:'alpha': uniform(0, 1)\n ; "))
            cv = int(input("Nombre de validations croisées (cv) : "))
            scoring = input("Métrique de scoring : ")
            n_iter = int(input("Nombre d'itérations (n_iter) : "))
        
            model = model_instance(model_name)

            random_search = RandomizedSearchCV(estimator=model, param_distributions=para, n_iter=n_iter, cv=cv, scoring=scoring)
            random_search.fit(Data, Data_Etiquette)
        
            print(f"Les meilleurs paramètres du modèle {model_name} sont :\n", random_search.best_params_)
            print(f"Le meilleur score du modèle {model_name} est :\n", random_search.best_score_)
            print("Les résultats :\n", pd.DataFrame(random_search.cv_results_))

            best_model = random_search.best_estimator_
            models.append(best_model)
        print("Les meilleurs modèles :\n", models)
        return models
    def SupHalvingGridSearch(Data, Data_Etiquette):
        print("Combien de modèles souhaitez-vous régler avec précision avec HalvingGridSearch ?")
        nb = int(input())
        models = []
        #
        for i in range(nb):
            print(f"Modèle {i+1}:") 
            print(model_names)
            model_name = input("Nom du modèle : ")
            display_hyperparameters(model_name)
            para = eval(input("Dictionnaire des paramètres : "))
            cv = int(input("Nombre de validations croisées (cv) : "))
            scoring = input("Métrique de scoring : ")
        
            model = model_instance(model_name)

            halving_grid = HalvingGridSearchCV(estimator=model, param_grid=para, cv=cv, scoring=scoring, factor=2, random_state=0)
            halving_grid.fit(Data, Data_Etiquette)

            print(f"Les meilleurs paramètres du modèle {model_name} sont :\n", halving_grid.best_params_)
            print(f"Le meilleur score du modèle {model_name} est :\n", halving_grid.best_score_)
            print("Les résultats :\n", pd.DataFrame(halving_grid.cv_results_))

            best_model = halving_grid.best_estimator_
            models.append(best_model)
        print("Les meilleurs modèles :\n", models)
        return models
    def SupHalvingRandomSearch(Data, Data_Etiquette):
        print("Combien de modèles souhaitez-vous régler avec précision avec HalvingRandomSearch ?")
        nb = int(input())
        models = []
        #
        for i in range(nb):
            print(f"Modèle {i+1}:") 
            print(model_names)
            model_name = input("Nom du modèle : ")
            display_hyperparameters(model_name)
            para = eval(input("Dictionnaire des paramètres : "))
            cv = int(input("Nombre de validations croisées (cv) : "))
            scoring = input("Métrique de scoring : ")
        
            model = model_instance(model_name)

            halving_random = HalvingRandomSearchCV(estimator=model, param_distributions=para, cv=cv, scoring=scoring, factor=2, random_state=0)
            halving_random.fit(Data, Data_Etiquette)

            print(f"Les meilleurs paramètres du modèle {model_name} sont :\n", halving_random.best_params_)
            print(f"Le meilleur score du modèle {model_name} est :\n", halving_random.best_score_)
            print("Les résultats :\n", pd.DataFrame(halving_random.cv_results_))

            best_model = halving_random.best_estimator_
            models.append(best_model)
        print("Les meilleurs modèles :\n", models)
        return models
    
    #Données
    X = Data_Set
    y = Data_Etiquette
    
    print("Voulez-vous utliser (Grid ou Randomized) SearchCV ou leurs equivalents respectifs halving(grid ou random) search ?")
    regulateur = input() 
    if regulateur.lower() == "grid":
        models = SupGridSearch(X, y)
    elif regulateur.lower() == 'randomized':
        models = SupRandomSearch(X, y)
    elif regulateur.lower() == 'halvinggrid':
        models = SupHalvingGridSearch(X, y)
    elif regulateur.lower() == 'halvingrandom':
        models = SupHalvingRandomSearch(X, y)  
    else: 
        raise ValueError("Regulateur non reconnu")
        
    #Modele
    import joblib

    def save_model(model, filename):
        """
        Sauvegarde le modèle donné dans un fichier.
    
        Parameters:
        model (object): Le modèle à sauvegarder.
        filename (str): Le nom du fichier où sauvegarder le modèle.
        """
        joblib.dump(model, filename)
        print(f"Modèle sauvegardé sous le nom de fichier : {filename}")
    
    print("Voulez-vous sauvegarder le modele ?")
    save = input("oui ou non \n")
    if save.lower() == 'oui':
        save_model(models, 'mon_modele_final.pkl')
        print("Pour charger le modèle utiliser: loaded_model = joblib.load('mon_modele_final.pkl')!!!")
    else:
        print("Vous n'avez pas sauvegarger le modele .")
