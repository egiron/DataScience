
from time import time
from itertools import product
import itertools
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import graphviz
from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from IPython.display import display, HTML

plt.style.use('seaborn-white')


description = "Clase con funciones predefinidas para realizar Análisis Exploratorio de datos - EDA"
author = "Ernesto Girón E."

# Define some global variables:
numberone = 1

# define some functions
def timesfour(input):
    print (input * 4)

class EDA:
    def __init__(self):
        #self.type = raw_input("What type of piano? ")
        pass

    # Carga los datos crudos de intrusos en redes
    # Devuelve los conjuntos de datos de entrenamiento y validación
    def load_attack_rawData():
        ataques_10perc = pd.read_csv('data/kddcup.data_10_percent', sep=',', decimal='.', header=None)
        ataques_correg_test_10perc = pd.read_csv('data/corrected', sep=',', decimal='.', header=None)
        # Como los datos estan sin cabeceera, se procede a colocarle
        atributos = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 
                     'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 
                     'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 
                     'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 
                     'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 
                     'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 
                     'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
                     'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_types']

        ataques_10perc.columns = atributos
        ataques_correg_test_10perc.columns = atributos

        print("Cantidad de observaciones (Entrenamiento): ",len(ataques_10perc))
        print("Cantidad de observaciones (Validación): ",len(ataques_correg_test_10perc))
        print("Cantidad de ataques (Entrenamiento): ",len(ataques_10perc.attack_types.unique()))
        print("Cantidad de ataques (Validación): ",len(ataques_correg_test_10perc.attack_types.unique()))

        return ataques_10perc, ataques_correg_test_10perc

    def create_category_attack(df):
        # Crear un nuevo predictor de categorías de tipos de ataques
        df['attack_category'] = df.attack_types.map({'normal': 'normal', 'buffer_overflow':'u2r', 
                                                       'loadmodule':'u2r', 'perl':'u2r', 'neptune':'dos',
                                                       'smurf':'dos','guess_passwd':'r2l', 'pod':'dos', 
                                                       'teardrop':'dos', 'portsweep':'probe','ipsweep':'probe',
                                                       'land':'dos','ftp_write':'r2l','back':'dos','imap':'r2l',
                                                       'satan':'probe','phf':'r2l','nmap':'probe',
                                                       'multihop':'r2l','warezmaster':'r2l','warezclient':'r2l',
                                                       'spy':'r2l','rootkit':'u2r', 'snmpgetattack':'unknown', 
                                                       'named':'unknown', 'xlock':'unknown', 'xsnoop':'unknown', 
                                                        'sendmail':'unknown', 'saint':'unknown','apache2':'unknown', 
                                                        'udpstorm':'unknown','xterm':'unknown', 'mscan':'unknown', 
                                                        'processtable':'unknown', 'ps':'unknown','httptunnel':'unknown', 
                                                        'worm':'unknown', 'mailbomb':'unknown','sqlattack':'unknown', 
                                                        'snmpguess':'unknown' })

        print("Categorías encontradas:", df.attack_category.unique())

    def create_category_binAttack(df):
        # Crear un nuevo predictor de categorías de tipos de ataques
        df['attack_category_bin'] = df.attack_category.map({'normal': 0, 'unknown':1, 'dos':1, 'probe':1, 'r2l':1, 'u2r':1 })
        print("Categorías encontradas:", df.attack_category_bin.unique())

    def replace_column_string(df, col, txt, new_txt):
        df[col] = df[col].str.replace(txt,new_txt)
    
    def hasNull(df):
        return df.isnull().sum()
    
    def delduplicates(df):
        l1 = len(df)
        df.drop_duplicates(inplace=True)
        print("Eliminó %i observaciones de un total de %i" % (len(df), l1))

    def clean_data(df):
        delduplicates(df)
    
    # Observar todas las columnas en el juego de datos
    def printall(X, max_rows=10):
        display(HTML(X.to_html(max_rows=max_rows)))

    # Función para mostrar las estadisticas descriptivas de las variables categóricas
    def describe_categorical(X):
        """
        Similar a .describe() de pandas, pero devulve los resultados para variables categóricas unicamente
        """
        display(HTML(X[X.columns[X.dtypes== "object"]].describe().to_html()))
    
    def display_info_numericas(X):
        # Obtenemos las variables númericas seleccionando solamente las variables que no son de tipo "object"
        X.info() # Muestra los tipos de datos
        numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
        X[numeric_variables].head()

    # categorical_variables = ['sex', 'cabin', 'embarked']
    def create_dummies(categorical_variables):
        for variable in categorical_variables:
            # Crear variables dummy 
            dummies = pd.get_dummies(X[variable], prefix=variable).iloc[:, 1:]
            # Update X para incluir las dummies y eliminar la variable principal
            X = pd.concat([X, dummies], axis=1)
            X.drop([variable], axis=1, inplace=True) # Eliminar las variables categóricas


    def graficar_importancia(X, regr):
        feature_importance = regr.feature_importances_*100
        rel_imp = pd.Series(feature_importance, index=X.columns).sort_values(inplace=False)
        print(rel_imp)
        rel_imp.T.plot(kind='barh', color='r', )
        plt.xlabel('Importancia de la Variable')
        plt.gca().legend_ = None

    def graficar_varianzaExp(pca):
        # Gráfico de Proporción de la Varianza Explicada
        plt.figure(figsize=(7,5))
        plt.plot([1,2,3,4], pca.explained_variance_ratio_, '-o', label='Componente Individual')
        plt.plot([1,2,3,4], np.cumsum(pca.explained_variance_ratio_), '-s', label='Acumulado')
        plt.ylabel('Proporción de la Varianza Explicada')
        plt.xlabel('Componente Principal')
        plt.xlim(0.75,4.25)
        plt.ylim(0,1.05)
        plt.xticks([1,2,3,4])
        plt.legend(loc=2)
    
    def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    
