
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
# numberone = 1

def run_kfold(clf, X, y):
    kf = KFold(n_splits=10, shuffle=False, random_state=35)
    outcomes = []
    fold = 0
    for train_index, test_index in kf.split(X):
        fold += 1
        X_train, X_test = X.values[train_index], X.values[test_index]
        y_train, y_test = y.values[train_index], y.values[test_index]
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        outcomes.append(accuracy)
        print("Grupo {0} precisión: {1}".format(fold, accuracy))     
    mean_outcome = np.mean(outcomes)
    print("Precisión promedio: {0}".format(mean_outcome)) 

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
        return print(df.isnull().sum())
    
    def delduplicates(df):
        l1 = len(df)
        df.drop_duplicates(inplace=True)
        print("Eliminó %i observaciones de un total de %i" % ((l1-len(df)), l1))
        print("Nuevas dimensiones: ", df.shape)

    def clean_data(df):
        delduplicates(df)
    
    def clean_dataset(df):
        assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
        df.dropna(inplace=True)
        indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
        return df[indices_to_keep].astype(np.float64)
    
    # Observar todas las columnas en el juego de datos
    def printall(X, max_rows=10):
        display(HTML(X.to_html(max_rows=max_rows)))

    # Función para mostrar las estadisticas descriptivas de las variables categóricas
    def describe_categorical(X):
        """
        Similar a .describe() de pandas, pero devulve los resultados para variables categóricas unicamente
        """
        try:
            display(HTML(X[X.columns[X.dtypes== "object"]].describe().to_html()))
        except TypeError:
            return print("No hay variables categóricas")

    def balanced_spl_by(df, lblcol, uspl=True):
        datas_l = [ df[df[lblcol]==l].copy() for l in list(set(df[lblcol].values)) ]
        lsz = [f.shape[0] for f in datas_l ]
        return pd.concat([f.sample(n = (min(lsz) if uspl else max(lsz)), replace = (not uspl)).copy() for f in datas_l ], axis=0 ).sample(frac=1) 
    
    def display_info_numericas(X):
        # Obtenemos las variables númericas seleccionando solamente las variables que no son de tipo "object"
        #X.info() # Muestra los tipos de datos
        #numeric_variables = list(X.dtypes[X.dtypes != "object"].index)
        #X[numeric_variables].head()
        try:
            display(HTML(X[X.columns[X.dtypes!= "object"]].describe().to_html()))
        except TypeError:
            return print("No hay variables númericas")
    
    #A helper method for pretty-printing linear models
    def pretty_print_linear(coefs, names = None, sort = False):
        if (names == None):
            names = ["X%s" % x for x in range(len(coefs))]
        lst = zip(coefs, names)
        if sort:
            lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
        return " + ".join("%s * %s" % (round(coef, 3), name) for coef, name in lst)


    def create_dummies(df, categorical_variables):
        '''
        Como todas las variables tienen mas de 2 valores en sus atributos, podemos usar la función get_dummies de pandas.
        De lo contrario solo debemos utilizar "factorize", para crear variables nominales binarias a numericas;
        de igual modo para variables categóricas con más de 2 valores pero "ordinales"
        '''
        for variable in categorical_variables:
            # Crear variables dummy
            if (variable in df.columns):
                dummies = pd.get_dummies(df[variable], prefix=variable).iloc[:, 1:]
                # Update X para incluir las dummies y eliminar la variable principal
                df = pd.concat([df, dummies], axis=1)
                df.drop([variable], axis=1, inplace=True) # Eliminar las variables categóricas
        
        return df

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
    
    def plot_confusion_matrix(cm, classes, normalize=False, title='Matríz de Confusión',
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Matríz de Confusión Normalizada")
        else:
            print('Matríz de Confusión sin Normalización')

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

    def save_data():
        X.to_csv('data/data_train_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        y_all.to_csv('data/data_y_all_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        y_4f.to_csv('data/data_y_4f_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        y_bin.to_csv('data/data_y_bin_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        ataques_10perc.to_csv('data/data_train_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        ataques_correg_test_10perc.to_csv('data/data_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        ataques_correg_test_10percAll.to_csv('data/data_10per_test_preprocessed.csv', sep=',', decimal='.', header=True, index=False)
        # Guardamos un archivo CSV con las 40 variables mas importantes
        feature_selected = pd.concat([X_train[X_train.columns[indices][:40]], y_train], axis=1)
        feature_selected_small = pd.concat([X_train[X_train.columns[indices][:15]], y_train], axis=1)
        feature_selected_small_var5 = pd.concat([X_train[X_train.columns[indices][:5]], y_train], axis=1)

        feature_selected.to_csv('data/data_train_preprocessed_40var.csv', sep=',', decimal='.', header=True, index=False)
        feature_selected_small.to_csv('data/data_train_preprocessed_15var.csv', sep=',', decimal='.', header=True, index=False)
        feature_selected_small_var5.to_csv('data/data_train_preprocessed_5var.csv', sep=',', decimal='.', header=True, index=False)
    


                                                    