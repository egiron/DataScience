# Detección de Intrusos en Redes LAN
# Solución Trabajo final Diplomado Data Mining Univalle 2017
# Author: Ernesto Girón E. - e.giron.e at gmail
# Nov 21, 2017

## DETECCIÓN DE INTRUSOS EN REDES

# Estos datos fueron usados para la edición de 1999 del KDD cup. 
# Los datos fueron generados por Lincoln Labs: Nueve semanas de registro de paquetes 
# TCP fueron recolectadas para una red LAN de una oficina de las fuerzas aéreas de USA.
# Durante el uso de la LAN, _varios ataques_ fueron ejecutados por el personal. 
# El paquete crudo fue agregado junto con la información de la conexión. 
# 
# Para cada registro, algunas características extra fueron derivadas, basados en 
# conocimiento del dominio sobre ataques a redes; hay 38 tipos diferentes de ataques, 
# pertenecientes a 4 categorías principales. Algunos tipos de ataque aparecen solo en 
# los datos de prueba(test data), y las frecuencias de los tipo de ataque en los 
# conjuntos de entrenamiento y prueba no son las mismas(para hacerlo más realista). 
# Información adicional sobre los datos puede ser encontrada 
# en (http://kdd.ics.uci.edu/databases/kddcup99/task.html) y los resumenes de los 
# resultados de la competencia KDD cup (http://cseweb.ucsd.edu/~elkan/clresults.html). 
# En la última página también se indica que hay una matriz de costo asociada con las 
# equivocaciones.  El ganador de la competencia usó árboles de decisión C5 en combinación 
# con boosting y bagging.
# 
# **Referencias**:
#   - PNrule: _A New Framework for Learning Classifier Models in Data Mining 
#    (A Case-Study in Network Intrusion Detection) (2000) by R. Agarwal and 
#    M. V. Joshi_. This paper proposes a new, very simple rule learning algorithm, 
#    and tests it on the network intrusion dataset. In the first stage, rules are 
#    learned to identify the target class, and then in the second stage, rules are 
#    learned to identify cases that were incorrectly classified as positive 
#    according to the first rules.

# Pasos solución 1
# 
# 1. Cargar las librerías a utilizar
# 2. Cargar los datos e importarlos a un dataframe
# 3. Visualizar los datos
# 4. Limpiar y transformar los datos
# 5. Códificar los datos
# 6. Seleccionar los parámetros más importantes
# 7. Separando el conjunto de datos de entrenamiento y de validación
# 8. Selección de algoritmos y métodos
# 9. Resumen de los métodos utilizados
# 10. Comparación de resultados con el ganador del KDDCup

# ---------------------------------------------------------
# 1. Cargar las librerías a utilizar
# ---------------------------------------------------------
# Instalar las librerias si no las han descargado
# install.packages("dplyr")
# install.packages("dummies")
# install.packages("mlr")
# install.packages("reshape")
# install.packages("Hmisc")
# install.packages("corrplot")
# install.packages("tree")
# install.packages("party")
# install.packages("partykit")
# install.packages("randomForest")
# install.packages("e1071")
# install.packages("gbm")
# install.packages("caret")
# install.packages("ranger")

# llamado a las librerías
library("dplyr")
library("plyr")
library("dummies")
# library("mlr")
library("ggplot2")
library("reshape")
library("leaps")
library("Hmisc")
library("corrplot")
library("RColorBrewer")
library("tree")
library("party")
library("rpart")
library("partykit")
library("randomForest")
library("ranger")
library("e1071")
library("gbm")
library("caret")


# ---------------------------------------------------------
# 2. Cargar los datos e importarlos a un dataframe
# ---------------------------------------------------------

# Definimos el directorio de trabajo
# setwd("C:/Users/Invitado/Desktop/egiron/Trabajo")  # Path PC No. 30 Univalle
# setwd("~/Desktop/DiplomadoUnivalle_DS2017/statistical_learning/ISLR/Ejercicios")

# Cargamos los datos de entrenamiento y validacón
# data_10_percent_train <- read.csv(file.choose())
data_10_percent_train <- read.csv("data/kddcup.data_10_percent", stringsAsFactors=FALSE, header = FALSE, sep = ',', dec = '.')
data_10_percent_test <- read.csv("data/corrected", stringsAsFactors=FALSE, header = FALSE, sep = ',', dec = '.')

# Asignar los nombres a las columnas
nomcols <- c("duration", "protocol_type", "service", "flag", "src_bytes", 
             "dst_bytes", "land", "wrong_fragment", 
             "urgent", "hot", "num_failed_logins", 
             "logged_in", "num_compromised", "root_shell",
             "su_attempted", "num_root", "num_file_creations", 
             "num_shells", "num_access_files", "num_outbound_cmds",
             "is_host_login", "is_guest_login", "count", "srv_count",
             "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate", 
             "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", 
             "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
             "dst_host_diff_srv_rate", "dst_host_same_src_port_rate", 
             "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
             "dst_host_srv_serror_rate", "dst_host_rerror_rate", 
             "dst_host_srv_rerror_rate", "attack_type")

# Asignar los nombres a las columnas a cada conjunto de datos
colnames(data_10_percent_train) <- nomcols
colnames(data_10_percent_test) <- nomcols

# Cantidad de observaciones y de variables
dim(data_10_percent_train)
dim(data_10_percent_test)


# ---------------------------------------------------------
# 3. Visualizar los datos
# ---------------------------------------------------------
View(data_10_percent_train) # Despliegue la tabla completa
# View(data_10_percent_test)

head(data_10_percent_train, n=3) # Muestre las primeras 3 observaciones
tail(data_10_percent_train, n=3) # Muestre las últimas 3 observaciones

summary(data_10_percent_train) # Muestre un resumen de los datos
#summary(data_10_percent_test)

# Visualizamos la estructura de los datos, el tipo de dato o clase de cada predictor
str(data_10_percent_train)
#str(data_10_percent_test)

# Visualizamos los tipos de ataques y su frecuencia.
unique(data_10_percent_train$attack_type) # Valores unicos de tipos de ataque
unique(data_10_percent_test$attack_type) # Valores unicos de tipos de ataque

# Convertimos los valores a factores para hacer operaciónes con ellos
tipos_ataque = as.factor(data_10_percent_train$attack_type)
levels(tipos_ataque)
# typeof(tipos_ataque)
summary(tipos_ataque) # Visualizamos la cantidad de intrusos por tipo de ataque

# Graficamos rápidamente estos datos
par(mfrow=c(1,1))
plot(tipos_ataque,xlab="Tipos de Ataques",ylab="Cantidad de ataques", 
     main="Cantidad de Intrusos por Tipo de Ataque", las=2, col="red", 
     cex.main=1.5, cex.lab=1.2, cex.axis=0.6, cex.sub=1.2) # font =2, family = 'Arial'

# ---------------------------------------------------------
# 4. Limpiar y transformar los datos
# ---------------------------------------------------------

# Reemplazamos el punto '.' final en los valores de la variable tipo de ataque (attack_type)
data_10_percent_train$attack_type <- gsub(".", "", data_10_percent_train$attack_type, fixed = TRUE)
data_10_percent_test$attack_type <- gsub(".", "", data_10_percent_test$attack_type, fixed = TRUE)
# View(data_10_percent_train)
# summary(as.factor(data_10_percent_train$attack_type))

# Gráficamos de nuevo
tipos_ataque_train = factor(data_10_percent_train$attack_type, ordered = TRUE)
tipos_ataque_test = as.factor(data_10_percent_test$attack_type)
levels(tipos_ataque_train)
levels(tipos_ataque_test)

par(mfrow=c(2,1))
plot(tipos_ataque_train, ylab="Cantidad de ataques", 
     main="Cantidad de Intrusos por Tipo de Ataque", las=2, col="red", 
     cex.main=1.5, cex.lab=1.2, cex.axis=0.6, cex.sub=1.2) # font=2, family = 'Arial'

plot(tipos_ataque_test,xlab="Tipos de Ataques",ylab="Cantidad de ataques", 
     las=2, col="red", cex.main=1.5, cex.lab=1.2, cex.axis=0.6, cex.sub=1.2) # font=2, family = 'Arial'

# Como se puede observar la cantidad de tipos de ataques en los datos de entrenamiento
# y los datos de validación son diferentes como se menciona en la literatura o página web del concurso

# Revisamos valores faltantes o nulos en los dos juegos de datos
sum(is.na.data.frame(data_10_percent_train))
sum(is.na.data.frame(data_10_percent_test))

# Eliminamos duplicados
# Existen varias formas de eliminar las observaciones duplicadas
# Utilizando, las funciones duplicated, unique o la librería dplyr
# data_10_percent_train<-data_10_percent_train[duplicated(data_10_percent_train),]

# Si desea ver cuales son los valores duplicados
#data_10_percent_train[duplicated(data_10_percent_train),]

# La función duplicated solo devuelve los indices donde se encuentran los valores duplicados
# Por lo que es necesario combinarla con otras operaciones así:
data_10_percent_train<-data_10_percent_train[!duplicated(data_10_percent_train), ]
dim(data_10_percent_train)

# Creamos un nuevo conjunto de datos o dataset con los datos no duplicados de Validación
# Esto con el fin de que nuestros modelos sean mas eficientes, pero dejando los datos de 
# Validación originales por que son los que nos permiten comparar al final con los 
# resultados del ganador del concurso
data_10_percent_test_small<-data_10_percent_test[duplicated(data_10_percent_test),]
dim(data_10_percent_test)
dim(data_10_percent_test_small)

# La otra forma de hacerlo
# unique(data_10_percent_train) Remueve duplicados en un solo comando

# Una forma mas eficiente de hacerlo en lugar de unique() es utilizando la libreria dplyr
# Remueve los valores duplicados basandose en todas las columnas
#distinct(data_10_percent_train)
#dim(data_10_percent_train)

# Si desea cambiar un valor específico manualmente
#data_10_percent_train_corregido <- edit(data_10_percent_train)
#fix(data_10_percent_train) # equivalente a anterior

# ---------------------------------------------------------
# 5. Códificar los datos
# ---------------------------------------------------------

# Como se observó anteriormente hay 4 variables categóricas incluyendo la independiente
# para lo cual debemos de modificar con el fin de que los algortimos entiendan y se pueda
# trabajar con ellas. Es mas eficiente para un computador entender números.

# Dicho lo anterior empezamos categorizando nuestros ataques en 4 categorias que especifican en el concurso
# Fuente: http://kdd.ics.uci.edu/databases/kddcup99/training_attack_types

data_10_percent_train$attack_category <- data_10_percent_train$attack_type
data_10_percent_test$attack_category <- data_10_percent_test$attack_type
data_10_percent_test_small$attack_category <- data_10_percent_test_small$attack_type
#dim(data_10_percent_train)

ataques_tipo = c('normal','buffer_overflow','loadmodule','perl','rootkit','back','neptune',
                 'smurf','pod','teardrop','land','guess_passwd','ftp_write','imap',
                 'phf','multihop','warezmaster','warezclient','spy','portsweep',
                 'ipsweep','satan','nmap','snmpgetattack', 'named', 'xlock', 
                 'xsnoop', 'sendmail', 'saint','apache2', 'udpstorm','xterm', 
                 'mscan', 'processtable', 'ps','httptunnel', 'worm', 'mailbomb',
                 'sqlattack', 'snmpguess')

ataques_ctg = c('normal','u2r','u2r','u2r','u2r','dos','dos','dos','dos','dos','dos',
                'r2l','r2l','r2l','r2l','r2l','r2l','r2l','r2l','probe','probe',
                'probe','probe','unknown', 'unknown', 'unknown', 'unknown', 'unknown',
                'unknown','unknown', 'unknown','unknown', 'unknown', 'unknown', 
                'unknown','unknown', 'unknown', 'unknown','unknown', 'unknown')

data_10_percent_train$attack_category <- mapvalues(data_10_percent_train$attack_category, 
                                                   from=ataques_tipo, 
                                                   to=ataques_ctg)

data_10_percent_test$attack_category <- mapvalues(data_10_percent_test$attack_category, 
                                                  from=ataques_tipo, 
                                                  to=ataques_ctg)

data_10_percent_test_small$attack_category <- mapvalues(data_10_percent_test_small$attack_category, 
                                                        from=ataques_tipo, 
                                                        to=ataques_ctg)
# Vefificamos los nuevos datos
summary(as.factor(data_10_percent_train$attack_category))
# View(data_10_percent_train)

# Graficamos
par(mfrow=c(1,1))
categoria_ataques = as.factor(data_10_percent_train$attack_category)
plot(categoria_ataques,xlab="Categorias de Tipos de Ataque",ylab="Cantidad de ataques", 
     main="Cantidad de Intrusos \npor Categorias de Ataque", las=2, col="red", 
     cex.main=1.5, cex.lab=1.2, cex.axis=0.6, cex.sub=1.2)

# Podriamos re-categorizar entre intrusos y no intrusos, buenas o malas conexiones, 0 o 1
# De tal forma que tengamos una variable independiente binaria y sea mas facil llevar a cabo
# las predicciones.

# Ahora procedemos a crear nuestras variables dummy, debido a que los predictores 
# categóricos que tenemos contienen mas de 2 valores nominales y no ordenados. 
# Esto significa que no podemos utilizar la función factor, usaremos una librería 
# llamada dummies que hace esto por nosotros:
# dummy(data_10_percent_train$protocol_type, sep = "_")

variables_categoricas = c('protocol_type', 'service', 'flag')
#data_10_percent_train <- cbind(data_10_percent_train, dummy(data_10_percent_train$service, sep = "_service_"))

data_10_percent_train_dummies <- dummy.data.frame( data_10_percent_train, sep = "_", all=FALSE )
protocol_type_dummy = get.dummy( data_10_percent_train_dummies, 'protocol_type' )
service_dummy = get.dummy( data_10_percent_train_dummies, 'service' )
flag_dummy = get.dummy( data_10_percent_train_dummies, 'flag' )
data_10_percent_train <- cbind(data_10_percent_train, protocol_type_dummy, service_dummy, flag_dummy)
dim(data_10_percent_train) # Verificamos las nuevas dimensiones de nuestros datos
names(data_10_percent_train)

data_10_percent_test_dummies <- dummy.data.frame( data_10_percent_test, sep = "_", all=FALSE )
protocol_type_dummy = get.dummy( data_10_percent_test_dummies, 'protocol_type' )
service_dummy = get.dummy( data_10_percent_test_dummies, 'service' )
flag_dummy = get.dummy( data_10_percent_test_dummies, 'flag' )
data_10_percent_test <- cbind(data_10_percent_test, protocol_type_dummy, service_dummy, flag_dummy)
dim(data_10_percent_test) # Verificamos las nuevas dimensiones de nuestros datos
#names(data_10_percent_train)

data_10_percent_testsmall_dummies <- dummy.data.frame( data_10_percent_test_small, sep = "_", all=FALSE )
protocol_type_dummy = get.dummy( data_10_percent_testsmall_dummies, 'protocol_type' )
service_dummy = get.dummy( data_10_percent_testsmall_dummies, 'service' )
flag_dummy = get.dummy( data_10_percent_testsmall_dummies, 'flag' )
data_10_percent_test_small <- cbind(data_10_percent_test_small, protocol_type_dummy, service_dummy, flag_dummy)
dim(data_10_percent_test_small)

# de igual forma el paquete "mlr" inluye la funcion createDummyFeatures para este proposito:
# data_10_percent_train_dummies <- createDummyFeatures(data_10_percent_train, cols = 'protocol_type')

# Ahora eliminamos las variables categóricas para que solo quede una matríz de números a análizar
data_10_percent_train <- data_10_percent_train[, !(names(data_10_percent_train) %in% variables_categoricas)]
data_10_percent_test <- data_10_percent_test[, !(names(data_10_percent_test) %in% variables_categoricas)]
data_10_percent_test_small <- data_10_percent_test_small[, !(names(data_10_percent_test_small) %in% variables_categoricas)]
# dim(data_10_percent_train)
# dim(data_10_percent_test)
# dim(data_10_percent_test_small)

# Separamos los datos de la variable independiente para luego utilizarlos como textos
# Y luego la convertimos a números para computarla en los siguientes análisis
y_train <- data_10_percent_train$attack_category
data_10_percent_train$attack_type <- NULL
data_10_percent_train$attack_category <- as.factor(data_10_percent_train$attack_category)
dim(data_10_percent_train)
y_test <- data_10_percent_test$attack_category
data_10_percent_test$attack_type <- NULL
data_10_percent_test$attack_category <- as.factor(data_10_percent_test$attack_category)
y_test_small <- data_10_percent_test_small$attack_category
data_10_percent_test_small$attack_type <- NULL
data_10_percent_test_small$attack_category <- as.factor(data_10_percent_test_small$attack_category)
# 
# # Comprobamos que todas las variables son númericas
str(data_10_percent_train)

# Liberamos un poco de memoria, removiendo objetos que no necesitamos
rm(data_10_percent_train_dummies, data_10_percent_test_dummies, data_10_percent_testsmall_dummies)
rm(flag_dummy, protocol_type_dummy, service_dummy)
rm(tipos_ataque, tipos_ataque_test, tipos_ataque_train, ataques_ctg, ataques_tipo)

# ---------------------------------------------------------
# 6. Seleccionar los parámetros más importantes
# ---------------------------------------------------------

# Para seleccionar las mejores variables podemos hacerlo manual revisando una matríz 
# de correlaciones entre las variables y bajo el conocimiento de la lógica del negocio.

data_10_percent_train_scaled <- data_10_percent_train
data_10_percent_train_scaled$attack_category <- NULL
data_10_percent_train_scaled = scale(data_10_percent_train_scaled)
corr_train <- cor(data_10_percent_train_scaled)
corr_train

# Gráficamos la matríz
# https://cran.r-project.org/web/packages/corrplot/vignettes/corrplot-intro.html
# corrplot(corr_train, method="number")
melted_cormat <- melt(corr_train)
# ggplot(melted_cormat, aes(X1, X2, fill = value)) + geom_tile() + 
#   scale_fill_gradient(low = "white",  high = "red")

# http://www.sthda.com/english/wiki/ggplot2-quick-correlation-matrix-heatmap-r-software-and-data-visualization
# http://www.sthda.com/english/wiki/correlation-matrix-a-quick-start-guide-to-analyze-format-and-visualize-a-correlation-matrix-using-r-software
# http://www.sthda.com/english/wiki/visualize-correlation-matrix-using-correlogram
ggplot(data = melted_cormat, aes(X1, X2, fill = value))+ geom_tile(color = "white") + 
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0, 
                       limit = c(-1,1), space = "Lab",  name="Correlación de\nPearson") + 
  theme_minimal() + theme(axis.text.x = element_text(angle = 45, vjust = 1, size = 12, hjust = 1)) + 
  coord_fixed()

# Otra forma de ver las correlaciones junto con los p-values
corr_train_2 <- rcorr(as.matrix(data_10_percent_train_scaled))
corr_train_2
# Extraer valores
corr_train_2$r # Extraer los coeficientes de correlación
corr_train_2$P # Extraer p-values

# ++++++++++++++++++++++++++++
# flattenCorrMatrix
# ++++++++++++++++++++++++++++
# cormat : matrix of the correlation coefficients
# pmat : matrix of the correlation p-values
flattenCorrMatrix <- function(cormat, pmat) {
  ut <- upper.tri(cormat)
  data.frame(
    row = rownames(cormat)[row(cormat)[ut]],
    column = rownames(cormat)[col(cormat)[ut]],
    cor  =(cormat)[ut],
    p = pmat[ut]
  )
}

# Visualizar la matríz de otra forma
flattenCorrMatrix(corr_train_2$r, corr_train_2$P)

# Liberamos memoria
rm(data_10_percent_train_scaled, corr_train, melted_cormat, corr_train_2)

# Ahora lo hacemos de forma automática, este metodo es para obtener la 
# mejor selección de predictores por lo que tomará algo de tiempo
# Este procedimiento toma demasiado tiempo por el numero de registros y variables 
# por lo que esta comentado
# regfit=regsubsets(data_10_percent_train$attack_category~.,data_10_percent_train, 
#                   really.big=T, nvmax=20)
# summary(regfit)
# reg.summary=summary(regfit)
# names(reg.summary)
# reg.summary$rsq

# Gráficamos para visualmente ver la importancia de las variables
# par(mfrow=c(2,2))
# plot(reg.summary$rss,xlab="Número de Variables",ylab="RSS",type="l")
# plot(reg.summary$adjr2,xlab="Número de Variables",ylab="R^2 Ajustado",type="l")
# which.max(reg.summary$adjr2)
# points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20)
# plot(reg.summary$cp,xlab="Número de Variables",ylab="Cp",type='l')
# which.min(reg.summary$cp)
# points(10,reg.summary$cp[10],col="red",cex=2,pch=20)
# which.min(reg.summary$bic)
# plot(reg.summary$bic,xlab="Número de Variables",ylab="BIC",type='l')
# points(6,reg.summary$bic[6],col="red",cex=2,pch=20)
# plot(regfit.full,scale="r2")
# plot(regfit.full,scale="adjr2")
# plot(regfit.full,scale="Cp")
# plot(regfit.full,scale="bic")
# coef(regfit.full,6)


# Como los procesos o metodos anteriores toman demasiado tiempo tomamos las 40 variables
# más importantes obtenidas del ejercicio ya resulto con python localizado en
# https://github.com/egiron/DataScience/blob/master/EjerciciosTmp/Deteccion%20intrusos%20redes.ipynb
predictores = c('same_srv_rate', 'flag_SF', 'dst_host_same_srv_rate', 'service_private',
                 'dst_host_srv_serror_rate', 'service_http', 'logged_in',
                 'dst_host_srv_count', 'count', 'srv_serror_rate', 'flag_S0',
                 'dst_host_serror_rate', 'dst_host_count', 'rerror_rate', 'serror_rate',
                 'dst_host_rerror_rate', 'src_bytes', 'srv_rerror_rate',
                 'dst_host_same_src_port_rate', 'dst_host_srv_rerror_rate',
                 'protocol_type_udp', 'service_ecr_i', 'flag_REJ', 'service_pop_3',
                 'protocol_type_tcp', 'diff_srv_rate', 'hot', 'dst_host_diff_srv_rate',
                 'service_telnet', 'service_domain_u', 'wrong_fragment',
                 'dst_host_srv_diff_host_rate', 'num_compromised', 'service_smtp',
                 'srv_count', 'dst_bytes', 'srv_diff_host_rate', 'service_ftp_data',
                 'duration', 'service_ftp', 'attack_category')

# ---------------------------------------------------------
# 7. Separando el conjunto de datos de entrenamiento y de validación
# ---------------------------------------------------------

X <- data_10_percent_train[,predictores]

# En este paso deberiamos de balancear la muestra o datos de entrenamiento 
# para obtener mejores resultados en nuestros clasificadores del paso sgte.
# Pero continuaremos adelante con el ejercicio y luego retomaremos este paso utilizando
# el conjunto de datos completo y no solo el del 10% de entrenamiento.

# set.seed(12345)
# data_10_percent_train_dos <- data_10_percent_train[data_10_percent_train$attack_type == 'dos',] #Sacamos solo los ejemplos tipo dos
# data_10_percent_train_normal <- data_10_percent_train[data_10_percent_train$attack_type == 'normal',] #Sacamos solo los ejemplos tipo normal.
# data_10_percent_train_probe <- data_10_percent_train[data_10_percent_train$attack_type == 'probe',] #Sacamos solo los ejemplos tipo probe
# data_10_percent_train_r2l <- data_10_percent_train[data_10_percent_train$attack_type == 'r2l',] #Sacamos solo los ejemplos tipo r2l
# data_10_percent_train_u2r <- data_10_percent_train[data_10_percent_train$attack_type == 'u2r',] #Sacamos solo los ejemplos tipo u2r
# 
# data_10_percent_train_dos <- data_10_percent_train_dos[sample(nrow(dos), 100000), ]# Tomamos 100.000 ejemplos aleatoriamente sin reemplazo. Podemos quitar muchos mas si queremos!!!!
# data_10_percent_train_normal <- data_10_percent_train_normal[sample(nrow(normal), 10000),]# Tomamos 10.000 ejemplos aleatoriamente sin reemplazo
# 
# # Ahora unimos las partes en un solo dataframe
# data_10_percent_train_balanced <- rbind(data_10_percent_train_dos, data_10_percent_train_normal,
#                  data_10_percent_train_probe, data_10_percent_train_r2l, 
#                  data_10_percent_train_u2r)


set.seed(1234)
ind <- sample(2, nrow(X), replace=TRUE, prob=c(0.7, 0.3))
Xr <- X[ind==1,]
yr <- Xr$attack_category
Xt <- X[ind==2,]
yt <- Xt$attack_category
rm(ind) # Liberamos memoria

# ---------------------------------------------------------
# 8. Selección de algoritmos y métodos
# ---------------------------------------------------------

# Iniciamos con un Árbol de decisión simple
tree.intrusos=tree(Xr$attack_category~.,Xr)
summary(tree.intrusos)
plot(tree.intrusos, cex=.75)
text(tree.intrusos,pretty=0)
tree.intrusos

# Otra forma de crearlos utilizando la librería party
tree.intrusos2 <- ctree(Xr$attack_category~. , data=Xr)
# Visualizar la matríz de confusión sobre los datos de validación
yt_pred2 = predict(tree.intrusos2, Xt, type="response") 
table(yt_pred2, Xt$attack_category) # yt
# Sin poda
plot(tree.intrusos2, type="simple")
text(tree.intrusos2)

# Transformemos las salidas de probabilidad a salidas categoricas
# maxidx <- function(arr) {
#   return(which(arr == max(arr)))
# }
# idx <- apply(yt_pred2, c(1), maxidx)
# prediction <- c('dos', 'normal', 'probe', 'r2l', 'u2r', 'unknown')[idx]
# table(prediction, Xt$attack_category)

# Como se observa anteriormente utilizamos un árbol de regresion por lo que la variable independiente era numerica
# Ahora procedemos a crear un Árbol de clasificación

X$attack_category <- y_train
#str(X)
set.seed(1234)
ind <- sample(2, nrow(X), replace=TRUE, prob=c(0.7, 0.3))
Xr <- X[ind==1,]
yr <- Xr$attack_category
Xt <- X[ind==2,]
yt <- Xt$attack_category
rm(ind)

# Ajustamos o creamos el árbol con la librería rpart
fit <- rpart(Xr$attack_category~. , data=Xr, method="class") # method="anova" para regresión
printcp(fit) # visualizar los resultados
plotcp(fit) # visualizar los resultados de la validación cruzada
summary(fit) # resumen detallado de los nodos

# Gráficar el árbol
plot(fit, uniform=TRUE, main="Arbol de Clasificacion de \nTipos de Ataques en Redes")
text(fit, use.n=TRUE, all=TRUE, cex=.8)
# Creamos un archivo postscript con el gráfico
# post(fit, file = "data/tree.ps", title = "Classification Tree for Attack types")

# Otra forma de visualizarlo
rparty.tree <- as.party(fit)
rparty.tree
plot(rparty.tree)

# Validación con los datos de prueba con duplicados
# Esta validación es con el fin de poder comparar con la matríz de confusión del ganador
X_test <- data_10_percent_test[,predictores]
table(predict(tree.intrusos2, X_test), X_test$attack_category)
# Precisión
# ((203047 + 60378 + 2033 + 188 + 18) / 311029) * 100
((201892 + 60377 + 2033 + 188 + 18) / 311029) * 100

# table(predict(rparty.tree, X_test), X_test$attack_category)
# ((59011 + 60355 + 1655 + 0 + 0) / 311029) * 100

# Como se observa la precisión fue de 85.% comparada con la del ganador de 92.7%

# predicted     0      1      2      3      4     %correct
# actual       --------------------------------------------------
#   0         |   60262    243     78      4      6       99.5%
#   1         |     511   3471    184      0      0       83.3%
#   2         |    5299   1328 223226      0      0       97.1%
#   3         |     168     20      0     30     10       13.2%
#   4         |   14527    294      0      8   1360        8.4%
#   

# Ahora probamos con otros algoritmos y metodos como random forest, bagging and boosting, KNN, etc.

# Random Forest
# system.time(Mod1 <- train(Xr$attack_category ~ ., method = "rf",
#                data = Xr, importance = T, verbose=T, keep.forest=TRUE,
#                trControl = trainControl(method = "cv", number = 3)))
# save(Mod1,file="data/Mod1.RData")
# 
# load("data/Mod1.RData")
# # Mod1$finalModel
# vi <- varImp(Mod1)
# vi$importance[1:10,]
# 
# # out-of-sample errors of random forest model using validation dataset 
# pred1 <- predict(Mod1, X_test) # type="class"
# cm1 <- confusionMatrix(pred1, X_test$attack_category)
# cm1$table

# Aplicamos Random Forest
# En cualquiera de los casos crash RStudio, es necesario investigar el porque... parece ser por que son datos de texto o como factor la variable independiente
# Como alternativa se utiliza ranger
# Xr$attack_category<-yr
# ?randomForest
# str(Xr)
# ataques_rf <- randomForest(Xr[1:40], Xr$attack_category, ntree=5, mtry=4, do.trace=T, 
#                            proximity=TRUE, importance=T)

# ataques_rf <- randomForest(formula=Xr$attack_category~., Xr, ntree=10,
#                            proximity=TRUE, importance=T, do.trace=T, keep.forest=T)
# table(predict(ataques_rf, type="class"),Xr$attack_category)

# ----------
# Random Forest utilizando la librería ranger
# ----------
# ?ranger
ataques_rf <- ranger(formula=Xr$attack_category~.,data=Xr, num.trees=1000,  
                     splitrule="gini", verbose=TRUE ) # classification=TRUE, mtry = 4, min.node.size=1,

ataques_rf$confusion.matrix
# Validamos con el datset de prueba completo
pred_rg <- predict(ataques_rf,X_test, type="response")
# summary(pred_rg)
table(y_test, pred_rg$predictions)
# Precisión
(223279 + 60304 + 2369 + 952 + 0)/ length(y_test) * 100


# Aplicamos Random Forest con Bagging
# En este ejemplo si funciona por que hemos reemplazado la variable categárica
set.seed(1)
Xr$attack_category <- as.integer(as.factor(yr))
# str(Xr)
# system.time(bag.intrusos <- randomForest(Xr$attack_category~.,data=Xr,mtry=13,
#                                          importance=TRUE, ntree=100, do.trace=T))
# save(bag.intrusos,file="data/Mbag_intrusos.RData")
load("data/Mbag_intrusos.RData")
bag.intrusos
#Xt$attack_category <- NULL
predictions <- predict(bag.intrusos, Xt, type = "response")
# mean((predictions-Xt)^2)
importance(bag.intrusos) # Importancia de los predictores
par(mfrow=c(1,1))
varImpPlot(bag.intrusos)

#table(yt, as.integer(predictions))

yt_pred <- predict(bag.intrusos, X_test, type = "response")
# mean((yt_pred - as.integer(X_test$attack_category))^2)
# table(y_test, as.integer(yt_pred)) # Es necesario revisar el orden
# # Precisión
# (56146 + 210 + 944 + 926 + 5) / length(y_test)


# Otra forma de hacerlo utilizando validación cruzada y la libreria "caret"
# Pero esto toma demasiado tiempo por lo que queda comentado
# Xr$attack_category <- factor(as.factor(yr)) #drop levels
# model_rf  <- train(Xr$attack_category~., tuneLength = 3, data = Xr, 
#                    method="rf", importance = TRUE,
#                    trControl = trainControl(method = "cv",
#                                             number = 5,
#                                             savePredictions = "final",
#                                             classProbs = T))
# # Predicciones
# model_rf$pred
# model_rf$pred[order(model_rf$pred$rowIndex),2] # Ordenar como en los datos originales
# # Matríz de confusion
# confusionMatrix(model_rf$pred[order(model_rf$pred$rowIndex),2], Xr$attack_category)

# Otra forma de hacerlo
# set.seed(415)
# Xr$attack_category <- as.factor(yr)
# fitRF <- cforest(Xr$attack_category ~ ., data = Xr,  ntree=100, mtry=5) # controls=cforest_unbiased
# predictionRF <- predict(fitRF, Xt, OOB=TRUE, type = "response")
# predictionRF

# --------------------------
# Usamos la libreria "e1071" para utilizar el método 'tune' que 
# permite obtener los mejores parametros o mejor modelo a utilizar
# Esta comentado por que toma mucho tiempo
# Xr$attack_category <- as.factor(yr)
# Xt$attack_category <- as.factor(yt)
# tuned.r <- tune(randomForest, train.x = Xr$attack_category ~ ., data = Xr, validation.x = Xt$attack_category)
# best.model <- tuned.r$best.model
# predictions <- predict(best.model, Xt)
# table.random.forest <- table(Xt$attack_category, predictions)
# table.random.forest
# # Computamos el error:
# error.rate <- 1 - sum(diag(as.matrix(table.random.forest))) / sum(table.random.forest)
# error.rate
# --------------------------


# Boosting
set.seed(12345)
Xr$attack_category <- yr # factor(yr) # as.character()
Xt$attack_category <- yt # factor(yt)
boost.ataques4 <- gbm(Xr$attack_category~.,data=Xr,distribution="multinomial", 
                      n.trees=50,interaction.depth=4, verbose=T)

summary(boost.ataques4)
boost.ataques4$num.classes
y_pred4=predict.gbm(boost.ataques4,Xt,n.trees=50, type = "response")

y_pred4[1:10,,]
p.y_pred4 <- apply(y_pred4, 1, which.max)
head(p.y_pred4)
p.yp4_class <- colnames(y_pred4)[p.y_pred4]
table(yt, p.yp4_class)
# Precisión sobre el Xt
# (15923 + 26275 + 224 + 87 + 0) / length(yt)

# Validamos con los datos de prueba completos
y_pred_4a=predict.gbm(boost.ataques4,X_test,n.trees=50, type = "response")
# prob_pred4a = as.matrix(y_pred_4a[,,1])
p.y_pred_4a <- apply(y_pred_4a, 1, which.max)
p.y_pred_4a_class <- colnames(y_pred_4a)[p.y_pred_4a]
table(y_test, p.y_pred_4a_class)
# Precisión
# (58996 + 60381 + 1574 + 35 + 0) / length(y_test)

# Otra forma
# boost.ataques6 <- gbm.fit(x=Xr[1:40], y=Xr$attack_category,distribution="multinomial", 
#                       n.trees=50,interaction.depth=4, verbose=T)
# 
# summary(boost.ataques6)
# # Validamos con los datos de prueba completos
# y_pred_6=predict.gbm(boost.ataques6,X_test[1:40],n.trees=50, type = "response")
# # prob_pred6 = as.matrix(y_pred_6[,,1])
# p.y_pred_6 <- apply(y_pred_6, 1, which.max)
# p.y_pred_6_class <- colnames(y_pred_6)[p.y_pred_6]
# table(y_test, p.y_pred_6_class)
# # Precisión
# (58994 + 60387 + 1571 + 35 + 0) / length(y_test)



# Otra forma de hacerlo usando ambos bagging and boosting pero toma demasiado tiempo procesar; para revisar luego...
# set.seed(123)
# fitControl = trainControl(method="cv", number=5) #returnResamp = "all"
# # method = "C5.0", metric = "kappa"
# Xr <- as.data.frame(Xr)
# model2 = train(factor(Xr$attack_category)~.,data=Xr, method="gbm",distribution="gaussian", 
#                trControl=fitControl, verbose=T, 
#                tuneGrid=data.frame(.n.trees=50, .shrinkage=0.01, 
#                                    .interaction.depth=1, .n.minobsinnode=1))
# 
# model2
# confusionMatrix(model2)


# ---------------------------------------------------------
# 10. Resumen de los métodos utilizados
# ---------------------------------------------------------

# De acuerdo a los modelos ejecutados anteriormente, y sin ser muy rigurosos o meticulosos
# en la elecciones de los mejores parametros o utilizando el conjunto de datos completos de 4 millones de registros
# Podemos concluir que el mejor podria ser para esta solución 1, el de random forest bagging and boosting
# utilizando la libreria ranger que fue la mas eficiente.
# Aqui va una tabla y/o gráfico con la comparación de los resultados de todos los modelos. 

# ---------------------------------------------------------
# 11. Comparación de resultados con el ganador del KDDCup
# ---------------------------------------------------------

# Matríz de confusión de nuestros mejores resultados:
#            dos normal  probe    r2l    u2r
# dos     223279     12      7      0      0
# normal      69  60304    216      2      2
# probe        3      5   2369      0      0
# r2l          0   5040      0    952      1
# u2r          0     34      0      5      0

# Precisión (223279 + 60304 + 2369 + 952 + 0)/ length(y_test) === 0.9224349


# Matríz de confusión del ganador del concurso del KDD Cup 1999
# predicted     0      1      2      3      4     %correct
# actual       --------------------------------------------------
#   0         |   60262    243     78      4      6       99.5%
#   1         |     511   3471    184      0      0       83.3%
#   2         |    5299   1328 223226      0      0       97.1%
#   3         |     168     20      0     30     10       13.2%
#   4         |   14527    294      0      8   1360        8.4%
#   

# Precisión === 0.9270808
(60262 + 3471 + 223226 + 30 + 1360)/ length(y_test) * 100  


# Queda mucho por explorar y realizar para mejorar la precisión, pero esto es solo un ejemplo
# de como puede abordar un problema de este tipo en R.

# Si desea conocer más o la manera como se implementa en python visite:
# https://github.com/egiron/DataScience/blob/master/EjerciciosTmp/Solucion%201.ipynb

