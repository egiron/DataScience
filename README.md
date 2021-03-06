# DataScience
Notas personales acerca de statistical Learning, machine learning y varios temas relacionados con la Ciencia de los datos.

## Aprendizaje Estadístico y Automático

A continuación encontrarán algunas guías sobre el aprendizaje estadístico y técnicas de aprendizaje automatico o machine learning que he abordado sobre varios cursos tanto presenciales como en línea.

Empezaremos desde los temas mas sencillos hasta ir abordando los mas dificiles, en lo posible tanto teórico como práctico.

Recuerde que estas notas son personales y no constituyen un tutorial o curso en línea de estas disciplinas . Dicho lo anterior recomiendo en todos los casos comprar los libros citados y tomar cursos adicionales para entender mejor los conceptos fundamentales detras de todo esto.

Ahora si manos a la obra.

## Aprendizaje Estadístico

Aunque el término aprendizaje estadístico (_Statistical Learning_) es relativamente nuevo, muchos de los conceptos que lo fundamentan se desarrollaron hace mucho tiempo.

> A principios del siglo XIX, **Legendre y Gauss** publicaron artículos sobre el método de los mínimos cuadrados, que implementaron la forma más antigua de lo que ahora se conoce como _regresión lineal_. El enfoque se aplicó primero 
con éxito a los problemas de la astronomía. La regresión lineal se utiliza hoy día en muchos campos de diferentes 
disciplinas.

Para llevar un orden en lo que vamos aprendiendo, seguiremos el texto "**An Introduction to Statistical Learning with Applications in R**", de _Gareth James, Daniela Witten, Trevor Hastie and Robert Tibshirani_. Este puede ser descargado de la [página del autor](http://www-bcf.usc.edu/~gareth/ISL/). 

![Libro ISLR](./assets/ISLCover.jpg)

This book provides an introduction to statistical learning methods. It is aimed for upper level undergraduate students, masters students and Ph.D. students in the non-mathematical sciences. The book also contains a number of R labs with detailed explanations on how to implement the various methods in real life settings, and should be a valuable resource for a practicing data scientist.

De igual modo utilizaremos parte del material del profesor **Elkin A. Castaño V.** que ha traducido al español gran parte del texto. Este material fue facilitado en el [Diplomado en Ciencia de Datos: Data Mining de la Universidad del Valle](http://escuelaestadistica.univalle.edu.co/diplomado-data-mining), Cali - Colombia.

El Data Mining ( Minería de Datos) se presenta en la actualidad como una nueva alternativa, que permite explorar grandes bases de datos, de manera automática o semiautomática, con el objetivo de encontrar patrones repetitivos, tendencias o reglas que expliquen el comportamiento de los datos en un determinado contexto, con el fin de que puedan usarse para predecir comportamientos futuros, transformando los datos en conocimiento proactivo, para la toma de decisiones de empresas públicas y privadas, científicos, universidades, entre otros.

### A quienes va dirigido estas notas

Este curso está dirigido a las personas que están interesadas en el empleo de métodos estadísticos para la modelación y la predicción a partir de datos. Este grupo de personas no solamente incluye a científicos, ingenieros, analistas de datos, o analistas cuantitativos, sino también a individuos menos técnicos en campos no-cuantitativos tales como las ciencias sociales o los negocios. Se espera que el participante haya tenido por lo menos un curso elemental en Estadística. El nivel matemático del curso es modesto y no se requiere un conocimiento detallado del álgebra matricial.

La comunidad de usuarios de las técnicas de aprendizaje estadístico ha venido creciendo y e incluye individuos con una gama más amplia de intereses y formaciones. Es importante que este grupo heterogéneo sea capaz de comprender los modelos, sus alcances y las fortalezas y debilidades de los diversos enfoques. Pero para este público, muchos de los detalles técnicos de los métodos de aprendizaje estadístico, como los algoritmos de optimización y las propiedades teóricas, no son de interés primordial. Estos estudiantes no necesitan una comprensión profunda de estos aspectos para convertirse en usuarios conocedores de las diversas metodologías, y para contribuir a sus campos de trabajo a través del uso de herramientas de aprendizaje estadístico.

Esta introducción al aprendizaje estadístico se basa en las siguientes cuatro premisas.

1. Muchos métodos de aprendizaje estadístico son relevantes y útiles en una amplia gama de disciplinas académicas y no académicas, más allá de la ciencia Estadística. 

2. El aprendizaje estadístico no debe ser visto como una serie de cajas negras.

3. Si bien es importante saber qué trabajo realiza cada engranaje, no es necesario tener las habilidades para construir la máquina dentro de la caja.

4. Suponemos que el lector está interesado en aplicar métodos de aprendizaje estadístico a problemas del mundo real. 

---
Si desea tener una introduccíon un poco mas detallada puede acceder a los videos y archivos de presentación que soportan el curso de Statistical Learning dictados por el Dr. Hastie y el Dr. Tibshirani discuss.

* [Presentación en PDF](https://lagunita.stanford.edu/c4x/HumanitiesScience/StatLearning/asset/introduction.pdf)
* [Opening Remarks and Examples](https://www.youtube.com/watch?v=2wLfFB_6SKI) (Video - duración 18:18)
[![](https://img.youtube.com/vi/2wLfFB_6SKI/0.jpg)](https://www.youtube.com/watch?v=2wLfFB_6SKI)
* [Supervised and Unsupervised Learning](https://www.youtube.com/watch?v=LvaTokhYnDw) (Video - duración 12:12)
[![](https://img.youtube.com/vi/LvaTokhYnDw/0.jpg)](https://www.youtube.com/watch?v=LvaTokhYnDw)



### Análisis Inteligente de Datos

1. [**Regresión lineal**](Regresion%20Lineal.ipynb#Regresio%CC%81n-Lineal)
    - [Regresión lineal simple](Regresion%20Lineal.ipynb#Regresi%C3%B3n-Lineal-Simple)
        * [Estimando "aprendiendo" los Coeficientes](Regresion%20Lineal.ipynb#Estimando-%22aprendiendo%22-los-Coeficientes)
        * [Evaluación de la exactitud de las estimaciones de coeficientes](Regresion%20Lineal.ipynb#Evaluacio%CC%81n-de-la-exactitud-de-las-estimaciones-de-coeficientes)
        * [Evaluación de la exactitud del modelo](Regresion%20Lineal.ipynb#Evaluacio%CC%81n-de-la-exactitud-del-modelo) 
            - [Error Estándar Residual (RSE)](Regresion%20Lineal.ipynb#Error-Estándar-Residual---RSE)
            - [Estadístico $R^2$](Regresion%20Lineal.ipynb#Estadi%CC%81stico-$R^2$)
        * [Confianza en nuestro modelo](Regresion%20Lineal.ipynb#Confianza-en-nuestro-modelo)
    - [Regresión lineal múltiple](Regresion%20Lineal%20Multiple.ipynb#Regresio%CC%81n-Lineal-Mu%CC%81ltiple)
        * [Estimación de los Coeficientes de Regresión](Regresion%20Lineal%20Multiple.ipynb#Estimacio%CC%81n-de-los-Coeficientes-de-Regresio%CC%81n)
        * [Estimación de los coeficientes utilizando scikit-learn](Regresion%20Lineal%20Multiple.ipynb#Estimaci%C3%B3n-de-los-coeficientes-utilizando-scikit-learn)
        * [Matriz de correlación](Regresion%20Lineal%20Multiple.ipynb#Matriz-de-correlaci%C3%B3n)
        * [Algunas Preguntas Importantes](Regresion%20Lineal%20Multiple.ipynb#Algunas-Preguntas-Importantes)
            - [¿Existe una relación entre la respuesta y los predictores?](Regresion%20Lineal%20Multiple.ipynb#%C2%BFExiste-una-relacio%CC%81n-entre-la-respuesta-y-los-predictores?)
            - [El Ajuste del Modelo](Regresion%20Lineal%20Multiple.ipynb#El-Ajuste-del-Modelo)
            - [Predicciones](Regresion%20Lineal%20Multiple.ipynb#Predicciones)
        * [Otras consideraciones en el modelo de regresión](Regresion%20Lineal%20Multiple.ipynb#Otras-consideraciones-en-el-modelo-de-regresio%CC%81n)
             - [Predictores cualitativos](Regresion%20Lineal%20Multiple.ipynb#Predictores-cualitativos)
             - [Predictores cualitativos con sólo dos niveles](Regresion%20Lineal%20Multiple.ipynb#Predictores-cualitativos-con-so%CC%81lo-dos-niveles)
             - [Creando variables dummy manualmente](Regresion%20Lineal%20Multiple.ipynb#Creando-variables-dummy-manualmente)
             - [Predictores cualitativos con más de dos niveles](Regresion%20Lineal%20Multiple.ipynb#Predictores-cualitativos-con-ma%CC%81s-de-dos-niveles)
    - [Extensiones del modelo lineal](Regresion%20Lineal%20Multiple.ipynb#Extensiones-del-modelo-lineal)
         * [Eliminación del supuesto de aditividad](Regresion%20Lineal%20Multiple.ipynb#Eliminacio%CC%81n-del-supuesto-de-aditividad)
         * [Interacción entre variables cualitativas y cuantitativas](Regresion%20Lineal%20Multiple.ipynb#Interacci%C3%B3n-entre-variables-cualitativas-y-cuantitativas)
         * [Relaciones no lineales](Regresion%20Lineal%20Multiple.ipynb#Relaciones-no-lineales)
    - [Problemas potenciales](Regresion%20Lineal%20Multiple.ipynb#Problemas-potenciales)
         - [No linealidad de las relaciones respuesta-predictor](Regresion%20Lineal%20Multiple.ipynb#1.-No-linealidad-de-los-datos)
         - [Correlación de los términos de error](Regresion%20Lineal%20Multiple.ipynb#2.-Correlacio%CC%81n-de-los-te%CC%81rminos-de-error)
         - [Variación no constante de los términos de error](Regresion%20Lineal%20Multiple.ipynb#3.-Variacio%CC%81n-no-constante-de-los-te%CC%81rminos-de-error)
         - [Valores atípicos](Regresion%20Lineal%20Multiple.ipynb#4.-Valores-ati%CC%81picos)
         - [Puntos de alta influncia o apalancamiento - Leverage](Regresion%20Lineal%20Multiple.ipynb#5.-Puntos-de-alta-influncia-o-apalancamiento---Leverage)
         - [Colinealidad](Regresion%20Lineal%20Multiple.ipynb#6.-Colinealidad)
             - [Fáctor de inflación de la varianza - VIF](Regresion%20Lineal%20Multiple.ipynb#F%C3%A1ctor-de-inflacio%CC%81n-de-la-varianza---VIF) 
    - [Algunos gráficos de Ayuda](Regresion%20Lineal%20Multiple.ipynb#Gráficos-similares-a-la-salida-de-R)
        * [Gráfico de Residuales](Regresion%20Lineal%20Multiple.ipynb#Gr%C3%A1fico-de-Residuales)
        * [Gráfico de Cuantiles - QQ](Regresion%20Lineal%20Multiple.ipynb#Gr%C3%A1fico-de-Cuantiles---QQ)
        * [Gráfico de Escala - Localización](Regresion%20Lineal%20Multiple.ipynb#Gr%C3%A1fico-de-Escala---Localizaci%C3%B3n)
        * [Gráfico de influencias o apalancamiento (Leverage)](Regresion%20Lineal%20Multiple.ipynb#Gr%C3%A1fico-de-influencias-o-apalancamiento---Leverage)
2. [**Clasificación**](Clasificacion.ipynb)
    - [Una visión general de la clasificación](Clasificacion.ipynb#Clasificaci%C3%B3n)
    - [¿Por qué no la regresión lineal?](Clasificacion.ipynb#%C2%BFPorque%CC%81-no-usar-Regresio%CC%81n-Lineal?)
    - [Regresión Logística](Clasificacion.ipynb#Regresio%CC%81n-Logi%CC%81stica)
        * [El Modelo Logístico](Clasificacion.ipynb#El-Modelo-Logi%CC%81stico)
        * [Estimación de los Coeficientes de Regresión](Clasificacion.ipynb#Estimacio%CC%81n-de-los-Coeficientes-de-Regresio%CC%81n)
        * [Predicciones](Clasificacion.ipynb#Predicciones)
        * [Regresión Logística Múltiple](Clasificacion.ipynb#Regresio%CC%81n-Logi%CC%81stica-Mu%CC%81ltiple)
    - [Análisis Discriminante Lineal](Clasificacion.ipynb#Ana%CC%81lisis-Discriminante-Lineal---LDA)
        * [Uso del teorema de Bayes para la clasificación](Clasificacion.ipynb#Uso-del-teorema-de-Bayes-para-la-clasificacio%CC%81n)
        * [Análisis Discriminante Lineal para p = 1](Clasificacion.ipynb#Ana%CC%81lisis-Discriminante-Lineal-para-p-=-1)
        * [Análisis Discriminante Lineal para p>1](Clasificacion.ipynb#Ana%CC%81lisis-Discriminante-Lineal-para-p%3E1)
        * [La curva ROC](Clasificacion.ipynb#La-curva-ROC)
    - [Análisis Discriminante Cuadrático](Clasificacion.ipynb#Ana%CC%81lisis-Discriminante-Cuadra%CC%81tico)
    - [Ejemplo con K-Nearest Neighbors](Clasificacion.ipynb#K-Nearest-Neighbors)
3. [**Aprendizaje no supervisado**](Aprendizaje%20no%20supervisado.ipynb#Aprendizaje-no-supervisado)
    - [El desafío del aprendizaje sin supervisión](Aprendizaje%20no%20supervisado.ipynb#El-desafi%CC%81o-del-aprendizaje-sin-supervisio%CC%81n)
    - [Análisis de componentes principales](Aprendizaje%20no%20supervisado.ipynb#Ana%CC%81lisis-de-componentes-principales)
        * [¿Qué son las componentes principales?](Aprendizaje%20no%20supervisado.ipynb#%C2%BFQue%CC%81-son-las-componentes-principales?)
        * [Otra Interpretación de las Componentes Principales](Aprendizaje%20no%20supervisado.ipynb#Otra-Interpretacio%CC%81n-de-las-Componentes-Principales)
        * [Más sobre el PCA](Aprendizaje%20no%20supervisado.ipynb#Ma%CC%81s-sobre-el-PCA)
            - [La escala de las variables](Aprendizaje%20no%20supervisado.ipynb#La-escala-de-las-variables)
            - [Unicidad de las Componentes Principales](Aprendizaje%20no%20supervisado.ipynb#Unicidad-de-las-Componentes-Principales)
            - [La proporción de la varianza explicada](Aprendizaje%20no%20supervisado.ipynb#La-proporcio%CC%81n-de-la-varianza-explicada)
            - [Cuántas componentes principales usar](Aprendizaje%20no%20supervisado.ipynb#Cua%CC%81ntas-componentes-principales-usar)
            - [Otros usos de las componentes principales](Aprendizaje%20no%20supervisado.ipynb#Otros-usos-de-las-componentes-principales)
    - [Métodos de agrupamiento (Clustering)](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Me%CC%81todos-de-agrupacio%CC%81n---Clustering)
        * [Agrupación de K-Means](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Agrupacio%CC%81n-de-K-Means)
        * [Agrupación jerárquica](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Agrupacio%CC%81n-jera%CC%81rquica)
            - [Interpretación de un dendrograma](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Interpretacio%CC%81n-de-un-dendrograma)
            - [El algoritmo de agrupación jerárquica](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#El-algoritmo-de-agrupacio%CC%81n-jera%CC%81rquica)
            - [Elección de la medida de disimilitud](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Eleccio%CC%81n-de-la-medida-de-disimilitud)
        * [Aspectos prácticos del agrupamiento](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Aspectos-pra%CC%81cticos-del-agrupamiento)
            - [Pequeñas decisiones con grandes consecuencias](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Pequen%CC%83as-decisiones-con-grandes-consecuencias)
            - [Validación de los clusters obtenidos](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Validacio%CC%81n-de-los-clusters-obtenidos)
            - [Otras consideraciones en la agrupación](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Otras-consideraciones-en-la-agrupacio%CC%81n)
            - [Un enfoque recomendado para interpretar los resultados del agrupamiento](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Un-enfoque-recomendado-para-interpretar-los-resultados-del-agrupamiento)
        * [Un ejemplo de agrupamiento con datos reales](Metodos%20de%20agrupacion%20o%20Clustering.ipynb#Un-Ejemplo-con-datos-reales)
4. [**Métodos basados en Árboles**](Metodos%20basados%20en%20arboles.ipynb)
    - [Arboles de regresión](Metodos%20basados%20en%20arboles.ipynb#Arboles-de-regresi%C3%B3n)
        * [Realizando predicciones sobre nuevos juegos de datos](Metodos%20basados%20en%20arboles.ipynb#Realizando-predicciones-sobre-nuevos-juegos-de-datos)
    - [Árboles de Clasificación](Metodos%20basados%20en%20arboles.ipynb#%C3%81rboles-de-Clasificaci%C3%B3n)
        * [Criterio de división en los árboles de clasificación](Metodos%20basados%20en%20arboles.ipynb#Criterio-de-divisi%C3%B3n-en-los-%C3%A1rboles-de-clasificaci%C3%B3n)
        * [Manipulando predictores categóricos](Metodos%20basados%20en%20arboles.ipynb#Manipulando-predictores-categ%C3%B3ricos)
        * [Ejemplo de Árbol de Clasificación](Metodos%20basados%20en%20arboles.ipynb#Ejemplo-de-%C3%81rbol-de-Clasificaci%C3%B3n)
        * [Otro ejemplo de Arboles de Clasificación con los datos de la base de datos IRIS](Metodos%20basados%20en%20arboles.ipynb#Otro-ejemplo-de-Arboles-de-Clasificaci%C3%B3n-con-los-datos-de-la-base-de-datos-IRIS)
        * [Ejemplo de Clasificación utilizando los datos de Aduanas](Metodos%20basados%20en%20arboles.ipynb#Ejemplo-de-Clasificaci%C3%B3n-utilizando-los-datos-de-Aduanas)
        * [Determinando donde o cuando podar](Metodos%20basados%20en%20arboles.ipynb#Determinando-donde-o-cuando-podar)
    - [Algunas consideraciones de los Árboles de Decisión](Metodos%20basados%20en%20arboles.ipynb#Algunas-consideraciones-de-los-%C3%81rboles-de-Decisi%C3%B3n)
    - [Ejemplo de Áboles de Decisión en R](Metodos%20basados%20en%20arboles.ipynb#Ejemplo-de-%C3%81boles-de-Decisi%C3%B3n-en-R)
    - [Ejemplo de Árboles de Decisión en R con libreria rpart y C50](Metodos%20basados%20en%20arboles.ipynb#Ejemplo-de-%C3%81rboles-de-Decisi%C3%B3n-en-R-con-libreria-rpart-y-C50)
    - [Random Forest y Bootstrap Aggregation (Bagging)](Random%20Forest.ipynb)
        * [Ejemplo con los datos del Titanic](Random%20Forest.ipynb#Ejemplo-con-los-datos-del-Titanic)
        * [Parametros para tener en cuenta y probar en el clasificador](Random%20Forest.ipynb#Parametros-para-tener-en-cuenta-y-probar-en-el-clasificador)
        * [Boosting](Random%20Forest.ipynb#Boosting)
        * [Bagging y Random Forests en R](Random%20Forest.ipynb#Bagging-y-Random-Forests-en-R)
5. [**Maquínas de Véctores Soporte - (SVM)**](SVM.ipynb)
    
6. [**Minería de texto**](TextMining.ipynb)
    - [Vectorización](TextMining.ipynb#Vectorizaci%C3%B3n)
    - [Bag of Words](TextMining.ipynb#Bag-of-Words)
    - [Procesamiento y análisis de textos](TextMining.ipynb#Procesamiento-y-an%C3%A1lisis-de-textos)
    - [¿Que son los NLP?](TextMining.ipynb#%C2%BFQue-son-los-NLP?)
    - [¿Por que utilizar NLP?](TextMining.ipynb#%C2%BFPor-que-utilizar-NLP?)
    - [Procesamiento utilizando NLTK](TextMining.ipynb#Procesamiento-utilizando-NLTK)
        * [Procesamiento de texto crudo](TextMining.ipynb#Procesamiento-de-texto-crudo)
        * [Tokenization](TextMining.ipynb#Tokenization)
        * [Otro ejemplo de vectorización o tokenization](TextMining.ipynb#Otro-ejemplo-de-vectorizaci%C3%B3n-o-tokenization)
        * [Proceso de textos crudos provenientes de internet o con formato HTML](TextMining.ipynb#Proceso-de-textos-crudos-provenientes-de-internet-o-con-formato-HTML)
        * [Utilizando expresiones regulares](TextMining.ipynb#Utilizando-expresiones-regulares)
        * [Normalización de textos](TextMining.ipynb#Normalizaci%C3%B3n-de-textos)
        * [Otro ejemplo de Stemming](TextMining.ipynb#Otro-ejemplo-de-Stemming)
        * [Etiquetado o Tagging](TextMining.ipynb#Etiquetado-o-Tagging)
        * [Clasificando Texto](TextMining.ipynb#Clasificando-Texto)
        * [Lemmatization](TextMining.ipynb#Lemmatization)
        * [Stopword Removal](TextMining.ipynb#Stopword-Removal)
        * [Named Entity Recognition](TextMining.ipynb#Named-Entity-Recognition)
        * [Term Frequency - Inverse Document Frequency (TF-IDF)](TextMining.ipynb#Term-Frequency---Inverse-Document-Frequency----TF-IDF)
        * [LDA - Latent Dirichlet Allocation](TextMining.ipynb#LDA---Latent-Dirichlet-Allocation)
        * [EXAMPLE Automatically summarize a document](TextMining.ipynb)
        * [Simplified Text Processing](TextMining.ipynb#Simplified-Text-Processing)
        * [Data Science Toolkit Sentiment](TextMining.ipynb#Data-Science-Toolkit-Sentiment)
    - [Uso de la librería Gensim](TextMining.ipynb#Uso-de-la-librer%C3%ADa-Gensim)
        * [Corpus Streaming – Un documento a la vez](TextMining.ipynb#Corpus-Streaming-%E2%80%93-Un-documento-a-la-vez)
    - [Ejemplo de Análisis de Sentimientos en Competencia Kaggle](TextMining.ipynb#Ejemplo-de-An%C3%A1lisis-de-Sentimientos-en-Competencia-Kaggle)
    - [Graficando Texto - WorldCloud](TextMining.ipynb#Gr%C3%A1ficando-Textos)
    
   
