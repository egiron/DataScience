# DataScience
Notas personales acerca de statistical Learning, machine learning y varios temas relacionados con la Ciencia de los datos.

## Aprendizaje Estadístico y Automático

A continuación encontrarán algunas guías sobre el aprendizaje estadístico y técnicas de aprendizaje automatico o machine learning que he abordado sobre varios cursos tanto presenciales como en línea.

Empezaremos desde los temas mas sencillos hasta ir abordando los mas dificiles, en lo posible tanto teórico como práctico.

Recuerde que estas notas son personales y no constituyen un tutorial o curso en línea de estas disciplinas . Dicho lo anterior recomiendo en todos los casos comprar los libros citados y tomar cursos adicionales para entender mejor los conceptos fundamentales detras de todo esto.

Ahora si manos a la obra.

### Análisis Inteligente de Datos

1. [**Regresión lineal**](Regresion%20Lineal.ipynb#Regresio%CC%81n-Lineal)
    - [Regresión lineal simple](Regresion%20Lineal.ipynb#Regresi%C3%B3n-Lineal-Simple)
        * [Estimando "aprendiendo" los Coeficientes](Regresion%20Lineal.ipynb#Estimando-%22aprendiendo%22-los-Coeficientes)
        * [Evaluación de la exactitud de las estimaciones de coeficientes](Regresion%20Lineal.ipynb#Evaluacio%CC%81n-de-la-exactitud-de-las-estimaciones-de-coeficientes)
        * [Evaluación de la exactitud del modelo](Regresion%20Lineal.ipynb#Evaluacio%CC%81n-de-la-exactitud-del-modelo) 
            - [Error Estándar Residual (RSE)](Regresion Lineal.ipynb#Error-Estándar-Residual---RSE)
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
    - [Algunos gráficos de Ayuda](Regresion Lineal Multiple.ipynb#Gráficos-similares-a-la-salida-de-R)
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
    - [Maximal Margin Classifier]()
    - [Support Vector Classifier]()
    - [Kernels and Support Vector Machines]()
    
## Próximos temas...
A continuación se presentan algunos de temas importantes ó secciones que se irán agregando poco a poco.
Estos temas corresponden a las clases magistrales y el libro guía de ISLR mencionado al comienzo. Stay tune!!!

**Chapter 5: Resampling Methods**

    Estimating Prediction Error and Validation Set Approach
    K-fold Cross-Validation
    Cross-Validation: The Right and Wrong Ways
    The Bootstrap
    More on the Bootstrap
    Lab: Cross-Validation
    Lab: The Bootstrap

**Chapter 6: Linear Model Selection and Regularization**

    Linear Model Selection and Best Subset Selection
    Forward Stepwise Selection
    Backward Stepwise Selection
    Estimating Test Error Using Mallow’s Cp, AIC, BIC, Adjusted R-squared
    Estimating Test Error Using Cross-Validation
    Shrinkage Methods and Ridge Regression
    The Lasso
    Tuning Parameter Selection for Ridge Regression and Lasso
    Dimension Reduction
    Principal Components Regression and Partial Least Squares
    Lab: Best Subset Selection
    Lab: Forward Stepwise Selection and Model Selection Using Validation Set
    Lab: Model Selection Using Cross-Validation
    Lab: Ridge Regression and Lasso

**Chapter 7: Moving Beyond Linearity (slides, playlist)**

    Polynomial Regression and Step Functions
    Piecewise Polynomials and Splines
    Smoothing Splines
    Local Regression and Generalized Additive Models
    Lab: Polynomials
    Lab: Splines and Generalized Additive Models
