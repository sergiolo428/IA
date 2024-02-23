function Template_T2_E8
% Este script contiene la resolución del ejercicio aplicado 8 del Tema 2
% de la asignatura 'Técnicas de Inteligencia Artificial'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Tema 2 parte 2

% Cargar base de datos

load('Carseats.mat');

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% EJERCICIO 8 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Apartado 1 - Ajustar un modelo de regresion lineal multiple para predecir 
% Sales en funcion de Price, Urban y US. 

% Por medio de notación Wilkinson

mdl_01=fitlm(Carseats,'Sales~Price+Urban+US','CategoricalVars',[7 10 11]);

% R^2 = 0.239 ; RSE = 2.47

% Apartado 2 - Interpreta cada coeficiente de regresión del modelo. 
% Ten en cuenta que algunas variables incorporadas en el modelo son cualitativas.

% Respuesta:
% Predictor Price ->  Obtenemos un valor de -0.05445, puede parecer un
% palor demasiado cercano a cero, pero si revisamos el error estandar,
% vemos que 0 no esta incluido en el intervalo de confianza. Ademas para
% asegurarnos, el p-valor es muy pequeño por lo que no garantiza la alta
% significancia del predictor

% Predictor Urban_No -> En este predictor cualitativo nos da una estimacion
% de 0.021916 el cual tambien esta bastantante cerca de 0, pero en este
% caso vemos que el error standard es bastante significativo, finalmente al
% comprobar el p-valor llegamos a la conlcusion que no es un predictor
% relevante en este modelo

% Predictor US_No -> Por ultimo, US si es un predictor significativo ya que
% vemos que tiene un p-valor decentemente pequeño. La estimacion nos indica
% que seha estimado que las observaciones con valor No en la variable US
% tienen un -1.2006 de ventas, para comprobar el concepto se ha incluido
% un ejemplo en el que unicamente modificamos el valor de US:


ej = Carseats(1,:);
ej.US{1,1} = 'Yes';
a=predict(mdl_01,ej);
ej.US{1,1} = 'No';
b=predict(mdl_01,ej);
a-b

% Apartado 3 - ¿Para qué predictores se puede rechazar la hipótesis 
% nula H0 : Bj = 0?

% Respuesta: Como se ha mencionado anteriormente, podemos ver que los
% predictores con influencia significativa en el modelo seran Price y US


% Apartado 4 - En base a la respuesta del apartado anterior, ajusta un modelo 
% con menos predictores que únicamente use predictores para los cuales existe 
% evidencia de asociacion con la respuesta.

mdl_02=fitlm(Carseats,'Sales~Price+US','CategoricalVars',[7 10 11]);

% Apartado 5 - ¿Cómo de bien se ajustan los modelos de los apartados 1 y 4?

% Respuesta: Parece ser que no hay grandes diferencias, pero si podemos
% destacar la alteraciondel p-valor al retirar una variable no
% significativa.
% 

% Apartado 6 - Usando el modelo del apartado 4, obtener los intervalos de 
% confianza del 95% para los coeficientes de regresión.

ci = coefCI(mdl_02,0.05)

% Apartado 7 - ¿Existe presencia de observaciones con valores attıpicos 
% o con influencia (leverage) inusualmente alta?

% Analizamos residuos --> Y

figure(1)
subplot(121);plotResiduals(mdl_02,'fitted','Marker','O');
subplot(122);plotResiduals(mdl_02,'fitted','ResidualType','studentized','Marker','O');

%Analizamos high leverage points --> X

figure(2)
plotDiagnostics(mdl_02,'leverage','Marker','o');xlabel('Indice fila');ylabel('leverage')

figure(3)
plot(mdl_02.Diagnostics.Leverage,mdl_02.Residuals.Studentized,'ko');ylabel('Residuos estudentizados');xlabel('Leverage');

% Conclusion, Haciendo el studentized vemos que no hay ningun valor atipico
% en Y, hay alguno que se acerca un poco al margen del valor 3, pero
% ninguno lo sobrepasa

% En cambio, alvisualizar los leveragee points, se puede apreciar que hay
% bastantes que sobresalen del umbral, sobre todo la observacion 43 de la
% cual sabemos:

% Price 24 | US No

% Viendo el resto de valores de la tabla podemos apreciar que 24 es el
% valor minimo que encontramos y es un valor que se aleja considerablemente
% del resto, por lo que este es el motivo por el que nos aparece al ahcer
% el analisis de residuos.

end