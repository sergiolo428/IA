function regresionLinealMultiple
%% Tema 2 parte 2 

load('Boston.mat')
a=1;
%% Ajustar regresion lineal multiple para predecir medv en base s lstat y
% age

var_names = Boston.Properties.VariableNames;

X1 = Boston.lstat;
X2 = Boston.age;
Y = Boston.medv;

mdl = fitlm([X1 X2],Y,'VarNames',var_names([13 7 14]));

% 1. Miramos el p-valor, los dos son < que 0.05, por lo que se asocian a la
% respuesta

% 2. p valor del estadistico F, debeser menor que 0.05 para indicar que al
% menos una de las variables esta asociada a la respuesta, (recordatrio 
% que esto se ve en las lineas finales del mdl)



% Ajustar RLM para predecir medv en base a todos los predictores

X = Boston{:,1:end-1};
Y = Boston.medv;

mdl_2 = fitlm(X,Y,'VarNames',var_names);


% Ajustar RLM par apredecir medv en base a todos menos age

X = Boston{:,[1:6 8:size(Boston,2)-1]};
Y = Boston.medv;

mdl_3 = fitlm(X,Y,'VarNames',var_names([1:6 8:size(Boston,2)-1 14]));

% Ojo, no estamos suponiendo las relaciones entre predictores
% Para arreglarlo metemos un termino de interaccion
% (Ditividad)

% Ajustar un modelo de regresion lineal multiple para predecir medv en 
% base al lstat, age y el termino de interacion lstat x age

X1 = Boston.lstat;
X2 = Boston.age;
X3 = X1.*X2;


mdl_4 = fitlm([X1 X2 X3],Y,'VarNames',{var_names{[13 7]},'lstat x age',var_names{14}});

% Problema linealidad
% Ajusta un modelo de RLM para predecir medv en base a los predictores
% lstat y la transformacion no-lineal lstat x lstat

X1 = Boston.lstat;
X2 = X1.*X1;

mdl_5 = fitlm([X1 X2],Y,'VarNames',{var_names{13},'lstat x lstat',var_names{14}});

mdl_6 = fitlm(X1,Y,'VarNames',{var_names{[13 14]}});


%% Ploteamos datos y modelos de regresion
[~,pos]=sort(X1);

scatter(X1,Y);xlabel('lstat'),ylabel('medv');
ypred = predict(mdl_6,X1);hold on;plot(X1,ypred); %% Modelo con todas las observaciones

ypred2 = predict(mdl_5,[X1 X2]);plot(X1(pos),ypred2(pos));%% Modelo ajustado con el cuadrado de lstat

% Ojo, no saldra la curva correctamente ya que los valores no estan ordenados
% En el caso de la recta no s eapreciaba ya que todas las uniones pasaban a
% lo largo de dicha recta.

%% Valores atipicos de Y

subplot(121);plotResiduals(mdl_5,'fitted','Marker','O');
subplot(122);plotResiduals(mdl_5,'fitted','ResidualType','studentized','Marker','O');
end
