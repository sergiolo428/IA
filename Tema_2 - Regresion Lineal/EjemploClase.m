function EjemploClase

% Cargamos Base de datos

load('Boston.mat');

% lstat --> Predictor, proporcion de la populaciond e bajo status

% medv valor mediano de las viviendas ocupadas en miles de dolares


var_names = Boston.Properties.VariableNames;

% Predicotr
X = Boston.lstat;

% Respuestas reales
Y = Boston.medv;

%% Ajustamos modelo

% Recta ajustada
% Le pasamos el nonbre de variables para evitar confusion % No funciona,
% mirar

mdl = fitlm(X,Y,"VarNames",var_names([13 14])); 

% Tabla entera
mdl.Coefficients;
mdl.Coefficients.Estimate;

%
mdl.RMSE;

% Estadistico R^2 
mdl.Rsquared.Ordinary;


%% Calculo de intervalo de confianza al 95% para coeficientes B0 B1
% (El 0.05 es por: --> 1 - intervalo de confianza)

ci = coefCI(mdl, 0.05);

ci_a_mano = [mdl.Coefficients.Estimate-1.965*mdl.Coefficients.SE mdl.Coefficients.Estimate+1.965*mdl.Coefficients.SE];



%% Calculo de intervalo de confianza para mis predicciones

% Siendo ypred , las respuestas estimadas,
% siendo yci, loq ue nos indican los intervalos de confianza
% EJ yci para un predictor de valor de, obtenedremos un valor entre 29.000 y 30.000

[ypred,yci] = predict(mdl, [5 10 15]');


%% Ploteamos datos y modelos de regresion

figure(1)
Ypred = predict(mdl,X);

% Vemos los valores de training
scatter(X,Y); xlabel('lstat');ylabel('medv');

% Que recta hemos obtenido
hold on;
plot(X,Ypred);
hold off;

%% Analisis de residuos tanto por x como y
% Valores atipicos de y
figure(2)
subplot(121)
plotResiduals(mdl,'fitted', Marker='o')
subplot(122)
plotResiduals(mdl,"fitted",'ResidualType','studentized', Marker='o')

%% Analisis high leverage points, Valores atipicos de predictores x

% Todos los puntos por enicma de la linea puntuadda negra, los
% consideraremos como atipicos de x, por loq ue se podrian eliminar
figure(4)
plotDiagnostics(mdl,'leverage','Marker','o');xlabel('Indice fila');ylabel('leverage')


%% Analisis soimulataneo liverage y residuos estudentizados, para identificar outlayers y highliberage points a la vez
figure(3)
plot(mdl.Diagnostics.Leverage,mdl.Residuals.Studentized,'ko');ylabel('Residuos estudentizados');xlabel('Leverage');

ylim = get(gca,'Ylim')
hold on;
p=1;
plot([2*(p+1)/length(X) 2*(p+1)/length(X)],[ylim(1) ylim(2)]);
hold off;






end