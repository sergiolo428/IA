function Template_T2_E6
% Este script contiene la resolución del ejercicio aplicado 6 del Tema 2
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% Cargar base de datos

load("Auto.mat");

disp('%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

% Apartado 1 - Ajustar un modelo de regresión lineal simple que tenga mpg como respuesta y
% horsepower como predictor

% Remover valores NaN

Auto2 = Auto;
Auto2(isnan(Auto2.horsepower),:)=[];

var_names = Auto.Properties.VariableNames;

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

X = Auto2.horsepower;
Y = Auto2.mpg;

mdl = fitlm(X,Y,"VarNames",var_names([4 1]));

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Predicción e intervalos de confianza del 95%

% Apartado 2 - Visualiza gráficamente predictor vs respuesta, y la recta de regresión.

Ypred = predict(mdl,X);

figure(1)
scatter(X,Y); xlabel('Horsepower');ylabel('Mpg')
hold on;
plot(X,Ypred);
hold off;

% Responde a las siguientes preguntas:
% ---------- a) ¿Existe relación entre predictor y respuesta? ---------

% Si la hay, podemos confirmarlo rapidameente viendo que el modelo generado
% tiene un valor de beta1 diferente de 0

% ---- b) ¿Cómo de fuerte es la relación entre predictor y respuesta? --

mdl.Rsquared.Ordinary;

% Obtenemos un 0.605 , podemos decir que tienen una relacion decente

% - c) ¿Es la relación entre predictor y respuesta negativa o positiva? 

mdl.Coefficients.Estimate(2);

% Vemos que nos devuelbe un valor negativo


% d) ¿Cuál es el mpg estimado para un horsepower igual a 98? 
 
predict(mdl,98); % 24.4671

% ¿Cuáles son los intervalos de confianza y predicción del 95%?
% Predicción e intervalos de predicción del 95%

ci = coefCI(mdl,0.05) % Intervalos de confianza

%b0 38.5 - 41.3
%b1 -0.17 - -0.14

[ypred,yci] = predict(mdl, [98]','Alpha',0.05)

%ypred --> 24.46

%yci --> 23.973 - 24.9611

% Apartado 3 - Visualiza residuos y puntos de alta influencia (high leverage points) y comenta los
% posibles problemas que existan con el ajuste por mínimos cuadráticos.
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Ploteamos datos y modelo de regresión

figure(2)
scatter(X,Y); xlabel('Horsepower');ylabel('Mpg')
hold on;
plot(X,Ypred);
hold off;

% Analizamos residuos 

figure(3)
subplot(121)
plotResiduals(mdl,'fitted', Marker='o')
subplot(122)
plotResiduals(mdl,"fitted",'ResidualType','studentized', Marker='o')


% Analizamos high leverage points (predictores)

figure(4)
plotDiagnostics(mdl,'leverage','Marker','o');xlabel('Indice fila');ylabel('leverage')


% Analizamos leverage vs studentized residuals

figure(5)
plot(mdl.Diagnostics.Leverage,mdl.Residuals.Studentized,'ko');ylabel('Residuos estudentizados');xlabel('Leverage');

ylim = get(gca,'Ylim');
hold on;
p=1;
plot([2*(p+1)/length(X) 2*(p+1)/length(X)],[ylim(1) ylim(2)]);
hold off;

end