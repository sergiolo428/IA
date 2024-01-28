function template_T1_E7
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cargar base de datos

load('Auto.mat')

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

% Apartado 1 - Remover valores perdidos

correctPos=~isnan(Auto.horsepower);
Auto2 = Auto(correctPos,:);

mpg2 = Auto2{:,1};
cylinders2 = Auto2{:,2};
displacement2 = Auto2{:,3};
horsepower2 = Auto2{:,4};
weight2 = Auto2{:,5};
acceleration2 = Auto2{:,6};
year2 = Auto2{:,7};
origin2 = Auto2{:,8};
name2 = Auto2{:,9};


% Apartado 2 - Identifica los predictores cuantitativos y los cualitativos
% fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(1);
sgtitle('Histograma variables')
subplot(2,4,1);
histogram(Auto.mpg);
title('MPG')

subplot(2,4,2);
histogram(Auto.cylinders);
title('Cylindres')

subplot(2,4,3);
histogram(Auto.displacement);
title('Displacement')

subplot(2,4,4);
histogram(Auto.horsepower);
title('Horsepower')

subplot(2,4,5);
histogram(Auto.weight);
title('Weight')

subplot(2,4,6);
histogram(Auto.acceleration);
title('Acceleration')

subplot(2,4,7);
histogram(Auto.year);
title('Year')

subplot(2,4,8);
histogram(Auto.origin);
title('Origin')



% Apartado 3 - Calcular la media, desviación estándar y rango de cada uno 
% de los predictores cuantitativos
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

media=mean(Auto2(:,1:6));

desviacion = std(Auto2(:,1:6));

maximo = max(Auto2(:,1:6));
minimo = min(Auto2(:,1:6));

rangoTabla = vertcat(maximo,minimo)
rango = maximo - minimo;

% Apartado 4 - Eliminar las observaciones en el rango $10-85$. 
% ¿Cuál es ahora el rango, media y desviación estándar de cada predictor?
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');



% Apartado 5 -  Usando toda la base de datos, analiza los predictores de 
% manera gráfica haciendo uso de la función \textit{scatter}. 
% Crea gráficos que resalten la relación entre predictores. 
% Resume los resultados obtenidos.
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(2)
sgtitle('Respecto al Nº de cilindros');

subplot(2,2,1)
scatter(Auto2{:,2},Auto2{:,1})
title('MPG')

subplot(2,2,2)
scatter(Auto2{:,2},Auto2{:,3})
title('Displacement')

subplot(2,2,3)
scatter(Auto2{:,2},Auto2{:,4})
title('Horsepower')

subplot(2,2,4)
scatter(Auto2{:,2},Auto2{:,5})
title('Weight')

% Apartado 5 -   Suponer que queremos predecir la autonom´ıa del coche dada 
% en millas por galón (mpg) en base a otros predicotres. ¿Alguno de los 
% gráficos obtenidos previamente sugieren que otras variables puedan ser de
% utilidad a la hora de predecir mpg?
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(3)
sgtitle('Respecto a MPG');

subplot(2,2,1)
scatter(Auto2{:,1},cylinders2)
title('Cylinders')

subplot(2,2,2)
scatter(Auto2{:,1},displacement2)
title('Displacement')

subplot(2,2,3)
scatter(Auto2{:,1},horsepower2)
title('Horsepower')

subplot(2,2,4)
scatter(Auto2{:,1},weight2)
title('Weight')

figure(4)
sgtitle('MPG respecto 2 variables')

subplot(1,3,1)
scatter3(horsepower2,displacement2,mpg2)
xlabel('Horsepower');ylabel('Displacement');zlabel('MPG');
title('Horsepower y Displacement')

subplot(1,3,2)
scatter3(horsepower2,weight2,mpg2)
xlabel('Horsepower');ylabel('Weight');zlabel('MPG');
title('Horsepower y Weight')

subplot(1,3,3)
scatter3(displacement2,weight2,mpg2)
xlabel('Displacement');ylabel('Weight');zlabel('MPG');
title('Displacement y Weight')

