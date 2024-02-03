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

%% --------------- Apartado 1 - Remover valores perdidos --------------- %%

correctPos=~isnan(Auto.horsepower);
Auto2 = Auto(correctPos,:);

%% Apartado 2 - Identifica los predictores cuantitativos y los cualitativos
% fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(5)

for z=1:8

    subplot(2,4,z)
    histogram(Auto2{:,z})
    titulo = Auto2.Properties.VariableNames{z};
    titulo(1) = upper(titulo(1));
    title(titulo);

end

% Apartado 3 - Calcular la media, desviación estándar y rango de cada uno 
% de los predictores cuantitativos
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

media=mean(Auto2(:,1:6));

desviacion = std(Auto2(:,1:6));

maximo = max(Auto2(:,1:6));
minimo = min(Auto2(:,1:6));

rangoTabla = vertcat(maximo,minimo);
rango = maximo - minimo;

disp('-----Media-----');
media
disp('-----Desviacion-----');
desviacion
disp('-----Rango-----');
rango

% Apartado 4 - Eliminar las observaciones en el rango $10-85$. 
% ¿Cuál es ahora el rango, media y desviación estándar de cada predictor?
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

Auto3 = Auto2;
Auto3(10:85,:)=[];

media=mean(Auto3(:,1:6));

desviacion = std(Auto3(:,1:6));

maximo = max(Auto3(:,1:6));
minimo = min(Auto3(:,1:6));

rangoTabla = vertcat(maximo,minimo);
rango = maximo - minimo;

disp('-----Media-----');
media
disp('-----Desviacion-----');
desviacion
disp('-----Rango-----');
rango

% Apartado 5 -  Usando toda la base de datos, analiza los predictores de 
% manera gráfica haciendo uso de la función \textit{scatter}. 
% Crea gráficos que resalten la relación entre predictores. 
% Resume los resultados obtenidos.

% Notas: Pintar todos los predictores frente a todos,
% Es decir una figura de 5x5 subplots

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(2)
num=5;

for i=1:num
    for j=1:num

        %%% Otenemos valor posicion
        pos=(i - 1) * 5 + j;

        %%% Subplot + scatter
        subplot(5,5,pos)
        scatter(Auto2{:,i},Auto2{:,j})

        %%% Primera en mayusculas
        nx=Auto.Properties.VariableNames{i};
        ny=Auto.Properties.VariableNames{j};
        nx(1)=upper(nx(1));
        ny(1)=upper(ny(1));
        
        %%% Title + labels
        titulo = sprintf('%s en funcion de %s',ny,nx);
        xlabel(nx);ylabel(ny);%title(titulo);        
    end
end    

% Apartado 5 -   Suponer que queremos predecir la autonom´ıa del coche dada 
% en millas por galón (mpg) en base a otros predicotres. ¿Alguno de los 
% gráficos obtenidos previamente sugieren que otras variables puedan ser de
% utilidad a la hora de predecir mpg?
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

 figure(3)

% Graficas utiles para predecir mpg

for j=2:num

        %%% Otenemos valor posicion
        pos=j-1;

        %%% Subplot + scatter
        subplot(2,2,pos)
        scatter(Auto2{:,1},Auto2{:,j})

        %%% Primera en mayusculas
        nx=Auto.Properties.VariableNames{1};
        ny=Auto.Properties.VariableNames{j};
        nx(1)=upper(nx(1));
        ny(1)=upper(ny(1));
        
        %%% Title + labels
        titulo = sprintf('%s en funcion de %s',ny,nx);
        xlabel(nx);ylabel(ny);%title(titulo);        
end

%% Cinclusion apartado 5:

% Podemos ver de una forma mas efectiva o menos, todas las variables
% incluidas en la figura 3 nos puede aportar informacion para predecir la
% variable mpg, habria que valorar cuales de ellas podrian aportar mas
% informacion respecto a mpg

end