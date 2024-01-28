function template_T1_E6
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cargar base de datos

load('College.mat')

disp('%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

% Apartado 1 - Calcular la media y desviación estándar de cada una de las 
% variables cuantitativas
disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

cuantis = [3:19];

means = mean(College(:,cuantis));

% Apartado 2 - Realizar el conteo de universidades públicas y privadas.
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

publicPrivateArr = strcmp(College{:,"Private"},'Yes')

numPriv = sum(publicPrivateArr);
numPublic = 777 - numPriv;

% Apartado 3 - Producir los siguientes diagramas de dispersión.
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(1)

subplot(2,2,1)
scatter(College.Apps,College.Accept)
xlim([-5000 max(College.Apps)+5000])
ylim([-5000 max(College.Accept)+5000])
title('Acceptance rate');
xlabel('Apps');ylabel('Accepted');

subplot(2,2,2)
scatter(College.Accept,College.Enroll)
xlim([-2000 max(College.Accept)+5000])
ylim([-2000 max(College.Enroll)+5000])
title('Enroll from accepted rate');
xlabel('Accepetd');ylabel('Enroll');


subplot(2,2,3)
scatter(College.Accept,College.Enroll)
xlim([-2000 max(College.Accept)+5000])
ylim([-2000 max(College.Enroll)+5000])
title('Enroll from accepted rate');
xlabel('Accepetd');ylabel('Enroll');

subplot(2,2,4)


% Apartado 4 - Binarizar variable y boxplot
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Crear una variable cualitativa Elite binarizando la variable Top10perc. 
% Si la proporción de estudiantes que estuvieron entre el mejor 10\% de su 
% clase en bachiller es superior al 50\% Elite será igual a 'Yes' de lo 
% contrario, será igual a 'No'. 

top10 = College{:,"Top10perc"};

for i=1:length(College{:,1})
    
    if College{i,"Top10perc"}
    
end

% Calcula el número de universidades de élite


% Boxplot de la variable Outstate en función de la variable Elite



% Apartado 5 - Histogramas
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

