function template_T1_E6

%% --- Sergio López --- %%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Cargar base de datos

load('College.mat')

disp('%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

%% Apartado 1 - Calcular la media y desviación estándar de cada una de las 
% variables cuantitativas
disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

means = mean(College(:,3:19));

%% Apartado 2 - Realizar el conteo de universidades públicas y privadas. %%
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

publicPrivateArr = strcmp(College{:,"Private"},'Yes');

numPriv = sum(publicPrivateArr);
numPublic = 777 - numPriv;

%% --- Apartado 3 - Producir los siguientes diagramas de dispersión. --- %%
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% (Uncomment para ver la figura 1)

figure(1)

subplot(2,3,1)
scatter(College.Apps,College.Accept)
xlim([-5000 max(College.Apps)+5000])
ylim([-5000 max(College.Accept)+5000])
title('Acceptance rate');
xlabel('Apps');ylabel('Accepted');


subplot(2,3,2)
scatter(College.Enroll,College.F_Undergrad)
xlim([-5000 max(College.Enroll)+5000])
ylim([-5000 max(College.F_Undergrad)+5000])
title('Enroll vs F_Undergrad');
xlabel('Enroll');ylabel('F_Undergrad');

subplot(2,3,3)
scatter(College.Room_Board,College.Outstate)
xlim([0 max(College.Room_Board)+5000])
ylim([0 max(College.Outstate)+5000])
title('Room_Board vs Outstate');
xlabel('Room_Board');ylabel('Outstate');

subplot(2,3,4)
scatter(College.PhD,College.Expend)
xlim([0 max(College.PhD)+0])
ylim([0 max(College.Expend)+0])
title('PhD vs Expend');
xlabel('PhD');ylabel('Expend');

subplot(2,3,5)
scatter(College.S_F_Ratio,College.Expend)
xlim([0 max(College.S_F_Ratio)+0])
ylim([0 max(College.Expend)+0])
title('S_F_Ratio vs Expend');
xlabel('S_F_ratio');ylabel('Expend');

subplot(2,3,6)
scatter(College.perc_alumni,College.Expend)
xlim([0 max(College.perc_alumni)+0])
ylim([0 max(College.Expend)+0])
title('perc_alumni vs Expend');
xlabel('perc_alumni');ylabel('Expend');

%% ------------- Apartado 4 - Binarizar variable y boxplot ------------- %%
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Crear una variable cualitativa Elite binarizando la variable Top10perc. 
% Si la proporción de estudiantes que estuvieron entre el mejor 10\% de su 
% clase en bachiller es superior al 50\% Elite será igual a 'Yes' de lo 
% contrario, será igual a 'No'. 

%% ------------- Calcula el número de universidades de élite ----------- %%

top10 = College{:,"Top10perc"};

elite(top10>50)="Yes";
elite(top10<50)="No";
elite = elite';

listauni=table(College.Names,elite);

numElite=sum(top10>50);

%% -- Boxplot de la variable Outstate en función de la variable Elite -- %%

outstate = College.Outstate;

figure(2)
boxplot(outstate,elite)

%% ---------------------- Apartado 5 - Histogramas --------------------- %%
fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

figure(3)

subplot(2,3,1)
histogram(College.Top10perc)
title('Top10perc')

subplot(2,3,2)

histogram(College.Top25perc)
title('Top25perc')

subplot(2,3,3)
histogram(College.PhD)
title('PhD')

subplot(2,3,4)
histogram(College.Terminal)
title('Terminal')

subplot(2,3,5)
histogram(College.Room_Board)
title('Room_Board')

subplot(2,3,6)
histogram(College.Outstate)
title('Outstate')

% Diferenciando entre Elite y no Elite

%% ------- Figura unica con diferenciacion entre Elite y NoElite ------- %%

siElite = top10>50;
noElite = top10<50;

figure(4)

subplot(2,3,1)
histogram(College.Top10perc(noElite))
hold on;
histogram(College.Top10perc(siElite))
title('Top10perc')
xlabel('No elite azul, Elite naranja')

subplot(2,3,2)

h1 = histogram(College.Top25perc(noElite))
hold on;
h2 = histogram(College.Top25perc(siElite))
h2.BinWidth = h1.BinWidth
title('Top25perc')
xlabel('No elite azul, Elite naranja')

subplot(2,3,3)
h1 = histogram(College.PhD(noElite))
hold on;
h2 = histogram(College.PhD(siElite))
h2.BinWidth = h1.BinWidth
title('PhD')
xlabel('No elite azul, Elite naranja')

subplot(2,3,4)
h1 = histogram(College.Terminal(noElite))
hold on;
h2 = histogram(College.Terminal(siElite))
h2.BinWidth = h1.BinWidth
title('Terminal')
xlabel('No elite azul, Elite naranja')

subplot(2,3,5)
h1 = histogram(College.Room_Board(noElite))
hold on;
h2 = histogram(College.Room_Board(siElite))
h2.BinWidth = h1.BinWidth
title('Room_Board')
xlabel('No elite azul, Elite naranja')

subplot(2,3,6)
h1 = histogram(College.Outstate(noElite))
hold on;
h2 = histogram(College.Outstate(siElite))
h2.BinWidth = h1.BinWidth
title('Outstate')
xlabel('No elite azul, Elite naranja')



%% Dibujar en una figura dos plaots uno de el histograma general y otro de
% los dos histogramas de elite y no elite juntos

figure(5)
subplot(2,2,1)
h1 = histogram(College.Top10perc)
title('Top10perc')
subplot(2,2,3)
h2 = histogram(College.Top10perc(noElite))
hold on;
h3 = histogram(College.Top10perc(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h3.BinWidth
h2.BinWidth = h3.BinWidth
hold off;

subplot(2,2,2)
h1 = histogram(College.Top25perc)
title('Top25perc')
subplot(2,2,4)
h2 = histogram(College.Top25perc(noElite))
hold on;
h3 = histogram(College.Top25perc(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h3.BinWidth
h2.BinWidth = h3.BinWidth
hold off;


figure(6)
subplot(2,2,1)
h1 = histogram(College.PhD)
title('PhD')
subplot(2,2,3)
h2 = histogram(College.PhD(noElite))
hold on;
h3 = histogram(College.PhD(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h3.BinWidth
h2.BinWidth = h3.BinWidth
hold off;

subplot(2,2,2)
h1 = histogram(College.Terminal)
title('Terminal')
subplot(2,2,4)
h2 = histogram(College.Terminal(noElite))
hold on;
h3 = histogram(College.Terminal(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h3.BinWidth
h2.BinWidth = h3.BinWidth
hold off;


figure(7)

subplot(2,2,1)
h1 = histogram(College.Room_Board)
title('Room_Board')
subplot(2,2,3)
h2 = histogram(College.Room_Board(noElite))
hold on;
h3 = histogram(College.Room_Board(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h2.BinWidth
h3.BinWidth = h2.BinWidth
hold off;

subplot(2,2,2)
h1 = histogram(College.Outstate)
title('Outstate')
subplot(2,2,4)
h2 = histogram(College.Outstate(noElite))
hold on;
h3 = histogram(College.Outstate(siElite))
xlabel('No elite azul, Elite naranja')
h1.BinWidth = h2.BinWidth
h3.BinWidth = h2.BinWidth
hold off;

end