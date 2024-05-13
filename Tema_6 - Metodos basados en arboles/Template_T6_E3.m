function Template_T6_E3
% Este script contiene la resolución del ejercicio aplicado 3 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En este problema usaremos ANN para predecir si un determinado coche
% tiene alta o baja autonomıa en la base de datos Auto.

% Cargamos base de datos

load Auto.mat

disp('%%%%%%%%%%%%%%%%% EJERCICIO 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN

Auto = rmmissing(Auto);

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 -  Crear una variable binaria que tome el valor 1 para coches
% con una autonomía (mpg) superior a la mediana, y que tome el valor 0
% para coches con una autonomía inferior a la mediana.

mediana=median(Auto{:,1})

Y(Auto.mpg>mediana) = 1;
Y(Auto.mpg<=mediana) = 0;

Y=Y';

X = Auto{:,2:8};


fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar una red neuronal artificial a los datos con varios
% valores de H (número de neuronas en la capa oculta de la red) con el objetivo
% de predecir si un coche tiene alta o baja autonom´ıa. Reportar el error
% CV asociado a los diferentes valores del parámetro H. Comentar los resultados.
rng(13);
% Crear red neuronal con 20 neuronas en capa oculta



rng(7);
% Dividir datos en train (0.7), validation (0.15) y test (0.15)

[trainInd, valInd, testInd] = dividerand(length(Y),0.7,0.15,0.15);


% CALCULAR NÚMERO ÓPTIMO DE NEURONAS EN LA CAPA OCULTA
% Usar 5-fold CV en los datos de entrenamiento para elegir H
rng(2)
K = 5;
th = 0.5;
cc = cvpartition(length(trainInd)+length(valInd),'KFold',K);

posCV = [trainInd valInd]; posCV = sort(posCV);
X1 = X(posCV,:);
Y1 = Y(posCV);

H_grid = [1 3 5 10 15 20 25 30 40 50];
CV_error=[];
for i=1:K
    
    pos_train = find(cc.training(i));
    pos_test_CV = find(cc.test(i));

    pos_val_CV = pos_train(1:length(pos_test_CV));
    pos_train_CV = pos_train(length(pos_test_CV)+1:end);


    for j=1:length(H_grid)
        net = patternnet(H_grid(j));

        % opciones de red
        net.layers{1}.transferFcn = 'logsig';
        net.divideFcn='divideind';% Por indice

        % dividerand random
        % divideblock bloques contiguos
        % divideint intercalado
        % divideind por indice

        net.divideParam.trainInd=pos_train_CV;
        net.divideParam.valInd=pos_val_CV;
        net.divideParam.testInd=pos_test_CV;
        net.inputs{1}.processFcns{2}='mapstd'; % Normalize inputs/targets 

        % mapminmax entre -1 y 1
        
        % Train ANN
        train_net = train(net,X1',Y1');
        
        % Test ANN
        ypred = train_net(X1(pos_test_CV,:)');
        yfit=ypred;
        yfit(ypred>th)=1;
        yfit(ypred<=th)=0;
        
        % calcular CV error
        CV_error(i,j) = 100 - 100*(sum(yfit==Y1(pos_test_CV)')/length(Y1(pos_test_CV)));
        % vaciamos yfit para siguiente iteración
        yfit = [];

    end

end


[val,pos] = min(mean(CV_error));

val

% 9.25 ; 25 neuronas k = 3

% 7.81 ; 25 neuronas k = 5

% 8.38 ; 15 neuronas k = 10

% Nota, largo periodo de entrenamiento, error decente, cuidado con overfitear con
% el numero de neuronas

% Dibujamos errorbar
errorbar(H_grid,mean(CV_error),std(CV_error));hold on;plot(H_grid,mean(CV_error),'ro');hold off;xlabel('#neuronas');ylabel('Error CV');

keyboard