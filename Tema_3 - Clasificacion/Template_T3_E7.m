function Template_T3_E7
% Este script contiene la resolución del ejercicio aplicado 7 del Tema 3
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Este ejercicio está relacionado con el uso de la base de datos Auto 
% que es parecida a Smarket, pero contiene 1089 retornos semanales de 21 
% años, desde principios de 1990 hasta el final de 2010.

% Cargamos base de datos

load('Auto.mat')

var_names = Auto.Properties.VariableNames;

disp('%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN

Auto(isnan(Auto.horsepower),:)=[];

% Scatters de las variables cuantitativas (todas menos Direction)

% [~,b]=plotmatrix(Auto{:,1:7});
% for i = 1:length(b)
%     axes(b(i,1));ylabel(var_names{i});
%     axes(b(end,i));xlabel(var_names{i});
% end

% Medimos correlación entre variables cuantitativas (todas menos Direction)

corr(Auto{:,1:end-1})

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Crear una variable binaria, mpg01, que contenga
% un 1 si mpg es superior al valor mediano, y un 0 si mpg es menor que la
% mediana.

mediana = median(Auto.mpg);

mpg01(Auto.mpg>=mediana)=1;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Utiliza scatterplots y boxplots  para analizar la asociación
% entre mpg01 y las otras características. ¿Qué predictores parecen
% ser más útiles a la hora de predecir mpg01?

AutoBinary = Auto;

mpg01 = mpg01';

AutoBinary.mpg = mpg01;

figure(1)
for i=1:8
    subplot(2,8,i)
    scatter(AutoBinary.mpg,AutoBinary{:,i});
    axis([-1 2 0 inf])
end

% 
% for i=8:14
%     subplot(2,7,i)
%     boxplot(AutoBinary.mpg,AutoBinary{:,i-7});
% end

corr(AutoBinary{:,1:end-1});

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 -Divide los datos en conjunto de entrenamiento (70%) y de 
% test (30%). Para cada valor posible de la variable year cuantifica el 
% número de coches diferentes que hay y asigna en orden de aparición el 
% 70% a entrenamiento y el resto a test. 

% Obtenemos el numero de coches por año
for i=1:13
    num(i) = sum(AutoBinary.year==i+69);
end

% Obtenemos el 70% del numero de coches por año
numCuant = round(num*0.7);

% Calculamos la primera posicion de cada grupo de 70% 
val=1;
for i=1:13
    firstPos(i) = val;
    val=val+num(i);
end

% Calculamops la ultima posicion
lastPos = firstPos+numCuant-1;

% Inicializamos las tablas de 70% y 30%
XtrainAll = AutoBinary(firstPos(1):lastPos(1),2:8);
YtrainAll = AutoBinary(firstPos(1):lastPos(1),1);

XtestAll = AutoBinary(lastPos(1)+1:lastPos(1)+num(1)-numCuant(1),2:8);
YtestAll = AutoBinary(lastPos(1)+1:lastPos(1)+num(1)-numCuant(1),1);

% Completamos las tablas con el reesto de datos
for i=2:13
    XtrainAll = [XtrainAll;AutoBinary(firstPos(i):lastPos(i),2:8)];
    YtrainAll = [YtrainAll;AutoBinary(firstPos(i):lastPos(i),1)];

    XtestAll = [XtestAll;AutoBinary(lastPos(i)+1:lastPos(i)+num(i)-numCuant(i),2:8)];
    YtestAll = [YtestAll;AutoBinary(lastPos(i)+1:lastPos(i)+num(i)-numCuant(i),1)];

end

height(XtrainAll) + height(XtestAll)
height(YtrainAll) + height(YtestAll)

Ytest = YtestAll{:,:};

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Utiliza el ALD en los datos de entrenamiento para predecir 
% mpg01 usando las variables que parecían estar más asociadas con 
% mpg01 en el apartado 2. ¿Cuál es el error de test del modelo?

Xtrain = XtrainAll(:,[4 5 6]);
Xtest = XtestAll{:,[4 5 6]};

mdl_ALD = fitcdiscr(Xtrain,YtrainAll,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{1});

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

C = confusionmat(Ytest,ypred_ALD);
figure(2)
title('ALD')
confusionchart(C,{'Down (0)','Up (1)'});

acc_ALD = 100*(C(1,1)+C(2,2))/length(Ytest);
err_ALD = 100-acc_ALD;


fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Utiliza el ACD en los datos de entrenamiento para predecir 
% mpg01 usando las variables que parecían estar más asociadas con 
% mpg01 en el apartado 2. ¿Cuál es el error de test del modelo?

Xtrain = XtrainAll(:,[4 5 6]);
Xtest = XtestAll{:,[4 5 6]};

mdl_ACD = fitcdiscr(Xtrain,YtrainAll,'DiscrimType','quadratic','PredictorNames',var_names([5 6 7]),'ResponseName',var_names{1});

[ypred_ACD,yprob_ACD,~] = predict(mdl_ACD,Xtest);

figure(4)
C = confusionmat(Ytest,ypred_ACD);
figure(3)
title('ACD')
confusionchart(C,{'Down (0)','Up (1)'});

acc_ACD = 100*(C(1,1)+C(2,2))/length(Ytest);
err_ACD = 100-acc_ACD;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Utiliza la regresión logística en los datos de entrenamiento 
% para predecir mpg01 usando las variables que parecían estar más asociadas con 
% mpg01 en el apartado 2. ¿Cuál es el error de test del modelo?

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

Ytrain = YtrainAll{:,:};

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([5 6 7 1]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

figure(4)
title('RL')
C = confusionmat(Ytest,Ypred_RL)
confusionchart(C,{'Down (0)','Up (1)'})

acc_RL = 100*(C(1,1)+C(2,2))/length(Ytest)
err_RL = 100-acc_RL;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 7 - Utiliza KNN en los datos de entrenamiento, con diferentes 
% valores de K, para predecir mpg01 usando las variables que parecían 
% estar más asociadas con mpg01 en el apartado 2. ¿Cuál es el error de test
% del modelo? ¿Qué valor de K parece obtener mejores resultados en test?

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KNN_1
rng(1);

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',1,'Standardize',1,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred_KNN);
figure(5)
title('KNN')
confusionchart(C,{'Down (0)','Up (1)'})
acc_KNN_1 = 100*(C(1,1)+C(2,2))/length(Ytest);
err_KNN_1 = 100-acc_KNN_1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KNN_5
rng(1);

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Standardize',1,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred_KNN);
figure(5)
title('KNN')
confusionchart(C,{'Down (0)','Up (1)'})
acc_KNN_5 = 100*(C(1,1)+C(2,2))/length(Ytest);
err_KNN_5 = 100-acc_KNN_5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KNN_10
rng(1);

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',10,'Standardize',1,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred_KNN);
figure(5)
title('KNN')
confusionchart(C,{'Down (0)','Up (1)'})
acc_KNN_10 = 100*(C(1,1)+C(2,2))/length(Ytest);
err_KNN_10 = 100-acc_KNN_10;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KNN_15
rng(1);

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',15,'Standardize',1,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred_KNN);
figure(5)
title('KNN')
confusionchart(C,{'Down (0)','Up (1)'})
acc_KNN_15 = 100*(C(1,1)+C(2,2))/length(Ytest);
err_KNN_15 = 100-acc_KNN_15;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% KNN_20
rng(1);

Xtrain = XtrainAll{:,[4 5 6]};
Xtest = XtestAll{:,[4 5 6]};

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',20,'Standardize',1,'PredictorNames',var_names([5 6 7]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred_KNN);
figure(5)
title('KNN')
confusionchart(C,{'Down (0)','Up (1)'})
acc_KNN_20 = 100*(C(1,1)+C(2,2))/length(Ytest);
err_KNN_20 = 100-acc_KNN_20;

fprintf('\nPRECISIONES DE LOS MODELOS \n')
fprintf('RL: %4.1f%% \n',acc_RL);
fprintf('ALD: %4.1f%% \n',acc_ALD);
fprintf('ACD: %4.1f%% \n',acc_ACD);
fprintf('KNN_1: %4.1f%% \n',acc_KNN_1);
fprintf('KNN_5: %4.1f%% \n',acc_KNN_5);
fprintf('KNN_10: %4.1f%% \n',acc_KNN_10);
fprintf('KNN_15: %4.1f%% \n',acc_KNN_15);
fprintf('KNN_20: %4.1f%% \n',acc_KNN_20);

% Viendo los resultados, vemos que con el uso dee los predictores weight
% acceleration y year obtenemos resultados decentes, destacar que el ajuste
% que mejores resultados ha dado ha sido con Regresion logistica con un
% 83.8%, y concretamente para el caso de KNN, parece ser que el mejor
% resultado lo obtenemos con K = 15 con un 82.1%
