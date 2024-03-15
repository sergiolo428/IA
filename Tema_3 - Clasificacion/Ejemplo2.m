function Ejemplo2

%% Practica ALD

load('Smarket.mat');

var_names = Smarket.Properties.VariableNames;


% [~,b]=plotmatrix(Smarket{:,1:end-1});
% for i = 1:length(b)
%     axes(b(i,1));ylabel(var_names{i});
%     axes(b(end,i));xlabel(var_names{i});
% end

% Matriz correlaciones

corr(Smarket{:,1:end-1});

% Ajustar un modelo de Analisis Lineal Discriminante ALD para predecir
% Direction usando Lag1 y Lag2. Al analisis liean discriminante igual que a
% la re greison logistica hay que pasarle un respuesta numerica.

% Para este caso digamos que se caracteriza por tener un umbral lineal,
% menos flexible

Y(strcmp(Smarket.Direction,'Down'))=0;
Y(strcmp(Smarket.Direction,'Up'))=1;
Y=Y';

var_sel = 2:3;

pos_train = Smarket.Year<2005;

Xtrain = Smarket{pos_train,var_sel};
Ytrain = Y(pos_train);


pos_test = Smarket.Year==2005;
Xtest = Smarket{pos_test,var_sel};
Ytest = Y(pos_test);

mdl = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names([var_sel]),'ResponseName',var_names{end});

% Como tenemos dos predictores tendremos una matriz de covarianzas:
% (recordar que esta es igual para todas las clases, matriz simetrica)

mdl.Sigma;

% Medias de lso rpedictores:
% (Estas si seran diferentes para cada clase)

mdl.Mu;

% La primera fila corresponde a los ceros y la segunda a la clase uno
% La primera columna es el predictor uno y la segunda el rpedictor 2


[ypred,yprob,~] = predict(mdl,Xtest);

% yrpob nos devuelve dos columnas, la priemra nos dice que probabilidades
% hay de que la observacion pertenezca a la clase 0, 

% ypred, por defecto le coloca a la matriz de probabilidades y le aplica
% una matriz de probabilidades, esta es la que nos vale

figure(1)
title("ALD")
C = confusionmat(Ytest,ypred);
confusionchart(C,{'Down (0)','Up (1)'});

acc = 100*(C(1,1)+C(2,2))/length(Ytest);
err = 100-acc;

% Analisis Cuadratico Discriminante ACD, 
% El umbral no es una recta, mas flexible menos sesgo, nos acercamos mas al
% problema de la vida real


%% AJustamos el analisis cuadratico discriminante 


mdl = fitcdiscr(Xtrain,Ytrain,'DiscrimType','quadratic','PredictorNames',var_names([var_sel]),'ResponseName',var_names{end});

[ypred,yprob,~] = predict(mdl,Xtest);

figure(2)
title("ACD")
C = confusionmat(Ytest,ypred);
confusionchart(C,{'Down (0)','Up (1)'});

acc = 100*(C(1,1)+C(2,2))/length(Ytest);
err = 100-acc;


%% Modelo KNN 1 vecino 

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',1,'Standardize',1,'PredictorNames',var_names([var_sel]),'ResponseName',var_names{end})

[ypred,yprob,~] = predict(mdl_KNN,Xtest);

figure(3)
title("KNN-1")
C = confusionmat(Ytest,ypred);
confusionchart(C,{'Down (0)','Up (1)'});

acc = 100*(C(1,1)+C(2,2))/length(Ytest);
err = 100-acc;

% OJO si solo usamos un vecino, estamos usando un modelo demasiado
% flexible, se ajusta demasiado, pero esto en el test no es util

%% Modelo KNN 7 vecino 

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',7,'Standardize',1,'PredictorNames',var_names([var_sel]),'ResponseName',var_names{end})

[ypred,yprob,~] = predict(mdl_KNN,Xtest);

C = confusionmat(Ytest,ypred);
confusionchart(C,{'Down (0)','Up (1)'});

acc = 100*(C(1,1)+C(2,2))/length(Ytest);
err = 100-acc;

end