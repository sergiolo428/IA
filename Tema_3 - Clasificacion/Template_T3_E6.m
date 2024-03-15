function Template_T3_E6
acc_RL = 0;
acc_ALD = 0;
acc_ACD = 0;
acc_KNN = 0;
% Este script contiene la resolución del ejercicio aplicado 6 del Tema 3
% de la asignatura 'Técnicas de Inteligencia Artificial'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Este ejercicio está relacionado con el uso de la base de datos Weekly 
% que es parecida a Smarket, pero contiene 1089 retornos semanales de 21 
% años, desde principios de 1990 hasta el final de 2010.

% Cargamos base de datos

load("Weekly.mat");

%% HACER  EJERS 1 2 3 4 

disp('%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Producir resúmenes numéricos y gráficos de la base de 
% datos Weekly. ¿Existe algún patrón?

var_names = Weekly.Properties.VariableNames;

% Scatters de las variables cuantitativas (todas menos Direction)

% [~,b]=plotmatrix(Weekly{:,1:end-1});
% for i=1:length(var_names)-1
% 
%     axes(b(i,1));ylabel(var_names{i});
%     axes(b(end,i));xlabel(var_names{i});
% end

%%% Relaciones posibles: Volume y Year

% Medimos correlación entre variables cuantitativas (todas menos Direction)

corr(Weekly{:,1:end-1})

%%% Mirando las correlaciones, observamos que efectivamente hay una 
%%% correlacion entre Volume y Year es de 0.8419

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Utiliza toda la base de datos para ajustar un modelo de 
% regresión logística para predecir Direction en base a las cinco variables lag y 
% Volume. ¿Es alguno de los predictores estadísticamente significativo? 
% En caso afirmativo, identifícalos.

Y(strcmp(Weekly.Direction,'Down'))=0;
Y(strcmp(Weekly.Direction,'Up'))=1;
Y=Y';

var_sel = 2:6;

X = Weekly{:,2:6};

mdl = fitglm(X,Y,'Distribution','binomial','VarNames',var_names([var_sel end]));

% Analizando el modelo podemos ver que el p-valor asociado al Chi^2 es
% menor que 0.05 , es decir, hay almenos un predictor que si tiene relacion
% con la variable a predecir

% En este caso vemos que Lag2 tiene un p-valor de 0.029

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas. Examina la matriz de confusión y explica lo que ésta indica sobre 
% los tipos de errores que el modelo de regresión logística comete.
Yprob = predict(mdl,X);

Ypred(Yprob>=0.5)=1;
Ypred(Yprob<0.5)=0;

figure(1)
C = confusionmat(Y,Ypred);
confusionchart(C,{'Down (0)','Up (1)'});

acc1 = 100*(C(1,1)+C(2,2))/length(Ypred);
err1 = 100-acc1;

% Fijandonos en la matriz vemos que hay un numero considerable de falsos
% positivos, es por ello por lo que tenemos un 56% de acierto.

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Ahora ajusta un modelo de regresión logística usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).

pos_train = Weekly.Year<2008;
pos_test = Weekly.Year>=2008;

Xtrain = Weekly{pos_train,3};
Xtest = Weekly{pos_test,3};

Ytrain = Y(pos_train);
YtrueTest = Y(pos_test);

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([3 9]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

figure(2)
C = confusionmat(YtrueTest,Ypred_RL)
confusionchart(C,{'Down (0)','Up (1)'})

acc_RL = 100*(C(1,1)+C(2,2))/length(YtrueTest)
err_RL = 100-acc_RL;

% Podeemos observar como en este caso obeservamos un caso parecido al
% ejemplo anterior, peero esta vez con un porcentaje de 55%

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Ahora ajusta un modelo ALD usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).

pos_train = Weekly.Year<2008;
pos_test = Weekly.Year>=2008;

Xtrain = Weekly{pos_train,3};
Xtest = Weekly{pos_test,3};

Ytrain = Y(pos_train);
YtrueTest = Y(pos_test);

mdl_ALD = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names{3},'ResponseName',var_names{9});

mdl_ALD.Sigma

mdl_ALD.Mu

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

figure(3)
C = confusionmat(YtrueTest,ypred_ALD);
confusionchart(C,{'Down (0)','Up (1)'});

acc_ALD = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ALD = 100-acc_ALD;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Ahora ajusta un modelo ACD usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).

mdl_ACD = fitcdiscr(Xtrain,Ytrain,'DiscrimType','quadratic','PredictorNames',var_names{3},'ResponseName',var_names{end});

[ypred_ACD,yprob_ACD,~] = predict(mdl_ACD,Xtest);

figure(4)
C = confusionmat(YtrueTest,ypred_ACD);
confusionchart(C,{'Down (0)','Up (1)'});

acc_ACD = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ACD = 100-acc_ACD;


fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 7 - Ahora ajusta un modelo 1-NN usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).
rng(1);% Controlamos la semilla para la creación de números aleatorios

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',1,'Standardize',1,'PredictorNames',var_names{3},'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

figure(5)
C = confusionmat(YtrueTest,ypred_KNN);
confusionchart(C,{'Down (0)','Up (1)'});

acc_KNN = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_KNN = 100-acc_KNN;

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 8 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 8 - ¿Cuál de los métodos parece obtener mejores resultados?
% Respuesta

% Podemos apreciar que RL y ALD serian los mejores ajustes que podriamos
% usar para esta base de datos.


fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 9 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 9 - Experimenta con diferentes combinaciones de los predictores 
% (posibles transformaciones o interacciones) para cada uno de los métodos. 
% Reporta las variables, método, y la matriz de  confusión que parecen 
% obtener los mejores resultados en los datos de test. Deberías de 
% experimentar con diferentes valores de $K$ para el clasificador KNN.


% RL

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 1 2
Xtrain = Weekly{pos_train,[2 3]};
Xtest = Weekly{pos_test,[2 3]};

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([2 3 9]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

C = confusionmat(YtrueTest,Ypred_RL);

acc_RL12 = 100*(C(1,1)+C(2,2))/length(YtrueTest)
err_RL12 = 100-acc_RL12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 3
Xtrain = Weekly{pos_train,[3 4]};
Xtest = Weekly{pos_test,[3 4]};

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([3 4 9]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

C = confusionmat(YtrueTest,Ypred_RL);

acc_RL23 = 100*(C(1,1)+C(2,2))/length(YtrueTest)
err_RL23 = 100-acc_RL23;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 4
Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly{pos_test,[3 5]};

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([3 5 9]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

C = confusionmat(YtrueTest,Ypred_RL);

acc_RL24 = 100*(C(1,1)+C(2,2))/length(YtrueTest)
err_RL24 = 100-acc_RL24;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 5
Xtrain = Weekly{pos_train,[3 6]};
Xtest = Weekly{pos_test,[3 6]};

mdl_RL = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([3 6 9]));
Yprob_RL = predict(mdl_RL,Xtest);
Ypred_RL(Yprob_RL>=0.5)=1;
Ypred_RL(Yprob_RL<0.5)=0;

C = confusionmat(YtrueTest,Ypred_RL);

acc_RL25 = 100*(C(1,1)+C(2,2))/length(YtrueTest)
err_RL25 = 100-acc_RL25;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ALD

%Log 1 2
Xtrain = Weekly(pos_train,[2 3]);
Xtest = Weekly{pos_test,[2 3]};

mdl_ALD = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names([2 3]),'ResponseName',var_names{9});

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

C = confusionmat(YtrueTest,ypred_ALD);

acc_ALD12 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ALD12 = 100-acc_ALD12;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 3
Xtrain = Weekly{pos_train,[3 4]};
Xtest = Weekly(pos_test,[3 4]);

mdl_ALD = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names([3 4]),'ResponseName',var_names{9});

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

C = confusionmat(YtrueTest,ypred_ALD);

acc_ALD23 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ALD23 = 100-acc_ALD23;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 4
Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly(pos_test,[3 5]);

mdl_ALD = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names([3 5]),'ResponseName',var_names{9});

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

C = confusionmat(YtrueTest,ypred_ALD);

acc_ALD24 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ALD24 = 100-acc_ALD24;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Log 2 5
Xtrain = Weekly{pos_train,[3 6]};
Xtest = Weekly(pos_test,[3 6]);

mdl_ALD = fitcdiscr(Xtrain,Ytrain,'PredictorNames',var_names([3 6]),'ResponseName',var_names{9});

[ypred_ALD,yprob_ALD,~] = predict(mdl_ALD,Xtest);

C = confusionmat(YtrueTest,ypred_ALD);

acc_ALD25 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ALD25 = 100-acc_ALD25;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% ACD
%Log 2 4
Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly(pos_test,[3 5]);

mdl_ACD = fitcdiscr(Xtrain,Ytrain,'DiscrimType','quadratic','PredictorNames',var_names([3 5]),'ResponseName',var_names{end});

[ypred_ACD,yprob_ACD,~] = predict(mdl_ACD,Xtest);

figure(4)
C = confusionmat(YtrueTest,ypred_ACD);
confusionchart(C,{'Down (0)','Up (1)'});

acc_ACD24 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_ACD24 = 100-acc_ACD24;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% KNN
%Log 2 4 _1

rng(1);

Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly(pos_test,[3 5]);

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',1,'Standardize',1,'PredictorNames',var_names([3 5]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(YtrueTest,ypred_KNN);

acc_KNN24_1 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_KNN24_1 = 100-acc_KNN24_1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Log 2 4 _2

rng(1);

Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly(pos_test,[3 5]);

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',2,'Standardize',1,'PredictorNames',var_names([3 5]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(YtrueTest,ypred_KNN);

acc_KNN24_2 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_KNN24_2 = 100-acc_KNN24_2;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Log 2 4 _5

rng(1);

Xtrain = Weekly{pos_train,[3 5]};
Xtest = Weekly(pos_test,[3 5]);

mdl_KNN = fitcknn(Xtrain,Ytrain,'NumNeighbors',5,'Standardize',1,'PredictorNames',var_names([3 5]),'ResponseName',var_names{end});

[ypred_KNN,yprob_KNN,~] = predict(mdl_KNN,Xtest);

C = confusionmat(YtrueTest,ypred_KNN);

acc_KNN24_5 = 100*(C(1,1)+C(2,2))/length(YtrueTest);
err_KNN24_5 = 100-acc_KNN24_5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('\nPRECISIONES DE LOS MODELOS \n')
fprintf('RL: %4.1f%% \n',acc_RL);
fprintf('RL12: %4.1f%% \n',acc_RL12);
fprintf('RL23: %4.1f%% \n',acc_RL23);
fprintf('RL24: %4.1f%% \n',acc_RL24);
fprintf('RL25: %4.1f%% \n',acc_RL25);

fprintf('\n');

fprintf('ALD: %4.1f%% \n',acc_ALD);
fprintf('ALD12: %4.1f%% \n',acc_ALD12);
fprintf('ALD23: %4.1f%% \n',acc_ALD23);
fprintf('ALD24: %4.1f%% \n',acc_ALD24);
fprintf('ALD25: %4.1f%% \n',acc_ALD25);

fprintf('\n');

fprintf('ACD: %4.1f%% \n',acc_ACD);
fprintf('ACD24: %4.1f%% \n',acc_ACD24);

fprintf('\n');

fprintf('KNN: %4.1f%% \n',100-acc_KNN);
fprintf('KNN24_1: %4.1f%% \n',acc_KNN24_1);
fprintf('KNN24_2: %4.1f%% \n',acc_KNN24_2);
fprintf('KNN24_5: %4.1f%% \n',100-acc_KNN24_5);

fprintf('\n');

% Viendo los resultados a pesar de que en un principio RL y LD eran las
% mejore sopciones, podemos observar como  con KNN para K=2 obtenemos casi
% un 60% 


end