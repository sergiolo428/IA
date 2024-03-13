function Template_T3_E6
acc_RL = 0;
acc_LDA = 0;
acc_QDA = 0;
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

var_sel = 2:7;

X = Weekly{:,2:7};

mdl = fitglm(X,Y,'Distribution','binomial','VarNames',var_names([var_sel end]))

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

C = confusionmat(Y,Ypred)
confusionchart(C,{'Down (0)','Up (1)'})

acc1 = 100*(C(1,2)+C(2,2))/length(Ypred)
err1 = 100-acc1;

% Fijandonos en la matriz, een la segunda columna veremos los valores
% acertados, es decir los que originalmente en nuestro datos tenian un
% valor y al predecir obtuvimos le mismo, en este caso 430 bajando, y 557
% subienedo. Por otro lado vemos que en la prediccion 48 se han
% categorizado como Down cuando en realidad eran Up, y 54 eran Down pero se
% situaron como Up.

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Ahora ajusta un modelo de regresión logística usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).

pos_train2 = Weekly.Year<2008;
pos_test2 = Weekly.Year>=2008;

Xtrain2 = Weekly{pos_train2,3};
Xtest2 = Weekly{pos_test2,3};

Ytrain2 = Y(pos_train2);
YtrueTest2 = Y(pos_test2);

mdl = fitglm(Xtrain2,Ytrain2,'Distribution','binomial','VarNames',var_names([3 9]));

Yprob2 = predict(mdl,Xtest2);

Ypred2(Yprob2>=0.5)=0;
Ypred2(Yprob2<0.5)=1;

C = confusionmat(YtrueTest2,Ypred2)
confusionchart(C,{'Down (0)','Up (1)'})

acc2 = 100*(C(1,1)+C(2,1))/length(YtrueTest2)
err2 = 100-acc2;

% Podeemos observar como en este caso hemos obtenido un beneficio de un 2%
% de acieerto, pasando de un 90% con todos los preditores a un 92% usando
% unicamente Log2.

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Ahora ajusta un modelo ALD usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Ahora ajusta un modelo ACD usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 7 - Ahora ajusta un modelo 1-NN usando como datos 
% de entrenamiento las observaciones desde 1990 hasta 2008, y utiliza Lag2 
% como único predictor. Calcula la matriz de confusión y el porcentaje de predicciones 
% correctas para los datos de test (observaciones desde 2009 a 2010).
rng(1);% Controlamos la semilla para la creación de números aleatorios



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 8 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 8 - ¿Cuál de los métodos parece obtener mejores resultados?
% Respuesta
% El ALD y la regresión logística presentan los mejores resultados. El 1-NN
% y QDA presentan resultados pobres. En este escenario un umbral de
% decisión de bayes lineal parece ser lo acertado.



fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 9 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 9 - Experimenta con diferentes combinaciones de los predictores 
% (posibles transformaciones o interacciones) para cada uno de los métodos. 
% Reporta las variables, método, y la matriz de  confusión que parecen 
% obtener los mejores resultados en los datos de test. Deberías de 
% experimentar con diferentes valores de $K$ para el clasificador KNN.



% RL


% LDA


% QDA


% KNN
rng(1);


fprintf('\nPRECISIONES DE LOS MODELOS \n')
fprintf('RL: %4.1f%% \n',acc_RL);
fprintf('LDA: %4.1f%% \n',acc_LDA);
fprintf('QDA: %4.1f%% \n',acc_QDA);
fprintf('KNN: %4.1f%% \n',acc_KNN);

end