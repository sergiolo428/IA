function Template_T5_E1
% Este script contiene la resolución del ejercicio aplicado 1 del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Este ejercicio está relacionado con el uso de la base de datos College. 
% Trataremos de predecir el número de solicitudes de mátriculación (Apps)
% recibidas usando las otras variables de la base de datos.

% Cargamos base de datos

disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')

load College.mat;
var_names=College.Properties.VariableNames;

% Y -> 3
% X-norm -> 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
% X-Dummy -> 2 

Dum = dummyvar(categorical(College{:,2}));
Dummy = Dum(:,2);

Y = College{:,3};
X = [Dummy College{:,4:19}];
var_names_x = [var_names(2) var_names(4:19)];

% Predecir Apps

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(13).

rng(13); % Fijamos semilla para el generado de números aleatorios

c = cvpartition(length(X),"HoldOut",0.5);
pos_train = c.training;
pos_test = c.test;

X_train = X(pos_train,:);
X_test = X(pos_test,:);

Y_train = Y(pos_train);
Y_test = Y(pos_test);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un modelo de regresión lineal por mínimos cuadráticos  
% usando el conjunto de entrenamiento. Reportar el error de test obtenido.

mdl = fitlm(X_train,Y_train);

Ypred = predict(mdl,X_test);


MSE = mean((Y_test-Ypred).^2);

fprintf('MSE REGRESION LINEAL: %d',MSE);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Ajustar un modelo ridge regression usando el conjunto de 
% entrenamiento y con una lambda seleccionada mediante 10-fold CV. 
% Reportar el error de test obtenido.

rng(3);
k = 10;

cc = cvpartition(sum(pos_train),'KFold',k);

lambda = linspace(0,100,80);
MSE_ridge_CV=[];
for i = 1:k

    pos_train_CV = cc.training(i);
    pos_test_CV = cc.test(i);

    X_train_CV = X_train(pos_train_CV,:);
    X_test_CV  = X_train(pos_test_CV,:);
    Y_train_CV = Y_train(pos_train_CV);
    Y_test_CV  = Y_train(pos_test_CV);

    for j=1:length(lambda)

        B = ridge(Y_train_CV,X_train_CV,lambda(j),0);
        Ypred = B(1)+X_test_CV*B(2:end);
        MSE_ridge_CV(i,j) = mean((Y_test_CV-Ypred).^2);

    end 
end

[val,pos] = min(mean(MSE_ridge_CV));

subplot(211);plot(lambda,mean(MSE_ridge_CV,1));title('RIDGE');
hold on;plot(lambda(pos),val,'ro');hold off;

fprintf('MSE RIDGE LINEAL: %d',val);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Ajustar un modelo lasso usando el conjunto de entrenamiento 
% y con una lambda seleccionada mediante 10-fold CV. Reportar el error de 
% test obtenido y el número de estimaciones de los coeficientes diferentes de 
% cero.

rng(3);
k = 10;

cc = cvpartition(sum(pos_train),'KFold',k);

lambda = linspace(0,100,80);
MSE_lasso_CV=[];
for i = 1:k

    pos_train_CV = cc.training(i);
    pos_test_CV = cc.test(i);

    X_train_CV = X_train(pos_train_CV,:);
    X_test_CV  = X_train(pos_test_CV,:);
    Y_train_CV = Y_train(pos_train_CV);
    Y_test_CV  = Y_train(pos_test_CV);

    for j=1:length(lambda)

        [B,FitInfo] = lasso(X_train_CV,Y_train_CV,"Lambda",lambda(j));
        ypred = FitInfo.Intercept+X_test_CV*B;
        MSE_lasso_CV(i,j) = mean((Y_test_CV-ypred).^2);

    end 
end

[val,pos] = min(mean(MSE_ridge_CV));

subplot(212);plot(lambda,mean(MSE_lasso_CV,1));title('LASSO');
hold on;plot(lambda(pos),val,'ro');hold off;

fprintf('MSE LASSO LINEAL: %d',val);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Ajustar un modelo PCR usando el conjunto de entrenamiento y 
% con una M seleccionada mediante 10-fold CV. Reportar el error de test 
% obtenido y el valor de $m$ obtenido a través de 10-fold CV.

X = zscore(X);

X1 = X(pos_train,:);
X2 = X(pos_test,:);
Y1 = Y(pos_train);
Y2 = Y(pos_test);

%%% APLICAMOS PCA %%%

% PCALoaddings son todos los fis
% PCAScores, predictores transformados (son como las Xtrain del total)
% PCAVar --> Varianza

[PCALoadings,PCAScores,PCAVar,~,explained,mu] = pca(X1);
explained;
cumsum(explained);
k = 10;
CV_MSE_PCR = [];
for aa=1:k
    pos_train_CV = cc.training(aa);
    pos_test_CV = cc.test(aa);
    Ytrain = Y1(pos_train_CV);
    Ytest = Y1(pos_test_CV);
   
    for bb=1:size(X,2)
        X_PCR_train = PCAScores(pos_train_CV,1:bb);
        X_PCR_test = PCAScores(pos_test_CV,1:bb);

        mdl = fitlm(X_PCR_train,Ytrain);
        CV_MSE_PCR(aa,bb) = mean((Ytest-predict(mdl,X_PCR_test)).^2);
        
    end

end

[val,pos] = min(mean(CV_MSE_PCR));

figure(2)
plot(mean(CV_MSE_PCR));
hold on;plot(pos,val);hold off;
xlabel('M');ylabel('CV MSE');

mdl = fitlm(PCAScores(:,1:pos),Y1);

X_PCR_test = (X2-mu)*PCALoadings(:,1:pos);

MSE_test_PCR=mean((Y2-predict(mdl,X_PCR_test)).^2);

[PCALoadings,PCAScores,PCAVar,~,explained,mu] = pca(X,'NumComponents',pos);
mdl_final = fitlm(PCAScores,Y);

fprintf('MSE PCR LINEAL: %d',MSE_test_PCR);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 6 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 6 - Comentar los resultados obtenidos. ¿Cómo de precisas son 
% nuestras predicciones del número de solicitudes recibidas? ¿Hay mucha 
% diferencia entre los errores de test reportados en los apartados anteriores?

% Parece ser que al comparar todo los modelos anteriores, tanto para Lasso
% Ridge, y PCR obtenemos valores muy cercanos al modelo de regresion
% lineal. Por lo que un modelo de regresion lineal seria suficiente.

