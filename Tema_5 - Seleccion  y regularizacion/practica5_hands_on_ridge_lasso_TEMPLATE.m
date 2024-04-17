function practica5_hands_on_ridge_lasso_TEMPLATE
% Este script contiene la resolución del tutorial práctico del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

load('Hitters');
% Nombre de las variables
var_names = Hitters.Properties.VariableNames;

% Dimensiones de la base de datos original
size(Hitters);

% Remover valores NaN
Hitters = rmmissing(Hitters);

% Dimensiones de la base de datos sin valores NaN
size(Hitters);

disp('%%%%%%%%%%%%%%% RIDGE REGRESSION & THE LASSO %%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Binarizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Hitters{:,14}));
League_N = D(:,2);
D = dummyvar(categorical(Hitters{:,15}));
Division_W = D(:,2);
D = dummyvar(categorical(Hitters{:,20}));
NewLeague_N = D(:,2);
X = [Hitters{:,1:13} League_N Division_W Hitters{:,16:18} NewLeague_N];
Y = Hitters{:,19};

% Creamos matriz de valores posible para lambda RIDGE REGRESSION
% (Creamos una matriz de lambdas)

A = linspace(-2,2,100);
lambda_grid = 10.^A;

%% RIDGE

B = ridge(Y,X,lambda_grid,0); 

% Scaled --> 0 Le decimos que a pesar de 
% standarizar los valores para el modelo que nos devuelva lso valores 
% originales para despuer plotearlos

subplot(211);plot(lambda_grid,B(2:end,:),'LineWidth',2);

grid on;
xlabel('Lambda');
ylabel('Coeficientes');
title('RIDGE REGRESSION');

% Creamos matriz de valores posible para lambda LASSO
% (Creamos una matriz de lambdas)

A = linspace(-2,2,100);
lambda_grid = 10.^A;

%% LASSO
% Vemos que lasso no nos devuelve beta0

B = lasso(X,Y,"Lambda",lambda_grid);

subplot(212);plot(lambda_grid,B,'LineWidth',2);

grid on;
xlabel('Lambda');
ylabel('Coeficientes');
title('LASSO');%pause;close;

% Dividimos la base de datos en train y test
rng(1); % Fijamos semilla para la generación de números aleatorios

asdasd = size(X,10);

%% Partición no estratificada 50% train y 50% test
hpartition = cvpartition(size(X,1),"HoldOut",0.5);
pos_train = hpartition.training;
pos_test = hpartition.test;

% RIDGE REGRESSION
lambda = 4;
B = ridge(Y(pos_train),X(pos_train,:),lambda,0);
ypred = B(1)+X(pos_test,:)*B(2:end);
MSE_test = mean((Y(pos_test)-ypred).^2);

fprintf('\nMSE test (RIDGE, lambda = %.1f) = %.0f \n',lambda,MSE_test);
fprintf('MSE test del modelo nulo = %.0f \n',mean((mean(Y(pos_train))-Y(pos_test)).^2));

lambda = 10^10;

B = ridge(Y(pos_train),X(pos_train,:),lambda,0);
ypred = B(1)+X(pos_test,:)*B(2:end);
MSE_test = mean((Y(pos_test)-ypred).^2);

fprintf('MSE test (RIDGE, lambda = %.0f) = %.0f \n',lambda,MSE_test);

lambda = 0;

B = ridge(Y(pos_train),X(pos_train,:),lambda,0);
ypred = B(1)+X(pos_test,:)*B(2:end);
MSE_test = mean((Y(pos_test)-ypred).^2);

fprintf('MSE test (RIDGE, lambda = %.0f) = %.0f \n',lambda,MSE_test);

% least squares

mdl = fitlm(X(pos_train,:),Y(pos_train));
ypred = predict(mdl,X(pos_test,:));
MSE_test = mean((Y(pos_test)-ypred).^2);

fprintf('MSE test (least squares) = %.0f \n',MSE_test);

% Con LASSO
lambda = 4;

[B,FitInfo] = lasso(X(pos_train,:),Y(pos_train),"Lambda",lambda);
ypred = FitInfo.Intercept+X(pos_test,:)*B;
MSE_test_LASSO = mean((Y(pos_test)-ypred).^2);

fprintf('\nMSE test (LASSO, lambda = %.0f) = %.0f \n',lambda,MSE_test_LASSO);

% Usamos ahora 10 FOLD CV en entrenamiento para seleccionar lambda óptima
% Creamos particiones
% OJO hacemos CV solo con el train!!

rng(2);
k = 10;

c = cvpartition(sum(pos_train),'KFold',k);
X1 = X(pos_train,:);
Y1 = Y(pos_train);


lambda_grid = linspace(0.01,100,100);
lambda_grid_LASSO = linspace(0.01,100,100);
CV_MSE=[];CV_MSE_LASSO=[];
for aa = 1:k

    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);

    Xtrain = X1(pos_train_CV,:);
    Xtest  = X1(pos_test_CV,:);
    Ytrain = Y1(pos_train_CV);
    Ytest  = Y1(pos_test_CV);


    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(lambda_grid) 
        % RIDGE REGRESSION
        
        B = ridge(Ytrain,Xtrain,lambda_grid(bb),0);
        ypred = B(1)+Xtest*B(2:end);
        CV_MSE(aa,bb) = mean((Ytest-ypred).^2);
        
        % LASSO
        
        [B,FitInfo] = lasso(Xtrain,Ytrain,"Lambda",lambda_grid(bb));
        ypred = FitInfo.Intercept+Xtest*B;
        CV_MSE_LASSO(aa,bb) = mean((Ytest-ypred).^2);
        
    end
    
end

[val,pos] = min(mean(CV_MSE));
[val2,pos2] = min(mean(CV_MSE_LASSO));


subplot(211);plot(lambda_grid,mean(CV_MSE,1));title('RIDGE');
hold on;plot(lambda_grid(pos),val,'ro');hold off;
subplot(212);plot(lambda_grid_LASSO,mean(CV_MSE_LASSO,1));title('LASSO');
hold on;plot(lambda_grid_LASSO(pos2),val2,'ro');hold off;
pause;close;


% Para RIDGE

lambda_RIDGE = lambda_grid(pos);
B = ridge(Y(pos_train),X(pos_train,:),lambda_RIDGE,0);
ypred = B(1)+X(pos_test,:)*B(2:end);
MSE_test_RIDGE = mean((Y(pos_test)-ypred).^2);


% Para LASSSO

lambda_LASSO = lambda_grid(pos2);
[B,FitInfo] = lasso(X(pos_train,:),Y(pos_train),"Lambda",lambda_LASSO);
ypred = FitInfo.Intercept+X(pos_test,:)*B;
MSE_test_LASSO = mean((Y(pos_test)-ypred).^2);


fprintf('\n Ridge regression, CV lambda = %.2f, el error de test = %4.0f \n',lambda_RIDGE,MSE_test_RIDGE);
fprintf('\n LASSO, CV lambda = %.2f, el error de test = %4.0f \n',lambda_LASSO,MSE_test_LASSO);

% Reajustamos el modelo usando todas las observaciones de las que disponemos 
% y el valor de lambda seleccionado

B = ridge(Y,X,lambda_RIDGE,0);

[B,FitInfo] = lasso(X,Y,"Lambda",lambda_LASSO);
[FitInfo.Intercept;B]

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PCR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
explained
cumsum(explained)

CV_MSE_PCR = [];
for aa=1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
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
plot(mean(CV_MSE_PCR))
hold on;plot(pos,val);hold off;
xlabel('M');ylabel('CV MSE');

mdl = fitlm(PCAScores(:,1:pos),Y1);

X_PCR_test = (X2-mu)*PCALoadings(:,1:pos);

MSE_test_PCR=mean((Y2-predict(mdl,X_PCR_test)).^2)

[PCALoadings,PCAScores,PCAVar,~,explained,mu] = pca(X,'NumComponents',pos);
mdl_final = fitlm(PCAScores,Y);


