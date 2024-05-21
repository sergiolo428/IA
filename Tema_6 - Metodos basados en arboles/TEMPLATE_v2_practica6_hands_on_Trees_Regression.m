function TEMPLATE_v2_practica6_hands_on_Trees_Regression
% Este script contiene la resolución del tutorial práctico del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

load Boston.mat;

disp('%%%%%%%%%%%%%%%%%%%% ÁRBOLES DE REGRESIÓN %%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Dividimos, por tanto, la base de datos en train y test
rng(4); % Fijamos semilla para el generado de números aleatorios
hpartition = cvpartition(size(Boston,1),"HoldOut",0.5); % Partición no estratificada
% 50% train y 50% test
pos_train = hpartition.training;
pos_test = hpartition.test;

Y = Boston.medv;
X = Boston(:,1:end-1);

% Entrenamos árbol de regresión
tree = fitrtree(X(pos_train,:),Y(pos_train));
alpha_grid = tree.PruneAlpha;

% Visualizamos árbol de clasificación
view(tree,'Mode','graph')
 
% Evaluamos rendimiento del árbol de clasificación en test
ypred = predict(tree,X(pos_test,:));
MSE = mean((y(pos_test)-ypred).^2);
disp('%%%%%%%%%%%%% TRAIN/TEST %%%%%%%%%%%%%')
fprintf('RMSE del árbol de regresión (nodos terminales=%d) = %4.2f \n\n',sum(~tree.IsBranchNode),sqrt(MSE));


% PODA DEL ÁRBOL
% Usar K-fold CV en los datos de entrenamiento para elegir ALPHA
rng(2)
k = 10;
c = cvpartition(sum(pos_train),'KFold',k);

X1 = X(pos_train,:);
Y1 = Y(pos_train);

CV_MSE=[];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = X1(pos_train_CV,:);
    Xtest = X1(pos_test_CV,:);
    Ytrain = Y1(pos_train_CV);
    Ytest = Y1(pos_test_CV);
    
    % Entrenamos árbol
    tree_train = fitrtree(Xtrain,Ytrain);
    
    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(alpha_grid)-1 %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria
        tree2 = prune(tree_train,'Alpha',alpha_grid(bb));
        ypred = predict(tree2,Xtest);
        CV_MSE(aa,bb) = mean((ypred-Ytest).^2);
    end
    
end
[val,pos] = min(mean(CV_MSE));

tree_pruned = prune(tree,'Alpha',alpha_grid(pos));
% Visualizamos árbol de clasificación
view(tree_pruned,'Mode','graph')

CV_RMSE = sqrt(CV_MSE);
% Evaluamos rendimiento del árbol de clasificación podado en test
ypred = predict(tree_pruned,X(pos_test,:));
MSE = mean((ypred-Y(pos_test)).^2);
fprintf('RMSE (TEST) del árbol podado (alpha=%.3f  nodos terminales=%d) = %4.2f \n\n',alpha_grid(pos),sum(~tree_pruned.IsBranchNode),sqrt(MSE));

errorbar(alpha_grid(1:end-1),mean(CV_RMSE),std(CV_RMSE));
hold on;plot(alpha_grid(1:end-1),mean(CV_RMSE),'ro');hold off;xlabel('alpha');ylabel('RMSE CV');
pause;close;

median(Y(pos_test))


disp('%%%%%%%%%%%%%%%%%%% BAGGING/RANDOM FORESTS %%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

rng(4);
mdl_bagged = TreeBagger(100,X(pos_train,:),Y(pos_train),"Method","regression",...
    "NumPredictorsToSample","all"); % "all" para considerar todos los predictores por arbol

% Visualizamos el primero de los árboles de regresión
view(mdl_bagged.Trees{1},'Mode','graph')

% Evaluamos rendimiento en test del árbol bagged 
ypred = predict(mdl_bagged,X(pos_test,:));
MSE = mean((ypred-Y(pos_test)).^2);
fprintf('RMSE (TEST) del árbol bagged = %4.2f \n\n',sqrt(MSE));

% ----------------------------------------------------------------------- %
rng(4);
mdl_RF = TreeBagger(100,X(pos_train,:),Y(pos_train),"Method","regression",...
    "NumPredictorsToSample",4); % "all" para considerar todos los predictores por arbol

% Evaluamos rendimiento en test del árbol bagged 
ypred = predict(mdl_RF,X(pos_test,:));
MSE = mean((ypred-Y(pos_test)).^2);
fprintf('RMSE (TEST) del RF = %4.2f \n\n',sqrt(MSE));


% ------------ 1/3 de los predictores que no se usan, out of bag oob --- %
% ---- Esas observaciones valen para testear ya que no se han usado ---- %

% Usando estos datos analizaremos la importancia de los prodictores en el
% modelo dependiendo de en la cantidad de veces que aparecen en el aroblm y
% la cantidad de error que dismnuye al usarlos.

% RF -> importancia de los predictores y OOB error
rng(4);
mdl_RF_OOB = TreeBagger(100,X(pos_train,:),Y(pos_train),"Method","regression",...
    "NumPredictorsToSample",4,"OOBPredictorImportance","on");

imp = mdl_RF_OOB.OOBPermutedPredictorDeltaError;

figure;
bar(imp);
ylabel('Importancia');
xlabel('Predictores');
h = gca;
h.XTickLabel = mdl_RF_OOB.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% OOB error

plot(sqrt(oobError(mdl_RF_OOB)));
xlabel('Número de árboles');
ylabel('OOB RMSE');

err = oobError(mdl_RF_OOB,'Mode','ensemble');
fprintf('RMSE OOB del RF = %4.2f \n\n',sqrt(err));

disp('%%%%%%%%%%%%%%%%%%%%%%%%%% BOOSTING %%%%%%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
rng(4);
mdl_boosted = fitrensemble(X(pos_train,:),Y(pos_train),"Method","Bag",...
    "NumLearningCycles",100,"Learners","tree","LearnRate",0.1);
% No ponemos la D ya que tiene 1 como valor por defecto

% Evaluamos rendimiento en test del árbol bagged 
ypred = predict(mdl_boosted,X(pos_test,:));
MSE = mean((ypred-Y(pos_test)).^2);
fprintf('RMSE (TEST) del boosted tree (B=100) = %4.2f \n\n',sqrt(MSE));
