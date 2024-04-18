function TEMPLATE_practica6_hands_on_Trees_Classification
% Este script contiene la resolución del tutorial práctico del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

load Carseats;
% Nombre de las variables
var_names=Carseats.Properties.VariableNames

% Dimensiones de la base de datos original
size(Carseats)

disp('%%%%%%%%%%%%%%%%%% ÁRBOLES DE CLASIFICACIÓN %%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Creo variable dicotómica cualitativa High en base a la variable Sales
High(Carseats.Sales>8) = {'Yes'};
High(Carseats.Sales<=8) = {'No'};
Y = High';


% Ajustar Arbol (probar Gini y entropia)
tree = fitctree(Carseats(:,2:end),Y,'CategoricalPredictors',[6,9,10],...
    'SplitCriterion','gdi');


% Predecir usando árbol generado

label = predict(tree,Carseats(:,2:end));

acierto = 100*sum(strcmp(label,Y))/length(Y);
error = 100-acierto

% Tasas error y acierto
% view(tree,'Mode','graph')

disp('%%%%%%%%%%%%% TODA LA BASE DE DATOS %%%%%%%%%%%%%')
fprintf('Tasa de predicciones correctas = %4.2f%% \n',acierto);
fprintf('Tasa de error = %4.2f%% \n\n',error);

% Visualizamos árbol de clasificación


% No queremos saber la tasa de predicciones correctas en training, estamos
% interesados en el rendimiento del clasificador en test.

% Dividimos, por tanto, la base de datos en train y test
rng(5); % Fijamos semilla para el generado de números aleatorios

hpartition = cvpartition(size(Carseats,1),'HoldOut',0.5);

pos_train = hpartition.training;
pos_test = hpartition.test;

tree = fitctree(Carseats(pos_train,2:end),Y(pos_train),'CategoricalPredictors',[6,9,10],...
    'SplitCriterion','gdi');

alpha_grid = tree.PruneAlpha;

label = predict(tree,Carseats(pos_test,2:end));

acierto = 100*sum(strcmp(label,Y(pos_test)))/length(Y(pos_test));
error = 100-acierto

% Partición no estratificada
% 50% train y 50% test


% Entrenamos árbol de clasificación


% Visualizamos árbol de clasificación
view(tree,'Mode','graph')

% Evaluamos rendimiento del árbol de clasificación en test


disp('%%%%%%%%%%%%% TRAIN/TEST %%%%%%%%%%%%%')
fprintf('Tasa de predicciones correctas (TEST) = %4.2f%% \n\n',acierto);


% PODA DEL ÁRBOL
% Usar K-fold CV en los datos de entrenamiento para elegir ALPHA
rng(2)
k = 10;

c = cvpartition(sum(pos_train),'KFold',10);

X1 = Carseats(pos_train,2:end);
Y1 = Y(pos_train);


CV_error=[];
for aa = 1:k
    
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    
    xtrain = X1(pos_train_CV,:);
    xtest = X1(pos_test_CV,:);

    ytrain = Y1(pos_train_CV);
    ytest = Y1(pos_test_CV);



    % Entrenamos árbol

    tree_train = fitctree(xtrain,ytrain,'CategoricalPredictors',[6,9,10]);

    % Para cada alpha, ajustamos y evaluamos los modelos
    for bb=1:length(alpha_grid)-1 %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria
        tree2 = prune(tree_train,'Alpha',alpha_grid(bb));
        label = predict(tree2,xtest);

        CV_error(aa,bb) = 100*(1-sum(strcmp(label,ytest))/length(ytest));
        
    end
end

[val,pos] = min(mean(CV_error))
tree_pruned = prune(tree,'Alpha',alpha_grid(pos));

% Visualizamos árbol de clasificación
view(tree_pruned,'Mode','graph')

% Evaluamos rendimiento del árbol de clasificación podado en test
label = predict(tree_pruned,Carseats(pos_test,2:end));
acierto = 100*sum(strcmp(label,Y(pos_test)))/length(Y(pos_test));
fprintf('Tasa de predicciones correctas (TEST) del árbol podado (alpha=%.3f  nodos terminales=%d) = %4.2f%% \n\n',alpha_grid(pos),sum(~tree_pruned.IsBranchNode),acierto);

% Dibujar error bar

errorbar(alpha_grid(1:end-1),mean(CV_error),std(CV_error));
hold on; plot(alpha_grid(1:end-1),mean(CV_error),'ro');hold off;
xlabel('Alpha');ylabel('CV Error');

