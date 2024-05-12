function Template_T6_E1
% Este script contiene la resolución del ejercicio aplicado 1 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En la práctica aplicamos un árbol de clasificación a la base de datos 
% Carseats después de convertir la variable respuesta Sales en una variable
% cualitativa. Ahora trataremos de predecir Sales usando árboles de regresión
% y aproximaciones relacionadas, tratando la respuesta como una variable 
% cuantitativa. 

% Cargamos base de datos

load Carseats.mat;

High(Carseats.Sales>8) = {'Yes'};
High(Carseats.Sales<=8) = {'No'};
Y = High';
X = Carseats(:,2:end);

disp('%%%%%%%%%%%%%%%%% EJERCICIO 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 - Divide los datos de manera aleatoria en conjuntos de 
% entrenamiento (50%) y de test (50%). Fijar la semilla para la 
% generación de números pseudo-aleatorios a rng(4).
rng(4); % Fijamos semilla para el generado de números aleatorios

c = cvpartition(length(Carseats{:,1}),"HoldOut",0.5);
pos_train = c.training;
pos_test = c.test;

X_train = X(pos_train,:);
X_test = X(pos_test,:);

Y_train = Y(pos_train);
Y_test = Y(pos_test);

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un árbol de regresión usando el conjunto de 
% entrenamiento. Visualiza el árbol e interpreta los resultados. 
% ¿Cuál es el MSE de test obtenido?
% Entrenamos árbol de regresión

tree = fitctree(X_train,Y_train,'CategoricalPredictors',[6,9,10],...
    'SplitCriterion','gdi');

view(tree,'Mode','graph')

alpha_grid = tree.PruneAlpha;

label = predict(tree,X_test);

acierto = 100*sum(strcmp(label,Y_test))/length(Y_test);
error = 100-acierto

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Utiliza CV para determinar el nivel óptimo de complejidad 
% del árbol. ¿Mejora el MSE de test con la poda del árbol?
rng(2);
k=10;
cc = cvpartition(sum(pos_train),"KFold",k);

error=[];

for i= 1:k

    pos_train_CV = cc.training(i);
    pos_test_CV = cc.test(i);

    X_train_CV = X_train(pos_train_CV,:);
    X_test_CV = X_train(pos_test_CV,:);

    Y_train_CV = Y_train(pos_train_CV);
    Y_test_CV = Y_train(pos_test_CV);
    
    tree_CV = fitctree(X_train_CV,Y_train_CV,'CategoricalPredictors',[6,9,10]);
    
    
    for j=1:length(alpha_grid)-1

        
        tree_pruned = prune(tree_CV,'Alpha',alpha_grid(j));
        label = predict(tree_pruned,X_test_CV);

        error(i,j) = 100*(1-sum(strcmp(label,Y_test_CV))/length(Y_test_CV));

    end
    
end

[val,pos] = min(mean(error));
val
tree_pruned_pos = prune(tree,'Alpha',alpha_grid(pos));
pos
view(tree_pruned_pos,'Mode','graph')

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 4 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 4 - Utiliza bagging para analizar los datos. ¿Cuál es el MSE de 
% test obtenido?  Determina qué variables son las más importantes.
rng(4);

mdl_bagged = TreeBagger(100,X_train,Y_train,"Method","classification"...
    ,'NumPredictorsToSample','all',"OOBPredictorImportance","on");

view(mdl_bagged.Trees{1},'Mode','graph');

label_bagged = predict(mdl_bagged,X_test);

acierto = 100*sum(strcmp(label_bagged,Y_test))/length(Y_test);
error = 100-acierto

mdl_bagged.OOBPermutedPredictorDeltaError

figure;
bar(mdl_bagged.OOBPermutedPredictorDeltaError);
ylabel('Importancia');
xlabel('Predictores');
h = gca;
h.XTickLabel = mdl_bagged.PredictorNames;
h.XTickLabelRotation = 45;
h.TickLabelInterpreter = 'none';

% Podemos ver como las variables 5 y 6 , Price y ShelveLoc son las mas
% relevantes

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 5 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 5 - Utiliza random forests para analizar los datos. ¿Cuál es el 
% MSE de test obtenido?  Determina qué variables son las más importantes. 
% Describe el efecto de m, el número de predictores considerado en cada 
% división, en la tasa de error obtenida.
rng(4);

mdl_RF = TreeBagger(100,X_train,Y_train,"Method","classification"...
    ,'NumPredictorsToSample',4,"OOBPredictorImportance","on");

label_RF = predict(mdl_bagged,X_test);

acierto = 100*sum(strcmp(label_RF,Y_test))/length(Y_test);
error = 100-acierto

% Analizamos el efecto de m usando CV
rng(4);

k = 10;
cc = cvpartition(sum(pos_train),"KFold",k);

error_RF=[];
for i= 1:k

    pos_train_RF = cc.training(i);
    pos_test_RF = cc.test(i);

    X_train_RF = X_train(pos_train_RF,:);
    X_test_RF = X_train(pos_test_RF,:);

    Y_train_RF = Y_train(pos_train_RF);
    Y_test_RF = Y_train(pos_test_RF);
    
    for j=1:10 % m

        
        mdl_RF = TreeBagger(50,X_train_RF,Y_train_RF,"Method","classification"...
    ,'NumPredictorsToSample',j,"OOBPredictorImportance","on");
        label = predict(mdl_RF,X_test_RF);

        error_RF(i,j) = 100*(1-sum(strcmp(label,Y_test_RF))/length(Y_test_RF));

    end
    
end

[val,pos] = min(mean(error_RF));
val
mean(error_RF)
tree_pruned_pos = prune(tree,'Alpha',alpha_grid(pos));
