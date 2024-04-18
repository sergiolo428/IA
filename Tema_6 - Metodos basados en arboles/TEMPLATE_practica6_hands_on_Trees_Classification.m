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


% Ajustar árbol (probar Gini y entropía)

% Predecir usando árbol generado

% Tasas error y acierto


disp('%%%%%%%%%%%%% TODA LA BASE DE DATOS %%%%%%%%%%%%%')
fprintf('Tasa de predicciones correctas = %4.2f%% \n',acierto);
fprintf('Tasa de error = %4.2f%% \n\n',error);

% Visualizamos árbol de clasificación


% No queremos saber la tasa de predicciones correctas en training, estamos
% interesados en el rendimiento del clasificador en test.

% Dividimos, por tanto, la base de datos en train y test
rng(5); % Fijamos semilla para el generado de números aleatorios

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




CV_error=[];
% leafs=[];
for aa = 1:k
    



    
    % Entrenamos árbol
    
    
    % Para cada alpha, ajustamos y evaluamos los modelos
    for bb=1:length(alpha_grid)-1 %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria




    end
    
end


% Visualizamos árbol de clasificación
view(tree_pruned,'Mode','graph')

% Evaluamos rendimiento del árbol de clasificación podado en test


fprintf('Tasa de predicciones correctas (TEST) del árbol podado (alpha=%.3f  nodos terminales=%d) = %4.2f%% \n\n',alpha_grid(pos),sum(~tree_pruned.IsBranchNode),acierto);

% Dibujar error bar


