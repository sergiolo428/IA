function Template_T6_E2
% Este script contiene la resolución del ejercicio aplicado 2 del Tema 6
% de la asignatura 'Técnicas de Inteligencia Artificial'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% En este problema usaremos SVC y SVM para predecir si un determinado coche
% tiene alta o baja autonom´ıa en la base de datos Auto.

% Cargamos base de datos

load Auto.mat;

disp('%%%%%%%%%%%%%%%%% EJERCICIO 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN

Auto = rmmissing(Auto);

disp('%%%%%%%%%%%%%%%%% Apartado 1 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 1 -  Crear una variable binaria que tome el valor 1 para coches
% con una autonomía (mpg) superior a la mediana, y que tome el valor 0
% para coches con una autonomía inferior a la mediana.

mediana=median(Auto{:,1})

Y(Auto.mpg>mediana) = 1;
Y(Auto.mpg<=mediana) = 0;

Y=Y';

X = Auto{:,2:8};

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 2 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 2 - Ajustar un SVC a los datos con varios valores de C con el
% objetivo de predecir si un coche tiene alta o baja autonomıa. Reportar
% el error CV asociado a los diferentes valores del parámetro C.
% Comentar los resultados.
rng(1); % Fijamos semilla para el generado de números aleatorios
c = cvpartition(length(Y),'Holdout',0.50); % Partición no estratificada
% 50% train y 50% test
% ¡OJO! -> NECESIDAD DE DEFINIR PREVIAMENTE 'y'

pos_train = c.training;
pos_test = c.test;

X_train = X(pos_train,:);
X_test = X(pos_test,:);

Y_train = Y(pos_train);
Y_test = Y(pos_test);


% Usamos 10-FOLD CV para buscar el C óptimo
rng(2)
k = 10;
cc = cvpartition(length(Y_train),'KFold',k);% ¡OJO! -> NECESIDAD DE DEFINIR PREVIAMENTE 'y1'
% % % C_grid = [0.001,0.01,0.1,1.5,10,100];
% % % 
% % % error_SVM = [];
% % % for i = 1:k
% % % 
% % %     X_train_CV = X_train(cc.training(i),:);
% % %     X_test_CV = X_train(cc.test(i),:);
% % % 
% % %     Y_train_CV = Y_train(cc.training(i));
% % %     Y_test_CV = Y_train(cc.test(i));
% % % 
% % %     for j=1:length(C_grid)
% % % 
% % %     mdl_SVM = fitcsvm(X_train_CV,Y_train_CV,"BoxConstraint",C_grid(j),"KernelFunction","linear");
% % %     label = predict(mdl_SVM,X_test_CV);
% % %     error_SVM(i,j) = 100*(1-sum(label==Y_test_CV)/length(Y_test_CV));
% % % 
% % %     end
% % % 
% % % end
% % % 
% % % [val,pos] = min(mean(error_SVM));
% % % val

% Val 9.2195 ; Pos 2 (C = 0.01)

fprintf('\n')
disp('%%%%%%%%%%%%%%%%% Apartado 3 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
% Apartado 3 - Repetir el apartado anterior, pero esta vez usandos SVMs con
% kernel radial con diferentes valores de gamma

% Usando las mismas particiones de 10-FOLD CV
rng(2)
CV_error=[];C_grid = [0.1,1,10,100,1000];KS_grid = [0.5 1 2 3 4 5 10];
k=10;
for i = 1:k

    X_train_CV = X_train(cc.training(i),:);
    X_test_CV = X_train(cc.test(i),:);

    Y_train_CV = Y_train(cc.training(i));
    Y_test_CV = Y_train(cc.test(i));
      
    % Para cada combinación C - KS ajustamos y evaluamos los modelos
    for j=1:length(C_grid)
        for z=1:length(KS_grid)
            
            mdl_gaus = fitcsvm(X_train_CV,Y_train_CV,"BoxConstraint",C_grid(j),"KernelFunction","gaussian","KernelScale",KS_grid(z));
            
            label = predict(mdl_gaus,X_test_CV);
            CV_error(j,z,i) = 100*(1-sum(label==Y_test_CV)/length(Y_test_CV));
        end
    end
end

CV_medios = mean(CV_error,3);

[val,pos] = min(CV_medios(:));

[row,col] = ind2sub(size(CV_medios),pos);

% val 29.6 ; C 1 ; gamma 10 ;
val
C_grid(row)
KS_grid(col)
keyboard;
