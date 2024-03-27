function Practica4


load('Auto.mat');

var_names = Auto.Properties.VariableNames;

% Limpia la bbdd
Auto = rmmissing(Auto);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Estrategia del conjunto de validacion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % % rng(4);
% % % 
% % % % Dividir la base de datos en un 50% 
% % % hpartition = cvpartition(size(Auto,1),"HoldOut",0.5);
% % % 
% % % % Obtenemos las posiciones las cuales dedicamos a training y cuales a test
% % % pos_train = hpartition.training;
% % % pos_test = hpartition.test;
% % % 
% % % % Selecionamos las posiciones de predictor y respuesta
% % % var_sel = 4;
% % % ans_sel = 1;
% % % 
% % % % Generamos X e Y tanto para training como para test
% % % Xtrain = Auto{pos_train,var_sel};
% % % Xtest = Auto{pos_test,var_sel};
% % % 
% % % Ytrain = Auto{pos_train,ans_sel};
% % % Ytest = Auto{pos_test,ans_sel};
% % % 
% % % 
% % % mdl_01 = fitlm(Xtrain,Ytrain);
% % % mdl_02 = fitlm([Xtrain Xtrain.^2],Ytrain);
% % % mdl_03 = fitlm([Xtrain Xtrain.^2 Xtrain.^3],Ytrain);
% % % 
% % % 
% % % MSE_01 = mean((Ytest-predict(mdl_01,Xtest)).^2);
% % % MSE_02 = mean((Ytest-predict(mdl_02,[Xtest Xtest.^2])).^2);
% % % MSE_03 = mean((Ytest-predict(mdl_03,[Xtest Xtest.^2 Xtest.^3])).^2);
% % % 
% % % fprintf('\n MSE de validacion (LINEAL) = %4.2f \n',MSE_01)
% % % fprintf('\n MSE de validacion (CUADRATICO) = %4.2f \n',MSE_02)
% % % fprintf('\n MSE de validacion (CUBICO) = %4.2f \n',MSE_03)

% NOTA: Observamos que al modificar el rng, varia considerablemente el MSE

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Leave one cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % tic
% % % 
% % % c = cvpartition(size(Auto,1),'LeaveOut');
% % % 
% % % X = Auto.horsepower;
% % % Y = Auto.mpg;
% % % 
% % % CVMSE = crossval('mse',X,Y,'Predfun',@RLIN,'partition',c);
% % % 
% % % fprintf('\n MSE de validacion (LINEAL) = %4.2f \n',CVMSE);
% % % 
% % % for cc = 2:4
% % % 
% % %     X = [X  Auto.horsepower.^cc];
% % %     CVMSE = crossval('mse',X,Y,'Predfun',@RLIN,'partition',c);
% % %     fprintf('\n MSE de validacion (Polinomio orden %d) = %4.2f \n',cc,CVMSE);
% % % end
% % % toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Fold cross validation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % % rng(17);
% % % tic
% % % 
% % % c = cvpartition(size(Auto,1),'KFold');
% % % 
% % % X = Auto.horsepower;
% % % Y = Auto.mpg;
% % % 
% % % CVMSE = crossval('mse',X,Y,'Predfun',@RLIN,'partition',c);
% % % 
% % % fprintf('\n MSE de validacion (LINEAL) = %4.2f \n',CVMSE);
% % % 
% % % for cc = 2:4
% % %     X = [X  Auto.horsepower.^cc];
% % %     CVMSE = crossval('mse',X,Y,'Predfun',@RLIN,'partition',c);
% % %     fprintf('\n MSE de validacion (Polinomio orden %d) = %4.2f \n',cc,CVMSE);
% % % end
% % % toc

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Bootstrap
% % % load Portfolio.mat;
% % % rng(13);
% % % keyboard
% % % 
% % % 
% % % d = 1000;
% % % bootstat = bootstrp(d,@compute_statistic,Portfolio{:,:});
% % % 
% % % fprintf('Media (SD) de alpha = %.3f (%.3f)',mean(bootstat),std(bootstat));


%%

% % % rng(37);
% % % 
% % % load Auto.mat
% % % d = 1000;
% % % X = Auto.horsepower;
% % % Y = Auto.mpg;
% % % 
% % % bootstat = bootstrp(d,@ajustar_RLIN,[Y X]);
% % % fprintf('Media B0: %.3f (%.3f)',mean(bootstat(:,1)),std(bootstat(:,1)));
% % % fprintf('Media B1: %.3f (%.3f)',mean(bootstat(:,2)),std(bootstat(:,2)));
% % % 
% % % mdl = fitlm(X,Y);
% % % mdl.Coefficients.Estimate';
% % % mdl.Coefficients.SE';

rng(37);

load Auto.mat
d = 1000;
X = Auto.horsepower;
Y = Auto.mpg;

bootstat = bootstrp(d,@ajustar_RLIN,[Y X X.^2]);
fprintf('Media B0: %.3f (%.3f)\n',mean(bootstat(:,1)),std(bootstat(:,1)));
fprintf('Media B1: %.3f (%.3f)\n',mean(bootstat(:,2)),std(bootstat(:,2)));
fprintf('Media B2: %.3f (%.3f)\n',mean(bootstat(:,3)),std(bootstat(:,3)));

mdl = fitlm([X X.^2],Y);
mdl.Coefficients.Estimate'
mdl.Coefficients.SE'

end

function yfit = RLIN(Xtrain,Ytrain,Xtest)

mdl = fitlm(Xtrain,Ytrain);
yfit = predict(mdl,Xtest);

end

function alpha = compute_statistic(samples)

X = samples(:,1);
Y = samples(:,2);

covM = cov(X,Y);

alpha = (var(Y) - covM(2,1))/(var(X)+var(Y)-2*covM(2,1));
mdl.Coefficients.Estimate';
mdl.Coefficients.Estimate';

end 

function coef = ajustar_RLIN(samples)

Y = samples(:,1);
X = samples(:,2:end);

mdl = fitlm(X,Y);

coef = mdl.Coefficients.Estimate';

end
