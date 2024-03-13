function Ejemplo1

load("Smarket.mat");

% 1250 dias, Empresas mas potentes de estados unidos

% Year - Lag1 - Lag2 - Lag3 - Lag4 - Lag5 - Volume - Today - Direction
% Año muestra - 
% Lag1: Indice El dia antes
% Lag2: Indice dos dias antes
% ...
% Volume: Numero acciones (En miles)
% Today: Rendimiento a dia de hoy
% Direction: Si el indice es esta subiendo o bajando

% ----------------------------------------------------------------------- %

var_names = Smarket.Properties.VariableNames;

% [~,b]=plotmatrix(Smarket{:,1:end-1});
% 
% for i=1:length(var_names)-1
% 
%     axes(b(i,1));ylabel(var_names{i});
%     axes(b(end,i));xlabel(var_names{i});
% 
% 
% end

% Correlacion entre volume y year

corr(Smarket{:,1:end-1});

% Vemos que la unica variable que tiene una correlacion decente es con
% volume

% plot(Smarket.Volume,'ko','Color','Red');title('Volume')

% AJustar modelo de regresion logistica para predecir direction usando:
% Lab1 Lab2 Lab3 Lab4 Lab5 y volume
% Hacemos esto ya que direction es cualitativa

% A la alta 1, a la baja 0

Y(strcmp(Smarket.Direction,'Down'))=0;

Y(strcmp(Smarket.Direction,'Up'))=1;
Y=Y';

var_sel=2:7;

X = Smarket{:,2:7};

% Usar funcion logistica
mdl = fitglm(X,Y,'Distribution','binomial','VarNames',var_names([var_sel end]));

% El estadistico Chi^2 es el equivalente al estadistico f
% Y vemos que ningun valor esta asociado a la respuesta

% Realizamos predicicones

Xtest = Smarket{1:10,var_sel};
Yprob = predict(mdl,Xtest);

% Realizamos predicicones para toda la BBDD

Xtest = Smarket{:,var_sel};
Yprob = predict(mdl,Xtest);

% plot(Yprob);

% Marcamos umbral de decision para decir si es up or down
ypred(Yprob>=0.5)=1;
ypred(Yprob<0.5)=0;

% Calculo de tasa de acierto y la tasa de error
% Usamos matriz de confusion

C = confusionmat(Y,ypred)
confusionchart(C,{'Down (0)','Up (1)'})

% 145 + 507 / Totla observaciones

% Tasa acierto 
acc = 100*(C(1,1)+C(2,2))/length(Y)
err = 100-acc;

fprintf('La tasa de acierto o precision es de %.2f %%\n',acc);
fprintf('La tasa de error o precision es de %.2f %%\n',err);

% Conclusion, el modelo es tan malo que solo e sun poco mejor que tirar una
% moneda al aire :v

%  Realizamos un modelo basado en una tabla de entrenamiento , y usaremos
%  test vara evaluar el rendimiento del modelo

% Se suele usar 60% training 40% test o incluso 70/30

pos_train = Smarket.Year<2005;
pos_test = Smarket.Year == 2005;

Xtrain = Smarket{pos_train,var_sel};
Ytrain = Y(pos_train);

Xtest = Smarket{pos_test,var_sel};
Ytest = Y(pos_test);

mdl = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([var_sel end]));

Yprob=predict(mdl,Xtest);

Ypred(Yprob>=0.5)=0;
Ypred(Yprob<0.5)=1;

C = confusionmat(Ytest,Ypred)
confusionchart(C,{'Down (0)','Up (1)'})

acc = 100*(C(1,1)+C(2,2))/length(Ytest)
err = 100-acc;

fprintf('La tasa de acierto o precision es de %.2f %%\n',acc);
fprintf('La tasa de error o precision es de %.2f %%\n',err);


% Ajustamos el modelo usando los predictores que parecen estar un poco mas
% asociados, Lag1 y Lag2

var_sel=[2 3];

Xtrain = Smarket{pos_train,var_sel};
Xtest = Smarket{pos_test,var_sel};

% Ajustamos modelo

mdl_3 = fitglm(Xtrain,Ytrain,'Distribution','binomial','VarNames',var_names([var_sel end]))

Yprob=predict(mdl_3,Xtest);

Ypred(Yprob>=0.5)=1;
Ypred(Yprob<0.5)=0;

C = confusionmat(Ytest,Ypred)
confusionchart(C,{'Down (0)','Up (1)'})

acc = 100*(C(1,1)+C(2,2))/length(Ytest)
err = 100-acc;

fprintf('La tasa de acierto o precision es de %.2f %%\n',acc);
fprintf('La tasa de error o precision es de %.2f %%\n',err);

% Valor predictivo positivo

PPV = 100*C(2,2)/(C(2,2)+C(1,2))
fprintf('\nValor predictivo positivoes %.2f %%\n',PPV);

% 

end