function TEMPLATE_practica6_hands_on_ANN_Classification
% Este script contiene la resolución del tutorial práctico del Tema 6 
% (RNA para clasificación) de la asignatura 'Técnicas de Inteligencia Artificial'

% cargamos base de datos

load Carseats.mat;

disp('%%%%%%%%%%%%%%% REDES NEURONALES ARTIFICIALES %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

dim = size(Carseats);
% Creamos variable dicotómica High en base a la variable Sales
High = zeros(dim(1),1);
High(Carseats.Sales>8) = 1;
Y = High;

% Dicotomizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Carseats{:,7}));
ShelveLoc_med = D(:,3);
ShelveLoc_good = D(:,2);
D = dummyvar(categorical(Carseats{:,10}));
Urban_Y = D(:,1);
D = dummyvar(categorical(Carseats{:,11}));
US_Y = D(:,1);

% Creamos vector predictor, X
X = [Carseats{:,2:6} ShelveLoc_med ShelveLoc_good Carseats{:,8:9} Urban_Y US_Y];



% Inicializamos red con 20 neuronas en la capa oculta

net = patternnet(20);

% Dividir datos en train/validation/test
rng(1);

[trainInd, valInd, testInd] = dividerand(length(Y),0.7,0.15,0.15);

% sigmoide como función de transferencia
net.layers{1}.transferFcn = 'logsig';


% fijar función usada para división de datos
net.divideFcn = 'divideind'; % Especificar la division

% fijar posiciones de datos train/validation/test
net.divideParam.trainInd = trainInd;
net.divideParam.valInd = valInd;
net.divideParam.testInd = testInd;

% estandarizar predictores a media 0 y std 1
net.inputs{1}.processFcns{2} = 'mapstd';
    
% Train ANN
tic
train_net = train(net,X',Y');
toc
% Test FNN 

th = 0.5;
ypred = train_net(X(testInd),:);
yfit = ypred;
yfit(ypred>th)=1;
yfit(ypred<=th)=0;


% Calcular y visualizar tasa de acierto
acierto = 100*sum(yfit==Y(testInd)')/length(Y(testInd));
fprintf('Tasa de predicciones correctas (TEST) = %4.2f%% \n\n',acierto);


% CALCULAR NÚMERO ÓPTIMO DE NEURONAS EN LA CAPA OCULTA
% Usar 5-fold CV en los datos de entrenamiento para elegir ALPHA
rng(2)
k = 5;
c = cvpartition(length(trainInd)+length(valInd),'KFold',k);

% idx para CV

posCV = [trainInd valInd]; posCV = sort(posCV);
X1 = X(posCV,:);
Y1 = Y(posCV);

CV_error=[];N_grid = [5 10 15 20 30 40 50];
for aa = 1:k
    pos_train = find(c.training(aa));
    pos_test_CV = find(c.test(aa));
    % fijar posiciones de train y validation    

    pos_val_CV = pos_train(1:length(pos_test_CV));
    pos_train_CV = pos_train(length(pos_test_CV)+1:end);
    
       
    for bb=1:length(N_grid)
        rng(13);
        % crear red
        
        net = patternnet(N_grid(bb));

        % opciones de red
        net.layers{1}.transferFcn = 'logsig';
        net.divideFcn='divideind';
        net.divideParam.trainInd=pos_train_CV;
        net.divideParam.valInd=pos_val_CV;
        net.divideParam.testInd=pos_test_CV;
        net.inputs{1}.processFcns{2}='mapstd';
        
        % Train ANN
        train_net = train(net,X1',Y1')
        
        % Test ANN
        ypred = train_net(X1(pos_test_CV,:)');
        yfit=ypred;
        yfit(ypred>th)=1;
        yfit(ypred<=th)=0;
        
        % calcular CV error
        CV_error(aa,bb) = 100 - 100*(sum(yfit==Y1(pos_test_CV)')/length(Y1(pos_test_CV)));
        % vaciamos yfit para siguiente iteración
        yfit = [];
        
    end
    
end
% calculamos min de la media del CV_error

[val,pos] = min(mean(CV_error));


% Dibujamos errorbar
errorbar(N_grid,mean(CV_error),std(CV_error));hold on;plot(N_grid,mean(CV_error),'ro');hold off;xlabel('#neuronas');ylabel('Error CV');
pause;

rng(13);
% creamos red con número óptimo de neuronas
net = patternnet(N_grid(pos));

% fijamos opciones de la red
net.layers{1}.transferFcn = 'logsig'; % Sigmoid as transfer function between layers
net.divideFcn='divideind';
net.divideParam.trainInd=trainInd;
net.divideParam.valInd=valInd;
net.divideParam.testInd=testInd;
net.inputs{1}.processFcns{2}='mapstd';
    
% Train ANN
train_net = train(net,X',Y');

% Test FNN 
ypred = train_net(X(testInd,:)');

th = 0.5;
yfit = ypred;
yfit(ypred>th) = 1;
yfit(ypred<=th) = 0;


% calculamos tasa de acierto
acierto = 100*sum(yfit==Y(testInd)')/length(Y(testInd));

fprintf('Tasa de predicciones correctas (TEST) con %d neuronas = %4.2f%% \n\n',N_grid(pos),acierto);