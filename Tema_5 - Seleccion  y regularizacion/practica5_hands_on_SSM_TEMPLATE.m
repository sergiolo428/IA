function practica5_hands_on_SSM_TEMPLATE
% Este script contiene la resolución del tutorial práctico del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

load Hitters.mat

% Nombre de las variables

var_names = Hitters.Properties.VariableNames;

% Dimensiones de la base de datos original

size(Hitters);

% Remover valores NaN

Hitters = rmmissing(Hitters);

% Dimensiones de la base de datos sin valores NaN

size(Hitters);

disp('%%%%%%%%%%%%%%%%%%%%% SELECCIÓN GRADUAL %%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Binarizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Hitters{:,14}));
Leage_A = D(:,1);

D = dummyvar(categorical(Hitters{:,15}));
Division_E = D(:,1);

dummyvar(categorical(Hitters{:,20}));
NewLeage_A = D(:,1);

X = [Hitters{:,1:13} Leage_A Division_E Hitters{:,16:18} NewLeage_A];
Y = Hitters.Salary;

% % % % Selección gradual hacia adelante
% % % opt = statset('Display','iter'); % Opciones de la seleccion gradual
% % % 
% % % [inmodel_fwd, history_fwd] = sequentialfs(@regLIN,X,Y,'direction','forward',...
% % %     'nfeatures',19,'cv','none','options',opt);
% % % 
% % % 
% % % % Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
% % % r2_adj_fwd=zeros(1,size(history_fwd.In,1));
% % % aic=zeros(1,size(history_fwd.In,1));
% % % bic=zeros(1,size(history_fwd.In,1));
% % % for cc=1:size(history_fwd.In,1)
% % %     Xtrain = X(:,history_fwd.In(cc,:));
% % %     mdl = fitlm(Xtrain,Y);
% % % 
% % %     r2_adj_fwd(cc) = mdl.Rsquared.Adjusted;
% % % 
% % %     aic(cc) = mdl.ModelCriterion.AIC;
% % %     bic(cc) = mdl.ModelCriterion.BIC;
% % % end
% % % 
% % % [val,pos] = max(r2_adj_fwd);
% % % [val_aic,pos_aic] = min(aic);
% % % [vaL_bic,pos_bic] = min(bic);
% % % 
% % % 
% % % fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',pos,val);
% % % fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a AIC\n #predictores = %d \n AIC = %4.3f \n\n',pos_aic,val_aic);
% % % fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a BIC\n #predictores = %d \n BIC ajustado = %4.3f \n\n',pos_bic,vaL_bic);
% % % 
% % % subplot(131);plot(r2_adj_fwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
% % % hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
% % %     'MarkerFaceColor','r','MarkerSize',8);hold off;
% % % subplot(132);plot(aic,'o-');xlabel('# predictores');ylabel('AIC');
% % % hold on;line(pos_aic,val_aic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
% % %     'MarkerFaceColor','r','MarkerSize',8);hold off;
% % % subplot(133);plot(bic,'o-');xlabel('# predictores');ylabel('BIC');
% % % hold on;line(pos_bic,vaL_bic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
% % %     'MarkerFaceColor','r','MarkerSize',8);hold off;
% % % pause;close;


% % % %% Selección gradual hacia atrás
% % % opt = statset('Display','iter');
% % % [inmodel_bwd, history_bwd] = sequentialfs(@regLIN,X,Y,'direction','backward',...
% % %     'nfeatures',1,'cv','none','options',opt);
% % % 
% % % 
% % % % Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
% % % r2_adj_bwd=zeros(1,size(history_bwd.In,1));
% % % for cc=1:size(history_bwd.In,1)
% % %     Xtrain = X(:,history_bwd.In(cc,:));
% % %     mdl = fitlm(Xtrain,Y);
% % %     r2_adj_bwd(cc) = mdl.Rsquared.Adjusted;
% % % 
% % % end
% % % 
% % % r2_adj_bwd = flip(r2_adj_bwd);
% % % [val, pos] = max (r2_adj_bwd);
% % % 
% % % 
% % % plot(r2_adj_bwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
% % % hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
% % %     'MarkerFaceColor','r','MarkerSize',8);hold off;
% % % pause;close;
% % % fprintf('\n SG hacia atrás (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',length(r2_adj_bwd)-pos+1,val);



%% Seleccionar el mejor modelo como aquel que menor MSE de CV presente.
%%% Es decir, usaremos todas las particiones menos una para entrenamiento y
%%% seleciconaremos predictores segun el MSE de la particion de test con el
%%% modelo


% Creamos particiones
rng(13);
k = 10;
c = cvpartition(size(Hitters,1),'KFold',k);


CV_MSE=[];
for aa = 1:k
    pos_train = c.training(aa);
    pos_test = c.test(aa);
    
    Xtrain = X(pos_train,:);
    Xtest = X(pos_test,:);
    Ytrain = Y(pos_train);
    Ytest = Y(pos_test);

    % SG hacia adelante 

    opt = statset('Display','iter');
    [inmodel_fwd, history_fwd] = sequentialfs(@regLIN,Xtrain,Ytrain,'direction','forward',...
        'nfeatures',19,'cv','none','options',opt);

    % Evaluamos los modelos seleccionados
    for bb = 1:size(history_fwd.In,1)
        
        Xtrain_1 = Xtrain(:,history_fwd.In(bb,:));
        Xtest_1 = Xtest(:,history_fwd.In(bb,:));
        mdl = fitlm(Xtrain_1,Ytrain);
        CV_MSE(aa,bb) = mean((Ytest-predict(mdl,Xtest_1)).^2)
    end 
    
end

[pos,val] = min(mean(CV_MSE))

plot(mean(CV_MSE,1),'o-');xlabel('# predictores');ylabel('MSE CV');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close;

keyboard

% SG hacia adelante con las 9 caracteristicas hemos visto que son mejores

[inmodel_fwd, history_fwd] = sequentialfs(@regLIN,X,Y,'direction','forward',...
        'nfeatures',9,'cv','none','options',opt);

% Ajustamos modelo para toda la base de datos

predNames = var_names;
predNames(19) = [];

mdl_final = fitlm(X(:,inmodel_fwd),Y,'VarNames',{predNames{inmodel_fwd,var_names{19}}})

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function criterio = regLIN(Xtrain,ytrain)
    % Ajustamos modelo de regresión lineal
    mdl = fitlm(Xtrain,ytrain);
    % Para elegir el modelo de k predictores con mayor R2
    criterio = 1 - mdl.Rsquared.Ordinary;

end