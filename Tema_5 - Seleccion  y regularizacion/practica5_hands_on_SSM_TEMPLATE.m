function practica5_hands_on_SSM_TEMPLATE
% Este script contiene la resolución del tutorial práctico del Tema 5
% de la asignatura 'Técnicas de Inteligencia Artificial'

load 
% Nombre de las variables


% Dimensiones de la base de datos original


% Remover valores NaN


% Dimensiones de la base de datos sin valores NaN


disp('%%%%%%%%%%%%%%%%%%%%% SELECCIÓN GRADUAL %%%%%%%%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');

% Binarizo las variables cualitativas League, Division y NewLeague
D = dummyvar(categorical(Hitters{:,14}));



% Selección gradual hacia adelante



% Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
r2_adj_fwd=zeros(1,size(history_fwd.In,1));
aic=zeros(1,size(history_fwd.In,1));
bic=zeros(1,size(history_fwd.In,1));
for cc=1:size(history_fwd.In,1)



end



fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',pos,val);
fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a AIC\n #predictores = %d \n AIC = %4.3f \n\n',pos_aic,val_aic);
fprintf('\n SG hacia adelante (sequentialfs) mejor modelo en base a BIC\n #predictores = %d \n BIC ajustado = %4.3f \n\n',pos_bic,val_bic);

subplot(131);plot(r2_adj_fwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
subplot(132);plot(aic,'o-');xlabel('# predictores');ylabel('AIC');
hold on;line(pos_aic,val_aic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
subplot(133);plot(bic,'o-');xlabel('# predictores');ylabel('BIC');
hold on;line(pos_bic,val_bic,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close;


% Selección gradual hacia atrás



% Seleccionar el mejor modelo como aquel que menor R2 ajustado presente
r2_adj_bwd=zeros(1,size(history_bwd.In,1));
for cc=1:size(history_bwd.In,1)



end


plot(r2_adj_bwd,'o-');xlabel('# predictores');ylabel('R^2 ajustado');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close;
fprintf('\n SG hacia atrás (sequentialfs) mejor modelo en base a R2 ajustado\n #predictores = %d \n R2 ajustado = %4.3f \n\n',length(r2_adj_bwd)-pos+1,val);



% Seleccionar el mejor modelo como aquel que menor MSE de CV presente.


% Creamos particiones
rng(13);



CV_MSE=[];
for aa = 1:k

    
    % SG hacia adelante 


    % Evaluamos los modelos seleccionados


end

plot(mean(CV_MSE,1),'o-');xlabel('# predictores');ylabel('MSE CV');
hold on;line(pos,val,'LineStyle','none','Marker','o','MarkerEdgeColor','r',...
    'MarkerFaceColor','r','MarkerSize',8);hold off;
pause;close;

% SG hacia adelante 


% Ajustamos modelo para toda la base de datos



end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% USER FUNCTIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function criterio = regLIN(Xtrain,ytrain)
    % Ajustamos modelo de regresión lineal
    
    % Para elegir el modelo de k predictores con mayor R2
    
end