function correlacionTry

load Xtrain.mat
load Ytrain.mat

Xtrain;

n = 256; % Número de colores en el colormap
quarter_n = n / 4;

% Modificar la interpolación para un desvanecimiento más rápido
blues = [linspace(0, 1, quarter_n)', linspace(0, 1, quarter_n)', linspace(1, 1, quarter_n)']; % De azul a blanco rápidamente
whites = [linspace(1, 1, quarter_n)', linspace(1, 1, quarter_n)', linspace(1, 1, quarter_n)']; % Blanco
reds = [linspace(1, 1, quarter_n)', linspace(0, 1, quarter_n)', linspace(0, 1, quarter_n)']; % De blanco a rojo rápidamente

% Combinar los colormaps
cmap = [blues; whites; flipud(reds)]; % Azul a blanco, luego blanco a rojo

corte = 0.95;
tablacorr = corr(Xtrain);

h = heatmap(tablacorr)
colormap(cmap)
h.ColorLimits = [-1,1]


[row, col] = find(abs(tablacorr)>corte & abs(tablacorr) <1);

pairs = [];
for k = 1:length(row)
    if row(k) < col(k)
        pairs = [pairs; row(k), col(k)];
    end
end

pairs'

% figure(1)
% num=size(pairs,1);
% 
% for i=1:num
% 
%     %%% Otenemos valor posicion
%     pos=i;
% 
%     %%% Subplot + scatter
%     subplot(4,4,pos)
%     scatter(Xtrain(pairs(i,1),:),Xtrain(pairs(i,2),:))
% 
%     %%% Title + labels
%     titulo = sprintf('%d vs %d',pairs(i,1),pairs(i,2));
%     title(titulo);
% end 

rng(22)
model = TreeBagger(300, Xtrain, Ytrain,"Method","classification","NumPredictorsToSample","all",'OOBPredictorImportance', 'On');
importance = model.OOBPermutedPredictorDeltaError;

figure(2)
bar(importance);
ylabel('Importancia');
xlabel('Predictores');

b = corr(Xtrain,Ytrain);
find(abs(b)>0.6);

[sorted_importance, idx] = sort(importance, 'descend');

% Seleccionar los 20 primeros predictores más importantes
top_20_importance = sorted_importance(1:20);
top_20_idx = idx(1:20);

% Obtener los nombres de los predictores correspondientes
predictor_names = model.PredictorNames(top_20_idx);

% Graficar
figure;
bar(top_20_importance);
xticks(1:20);
xticklabels(predictor_names);
xtickangle(45); % Rotar los nombres de los predictores para mayor legibilidad
ylabel('Importancia');
title('Top 20');



% Predictores Importantes:

% Muy:
% 7 - 13

% Decente:
% 8 - 11

% Normal:
% 3 - 22

% Justo:
% 1 - 14 - 44 - 47

rng(22)
%Xtrain = Xtrain(:,[1 3 7 8 11 13 14 22 44 47]);
%Xtrain = Xtrain(:,[3 7 8 11 13 22]);
c = cvpartition(length(Ytrain),"HoldOut",0.25);

pos_train = c.training;
pos_test = c.test;

x_train = Xtrain(pos_train,:);
x_test = Xtrain(pos_test,:);

y_train = Ytrain(pos_train);
y_test = Ytrain(pos_test);

k=15;
rng(22)
cc = cvpartition(sum(pos_train),"KFold",k);

num = 5;
rng(22)
mat = [];
for aa = 1:k
    
    x_train_CV = x_train(cc.training(aa),:);
    x_test_CV = x_train(cc.test(aa),:);

    y_train_CV = y_train(cc.training(aa),:);
    y_test_CV = y_train(cc.test(aa),:);

    %for bb = 1:num

        rng(22)
        model = TreeBagger(100, x_train_CV, y_train_CV,"Method","classification","NumPredictorsToSample","all");
        
        label = predict(model, x_test_CV);
        mylabel = cellfun(@str2num,label);
        [SE,SP,ACC,BAC] = compute_metrics(mylabel,y_test_CV);
        mat(aa) = ACC;
    %end

end
fprintf("CV:")
totalacc = mean(mat)

rng(22)
model2 = TreeBagger(100, x_train_CV, y_train_CV,"Method","classification","NumPredictorsToSample","all");

fprintf("Test:")
label = predict(model2, x_test);
mylabel = cellfun(@str2num,label);
[SE,SP,ACC,BAC] = compute_metrics(mylabel,y_test);
ACC

end