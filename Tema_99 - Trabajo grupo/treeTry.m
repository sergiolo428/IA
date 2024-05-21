function treeTry

trainPorcen = 0.5;

load Xtrain.mat;
load Ytrain.mat;

rng(2)
c = cvpartition(size(Ytrain,1),"HoldOut",trainPorcen);
pos_train = c.training;
pos_test = c.test;

x_train = Xtrain(pos_train,:);
y_train = Ytrain(pos_train);

x_test = Xtrain(pos_test,:);
y_test = Ytrain(pos_test);

tree = fitctree(x_train,y_train,...
    'SplitCriterion','gdi');
alpha_grid = tree.PruneAlpha;
view(tree,'Mode','graph')

label = predict(tree,x_test);

fprintf("ERROR UN ARBOL");
acierto = 100*sum(label==y_test)/length(y_test)

%--------------------------------------------------------------------------

rng(2)
k = 10;
c = cvpartition(sum(pos_train),'KFold',k);

X1 = x_train;
Y1 = y_train;

CV_MSE=[];
for aa = 1:k
    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);
    Xtrain = X1(pos_train_CV,:);
    Xtest = X1(pos_test_CV,:);
    Ytrain = Y1(pos_train_CV);
    Ytest = Y1(pos_test_CV);
    
    % Entrenamos árbol
    tree_train = fitctree(Xtrain,Ytrain);
    
    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(alpha_grid)-1 %Si hay M niveles de poda, hay M+1 alphas -> la última no cogemos sería poda completa -> decir clase mayoritaria
        tree2 = prune(tree_train,'Alpha',alpha_grid(bb));
        ypred = predict(tree2,Xtest);
        CV_MSE(aa,bb) = 100*sum(ypred==Ytest)/length(y_test);
    end
    
end
[val,pos] = min(mean(CV_MSE));
fprintf("ERROR UN ARBOL CORTADO");
val
%--------------------------------------------------------------------------

rng(2);
mdl_bagged = TreeBagger(60,x_train,y_train,"Method","classification",...
    "NumPredictorsToSample",7);


label2 = predict(mdl_bagged,x_test);

fprintf("ERROR 100 ARBOLES");
mylabel2 = cellfun(@str2num,label2);
acierto2 = 100*sum(mylabel2==y_test)/length(y_test)

end