function ridgelassoTry


load Xtrain.mat
load Ytrain.mat

trainPorcen = 0.5;

rng(2)
c = cvpartition(size(Ytrain,1),"HoldOut",trainPorcen);
pos_train = c.training;
pos_test = c.test;

x_train = Xtrain(pos_train,:);
y_train = Ytrain(pos_train);

x_test = Xtrain(pos_test,:);
y_test = Ytrain(pos_test);


rng(2);
k = 10;

c = cvpartition(sum(pos_train),'KFold',k);
X1 = x_train;
Y1 = y_train;

lambda_grid = linspace(0.01,100,100);
lambda_grid_LASSO = linspace(0.01,100,100);
CV_MSE=[];CV_MSE_LASSO=[];
for aa = 1:k

    pos_train_CV = c.training(aa);
    pos_test_CV = c.test(aa);

    Xtrain = X1(pos_train_CV,:);
    Xtest  = X1(pos_test_CV,:);
    Ytrain = Y1(pos_train_CV);
    Ytest  = Y1(pos_test_CV);


    % Para cada lambda, ajustamos y evaluamos los modelos
    for bb=1:length(lambda_grid) 
        % RIDGE REGRESSION
        
        B = ridge(Ytrain,Xtrain,lambda_grid(bb),0);
        ypred = B(1)+Xtest*B(2:end);


        CV_MSE(aa,bb) = mean((Ytest-ypred).^2);
        
        % LASSO
        
        [B,FitInfo] = lasso(Xtrain,Ytrain,"Lambda",lambda_grid(bb));
        ypred = FitInfo.Intercept+Xtest*B;
        CV_MSE_LASSO(aa,bb) = mean((Ytest-ypred).^2);
        
    end
    
end

[val,pos] = min(mean(CV_MSE));
[val2,pos2] = min(mean(CV_MSE_LASSO));

subplot(211);plot(lambda_grid,mean(CV_MSE,1));title('RIDGE');
hold on;plot(lambda_grid(pos),val,'ro');hold off;
subplot(212);plot(lambda_grid_LASSO,mean(CV_MSE_LASSO,1));title('LASSO');
hold on;plot(lambda_grid_LASSO(pos2),val2,'ro');hold off;


[B,FitInfo] = lasso(Xtrain,Ytrain,"Lambda",lambda_grid(pos2));


end