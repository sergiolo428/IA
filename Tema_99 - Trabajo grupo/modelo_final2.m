function modelo_final2

load Xtrain.mat
load Ytrain.mat

Xtest = [];
Ytest = [];

Xtrain = Xtrain(:,[1 3 4 9 13 14 15 17 36 46 47]);

rng(2)
C_grid = 10; KS_grid = 1;
mdl_gaus = fitcsvm(Xtrain,Ytrain,"BoxConstraint",C_grid,"KernelFunction","gaussian","KernelScale",KS_grid);


label = predict(mdl_gaus,Xtest);
[SE,SP,ACC,BAC] = compute_metrics(label,Ytest);

ACC

end