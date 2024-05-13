function a = firstTry(trainPorcen)

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

mdl = fitlm(x_train,y_train);

yprob = predict(mdl,x_test);
ypred(yprob>=0.51)=1;
ypred(yprob<0.51)=0;

ypred = ypred';

[SE,SP,ACC,BAC] = compute_metrics(ypred,y_test);

fprintf("\nACC: %.5f ",ACC)

end