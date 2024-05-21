function tryPCATree

% Cargar datos
load Xtrain.mat;
load Ytrain.mat;


trainPorcen = 0.5;

rng(2)
c = cvpartition(size(Ytrain,1),"HoldOut",trainPorcen);
pos_train = c.training;
pos_test = c.test;

x_train = Xtrain(pos_train,:);
y_train = Ytrain(pos_train);

x_test = Xtrain(pos_test,:);
y_test = Ytrain(pos_test);


X1 = x_train;
X2 = x_test;
Y1 = y_train;
Y2 = y_test;

%%% APLICAMOS PCA %%%

% PCALoaddings son todos los fis
% PCAScores, predictores transformados (son como las Xtrain del total)
% PCAVar --> Varianza

k = 10;
cc = cvpartition(sum(pos_train),'KFold',k);

[PCALoadings,PCAScores,PCAVar,~,explained,mu] = pca(X1);
explained;
cumsum(explained);
k = 10;
CV_MSE_PCR = [];
for aa=1:k
    pos_train_CV = cc.training(aa);
    pos_test_CV = cc.test(aa);
    Ytrain = Y1(pos_train_CV);
    Ytest = Y1(pos_test_CV);
   
    for bb=1:size(Xtrain,2)
        X_PCR_train = PCAScores(pos_train_CV,1:bb);
        X_PCR_test = PCAScores(pos_test_CV,1:bb);

        %mdl_bagged = TreeBagger(60,X_PCR_train,Ytrain,"Method","classification","NumPredictorsToSample",7);

        mdl_gaus = fitcsvm(X_PCR_train,Ytrain,"BoxConstraint",10,"KernelFunction","gaussian","KernelScale",2);

        label2 = predict(mdl_gaus,X_PCR_test);

        %mylabel2 = cellfun(@str2num,label2);
        %acierto2 = 100*sum(mylabel2==Ytest/length(Ytest));

        [SE,SP,ACC,BAC] = compute_metrics(label2,Ytest);
        
        CV_MSE_PCR(aa,bb)=ACC;

    end

end

[val,pos] = min(mean(CV_MSE_PCR));

val


end