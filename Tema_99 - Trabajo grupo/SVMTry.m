function SVMTry

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


rng(2)
k = 10;
cc = cvpartition(length(y_train),'KFold',k);

rng(2)
CV_error=[];C_grid = [0.1,1,10,100,1000];KS_grid = [0.5 1 2 3 4 5 10];
k=10;
for i = 1:k

    X_train_CV = x_train(cc.training(i),:);
    X_test_CV = x_train(cc.test(i),:);

    Y_train_CV = y_train(cc.training(i));
    Y_test_CV = y_train(cc.test(i));
      
    % Para cada combinación C - KS ajustamos y evaluamos los modelos
    for j=1:length(C_grid)
        for z=1:length(KS_grid)
            
            mdl_gaus = fitcsvm(X_train_CV,Y_train_CV,"BoxConstraint",C_grid(j),"KernelFunction","linear");
            %mdl_gaus = fitcsvm(X_train_CV,Y_train_CV,"BoxConstraint",C_grid(j),"KernelFunction","g","KernelScale",KS_grid(z));

            label = predict(mdl_gaus,X_test_CV);
            CV_error(j,z,i) = 100*(1-sum(label==Y_test_CV)/length(Y_test_CV));
        end
    end
end

CV_medios = mean(CV_error,3);

[val,pos] = min(CV_medios(:));

[row,col] = ind2sub(size(CV_medios),pos);

% val 29.6 ; C 1 ; gamma 10 ;
val
C_grid(row)
KS_grid(col)


end