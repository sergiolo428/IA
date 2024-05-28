function modelo_final

load Xtrain.mat
load Ytrain.mat

Xtrain = Xtrain(:,[1 3 4 9 13 14 15 17 36 46 47]);

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
CV_error=[];C_grid = 10;KS_grid = 1;
k=10;
for i = 1:k

    X_train_CV = x_train(cc.training(i),:);
    X_test_CV = x_train(cc.test(i),:);

    Y_train_CV = y_train(cc.training(i));
    Y_test_CV = y_train(cc.test(i));
      
    % Para cada combinación C - KS ajustamos y evaluamos los modelos
    for j=1:length(C_grid)
        for z=1:length(KS_grid)
            
            mdl_gaus = fitcsvm(X_train_CV,Y_train_CV,"BoxConstraint",C_grid(j),"KernelFunction","gaussian","KernelScale",KS_grid(z));

            label = predict(mdl_gaus,X_test_CV);
            [SE,SP,ACC,BAC] = compute_metrics(label,Y_test_CV);
            CV_error(j,z,i) = 100*(ACC);
            
        end
    end
end

CV_medios = mean(CV_error,3);

[val,pos] = min(CV_medios(:));

[row,col] = ind2sub(size(CV_medios),pos);

fprintf("\nEl acierto es de %4.2f\n",val);

end