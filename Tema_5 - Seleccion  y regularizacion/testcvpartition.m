rng(13);
a = 1:12;

c = cvpartition(length(a),"HoldOut",0.5);

a_train = a(c.training)
a_test = a(c.test);

cc = cvpartition(sum(c.training),"KFold",3);

a_train(cc.training(1))

a_train(cc.test(1))