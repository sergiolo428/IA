function regresionLinealMultiple2

load('Carseats.mat');

var_names = Carseats.Properties.VariableNames;

% Definimos matriz identidad de dimensiones carseats

% Por medio de matriz de terminos

T = eye(size(Carseats,2));

T(1,1)=0;
TI_1 = zeros(1,size(T,2));
TI_1([3 4])=1; % Income*advertaising
TI_2 = zeros(1,size(T,2));

TI_2([6 8])=1%Interaccion entre price*age
T = [T;TI_1;TI_2];

% Las 11 primeras filas indican que predictores se van a usar para el
% modelo, las filas que tengan un valor 1 en la diagonal principal seran
% las que se van a usar

% En las siguietnes filas se indican lso terminos de interaccion, estos se
% definen colocando un 1 en las variables a interaccionar


mdl = fitlm(Carseats,T,'CategoricalVars',[7 10 11]);


% Notacion de Wilkinson

mdl = fitlm(Carseats,'Sales~CompPrice+Income+Advertising+Population+Price+ShelveLoc+Age+Education+Urban+Income:Advertising+Price:Age','CategoricalVars',[7 10 11]);

end