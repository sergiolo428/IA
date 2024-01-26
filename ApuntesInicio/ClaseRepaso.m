function ClaseRepaso

% Crear vectores


x1 = [1 2 3 4 5];

% Vector columna

x1 = x1';
% o 
x2 = [1;2;3;4;5];

% Obtener longitud

lenx1 = length(x1); % Funciolna tanto para columna como fila

% Sumar vectores

c = x1 + x2;

%% Matrices

A = [1 2 3; 4 5 6];% ; separa las filas

% DImensiones de matriz

sizeA = size(A); % [Filas, Columnas] | OJO los arrays empiezan en 1!!!!


%% Operar con vectore sy matrices
% Raiz cuadrada de cada posicion

B = sqrt(A);

% Elevar la raiz completa

d = x1*x1';

% Elevar cada valor

x1.^2;

% Crear vectores aleatorios

randn(1,20); % Vector de (filas, columnas) \ distribucion estandar

x3 = randn(1,20);


x4 = randn(1,20);


% Establecer semilla

rng(13);
x3 = randn(1,20);
rng(13);
x4 = randn(1,20);


% Corelacion entre dos vectores

corr(x3',x4'); % Ojo, siempre hay que pasarlos en forma de columna
% ^^ Nos da 1 ya que hemos generado los dos vectores ocn la misma semilla
% por lo que son iguales


rng(22);
x5 = randn(1,20);

b =mean(x5); % media

c = std(x5); % Desviacion

e = var(x5); % Varianza


%% Graficso 2D

y6 = randn(1,100);


%plot(y6);title('Señal');xlabel('Muestras');ylabel('y6');


x=2:2:200; % Desde 2 en saltos de 2 hasta 200

%plot(x,y6);title('Señal');xlabel('Muestras');ylabel('y6');



x6 = randn(1,100);
x7 = randn(1,100);
myBlue = [0 0.4470 0.7410];
%plot(x6,x7,'o','MarkerEdgeColor','r','MarkerFaceColor',myBlue,'MarkerSize',10); % Usamos o para indicar que es un diagrama de dispersion, no une los puntos


% MarkerEdgeColor --> Borde de los puntos
% MarkerFaceColor --> Color 
% MarkerSize --> Tamaño de los indicadores

% Funcion scatter | e slo mismo que lo anteriore pero no s  permiten mas
% cosas

%scatter(x6,x7,8,"red","+");

%% Grafico 3D

t = linspace(-10,10,1000); % (Origen,Destino, CantidadDeSaltos)
x = exp(-t/10).*sin(5*t);
y = exp(-t/10).*cos(5*t);

%plot3(x,y,t);xlabel('x');ylabel('y');zlabel('t');

% Superficies (una variable depende de dos independientes)

% COntorno 2D
x = -1:0.01:0.99;
y = -1:0.01:0.99;
[X,Y] = meshgrid(x,y); % Crea dos matrices lenx * lenx repitiendo el array x lenx veces, hace lo mismo con la y
Z = real(sqrt(1-X.^2-Y.^2));
%contour3(X,Y,Z,50);
%contour(X,Y,Z,50); % Visto desde arriba


%% Indexacion 

A = rand(10,10);% Distribucion uniforme, todos los valores con misma prob de salir, (filas, columnas)

b = A(2,2); % Extraemos la posicion fila 2 columna 2

c = A(1,:); % Extraemos fila 1 completa

d = A(:,5); % Extraemos la columna 5

e = A(1:3, 5:7);

w = A([8 10],[1 3]); % Fila 8 y fila 10, columna 1 y 3

%% Bases de datos


load('Auto.mat');

mpg = Auto{:,1}; % E suna tabla, usamos llaves

cilindros = Auto{:,2};

%scatter(cilindros,mpg);xlabel('Cilindros');ylabel('Mpg'); % Ver relaicon entre dos variables

%histogram(mpg);% Ver la distribucion de coches en uan variable
mean(mpg); 
median(mpg);


%boxplot(mpg); % Lo rojo indica la mediana es decir el 50% estar por encima 
% y el 50% por debajo, las azules horizontales indican los percentiles 25 y 75
% Las lineas extremas negras indican valores extremos

prctile(mpg,50); % Es la mediana

var_names = Auto.Properties.VariableNames; % Ver nombres de las variables en la tabla

fprintf('La media de mpg es %s es de: %.2f',var_names{1},mean(mpg));

for i =1: length(var_names)
    fprintf('La media de mpg es %s es de: %.2f\n',var_names{1},mean(Auto{:,i}));
end


marcas = Auto{:,end};

p = sum(strcmp(marcas,'chebrolet chevelle malibu'));

length(find(cilindros==4));