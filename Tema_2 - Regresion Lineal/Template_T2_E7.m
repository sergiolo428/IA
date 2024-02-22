function Template_T2_E7
% Este script contiene la resolución del ejercicio aplicado 7 del Tema 2
% de la asignatura 'Técnicas de Inteligencia Artificial'


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Tema 2 parte 2

% Cargar base de datos

load('Auto.mat');

fprintf('\n\n')
disp('%%%%%%%%%%%%%%%%% EJERCICIO 7 %%%%%%%%%%%%%%%%%');
disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%');
fprintf('\n\n')


% Remover valores NaN

var_names = Auto.Properties.VariableNames;

Auto2 = Auto;
Auto2(isnan(Auto2.horsepower),:)=[];

% Apartado 1 - Producir una matriz con las diferentes gráficas de dispersión 
% (scatterplots) para las diferentes variables de la base de datos. 

num = 8;

% Quitar comment para ejecutar
% figure(1);
% for i=1:num
%     for j=1:num
% 
%         %%% Otenemos valor posicion
%         pos=(i - 1) * num + j;
% 
%         %%% Subplot + scatter
%         subplot(8,8,pos)
%         scatter(Auto2{:,i},Auto2{:,j})
% 
%         %%% Primera en mayusculas
%         nx=Auto.Properties.VariableNames{i};
%         ny=Auto.Properties.VariableNames{j};
%         nx(1)=upper(nx(1));
%         ny(1)=upper(ny(1));
% 
%         %%% Title + labels
%         titulo = sprintf('%s en funcion de %s',ny,nx);
%         xlabel(nx);ylabel(ny);%title(titulo);        
%     end
% end   

% [a,b,c,d,e]=plotmatrix(Auto2{:,1:8});
% for k=1:8
%     axes(b(8,k));xlabel(var_names{i});
%     axes(b(k,k));xlabel(var_names{i});

% Apartado 2 - Calcular la matriz de correlación entre las diferentes variables. 
% Únicamente excluir la variable name, única no numérica.
Auto3 = Auto2(:,[1:8]);
num=8;

%// TO DO

% for i = 1:num
%     for j = 1:num
%         pos=(i - 1) * 5 + j;
%         
%     end
% end

corr(Auto2{:,1:8});

% Apartado 3 -  Ajustar un modelo de regresion lineal múltiple que tenga 
% mpg como respuesta y el resto de variables, excepto name, como predictores.
Auto3(14,:)=[];
Y = Auto3.mpg;

mdl = fitlm(Auto3{:,2:8},Y,'VarNames',var_names([2:8 1]));

% a)

% Si, simplemente mirando el eestadistico F, vemos que su p-valor es menor
% que 0.05 

% b)

% Podemos destacar Displacement, Weight, Year y Origin como las variables
% con mas relacion ya que su p-valor esta por debajo de 0.05

% c)

%// TO DO

% Apartado 4 -  Visualiza gráficas de residuos y de influencia (leverage) y comenta los posibles
% problemas que existan con el ajuste por mínimos cuadráticos. ¿Los gráficos de
% residuos sugieren la presencia de valores atípicos inusualmente grandes? ¿El gráfico de
% influencia sugiere la presencia de observaciones con influencia (leverage) inusualmente
% alta?

% Analizamos residuos --> Y

% figure(2)
% subplot(121);plotResiduals(mdl,'fitted','Marker','O');
% subplot(122);plotResiduals(mdl,'fitted','ResidualType','studentized','Marker','O');

% Analizamos high leverage points --> X

% figure(3)
% plotDiagnostics(mdl,'leverage','Marker','o');xlabel('Indice fila');ylabel('leverage')

% Analizamos leverage vs studentized residuals

% figure(4)
% plot(mdl.Diagnostics.Leverage,mdl.Residuals.Studentized,'ko','Color','Red');ylabel('Residuos estudentizados');xlabel('Leverage');


% Vemos que en la figura hay 4 valores que excenden un poco En el analisis
% de residuos

% Sin embargo, un valor que llama mas la atencion es la observacion numero
% 14, la cual excede considerablemente respecto a las otras observaciones


% Apartado 5 -  Introduce términos de interacción en el modelo. 
% ¿Es posible que algún término de interacción sea estadísticamente 
% significativo?

% Introducimos términos de interacción en el modelo
XX = Auto3.year.*Auto3.year;
X3 = Auto3.cylinders.*Auto3.displacement;
X4 = Auto3.weight.*Auto3.weight;
X5 = Auto3.displacement.*Auto3.displacement;
X6 = Auto3.horsepower.*Auto3.displacement;
X7 = Auto3.horsepower.*Auto3.weight;
X8 = Auto3.horsepower.*Auto3.horsepower;
X9 = Auto3.acceleration.*Auto3.acceleration;

% Superar R^2 = 0.869 y RSE = 2.86

mdlX = fitlm([Auto3{:,2:8},XX],Y,'VarNames',{var_names{[2:8]},'XX',var_names{1}});

mdl_1 = fitlm([Auto3{:,2:8},X3],Y,'VarNames',{var_names{[2:8]},'Cylinders*Displacement',var_names{1}});

mdl_2 = fitlm([Auto3{:,2:8},X4],Y,'VarNames',{var_names{[2:8]},'Weight^2',var_names{1}});

mdl_3 = fitlm([Auto3{:,2:8},X3,X4],Y,'VarNames',{var_names{[2:8]},'Cylinders*Displacement','Weight^2',var_names{1}});

mdl_4 = fitlm([Auto3{:,2:8},X3,X4,X5],Y,'VarNames',{var_names{[2:8]},'Cylinders*Displacement','Weight^2','Displacement^2',var_names{1}});

mdl_5 = fitlm([Auto3{:,2:8},X6,X4,X5],Y,'VarNames',{var_names{[2:8]},'Horsepower*Displacement','Weight^2','Displacement^2',var_names{1}});
% Quitamos displacement^2 , no aporta nada
% Mantener Horsepower*Displacement ; Weight^2
mdl_6 = fitlm([Auto3{:,2:8},X6,X4,X3],Y,'VarNames',{var_names{[2:8]},'Horsepower*Displacement','Weight^2','Cylinders*Displacement',var_names{1}});
% Quitamos Cylinder*Displacement
% Añadimos Horsepower*Weight
mdl_7 = fitlm([Auto3{:,2:8},X6,X4,X7],Y,'VarNames',{var_names{[2:8]},'Horsepower*Displacement','Weight^2','Horsepower*Weight',var_names{1}});
% Quitamos Horsepower*Weight
% Añadimos Horsepower^2
mdl_8 = fitlm([Auto3{:,2:8},X6,X4,X8],Y,'VarNames',{var_names{[2:8]},'Horsepower*Displacement','Weight^2','Horsepower^2',var_names{1}});
% Quitamos Horsepower^2
% Añadimos Acceleration^2
mdl_9 = fitlm([Auto3{:,2:8},X6,X4,X9],Y,'VarNames',{var_names{[2:8]},'Horsepower*Displacement','Weight^2','Acceleration^2',var_names{1}});
mdl_9

% R^2 = 0.869 y RSE = 2.86
% Superado con: R^2 = 0.87 y RSE = 2.85

% Apartado 6 -  Prueba diferentes transformaciones de las variables
% Introducimos términos de interacción en el modelo

end