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

figure(1);

% Quitar comment para ejecutar

for i=1:num
    for j=1:num

        %%% Otenemos valor posicion
        pos=(i - 1) * num + j;

        %%% Subplot + scatter
        subplot(8,8,pos)
        scatter(Auto2{:,i},Auto2{:,j})

        %%% Primera en mayusculas
        nx=Auto.Properties.VariableNames{i};
        ny=Auto.Properties.VariableNames{j};
        nx(1)=upper(nx(1));
        ny(1)=upper(ny(1));

        %%% Title + labels
        titulo = sprintf('%s en funcion de %s',ny,nx);
        xlabel(nx);ylabel(ny);%title(titulo);        
    end
end   

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

% Apartado 3 -  Ajustar un modelo de regresion lineal múltiple que tenga 
% mpg como respuesta y el resto de variables, excepto name, como predictores.

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

% Analizamos residuos

figure(2)
subplot(121);plotResiduals(mdl,'fitted','Marker','O');
subplot(122);plotResiduals(mdl,'fitted','ResidualType','studentized','Marker','O');


% Analizamos high leverage points

figure(3)
plotDiagnostics(mdl,'leverage','Marker','o');xlabel('Indice fila');ylabel('leverage')

% Analizamos leverage vs studentized residuals

figure(4)
plot(mdl.Diagnostics.Leverage,mdl.Residuals.Studentized,'ko','Color','Red');ylabel('Residuos estudentizados');xlabel('Leverage');


% Vemos que en la figura hay 4 valores que excenden un poco En el analisis
% de residuos

% Sin embargo, un valor que llama mas la atencion es la observacion numero
% 14, la cual excede considerablemente respecto a las otras observaciones

% Apartado 5 -  Introduce términos de interacción en el modelo. 
% ¿Es posible que algún término de interacción sea estadísticamente 
% significativo?

% Introducimos términos de interacción en el modelo

X3;

mdl_1 = fitlm(Auto3{:,2:8},Y,'VarNames',var_names([2:8 1]));

% Introducimos más términos de interacción en el modelo

X3;

mdl_2 = fitlm(Auto3{:,2:8},Y,'VarNames',var_names([2:8 1]));

% Apartado 6 -  Prueba diferentes transformaciones de las variables
% Introducimos términos de interacción en el modelo

end