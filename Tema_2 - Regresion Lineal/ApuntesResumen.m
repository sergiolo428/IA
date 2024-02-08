%% Error estandar --> SE(B0') y SE(B1') indica la cantidad en la que 
% B0' y B1' distan de los valores reales B0 y B1

%% Error estandar 

% Intervalo de confianza 95% B0 -> [B0 - 2SE(B0') , B0 + 2SE(B0')]

% Contraste de hipotesis --> H0: Hipotesis nula B1 = 0
    
    % Para aceptar o rechazar hipotesis:

        % 1. Calculamos estadistico t'
        % 2. Calculamos p-valor de t' p(|t|>=t') = p-valor
        % 3. Dependiendo del nivel de significancia "alpha"
            % p-valor < alpha --> Rechazamos hipotesis
            % p-valor > alpha --> Aceptamos hipotesis

%% Estimando precision de Y'

% 1 Error estandar residual RSE = Raiz((RSS)/(n-2))

% 2 Estadistico R^2 -> [0 - 1]
    % 0 -> No lineal ; 1 -> Altamente lineal

% 3 Correlacion (r) ; R^2 = r^2
