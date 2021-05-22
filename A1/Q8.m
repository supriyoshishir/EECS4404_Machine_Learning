%% Assignment -1 EECS 4404/5327
% Name: Supriyo Shishir Ghosh
% Student Number: 215318728
% Email: shishir@my.yorku.ca

% LOAD DATASET %
load('wine.mat');

%  LOAD VALUES INTO VARIABLES %
wine=double(A);

wine_label = wine(:, 14);
wine_data = wine(:, 1:13);
% DEFINE ATTRIBUTES AND CLASSES %
categories = {'Alcohol'; 'Malic acid'; 'Ash'; 'Alcalinity of ash'; 'Magnesium'; 'Total phenols'; 'Flavanoids'; 'Nonflavanoid phenols'; 'Proanthocyanins'; 'Color intensitys'; 'Hue'; 'OD280/OD315 of diluted wines'; 'Proline'};
classnumber = 3;

idx = (wine_label > 2);
wine(idx,:) = [];
wine_label = wine(:, 14);
wine_data = wine(:, 1:13);

% Question-8 (1 bonus pts)
% Fit the data with a Perceptron classifier and compare the loss with respect to your fitted lines
% (question-3)
magnesium=double(magnesium);
color_intensity=double(color_intensity);
hue=double(hue);

mag_norm=magnesium / max(magnesium);
color_intensity=color_intensity / max(color_intensity);
hue_norm=hue / max(hue);

input = [mag_norm, color_intensity, hue_norm];

numIn = 130;
desired_out = wine_label/max(wine_label);
bias = -1;
coeff = 0.01;
rand('state',sum(100*clock));
weights = -1*2.*rand(4,1);
iterations = 1000;

for i = 1:iterations
     out = zeros(4,1);
     for j = 1:numIn
          y = bias*weights(1,1)+...
               input(j,1)*weights(2,1)+input(j,2)*weights(3,1)+input(j,3)*weights(4,1);
          out(j) = 1/(1+exp(-y));
          delta = desired_out(j)-out(j);
          weights(1,1) = weights(1,1)+coeff*bias*delta;
          weights(2,1) = weights(2,1)+coeff*input(j,1)*delta;
          weights(3,1) = weights(3,1)+coeff*input(j,2)*delta;
          weights(4,1) = weights(4,1)+coeff*input(j,3)*delta;
     end
end

rmse_comp=[RMSE(10),RMSE_surf,delta];
figure()
h=bar(rmse_comp);
xlabel('Predictors','FontSize', 12)
ylabel('ERM','FontSize', 12)
title('Question-8 ERM Surf Fit vs ERM(Poly) vs Perceptron', 'FontSize', 10);
l{2}='Surface'; l{1}='Polynomial';   l{3}='Perceptron';  
set(gca,'xticklabel', l) 
grid on



