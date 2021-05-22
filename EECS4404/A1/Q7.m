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

surffit = fit( [x, y], z, 'poly10','normalize','on' );
plot(surffit, [x,y], z)
xlabel('Color Intensity')
ylabel('Hue')
zlabel('Magnesium')
xi = [5;10;7];
yi = [0.7;0.6;0.5];

% Question-7 (0.5 pts)
% Compare the ERM loss of your surface (question 6) and line (question 3) predictors. 
se=surffit(xi,yi);
RMSE_surf=sqrt(mean(se.^2))/100;
rmse_comp=[RMSE_surf,RMSE(10)];
figure()
h=bar(rmse_comp);
xlabel('Predictors','FontSize', 12)
ylabel('ERM','FontSize', 12)
title('Question-7  ERM Surf Fit   vs   ERM(Poly) at degree=10', 'FontSize', 10);
l{1}='Surface'; l{2}='Polynomial';   
set(gca,'xticklabel', l) 
grid on