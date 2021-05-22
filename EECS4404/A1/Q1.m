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

%Question-1 (0.25 pts)
% Load the data and plot (visualize) the data points of wines by their Alcohol (feature 1 in x axis) and
% Malic acid (feature 2 in y axis). 
alcohol=wine(:,1);
malic_acid=wine(:,2);
c = linspace(1,100,length(wine(:,14)));
figure
gscatter(alcohol,malic_acid,wine_label)
title('Question-1 :Alcohol    vs    Malic Acid', 'FontSize', 12);
xlabel('Alcohol','FontSize', 12)
ylabel('Malic Acid','FontSize', 12)
grid on