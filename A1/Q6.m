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

% Question-6 (1 pts)
% For your three selected features, fit a surface to your data of a degree 10.
figure()
surffit = fit( [x, y], z, 'poly10','normalize','on' );
plot(surffit, [x,y], z)
xlabel('Color Intensity')
ylabel('Hue')
zlabel('Magnesium')
title('Question-6 Surface Fit with degree 10')
xi = [5;10;7];
yi = [0.7;0.6;0.5];