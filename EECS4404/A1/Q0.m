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

% Question-0 (Preprocessing) %
%Remove all row corresponding to the labeled winery 3. After this process,
%you should have only 2 %
idx = (wine_label > 2);
wine(idx,:) = [];
wine_label = wine(:, 14);
wine_data = wine(:, 1:13);