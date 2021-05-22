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

% Question-4 (1 pts)
% Now, fix the n=10 and add a lasso regularization for your predictor of data. Vary the regularization
% parameter in a loop of 20 and visualize the RLM loss. You can check the correctness of your solution
% with MALAB’s built-in Lasso.
figure()
grid on

for i=1:20
    p = polyfit(magnesium,color_intensity,i);
    y1 = polyval(p,magnesium);
    [b,fitinfo] = lasso(magnesium,y1,'CV',10);
    lam = fitinfo.Index1SE;
    fitinfo.MSE(lam);
    hold on
    lassoPlot(b,fitinfo,'PlotType','CV');
    str = sprintf('Question-4 Lasso Regularization Cross Validation fold:%d', i);
    title(str, 'FontSize',10);
    hold off
end