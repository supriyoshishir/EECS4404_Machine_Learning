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

% Question-3 (1 pts)
% For each learned function (n=1, …, 10), compute the empirical square loss (ERM) on data and plot
% it as a function of n. 
RMSE=[10];

for i=1:10
    p = polyfit(magnesium,color_intensity,i);
    f = polyval(p,magnesium);
%     T = table(magnesium,color_intensity,f,color_intensity-f,'VariableNames',{'X','Y','Fit','FitError'})
    RMSE(i)=sqrt(mean((color_intensity-f).^2));
end
figure()
t=1:10;
bar(t,RMSE);
xlabel('Polynomial','FontSize', 12)
ylabel('ERM','FontSize', 12)
title('Question-3 ERM   vs   Polynomial(1:10)', 'FontSize', 15);
grid on