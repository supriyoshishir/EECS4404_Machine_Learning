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

% Question-2 (1 pts)
% Pick Magnesium and Color intensity as your two features and for degrees n =1, …, 10 fit a polynomial
% of degree n to your data. Plot those fitting lines on the data. You can check the correctness of your
% solution with MALAB’s built-in curve fitting function.
magnesium=wine(:,5);
color_intensity=wine(:,10);
figure('Renderer', 'painters', 'Position', [0 0 900 900])

xlabel('Magnesium','FontSize', 8)
ylabel('Color Intensity','FontSize', 8)
grid on

for i=1:10
    p = polyfit(magnesium,color_intensity,i);
    y1 = polyval(p,magnesium);
    B = [magnesium, y1];
    C = sortrows(B,1);
    
    subplot(5,2,i);
    hold on
    gscatter(magnesium,color_intensity,wine_label)
    plot(C(:,1),C(:,2),'k','LineWidth',3);
    hold off
    str = sprintf('Question-2 Polynomial:%d', i);
    title(str, 'FontSize', 8);
end