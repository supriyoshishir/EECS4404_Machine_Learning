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

% Question-5 (0.25 pts)
% Now, add a third feature of Hue to your data and plot the three in a 3D plot.
hue=wine(:,11);
A = [magnesium,color_intensity,hue];
x = A(:,2) ; y = A(:,3) ; z = A(:,1) ;

[xq,yq] = meshgrid(0:.2:5, 0:.2:5);   % change these values to suitable ones for your problem. Maybe use min and max on the x and y data etc.
zq = griddata(x,y,z,xq,yq);    
figure; surf(xq,yq,zq); 
xlabel('Color Intensity')
ylabel('Hue')
zlabel('Magnesium')
title('Question-5 3D Plot of Manesium/Color Intensity/Hue')