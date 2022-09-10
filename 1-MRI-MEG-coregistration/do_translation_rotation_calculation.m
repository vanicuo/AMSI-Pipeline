clear;
close all;
clc;
load('trans_matrix_auto.mat');
for i=1:numel(trans_matrix)   
    R=trans_matrix{i}(1:3,1:3);
    T_auto(:,i)=trans_matrix{i}(:,4);
    x_auto(i) = atan2(R(3,2),R(3,3));  %pitch
    y_auto(i) = atan2(-R(3,1),sqrt(R(3,2)^2+R(3,3)^2)); %roll
    z_auto(i) = atan2(R(2,1),R(1,1)); %yaw
end

clear trans_matrix R
load('trans_matrix_manu.mat');
for i=1:numel(trans_matrix)   
    R=trans_matrix{i}(1:3,1:3);
    T_manu(:,i)=trans_matrix{i}(:,4);
    x_manu(i) = atan2(R(3,2),R(3,3));
    y_manu(i) = atan2(-R(3,1),sqrt(R(3,2)^2+R(3,3)^2));
    z_manu(i) = atan2(R(2,1),R(1,1));
end