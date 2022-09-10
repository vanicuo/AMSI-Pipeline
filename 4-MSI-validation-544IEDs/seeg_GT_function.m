function [GT_index]=seeg_GT_function(seeg_channel_name,seeg_individual)
num_channel=numel(seeg_channel_name);
num_individual= numel(seeg_individual);
GT_index=[];
for i=1:num_channel
    for j=1:num_individual
        if strcmp(seeg_channel_name{i},seeg_individual{j})==1
            GT_index=[GT_index,i];
        end
    end
end