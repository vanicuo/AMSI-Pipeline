function [l1,l2,l3,l4]=compute_sequence_function(fslxyz_mat)
load index_level.mat
% addpath(genpath('/disk2/zhengli/simulation/tool/REST_V1_8'));
[aal_resample,head]=rest_ReadNiftiImage('AAL_resample.nii');
% rmpath(genpath('/disk2/zhengli/simulation/tool/REST_V1_8'));
aal=aal_resample(:);
index_1_90= find(aal>0&aal<=90);
[aal_x,aal_y,aal_z]=ind2sub([182 ,218 ,182],index_1_90);
aal_xyz=[aal_x,aal_y,aal_z];

for i=1:numel(fslxyz_mat)/3
    x=fslxyz_mat(i,1)+1;y=fslxyz_mat(i,2)+1;z=fslxyz_mat(i,3)+1;
    AAL_label= aal_resample(round(x),round(y),round(z));
    
    if AAL_label~=0&&AAL_label<=90
        
        l1(i,1)=rl_index(AAL_label);
        l2(i,1)=labor_index(AAL_label);
        l3(i,1)=medial_lateral_index(AAL_label);
        l4(i,1)=sublabor_index(AAL_label);
    else
       AAL_label_before=AAL_label;
        delta = rownorm(repmat([x,y,z],numel(aal_x),1) -aal_xyz);
        [min_value,min_index]=min(delta);
        AAL_label= aal_resample(aal_xyz(min_index,1),aal_xyz(min_index,2),aal_xyz(min_index,3));
        l1(i,1)=rl_index(AAL_label);
        l2(i,1)=labor_index(AAL_label);
        l3(i,1)=medial_lateral_index(AAL_label);
        l4(i,1)=sublabor_index(AAL_label);       
%         [i,AAL_label_before,  AAL_label,min_value]
        
        
    end
        
        
end
