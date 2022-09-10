clc;clear;close all;
home_dir = pwd;
path = [home_dir,'/results_PKU'];
path_GT = [home_dir,'/electrode_coor/'];
subject_name={'sub01','sub02','sub03'...
              'sub04','sub05','sub06','sub07',...
              'sub08','sub09','sub10',...
              'sub11','sub12','sub13'};     
method_name={'wmne','sLoreta','dspm','beam','dipolefit','MUSIC','fvestal','STOUT'};
source_type = 'first';  % another option:'first','center'
for method_num=1:numel(method_name)  
    tempall = [];
    for sub_num=1:numel(subject_name)  
        
        load([path,'/spike_mni_coordinate_',method_name{method_num},'_',subject_name{sub_num},'.mat'])
        load([path_GT,'seeg_',subject_name{sub_num},'.mat']);
        fslxyz_mat=seeg_fslxyz(seeg_index_GT,:);
        [l1,l2,l3,l4]=compute_sequence_function(fslxyz_mat);
        sequence_GT=[l1,l2,l3,l4];
        
        switch source_type
            case 'center'    
                [s1c,s2c,s3c,s4c]=compute_sequence_function(final_fsl_center);
                sequence_spike_center=[s1c,s2c,s3c,s4c];
                [l1_num(sub_num),l2_num(sub_num),l3_num(sub_num),l4_num(sub_num),all_num(sub_num)]=compare_concordance_function(sequence_GT,sequence_spike_center);
            case 'first'
                [s1f,s2f,s3f,s4f]=compute_sequence_function(final_fsl_first);
                sequence_spike_first=[s1f,s2f,s3f,s4f];  
                 [l1_num(sub_num),l2_num(sub_num),l3_num(sub_num),l4_num(sub_num),all_num(sub_num)]=compare_concordance_function(sequence_GT,sequence_spike_first);
        end
        
        level1(method_num)=l1_num(sub_num)/all_num(sub_num);
        level2(method_num)=l2_num(sub_num)/all_num(sub_num);
        level3(method_num)=l3_num(sub_num)/all_num(sub_num);
        level4(method_num)=l4_num(sub_num)/all_num(sub_num);  
%       subject_name{sub_num}
%       [l1_num(sub_num),l2_num(sub_num),l3_num(sub_num),l4_num(sub_num),all_num(sub_num)]

        templevel = [level1(method_num),level2(method_num),level3(method_num),level4(method_num),all_num(sub_num)];
        tempall = [tempall;templevel];
    end
    disp(['concordance outcomes of dataset B (13 patients) using ', method_name{method_num},':'])
    
    tempall
    
    clear tempall
    
    level1_all(method_num)=sum(l1_num)/sum(all_num);
    level2_all(method_num)=sum(l2_num)/sum(all_num);
    level3_all(method_num)=sum(l3_num)/sum(all_num);
    level4_all(method_num)=sum(l4_num)/sum(all_num); 
end

switch source_type
    case 'center'
        four_levels_center = [level1_all;level2_all;level3_all;level4_all]
    case 'first'
        four_levels_first = [level1_all;level2_all;level3_all;level4_all]
end

% 
% switch source_type
%     case 'center'
%         four_levels_center = [level1;level2;level3;level4]
%     case 'first'
%         four_levels_first = [level1;level2;level3;level4]
% end



