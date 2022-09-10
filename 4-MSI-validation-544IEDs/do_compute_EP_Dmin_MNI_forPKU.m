clear;clc;
main_folder = pwd;
path = [main_folder,'/results_PKU_fsllabel/'];
path_GT = [main_folder,'/electrode_coor/'];
subject_name={'sub01','sub02','sub03'...
                  'sub04','sub05','sub06','sub07',...
                  'sub08','sub09','sub10','sub11','sub12','sub13'};              
method_name={'wmne','sLoreta','dspm','beam','dipolefit','MUSIC','fvestal','STOUT'};
file_name={'mne','sLoreta','dspm','bf','dipole','MUSIC','fvestal','STOUT'};  
tempall=[]
numnum=[]
nummm=[]
for method_num=1:numel(method_name)

    for sub_num=1:numel(subject_name)
        subject_name{sub_num}  
        load([path,subject_name{sub_num},'/fsllabel/',method_name{method_num},'/final_fsl_first_MNI.mat']);
        load([path,subject_name{sub_num},'/fsllabel/',method_name{method_num},'/final_fsl_center_MNI.mat']); 

             
        load([path_GT,'seeg_',subject_name{sub_num},'.mat']);
        Dmin_IID_center_value=zeros(size(final_fsl_center,1),1);Dmin_IID_center_id=zeros(size(final_fsl_center,1),1);
        Dmin_SZ_center_value=zeros(size(final_fsl_center,1),1);Dmin_SZ_center_id=zeros(size(final_fsl_center,1),1);
        Dmin_GT_center_value=zeros(size(final_fsl_center,1),1);Dmin_GT_center_id=zeros(size(final_fsl_center,1),1);
        
        Dmin_IID_first_value=zeros(size(final_fsl_first,1),1);Dmin_IID_first_id=zeros(size(final_fsl_first,1),1);
        Dmin_SZ_first_value=zeros(size(final_fsl_first,1),1);Dmin_SZ_first_id=zeros(size(final_fsl_first,1),1);
        Dmin_GT_first_value=zeros(size(final_fsl_first,1),1);Dmin_GT_first_id=zeros(size(final_fsl_first,1),1);  
        
        for spike_num=1:size(final_fsl_center,1)
            
            delta_IID_center = rownorm(repmat(final_fsl_center(spike_num,:),numel(seeg_index_IID),1)...
                         -seeg_fslxyz(seeg_index_IID,:));
            
            delta_SZ_center = rownorm(repmat(final_fsl_center(spike_num,:),numel(seeg_index_SZ),1)...
                         -seeg_fslxyz(seeg_index_SZ,:));   
            
            delta_GT_center = rownorm(repmat(final_fsl_center(spike_num,:),numel(seeg_index_GT),1)...
                         -seeg_fslxyz(seeg_index_GT,:));
                     
            delta_IID_first = rownorm(repmat(final_fsl_first(spike_num,:),numel(seeg_index_IID),1)...
                         -seeg_fslxyz(seeg_index_IID,:));
                     
            delta_SZ_first = rownorm(repmat(final_fsl_first(spike_num,:),numel(seeg_index_SZ),1)...
                         -seeg_fslxyz(seeg_index_SZ,:));   
            
            delta_GT_first = rownorm(repmat(final_fsl_first(spike_num,:),numel(seeg_index_GT),1)...
                         -seeg_fslxyz(seeg_index_GT,:));
                     
            [Dmin_IID_center_value(spike_num),Dmin_IID_center_id(spike_num)]=min(delta_IID_center) ;  
            [Dmin_SZ_center_value(spike_num),Dmin_SZ_center_id(spike_num)]=min(delta_SZ_center) ;
            [Dmin_GT_center_value(spike_num),Dmin_GT_center_id(spike_num)]=min(delta_GT_center) ; 
            
            [Dmin_IID_first_value(spike_num),Dmin_IID_first_id(spike_num)]=min(delta_IID_first) ;  
            [Dmin_SZ_first_value(spike_num),Dmin_SZ_first_id(spike_num)]=min(delta_SZ_first) ;
            [Dmin_GT_first_value(spike_num),Dmin_GT_first_id(spike_num)]=min(delta_GT_first) ;            
            
        end
        all_data_subject_IID_center{sub_num,method_num}=Dmin_IID_center_value;
        all_data_subject_SZ_center{sub_num,method_num}=Dmin_SZ_center_value;
        all_data_subject_GT_center{sub_num,method_num}=Dmin_GT_center_value;  

        all_data_subject_IID_first{sub_num,method_num}=Dmin_IID_first_value;
        all_data_subject_SZ_first{sub_num,method_num}=Dmin_SZ_first_value;
        all_data_subject_GT_first{sub_num,method_num}=Dmin_GT_first_value;       

        mean_data_subject_IID_center(sub_num,method_num)=mean(Dmin_IID_center_value);
        mean_data_subject_SZ_center(sub_num,method_num)=mean(Dmin_SZ_center_value);
        mean_data_subject_GT_center(sub_num,method_num)=mean(Dmin_GT_center_value);  

        mean_data_subject_IID_first(sub_num,method_num)=mean(Dmin_IID_first_value);
        mean_data_subject_SZ_first(sub_num,method_num)=mean(Dmin_SZ_first_value);
        mean_data_subject_GT_first(sub_num,method_num)=mean(Dmin_GT_first_value);
        
        tempall = [tempall;[mean(Dmin_GT_first_value),std(Dmin_GT_first_value),...
            median(Dmin_GT_first_value),size(final_fsl_center,1)]]

    end
end

for method_num=1:numel(method_name)
    IID_c=[];
    SZ_c=[];
    GT_c=[];
    
    IID_f=[];
    SZ_f=[];
    GT_f=[];

    for sub_num=1:numel(subject_name)     
        IID_c= [IID_c;all_data_subject_IID_center{sub_num,method_num}];
        SZ_c = [SZ_c;all_data_subject_SZ_center{sub_num,method_num}];
        GT_c=[GT_c;all_data_subject_GT_center{sub_num,method_num}];

        IID_f=[IID_f;all_data_subject_IID_first{sub_num,method_num}];
        SZ_f=[SZ_f;all_data_subject_SZ_first{sub_num,method_num}];
        GT_f=[GT_f;all_data_subject_GT_first{sub_num,method_num}];             
    end
    disp(['------',method_name{method_num},'------']);
    [mean(IID_c),mean(SZ_c),mean(GT_c);mean(IID_f),mean(SZ_f),mean(GT_f)]
end