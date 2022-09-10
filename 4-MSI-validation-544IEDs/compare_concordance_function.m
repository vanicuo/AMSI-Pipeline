function [l1_num,l2_num,l3_num,l4_num,all_num]=compare_concordance_function(sequence_GT,sequence_spike_center)

   all_num = size(sequence_spike_center,1);
   l1_num=0;
   l2_num=0;
   l3_num=0;
   l4_num=0;
   
   l1_index=[];
   for i=1:size(sequence_spike_center,1)
       for j=1:size(sequence_GT,1)
           if sequence_spike_center(i,1)==sequence_GT(j,1)
               l1_num=l1_num+1;
               l1_index=[l1_index,i];
               break;        
           end
       end
   end
   
   
   
%    load index_level.mat       
   temp2=[];
   for j=1:size(sequence_GT,1)
       l2_GT(j) = 2^sequence_GT(j,1)+2^sequence_GT(j,2);
       temp2=union(temp2,l2_GT(j));
   end  
   
   for j=1:size(sequence_spike_center,1)
       l2_spike(j) = 2^sequence_spike_center(j,1)+2^sequence_spike_center(j,2);
%        temp2=union(temp2,l2_GT(j));
   end
   
   for i=1:numel(l2_spike)
       for j=1:numel(temp2)
            if l2_spike(i)==temp2(j)
                l2_num=l2_num+1;
                break;
            end
       end
   end
   

%    load index_level.mat       
   temp3=[];
   for j=1:size(sequence_GT,1)
       l3_GT(j) = 2^sequence_GT(j,1)+2^sequence_GT(j,2)+2^(sequence_GT(j,3)+4);
       temp3=union(temp3,l3_GT(j));
   end  
   
   for j=1:size(sequence_spike_center,1)
       l3_spike(j) = 2^sequence_spike_center(j,1)+2^sequence_spike_center(j,2)+2^(sequence_spike_center(j,3)+4);
%        temp2=union(temp2,l2_GT(j));
   end
   
   for i=1:numel(l3_spike)
       for j=1:numel(temp3)
            if l3_spike(i)==temp3(j)
                l3_num=l3_num+1;
                break;
            end
       end
   end



%    load index_level.mat       
   temp4=[];
   for j=1:size(sequence_GT,1)
       l4_GT(j) = 2^sequence_GT(j,1)+2^sequence_GT(j,2)+2^(sequence_GT(j,3)+4)-35+10^(sequence_GT(j,4));
       temp4=union(temp4,l4_GT(j));
   end  
   
   for j=1:size(sequence_spike_center,1)
       l4_spike(j) = 2^sequence_spike_center(j,1)+2^sequence_spike_center(j,2)+...
           2^(sequence_spike_center(j,3)+4)-35+10^(sequence_spike_center(j,4));
%        temp2=union(temp2,l2_GT(j));
   end
   
   for i=1:numel(l4_spike)
       for j=1:numel(temp4)
            if l4_spike(i)==temp4(j)
                l4_num=l4_num+1;
                break;
            end
       end
   end


% 
% load index_level.mat
% temp4=[];
% for i=1:90
%     group_label(i)=2^rl_index(i)+2^labor_index(i)+2^(medial_lateral_index(i)+4);
%     temp4=union(temp4,group_label(i));
% end
% numel(temp4)
   
%    l2_index=[];
%    for ii=1:numel(l1_index)
%        for j=1:size(sequence_GT,1)
%            if sequence_spike_center(l1_index(l1_index(ii)),2)==sequence_GT(j,2)
%                l2_num=l2_num+1;
%                l2_index=[l2_index,l1_index(ii)];
%                break;
%            end
%        end
%    end
%    
%    l3_index=[];
%    if isempty(l2_index)~=1
%        for iii=1:numel(l2_index)
%            for j=1:size(sequence_GT,1)
%                if sequence_spike_center(l2_index(iii),3)==sequence_GT(j,3)
%                    l3_num=l3_num+1;
%                    l3_index=[l3_index,l2_index(iii)];
%                    break;
%                end
%            end
%        end
%    end
% 
%    l4_index=[];
%    if isempty(l3_index)~=1
%        for iiii=1:numel(l3_index)
%            for j=1:size(sequence_GT,1)
%                if sequence_spike_center(l3_index(iiii),4)==sequence_GT(j,4)
%                    l4_num=l4_num+1;
%                    l4_index=[l4_index,l3_index(iiii)];
%                    break;
%                end
%            end
%        end
%    end
   
   
           
