clear;
close all;
channelloc_all=zeros(306,3,48);
channelori_all=zeros(306,3,48);
auto_all=[];
manu_all=[];

for kkk=1:48 
    path= pwd;
    path_auto = [path,'/auto_mat/'];
    
    load([path_auto,'anony_auto_sub',num2str(kkk),'.mat'])
    channelloc=[];
    channelori=[];

    EP_refined_channel= Header;
    headpoint_auto = double(Header.HeadPoints.Loc');
    Mask_auto=Mask;
    
    channel_num=numel(EP_refined_channel.Channel);
    MEG_id=[];
    for temp=1:channel_num
        channel_name = EP_refined_channel.Channel(temp).Type;
        if strcmp(channel_name(1:3),'MEG')==1
            MEG_id=[MEG_id,temp];
        end
    end
    
    if numel(MEG_id)~=306
        disp('-------error------')
    end
    
    for i=1:306
        k=MEG_id(i);
        channelloc=[channelloc; mean(EP_refined_channel.Channel(k).Loc',1)];
        channelori=[channelori;mean(EP_refined_channel.Channel(k).Orient',1)];
    end
    channelloc_all(:,:,kkk)=channelloc;
    channelori_all(:,:,kkk)=channelori;
    
    clear Header Mask

    for i=1:size(headpoint_auto,1)
        minidisk_auto(i)=find_min_dist(headpoint_auto(i,:),Mask_auto.Vertices);
    end
    
    auto_all=[auto_all,minidisk_auto];
    auto_48(kkk)=mean(minidisk_auto);

    path_manu = [path,'/manu_mat/'];
    load([path_manu,'anony_manu_sub',num2str(kkk),'.mat'])
    channelloc=[];
    channelori=[];

    EP_refined_channel= Header;
    headpoint_man = double(Header.HeadPoints.Loc');
    Mask_man=Mask;
    
    channel_num=numel(EP_refined_channel.Channel);
    MEG_id=[];
    for temp=1:channel_num
        channel_name = EP_refined_channel.Channel(temp).Type;
        if strcmp(channel_name(1:3),'MEG')==1
            MEG_id=[MEG_id,temp];
        end
    end
    
    if numel(MEG_id)~=306
        disp('-------error------')
    end
    
    for i=1:306
        k=MEG_id(i);
        channelloc=[channelloc; mean(EP_refined_channel.Channel(k).Loc',1)];
        channelori=[channelori;mean(EP_refined_channel.Channel(k).Orient',1)];
    end 
    
    channelloc_all(:,:,kkk)=channelloc;
    channelori_all(:,:,kkk)=channelori;
    clear Header Mask
    for i=1:size(headpoint_auto,1)
        minidisk_manual(i)=find_min_dist(headpoint_man(i,:),Mask_man.Vertices);
    end
    
    manu_all=[manu_all,minidisk_manual];
    hp_num(kkk)=numel(minidisk_manual);
    manu_48(kkk)=mean(minidisk_manual);
    
    [mean(minidisk_auto),mean(minidisk_manual)]
    clear minidisk_manual minidisk_auto  
    close all;
    
end

[r1,p1]=corr(hp_num',auto_48');
[r2,p2]=corr(hp_num',manu_48');
[H,P,CI,STATS] = ttest(manu_48',auto_48');
