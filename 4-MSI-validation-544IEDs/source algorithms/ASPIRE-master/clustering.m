function [cluster] = clustering(spike_ind, G3, Nmin, ValMax, IndMax, ...
    ind_m, thr_dist, draw, cortex, RAP, spikeind)

% -------------------------------------------------------------------------
% Spatial clustering of localized spikes
% -------------------------------------------------------------------------
% INPUTS:
%   spike_ind -- indices of all detected spikes
%   G3 -- brainstorm structure with forward operator
%   Nmin -- minimal number of sources in one cluster
%   ValMax -- subcorr values from RAP-MUSIC
%   IndMax -- locations of sources from RAP-MUSIC
%   ind_m -- indices of sources survived after the subcorr thresholding
%   draw -- draw plot or not 
%   cortex -- cortical structure from brainstorm
%   RAP -- 'RAP' to use complete RAP-MUSIC procedure, smth else for one-round 
%   spikeind -- timeindices from RAP-MUSIC procedure
% 
% OUTPUTS:
%   cluster -- structure [length(ind_m)x4], first column -- source
%           location, second column -- spike timestamp, third -- the
%           subcorr value, fourth -- index of the spike from the whole set
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

if size(spike_ind,2) == 1
    spike_ind = spike_ind';
end
if strcmp(RAP, 'RAP') == 0
   src_idx = [IndMax(ind_m); spike_ind(ind_m); ValMax(ind_m); ind_m];
else
   src_idx = [IndMax; spikeind; ValMax; ind_m];
end    

locs = G3.GridLoc;
for i = 1:size(src_idx,2) % distances between each vertex
    for j = 1:size(src_idx, 2)
        dist(i,j) = norm(locs(src_idx(1,i),:)-locs(src_idx(1,j),:));
    end
end

clear cluster
fl = 1; 
k = 1;
while fl == 1
    dst = sum(dist < thr_dist, 2); 
    [val, ind] = max(dst); % vertex with the highest number of close neighbours
    if val > Nmin
        ind_nbh = find(dist(ind,:) < thr_dist); % neighbours
        cluster{k} = src_idx(:,ind_nbh);
        src_idx = src_idx(:,setdiff(1:size(dist,1), ind_nbh));
        dist = dist(setdiff(1:size(dist,1), ind_nbh),setdiff(1:size(dist,1), ind_nbh));
        k = k + 1;
        fl = 1;
    else
        fl = 0;
    end
end

if draw == 1
    cortex_lr = cortex;
    cortex_hr = cortex;
    c = [lines(7); 0.15, 0.15, 0.15; ...
        0, 0.5, 0.5; 0.3, 0.18, 0.4; ...
        0.8, 0.08, 0.5; 0.9, 0.5, 0.5;lines(7); 0.15, 0.15, 0.15; ...
        0, 0.5, 0.5; 0.3, 0.18, 0.4; ...for src_hemi in fwd_fixed['src']]
        0.8, 0.08, 0.5; 0.9, 0.5, 0.5]; % colors
    data_lr = ones(length(cortex_lr.Vertices),1);
    mask_lr = zeros(size(data_lr));
    figure
    subplot(1,2,1)
    plot_brain_cmap2(cortex_lr, cortex_lr, [], data_lr, ...
        mask_lr, 0.05)
    hold on
    for i = 1:length(cluster)
        ind = cluster{1,i}(1,:);
        scatter3(cortex.Vertices(ind,1), cortex.Vertices(ind,2), ...
            cortex.Vertices(ind,3), 100, 'filled', 'MarkerEdgeColor','k',...
                'MarkerFaceColor',c(i,:));
    end
    view(270, 90)
    subplot(1,2,2)
    plot_brain_cmap2(cortex_lr, cortex_lr, [], data_lr, ...
        mask_lr, 0.05)
    hold on
    for i = 1:length(cluster)
        ind = cluster{1,i}(1,:);
        scatter3(cortex.Vertices(ind,1), cortex.Vertices(ind,2), ...
            cortex.Vertices(ind,3), 100, 'filled', 'MarkerEdgeColor','k',...
                'MarkerFaceColor',c(i,:));
    end
    view(0, 0)

end
end
