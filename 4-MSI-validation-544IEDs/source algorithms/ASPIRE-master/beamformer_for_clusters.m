function [bf_ts, spike_sources, spike_sources_av] = beamformer_for_clusters(Data, ...
    G3, cluster, channel_type, spikydata, picked_components, picked_comp_top)

if strcmp(channel_type, 'grad') == 1
    grad_idx = setdiff(1:306, 3:3:306);
    channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
elseif strcmp(channel_type, 'mag') == 1
    magn_idx = 3:3:306;
    channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
end

% 2D forward operator
[G2, ~] = G3toG2(G3, channel_idx); 

Fs = 1/(Data.Time(2)-Data.Time(1));

% [u s] = eig(G2*G2');
% h = cumsum(diag(s)/sum(diag(s)));
% n = min(find(h>0.95));

% UP = u(:,1:n)'; % direction of dimention reduction
% G2U = UP*G2;

% filtering for localization
f_low = 8;
f_high = 70;
[b,a] = butter(4, [f_low f_high]/(Fs/2)); 
Ff_870 = filtfilt(b, a, Data.F(channel_idx,:)')';

% filtering for beamformer reconstruction
f_low = 2;
f_high = 70;
Fs = 1/(Data.Time(2)-Data.Time(1));
[b,a] = butter(4, [f_low f_high]/(Fs/2)); 
Ff_270 = filtfilt(b, a, Data.F(channel_idx,:)')';
% FfU_270 = UP*Ff_270;

if spikydata == 1
    data_spike = picked_comp_top*picked_components;
end

Gtotal = [];
for cluster_num = 1:size(cluster, 2)
    
    spike_ind = cluster{1,cluster_num}(2,:); % time samples with spikes detected
    spike_src = cluster{1,cluster_num}(1,:); % the dipole localized with MUSIC previosly

    % find orientations of dipoles localized with MUSIC
    Gact = [];

    for i = 1:length(spike_ind)
        if spikydata == 0
            spike = Ff_870(:,(spike_ind(i)-30):(spike_ind(i)+30));
        else
            spike = data_spike(:,(spike_ind(i)-30):(spike_ind(i)+30));
        end
        [U,S,V] = svd(spike);
        h = cumsum(diag(S)/sum(diag(S)));
        n = find(h>=0.95);
        g = G2(:,(spike_src(i)*2-1):spike_src(i)*2);
        B = g'*U(:,1:n(1));
        [U,S,V] = svd(B);
        orient(i,:) = U(:,1);
        Gact = [Gact g*orient(i,:)'];
    end
    Gact_clust{cluster_num} = Gact;
    Gtotal = [Gtotal, Gact];
    cluster_size(cluster_num) = length(spike_ind);
end

clear w
for cluster_num = 1:size(cluster, 2)
    
    spike_ind = cluster{1,cluster_num}(2,:); % time samples with spikes detected
    spike_src = cluster{1,cluster_num}(1,:); % the dipole localized with MUSIC previosly

    % correlation matrix with spikes 
    C = zeros(length(channel_idx), length(channel_idx));
    for i = 1:length(spike_ind)
        if spikydata == 0
            spike = Ff_270(:,(spike_ind(i)-20):(spike_ind(i)+30));
        else
            spike = data_spike(:,(spike_ind(i)-20):(spike_ind(i)+30));
        end
        C = C + spike*spike';
    end
    C = C/length(spike_ind);
    iC = tihinv(C, 0.01); 

    % beamformer for the considered cluster
    Gact = Gact_clust{cluster_num};
    Gsupp = Gtotal;
    
    Gsupp(:, 1+sum(cluster_size(1:(cluster_num-1))):sum(cluster_size(1:cluster_num))) = [];
    
    [ua sa va] = svd(Gact);
    h = cumsum(diag(sa)/sum(diag(sa)));
    na = min(find(h>=0.99));
    
    [us ss vs] = svd(Gsupp);
    h = cumsum(diag(ss)/sum(diag(ss)));
    ns = min(find(h>=0.99));

    f = size(Gact, 2)/cluster_size(cluster_num);
    ra = kron([ones(1, cluster_size(cluster_num))], eye(f,f)); % unit gain constraints for sources from the cluster
    rs = kron([zeros(1, sum(cluster_size)-cluster_size(cluster_num))], eye(f,f)); % zero constraints for sources from the other clusters

    w(cluster_num,:) = iC*[ua(:,1:na), us(:,1:ns)]*[ua(:,1:na),us(:,1:ns)]'*iC*...
        [ua(:,1:na), us(:,1:ns)]*[ra*va(:,1:na)*inv(sa(1:na,1:na)), rs*vs(:,1:ns)*inv(ss(1:ns,1:ns))]'; 

    if spikydata == 0
        bf_ts(cluster_num,:) = w(cluster_num,:)*Ff_270; % bf timeseries
    else
        bf_ts(cluster_num,:) = w(cluster_num,:)*data_spike; % bf timeseries
    end

    for i = 1:cluster_size(cluster_num)
        spike_sources{cluster_num}(i,:) = bf_ts(cluster_num, ...
            (spike_ind(i)-40):(spike_ind(i)+120));
    end
    spike_sources_av{cluster_num} = mean(spike_sources{cluster_num}, 1);
    
end

% plot beamformer timeseries
% x = 1/Fs:1/Fs:(size(bf_ts,2)/Fs);
% num_val = int32(Fs*5);
% Y = bf_ts;
% nw = 2;
% event = [];
% 
% scrolling_plot3(x, Y, num_val, event, nw)
end