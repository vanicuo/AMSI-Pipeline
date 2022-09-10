function [IndMax, ValMax, ind_m, spikeind] = myspike_localization(spike_ind, Data, G3, G2, ...
    id_grad, spikydata, picked_components, ...
    picked_comp_top, corr_thresh, RAP)

% -------------------------------------------------------------------------
% Spike localization with RAP-MUSIC
% -------------------------------------------------------------------------
% INPUTS:
%   spike_ind -- time stamps with spiking events
%   Data -- brainstorm structure with artifact corrected maxfiltered MEG data
%   G3 -- brainstorm structure with forward operator
%   channel_type -- channels used ('grad' or 'mag')
%   f_low, f_high -- the bands for prefiltering before fitting
%   spikydata -- indicatior, showing whether you want to fit
%                   dipoles for the raw data (0) or to the ICA composed data (1)  
%   picked_components -- if spikydata == 1, the picked ICA components
%                           timeseries
%   picked_comp_top -- if spikydata == 1, the picked ICA components
%                           topographies
%   corr_thresh -- subcorr threshold level
%   RAP -- 'RAP' to run RAP-MUSIC and something else for fast one-round
%       MUSIC
%   
% OUTPUTS:
%   IndMax -- locations of fitted dipoles
%   ValMax -- values of subspace correlation
%   ind_m -- indices of spikes survived the subcorr threshold
%   spikeind -- time indices of spikes in RAP-MUSIC
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com
% channel_type='grad';
% % channels: gradiometers or magnetometers
% if strcmp(channel_type, 'grad') == 1
%     grad_idx = setdiff(1:306, 3:3:306);
%     channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
% elseif strcmp(channel_type, 'mag') == 1
%     magn_idx = 3:3:306;
%     channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
% end
channel_idx=id_grad;

% 2D forward operator
% [G2, ~] = G3toG2(G3, channel_idx); 
G2= G2(id_grad,:);

% % %Fs = 1/(Data.Time(2)-Data.Time(1));
% % Fs=1000;
% % [b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
% % % Ff = filtfilt(b, a, Data.F(channel_idx,:)')';
% % Ff = filtfilt(b, a, Data(channel_idx,:)')';
Ff = Data(channel_idx,:);

clear ValMax IndMax

if spikydata == 0
    picked_components = [];
    picked_comp_top = [];
    if spike_ind(1) <= 20
        spike = Ff(:,1:(spike_ind(1)+30));
    else
        spike = Ff(:,(spike_ind(1)-20):(spike_ind(1)+30));
    end
else
    data_spike = picked_comp_top*picked_components;
    if spike_ind(1) <= 20
        spike = data_spike(:,1:(spike_ind(1)+30));
    else
        spike = data_spike(:,(spike_ind(1)-20):(spike_ind(1)+30));
    end
end

T = size(Ff, 2);

if strcmp(RAP, 'RAP') == 0
    spikeind = [];
    [U,S,V] = svd(spike);
    h = cumsum(diag(S)/sum(diag(S)));
    n = find(h>=0.95);
    corr = MUSIC_scan(G2, U(:,1:n(1)));
    [ValMax(1), IndMax(1)] = max(corr);

    for j = 2:length(spike_ind(spike_ind<T-30))
        if spikydata == 0
            spike = Ff(:,(spike_ind(j)-20):(spike_ind(j)+30));
        else
            spike = data_spike(:,(spike_ind(j)-20):(spike_ind(j)+30));
        end
        [U,S,V] = svd(spike);
        h = cumsum(diag(S)/sum(diag(S)));
        n = find(h>=0.95);
        p(j) = n(1);
        corr = MUSIC_scan(G2, U(:,1:n(1)));

        [ValMax(j), IndMax(j)] = max(corr);
        j
    end
else
    ValMax = [];
    IndMax = [];
    spikeind = [];
    [Valmax, Indmax] = RAP_MUSIC_scan(spike, G3.Gain(channel_idx,:), ...
        G2, corr_thresh);
    ValMax = [ValMax, Valmax];
    IndMax = [IndMax, Indmax];
    spikeind = [spikeind, repmat(spike_ind(1), 1, length(Indmax))];

    for j = 2:length(spike_ind(spike_ind<T-30))
        if spikydata == 0
            spike = Ff(:,(spike_ind(j)-20):(spike_ind(j)+30));
        else
            spike = data_spike(:,(spike_ind(j)-20):(spike_ind(j)+30));
        end
        [Valmax, Indmax] = RAP_MUSIC_scan(spike, G3.Gain(channel_idx,:), ...
            G2, corr_thresh);
        ValMax = [ValMax, Valmax];
        IndMax = [IndMax, Indmax];
        spikeind = [spikeind, repmat(spike_ind(j), 1, length(Indmax))];
        j
    end
end

% figure
% histogram(ValMax)

% corr_thresh = quantile(ValMax, 0.95);
ind_m = find((ValMax > corr_thresh));
disp(['Subcorr threshold: ', num2str(corr_thresh), ' Number of spike found: ', ...
    num2str(length(ind_m))]);

end