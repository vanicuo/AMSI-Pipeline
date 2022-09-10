function [spike_ind, picked_components, picked_comp_top] = SpikeDetect(Data, ...
    channel_idx, G2, f_low, f_high, ncomp, check, decision,...
    bad_idx, num_event)

Fs = 1/(Data.Time(2)-Data.Time(1));

[b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
Ff = filtfilt(b, a, Data.F(channel_idx,:)')';
[b,a] = butter(4, [8 12]/(Fs/2), 'stop'); % butterworth filter before ICA
Ff = filtfilt(b, a, Ff')';

% delete bad segments
Ff_wb = Ff;
Ff_wb(:, bad_idx) = [];

[w,s] = runica(Ff_wb, 'pca', ncomp); % compute ICA components
% [spike, Q, W] = fastica(Ff_wb, 'numOfIC', ncomp);
W = w*s; % unmixing matrix
Q = pinv(W); % matrix of ICA topographies

Ff(:,bad_idx) = zeros(size(Ff,1), length(bad_idx));

ica_ts = W*Ff; % ICA timeseries

if isempty(num_event)
    spikes = [];
else
    spikes = int32((Data.Events(num_event).times-Data.Time(1))*Fs+1); % manually detected spikes
end

% 2.1. Kurtosis (how outlier-prone the distribution is)
% we are looking for a high kurtosis (but not too high)

kurt = kurtosis(ica_ts');
[val_kurt_init, ind_kurt_init] = sort(kurt, 'descend');

% Delete components with too high kurtosis
ind_kurt = ind_kurt_init(val_kurt_init <= 50);
val_kurt = val_kurt_init(val_kurt_init <= 50);

if length(ind_kurt)>20
    pos_ind = ind_kurt(1:20);
else pos_ind = ind_kurt;
end

ica_ts = W(pos_ind,:)*Ff;
kurt = kurtosis(ica_ts'); 
[val_kurt, ind_kurt] = sort(kurt, 'descend');
ica_ts_sort = ica_ts(ind_kurt,:);  % ICA components sorted with kurtosis
Q = Q(:,pos_ind(ind_kurt));

if check == 1
    num_val = int32(Fs*10);
    time = 1/Fs:1/Fs:(size(ica_ts,2)/Fs);
    events = zeros(1, size(time, 2));
    events(spikes) = 1;
    scrolling_plot(time, ica_ts_sort, num_val, events, 2, val_kurt)
end


% 2.4. Maximal absolute amplitude of the contribution of one component
for i = 1:size(ica_ts_sort,1)
    maxamp(i) = max(max(abs(Q(:,i)*ica_ts_sort(i,:))));
end
maxamp_ind = find(maxamp>(quantile(maxamp, 0.7)));


% 2.4 Dipole fit for the ica_component topography
% even better decision
for i = 1:size(ica_ts_sort,1)
    corr = MUSIC_scan(G2, Q(:,i)/norm(Q(:,i)));
    icadip(i) = max(corr);
end


icadip_ind = find(icadip>(quantile(icadip, 0.7)));


% 3. Match filter
% for the first 50 spikes
% ica_ts_sort_n = sqrt(sum(ica_ts_sort.^2,2));
% ica_ts_sort = bsxfun(@rdivide, ica_ts_sort, ica_ts_sort_n);
% 
% for i = 1:50
%     x = Ff(:,spikes(i)); % ideal spike
%     x = x/norm(x);
%     X_n = sqrt(sum(Ff.^2,1)); % norm of the each time sample
%     X_n(X_n == 0) = 10^(-5);
%     X = bsxfun(@rdivide, Ff, X_n);
%     match_filter = x'*X;
%     match_filter(match_filter<0.8) = 0;
%     match_filter = match_filter/norm(match_filter);
%     match_corr(i,:) = abs(ica_ts_sort*match_filter');
% end
% 
% % figure
% % imagesc(match_corr)
% % colorbar
% 
% ind_match = find(max(match_corr)> quantile(max(match_corr), 0.75));

% Final decision
ind_spikecomp = unique([maxamp_ind, icadip_ind]);

[d, ~] = max(abs(ica_ts_sort(ind_spikecomp,:))); % maximum values
[val_d, ind_d] = sort(d); % threshold
thr = val_d(round(decision*size(d,2))); 

decision_ts = d;
spike_ind = find(d>=thr);
picked_comp_top = Q(:,ind_spikecomp);
picked_components = ica_ts_sort(ind_spikecomp,:);

% size(spike_ind)
clear spike_ind_red
diff = [10, spike_ind(2:length(spike_ind))-spike_ind(1:(length(spike_ind)-1))];
amp = 0;
k = 1;
for i = 1:length(diff)
    if (diff(i)<20)
        if (d(spike_ind(i)) > amp)
            amp = d(spike_ind(i));
            spike_ind_red(k) = spike_ind(i);
        end
    else
        k = k+1;
        spike_ind_red(k) = spike_ind(i);
        amp = d(spike_ind(i));
    end
end  

% find which particular component contributed in each particular spike
for i = 1:length(spike_ind_red)
    for j = 1:length(ind_spikecomp)
        component_indicatior(i,j) = (abs(ica_ts_sort(ind_spikecomp(j),spike_ind_red(i)))>thr)*j;
    end
end

disp(['Picked components ', num2str(ind_spikecomp)])

disp(['Number of detected spikes ', num2str(length(spike_ind_red))])

spike_ind = spike_ind_red;

end
