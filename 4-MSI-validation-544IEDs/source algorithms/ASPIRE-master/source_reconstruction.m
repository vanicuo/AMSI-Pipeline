function [spike_trials, maxamp_spike_av, channels_maxamp, spike_ts] = ...
                source_reconstruction(Data, G3, channel_type, cluster, ...
                f_low, f_high)
% -------------------------------------------------------------------------
% Source timeseries (dipole-fitting)
% -------------------------------------------------------------------------
% INPUTS:
%   Data -- brainstorm structure with artifact corrected maxfiltered MEG data
%   G3 -- brainstorm structure with forward operator
%   channel_type -- channels used ('grad' or 'mag')
%   cluster -- structure with spatial clusters
%   f_low, f_high -- bandpass filtering for visualization
%   
% OUTPUTS:
%   spike_trials -- cell (number of clusters size) with (number_of_channels x
%                       121 x number_of_spikes_in_cluster) with cropped spikes
%   maxamp_spike_av -- cell (number of clusters size) with average spike on
%                       5 top amplitude channels (5 x 121)
%   channels_maxamp -- cell (number of clusters size) with indices of 20 top 
%                       amplitude channels (20 x 1)
%   spike_ts -- cell (number of clusters size) with reconstructed timeseries
%                       for each spike from the cluster (number_of_spike_in_cluster
%                          x 121)
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

    % channels mag or grad
    if strcmp(channel_type, 'grad') == 1
        grad_idx = setdiff(1:306, 3:3:306);
        channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
    elseif strcmp(channel_type, 'mag') == 1
        magn_idx = 3:3:306;
        channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
    end
    
    % 2D forward operator
    [G2, ~] = G3toG2(G3, channel_idx);
    
    % Filtering before source reconstruction
    Fs = 1/(Data.Time(2)-Data.Time(1));
    [b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
    Ff = filtfilt(b, a, Data.F(channel_idx,:)')';

    % Average spike on the channels with maximal amplitude
    clear spike_trials
    for i = 1:length(cluster)
        for j = 1:size(cluster{1,i}, 2) % number of spikes in the cluster
            indtime = cluster{1,i}(2,j); % spike timesamples
            spike_trials{i}(:,:,j) = Ff(:,(indtime-40):(indtime+80)); % spikes from recordings
        end
        spike_av{i} = mean(spike_trials{i}, 3);
        spike_std{i} = std(spike_trials{i},[], 3);
        gfp{i} = sum(spike_av{i}.^2, 1);
        
%         figure
%         plot(gfp{i})
        [valmax indmax] = max(gfp{i}(20:60));
        
        [val ind] = sort(abs(spike_av{i}(:,20+indmax)), 'descend'); % sort by peak amplitude
        channels_maxamp{i} = ind(1:20);
        maxamp_spike_av{i} = spike_av{i}(ind(1:5,:),:); % ten channels with the highest amplitude
        for j = 1:5
            if maxamp_spike_av{i}(j,20+indmax) > 0
                maxamp_spike_av{i}(j,:) = -maxamp_spike_av{i}(j,:);
            end
        end
    end

    % plot the average spike
%     figure
%     for i = 1:length(cluster)
%         subplot(2,4,i)
%         plot(maxamp_spike_av{i}')
%     end
%   
%     figure
%     for i = 1:9
%         subplot(3,3,i)
%         plot(spike_trials{2}(channels_maxamp{2}(1:5),:,i)')
%     end
    
    for i = 1:size(cluster, 2)
        for j = 1:length(cluster{1,i})
            dip_ind = cluster{1,i}(1,j);
            sp_ind = cluster{1,i}(2,j);
            
            spike = Ff(:, (sp_ind-40):(sp_ind+80));
            
%             figure
%             plot(spike')
            
            g = G2(:,(dip_ind*2-1):dip_ind*2);
            [U S ~] = svd(spike);
            h = cumsum(diag(S)/sum(diag(S)));
            n = min(find(h>=0.95));
            [u s v] = svd(U(:,1:n)'*g);
            g_fixed = g*v(1,:)';
            spike_ts{i}(j,:) = spike'*g_fixed;
            if spike_ts{i}(j,40) > 0
                spike_ts{i}(j,:) = -spike_ts{i}(j,:);
            end
                            
%             figure
%             plot(spike_ts{i}(j,:))
        end
    end

    
%     figure
%     for i = 1:length(cluster)
%         subplot(4,2,i)
%         plot(spike_ts{i}', 'Color', 'b')
%         hold on
%         plot(mean((spike_ts{i})), 'LineWidth', 2)
%     end
end
            
            
            
            