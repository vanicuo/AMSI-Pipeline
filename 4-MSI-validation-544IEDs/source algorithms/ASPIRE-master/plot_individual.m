function plot_individual(spike_time, Data, f_low, f_high, ...
    channel_type, G3, MRI, channels)

% -------------------------------------------------------------------------
% Plot separate spikes
% -------------------------------------------------------------------------
% INPUTS:
%   spike_time -- time stamps with spiking events
%   Data -- brainstorm structure with artifact corrected maxfiltered MEG data
%   G3 -- brainstorm structure with forward operator
%   channel_type -- channels used ('grad' or 'mag')
%   f_low, f_high -- the bands for prefiltering before fitting
%   MRI structure from bst
%   channels -- channels info
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

    if strcmp(channel_type, 'grad') == 1
        grad_idx = setdiff(1:306, 3:3:306);
        cfg = [];
        cfg.layout = 'neuromag306planar.lay';
        channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
    elseif strcmp(channel_type, 'mag') == 1
        magn_idx = 3:3:306;
        cfg = [];
        cfg.layout = 'neuromag306mag.lay';
        channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
    end
    Fs = 1/(Data.Time(2)-Data.Time(1));
    [b,a] = butter(4, [f_low f_high]/(Fs/2)); % butterworth filter before ICA
    Ff = filtfilt(b, a, Data.F(channel_idx,:)')';
    
    [G2, ~] = G3toG2(G3, channel_idx);
    R = G3.GridLoc;

    k = 1;
    for i = 1:length(channel_idx)
        namechan{i} = channels.Channel(channel_idx(i)).Name;
        k = k+1;
    end

    avgFC = load('avgFC.mat');
    for i = 1:length(spike_time)
        spike = Ff(:,(spike_time(i)-20):(spike_time(i)+30));
        [U,S,V] = svd(spike);
        h = cumsum(diag(S)/sum(diag(S)));
        n = find(h>=0.95);
        corr = MUSIC_scan(G2, U(:,1:n(1)));
        [~, dip_ind] = max(corr);
                    
        spike = Ff(:, (spike_time(i)-40):(spike_time(i)+80));
        g = G2(:,(dip_ind*2-1):dip_ind*2);
        [U S ~] = svd(spike);
        h = cumsum(diag(S)/sum(diag(S)));
        n = min(find(h>=0.95));
        [u s v] = svd(U(:,1:n)'*g);
        g_fixed = g*v(1,:)';
        spike_ts = spike'*g_fixed;
%         if spike_ts(40) > 0
%             spike_ts = -spike_ts;
%         end
        
        coord_scs = R(dip_ind,:);
        coord_mri = cs_convert(MRI, 'scs', 'voxel', coord_scs);
        coord_mri = round(coord_mri);

        figure
        subplot(4,4,4)
        plot(-40:80, spike_ts, 'LineWidth', 2)
        xlim([-40, 80])
        title('Timecourse on sources')
        subplot(4,4,8)   
        mri = flipud(MRI.Cube(:,:,coord_mri(3))');
        imagesc(mri)
        colormap('gray')
        axis equal
        grid off
        axis off
        hold on
        scatter(coord_mri(:,1), 257-coord_mri(:,2),  50, 'filled', 'MarkerFaceColor', ...
            'r', 'MarkerEdgeColor', 'k');
        subplot(4,4,12)   
        mri = flipud(squeeze(MRI.Cube(:,coord_mri(2),:))');
        imagesc(mri)
        axis equal
        grid off
        axis off
        hold on
        scatter(coord_mri(:,1), 257-coord_mri(:,2), 50, 'filled', 'MarkerFaceColor', ...
            'r', 'MarkerEdgeColor', 'k');
        subplot(4,4,16)   
        mri = flipud(fliplr(squeeze(MRI.Cube(coord_mri(1),:,:))'));
        imagesc(mri)
        colormap('gray')
        hold on
        scatter(257-coord_mri(:,1), 257-coord_mri(:,2), 50, 'filled', 'MarkerFaceColor', ...
                'r', 'MarkerEdgeColor', 'k');
        axis equal
        grid off
        axis off
   
        subplot(4,4,[1:3,5:7,9:11,13:15])
        
        data = avgFC;
        data.avg = spike;
        data.var = spike;
        data.dof = repmat(72, size(data.avg));
        data.label = namechan';
        data.time = -40:80;
        ft_multiplotER(cfg, data);

end