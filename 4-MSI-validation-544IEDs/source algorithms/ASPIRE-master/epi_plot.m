function epi_plot(Data, channel_type, f_low, f_high, cortex, ...
    spike_trials, maxamp_spike_av, spike_ts, cluster, ...
    channels, G3, MRI, corr_thresh, default_anat, channels_maxamp, ...
    spike_ind)
% 
%  spike_ind, picked_components, picked_comp_top, spike_sources, ...
%     bf_ts, corr_thresh, hemi)

% -------------------------------------------------------------------------
% Visualization of automatic spike detection procedure
% -------------------------------------------------------------------------
% FORMAT:
%   epi_plot(cluster)
% INPUTS:
%   cluster -- structure with all detected clusters
%
%   
% NOTE:
% 
% _________________________________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

% 1. Data is filtered for vizualization 
Fs = 1/(Data.Time(2)-Data.Time(1)); % sampling frequency
R = G3.GridLoc; % locations of sources

% channel indices
if strcmp(channel_type, 'grad') == 1
    grad_idx = setdiff(1:306, 3:3:306);
    channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
elseif strcmp(channel_type, 'mag') == 1
    magn_idx = 3:3:306;
    channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
end

% channel names
k = 1;
for i = 1:length(channel_idx)
    namechan{i} = channels.Channel(channel_idx(i)).Name;
    k = k+1;
end

[b,a] = butter(4, [f_low f_high]/(Fs/2)); % filtering for visualization
Ff = filtfilt(b, a, Data.F(channel_idx,:)')';

% MAIN FIGURE

c = [lines(7); 0.15, 0.15, 0.15; ...
    0, 0.5, 0.5; 0.3, 0.18, 0.4; ...
    0.8, 0.08, 0.5; 0.9, 0.5, 0.5;  0.1, 0.3, 0.2; ...
    lines(7); 0.15, 0.15, 0.15; ...
    0, 0.5, 0.5; 0.3, 0.18, 0.4; ...
    0.8, 0.08, 0.5; 0.9, 0.5, 0.5;  0.1, 0.3, 0.2]; % colors

h = figure;

% top brain view
subplot(3,4,[5:6,9:10])
cortex_lr = cortex;
cortex_hr = cortex;
data_lr = ones(length(cortex_lr.Vertices),1);
mask_lr = zeros(size(data_lr));
plot_brain_cmap2(cortex_lr, cortex_lr, [], data_lr, ...
    mask_lr, 0.05)
axis equal
grid off
axis off
hold on
view(270, 90)
for i = 1:length(cluster)
    ind = cluster{1,i}(1,:);
    ff = scatter3(cortex.Vertices(ind,1), cortex.Vertices(ind,2), ...
        cortex.Vertices(ind,3), 150, 'filled', 'MarkerEdgeColor','k',...
            'MarkerFaceColor',c(i,:));
end

% bars for clicking
subplot(3,4,7)
x = ones(2, length(cluster));
ff = area(x);
colormap(c)
axis off
grid off
ylim([0, length(cluster)])
set(ff,'ButtonDownFcn', @cluststatistics, 'HitTest','on')

% plots for statistics
h1 = subplot(3,4,1);
title('Distribution of subcorrs')
axis equal
grid off
set(gca,'fontsize', 14)
h2 = subplot(3,4,2);
title('Distribution of events in time')
axis equal
grid off
set(gca,'fontsize', 14)
h3 = subplot(3,4,3);
title('Events timeseries (sensors)')
axis equal
grid off
set(gca,'fontsize', 14)
h4 = subplot(3,4,4);
title('Events timeseries (sources)')
axis equal
grid off
set(gca,'fontsize', 14)
subplot(3,4,7)
axis off
h5 = subplot(3,4,8);
axis equal
grid off
h6 = subplot(3,4,11);
axis equal
grid off
h7 = subplot(3,4,12);
axis equal
grid off


button = uicontrol('Style', 'pushbutton',...
   'String', 'Show timeseries',...
   'Position', [300 15 310 30],...
   'Callback', @(source,event)plotMEG(h),...
   'FontSize', 14);

function cluststatistics(source, event)
    
    loc = event.IntersectionPoint;
    clust_num = ceil(loc(2));
    
    % Distribution of subcorrs inside the cluster
    cla(h1)
    h1 = subplot(3,4,1);
    histogram(cluster{1,clust_num}(3,:), 'EdgeColor', 'k', 'FaceColor', ...
        c(clust_num,:), 'BinWidth', 0.001)
    xlim([corr_thresh-0.01 1])
    title('Distribution of subcorrs')
    set(gca,'fontsize', 14)

    % Distribution of this cluster events in time
    cla(h2)
    h2 = subplot(3,4,2);
    stem(cluster{1,clust_num}(2,:), ones(1, length(cluster{1,clust_num}(2,:))), ...
        'MarkerEdgeColor', 'k', 'MarkerFaceColor', c(clust_num,:))
    title('Distribution of events in time')
    xlim([0 size(Ff, 2)])
    set(gca,'fontsize', 14)
    
    % Average spike on sensors
    cla(h3)
    h3 = subplot(3,4,3);
    plot(-40:80, maxamp_spike_av{clust_num}', 'Color', c(clust_num,:), 'LineWidth', 2)
    hold on
    xlim([-40 80])
    title('Average spike on 5 top amplitude sensors')
    set(gca,'fontsize', 14)
    
    % Source activations timeseries
    cla(h4)
    h4 = subplot(3,4,4);
    plot(-40:80, mean(spike_ts{clust_num}, 1), 'Color', c(clust_num, :), 'LineWidth', 2)
    title('Events timeseries (sources)')
    xlim([-40 80])
    setappdata(h, 'cluster', clust_num)
    set(gca,'fontsize', 14)

    coord_scs = R(cluster{clust_num}(1,:),:);
    coord_mri = cs_convert(MRI, 'scs', 'voxel', coord_scs);
    mnslice = round(mean(coord_mri, 1));
    coord_mri = round(coord_mri);
    
    cla(h5)
    h5 = subplot(3,4,8);
    mri = flipud(MRI.Cube(:,:,mnslice(3))');
    imagesc(mri)
    colormap('gray')
    axis equal
    grid off
    axis off
    hold on
    idx = coord_mri(:,1:2);
    if default_anat == 0
        scatter(idx(:,1), 257-idx(:,2),  50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    else
        scatter(idx(:,1), 257-idx(:,2),  50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    end
   
    cla(h6)
    h6 = subplot(3,4,11)
    mri = flipud(squeeze(MRI.Cube(:,mnslice(2),:))');
    imagesc(mri)
    axis equal
    grid off
    axis off
    hold on
    idx = coord_mri(:,[1,3]);
    if default_anat == 0
        scatter(idx(:,1), 257-idx(:,2), 50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    else
        scatter(idx(:,1), idx(:,2), 50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    end
    
    cla(h7)
    h7 = subplot(3,4,12)
    mri = flipud(fliplr(squeeze(MRI.Cube(mnslice(1),:,:))'));
    imagesc(mri)
    colormap('gray')
    hold on
    idx = coord_mri(:,[2,3]);
    if default_anat == 0
        scatter(257-idx(:,1), 257-idx(:,2), 50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    else
        scatter(257-idx(:,1), idx(:,2), 50, 'filled', 'MarkerFaceColor', ...
            c(clust_num,:), 'MarkerEdgeColor', 'k');
    end
    axis equal
    grid off
    axis off
     
end
   

function plotMEG(h)

    channel_groups{1} = {'MEG0113', 'MEG0112', 'MEG0111' 'MEG0132', 'MEG0133', 'MEG0131',...
        'MEG0143', 'MEG0142', 'MEG0141', 'MEG0213', 'MEG0212', 'MEG0211', ...
        'MEG0222', 'MEG0223', 'MEG0221', 'MEG0232', 'MEG0233', 'MEG0231', ...
        'MEG0243', 'MEG0242', 'MEG0241', 'MEG1512', 'MEG1513', 'MEG1511', ...
        'MEG1522', 'MEG1523', 'MEG1521', 'MEG1533', 'MEG1532', 'MEG1531', ...
        'MEG1543', 'MEG1542', 'MEG1541' 'MEG1613', 'MEG1612', 'MEG1611',...
        'MEG1622', 'MEG1623', 'MEG1621'};
        channel_groups{2} = {'MEG1312', 'MEG1313', 'MEG1311' 'MEG1322', 'MEG1323', 'MEG1321',...
        'MEG1333', 'MEG1332', 'MEG1331', 'MEG1343', 'MEG1342', 'MEG1341', ...
        'MEG1422', 'MEG1423', 'MEG1421', 'MEG1432', 'MEG1433', 'MEG1431', ...
        'MEG1443', 'MEG1442', 'MEG1441', 'MEG2412', 'MEG2413', 'MEG2411', ...
        'MEG2422', 'MEG2423', 'MEG2421', 'MEG2613', 'MEG2612', 'MEG2611', ...
        'MEG2623', 'MEG2622', 'MEG2621' 'MEG2633', 'MEG2632', 'MEG2631',...
        'MEG2642', 'MEG2643', 'MEG2641'};
        channel_groups{3} = {'MEG0412', 'MEG0413', 'MEG0411' 'MEG0422', 'MEG0423', 'MEG0421',...
        'MEG0433', 'MEG0432', 'MEG0431', 'MEG0443', 'MEG0442', 'MEG0441', ...
        'MEG0632', 'MEG0633', 'MEG0631', 'MEG0712', 'MEG0713', 'MEG0711', ...
        'MEG0743', 'MEG0742', 'MEG0741', 'MEG1632', 'MEG1633', 'MEG1631', ...
        'MEG1812', 'MEG1813', 'MEG1811', 'MEG1823', 'MEG1822', 'MEG1821', ...
        'MEG1833', 'MEG1832', 'MEG1831' 'MEG1843', 'MEG1842', 'MEG1841',...
        'MEG2012', 'MEG2013', 'MEG2011'};
        channel_groups{4} = {'MEG0722', 'MEG0723', 'MEG0721' 'MEG0732', 'MEG0733', 'MEG0731',...
        'MEG1043', 'MEG1042', 'MEG1041', 'MEG1113', 'MEG1112', 'MEG1111', ...
        'MEG1122', 'MEG1123', 'MEG1121', 'MEG1132', 'MEG1133', 'MEG1131', ...
        'MEG1143', 'MEG1142', 'MEG1141', 'MEG2022', 'MEG2023', 'MEG2021', ...
        'MEG2212', 'MEG2213', 'MEG2211', 'MEG2223', 'MEG2222', 'MEG2221', ...
        'MEG2233', 'MEG2232', 'MEG2231' 'MEG2243', 'MEG2242', 'MEG2241',...
        'MEG2442', 'MEG2443', 'MEG2441'};
        channel_groups{5} = {'MEG1643', 'MEG1642', 'MEG1641' 'MEG1712', 'MEG1713', 'MEG1711',...
        'MEG1723', 'MEG1722', 'MEG1721', 'MEG1733', 'MEG1732', 'MEG1731', ...
        'MEG1742', 'MEG1743', 'MEG1741', 'MEG1912', 'MEG1913', 'MEG1911', ...
        'MEG1923', 'MEG1922', 'MEG1921', 'MEG1932', 'MEG1933', 'MEG1931', ...
        'MEG1942', 'MEG1943', 'MEG1941', 'MEG2043', 'MEG2042', 'MEG2041', ...
        'MEG2113', 'MEG2112', 'MEG2111' 'MEG2143', 'MEG2142', 'MEG2141'};
        channel_groups{6} = {'MEG2032', 'MEG2033', 'MEG2031' 'MEG2122', 'MEG2123', 'MEG2121',...
        'MEG2133', 'MEG2132', 'MEG2131', 'MEG2313', 'MEG2312', 'MEG2311', ...
        'MEG2322', 'MEG2323', 'MEG2321', 'MEG2332', 'MEG2333', 'MEG2331', ...
        'MEG2343', 'MEG2342', 'MEG2341', 'MEG2432', 'MEG2433', 'MEG2431', ...
        'MEG2512', 'MEG2513', 'MEG2511', 'MEG2523', 'MEG2522', 'MEG2521', ...
        'MEG2533', 'MEG2532', 'MEG2531' 'MEG2543', 'MEG2542', 'MEG2541'};
        channel_groups{7} = {'MEG0122', 'MEG0123', 'MEG0121' 'MEG0312', 'MEG0313', 'MEG0311',...
        'MEG0323', 'MEG0322', 'MEG0321', 'MEG0333', 'MEG0332', 'MEG0331', ...
        'MEG0342', 'MEG0343', 'MEG0341', 'MEG0512', 'MEG07513', 'MEG0511', ...
        'MEG0523', 'MEG0522', 'MEG0521', 'MEG0532', 'MEG0533', 'MEG0531', ...
        'MEG0542', 'MEG0543', 'MEG0541', 'MEG0613', 'MEG0612', 'MEG0611', ...
        'MEG0623', 'MEG0622', 'MEG0621' 'MEG0643', 'MEG0642', 'MEG0641',...
        'MEG0822', 'MEG0823', 'MEG0821'};
        channel_groups{8} = {'MEG0812', 'MEG0813', 'MEG0811' 'MEG0912', 'MEG0913', 'MEG0911',...
        'MEG0923', 'MEG0922', 'MEG0921', 'MEG0933', 'MEG0932', 'MEG0931', ...
        'MEG0942', 'MEG0943', 'MEG0941', 'MEG1012', 'MEG1013', 'MEG1011', ...
        'MEG1023', 'MEG1022', 'MEG1021', 'MEG1032', 'MEG1033', 'MEG1031', ...
        'MEG1212', 'MEG1213', 'MEG1211', 'MEG1223', 'MEG1222', 'MEG1221', ...
        'MEG1233', 'MEG1232', 'MEG1231' 'MEG1243', 'MEG1242', 'MEG1241',...
        'MEG1412', 'MEG1413', 'MEG1411'};

        for j = 1:length(channel_groups)
            channels_group_idx{j} = [];
            for i = 1:length(channel_idx)
                if sum(~cellfun('isempty', strfind(cellstr(channel_groups{j}), channels.Channel(channel_idx(i)).Name))) == 1
                    channels_group_idx{j} = [channels_group_idx{j}, i];
                end
            end
        end
        
        
        clust_num = getappdata(h, 'cluster'); % number of cluster to show
        channels_group_idx{9} = channels_maxamp{clust_num};
           
        h8 = figure;
        valind = 1;
        chanind = channels_group_idx{valind};
        timeind = 1; % starting point for time
        num_time = int32(Fs*10); % number of time samples to show on one window
        datatime = Data.Time;
        
        bf_ts_norm = bf_ts(clust_num,:)/(norm(bf_ts(clust_num,:))/norm(Ff(1,:)));
        [fpoint, minv, maxv] = plotmatr(datatime(1:num_time), ...
        [bf_ts_norm(1:num_time); Ff(chanind, 1:num_time)], 1, ...
        2, 1, c(clust_num,:));

        marker = spike_ind(spike_ind<num_time);
        for k = 1:length(marker)
            h = line([datatime(marker(k)), datatime(marker(k))], [minv maxv], ...
                'color','k');
            set(h, 'LineStyle', ':', 'LineWidth', 1);
        end
    
        for clust = 1:size(cluster,2)
            spike_time = cluster{1, clust}(2,:);
            marker = spike_time(spike_time<num_time);
            for k = 1:length(marker)
                h = line([datatime(marker(k)), datatime(marker(k))], [minv maxv], ...
                    'color', c(clust,:));
                set(h,'LineStyle', ':', 'LineWidth', 2);
            end
        end      
        
        yticks(sort(double(fpoint), 'ascend'))
        yticklabels(['Bf', namechan(chanind)])
                
        step = (num_time/size(Ff,2));
        lims = [step, 1];
        sld1 = uicontrol('Style', 'slider',...
        'Min',1,'Max',size(Ff,2)-num_time-mod((size(Ff,2)-num_time), 100),...
        'Value',1,...
        'Position', [300 50 1300 20],...
        'SliderStep', [0.025 1], ... % step = 0.01*T
        'Callback', @slidetime);
    
        function slidetime(source, event)
        timeind = int32(source.Value);
        cla(h8)

        [fpoint, minv, maxv] = plotmatr(datatime(timeind:(timeind+num_time-1)), ...
            [bf_ts_norm(timeind:(timeind+num_time-1)); ...
            Ff(chanind,timeind:(timeind+num_time-1))],timeind, 2, 1, c(clust_num,:));
        
        marker = spike_ind((spike_ind>timeind)&(spike_ind<(timeind+num_time-1)));
        for k = 1:length(marker)
            h = line([datatime(marker(k)), datatime(marker(k))], [minv maxv], ...
                'color','k');
            set(h, 'LineStyle', ':', 'LineWidth', 1);
        end
        
        for clust = 1:size(cluster,2)
            spike_time = cluster{1, clust}(2,:);
            marker = spike_time((spike_time>timeind)&(spike_time<(timeind+num_time-1)));
            for k = 1:length(marker)
                h = line([datatime(marker(k)), datatime(marker(k))], ...
                    [minv maxv], 'color', c(clust,:));
                set(h, 'LineStyle', ':','LineWidth', 2);
            end
        end

        yticks(sort(double(fpoint), 'ascend'))
        yticklabels(['Bf', namechan(chanind)])           
        end
        
        popup = uicontrol('Style', 'popup',...
       'String', {'Left temporal'; 'Right temporal'; 'Left parietal'; 
       'Right parietal'; 'Left occipital'; 'Right occipital'; 'Left frontal'; ...
       'Right frontal'; '20 top amplitude'},...
       'Position', [100 5 120 20],...
       'Callback', @changechannels);
       
        function changechannels(source, event)
            valchan = source.Value;
            chanind = channels_group_idx{valchan};

             [fpoint, minv, maxv] = plotmatr(datatime(timeind:(timeind+num_time-1)), ...
                [bf_ts_norm(timeind:(timeind+num_time-1)); ...
                Ff(chanind, timeind:(timeind+num_time-1))], ...
            timeind, 2, 1, c(clust_num,:));

            marker = spike_ind((spike_ind>timeind)&(spike_ind<(timeind+num_time-1)));
            for k = 1:length(marker)
                h = line([datatime(marker(k)), datatime(marker(k))], [minv maxv], ...
                    'color','k');
                set(h, 'LineStyle', ':', 'LineWidth', 1);
            end

            for clust = 1:size(cluster,2)
                spike_time = cluster{1, clust}(2,:);
                marker = spike_time((spike_time>timeind)&(spike_time<(timeind+num_time-1)));
                for k = 1:length(marker)
                    h = line([datatime(marker(k)), datatime(marker(k))], ...
                        [minv maxv], 'color', c(clust,:));
                    set(h, 'LineStyle', ':', 'LineWidth', 2);
                end
            end

            yticks(sort(double(fpoint), 'ascend'))
            yticklabels(['Bf', namechan(chanind)])   
        end
            
                

end
end