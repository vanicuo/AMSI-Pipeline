function [spike_ind, picked_components, picked_comp_top] = ...
    ICA_detection(Data, G3, channel_type, decision, f_low, f_high)

% -------------------------------------------------------------------------
% ICA-based spike detection
% -------------------------------------------------------------------------
% INPUTS:
%   Data -- brainstorm structure with artifact corrected maxfiltered MEG data
%   G3 -- brainstorm structure with forward operator
%   channel_type -- channels used ('grad' or 'mag')
%   decision -- the threshold to pick the spikes
%   f_low, f_high -- the bands for prefiltering before ICA
%   
% OUTPUTS:
%   spike_ind -- time stamps with potentially spiking events
%   picked_components --
%   picked_comp_top -- 
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

Fs = 1/(Data.Time(2)-Data.Time(1)); % sampling frequency

% channels: gradiometers or magnetometers
if strcmp(channel_type, 'grad') == 1
    grad_idx = setdiff(1:306, 3:3:306);
    channel_idx = grad_idx(Data.ChannelFlag(grad_idx)~=-1);
elseif strcmp(channel_type, 'mag') == 1
    magn_idx = 3:3:306;
    channel_idx = magn_idx(Data.ChannelFlag(magn_idx)~=-1);
end

% 2D forward operator without radial component
[G2, ~] = G3toG2(G3, channel_idx); 

% ICA-based detection
ncomp = 40; % number of computed components
check = 1;
num_event = [];

if length(Data.Events) > 0 
    for i = 1:length(Data.Events)
        events_name{i} = Data.Events(i).label;
    end

    if ismember('BAD', events_name)
        addit = Data.Time(1);
        num_event = find(strcmp(events_name, 'BAD'));
        bad_idx = [];
        for i = 1:size(Data.Events(num_event).times, 2)
            bad_idx = [bad_idx, int32((Data.Events(num_event).times(1,i)-addit+0.001)*Fs):...
                int32((Data.Events(num_event).times(2,i)-addit+0.001)*Fs)];
        end
    else
        bad_idx = [];
    end
else
    bad_idx = [];
end

[spike_ind, picked_components, picked_comp_top] = SpikeDetect(Data, channel_idx, G2, f_low, f_high, ncomp, check, decision,...
    bad_idx, num_event);

end
