subj_name = 'B1C2';
protocol_dir = '/home/ksasha/Documents/brainstorm_db/MEG_Tommaso/';
file_name = 'B1C2_ii_run1_raw_tsss_mc_art_corr';
channel_type = 'grad'; % channels you want to analyse ('grad' or 'mag')

MRI = load(strcat([protocol_dir, 'anat/', subj_name, '/subjectimage_T1.mat']));
Data = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name,'/data_block001.mat']));
channels = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name, '/channel_vectorview306_acc1.mat']));
G3 = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name, '/headmodel_surf_os_meg.mat']));

% Here load your manually detected spikes
spike_time = csvread('Manual_spikes_B1C2_ii_run1_raw_tsss_mc.csv');
spike_time = spike_time(spike_time<600); % pick only the first 10 minutes 
spike_time = round(spike_time*1000); % replace seconds for time samples

% Be careful, the output of the next step is the set of plots -- one for
% each event you have in spike_time!
% spike_time -- should be time samples, not seconds!

f_low = 2;
f_high = 50;
plot_individual(spike_time, Data, f_low, f_high, ...
    channel_type, G3, MRI, channels)