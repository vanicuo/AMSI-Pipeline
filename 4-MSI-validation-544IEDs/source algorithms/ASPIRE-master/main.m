% 1. Export everything from brainstorm
subj_name = 'B1C2';
protocol_dir = '/home/ksasha/Documents/brainstorm_db/MEG_Tommaso/';
file_name = 'B1C2_ii_run1_raw_tsss_mc_art_corr';
channel_type = 'grad'; % channels you want to analyse ('grad' or 'mag')
default_anat = 0; % 0 if individual anatomy, 1 if default

if default_anat == 0
    cortex = load(strcat([protocol_dir, 'anat/', subj_name,'/tess_cortex_pial_low.mat']));
    MRI = load(strcat([protocol_dir, 'anat/', subj_name, '/subjectimage_T1.mat']));
else
    cortex = load(strcat([protocol_dir, 'anat/@default_subject','/tess_cortex_pial_low.mat']));
    MRI = load(strcat([protocol_dir, 'anat/@default_subject', '/subjectimage_T1.mat']));
end    
Data = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name,'/data_block001.mat']));
channels = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name, '/channel_vectorview306_acc1.mat']));
G3 = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name, '/headmodel_surf_os_meg.mat']));

% 2. ICA-based spike detection
decision = 0.9; % the amplitude threshold for decision 
f_low = 3; % bandpass filter before the ICA decomposition
f_high = 70;
[spike_ind, picked_components, picked_comp_top] = ...
    ICA_detection(Data, G3, channel_type, decision, f_low, f_high);

% 3. RAP-MUSIC (2) dipole fitting
f_low = 10;
f_high = 200;
spikydata = 0;
corr_thresh = 0.89;
RAP = 'RAP';
[IndMax, ValMax, ind_m, spikeind] = spike_localization(spike_ind, Data, G3, ...
    channel_type, f_low, f_high, spikydata, picked_components, ...
    picked_comp_top, corr_thresh, RAP);

% 4. Clustering
thr_dist = 0.01; % maximal distance from the center of the cluster (radius)
Nmin = 8; % minimum number of sources in one cluster
cluster = clustering(spike_ind, G3, Nmin, ValMax, IndMax, ind_m, ...
    thr_dist, 1, cortex, RAP, spikeind);

% 5. Activation on sources 
f_low = 3;
f_high = 50;
[spike_trials, maxamp_spike_av, channels_maxamp, spike_ts] = ...
                source_reconstruction(Data, G3, channel_type, cluster, ...
                f_low, f_high);

% 6. Big plot
f_low = 2; % bandpass filter for visualization
f_high = 50;
epi_plot(Data, channel_type, f_low, f_high, cortex, ...
    spike_trials, maxamp_spike_av, spike_ts, cluster, ...
    channels, G3, MRI, corr_thresh, default_anat, channels_maxamp, spike_ind)

% 7. Individual plot for manually detected spikes
spike_time = csvread('Manual_spikes_B1C2_ii_run1_raw_tsss_mc.csv');
spike_time = spike_time(spike_time<600);
spike_time = round(spike_time*1000);

f_low = 2;
f_high = 50;
plot_individual(spike_time, Data, f_low, f_high, ...
    channel_type, G3, MRI, channels)
