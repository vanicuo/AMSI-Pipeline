subj_name = 'B1C2';
protocol_dir = '/home/ksasha/Documents/brainstorm_db/MEG_Tommaso/';
file_name = 'B1C2_ii_run1_raw';
channel_type = 'grad'; % channels you want to analyse ('grad' or 'mag')

cortex = load(strcat([protocol_dir, 'anat/', subj_name,'/tess_cortex_pial_low.mat']));
G3 = load(strcat([protocol_dir, 'data/', subj_name, '/', file_name, '/headmodel_surf_os_meg.mat']));

if strcmp(channel_type, 'grad') == 1
    channel_idx = setdiff(1:306, 3:3:306);
elseif strcmp(channel_type, 'mag') == 1
    channel_idx = 3:3:306;
end

[G2, G20, Nsites] = G3toG2(G3, channel_idx);

% MUSIC test
ind_target = randperm(size(G3.Gain, 2)/3);
ind_target = ind_target(1);
I = ind_target*2;

snr = 1; % snr level in the data
Fs = 1000; % sampling frequency
T = 200; % number of time points in one trial

Noise = GenerateBrainNoise(G20,T,200,500,Fs);
Noise_0 = Noise/norm(Noise); % normalized noise

t = linspace(0,1,1000);
s = sin(2*pi*5*t(1:T)).*exp(-10*(t(1:T)+0.2).^2);
           
X = G20(:,I)*s; % signal, activations only by y-axis of G
X_0 = X/norm(X);

spike  = snr*X_0 + Noise_0; % add noise to the data

figure
plot(spike')

[U,S,V] = svd(spike);
h = cumsum(diag(S)/sum(diag(S)));
n = find(h>=0.95);
corr = MUSIC_scan(G2, U(:,1:n(1)));
[ValMax, IndMax] = max(corr);
IndMax == ind_target
ValMax

% RAP-MUSIC test
ind_target = randperm(size(G3.Gain, 2)/3);
ind_target = ind_target(1:2);
I = ind_target*2;

Fs = 1000; % sampling frequency
T = 200; % number of time points in one trial
Noise = GenerateBrainNoise(G20,T,200,500,Fs);
Noise_0 = Noise/norm(Noise); % normalized noise

t = linspace(0,1,1000);
s(1,:) = sin(2*pi*6*t(1:T)).*exp(-10*(t(1:T)+0.2).^2);
s(2,:) = sin(2*pi*6*t(1:T)+randn*0.1).*exp(-10*(t(1:T)+0.2 +randn*0.2).^2);

% figure
% plot(s(1,:))
% hold on
% plot(s(2,:))

X = G20(:,I(1))*s(1,:)+G20(:,I(2))*s(2,:); % signal, activations only by y-axis of G
X_0 = X/norm(X);
snr = 5; 
spike  = snr*X_0 + Noise_0; % add noise to the data

figure
plot(spike')

[Ns, Nsrc2] = size(G2);
Nsrc = Nsrc2/2;
Valmax = [];
Indmax = [];
[U,S,V] = svd(spike);
h = cumsum(diag(S)/sum(diag(S)));
n = find(h>=0.95);
corr = MUSIC_scan(G2, U(:,1:n(1)));
[valmax, indmax] = max(corr);

Gain = G3.Gain(channel_idx,:);
while valmax > thresh
    Valmax = [Valmax, valmax];
    Indmax = [Indmax, indmax];
    
    A = Gain(:,(indmax*3-2):indmax*3);
    P = eye(Ns, Ns)-A*inv(A'*A)*A';
    spike_proj = P*spike;
    G_proj = P*Gain;
    Gain = G_proj;
    
    G2 = zeros(Ns,2*Nsrc);
    range = 1:2;
    for i = 1:Nsrc
        g = [G_proj(:,1+3*(i-1)) G_proj(:,2+3*(i-1)) G_proj(:,3+3*(i-1))];
        [u sv v] = svd(g);
        gt = g*v(:,1:2);
        G2(:,range) = gt*diag(1./sqrt(sum(gt.^2,1)));
        range = range + 2;
    end

    [U,S,V] = svd(spike_proj);
    h = cumsum(diag(S)/sum(diag(S)));
    n = find(h>=0.95);
    corr = MUSIC_scan(G2, U(:,1:n(1)));
    [valmax, indmax] = max(corr);
end

