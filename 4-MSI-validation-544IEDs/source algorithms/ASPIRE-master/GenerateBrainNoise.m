function [ BrainNoise ] = GenerateBrainNoise(G,T,Tw,N,Fs)

Nsrc = size(G,2);
SrcIndex = fix(rand(1,N)*Nsrc+1);

q = randn(N,T+2*Tw);

alpha_band  = [8,  12]/(Fs/2);
beta_band   = [15, 30]/(Fs/2);
gamma1_band = [30, 50]/(Fs/2);
gamma2_band = [50, 70]/(Fs/2);


[b_alpha, a_alpha] = butter(4,alpha_band);
[b_beta, a_beta] = butter(4,beta_band);
[b_gamma1, a_gamma1] = butter(4,gamma1_band);
[b_gamma2, a_gamma2] = butter(4,gamma2_band);

SourceNoise = 1/mean(alpha_band)* filtfilt(b_alpha,a_alpha,q')' + ...
              1/mean(beta_band)*filtfilt(b_beta,a_beta,q')' + ...
              1/mean(gamma1_band)*filtfilt(b_gamma1,a_gamma1,q')'+...
              1/mean(gamma2_band)*filtfilt(b_gamma2,a_gamma2,q')';

BrainNoise = G(:,SrcIndex)*SourceNoise(:,Tw+1:Tw+T)/N;



end

