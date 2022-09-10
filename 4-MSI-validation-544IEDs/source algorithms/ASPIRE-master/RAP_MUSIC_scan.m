function [Valmax, Indmax] = RAP_MUSIC_scan(spike, Gain, G2, thresh)
% -------------------------------------------------------
% RAP-MUSIC scan
% -------------------------------------------------------
% INPUTS:
%   spike -- Nch x T MEG data with spike
%   Gain -- Nch x Nsources*3 forward operator
%   G2 -- Nch x Nsources*2 forward operator in tangential plane
%   thresh -- minimal considered correlation value
%
% OUTPUT:
%   Valmax -- subcorr values for all found dipoles higher than threshold
%   IndMax -- location of this sources
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

[Ns, Nsrc2] = size(G2);
Nsrc = Nsrc2/2;

Valmax = [];
Indmax = [];
[U,S,V] = svd(spike);
h = cumsum(diag(S)/sum(diag(S)));
n = find(h>=0.95);
corr = MUSIC_scan(G2, U(:,1:n(1)));
[valmax, indmax] = max(corr);

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

end

