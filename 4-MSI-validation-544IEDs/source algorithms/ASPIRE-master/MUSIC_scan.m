function corr = MUSIC_scan(G2, U)
% -------------------------------------------------------
% One step of RAP-MUSIC scan
% -------------------------------------------------------
% INPUTS:
%   G2 -- Nch x Nsources*2 forward model matrix
%   U -- Nch x k signal subspace matrix
%
% OUTPUT:
%   corr -- subspace correlations for all sources
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com

[Nsns, Nsrc2] = size(G2);
Nsrc = Nsrc2/2;

tmp = U'*G2;

c11c22 = sum(tmp.^2, 1);
tmp1 = tmp(:,1:2:2*Nsrc);
tmp2 = tmp(:,2:2:2*Nsrc);
c12 = sum(tmp1.*tmp2, 1);

tr = c11c22(1:2:2*Nsrc)+c11c22(2:2:2*Nsrc);
d = c11c22(1:2:2*Nsrc).*c11c22(2:2:2*Nsrc) - c12.^2;

l1 = sqrt(0.5*(tr+sqrt(tr.^2-4*d)));
l2 = sqrt(tr-l1.^2);
corr = max(l1, l2);

end

