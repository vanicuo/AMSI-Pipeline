function [G2d, G2d0, Nsites] = G3toG2(G3, ChUsed)
% -------------------------------------------------------------------------
% Compute the forward operator for MEG without the radial component
% -------------------------------------------------------------------------
% INPUTS:
%   G3 -- brainstorm structure with forward operator
%   ChUsed -- indices of channels used
%   
% OUTPUTS:
%   G2d -- normalized forward operator with tangential dipoles
%   G2d0 -- non-normalized forward operator with tangential dipoles
%   Nsites -- number of sources in the cortical model
% _______________________________________________________
% Aleksandra Kuznetsova, kuznesashka@gmail.com
% Alexei Ossadtchi, ossadtchi@gmail.com
    
    [Nch, Nsites] = size(G3.Gain(ChUsed,1:3:end));
    G_pure = G3.Gain(ChUsed,:); % 2D dense forward matrix 
    G2d = zeros(Nch,Nsites*2);
    G2d0 = zeros(Nch,Nsites*2);
    range = 1:2;
    for i = 1:Nsites
        g = [G_pure(:,1+3*(i-1)) G_pure(:,2+3*(i-1)) G_pure(:,3+3*(i-1))];
        [u sv v] = svd(g);
        gt = g*v(:,1:2);
        G2d(:,range) = gt*diag(1./sqrt(sum(gt.^2,1)));
        G2d0(:,range) = gt;
        range = range + 2;
    end
end