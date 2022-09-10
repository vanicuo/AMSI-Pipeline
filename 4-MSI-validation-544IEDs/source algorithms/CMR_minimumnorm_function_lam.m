function [w]=CMR_minimumnorm_function_lam(A,C,snr,noiselambda,doscale,dowhiten,wmne_p,id_keep,lam)
%A=leadfiled
%C=sensor noise
%wmne  depth weighted
%x=w*S

Nsource=size(A,2);  
if wmne_p
Depth_Weighted=diag(A'*A);
% diagR=(1./(Depth_Weighted.^1)).^2;
% inv_diagR=1./diagR;
% R=diag(inv_diagR);
R=diag((1./Depth_Weighted).^wmne_p);
% omega_keep=ones(length(id_keep),1);
% 
%     for i=1:length(id_keep)
%         if Nsource==2*length(id_keep)
%            temp=Depth_Weighted(2*i-1)+Depth_Weighted(2*i);
%         else
%            temp=Depth_Weighted(3*i-1)+Depth_Weighted(3*i-2)+Depth_Weighted(3*i);
%         end
%         omega_keep(i)=temp^wmne_p;
%     end 
% 
%     inv_diagR=1./omega_keep;
%     R=diag(inv_diagR);
%         
%     if Nsource==2*length(id_keep)
%          R=kron(R,diag([1 1]));
%     else
%          R=kron(R,diag([1 1 1]));
%     end
    
else
    
R=speye(Nsource);                     %R =source covariance 
end

if dowhiten,
  fprintf('prewhitening the leadfields using the noise covariance\n');
  % compute the prewhitening matrix
  if ~isempty(noiselambda)
    fprintf('using a regularized noise covariance matrix\n');
    % note: if different channel types are present, one should probably load the diagonal with channel-type specific stuff
    [U,S,V] = svd(C+eye(size(C))*noiselambda);
  else
    [U,S,V] = svd(C);
  end

  Tol     = 1e-12;
  diagS   = diag(S);
  sel     = find(diagS>Tol.*diagS(1));
  P       = diag(1./sqrt(diag(S(sel,sel))))*U(:,sel)'; % prewhitening matrix
  A       = P*A;              % prewhitened leadfields
  C       = eye(size(P,1));   % prewhitened noise covariance matrix
end

if doscale,
    scale = trace(A*(R*A'))/trace(C);
    R     = R./scale;
end

if ~isempty(snr)
    lambda = trace(A * R * A')/(trace(C)*snr^2);
else
    lambda = lam
end

if dowhiten,
      % as documented on MNE website, this is replacing the part of the code below, it gives
      % more stable results numerically.
      Rc      = chol(R, 'lower');
      [U,S,V] = svd(A * Rc, 'econ');
      s  = diag(S);
      ss = s ./ (s.^2 + lambda);
      w  = Rc * V * diag(ss) * U';   
      % unwhiten the filters to bring them back into signal subspace
      w = w*P;
   
else
      % equation 5 from Lin et al 2004 (this implements Dale et al 2000, and Liu et al. 2002)
      denom = (A*R*A'+(lambda^2)*C);
      if cond(denom)<1e12
        w = R * A' / denom;
      else
        fprintf('taking pseudo-inverse due to large condition number\n');
        w = R * A' * pinv(denom);
      end
end

end

