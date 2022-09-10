function d=find_min_dist(headpoint_auto,Vertices)
  d=rownorm(Vertices-repmat(headpoint_auto(1,:),size(Vertices,1),1));
  d=min(d);
end