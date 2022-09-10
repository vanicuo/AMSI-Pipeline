function plot_brain_cmap2(surf_hr, surf_lr, vert_inds_lr, data_lr, mask_lr, brain_opacity)
% DESCRIPTION:
% -----------
% Extrapolate low-resolution data on high resolution brain surface
% and plot as a colormap
%
% USAGE:
% colormap(hsv); h = plot_brain_cmap(hr_infl, surf_lr, vert_inds_lr, data_lr, mask_lr, [])
%
% INPUT:
% -----------
% surf_hr and surf_lr should be exported from brainstorm
% (typical name for surf_hr would be cortex_37718V)
%
% SETUP vert_inds_lr:
% Import data for low res. surface from brainstorm and input:
% vert_inds_lr = [test.Atlas.Scouts.Vertices];
% in this case size(vert_inds_lr) should be the same as size(data_lr)
%
% OR if data is present for each vertex in surf_lr, set
% vert_inds_lr = []
%
% SETUP data_lr
% data_lr must be a column vector defined for each
% vertex in  either surf_lr or vert_inds_lr
%
% SETUP mask_lr
% mask_lr should have the same size as data_lr
% mask_lr setup example:
% mask_lr = data_lr < (max(data_lr) + min(data_lr)) / 2 * 0.7;
%
% SETUP atlas:
% Import original surface from bst (not inflated)
% to a variable surf_hr_orig and run
% >> atlas = surf_hr_orig.Atlas(2)
% to get Destrieux atlas
%
% NOTE:
% With atlas produced pictures look ugly by now.
% Better stick with atlas = []
%
% brain_opacity - scalar between 0 and 1. default = 1
%
% OUTPUT:
% returns handle to camlight;
% to reset lighting after moving the camera, run
% camlight(h,'left')
% ____________________________________________________________

    if nargin < 6
        brain_opacity = 1;
    end

    if nargin < 5
        mask_lr = zeros(size(data_lr));
    end

    verts_hr = surf_hr.Vertices;
    faces_hr = surf_hr.Faces;

    vx_hr = verts_hr(:,1);
    vy_hr = verts_hr(:,2);
    vz_hr = verts_hr(:,3);

    % --- Constrain lowres vertices to subset --- %
    % Useful when lowres surface contains subcortical structures
    % but data are present only on cortex
    verts_lr = surf_lr.Vertices;
    if isempty(vert_inds_lr)
        vert_inds_lr = 1:size(data_lr, 1);

    end
    vx_lr_ = verts_lr(vert_inds_lr,1);
    vy_lr_ = verts_lr(vert_inds_lr,2);
    vz_lr_ = verts_lr(vert_inds_lr,3);
    c_lr = data_lr;
    % ------------------------------------ %

    % ------------ Extrapolate data onto HR surface ------- %
    F_data = scatteredInterpolant(vx_lr_, vy_lr_, vz_lr_, c_lr);
    c_hr = F_data(vx_hr, vy_hr, vz_hr);

    % ----------- Extrapolate mask onto HR surface ----------- %
    F_mask = scatteredInterpolant(vx_lr_, vy_lr_, vz_lr_, double(mask_lr));
    mask_extrap = F_mask(vx_hr, vy_hr, vz_hr);
    mask_hr = ~(mask_extrap < 0.5);

    % ---- Mask out data --- %
    r = 1:length(vz_hr);
    bad_inds = r(~mask_hr);
    faces_hr_ = prune_faces(faces_hr, bad_inds);

    % ---- Plot brain ---- %
    % brain_col = [0.9,0.9,0.9];
    % trisurf(faces_hr, vx_hr, vy_hr, vz_hr, 'EdgeColor', 'None', 'FaceColor', brain_col, 'FaceAlpha', brain_opacity);

    % % ---- Plot atlas if present ---- %
    % if ~isempty(atlas)
    %     cmap = lines(length(scouts));
    %     for i = 1:length(scouts)
    %         verts_id_sc = scouts{i};
    %         faces_sc = prune_faces(faces_hr, verts_id_sc);
    %         hold on;
    %         trisurf(faces_sc, vx_hr, vy_hr, vz_hr, 'EdgeColor', 'None', 'FaceAlpha', 0.8, 'FaceColor', cmap(i,:) );
    %     end
    % end

    % ---- Plot masked data ---- %
    c_hr = repmat(0.1, length(verts_hr), 1)';
    hold on;
    trisurf(faces_hr_, vx_hr, vy_hr, vz_hr, c_hr, 'EdgeColor', 'None', 'FaceAlpha', brain_opacity);
    axis equal
    grid off
    axis off
    view(360,360)
    h = camlight('left');
    lighting phong;
    material dull
%    hold on
%    sphere_marker(vertices(ind_n(ind),1),vertices(ind_n(ind),2), vertices(ind_n(ind),3), 0.003, [0,0,0])
  
    %     cm_viridis = viridis(100);
%     cm_inferno = inferno(100);
%     cm_magma = magma(100);
%     cm_plasma = plasma(100);
%     colormap(cm_plasma);

    % - Set lighting - %

end


function faces_ = prune_faces(faces, idx)
% Remove faces if one of 3 vertex indices is in idx
% ________________________________________________

    mask_faces = ismember(faces(:,1), idx) | ismember(faces(:,2), idx) |  ismember(faces(:,3), idx);
    faces_ = faces(mask_faces,:);
end