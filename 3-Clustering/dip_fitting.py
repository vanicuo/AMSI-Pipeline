import torch
import mne
import numpy as np


def dip_do_fit_PCA(data, info, SampWin=5, initRes=10, inter=4, Jobs=10, device=torch.device('cpu')):
    dipRes = initRes / 1000
    if SampWin == 'PCA':
        Dat = torch.tensor(data).clone().to(device)
        U, E, V = torch.linalg.svd(Dat)
        Dat = U[:, :, 0].t()
    else:
        Dat = torch.tensor(data)[:, :,
              round(data.shape[-1] / 2) + torch.arange(-SampWin, SampWin)].mean(dim=-1).to(device).t()
    # First low resolution
    lf_X_lf_inv, lf_inv, src_pos, Idx = dip_get_lf(info, pos=dipRes * 1000, device=device)
    dip_pos, dip_error, dip_ori = dip_fit(lf_X_lf_inv=lf_X_lf_inv, lf_inv=lf_inv, Dat=Dat, src_pos=src_pos,
                                 Chunk=int(np.ceil(2000 / Dat.shape[1])), device=device)
    # Second higher resolution
    for i in range(inter):
        dip_pos_unique, invIdx = torch.unique(dip_pos, dim=0, return_inverse=True)
        pos = [dip_sub_sample(center=x, rad=dipRes) for x in dip_pos_unique]
        lf_X_lf_inv, lf_inv, src_pos, Idx = dip_get_lf(info, pos=pos, Idx=invIdx, Jobs=Jobs, device=device)
        dip_pos, dip_error, dip_ori = dip_fit(lf_X_lf_inv=lf_X_lf_inv, lf_inv=lf_inv, Dat=Dat, src_pos=src_pos, Idx=Idx,
                                     Chunk=int(np.ceil(2000 / Dat.shape[1])), device=device)
        dipRes = dipRes / 2
    fitGood = 1 - dip_error
    return dip_pos, dip_ori, fitGood


def dip_get_lf(info, pos=30, Idx=None, Jobs=10, device=torch.device('cpu')):
    # Merge grid Pos and get Idx (Event*1 list, the Idx of grid pos of each event)
    if isinstance(pos, list):
        posTemp = {'rr': np.array([[], [], []]).T, 'nn': np.array([[], [], []]).T}
        for x in pos:
            for k, v in x.items():
                posTemp[k] = np.concatenate([posTemp[k], v])
        posIdx = torch.cat([torch.tensor([i]).repeat(x['rr'].shape[0]) for (i, x) in enumerate(pos)]).to(
            device)
        posTemp['rr'], invIdx = torch.unique(torch.tensor(posTemp['rr']), dim=0, return_inverse=True)
        posTemp['nn'] = posTemp['nn'][:posTemp['rr'].shape[0]]
        Idx = [invIdx[torch.where(posIdx == x)[0]].unique().to(device) for x in Idx]
        pos = posTemp
    # Get lf
    bem = mne.make_sphere_model(r0='auto', head_radius=None, info=info, sigmas=(0.33,), relative_radii=(1.0,),
                                verbose='error')
    src = mne.setup_volume_source_space(pos=pos, sphere_units='m', verbose='error', sphere=np.append(bem['r0'], 0.09))
    fwd = mne.make_forward_solution(info=info, trans=None, src=src, bem=bem, verbose='error', n_jobs=Jobs)
    # Get lf, reduce_rank lf, lf * lf_inv
    lf = torch.tensor(fwd['sol']['data']).to(device).t().reshape(fwd['nsource'], 3, fwd['nchan']).float()
    lf = ReduceRank_LF(lf, Rank=2, device=device).transpose(-1, -2)
    lf_inv = torch.linalg.pinv(lf).float()
    lf_X_lf_inv = torch.bmm(lf.cpu(), lf_inv.cpu())
    src_pos = torch.tensor(fwd['source_rr']).to(device)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return lf_X_lf_inv, lf_inv, src_pos, Idx


def dip_fit(lf_X_lf_inv=None, lf_inv=None, src_pos=None, Dat=None, Idx=None, Chunk=2000, device=torch.device('cpu')):
    # get chunk for save GPU RAM
    Chunk = [torch.tensor(range(x * Chunk, min(lf_X_lf_inv.shape[0], (x + 1) * Chunk))).to(device)
             for x in range(np.ceil(lf_X_lf_inv.shape[0] / Chunk).astype('int'))]
    # get grid_error
    grid_error = torch.cat([torch.pow(torch.bmm(
        torch.eye(lf_X_lf_inv.shape[1]).repeat(x.shape[0], 1, 1).to(device) - lf_X_lf_inv[x].to(device),
        Dat.repeat(x.shape[0], 1, 1)), 2).sum(dim=1) for x in Chunk])
    if Idx is None:
        grid_Idx = grid_error.argmin(dim=0)
        dip_pos = src_pos[grid_Idx.cpu()].clone().detach()
        dip_error = grid_error.amin(dim=0) / torch.pow(Dat, 2).sum(dim=0)
        # get ori
        src = torch.bmm(lf_inv[grid_Idx.cpu()], Dat.t().unsqueeze(-1))
        dip_ori = torch.linalg.svd(torch.bmm(src, src.transpose(-1, -2)))[0][:, :1].transpose(1, 2)
    else:
        grid_Idx = torch.tensor([grid_error[x, i].argmin() for (i, x) in enumerate(Idx)])
        dip_pos = torch.stack([src_pos[Idx[i][x]] for (i, x) in enumerate(grid_Idx)])
        dip_error = torch.stack([grid_error[Idx[i][x], i] for (i, x) in enumerate(grid_Idx)]) / \
                    torch.pow(Dat, 2).sum(dim=0)
        # get ori
        src = torch.bmm(torch.stack([lf_inv[Idx[i][x]] for (i, x) in enumerate(grid_Idx)]), Dat.t().unsqueeze(-1))
        dip_ori = torch.linalg.svd(torch.bmm(src, src.transpose(-1, -2)))[0][:, :1].transpose(1, 2)
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return dip_pos.cpu().float(), dip_error.cpu(), (dip_ori.squeeze() / torch.linalg.norm(dip_ori, dim=1)).cpu()


def dip_sub_sample(center=None, rad=0.03, device=torch.device('cpu')):
    rr = torch.meshgrid(torch.tensor([center[0] - rad / 2, center[0] + rad / 2]),
                        torch.tensor([center[1] - rad / 2, center[1] + rad / 2]),
                        torch.tensor([center[2] - rad / 2, center[2] + rad / 2]))
    x, y, z = rr[0].ravel(), rr[1].ravel(), rr[2].ravel()
    rr = torch.cat((torch.stack([x, y, z]).t().to(device), center.reshape(1, -1).to(device)))
    pos = {'rr': rr.cpu().numpy(), 'nn': np.array([0., 0., 1.]).reshape(1, -1).repeat(rr.shape[0], 0)}
    return pos


def ReduceRank_LF(lf, Rank=2, device=torch.device('cpu')):
    # Compute Reduced rank LeadField
    # SVD(LF), U*S(Rank)*V'
    U, s, V = torch.linalg.svd(lf.to(device), full_matrices=True)
    V = V.transpose(-2, -1)
    s[:, Rank:] = 0
    return torch.bmm(torch.bmm(
        U, torch.stack([torch.cat([x.diag(), torch.zeros(x.shape[0], V.shape[1] - x.shape[0]).to(device)],
                                  dim=1) for x in s], dim=0)), V.transpose(1, 2))
