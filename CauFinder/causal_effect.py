import numpy as np
import torch
import torch.nn.functional as F


def joint_uncond_v1(params, model, data, index, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond:
        Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
    Inputs:
        - params['N_alpha'] monte-carlo samples per causal factor
        - params['N_beta']  monte-carlo samples per noncausal factor
        - params['K']      number of causal factors
        - params['L']      number of non-causal factors
        - params['M']      number of classes (dimensionality of classifier output)
        - model
        - data
        - device
    Outputs:
        - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
        - info['xhat']
        - info['yhat']
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)
    # mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], params['K']), device=device).mul(alpha_std).add_(alpha_mu).repeat(1, params[
        'N_beta']).view(params['N_alpha'] * params['N_beta'], params['K'])
    beta = torch.randn((params['N_alpha'] * params['N_beta'], params['L']), device=device).mul(beta_std).add_(beta_mu)
    zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    logit, prob = model.dpd_model(zs).values()

    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return -I, None


def beta_info_flow_v1(params, model, data, index, alpha_vi=True, beta_vi=False, eps=1e-8, device=None):
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'] * params['N_beta'], params['K']), device=device).mul(alpha_std).add_(
        alpha_mu)
    beta = torch.randn((params['N_alpha'], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], params['L'])

    zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    logit, prob = model.dpd_model(zs).values()

    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return -I, None


def joint_uncond_single_dim_v1(params, model, data, index, dim, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond_single_dim:
        Sample-based estimate of "joint, unconditional" causal effect
        for single latent factor, -I(z_i; Yhat). Note the interpretation
        of params['Nalpha'] and params['Nbeta'] here: Nalpha is the number
        of samples of z_i, and Nbeta is the number of samples of the other
        latent factors.
    Inputs:
        - params['Nalpha']
        - params['Nbeta']
        - params['K']
        - params['L']
        - params['M']
        - model
        - data
        - device
        - dim (i : compute -I(z_i; Yhat) **note: i is zero-indexed!**)
    Outputs:
        - negCausalEffect (sample-based estimate of -I(z_i; Yhat))
        - info['xhat']
        - info['yhat']
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x1, _ = model.feature_mapper(feat, mode='causal')
    latent1 = model.encoder1(x1)
    x2, _ = model.feature_mapper(feat, mode='spurious')
    latent2 = model.encoder2(x2)
    mu = torch.cat((latent1["qz_m"], latent2["qz_m"]), dim=-1)
    std = torch.cat((latent1["qz_v"].sqrt(), latent2["qz_v"].sqrt()), dim=-1)
    if alpha_vi:
        alpha_mu = mu[:, dim].mean(0)
        alpha_std = std[:, dim].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu.mean(0)
        beta_std = std.mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], 1), device=device).mul(alpha_std).add_(alpha_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], 1)
    zs = torch.randn((params['N_alpha'] * params['N_beta'], params['z_dim']), device=device).mul(beta_std).add_(beta_mu)
    zs[:, dim] = alpha[:, 0]
    logit, prob = model.dpd_model(zs).values()
    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return I


def joint_uncond_v2(params, model, data, index, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond:
        Sample-based estimate of "joint, unconditional" causal effect, -I(alpha; Yhat).
    Inputs:
        - params['N_alpha'] monte-carlo samples per causal factor
        - params['N_beta']  monte-carlo samples per noncausal factor
        - params['K']      number of causal factors
        - params['L']      number of non-causal factors
        - params['M']      number of classes (dimensionality of classifier output)
        - model
        - data
        - device
    Outputs:
        - negCausalEffect (sample-based estimate of -I(alpha; Yhat))
        - info['xhat']
        - info['yhat']
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x, _ = model.feature_mapper(feat)
    latent = model.encoder(x)
    mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], params['K']), device=device).mul(alpha_std).add_(alpha_mu).repeat(1, params[
        'N_beta']).view(params['N_alpha'] * params['N_beta'], params['K'])
    beta = torch.randn((params['N_alpha'] * params['N_beta'], params['L']), device=device).mul(beta_std).add_(beta_mu)
    zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    logit, prob = model.dpd_model(zs).values()

    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return -I, None


def beta_info_flow_v2(params, model, data, index, alpha_vi=True, beta_vi=False, eps=1e-8, device=None):
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x, _ = model.feature_mapper(feat)
    latent = model.encoder(x)
    mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, :params['K']].mean(0)
        alpha_std = std[:, :params['K']].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu[:, params['K']:].mean(0)
        beta_std = std[:, params['K']:].mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'] * params['N_beta'], params['K']), device=device).mul(alpha_std).add_(
        alpha_mu)
    beta = torch.randn((params['N_alpha'], params['L']), device=device).mul(beta_std).add_(beta_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], params['L'])

    zs = torch.cat([alpha, beta], dim=-1)
    # x_top_rec = model.decoder(zs)
    # x_down, _ = model.feature_selector(x_top_rec, keep_top=False, keep_not_top=True)
    logit, prob = model.dpd_model(zs).values()

    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return -I, None


def joint_uncond_single_dim_v2(params, model, data, index, dim, alpha_vi=False, beta_vi=True, eps=1e-8, device=None):
    """
    joint_uncond_single_dim:
        Sample-based estimate of "joint, unconditional" causal effect
        for single latent factor, -I(z_i; Yhat). Note the interpretation
        of params['Nalpha'] and params['Nbeta'] here: Nalpha is the number
        of samples of z_i, and Nbeta is the number of samples of the other
        latent factors.
    Inputs:
        - params['Nalpha']
        - params['Nbeta']
        - params['K']
        - params['L']
        - params['M']
        - model
        - data
        - device
        - dim (i : compute -I(z_i; Yhat) **note: i is zero-indexed!**)
    Outputs:
        - negCausalEffect (sample-based estimate of -I(z_i; Yhat))
        - info['xhat']
        - info['yhat']
    """
    I = 0.0
    q = torch.zeros(params['M'], device=device)
    feat = data[index].repeat(params['N_alpha'] * params['N_beta'], 1)
    x, _ = model.feature_mapper(feat)
    latent = model.encoder(x)
    mu, std = latent["qz_m"], latent["qz_v"].sqrt()
    if alpha_vi:
        alpha_mu = mu[:, dim].mean(0)
        alpha_std = std[:, dim].mean(0)
    else:
        alpha_mu = 0
        alpha_std = 1

    if beta_vi:
        beta_mu = mu.mean(0)
        beta_std = std.mean(0)
    else:
        beta_mu = 0
        beta_std = 1

    alpha = torch.randn((params['N_alpha'], 1), device=device).mul(alpha_std).add_(alpha_mu).repeat(
        1, params['N_beta']).view(params['N_alpha'] * params['N_beta'], 1)
    zs = torch.randn((params['N_alpha'] * params['N_beta'], params['z_dim']), device=device).mul(beta_std).add_(beta_mu)
    zs[:, dim] = alpha[:, 0]
    logit, prob = model.dpd_model(zs).values()
    yhat = torch.cat((prob, 1 - prob), dim=1).view(params['N_alpha'], params['N_beta'], params['M'])
    p = yhat.mean(1)
    I = torch.sum(torch.mul(p, torch.log(p + eps)), dim=1).mean()
    # I = torch.sum(torch.mul(p, F.log_softmax(p, dim=1)), dim=1).mean()
    q = p.mean(0)
    I = I - torch.sum(torch.mul(q, torch.log(q + eps)))
    return I