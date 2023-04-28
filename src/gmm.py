import torch
from torch import nn


# GMM Parameters
class GMMParameters(nn.Module):

    def __init__(self, C, K, sigma_eps=0.01, rho_eps=0.01):
        super(GMMParameters, self).__init__()

        self.C = C
        self.K = K
        self.sigma_eps = sigma_eps
        self.rho_eps = rho_eps

        self.color_params = nn.Parameter(torch.randn(1, self.C, 1, 1, self.K))
        self.mu_x_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.mu_y_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.sigma_x_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.sigma_y_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.rho_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.pi_params = nn.Parameter(torch.randn(1, 1, 1, 1, self.K))
        self.scale_params = nn.Parameter(torch.randn(1, 1, 1, 1, 1))

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=4)

    def reset(self):
        device = self.color_params.device
        self.color_params.data = torch.randn(1, self.C, 1, 1, self.K).to(device)
        self.mu_x_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.mu_y_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.sigma_x_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.sigma_y_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.rho_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.pi_params.data = torch.randn(1, 1, 1, 1, self.K).to(device)
        self.scale_params.data = torch.randn(1, 1, 1, 1, 1).to(device)

    def forward(self):
        color = self.tanh(self.color_params)  # {N, C, 1, 1, K}
        mu_x = self.tanh(self.mu_x_params)  # {N, 1, 1, 1, K}
        mu_y = self.tanh(self.mu_y_params)  # {N, 1, 1, 1, K}
        sigma_x = self.softplus(self.sigma_x_params) + self.sigma_eps  # {N, 1, 1, 1, K}
        sigma_y = self.softplus(self.sigma_y_params) + self.sigma_eps  # {N, 1, 1, 1, K}
        rho = self.tanh(self.rho_params) * (1.0 - self.rho_eps)  # {N, 1, 1, 1, K}
        pi = self.softmax(self.pi_params)  # {N, 1, 1, 1, K}
        scale = self.sigmoid(self.scale_params)  # {N, 1, 1, 1, 1}
        return color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale


# Random GMM Parameters
class RandomGMMParameters(nn.Module):

    def __init__(self, C, K, sigma_eps=0.01, rho_eps=0.01):
        super(RandomGMMParameters, self).__init__()

        self.C = C
        self.K = K
        self.sigma_eps = sigma_eps
        self.rho_eps = rho_eps

        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=4)

    def forward(self, N, device):

        color_params = (torch.rand(N, self.C, 1, 1, self.K).to(device) * 2.0 - 1.0)  # -1 <= x <= 1
        mu_x_params = (torch.rand(N, 1, 1, 1, self.K).to(device) * 2.0 - 1.0)  # -1 <= x <= 1
        mu_y_params = (torch.rand(N, 1, 1, 1, self.K).to(device) * 2.0 - 1.0)  # -1 <= x <= 1
        sigma_x_params = torch.randn(N, 1, 1, 1, self.K).to(device)  # x
        sigma_y_params = torch.randn(N, 1, 1, 1, self.K).to(device)  # x
        rho_params = (torch.rand(N, 1, 1, 1, self.K).to(device) * 2.0 - 1.0)  # -1 <= x <= 1
        pi_params = torch.randn(N, 1, 1, 1, self.K).to(device)  # x
        scale_params = torch.rand(N, 1, 1, 1, 1).to(device)  # 0 <= x <= 1

        color = color_params  # {N, C, 1, 1, K}, -1 <= x <= 1
        mu_x = mu_x_params  # {N, 1, 1, 1, K}, -1 <= x <= 1
        mu_y = mu_y_params  # {N, 1, 1, 1, K}, -1 <= x <= 1
        sigma_x = self.softplus(sigma_x_params) + self.sigma_eps  # {N, 1, 1, 1, K}, x > eps
        sigma_y = self.softplus(sigma_y_params) + self.sigma_eps  # {N, 1, 1, 1, K}, x > eps
        rho = rho_params * (1.0 - self.rho_eps)  # {N, 1, 1, 1, K}, -1+eps <= x <= 1-eps
        pi = self.softmax(pi_params)  # {N, 1, 1, 1, K}, x > 0
        scale = scale_params # {N, 1, 1, 1, 1}, 0 <= x <= 1

        return color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale


# GMM Generator
class GMMGenerator(nn.Module):

    def __init__(self):
        super(GMMGenerator, self).__init__()

    def get_prob(self, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale, N, H, W, device):
    
        x = (torch.arange(W) / (W - 1) * 2.0 - 1.0).view(1, 1, 1, W, 1).to(device)  # {1, 1, 1, W, 1}
        y = (torch.arange(H) / (H - 1) * 2.0 - 1.0).view(1, 1, H, 1, 1).to(device)  # {1, 1, H, 1, 1}

        z_x = (x - mu_x) / sigma_x  # {N, 1, 1, W, K}
        z_y = (y - mu_y) / sigma_y  # {N, 1, H, 1, K}

        f_xy = torch.exp(-(z_x * z_x - 2.0 * rho * z_x * z_y + z_y * z_y) / (2.0 * (1.0 - rho * rho)))  # {N, 1, H, W, K}
        gauss_prob = f_xy * pi  # {N, 1, H, W, K}
        gmm_prob = gauss_prob.sum(dim=-1)  # {N, 1, H, W}
        gmm_prob_max = gmm_prob.view(N, -1).max(dim=-1)[0].view(N, 1, 1, 1, 1)  # {N, 1, 1, 1, 1}
        norm_gauss_prob = gauss_prob / gmm_prob_max * scale  # {N, 1, H, W, K}

        return gauss_prob, norm_gauss_prob, gmm_prob_max

    def component(self, gmm_params, N, W, H, device):

        color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
        gauss_prob, norm_gauss_prob, _ = self.get_prob(mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale, N, H, W, device)

        gamma = gauss_prob / gauss_prob.sum(dim=-1, keepdim=True)  # {N, 1, H, W, K}
        imageColor = (color * gamma).sum(dim=-1).expand(N, 3, H, W)  # {N, 3, H, W}
        imageAlpha = norm_gauss_prob.sum(dim=-1) * 2.0 - 1.0  # {N, 1, H, W}
        imageGMM = torch.cat((imageColor, imageAlpha), dim=1).contiguous()  # {N, 4, H, W}

        return imageGMM

    def gauss_mask(self, gmm_params, N, W, H, device):

        color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
        _, norm_gauss_prob, gmm_prob_max = self.get_prob(mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale, N, H, W, device)

        mask = 1.0 - norm_gauss_prob  # {N, 1, H, W, K}
        noise = norm_gauss_prob * color  # {N, C, H, W, K}

        return mask, noise, gmm_prob_max

    def forward(self, gmm_params, N, W, H, device):

        color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
        _, norm_gauss_prob, _ = self.get_prob(mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale, N, H, W, device)

        mask = 1.0 - norm_gauss_prob.sum(dim=-1)  # {N, 1, H, W}
        noise = (norm_gauss_prob * color).sum(dim=-1)  # {N, C, H, W}

        return mask, noise


# Anomaly Generator
class AnomalyGenerator(nn.Module):

    def __init__(self, args):
        super(AnomalyGenerator, self).__init__()
        self.gmm_params = GMMParameters(C=args.nc, K=args.K)
        self.gmm_gen = GMMGenerator()

    def reset(self):
        self.gmm_params.reset()
    
    def component(self, N, W, H, device):

        gmm_params = self.gmm_params()
        imageGMM = self.gmm_gen.component(gmm_params, N, W, H, device)

        return imageGMM
    
    def forward(self, imageN):

        N = imageN.shape[0]
        W = imageN.shape[3]
        H = imageN.shape[2]
        device = imageN.device

        gmm_params = self.gmm_params()
        mask, noise = self.gmm_gen(gmm_params, N, W, H, device)
        imageA = imageN * mask + noise

        return imageA, 1.0 - mask


# Constant Anomaly Generator
class ConstantAnomalyGenerator(nn.Module):

    def __init__(self):
        super(ConstantAnomalyGenerator, self).__init__()
        self.gmm_gen = GMMGenerator()
    
    def component(self, gmm_params, N, W, H, device):

        imageGMM = self.gmm_gen.component(gmm_params, N, W, H, device)
        
        return imageGMM

    def for_target(self, imageN, gmm_params, thresh, eps=1e-9):

        N = imageN.shape[0]
        C = imageN.shape[1]
        W = imageN.shape[3]
        H = imageN.shape[2]
        device = imageN.device

        mask, noise, gmm_prob_max_before = self.gmm_gen.gauss_mask(gmm_params, N, W, H, device)  # {N, 1, H, W, K}, {N, C, H, W, K}, {N, 1, 1, 1, 1}
        imageA = imageN.unsqueeze(dim=-1) * mask + noise  # {N, C, H, W, K}
        imageS = torch.abs(imageN.unsqueeze(dim=-1) - imageA).view(N, C * H * W, -1)  # {N, C*H*W, K}
        sub = imageS.max(dim=1)[0].view(N, 1, 1, 1, -1)  # {N, 1, 1, 1, K}
        effect_flag = (sub > thresh).to(torch.float)  # {N, 1, 1, 1, K}

        color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale = gmm_params
        pi = pi * effect_flag + torch.zeros_like(pi) * (1.0 - effect_flag) + eps  # {N, 1, 1, 1, K}
        _, _, gmm_prob_max_after = self.gmm_gen.get_prob(mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale, N, H, W, device)  # {N, 1, 1, 1, 1}
        scale = scale * gmm_prob_max_after / gmm_prob_max_before  # {N, 1, 1, 1, 1}
        gmm_params = (color, mu_x, mu_y, sigma_x, sigma_y, rho, pi, scale)

        return gmm_params

    def forward(self, imageN, gmm_params):

        N = imageN.shape[0]
        W = imageN.shape[3]
        H = imageN.shape[2]
        device = imageN.device

        mask, noise = self.gmm_gen(gmm_params, N, W, H, device)
        imageA = imageN * mask + noise

        return imageA, 1.0 - mask


# Random Anomaly Generator
class RandomAnomalyGenerator(nn.Module):

    def __init__(self, args):
        super(RandomAnomalyGenerator, self).__init__()
        self.gmm_params = RandomGMMParameters(C=args.nc, K=args.K)
        self.gmm_gen = GMMGenerator()

    def component(self, N, W, H, device):

        gmm_params = self.gmm_params(N, device)
        imageGMM = self.gmm_gen.component(gmm_params, N, W, H, device)
        
        return imageGMM

    def forward(self, imageN):

        N = imageN.shape[0]
        W = imageN.shape[3]
        H = imageN.shape[2]
        device = imageN.device

        gmm_params = self.gmm_params(N, device)
        mask, noise = self.gmm_gen(gmm_params, N, W, H, device)
        imageA = imageN * mask + noise

        return imageA


