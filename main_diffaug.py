import lightning as pl
import numpy as np
import plotly.graph_objects as go
import scipy
import torch
import torch.nn.functional as F
from kornia.augmentation import (
    ColorJitter,
    Normalize,
    RandomGrayscale,
    RandomHorizontalFlip,
    RandomResizedCrop,
    RandomSolarize,
)
from lightning import LightningModule
from lightning.pytorch.cli import LightningCLI
from plotly.subplots import make_subplots
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import models

import eval.eval_core_base as ecb
import manifolds
from dataloader import data_DVIT as data_base
from diff_model.diffusion import GaussianDiffusion, make_beta_schedule
from diff_model.model import AE_CNN_bottleneck_deep, AE
from util import Layout, LineBackground, OuterRing, ScatterCenter, ScatterVis, pad_table_data

import wandb

torch.set_num_threads(1)


class PositionalEncoding(nn.Module):
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        # self.dropout = nn.Dropout(dropout)
        self.P = torch.zeros((1, max_len, num_hiddens))
        r_a = torch.arange(0, num_hiddens, 2, dtype=torch.float32)
        n_h = torch.pow(10000, r_a / num_hiddens)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / n_h
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return X


class ToPoincareModel(nn.Module):
    """
    Module which maps points in n-dim Euclidean space to n-dim Poincare space
    """

    def __init__(self, c, manifold="PoincareBall"):
        super(ToPoincareModel, self).__init__()
        self.c = c
        self.manifold = manifold  # 'PoincareBall'
        self.manifold = getattr(manifolds, self.manifold)()

    def forward(self, x):
        emb = self.manifold.expmap0(self.manifold.proj_tan0(x, self.c), c=self.c)
        z = self.manifold.proj(emb, c=self.c) / 1.414
        return z


def gpu2np(a):
    return a.cpu().detach().numpy()


class NN_FCBNRL_MM(nn.Module):
    def __init__(self, in_dim, out_dim, channel=8, use_RL=True):
        super(NN_FCBNRL_MM, self).__init__()
        m_l = []
        m_l.append(
            nn.Linear(
                in_dim,
                out_dim,
            )
        )
        if use_RL:
            m_l.append(nn.LeakyReLU(0.1))
        m_l.append(nn.BatchNorm1d(out_dim))

        self.block = nn.Sequential(*m_l)

    def forward(self, x):
        return self.block(x)


class DMTEVT_Encoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Encoder, self).__init__()
        self.data_name = data_name
        self.l_token = l_token
        self.l_token2 = l_token2
        (self.model_token,) = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    # def tokenlization(self, data_after_tokened, token_fea, batch_size):
    #     token_index = token_fea[0].reshape(-1).to(data_after_tokened.device)
    #     data_tokened = data_after_tokened[:, token_index].reshape(
    #         batch_size, token_fea.shape[1], token_fea.shape[2]
    #     )
    #     return data_tokened

    def forward(self, x):
        # def forward_fea(self, x, batch_idx):
        lat_high_dim = self.model_token(x)

        return lat_high_dim  # , lat_vis

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        # self.hparams.l_token_2 = 20
        m_p = []
        m_p.append(NN_FCBNRL_MM(num_input_dim, 500))
        m_p.append(NN_FCBNRL_MM(500, 500))
        m_p.append(NN_FCBNRL_MM(500, 500))
        m_p.append(NN_FCBNRL_MM(500, l_token_2))
        model_token = nn.Sequential(*m_p)

        # if "Cifar" not in data_name:
        #     # Transformer_layer = nn.TransformerEncoderLayer(
        #     #     d_model=l_token_2,
        #     #     nhead=10,
        #     #     batch_first=True,
        #     # )
        #     # model_transformer1 = nn.TransformerEncoder(
        #     #     Transformer_layer,
        #     #     num_layers=num_layers_Transformer,
        #     # )
        #     model_transformer2 = NN_FCBNRL_MM(transformer2_indim, l_token_2)
        # else:
        #     r18 = models.__dict__["resnet18"]()
        #     model_transformer2 = nn.Sequential(
        #         *list(r18.children())[:-1],
        #         nn.Flatten(),
        #         NN_FCBNRL_MM(512, 512),
        #         NN_FCBNRL_MM(512, 512),
        #     )
        #     model_transformer2[0] = nn.Conv2d(
        #         3, 64, kernel_size=3, stride=1, padding=1, bias=False
        #     )
        #     model_transformer2[3] = nn.Identity()

        #     model_transformer1 = NN_FCBNRL_MM(20, 10)

        return (
            model_token,
            # model_transformer1,
            # model_transformer2,
            # model_down,
            # model_decoder,
        )  # , model_d, model_d_encode


class DMTEVT_Vis(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Vis, self).__init__()
        self.model_down = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, lat_high_dim):
        lat_vis = self.model_down(lat_high_dim)
        return lat_vis

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        if "Cifar" not in data_name:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            model_down = nn.Sequential(*m_b)
        else:
            m_b = []
            m_b.append(NN_FCBNRL_MM(l_token_2, laten_down))
            m_b.append(NN_FCBNRL_MM(laten_down, 2, use_RL=False))
            model_down = nn.Sequential(*m_b)

        return model_down


class DMTEVT_Decoder(nn.Module):
    def __init__(
        self,
        l_token,
        l_token2,
        data_name,
        transformer2_indim,
        laten_down,
        num_layers_Transformer,
        num_input_dim,
    ):
        super(DMTEVT_Decoder, self).__init__()
        self.model_decoder = self.InitNetworkMLP(
            l_token=l_token,
            l_token_2=l_token2,
            data_name=data_name,
            transformer2_indim=transformer2_indim,
            laten_down=laten_down,
            num_layers_Transformer=num_layers_Transformer,
            num_input_dim=num_input_dim,
        )

    def forward(self, x):
        return self.model_decoder(x)

    def InitNetworkMLP(
        self,
        l_token=50,
        l_token_2=100,
        data_name="mnist",
        transformer2_indim=3750,
        laten_down=500,
        num_layers_Transformer=1,
        num_input_dim=64,
    ):
        m_decoder = []
        m_decoder.append(NN_FCBNRL_MM(l_token_2, 100))
        m_decoder.append(NN_FCBNRL_MM(100, 100))
        m_decoder.append(NN_FCBNRL_MM(100, num_input_dim, use_RL=False))
        m_decoder.append(nn.Sigmoid())
        model_decoder = nn.Sequential(*m_decoder)

        return model_decoder


def accumulate(model1, model2, decay=0.9999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def progressive_samples_fn(
    model, diffusion, shape, device, cond, include_x0_pred_freq=50
):
    samples, progressive_samples = diffusion.p_sample_loop_progressive(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        cond=cond,
    )
    return {"samples": samples, "progressive_samples": progressive_samples}


def progressive_samples_fn_simple(
    model, diffusion, shape, device, cond, include_x0_pred_freq=50
):
    samples, history = diffusion.p_sample_loop_progressive_simple(
        model=model,
        shape=shape,
        noise_fn=torch.randn,
        device=device,
        include_x0_pred_freq=include_x0_pred_freq,
        cond=cond,
    )
    return {"samples": samples}


class DMTEVT_model(LightningModule):
    def __init__(
        self,
        lr=0.001,
        nu=0.01,
        data_name="mnist",
        max_epochs=1000,
        class_num=30,
        steps=20000,
        num_pat=8,
        num_fea_aim=500,
        n_timestep=100,
        l_token=50,
        l_token_2=50,
        num_layers_Transformer=1,
        num_latent_dim=2,
        num_input_dim=64,
        laten_down=500,
        weight_decay=0.0001,
        kmean_scale=0.01,
        loss_rec_weight=0.01,
        preprocess_epoch=100,
        transformer2_indim=3750,
        marker_size=2,
        joint_epoch=1400,
        rand_rate=0.9,
        **kwargs,
    ):
        super().__init__()

        # Set our init args as class attributes
        self.setup_bool_zzl = False
        # self.learning_rate = learning_rate
        self.save_hyperparameters()

        self.dictinputdict = {}
        self.t = 0.1
        self.alpha = None
        self.stop = False
        self.bestval = 0
        self.aim_cluster = None
        self.importance = None
        self.wandb_logs = {}
        self.mse = torch.nn.MSELoss()  # torch.nn.CrossEntropyLoss()
        self.center_change = False
        self.train_state = "train"
        self.base_acc = 0

        self.validation_step_outputs_data = []
        self.validation_step_outputs_lat = []
        self.validation_step_outputs_lat_high = []
        self.validation_step_outputs_label = []

        self.enc = DMTEVT_Encoder(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )

        self.vis = DMTEVT_Vis(
            l_token=self.hparams.l_token,
            l_token2=self.hparams.l_token_2,
            data_name=self.hparams.data_name,
            transformer2_indim=self.hparams.transformer2_indim,
            laten_down=self.hparams.laten_down,
            num_layers_Transformer=self.hparams.num_layers_Transformer,
            num_input_dim=self.hparams.num_input_dim,
        )
        self.data_aug = torch.zeros(48000, self.hparams.num_input_dim)

        self.UNet_model = AE(in_dim=self.hparams.num_input_dim, mid_dim=2000)
        self.UNet_ema = AE(in_dim=self.hparams.num_input_dim, mid_dim=2000)

        n_timestep = self.hparams.n_timestep
        self.betas = make_beta_schedule(
            schedule="linear", start=1e-4, end=2e-2, n_timestep=n_timestep
        )
        self.diffusion = GaussianDiffusion(
            betas=self.betas,
            model_mean_type="eps",
            model_var_type="fixedlarge",
            loss_type="mse",
        )

        if "Cifar" in data_name:
            mean = [0.4914, 0.4822, 0.4465]
            std = [0.4914, 0.4822, 0.4465]
            self.transform_base = torch.nn.Sequential(
                Normalize(mean=mean, std=std),
            )
            self.transform = torch.nn.Sequential(
                RandomResizedCrop((32, 32)),  # 随机裁剪和调整大小
                RandomHorizontalFlip(),  # 随机水平翻转
                ColorJitter(
                    brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2, p=0.8
                ),
                RandomGrayscale(p=0.2),
                RandomSolarize(0.2),
                Normalize(mean=mean, std=std),
            )

        if self.hparams.num_fea_aim < 1:
            self.hparams.num_fea_aim = int(
                self.hparams.num_input_dim * self.hparams.num_fea_aim
            )
        else:
            self.hparams.num_fea_aim = int(self.hparams.num_fea_aim)
        self.hparams.num_fea_aim = min(
            self.hparams.num_fea_aim, self.hparams.num_input_dim
        )

        self.Kcluster_layer = nn.Linear(
            self.hparams.l_token_2, self.hparams.class_num, bias=False
        )
        self.kmean_normalize_cluster_center()
        self.ToPoincare = ToPoincareModel(
            c=0.5,
        )
        self.start_from_ckpt = True

    def _DistanceSquared(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)

        return dist

    def _CalGamma(self, v):
        a = scipy.special.gamma((v + 1) / 2)
        b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
        out = a / b

        return out

    def _TwowaydivergenceLoss(self, P_, Q_, select=None):
        EPS = 1e-5
        losssum1 = P_ * torch.log(Q_ + EPS)
        losssum2 = (1 - P_) * torch.log(1 - Q_ + EPS)
        losssum = -1 * (losssum1 + losssum2)
        return losssum.mean()

    def _Similarity(self, dist, gamma, v=100, h=1, pow=2):
        dist_rho = dist

        dist_rho[dist_rho < 0] = 0
        Pij = (
            gamma
            * torch.tensor(2 * 3.14)
            * gamma
            * torch.pow((1 + dist_rho / v), exponent=-1 * (v + 1))
        )
        return Pij

    def LossManifold(
        self,
        v_input,
        input_data,
        latent_data,
        v_latent,
        metric="euclidean",
    ):
        batch_size = input_data.shape[0] // 2
        data_1 = input_data[:batch_size]
        dis_P = self._DistanceSquared(data_1, data_1)
        latent_data_1 = latent_data[:batch_size]
        dis_P_2 = dis_P  # + nndistance.reshape(1, -1)
        gamma = self._CalGamma(v_input)
        P_2 = self._Similarity(dist=dis_P_2, gamma=gamma, v=v_input)
        latent_data_2 = latent_data[batch_size:]
        dis_Q_2 = self._DistanceSquared(latent_data_1, latent_data_2)
        Q_2 = self._Similarity(
            dist=dis_Q_2,
            gamma=self._CalGamma(v_latent),
            v=v_latent,
        )
        loss_ce_2 = self._TwowaydivergenceLoss(P_=P_2, Q_=Q_2)

        return loss_ce_2

    def augment_data_simple(self, cond_input_val):
        shape = (cond_input_val.shape[0], 1, self.hparams.num_input_dim)
        self.UNet_ema.eval()
        sample = progressive_samples_fn_simple(
            self.UNet_ema,
            self.diffusion,
            shape,
            device="cuda",
            cond=cond_input_val,
            include_x0_pred_freq=50,
        )
        return sample["samples"]

    def kmean_normalize_cluster_center(self):
        self.Kcluster_layer.weight.data = (
            F.normalize(self.Kcluster_layer.weight.data, dim=1) * 2.0
        )

    def kcompute_cluster_center(self):
        center_high_dim = self.Kcluster_layer.weight
        return self.vis(center_high_dim).detach().cpu().numpy()

    def kmeans_hand_center(self, latenvec):
        index = torch.randint(
            low=0,
            high=len(latenvec) // 2,
            size=(self.hparams.class_num,),
        )
        center = latenvec[index]
        return center

    def kmeans_loss(self, mid):
        mid = mid.detach()
        dis_cluster = self._DistanceSquared(mid, self.Kcluster_layer.weight)
        dis_cluster = dis_cluster / 100
        sim = self._Similarity(dist=dis_cluster, gamma=self._CalGamma(1), v=1)
        soft_label = F.softmax(sim.detach(), dim=1)
        hard_label = torch.argmax(soft_label, dim=1)
        delta = torch.zeros(
            (mid.shape[0], self.hparams.class_num), requires_grad=False
        ).cuda()
        label_range = torch.arange(0, delta.size(0)).long()
        delta[label_range, hard_label] = 1
        loss_clu_batch = 1 - torch.mul(delta, sim)
        loss_clu_batch = self.hparams.kmean_scale * loss_clu_batch.mean()
        return loss_clu_batch

    def forward_fea(self, x, batch_idx=None):
        lat_high_dim = self.enc(x)
        lat_vis = self.vis(lat_high_dim)

        return lat_high_dim, lat_vis

    def forward(self, x, batch_idx):
        return self.forward_fea(x, batch_idx)

    def encoder(self, x):
        return (self.forward_fea(torch.tensor(x), 0)[0]).detach().cpu().numpy()

    def decoder(self, x):
        return self.model_decoder(torch.tensor(x)).detach().cpu().numpy()

    def predict_get_emb(self, x):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()

            x = self.tokenlization(x, self.n_token_feature, x.shape[0])

            return gpu2np(self.forward_fea(x)[1])

    def predict(self, x, tau=1):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()

            x = self.tokenlization(x, self.n_token_feature, x.shape[0])

            return gpu2np(self.forward_simi(x, tau=tau))

    def predict_label(self, x, tau=1):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()
            x = self.tokenlization(x, self.n_token_feature, x.shape[0])
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)

            lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)
            center = self.Kcluster_layer.weight.data.detach()
            center_vis = self.vis(center)
            distance = torch.cdist(lat_vis, center_vis, p=2)
            label = distance.argmin(dim=1)

            r_class = []

            for i in range(distance.shape[1]):
                if (label == i).sum() < 10:
                    print("class {} is empty".format(i))
                else:
                    r_class.append(i)
            # import pdb; pdb.set_trace()
            center_vis = center_vis[r_class]
            distance = torch.cdist(lat_vis, center_vis, p=2)
            label = distance.argmin(dim=1)

            return label

    def predict_high_latent(self, x, tau=1):
        # import pdb; pdb.set_trace()
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()

            x = self.tokenlization(x, self.n_token_feature, x.shape[0])

            # check x is tensor
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x)
            lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)

        return lat_high_dim

    def _Distance_squared_CPU(self, x, y):
        x = torch.tensor(x)
        y = torch.tensor(y)
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-36)
        return d.detach().cpu().numpy()

    def pw_cosine_distance(self, input_a, input_b):
        normalized_input_a = torch.nn.functional.normalize(input_a)
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        res *= -1  # 1-res without copy
        res += 1
        return res

    def forward_simi(self, x, nu=1, tau=1):
        # check x is tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x)
        lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)
        out = self.classifier(lat_vis)
        # data_pre = out.argmax(dim=1)
        # for i in range(30):
        # print([(data_pre==i).sum() for i in range(30)])
        # import pdb; pdb.set_trace()
        # self.aim_cluster = 0
        # center = self.Kcluster_layer.weight.data.detach()
        # center_vis = self.vis(center)
        # # distance = torch.cdist(lat_high_dim, center, p=1)
        # distance = torch.cdist(lat_vis, center_vis, p=2)
        return out

    def forward_vis(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()
            x = self.tokenlization(x, self.n_token_feature, x.shape[0])
            lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)
        return lat_vis

    def forward_center(
        self,
    ):
        center = self.Kcluster_layer.weight.data.detach()
        return self.vis(center)

    def AE_predict(self, x):
        with torch.no_grad():
            if len(x.shape) == 1:
                x = x.reshape(1, -1)
            x = torch.tensor(x).to(self.device).float()

            x = self.tokenlization(x, self.n_token_feature, x.shape[0])

            lat_high_dim, lat_vis = self.forward_fea(x, batch_idx=0)
            decode_data = self.dec(lat_high_dim)
            return gpu2np(decode_data)

    def tokenlization(self, data_after_tokened, n_token_feature, batch_size):
        token_index = n_token_feature[0].reshape(-1)
        token_index = token_index.to(data_after_tokened.device)
        # if self.dbug:
        #     import pdb; pdb.set_trace()
        data_tokened = data_after_tokened[:, token_index].reshape(
            batch_size, n_token_feature.shape[1], n_token_feature.shape[2]
        )
        return data_tokened

    def on_save_checkpoint(self, checkpoint) -> None:
        "Objects to include in checkpoint file"
        checkpoint["n_token_feature"] = self.n_token_feature

    def on_load_checkpoint(self, checkpoint) -> None:
        "Objects to retrieve from checkpoint file"
        self.n_token_feature = checkpoint["n_token_feature"]
        self.enc.n_token_feature = checkpoint["n_token_feature"]

    def f_augment_data(self, data_at, data_aat, token_fea, batch_size):
        if "Cifar" in self.hparams.data_name:
            data = self.transform_base(data_at.permute(0, 3, 1, 2) / 255)
            data_aug = self.transform(data_aat.permute(0, 3, 1, 2) / 255)
        else:
            # data = self.tokenlization(data_at, token_fea, batch_size)
            # data_aug = self.tokenlization(data_aat, token_fea, batch_size)
            data = data_at
            data_aug = data_aat

        data = torch.cat([data, data_aug])
        return data

    def diffusion_loss(self, data_after_tokened, lat_high_dim):
        data_diff = data_after_tokened
        views = data_diff.reshape(data_diff.shape[0], -1)
        time = (
            (torch.rand(data_diff.shape[0]) * self.hparams.n_timestep)
            .type(torch.int64)
            .to(data_diff.device)
        )
        loss_diff = self.diffusion.training_losses(
            model=self.UNet_model,
            x_0=views,
            t=time,
            lab=lat_high_dim.detach(),
        ).mean()
        return loss_diff * 0.01

    def EM_switch(self):
        if self.current_epoch >= self.hparams.joint_epoch:
            # if (self.current_epoch // 100) % 2 == 0:
            em_state = "e"
            # if (self.current_epoch // 100) % 2 == 1:
            #     em_state = "m"
        elif self.current_epoch > self.hparams.preprocess_epoch:
            em_state = "s2"
        else:
            em_state = "s1"
        return em_state

    def training_step(self, batch, batch_idx):
        (
            (data_at, data_aat, data_rec),
            label,
            index,
            n_token_feature,
        ) = batch
        batch_size = index.shape[0]

        self.em_state = self.EM_switch()
        self.n_token_feature = n_token_feature
        self.enc.n_token_feature = n_token_feature

        if self.current_epoch > self.hparams.joint_epoch+1:            
            new_aug = self.data_aug[index.cpu()].to(self.device)
            rand_bool = torch.randn(new_aug.shape[0]).to(self.device) > self.hparams.rand_rate
            data_aat[rand_bool] = new_aug[rand_bool]

        data = self.f_augment_data(data_at, data_aat, n_token_feature, batch_size)
        # if self.current_epoch > self.hparams.joint_epoch + 1:

        # if self.current_epoch < self.hparams.preprocess_epoch:
        if self.em_state == "s1" or self.em_state == "e":
            
            self.enc.train()
            self.vis.train()
            lat_high_dim, lat_vis = self(data, batch_idx)
            loss_topo = self.LossManifold(
                v_input=100,
                input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
                latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
                v_latent=self.hparams.nu,
            )
            loss_diff = torch.tensor(0)
        elif self.em_state == "s2" or self.em_state == "m":

            self.enc.eval()
            self.vis.eval()
            cond, _ = self(data, batch_idx)
            loss_topo = torch.tensor(0)
            loss_diff = self.diffusion_loss(data_at, cond[:batch_size])
        # else:
        #     loss_topo = self.LossManifold(
        #         v_input=100,
        #         input_data=lat_high_dim.reshape(lat_high_dim.shape[0], -1),
        #         latent_data=lat_vis.reshape(lat_vis.shape[0], -1),
        #         v_latent=self.hparams.nu,
        #     )
        #     loss_diff = self.diffusion_loss(data_at, lat_high_dim[:batch_size])
    
        self.wandb_logs = {
            "loss_topo": loss_topo.item(),
            "loss_diff": loss_diff.item(),
            "lr": float(self.trainer.optimizers[0].param_groups[0]["lr"]),
            "epoch": float(self.current_epoch),
            # ""
        }
        # self.log_dict(self.wandb_logs, sync_dist=True)

        accumulate(
            self.UNet_ema,
            self.UNet_model.module
            if isinstance(self.UNet_model, nn.DataParallel)
            else self.UNet_model,
            0.9999,
        )
        return loss_topo + loss_diff

    def validation_step(self, batch, batch_idx, test=False):
        if test:
            data_at, label, index, n_token_feature = batch
        else:
            (
                (data_at, data_aat, data_rec),
                label,
                index,
                n_token_feature,
            ) = batch
        batch_size = index.shape[0]
        self.em_state = self.EM_switch()
        self.n_token_feature = n_token_feature

        if self.em_state == "e" or self.em_state == "m":
            if self.current_epoch % 1000000000 == 0:
                self.vis.eval()
                self.enc.eval()
                cond, _ = self(data_at, batch_idx)
                with torch.no_grad():
                    samples = self.augment_data_simple(cond[:batch_size])
                    self.data_aug[index] = samples.reshape(batch_size, -1).cpu()

        # if "Cifar" in self.hparams.data_name:
        #     data = self.transform_base(data_at.permute(0, 3, 1, 2) / 255)
        # else:
        #     data = self.tokenlization(data_at, n_token_feature, batch_size)
        data = data_at

        lat_high_dim, lat_vis = self(data, batch_idx)
        if batch_idx == 0:
            self.lat_high_dim = lat_high_dim
            self.data = data_at

        self.validation_step_outputs_data.append(gpu2np(data)[:batch_size])
        self.validation_step_outputs_lat.append(gpu2np(lat_vis)[:batch_size])
        self.validation_step_outputs_lat_high.append(gpu2np(lat_high_dim)[:batch_size])
        self.validation_step_outputs_label.append(gpu2np(label))
        return data.mean()

    def on_validation_epoch_end(self):
        data = np.concatenate(self.validation_step_outputs_data)
        emb_vis = np.concatenate(self.validation_step_outputs_lat)
        emb_high = np.concatenate(self.validation_step_outputs_lat_high)
        label = np.concatenate(self.validation_step_outputs_label)
        
        ecb_e_train = ecb.Eval(input=data, latent=emb_vis, label=label, k=10)
        ecb_lat_train = ecb.Eval(input=data, latent=emb_high, label=label, k=10)
        SVC_acc = ecb_e_train.E_Classifacation_SVC()
        KNN_acc = ecb_e_train.E_Clasting_Kmeans()
        SVC_high_acc = ecb_lat_train.E_Classifacation_SVC()
        KNN_high_acc = ecb_lat_train.E_Clasting_Kmeans()
        
        if self.current_epoch == self.hparams.preprocess_epoch-1:
            self.base_acc = SVC_high_acc
        
        if data.shape[0] <= 1000 and self.train_state == "train":
            pass
        else:
            self.wandb_logs.update(
                {
                    f"data_number_{self.train_state}": data.shape[0],
                    f"SVC_{self.train_state}": SVC_acc,
                    f"SVC_high_{self.train_state}": SVC_high_acc,
                    f"KNN_{self.train_state}": KNN_acc,
                    f"KNN_high_{self.train_state}": KNN_high_acc,
                    f"acc_delta_{self.train_state}": SVC_high_acc - self.base_acc,
                }
            )
        formatted_logs = {key: (f"{value:.4f}" if isinstance(value, float) else value) for key, value in self.wandb_logs.items()}
        sorted_keys = sorted(formatted_logs)
        for key in sorted_keys:
            print(f"{key:<15} : {formatted_logs[key]}")
        print('-------------------')
        
        self.log_dict(self.wandb_logs, sync_dist=True)
        self.wandb_logs.clear()
        
        if self.start_from_ckpt:
            self.start_from_ckpt = False
            self.log_dict({'SVC_train': 0})
        
        self.samples = self.augment_data_simple(self.lat_high_dim[:50])

        if "MNIST" in self.hparams.data_name:
            self.logger.log_table(
                "samples",
                [self.samples[i].detach().cpu().numpy().reshape(28, 28) for i in range(20)],
            )
            self.logger.log_table(
                "data",
                [self.data[i].detach().cpu().numpy().reshape(28, 28) for i in range(20)],
            )
        else:
            dim = self.samples.shape[-1]
            d1 = np.sqrt(dim).astype(int)
            d2 = (dim//d1)+1
            pad_dim = d1*d2-dim
            sample = self.samples.reshape(-1, dim)
            for i in range(5):
                wandb.log({f'sample_{self.current_epoch}': pad_table_data(sample[i], pad_dim, d1, d2)})
                if self.current_epoch == 0:
                    wandb.log({f'data_{self.current_epoch}': pad_table_data(self.data[i], pad_dim, d1, d2)})

        self.plot_scatter(
            emb_vis, label, emb_vis[:10], P=False
            ).write_image("fig1.png", scale=3)
        self.logger.log_image(f"fig_{self.train_state}", ["fig1.png"])



        # dict_imgs = {
        #     "fig_{}".format(self.train_state): self.plot_scatter(emb_vis, label, cluster_center, P=False),
        #     "fig_H_{}".format(self.train_state): self.plot_scatter(emb_vis, label, cluster_center, P=True),
        #     "fig_H_{}_clearn".format(self.train_state): self.plot_scatter(emb_vis, label, cluster_center, P=True, clean=True),
        #     }
        # if reconstruction.shape[1] == 784:
        #     rec_tensor = torch.tensor(reconstruction[:64]).reshape(-1, 1, 28, 28)
        #     grid_image_recons = vutils.make_grid(rec_tensor, normalize=False, nrow=8).detach().cpu().numpy()
        #     grid_image_recons_image = wandb.Image(grid_image_recons.transpose(1,2,0))
        #     dict_imgs.update({"reconstruction_{}".format(self.train_state): grid_image_recons_image,})

        # wandb.log(dict_imgs)
        
        self.validation_step_outputs_lat_high.clear()
        self.validation_step_outputs_data.clear()
        self.validation_step_outputs_lat.clear()
        self.validation_step_outputs_label.clear()
        # self.validation_step_reconstruct.clear()

    def test_step(self, batch, batch_idx):
        # Here we just reuse the validation_step for testing
        self.train_state = "test"
        self.validation_step(batch, batch_idx, test=True)

    def on_test_epoch_end(self):
        # Here we just reuse the validation_epoch_end for testing
        print("------------")
        print("----test----")
        print("------------")
        self.on_validation_epoch_end()

    def configure_optimizers(self):
        parameters = [
            {"params": self.enc.parameters(), "lr": self.hparams.lr},
            {"params": self.vis.parameters(), "lr": self.hparams.lr},
            {"params": self.UNet_model.parameters(), "lr": self.hparams.lr},
        ]

        optimizer = torch.optim.AdamW(
            parameters, weight_decay=self.hparams.weight_decay
        )
        self.scheduler = StepLR(
            optimizer, step_size=self.hparams.max_epochs // 10, gamma=0.8
        )
        return [optimizer], [self.scheduler]

    def plot_scatter(self, emb_vis, labels, cluster_c, P=True, clean=False):
        if cluster_c.shape[1] != 2:
            cluster_c = cluster_c[:, :2]
            emb_vis = emb_vis[:, :2]

        fig = go.Figure()
        if P:
            mean_values_ins = np.mean(emb_vis, axis=0)
            var_values_ins = np.std(emb_vis, axis=0) * 0.5
            emb_c_vis_center = np.concatenate([cluster_c, emb_vis])

            emb_c_dis = (emb_c_vis_center - mean_values_ins) / var_values_ins
            emb_c_vis_center = gpu2np(self.ToPoincare(torch.Tensor(emb_c_dis)))
            emb_vis = emb_c_vis_center[cluster_c.shape[0] :]
            cluster_c = emb_c_vis_center[: cluster_c.shape[0]]
            if not clean:
                fig = LineBackground(fig, self)
                fig = OuterRing(fig, cluster_c, labels)
            else:
                fig = LineBackground(fig, self)

        fig = ScatterVis(fig, emb_vis, labels, size=self.hparams.marker_size)
        fig = ScatterCenter(fig, cluster_c, labels)
        fig = Layout(fig)

        return fig

    def pdist2(self, x: torch.Tensor, y: torch.Tensor):
        # calculate the pairwise distance
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist = torch.addmm(dist, mat1=x, mat2=y.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-12)
        return dist


def plot_cf_figure(cf_origin, img_from_origin, model, pix, text, name=""):
    cf = np.copy(cf_origin)
    img_from = np.copy(img_from_origin)

    fig_all_img = make_subplots(3, cf.shape[0])
    for j in range(cf.shape[0]):
        cf_0 = cf[j]
        cf_from = img_from[j]

        cf_from[gpu2np(model.mask < 1)] = None
        cf_0[gpu2np(model.mask < 1)] = None

        fig_all_img.add_trace(go.Heatmap(z=cf_from.reshape(pix, pix)[::-1]), 1, j + 1)
        fig_all_img.add_trace(
            go.Heatmap(z=np.abs(cf_from - cf_0).reshape(pix, pix)[::-1]), 2, j + 1
        )
        fig_all_img.add_trace(go.Heatmap(z=cf_0.reshape(pix, pix)[::-1]), 3, j + 1)


class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_name: str = "Digits",
        data_path: str = "/zangzelin/data",
        batch_size: int = 32,
        num_workers: int = 1,
        K: int = 3,
        uselabel: bool = False,
        pca_dim: int = 50,
        n_cluster: int = 25,
        n_f_per_cluster: int = 3,
        l_token: int = 10,
        seed: int = 0,
        rrc_rate: float = 0.8,
        trans_range: int = 2,
        preprocess_bool: bool = True,
    ):
        super().__init__()
        self.data_name = data_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.uselabel = uselabel
        self.pca_dim = pca_dim
        self.n_cluster = n_cluster
        self.n_f_per_cluster = n_f_per_cluster
        self.l_token = l_token
        self.K = K
        self.seed = seed
        self.rrc_rate = rrc_rate
        self.trans_range = trans_range
        self.preprocess_bool = preprocess_bool

    def setup(self, stage: str):
        dataset_f = getattr(data_base, self.data_name + "Dataset")
        self.data_train = dataset_f(
            data_name=self.data_name,
            train=True,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            n_cluster=self.n_cluster,
            n_f_per_cluster=self.n_f_per_cluster,
            l_token=self.l_token,
            seed=self.seed,
            rrc_rate=self.rrc_rate,
            trans_range=self.trans_range,
            preprocess_bool=self.preprocess_bool,
        )
        self.data_val = self.data_train
        self.data_test = dataset_f(
            data_name=self.data_name,
            train=False,
            data_path=self.data_path,
            k=self.K,
            pca_dim=self.pca_dim,
            n_cluster=self.n_cluster,
            n_f_per_cluster=self.n_f_per_cluster,
            l_token=self.l_token,
            seed=self.seed,
            rrc_rate=self.rrc_rate,
            trans_range=0,
            preprocess_bool=self.preprocess_bool,
        )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            drop_last=True,
            shuffle=True,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            drop_last=False,
            batch_size=min(self.batch_size, self.data_train.data.shape[0]),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
        )


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("trainer.max_epochs", "model.max_epochs")
        parser.link_arguments(
            "model.l_token",
            "data.l_token",
        )
        parser.link_arguments(
            "data.data_name",
            "model.data_name",
        )


def main():
    cli = MyLightningCLI(
        DMTEVT_model,
        # CIFAR10DataModule,
        MyDataModule,
        save_config_callback=None,
    )
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.model.eval()
    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
