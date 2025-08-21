from __future__ import absolute_import
from torch import nn
import torch
import collections
import torch.nn.functional as F
import numpy as np


class F2TDReid(nn.Module):
    def __init__(self, channel=64):
        super(F2TDReid, self).__init__()
        self.channel = channel
        self.adaptiveFC1 = nn.Linear(2 * channel, channel)
        self.adaptiveFC2 = nn.Linear(channel, int(channel / 2))
        self.adaptiveFC3 = nn.Linear(int(channel / 2), 2)
        self.softmax = nn.Softmax(dim=1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.relu = nn.ReLU()
        self.bn_afterDL = nn.BatchNorm2d(channel)
        # 自动更新的参数a
        self.aSource = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.aTarget = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.F2TDReidepsilon = 1e-8
        self.KSVDMaxIter = 1
        self.KSVDTolerance = 1e-6

    def forward(self, x, s_t_targets, HDiuRatio, DMomentumUpdate):
        if (not self.training):
            return x
        bs = x.size(0)
        assert (bs % 2 == 0)
        split = torch.split(x.clone(), int(bs / 2), 0)
        x_s = split[0].contiguous()  # [B, C, H, W]
        x_t = split[1].contiguous()

        # 生成源域字典和目标域字典
        # BCHW 转为 BHWC
        image_tensor = x.clone().permute(0, 2, 3, 1)

        # 公共字典D
        D_dict = collections.defaultdict(list)
        B, H, W, C = image_tensor.shape
        # 对批次的每张图片都计算字典
        # 将图像展平为二维
        Y = image_tensor.reshape(B, H, -1).to(torch.float32)

        # KS最大是图片的行数，即高度
        # K的值待确定
        K = H - (H // HDiuRatio)

        #  由于二范数计算出的损失很大，传入BHCW求损失的均值
        HWmul = H * W
        # finishDict_D, finishDict_X, allDLfinal_loss = self.createDictory(Y, K, HWmul)

        # # 动量更新公共字典
        # with torch.no_grad():
        #     for Bitem in range(B):
        #         if s_t_targets[Bitem].item() not in D_dict.keys():
        #             D_dict[s_t_targets[Bitem].item()] = finishDict_D[Bitem]
        #         else:
        #             # 使用动量更新公共字典
        #             D_dict[s_t_targets[Bitem].item()] = DMomentumUpdate * D_dict[s_t_targets[Bitem].item()] + \
        #                                                 (1 - DMomentumUpdate) * finishDict_D[Bitem]
        # # 使用生成的字典对输入x进行更新
        # with torch.no_grad():
        #     D_dict_items = [D_dict[s_t_targets[i].item()].clone() for i in range(B)]
        #     Y_items = [image_tensor[i].clone().reshape(H, -1) for i in range(B)]
        #     D_normalized = torch.stack(
        #         [(D_dict_item - D_dict_item.min()) / (D_dict_item.max() - D_dict_item.min() + self.F2TDReidepsilon) for D_dict_item in D_dict_items])
        #     finalX = torch.linalg.lstsq(D_normalized, torch.stack(Y_items)).solution
        #
        #     # 重构图片
        #     x_after_DL = torch.clamp(torch.matmul(D_normalized, finalX).reshape(B, H, W, C).permute(0, 3, 1, 2),
        #                              0, 1)

        # x_after_DLsplit = torch.split(x_after_DL, int(bs / 2), 0)
        # x_after_DLx_s = x_after_DLsplit[0].contiguous()  # [B, C, H, W]
        # x_after_DLx_t = x_after_DLsplit[1].contiguous()
        # # 对新生成的样例进行BN操作
        # BN_x_after_DLx_s = self.bn_afterDL(x_after_DLx_s)
        # BN_x_after_DLx_t = self.bn_afterDL(x_after_DLx_t)

        # # 先使用注意力机制再使用池化
        x_s_attention, aSourceLoss = self.attentionModel(x_s, self.channel, self.aSource)
        x_t_attention, aTargetLoss = self.attentionModel(x_t, self.channel, self.aTarget)
        aSTLoss = aSourceLoss + aTargetLoss

        x_embd_s = torch.cat(
            (self.avg_pool(x_s_attention.detach()).squeeze(), self.max_pool(x_s_attention.detach()).squeeze()),
            1)  # [B, 2*C]
        x_embd_t = torch.cat(
            (self.avg_pool(x_t_attention.detach()).squeeze(), self.max_pool(x_t_attention.detach()).squeeze()), 1)

        # x_embd_s = torch.cat(
        #     (self.avg_pool(x_s.detach()).squeeze(), self.max_pool(x_s.detach()).squeeze()), 1)  # [B, 2*C]
        # x_embd_t = torch.cat(
        #     (self.avg_pool(x_t.detach()).squeeze(), self.max_pool(x_t.detach()).squeeze()), 1)
        x_embd_s, x_embd_t = self.adaptiveFC1(x_embd_s), self.adaptiveFC1(x_embd_t)  # [B, C]
        x_embd = x_embd_s + x_embd_t
        x_embd = self.adaptiveFC2(x_embd)
        lam = self.adaptiveFC3(x_embd)
        lam = self.softmax(lam)  # [B, 2]

        # x_inter = lam[:, 0].reshape(-1, 1, 1, 1) * BN_x_after_DLx_s + lam[:, 1].reshape(-1, 1, 1, 1) * BN_x_after_DLx_t
        x_inter = lam[:, 0].reshape(-1, 1, 1, 1) * x_s + lam[:, 1].reshape(-1, 1, 1, 1) * x_t
        allDLfinal_loss = torch.tensor(0.00)
        out = torch.cat((x_s, x_t, x_inter), 0)

        # return out, lam, allDLfinal_loss
        return out, lam, allDLfinal_loss, aSTLoss

    # 混合高斯分布随机初始化稀疏编码X
    def gaussian_random_x(self, num_samples, num_features, sparsity):
        num_clusters = 5  # 高斯分布数量
        # 随机初始化混合高斯分布的均值和协方差矩阵
        means = torch.randn(num_clusters, num_features, device='cuda')  # 均值张量在GPU上
        covariances = [torch.eye(num_features, device='cuda') * torch.rand(num_features, device='cuda') for _ in
                       range(num_clusters)]

        # 生成混合高斯分布的随机样本
        torch.manual_seed(0)
        X = []
        for _ in range(num_samples):
            cluster_idx = torch.randint(num_clusters, (1,), device='cuda').item()
            sample = torch.distributions.MultivariateNormal(means[cluster_idx], covariances[cluster_idx]).sample()
            X.append(sample)
        X = torch.stack(X)

        # 根据稀疏度将部分元素置为0
        mask = torch.rand(X.size(), device='cuda') < sparsity
        X = X * mask.float()

        return X

    # KSVD算法
    def KSVD(self, Y, D, X, K):
        for k in range(K):
            index = torch.nonzero(X[k, :]).squeeze()
            # 0维张量变为1维张量
            if index.dim() == 0:
                index = index.unsqueeze(0)
            if len(index) == 0:
                continue
            r = (Y - torch.matmul(D, X)).clone()[:, index]
            with torch.no_grad():
                U, S, V_T = torch.linalg.svd(r.clone())
                D[:, k] = U[:, 0].clone()
                D[:, k] = U[:, 0]
                V_T = V_T.permute(1, 0)
                Xk = S[0] * V_T[0, :len(index)]
                X[k, index] = Xk
        return D, X

    # 每张图片创建自己的字典
    def createDictory(self, Y, K, HWmul):
        # 初始化D，从Y中随机选取K列作为D
        # U最大是图片的行数
        oldY = Y.clone()
        # oldY = (oldY - oldY.mean()) / oldY.std()
        # 若Y中含有Nan值，则在此处替换
        # if torch.isnan(oldY).any():
        #     oldY = torch.where(torch.isnan(oldY), torch.tensor(self.F2TDReidepsilon, device=oldY.device), oldY)
        # print(oldY)

        # 由于奇异值分解有时会不收敛，若不收敛把nan换为其他的值
        # try:
        # U,   _, _ = torch.linalg.svd(oldY)
        # except:
        #     U[torch.isnan(U)] = torch.rand(torch.sum(torch.isnan(U)))

        B = oldY.shape[0]
        for i in range(B):
            # if  K > U.shape[-1]:
            #     K = U.shape[-1]
            U, _, _ = torch.linalg.svd(oldY[i].clone())

            # DUK = U[i, :, :K].clone()
            DUK = U[:, :K].clone()
            newY = oldY[i].clone()
            # #  由于OMP传入的参数不能线性相关，此处标准化DUK和newY
            # normalized_DUK = (DUK - DUK.mean()) / DUK.std()
            # normalized_newY = (newY - newY.mean()) / newY.std()

            # 对抽取的特征迭代多次，计算更准确的字典
            # for KSVDIter in range(self.KSVDMaxIter):
            # 计算稀疏编码矩阵
            # X = torch.from_numpy(linear_model.orthogonal_mp(normalized_DUK.detach().cpu(),
            #                                                 normalized_newY.detach().cpu())).float().cuda()

            # 随机初始化稀疏编码矩阵
            # 定义稀疏编码矩阵的维度和稀疏度
            sparsity = 0.4  # 稀疏度
            # 随机初始化稀疏编码矩阵X
            X = torch.rand(DUK.shape[1], newY.shape[1]).cuda()  # 随机生成[0, 1]之间的值
            X[X > sparsity] = 0  # 根据稀疏度将部分元素置为0
            # 每一次更新D之后由OMP算法求得稀疏矩阵X
            # X = torch.Tensor(linear_model.orthogonal_mp((D - D.min()) / (D.max() - D.min()), Y))
            # 混合高斯分布随机初始化稀疏编码Xfull
            # X = self.gaussian_random_x(DUK.shape[1], newY.shape[1], sparsity).cuda()

            # KSVD算法更新D
            # D_KSVD, X_KSVD = self.KSVD(normalized_newY, normalized_DUK, X, K)
            D_KSVD, X_KSVD = self.KSVD(newY, DUK, X, K)

            # 计算二范式
            # 标准化
            # DXmul = torch.matmul(D_KSVD.clone(), X_KSVD.clone())
            # normalized_DXmul = (DXmul - DXmul.mean()) / DXmul.std()

            # loss = torch.norm(normalized_newY.clone() - normalized_DXmul) / (HWmul / 64)
            # if loss.item() < self.KSVDTolerance:
            #     break

            if i == 0:
                AllD_KSVD = D_KSVD.clone().unsqueeze(0)
                AllX_KSVD = X_KSVD.clone().unsqueeze(0)
            else:
                AllD_KSVD = torch.cat((AllD_KSVD.clone(), D_KSVD.clone().unsqueeze(0)), dim=0)
                AllX_KSVD = torch.cat((AllX_KSVD.clone(), X_KSVD.clone().unsqueeze(0)), dim=0)

        # 计算损失前先检查有没有Nan值，若有则替换。若不替换损失函数会为Nan
        # if torch.isnan(AllD_KSVD).any():
        #     AllD_KSVD = torch.where(torch.isnan(AllD_KSVD), torch.tensor(self.F2TDReidepsilon, device=AllD_KSVD.device), AllD_KSVD)
        # if torch.isnan(AllX_KSVD).any():
        #     AllX_KSVD = torch.where(torch.isnan(AllX_KSVD), torch.tensor(self.F2TDReidepsilon, device=AllX_KSVD.device), AllX_KSVD)

        # # 计算损失并输出
        # allDL_loss = torch.sqrt(
        #     torch.pow(oldY.clone() - torch.matmul(AllD_KSVD.clone(), AllX_KSVD.clone()), 2) + self.F2TDReidepsilon
        # )
        # # 由于数值比较大，把数值归一化到 0-1 之间
        # allDL_loss_02 = (allDL_loss - allDL_loss.min()) / (
        #         allDL_loss.max() - allDL_loss.min() + self.F2TDReidepsilon)
        # # 由于很多损失比较大，这里把损失重新限制在0-0.3之间
        # # 选择0.3的原因是，batch为64时，损失最大为38左右，不至于损失在总的比重中占比过大
        # loss_each_img = allDL_loss_02.clone().sum(dim=(-2, -1))
        # loss_each_img_02 = (loss_each_img - loss_each_img.min()) / (
        #         loss_each_img.max() - loss_each_img.min() + self.F2TDReidepsilon) * 0.3
        # allDLfinal_loss = loss_each_img_02.sum()

        # 计算损失
        # 标准化
        DXmul = torch.matmul(D_KSVD.clone(), X_KSVD.clone())
        normalized_DXmul = (DXmul - DXmul.mean()) / DXmul.std()

        allDL_loss = torch.norm(oldY.clone() - normalized_DXmul)
        allDL_loss = allDL_loss / HWmul

        return AllD_KSVD, AllX_KSVD, allDL_loss

    # 注意力模块
    def attentionModel(self, image_tensor, inChannel, a):
        # 创建1x1卷积层q
        qConv = nn.Conv2d(in_channels=inChannel, out_channels=inChannel // 8, kernel_size=1) \
            .to(image_tensor.device)
        # 创建1x1卷积k
        kConv = nn.Conv2d(in_channels=inChannel, out_channels=inChannel // 8, kernel_size=1) \
            .to(image_tensor.device)
        # 创建1x1卷积层v
        vConv = nn.Conv2d(in_channels=inChannel, out_channels=inChannel, kernel_size=1) \
            .to(image_tensor.device)

        m_batchsize, C, height, width = image_tensor.size()

        Q = qConv(image_tensor).view(m_batchsize, -1, width * height).permute(0, 2, 1).contiguous()
        K = kConv(image_tensor).view(m_batchsize, -1, width * height).contiguous()
        V = vConv(image_tensor).view(m_batchsize, -1, width * height).contiguous()

        scores = torch.bmm(Q, K)

        # 计算亲和力矩阵
        QKaffinityMatrix = F.softmax(scores, dim=-1)

        # 计算注意力图（attention map）
        attentionMap = torch.matmul(V, QKaffinityMatrix.permute(0, 2, 1))
        attentionMap = attentionMap.view(m_batchsize, C, height, width).contiguous()

        resultImg = a * attentionMap + image_tensor
        loss = F.mse_loss(a * resultImg, image_tensor)

        return resultImg, loss
