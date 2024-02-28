import torch

import torch.nn as nn
import torch.nn.functional as F

import numpy as np


def affinity(xy, std, sam):
    # similarity = isotropic scaling + isometry(rotation + translation)
    # isotropic scaling이 아니므로 similarity가 아니다 -> affinity

    device = std.device
    h = torch.tensor([[1.0], [0.0]]).to(device)
    x = xy @ h
    v = torch.tensor([[0.0], [1.0]]).to(device)
    y = xy @ v

    r = torch.sqrt(x**2 + y**2 + 1e-6)
    # sqrt 0이 되지 않도록 1e-6 추가

    # s = torch.sin(theta)
    s = y / (r + 1e-6)
    # 0으로 나누지 않도록 1e-6추가
    # c = torch.cos(theta)
    c = x / (r + 1e-6)

    scale = torch.diag(std)
    # scale = sqrt(eigen_value)
    # 여기서 eigen_value는 covariance matrix가 diagonal일때 diagonal entries (x, y의 분산)

    rot = torch.squeeze(torch.stack([torch.stack([c, -s]), torch.stack([s, c])]))
    # print(f"==>> rot.shape: {rot.shape}")

    sample = sam @ scale @ rot.T
    # scale, rotation 변환
    m = torch.tensor([1.0, 0.0]).to(device)
    translation = r * m
    translation = translation @ rot.T
    # TODO: (x,y)를 다시 계산하지않고 단순히 xy를 그대로 사용하도록 수정하기
    sample = sample + translation
    # print(f"==>> sample.shape: {sample.shape}")
    # 좌표 이동
    return sample
    # return torch.squeeze(sample)


def reparameterization(xy, std):
    device = std.device

    sam = torch.Tensor(np.random.normal(0, 1, (std.size(0), 2))).to(device)

    z = torch.vmap(affinity)(xy, std, sam)
    # print(f"==>> z.shape: {z.shape}")

    return z


class EncoderWithReParam(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.Linear(1000, odim),
            # nn.ReLU(),
        )

        # self.logvar = nn.Linear(1000, odim)
        self.std = nn.Sequential(nn.Linear(1000, odim), nn.ReLU())
        self.xy = nn.Linear(1000, odim)

        # self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.seq(x)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # r, θ, σ의 범위안의 값이 나오도록 해야 한다
        xy = self.xy(x)

        std = self.std(x)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        z = reparameterization(xy=xy, std=std)

        return z

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


def reparameterization5(mean, std, odim):
    device = std.device

    sam = torch.Tensor(np.random.normal(0, 1, (std.size(0), odim))).to(device)

    z = mean + std * sam
    # print(f"==>> z.shape: {z.shape}")

    return z


class EncoderExp5(nn.Module):
    def __init__(self, idim, odim, tau=1):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.Linear(1000, odim),
            # nn.ReLU(),
        )
        self.odim = odim
        self.tau = tau

        # self.logvar = nn.Linear(1000, odim)
        self.std = nn.Sequential(nn.Linear(1000, odim), nn.ReLU())
        self.mean = nn.Linear(1000, odim)
        # self.label = nn.Sequential(nn.Linear(1000, odim), nn.Softmax(dim=-1))
        self.label = nn.Linear(1000, odim)

        # self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.seq(x)

        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        mean = self.mean(x)

        std = self.std(x)

        z = reparameterization5(mean=mean, std=std, odim=self.odim)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@

        label = self.label(x)
        label = F.gumbel_softmax(label, tau=self.tau, dim=-1)
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html#torch.nn.functional.gumbel_softmax

        return z, label

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


class Decoder(nn.Module):
    def __init__(self, idim, odim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, odim),
            nn.Tanh(),
            # nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, x):
        x = self.seq(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


class Discriminator(nn.Module):
    def __init__(self, idim, classdim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim + classdim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, x):
        # print(f"==>> d input x: {x}")
        x = self.seq(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


class Discriminator5_Style(nn.Module):
    def __init__(self, idim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, x):
        # print(f"==>> d input x: {x}")
        x = self.seq(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화


class Discriminator5_Label(nn.Module):
    def __init__(self, idim):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Linear(idim, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            # nn.Linear(1000, 1000),
            # nn.BatchNorm1d(1000),
            # nn.ReLU(),
            nn.Linear(1000, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            # nn.Dropout(p=0.25),
            nn.Linear(1000, 1),
            nn.Sigmoid(),
        )

        self._initialize_weights()

    def forward(self, x):
        # print(f"==>> d input x: {x}")
        x = self.seq(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # self.modules() 은 모델 안에 들어있는 모든 module들(nn.Conv2d, nn.ReLu, nn.Linear 등)을 순회하는 iterator

            if isinstance(m, nn.Linear):
                # isinstance(확인할 클래스 객체, 클래스이름) 으로 한 객체가 특정 클래스의 객체인지를 알 수 있다
                nn.init.normal_(m.weight, mean=0.0, std=0.01)

                if m.bias is not None:
                    # batch normalization을 하고 있기때문에 conv는 bias를 따로 만들지 않아야한다.
                    nn.init.constant_(m.bias, 0)
                    # nn.init.constant_(tensor, value) 는 주어진 tensor의 모든 entry들을 정해진 value값으로 바꿔준다

            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                # BatchNorm(x) = γx + β 에서
                # γ = 1, β = 0 으로 초기화
