import torch

import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import numpy as np

import matplotlib.pyplot as plt

import torchvision.transforms as transforms

# from torchvision import datasets
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader, Subset

import itertools

# 하나의 optimizer에 여러 모듈의 parameter 넣을 때 사용
# https://discuss.pytorch.org/t/giving-multiple-parameters-in-optimizer/869/8

import os
import json

from sklearn.model_selection import train_test_split

from modules import *

import argparse

import datetime
from pytz import timezone

from torchvision.utils import save_image
from PIL import Image

import wandb
import matplotlib.pyplot as plt
import seaborn as sns


def similarity(l, sam):
    # similarity = isotropic scaling + isometry(rotation + translation)
    s = torch.sin(l * np.pi / 5)
    c = torch.cos(l * np.pi / 5)

    device = l.device

    scale = torch.tensor([[np.sqrt(5.0, dtype=np.float32), 0], [0, np.sqrt(0.5, dtype=np.float32)]]).to(
        device
    )
    # scale = sqrt(eigen_value)
    # 여기서 eigen_value는 covariance matrix가 diagonal일때 diagonal entries (x, y의 분산)

    rot = torch.stack([torch.stack([c, -s]), torch.stack([s, c])])

    sample = sam @ scale @ rot.T
    # scale, rotation 변환
    sample = sample + (torch.tensor([10.0, 0.0]).to(device) @ rot.T)
    # 좌표 이동
    return sample


def create_loader(dir, batch_size):
    train_dataset = MNIST(
        root="C:/Users/paoki/workspace/AAE/mnist",
        train=True,
        # transform=transforms.Compose([transforms.ToTensor()]),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
        # ToTensor: Tensor 변환 + 0~1 사이 값 갖도록 변경
        # Normalize: mean을 뺀 후 std로 나눠준다 0-0.5~1-0.5 => -0.5~0.5 => -0.5/0.5~0.5/0.5 => -1~1
    )
    test_dataset = MNIST(
        root="C:/Users/paoki/workspace/AAE/mnist",
        train=False,
        # transform=transforms.Compose([transforms.ToTensor()]),
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]),
    )

    unsv_idx, sv_idx = train_test_split(
        np.arange(len(train_dataset)), test_size=10000, shuffle=True, stratify=train_dataset.targets
    )

    unsv_set = Subset(train_dataset, unsv_idx)
    sv_set = Subset(train_dataset, sv_idx)

    unsv_loader = DataLoader(dataset=unsv_set, batch_size=batch_size, shuffle=True)
    sv_loader = DataLoader(dataset=sv_set, batch_size=batch_size, shuffle=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

    return unsv_loader, sv_loader, test_loader


def get_lr(self, optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


# normal distribution을 따르는 임의의 z를 건네어 주면 진짜와 유사한 MNIST 이미지 생성
def sample_image(n_row, epoch, decoder, device, timestamp):
    """Saves a grid of generated digits"""
    # Sample noise
    sam = torch.Tensor(np.random.normal(0, 1, (n_row, 2))).to(device)

    # label = torch.from_numpy(np.random.randint(low=0, high=10, size=n_row)).to(device)
    label = torch.from_numpy(np.arange(0, 10)).to(device)

    z_real = torch.vmap(similarity)(label, sam)
    # 0~9 생성에 사용할 z_real
    gen_imgs = decoder(z_real)

    gen_imgs = gen_imgs.view(n_row, 1, 28, 28)

    save_image(gen_imgs.data, f"images/{timestamp}_{epoch+1}.png", nrow=n_row, normalize=True)

    return gen_imgs, label, z_real


def train(hp):
    # hyperparameter
    epochs = hp["epochs"]

    # dataset, dataloader
    dataset_dir = "C:/Users/paoki/workspace/AAE/mnist"
    batch_size = hp["batch_size"]

    unsv_loader, sv_loader, test_loader = create_loader(dataset_dir, batch_size)

    # model intance 생성 및 초기화
    encoder = hp["encoder"]
    decoder = hp["decoder"]
    discriminator = hp["discriminator"]

    # loss, optimizer, scheduler
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCELoss()

    params = [encoder.parameters(), decoder.parameters()]
    # https://discuss.pytorch.org/t/giving-multiple-parameters-in-optimizer/869/8
    optim_ae = hp["optim_ae"]
    scheduler_ae = hp["scheduler_ae"]
    optim_en = hp["optim_en"]
    scheduler_en = hp["scheduler_en"]
    optim_dc = hp["optim_dc"]
    scheduler_dc = hp["scheduler_dc"]

    wandb.init(
        project="AAE",
        entity="pao-kim-si-woong",
        config={
            "lr_ae": 0.01,
            "lr_en": 0.1,
            "lr_dc": 0.1,
            "dataset": "MNIST",
            "n_epochs": epochs,
            "loss": "BCE",
            "notes": "AAE 실험",
        },
        name=hp["run_name"],
        mode=hp["wandb"],
    )

    wandb.watch((encoder, decoder, discriminator))

    # 샘플 이미지 생성에 사용
    z_real_list = np.zeros((epochs, 2), dtype=np.float32)
    label_list = np.zeros(epochs, dtype=np.float32)

    real = torch.full((batch_size, 1), 1.0).to(hp["device"])
    fake = torch.full((batch_size, 1), 0.0).to(hp["device"])

    for epoch in range(epochs):
        # train loop
        encoder.train()
        decoder.train()
        discriminator.train()

        loss_sv_r_value = 0
        loss_sv_d_value = 0
        loss_sv_g_value = 0

        # supervised
        for i, (img, label) in enumerate(sv_loader):
            img = img.to(hp["device"])
            label = label.to(hp["device"])

            # Reconstruction Phase
            optim_ae.zero_grad()
            z = encoder(img)

            x_rc = decoder(z)
            x_rc = x_rc.view(x_rc.size(0), 1, 28, 28)

            loss_reconstruction = criterion_mse(x_rc, img)  # loss_reconstruction 계산
            loss_reconstruction.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(*params), 1)
            optim_ae.step()

            # Regularization Phase

            # discriminator
            optim_dc.zero_grad()

            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # https://www.visiondummy.com/2014/04/geometric-interpretation-covariance-matrix/
            # N(0,1) 2차원 white data를 먼저 추출한 후 변환하는 방식으로 변경하기
            sam = torch.Tensor(np.random.normal(0, 1, (img.size(0), 2))).to(hp["device"])

            z_real = torch.vmap(similarity)(label, sam)

            # label이 지정하는 분포에서 표본 추출

            class_label = F.one_hot(label, num_classes=11)
            # 11차원 one-hot vector로 변경
            # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
            # MultivariateNormal
            # torch.distributions.multivariate_normal.MultivariateNormal(loc, covariance_matrix=None, precision_matrix=None, scale_tril=None, validate_args=None)
            # https://stackoverflow.com/questions/69024270/how-to-create-a-normal-2d-distribution-in-pytorch

            z = encoder(img)
            # simultaneous gradient ascent-descent를 할지, alternating gradient ascent-descent를 할지 결정해야한다

            z_fake_input = torch.concat((z, class_label), dim=-1)
            z_real_input = torch.concat((z_real, class_label), dim=-1)

            # BCE에 넣을 real, fake label 정보
            p_z_real = discriminator(z_real_input)
            p_z_fake = discriminator(z_fake_input)

            loss_D = criterion_bce(p_z_real, real) + criterion_bce(p_z_fake, fake)  # loss_D 계산
            # loss_D.backward(retain_graph=True)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            optim_dc.step()

            mean_p_real_sv_d = torch.mean(p_z_real).item()
            mean_p_fake_sv_d = torch.mean(p_z_fake).item()

            # generator
            optim_en.zero_grad()

            z = encoder(img)

            z_fake_input = torch.concat((z, class_label), dim=-1)
            # simultaneous gradient ascent-descent를 할지, alternating gradient ascent-descent를 할지 결정해야한다

            loss_G = -1 * (
                criterion_bce(discriminator(z_real_input), real)
                + criterion_bce(discriminator(z_fake_input), fake)
            )  # loss_G 계산 -1 * loss_D
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            optim_en.step()

            # loss 값들 print, log
            loss_sv_r_value += (loss_reconstruction).item()
            loss_sv_d_value += (loss_D).item()
            loss_sv_g_value += (loss_G).item()

            log_interval = 100
            if (i + 1) % log_interval == 0:
                loss_sv_r_value_mean = loss_sv_r_value / log_interval
                loss_sv_d_value_mean = loss_sv_d_value / log_interval
                loss_sv_g_value_mean = loss_sv_g_value / log_interval

                print(
                    f"Epoch[{epoch}/{epochs}]({i + 1}/{len(sv_loader)}) || "
                    f"r_loss_sv {loss_sv_r_value_mean:4.4} d_loss_sv {loss_sv_d_value_mean:4.4} g_loss_sv {loss_sv_g_value_mean:4.4}"
                )

                loss_sv_r_value = 0
                loss_sv_d_value = 0
                loss_sv_g_value = 0

        loss_unsv_r_value = 0
        loss_unsv_d_value = 0
        loss_unsv_g_value = 0

        # unsupervised
        for i, (img, _) in enumerate(unsv_loader):
            img = img.to(hp["device"])

            # Reconstruction Phase
            optim_ae.zero_grad()

            z = encoder(img)

            x_rc = decoder(z)
            x_rc = x_rc.view(x_rc.size(0), 1, 28, 28)

            loss_reconstruction = criterion_mse(x_rc, img)  # loss_reconstruction 계산
            loss_reconstruction.backward()
            torch.nn.utils.clip_grad_norm_(itertools.chain(*params), 1)
            optim_ae.step()

            # Regularization Phase
            # discriminator
            optim_dc.zero_grad()
            class_label = torch.zeros(img.size(0), 11).to(hp["device"])

            sam = torch.Tensor(np.random.normal(0, 1, (img.size(0), 2))).to(hp["device"])

            label = torch.from_numpy(np.random.randint(low=0, high=10, size=img.size(0))).to(hp["device"])

            z_real = torch.vmap(similarity)(label, sam)
            # 전체 mixture 분포에서 표본 추출
            class_label[:, -1] = 1
            # 11차원 one-hot vector, label 없음을 마지막 11번째 자리에 1을 넣어 표현

            z = encoder(img)
            # simultaneous gradient ascent-descent를 할지, alternating gradient ascent-descent를 할지 결정해야한다

            z_fake_input = torch.concat((z, class_label), dim=-1)
            z_real_input = torch.concat((z_real, class_label), dim=-1)

            p_z_real = discriminator(z_real_input)
            p_z_fake = discriminator(z_fake_input)

            loss_D = criterion_bce(p_z_real, real) + criterion_bce(p_z_fake, fake)  # loss_D 계산
            # loss_D.backward(retain_graph=True)
            loss_D.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1)
            optim_dc.step()

            mean_p_real_unsv_d = torch.mean(p_z_real).item()
            mean_p_fake_unsv_d = torch.mean(p_z_fake).item()

            # generator
            optim_en.zero_grad()

            z = encoder(img)

            z_fake_input = torch.concat((z, class_label), dim=-1)
            # simultaneous gradient ascent-descent를 할지, alternating gradient ascent-descent를 할지 결정해야한다

            loss_G = -1 * (
                criterion_bce(discriminator(z_real_input), real)
                + criterion_bce(discriminator(z_fake_input), fake)
            )  # loss_G 계산 -1 * loss_D
            loss_G.backward()
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 1)
            optim_en.step()

            # loss 값들 print, log
            loss_unsv_r_value += (loss_reconstruction).item()
            loss_unsv_d_value += (loss_D).item()
            loss_unsv_g_value += (loss_G).item()

            log_interval = 500
            if (i + 1) % log_interval == 0:
                loss_unsv_r_value_mean = loss_unsv_r_value / log_interval
                loss_unsv_d_value_mean = loss_unsv_d_value / log_interval
                loss_unsv_g_value_mean = loss_unsv_g_value / log_interval

                print(
                    f"Epoch[{epoch}/{epochs}]({i + 1}/{len(unsv_loader)}) || "
                    f"r_loss_unsv {loss_unsv_r_value_mean:4.4} d_loss_unsv {loss_unsv_d_value_mean:4.4} g_loss_unsv {loss_unsv_g_value_mean:4.4}"
                )

                loss_unsv_r_value = 0
                loss_unsv_d_value = 0
                loss_unsv_g_value = 0

        scheduler_ae.step()
        scheduler_en.step()
        scheduler_dc.step()

        # test loop
        with torch.no_grad():
            print("Calculating validation results...")
            encoder.eval()
            decoder.eval()
            discriminator.eval()

            loss_test_r_list = []
            loss_test_d_list = []
            loss_test_g_list = []

            for img_t, label in test_loader:
                img_t = img_t.to(hp["device"])
                label = label.to(hp["device"])

                z_t = encoder(img_t)
                x_rc_t = decoder(z_t)
                x_rc_t = x_rc_t.view(x_rc_t.size(0), 1, 28, 28)
                loss_test_reconstruction = criterion_mse(x_rc_t, img_t)  # loss_reconstruction 계산

                # Regularization Phase
                # discriminator
                class_label = torch.zeros(img_t.size(0), 11).to(hp["device"])

                sam = torch.Tensor(np.random.normal(0, 1, (img_t.size(0), 2))).to(hp["device"])

                z_real = torch.vmap(similarity)(label, sam)
                # label이 지정하는 분포에서 표본 추출

                class_label = F.one_hot(label, num_classes=11)
                # 11차원 one-hot vector로 변경

                z_fake_input = torch.concat((z_t, class_label), dim=-1)
                z_real_input = torch.concat((z_real, class_label), dim=-1)

                p_z_real_test = discriminator(z_real_input)
                p_z_fake_test = discriminator(z_fake_input)

                loss_test_D = criterion_bce(p_z_real_test, real) + criterion_bce(
                    p_z_fake_test, fake
                )  # loss_D 계산

                mean_p_real_test_d = torch.mean(p_z_real_test).item()
                mean_p_fake_test_d = torch.mean(p_z_fake_test).item()

                # generator

                loss_test_G = -1 * (
                    criterion_bce(discriminator(z_real_input), real)
                    + criterion_bce(discriminator(z_fake_input), fake)
                )  # loss_G 계산 -1 * loss_D

                loss_test_r_list.append(loss_test_reconstruction.item())
                loss_test_d_list.append(loss_test_D.item())
                loss_test_g_list.append(loss_test_G.item())

            loss_test_r = np.sum(loss_test_r_list) / len(test_loader)
            loss_test_d = np.sum(loss_test_d_list) / len(test_loader)
            loss_test_g = np.sum(loss_test_g_list) / len(test_loader)

            print(f"[test] loss_r: {loss_test_r:4.2}, loss_d: {loss_test_d:4.2}, loss_g: {loss_test_g:4.2}")

        new_wandb_metric_dict = {
            "valid_loss_r": loss_test_r,
            "valid_loss_d": loss_test_d,
            "valid_loss_g": loss_test_g,
            "train_loss_sv_r": loss_sv_r_value_mean,
            "train_loss_sv_d": loss_sv_d_value_mean,
            "train_loss_sv_g": loss_sv_g_value_mean,
            "train_loss_unsv_r": loss_unsv_r_value_mean,
            "train_loss_unsv_d": loss_unsv_d_value_mean,
            "train_loss_unsv_g": loss_unsv_g_value_mean,
            "mean_p_real_sv_d": mean_p_real_sv_d,
            "mean_p_fake_sv_d": mean_p_fake_sv_d,
            "mean_p_real_unsv_d": mean_p_real_unsv_d,
            "mean_p_fake_unsv_d": mean_p_fake_unsv_d,
            "mean_p_real_test_d": mean_p_real_test_d,
            "mean_p_fake_test_d": mean_p_fake_test_d,
        }
        wandb.log(new_wandb_metric_dict)

        # 10 epoch마다 10개 표본 추출해 생성한 이미지 저장
        if (epoch + 1) % 10 == 0:
            timestamp = str(datetime.datetime.now()).split(" ")
            timestamp = timestamp[0] + "-" + ";".join(timestamp[1].split(".")[0].split(":"))

            with torch.no_grad():
                gen_imgs, label, z_real = sample_image(
                    n_row=10, epoch=epoch, decoder=decoder, device=hp["device"], timestamp=timestamp
                )
            gen_imgs = gen_imgs.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            z_real = z_real.detach().cpu().numpy()

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, aspect=1)
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)

            if epoch != 9:
                sns.scatterplot(
                    x=z_real_list[: epoch - 9, 0],
                    y=z_real_list[: epoch - 9, 1],
                    hue=label_list[: epoch - 9],
                    palette="tab10",
                    ax=ax,
                )

            sns.scatterplot(
                x=z_real[:, 0],
                y=z_real[:, 1],
                hue=label,
                palette="tab10",
                marker="+",
                s=200,
                legend=False,
                ax=ax,
            )
            z_real_list[epoch - 9 : epoch + 1, :] += z_real
            label_list[epoch - 9 : epoch + 1] += label

            plt.savefig(f"images/{timestamp}_z_real_scatter.png")

            images = []
            images.append(
                wandb.Image(f"images/{timestamp}_z_real_scatter.png", caption=f"epoch {epoch+1} z_real")
            )

            # wandb 원본, 복원 이미지 비교
            zor_images = []

            imgs_o = img[:10].detach().clone().cpu().numpy()
            imgs_r = x_rc[:10].detach().clone().cpu().numpy()
            label_o = _[:10].detach().clone().cpu().numpy()
            z_encoded = z[:10].detach().clone().cpu().numpy()

            fig = plt.figure(figsize=(7, 7))
            ax = fig.add_subplot(111, aspect=1)
            ax.set_xlim(-25, 25)
            ax.set_ylim(-25, 25)

            sns.scatterplot(
                x=z_encoded[:, 0],
                y=z_encoded[:, 1],
                hue=label_o,
                palette="tab10",
                ax=ax,
            )
            plt.savefig(f"images/{timestamp}_z_encoded_scatter.png")

            for i in range(10):
                image = gen_imgs[i]
                caption = f"epoch {epoch+1} gen_label: {label[i]}"
                images.append(wandb.Image(image, caption=caption))

                image_o = imgs_o[i]
                caption = f"epoch {epoch+1} o_label: {label_o[i]}"
                zor_images.append(wandb.Image(image_o, caption=caption))
                image_r = imgs_r[i]
                caption = f"epoch {epoch+1} r_label: {label_o[i]}"
                zor_images.append(wandb.Image(image_r, caption=caption))

            zor_images.append(
                wandb.Image(f"images/{timestamp}_z_encoded_scatter.png", caption=f"epoch {epoch+1} z_real")
            )

            wandb.log({"Img_generated": images})
            wandb.log({"original vs reconstruction": zor_images})

    return encoder, decoder, discriminator


if __name__ == "__main__":
    hp = {}

    parser = argparse.ArgumentParser()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs to train (default: 100)")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="input batch size for training (default: 100)",
    )

    args = parser.parse_args()
    print(args)

    hp["device"] = device
    hp["epochs"] = args.epochs
    hp["batch_size"] = args.batch_size

    start_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 시작 : {str(start_time)[:19]}")

    train(hp)

    end_time = datetime.datetime.now(timezone("Asia/Seoul"))
    print(f"학습 끝 : {str(end_time)[:19]}")

    # 학습 소요 시간 계산 및 출력
    elapsed_time = end_time - start_time
    print(f"학습 소요 시간: {elapsed_time}")
