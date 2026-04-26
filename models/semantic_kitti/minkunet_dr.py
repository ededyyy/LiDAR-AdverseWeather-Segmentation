import math
import random
from collections import deque, namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsparse
import torchsparse.nn as spnn

__all__ = ['MinkUNet_DR']


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory:

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size: int):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class PointDropDQN(nn.Module):
    """A tiny point-wise policy network for learnable point dropping.

    This mirrors LiDARWeather's idea: use (loss, uncertainty) as a global state,
    combine it with per-point features, and output a keep/drop probability per point.
    """

    def __init__(self, n_observations: int = 2, point_feat_dim: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(n_observations + point_feat_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state: torch.Tensor, point_feats: torch.Tensor) -> torch.Tensor:
        # state: (1, n_observations) or (B, n_observations)
        # point_feats: (N, point_feat_dim)
        if state.dim() == 1:
            state = state.view(1, -1)
        state = state.expand(point_feats.shape[0], -1)
        x = torch.cat([state, point_feats], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # keep probability per point
        return torch.sigmoid(self.fc3(x)).squeeze(1)

class BasicConvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        stride=stride,
                        transposed=True),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):

    def __init__(self, inc, outc, ks=3, stride=1, dilation=1):
        super().__init__()
        self.net = nn.Sequential(
            spnn.Conv3d(inc,
                        outc,
                        kernel_size=ks,
                        dilation=dilation,
                        stride=stride),
            spnn.BatchNorm(outc),
            spnn.ReLU(True),
            spnn.Conv3d(outc, outc, kernel_size=ks, dilation=dilation,
                        stride=1),
            spnn.BatchNorm(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                spnn.Conv3d(inc, outc, kernel_size=1, dilation=1,
                            stride=stride),
                spnn.BatchNorm(outc),
            )

        self.relu = spnn.ReLU(True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out

class MinkUNet_DR(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        cr = kwargs.get('cr', 1.0)
        cs = [32, 32, 64, 128, 256, 256, 128, 96, 96]
        cs = [int(cr * x) for x in cs]
        self.run_up = kwargs.get('run_up', True)

        self.stem = nn.Sequential(
            spnn.Conv3d(4, cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True),
            spnn.Conv3d(cs[0], cs[0], kernel_size=3, stride=1),
            spnn.BatchNorm(cs[0]), spnn.ReLU(True))

        self.stage1 = nn.Sequential(
            BasicConvolutionBlock(cs[0], cs[0], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[0], cs[1], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[1], cs[1], ks=3, stride=1, dilation=1),
        )

        self.stage2 = nn.Sequential(
            BasicConvolutionBlock(cs[1], cs[1], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[1], cs[2], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[2], cs[2], ks=3, stride=1, dilation=1))

        self.stage3 = nn.Sequential(
            BasicConvolutionBlock(cs[2], cs[2], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[2], cs[3], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[3], cs[3], ks=3, stride=1, dilation=1),
        )

        self.stage4 = nn.Sequential(
            BasicConvolutionBlock(cs[3], cs[3], ks=2, stride=2, dilation=1),
            ResidualBlock(cs[3], cs[4], ks=3, stride=1, dilation=1),
            ResidualBlock(cs[4], cs[4], ks=3, stride=1, dilation=1),
        )

        self.up1 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[4], cs[5], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[5] + cs[3], cs[5], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[5], cs[5], ks=3, stride=1, dilation=1),
            )
        ])

        self.up2 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[5], cs[6], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[6] + cs[2], cs[6], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[6], cs[6], ks=3, stride=1, dilation=1),
            )
        ])

        self.up3 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[6], cs[7], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[7] + cs[1], cs[7], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[7], cs[7], ks=3, stride=1, dilation=1),
            )
        ])

        self.up4 = nn.ModuleList([
            BasicDeconvolutionBlock(cs[7], cs[8], ks=2, stride=2),
            nn.Sequential(
                ResidualBlock(cs[8] + cs[0], cs[8], ks=3, stride=1, dilation=1),
                ResidualBlock(cs[8], cs[8], ks=3, stride=1, dilation=1),
            )
        ])

        self.classifier = nn.Sequential(nn.Linear(cs[8], kwargs['num_classes']))

        self.point_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(cs[0], cs[4]),
                nn.BatchNorm1d(cs[4]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[4], cs[6]),
                nn.BatchNorm1d(cs[6]),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Linear(cs[6], cs[8]),
                nn.BatchNorm1d(cs[8]),
                nn.ReLU(True),
            )
        ])

        self.weight_initialization()
        self.dropout = nn.Dropout(0.3, True)

        # projection head
        self.proj = nn.Sequential(
            nn.Linear(cs[8], cs[8]),
            nn.ReLU(inplace=True),
            nn.Linear(cs[8], 128))

        # create the momentum memory bank to save prototypes
        self.m = 0.99  # momentum update rate
        self.register_buffer("memo_bank", torch.randn(kwargs['num_classes'], 128))
        self.memo_bank = self.memo_bank * 0.

        # Learnable point drop (LPD) - optional.
        self.learnable_drop = kwargs.get('learnable_drop', False)
        if self.learnable_drop:
            self._init_lpd(kwargs)

    def _init_lpd(self, kwargs):
        # Hyper-parameters (defaults follow LiDARWeather's style).
        self.dqn_train_start_iter = int(kwargs.get('dqn_train_start_iter', 32))
        self.dqn_gamma = float(kwargs.get('dqn_gamma', 0.99))
        self.dqn_eps_start = float(kwargs.get('dqn_eps_start', 0.9))
        self.dqn_eps_end = float(kwargs.get('dqn_eps_end', 0.05))
        self.dqn_eps_decay = float(kwargs.get('dqn_eps_decay', 1000))
        self.dqn_tau = float(kwargs.get('dqn_tau', 0.005))
        self.dqn_replay_memory_size = int(kwargs.get('dqn_replay_memory_size', 10000))
        self.drop_threshold = float(kwargs.get('dqn_drop_threshold', 0.5))
        self.drop_threshold_explore = float(kwargs.get('dqn_drop_threshold_explore', 0.5))

        self.policy_net = PointDropDQN(n_observations=2, point_feat_dim=4)
        self.target_net = PointDropDQN(n_observations=2, point_feat_dim=4)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        for p in self.target_net.parameters():
            p.requires_grad = False
        self.memory = ReplayMemory(self.dqn_replay_memory_size)
        self.steps_done = 0

    @torch.no_grad()
    def lpd_select_action(self, state: torch.Tensor, point_feats: torch.Tensor) -> torch.Tensor:
        """Return a boolean keep-mask for points (True=keep)."""
        sample = random.random()
        eps_threshold = self.dqn_eps_end + (self.dqn_eps_start - self.dqn_eps_end) * math.exp(
            -1.0 * self.steps_done / self.dqn_eps_decay
        )
        self.steps_done += 1

        if sample > eps_threshold:
            keep_prob = self.policy_net(state, point_feats)
            return keep_prob >= self.drop_threshold
        else:
            out = torch.rand(point_feats.shape[0], device=point_feats.device)
            return out >= self.drop_threshold_explore

    def lpd_soft_update_target(self) -> None:
        """Polyak averaging update: target = tau*policy + (1-tau)*target."""
        if not self.learnable_drop:
            return
        with torch.no_grad():
            for t, s in zip(self.target_net.parameters(), self.policy_net.parameters()):
                t.data.mul_(1.0 - self.dqn_tau).add_(self.dqn_tau * s.data)

    def lpd_dqn_loss(self, batch_state: torch.Tensor, batch_point_feats: torch.Tensor,
                     batch_reward: torch.Tensor, batch_next_state: torch.Tensor) -> torch.Tensor:
        """A lightweight DQN-style regression loss (broadcasted to points).

        Note: This follows LiDARWeather's implementation pattern (scalar reward/state).
        """
        q = self.policy_net(batch_state, batch_point_feats)  # (N,)
        with torch.no_grad():
            q_next = self.target_net(batch_next_state, batch_point_feats)  # (N,)
            target = batch_reward + self.dqn_gamma * q_next.mean()
        return F.smooth_l1_loss(q, target.expand_as(q))

    @torch.no_grad()
    def momentum_update_key_encoder(self, feat, init=False):
        """
        Momentum update of the memo_bank
        """
        if init:
            self.memo_bank = feat
        else:
            self.memo_bank = self.memo_bank * self.m + feat * (1. - self.m)

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x0 = self.stem(x)
        x1 = self.stage1(x0)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        y1 = self.up1[0](x4)
        y1 = torchsparse.cat([y1, x3])
        y1 = self.up1[1](y1)

        y2 = self.up2[0](y1)
        y2 = torchsparse.cat([y2, x2])
        y2 = self.up2[1](y2)

        y3 = self.up3[0](y2)
        y3 = torchsparse.cat([y3, x1])
        y3 = self.up3[1](y3)

        y4 = self.up4[0](y3)
        y4 = torchsparse.cat([y4, x0])
        y4 = self.up4[1](y4)

        out = self.classifier(y4.F)

        feat = self.proj(y4.F)
        return out, feat