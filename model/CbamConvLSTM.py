import torch
from torch import nn
from model.CBAM import CBAM
from model.ConvLSTMCell import ConvLSTMCell


class CbamConvLSTM(nn.Module):

    def __init__(
        self,
        in_channels: int | tuple[int] | list[tuple[int]],
        hidden_channels: int,
        num_layers: int,
        kernel_size: int | tuple[int] | list[tuple[int]],
        cbam_sa_kernel_size: int,
        bias: bool = True,
        dropout: float = 0.0,
        batch_norm: bool = False,
        retain_attention_outputs=False,
    ):
        super().__init__()

        # hidden_channels = self._extend_for_multi_layers(hidden_channels, num_layers)
        # kernel_size = self._extend_for_multi_layers(kernel_size, num_layers)

        self.num_layers = num_layers

        self.cbam = CBAM(
            ca_num_channels=in_channels,
            sa_kernel_size=cbam_sa_kernel_size,
            # dropout=dropout,
            retain_output=retain_attention_outputs,
        )

        self.embed = nn.Conv2d(in_channels, hidden_channels[0], 1, 1, 0)
        self.activation = nn.ReLU()

        self.cell_list = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_channels=hidden_channels[i],
                    hidden_channels=hidden_channels[i + 1],
                    kernel_size=kernel_size[i],
                    bias=bias,
                    dropout=dropout,
                    batch_norm=batch_norm,
                )
                for i in range(self.num_layers)
            ]
        )

        self.output = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels[-1],
                out_channels=1,
                kernel_size=1,
                bias=bias,
            ),
            nn.Sigmoid(),
        )

    @staticmethod
    def _extend_for_multi_layers(size: int | list[int] | tuple[int], num_layers: int):
        if isinstance(size, int):
            return [size] * num_layers
        else:
            assert isinstance(size, (list, tuple))
            assert len(size) == num_layers
            return size

    def forward(
        self,
        prefix: torch.Tensor,
        pred_len: int,
        spatial: torch.Tensor | None,
        teacher_tensor: torch.Tensor | None = None,
        teacher_mask: list[torch.Tensor] | None = None,
    ):
        # prefix, teacher  (t, b, c, h, w)
        # spatial          (   b, c, h, w)

        assert prefix.size()[-2:] == spatial.size()[-2:]
        if teacher_tensor is not None:
            assert pred_len == teacher_tensor.size(0)
            assert prefix.size()[-2:] == teacher_tensor.size()[-2:]
            if teacher_mask is not None:
                assert teacher_tensor.size() == teacher_mask.size()

        state = [
            self.cell_list[i].begin_state(
                prefix.size(1), (prefix.size(-2), prefix.size(-1))
            )
            for i in range(self.num_layers)
        ]

        # Warm-up
        for t in range(prefix.size(0) - 1):
            # x_step (b, c, h, w)
            x_step = prefix[t]
            if spatial is not None:
                x_step = torch.cat((x_step, spatial), dim=1)
            _, state = self._forward_step(x_step, state)

        # Prediction
        y_hat = []
        for t in range(pred_len):
            # x_step (b, c, h, w)
            x_step = y_hat[-1] if y_hat else prefix[-1]

            if teacher_tensor is not None:
                if teacher_mask is None:
                    x_step = teacher_tensor[t]
                else:
                    x_step = teacher_tensor[t] * teacher_mask[t] + x_step * (
                        1 - teacher_mask[t]
                    )

            if spatial is not None:
                x_step = torch.cat((x_step, spatial), dim=1)
            y_step, state = self._forward_step(x_step, state)
            y_hat.append(y_step)

        # y_hat (pred_len, b, c, h, w)
        y_hat = torch.stack(y_hat)

        return y_hat

    def _forward_step(self, x, h):
        x = self.cbam(x)
        x = self.embed(x)
        x = self.activation(x)

        for i, cell in enumerate(self.cell_list):
            h[i] = cell(input_tensor=x if i == 0 else h[i - 1][0], cur_state=h[i])

        y_hat = self.output(h[-1][0])

        return y_hat, h
