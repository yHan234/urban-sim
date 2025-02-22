# based on https://github.com/ndrplz/ConvLSTM_pytorch

import torch
from torch import nn


class ConvLSTMCell(nn.Module):

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        kernel_size: tuple,
        bias: bool = True,
        dropout: float = 0.0,
        layer_norm: bool = False,
        input_shape: tuple[int] | None = None,
    ):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.bias = bias

        self.kernel_size = kernel_size
        if isinstance(kernel_size, int):
            self.padding = (kernel_size // 2, kernel_size // 2)
        else:
            self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.conv = nn.Conv2d(
            in_channels=in_channels + self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias,
        )

        ln_shape = (hidden_channels,) + input_shape
        self.ln_i = nn.LayerNorm(ln_shape) if layer_norm else lambda x: x
        self.ln_f = nn.LayerNorm(ln_shape) if layer_norm else lambda x: x
        self.ln_o = nn.LayerNorm(ln_shape) if layer_norm else lambda x: x
        self.ln_g = nn.LayerNorm(ln_shape) if layer_norm else lambda x: x

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor, cur_state):
        # input tensor (batch_size, in_channels, height, width)

        # h, c (batch_size, hidden_channels, height, width)
        h_cur, c_cur = cur_state

        # combined (batch_size, in_channels + hidden_channels, height, width)
        combined = torch.cat(
            (input_tensor, h_cur), dim=1
        )  # concatenate along channel axis

        # combined_conv (batch_size, 4 * hidden_channels, height, width)
        combined_conv = self.conv(combined)

        # i, f, o, g (batch_size, hidden_channels, height, width)
        i, f, o, g = torch.split(combined_conv, self.hidden_channels, dim=1)
        i, f, o, g = self.ln_i(i), self.ln_f(f), self.ln_o(o), self.ln_g(g)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)

        # h, c (batch_size, hidden_channels, height, width)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        h_next = self.dropout(h_next)

        return h_next, c_next

    def begin_state(self, batch_size: int, image_size: tuple):
        height, width = image_size
        return (
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
            torch.zeros(
                batch_size,
                self.hidden_channels,
                height,
                width,
                device=self.conv.weight.device,
            ),
        )
