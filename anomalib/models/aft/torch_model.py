from torch import Tensor
import torch.nn as nn
import torch
import math


class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_input_channel,
        n_feature,
        latent_vec_size,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()

        self.input_layers=nn.Sequential()
        self.input_layers.add_module(
            f"initial-conv-{num_input_channel}-{n_feature}",
            nn.Conv1d(num_input_channel, n_feature, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.input_layers.add_module(f"initial-relu-{n_feature}", nn.LeakyReLU(0.2,inplace=True))

        self.extra_layer = nn.Sequential()

        self.pyramid_features = nn.Sequential()
        pyramid_dim = input_size // 2
        while pyramid_dim > 50:
            in_features = n_feature
            out_features = n_feature * 2
            self.pyramid_features.add_module(
                f"pyramid-{in_features}-{out_features}-conv",
                nn.Conv1d(in_features, out_features, kernel_size=4, stride=2, padding=1, bias=False)
            )
            self.pyramid_features.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm1d(out_features))
            self.pyramid_features.add_module(f"pyramid-{out_features}-relu", nn.LeakyReLU(0.2, inplace=True))
            n_feature = out_features
            pyramid_dim=pyramid_dim // 2
        
        if add_final_conv_layer:
            self.final_conv_layer = nn.Conv1d(
                n_feature,
                latent_vec_size,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False
            )


    def forward(self,input_tensor:Tensor):
        output = self.input_layers(input_tensor)
        output = self.extra_layer(output)
        output = self.pyramid_features(output)
        if self.final_conv_layer is not None:
            output = self.final_conv_layer(output)
        return output

class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        n_features,
        latent_vec_size,
        num_input_channels,
    ) -> None:
        super().__init__()
        
        self.latent_input = nn.Sequential()
        exp_factor = math.ceil(math.log(input_size // 2, 2)) - 2
        n_input_features = n_features * (2**exp_factor)

        # CNN layer for latent vector input
        self.latent_input.add_module(
            f"initial-{latent_vec_size}-{n_input_features}-convt",
            nn.ConvTranspose1d(
                latent_vec_size,
                n_input_features,
                kernel_size=4,
                stride=2,
                padding=0,
                bias=False,
            ),
        )
        self.latent_input.add_module(f"initial-{n_input_features}-batchnorm", nn.BatchNorm1d(n_input_features))
        self.latent_input.add_module(f"initial-{n_input_features}-relu", nn.ReLU(True))

        # Create inverse pyramid
        self.inverse_pyramid = nn.Sequential()
        pyramid_dim = input_size // 2  # Use the smaller dimension to create pyramid.
        while pyramid_dim > 50:
            in_features = n_input_features
            out_features = n_input_features // 2
            self.inverse_pyramid.add_module(
                f"pyramid-{in_features}-{out_features}-convt",
                nn.ConvTranspose1d(
                    in_features,
                    out_features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
            )
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-batchnorm", nn.BatchNorm1d(out_features))
            self.inverse_pyramid.add_module(f"pyramid-{out_features}-relu", nn.ReLU(True))
            n_input_features = out_features
            pyramid_dim = pyramid_dim // 2

        #extra_layer
        self.extra_layers = nn.Sequential()

        self.final_layers = nn.Sequential()
        self.final_layers.add_module(
            f"final-{n_input_features}-{num_input_channels}-conv",
            nn.ConvTranspose1d(
                n_input_features,
                num_input_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
            ),
        )
        self.final_layers.add_module(f"final-{num_input_channels}-tanh",nn.Tanh())

    def forward(self,input_tensor):
        output = self.latent_input(input_tensor)
        output = self.inverse_pyramid(output)
        output = self.extra_layers(output)
        output = self.final_layers(output)
        return output


class DiscriminatorClassifier(nn.Module):
    def __init__(self, latent_vec_size: int = 64) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(latent_vec_size, latent_vec_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_vec_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(latent_vec_size, latent_vec_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_vec_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_vec_size, latent_vec_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_vec_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv1d(latent_vec_size, latent_vec_size, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(latent_vec_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(latent_vec_size, latent_vec_size, kernel_size=3, stride=2, padding=1),
        )

    def forward(self, input_tensor):
        feature = self.layer1(input_tensor)
        return feature

class Discriminator(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channels,
        n_features,
    ) -> None:
        super().__init__()
        encoder = Encoder(input_size=input_size, latent_vec_size=1, num_input_channel=num_input_channels, n_feature=n_features)
        layers = []

        for block in encoder.children():
            if isinstance(block,nn.Sequential):
                layers.extend(list(block.children()))
            else:
                layers.append(block)
        
        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module("Sigmoid", nn.Sigmoid())

    def forward(self, input_tensor):
        features = self.features(input_tensor)
        classifier = self.classifier(features)
        # torch.Size([64, 1, 24])
        classifier = classifier.view(input_tensor.shape[0], -1).squeeze(1)
        classifier = torch.mean(classifier, dim=1)
        return classifier, features


class Generator(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channel,
        n_features,
        latent_vec_size,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()
        self.encoder1 = Encoder(input_size=input_size,
                              num_input_channel=num_input_channel,
                              n_feature=n_features,
                              add_final_conv_layer=add_final_conv_layer,
                              latent_vec_size=latent_vec_size)
        self.decoder = Decoder(input_size=input_size,
                               n_features=n_features,
                               latent_vec_size=latent_vec_size,
                               num_input_channels=num_input_channel)
        self.encoder2 = Encoder(input_size=input_size,
                                num_input_channel=num_input_channel,
                                n_feature=n_features,
                                add_final_conv_layer=add_final_conv_layer,
                                latent_vec_size=latent_vec_size)

    def forward(self,input_tensor):
        # [64, 1, 1600]
        latent_i = self.encoder1(input_tensor)
        gen_image = self.decoder(latent_i)
        latent_o = self.encoder2(gen_image)
        return latent_i, gen_image, latent_o


class Ganomaly1d(nn.Module):
    def __init__(
        self,
        input_size,
        num_input_channel,
        n_feature,
        latent_vec_size,
        add_final_conv_layer: bool = True,
    ) -> None:
        super().__init__()

        self.generator:Generator = Generator(
            input_size=input_size,
            num_input_channel=num_input_channel,
            n_features=n_feature,
            latent_vec_size=latent_vec_size,
            add_final_conv_layer=add_final_conv_layer,
        )

        self.discriminator:Discriminator = Discriminator(
            input_size=input_size,
            num_input_channels=num_input_channel,
            n_features=n_feature,
        )

    def forward(self,batch):
        batch = batch.unsqueeze(1)
        latent_i, gen_image, latent_o = self.generator(batch)
        # if self.training:
        #     return latent_i, latent_o, batch, gen_image
        # return torch.mean(torch.pow((latent_i - latent_o), 2), dim=(1, 2)).view(-1)
        return latent_i, latent_o, batch, gen_image
        