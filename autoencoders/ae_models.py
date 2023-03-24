import torch

class DeepAutoEncoder(torch.nn.Module):
    """Class for classic auto-encoders"""
    def __init__(self, encoder_dims, decoder_dims, dropout):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims:    list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        """
        super(DeepAutoEncoder, self).__init__()
        layers = []
        for i in range(len(encoder_dims) - 2):
            layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(encoder_dims[-2], encoder_dims[-1]))
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

    def xi_ae(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return self.encoder(x).detach().numpy()

    def grad_xi_ae(self, x):
        """Gradient of the collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 1]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x.requires_grad_()
        enc = self.encoder(x)
        return ((torch.autograd.grad(outputs=enc.sum(), inputs=x)[0][:, :2]) ** 2).mean()

class DeepAutoEncoderDoubleDec(torch.nn.Module):
    """Class for auto-encoders with two decoders. This class does not contain a forward function."""
    def __init__(self, encoder_dims, decoder_dims, dropout):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims:    list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        """
        super(DeepAutoEncoderDoubleDec, self).__init__()
        layers = []
        for i in range(len(encoder_dims) - 2):
            layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(encoder_dims[-2], encoder_dims[-1]))

        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))

        self.decoder1 = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))

        self.decoder2 = torch.nn.Sequential(*layers)

    def decoded1(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder1(enc)
        return dec

    def decoded2(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder2(enc)
        return dec

    def xi_ae(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return self.encoder(x).detach().numpy()

    def grad_xi_ae(self, x):
        """Gradient of the collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 1]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x.requires_grad_()
        enc = self.encoder(x)
        return ((torch.autograd.grad(outputs=enc.sum(), inputs=x)[0][:, :2]) ** 2).mean()

class EffectiveDynamicsAutoEncoder(torch.nn.Module):
    """Class for auto-encoders with effective dynamics. """
    def __init__(self, encoder_dims, decoder_dims, bias_dims, diff_dims, dropout):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param encoder_dims:    list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param bias_dims:       list, List of dimensions for the decoder encoding the bias of the effective dynamics,
                                including input/output layers
        :param diff_dims:       list, List of dimensions for the decoder encoding the bias of the effective dynamics,
                                including input/output layers
        :param dropout:         int, value of the dropout probability
        """
        super(EffectiveDynamicsAutoEncoder, self).__init__()
        layers = []
        for i in range(len(encoder_dims) - 2):
            layers.append(torch.nn.Linear(encoder_dims[i], encoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(encoder_dims[-2], encoder_dims[-1]))
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(bias_dims) - 2):
            layers.append(torch.nn.Linear(bias_dims[i], bias_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(bias_dims[-2], bias_dims[-1]))
        self.ed_b = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(diff_dims) - 2):
            layers.append(torch.nn.Linear(diff_dims[i], diff_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(diff_dims[-2], diff_dims[-1]))
        self.ed_d = torch.nn.Sequential(*layers)

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded

    def ed_bias(self, enc):
        ed_bias = self.ed_b(enc)
        ed_bias = torch.reshape(ed_bias, (ed_bias.shape + (1,)))
        return ed_bias

    def ed_diff(self, enc):
        ed_diff = self.ed_d(enc)
        ed_diff = torch.reshape(ed_diff, ed_diff.shape + (ed_diff.shape[-1],))
        return ed_diff

    def xi_ae(self,  x):
        """Collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 2]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        return self.encoder(x).detach().numpy()

    def grad_xi_ae(self, x):
        """Gradient of the collective variable defined through an auto encoder model

        :param x: np.array, position, ndim = 2, shape = [any, 1]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(x):
            x = torch.from_numpy(x).float()
        x.requires_grad_()
        enc = self.encoder(x)
        return ((torch.autograd.grad(outputs=enc.sum(), inputs=x)[0][:, :2]) ** 2).mean()

