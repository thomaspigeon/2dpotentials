import torch


class CommittorOneDecoder(torch.nn.Module):
    def __init__(self, committor_dims, decoder_dims, dropout, pot, boundary_width=0.1, handtanh=True):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param committor_dims:  list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        :param pot:             General2DPotential, object containing information concerning the potential et the
                                definition of the reactant and product state
        :param boundary_width:  float, witdth of the boundary of the definition of states
        """
        super(CommittorOneDecoder, self).__init__()
        self.boundary_width = boundary_width
        layers = []
        for i in range(len(committor_dims) - 2):
            layers.append(torch.nn.Linear(committor_dims[i], committor_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(committor_dims[-2], committor_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

        layers = []
        for i in range(len(decoder_dims) - 2):
            layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = torch.nn.Sequential(*layers)
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.minR = torch.tensor(pot.minR.astype('float32')).to(device)
        self.R_radius = pot.R_radius
        self.minP = torch.tensor(pot.minP.astype('float32')).to(device)
        self.P_radius = pot.P_radius
        if handtanh:
            self.HT = torch.nn.Hardtanh()
        else:
            self.HT = torch.nn.Tanh()

    def inR(self, inp):
        shape = list(inp.shape)
        shape[-1] = 1
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (2 * (1 / self.boundary_width) * (torch.sqrt(torch.sum((inp - self.minR) ** 2, dim=-1)).reshape(shape) - self.R_radius) / self.R_radius) + 1))

    def inP(self, inp):
        shape = list(inp.shape)
        shape[-1] = 1
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (2 * (1 / self.boundary_width) * (torch.sqrt(torch.sum((inp - self.minP) ** 2, dim=-1)).reshape(shape) - self.P_radius) / self.P_radius) + 1))

    def decoded(self, inp):
        enc = self.encoder(inp)
        dec = self.decoder(enc)
        return dec

    def committor(self, inp):
        shape = inp.shape
        if shape[-1] != 2:
            if len(shape) == 2:
                committor = (1 - self.inR(inp[:, :2])) * \
                                ((1 - self.inP(inp[:, :2])) * self.encoder(inp) + self.inP(inp[:, :2]))
            elif len(shape) == 3:
                committor = (1 - self.inR(inp[:, :, :2])) * \
                                ((1 - self.inP(inp[:, :, :2])) * self.encoder(inp) + self.inP(inp[:, :, :2]))
            elif len(shape) == 4:
                committor = (1 - self.inR(inp[:, :, :, :2])) * \
                                ((1 - self.inP(inp[:, :, :, :2])) * self.encoder(inp) + self.inP(inp[:, :, :, :2]))
        else:
            committor = (1 - self.inR(inp)) * ((1 - self.inP(inp)) * self.encoder(inp) + self.inP(inp))
        return committor

    def xi_forward(self,  X):
        """Collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if not torch.is_tensor(X):
            X = torch.tensor(X.astype('float32'), device=device)
        return self.committor(X).cpu().detach().numpy()

    def xi_backward(self,  X):
        """Collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if not torch.is_tensor(X):
            X = torch.tensor(X.astype('float32'), device=device)
        return 1 - self.committor(X.to(device)).cpu().detach().numpy()

    def grad_xi(self, X):
        """Gradient of the collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        X.requires_grad_()
        enc = self.encoder(X)
        if X.shape[1] == 2:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :2]
        if X.shape[1] == 4:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :4]


class CommittorMultiDecoder(torch.nn.Module):
    def __init__(self, committor_dims, decoder_dims, pot, number_decoders=2, dropout=0., boundary_width=0.0001, handtanh=True):
        """Initialise auto encoder with hyperbolic tangent activation function

        :param committor_dims:  list, List of dimensions for encoder, including input/output layers
        :param decoder_dims:    list, List of dimensions for decoder, including input/output layers
        :param dropout:         int, value of the dropout probability
        :param pot:             General2DPotential, object containing information concerning the potential et the
                                definition of the reactant and product state
        :param boundary_width:  float, witdth of the boundary of the definition of states
        """
        super(CommittorMultiDecoder, self).__init__()
        self.boundary_width = boundary_width
        layers = []
        for i in range(len(committor_dims) - 2):
            layers.append(torch.nn.Linear(committor_dims[i], committor_dims[i + 1]))
            layers.append(torch.nn.Dropout(dropout))
            layers.append(torch.nn.Tanh())
        layers.append(torch.nn.Linear(committor_dims[-2], committor_dims[-1]))
        layers.append(torch.nn.Sigmoid())
        self.encoder = torch.nn.Sequential(*layers)

        self.decoders = []
        for i in range(number_decoders):
            layers = []
            for i in range(len(decoder_dims) - 2):
                layers.append(torch.nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
                layers.append(torch.nn.Dropout(dropout))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(decoder_dims[-2], decoder_dims[-1]))
            self.decoders.append(torch.nn.Sequential(*layers))

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        self.minR = torch.tensor(pot.minR.astype('float32')).to(device)
        self.R_radius = pot.R_radius
        self.minP = torch.tensor(pot.minP.astype('float32')).to(device)
        self.P_radius = pot.P_radius
        if handtanh:
            self.HT = torch.nn.Hardtanh()
        else:
            self.HT = torch.nn.Tanh()

    def inR(self, inp):
        shape = list(inp.shape)
        shape[-1] = 1
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (2 * (1 / self.boundary_width) * (torch.sqrt(torch.sum((inp - self.minR) ** 2, dim=-1)).reshape(
                shape) - self.R_radius) / self.R_radius) + 1))

    def inP(self, inp):
        shape = list(inp.shape)
        shape[-1] = 1
        return 1 - (1 / 2 + (1 / 2) * self.HT(
            (2 * (1 / self.boundary_width) * (torch.sqrt(torch.sum((inp - self.minP) ** 2, dim=-1)).reshape(
                shape) - self.P_radius) / self.P_radius) + 1))

    def decoded(self, inp):
        enc = self.encoder(inp)
        decs = []
        for i in range(len(self.decoders)):
            decs.append(self.decoders[i](enc))
        return decs

    def committor(self, inp):
        shape = inp.shape
        if shape[-1] != 2:
            if len(shape) == 2:
                committor = (1 - self.inR(inp[:, :2])) * \
                            ((1 - self.inP(inp[:, :2])) * self.encoder(inp) + self.inP(inp[:, :2]))
            elif len(shape) == 3:
                committor = (1 - self.inR(inp[:, :, :2])) * \
                            ((1 - self.inP(inp[:, :, :2])) * self.encoder(inp) + self.inP(inp[:, :, :2]))
            elif len(shape) == 4:
                committor = (1 - self.inR(inp[:, :, :, :2])) * \
                            ((1 - self.inP(inp[:, :, :, :2])) * self.encoder(inp) + self.inP(inp[:, :, :, :2]))
        else:
            committor = (1 - self.inR(inp)) * ((1 - self.inP(inp)) * self.encoder(inp) + self.inP(inp))
        return committor

    def xi_forward(self, X):
        """Collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if not torch.is_tensor(X):
            X = torch.tensor(X.astype('float32'), device=device)
        return self.committor(X).cpu().detach().numpy()

    def xi_backward(self, X):
        """Collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 1]
        """
        self.eval()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if not torch.is_tensor(X):
            X = torch.tensor(X.astype('float32'), device=device)
        return 1 - self.committor(X).cpu().detach().numpy()

    def grad_xi(self, X):
        """Gradient of the collective variable defined through an auto encoder model

        :param X: np.array, position (or position and momenta), ndim = 2 (4), shape = [any, 2(4)]
        :return: xi: np.array, ndim = 2, shape = [any, 2]
        """
        self.eval()
        if not torch.is_tensor(X):
            X = torch.from_numpy(X).float()
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        X.to(device)
        X.requires_grad_()
        enc = self.encoder(X)
        if X.shape[1] == 2:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :2]
        if X.shape[1] == 4:
            return torch.autograd.grad(outputs=enc.sum(), inputs=X)[0][:, :4]


