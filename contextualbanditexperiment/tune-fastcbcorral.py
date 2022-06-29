import torch

class EasyAcc:
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sumsq = 0

    def __iadd__(self, other):
        self.n += 1
        self.sum += other
        self.sumsq += other*other
        return self

    def __isub__(self, other):
        self.n += 1
        self.sum -= other
        self.sumsq += other*other
        return self

    def mean(self):
        return self.sum / max(self.n, 1)

    def var(self):
        from math import sqrt
        return sqrt(self.sumsq / max(self.n, 1) - self.mean()**2)

    def semean(self):
        from math import sqrt
        return self.var() / sqrt(max(self.n, 1))

def isfloat(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

def floatorzero(element):
    try:
        return float(element)
    except ValueError:
        return 0.0

# match vw's flexible parsing
def makeCategoricalData(filename):
    from collections import defaultdict
    import numpy
    
    print('using categorical parsing')
        
    isNumeric = None
    extras = defaultdict(set)

    with open(filename, 'r') as f:
        for line in f:
            targetstr, rest = line.strip().split('|')
            target = float(targetstr)
            stringfeatures = rest.split()
            for col, (isnum, v) in enumerate((isfloat(x), x) for x in stringfeatures):
                if not isnum:
                    extras[col].add(v)
                    
    onehotmap = {}
    for col, values in extras.items():
        for v in values:
            if (col, v) not in onehotmap:
                onehotmap[col, v] = len(onehotmap)
             
    print(f'creating {len(onehotmap)} additional one-hot columns')        
    
    Y = []
    X = []
    with open(filename, 'r') as f:
        for line in f:
            targetstr, rest = line.strip().split('|')
            target = float(targetstr)
            stringfeatures = rest.split()
            features = [0]*len(onehotmap) + [ floatorzero(x) for x in stringfeatures ]
            for col, v in enumerate(stringfeatures):
                if (col, v) in onehotmap:
                    features[onehotmap[col, v]] = 1
            
            Y.append(target)
            X.append(features)
 
    Y = numpy.array(Y)
    Ymin, Ymax = numpy.min(Y), numpy.max(Y)
    Y = (Y - Ymin) / (Ymax - Ymin)
    X = numpy.array(X)
    Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(X, axis=0, keepdims=True)
    X = (X - Xmin) / numpy.maximum(Xmax - Xmin, 1e-9)
    
    return X, Y

def makeData(filename):
    import numpy
    
    Y = []
    X = []
    with open(filename, 'r') as f:
        for line in f:
            targetstr, rest = line.strip().split('|')
            target = float(targetstr)
            try:
                features = [ float(x) for x in rest.split() ]
            except ValueError:
                return makeCategoricalData(filename)
            
            Y.append(target)
            X.append(features)
 
    Y = numpy.array(Y)
    Ymin, Ymax = numpy.min(Y), numpy.max(Y)
    Y = (Y - Ymin) / (Ymax - Ymin)
    X = numpy.array(X)
    Xmin, Xmax = numpy.min(X, axis=0, keepdims=True), numpy.max(X, axis=0, keepdims=True)
    X = (X - Xmin) / (Xmax - Xmin)
    
    return X, Y

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filename):
        X, Y = makeData(filename)
        self.Xs = torch.Tensor(X)
        self.Ys = torch.Tensor(Y).unsqueeze(1)
            
    def __len__(self):
        return self.Xs.shape[0]

    def __getitem__(self, index):
        # Select sample
        return self.Xs[index], self.Ys[index]

class ArgmaxPlusDispersion(torch.nn.Module):
    def __init__(self, argmaxblock):
        super(ArgmaxPlusDispersion, self).__init__()
        
        self.argmaxblock = argmaxblock
        self.logitsigma = torch.nn.Parameter(torch.ones(1))
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.sigmoid = torch.nn.Sigmoid()
        
    def argmax(self, Xs):
        return 1 - self.sigmoid(self.logitsigma).unsqueeze(0).expand(Xs.shape[0], -1), self.argmaxblock(Xs)

    def forward(self, Xs, As):
        _, Yhat = self.argmax(Xs)
        return 1 - self.sigmoid(torch.abs(self.scale * (Yhat - As)) + self.logitsigma)
    
class LinearArgmax(torch.nn.Module):
    def __init__(self, dobs):
        super(LinearArgmax, self).__init__()
        
        self.linear = torch.nn.Linear(in_features=dobs, out_features=1)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, Xs):
        return self.sigmoid(self.linear(Xs))
    
class CauchyRFF(torch.nn.Module):
    def __init__(self, dobs, numrff, sigma, device):
        from math import pi, sqrt
        
        super(CauchyRFF, self).__init__()
        
        self.rffW = torch.nn.Parameter(torch.empty(dobs, numrff).cauchy_(sigma = sigma).to(device), 
                                       requires_grad=False)
        self.rffb = torch.nn.Parameter((2 * pi * torch.rand(numrff)).to(device),
                                       requires_grad=False)
        self.sqrtrff = torch.nn.Parameter(torch.Tensor([sqrt(numrff)]).to(device), 
                                          requires_grad=False)
        self.linear = torch.nn.Linear(in_features=numrff, out_features=1, device=device)
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, Xs):
        with torch.no_grad():
            rff = (torch.matmul(Xs, self.rffW) + self.rffb).cos() / self.sqrtrff
            
        return self.sigmoid(self.linear(rff))
    
class CorralFastIGW(object):
    def __init__(self, *, eta, gammamin, gammamax, nalgos, device):
        import numpy
        
        super(CorralFastIGW, self).__init__()
        
        self.eta = eta / nalgos
        self.gammas = torch.Tensor(numpy.geomspace(gammamin, gammamax, nalgos)).to(device)
        self.invpalgo = torch.Tensor([ self.gammas.shape[0] ] * self.gammas.shape[0]).to(device)
        
    def update(self, algo, invprop, reward):
        import numpy
        from scipy import optimize
        
        assert torch.all(reward >= 0) and torch.all(reward <= 1), reward
        
        weightedlosses = self.eta * (-reward.squeeze(1)) * invprop.squeeze(1)
        newinvpalgo = torch.scatter(input=self.invpalgo,
                                    dim=0,
                                    index=algo,
                                    src=weightedlosses,
                                    reduce='add')
                                    
        # just do this calc on the cpu
        invp = newinvpalgo.cpu().numpy() 
        invp += 1 - numpy.min(invp)
        Zlb = 0
        Zub = 1
        while (numpy.sum(1 / (invp + Zub)) > 1):
            Zlb = Zub
            Zub *= 2 
        root, res = optimize.brentq(lambda z: 1 - numpy.sum(1 / (invp + z)), Zlb, Zub, full_output=True)
        assert res.converged, res
        
        self.invpalgo = torch.Tensor(invp + root, device=self.invpalgo.device)
 
    def sample(self, fhatstar, ahatstar, fhat, X):
        N, _ = fhatstar.shape

        algosampler = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False)
        algo = algosampler.sample((N,))
        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1),
                                dim=1,
                                index=algo.unsqueeze(1))
        gamma = torch.gather(input=self.gammas.unsqueeze(0).expand(N, -1),
                             dim=1,
                             index=algo.unsqueeze(1))
        
        rando = torch.rand(size=(N, 1), device=X.device)
        fhatrando = fhat(X, rando)
        probs = 1 / (1 + gamma * (1 - fhatrando / fhatstar))
        unif = torch.rand(size=(N, 1), device=X.device)
        shouldexplore = (unif <= probs).long()
        return (ahatstar + shouldexplore * (rando - ahatstar)), algo, invpalgo        
    
def learnOnline(dataset, *, seed, batch_size, modelfactory, initlr, tzero, eta, gammamin, gammamax, nalgos):
    import time
    
    torch.manual_seed(seed)
        
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = None
    l1_loss = torch.nn.L1Loss(reduction='none')
    log_loss = torch.nn.BCELoss()
    
    print('{:<5s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}'.format(
            'n', 'loss', 'since last', 'acc', 'since last', 'reward', 'since last', 'dt (sec)'),
          flush=True)
    avloss, sincelast, acc, accsincelast, avreward, rewardsincelast = [ EasyAcc() for _ in range(6) ]
    
    for bno, (Xs, ys) in enumerate(generator):
        if model is None:
            from math import sqrt
            model = modelfactory(Xs)
            opt = torch.optim.Adam(( p for p in model.parameters() if p.requires_grad ), lr=initlr)
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda = lambda t: sqrt(tzero) / sqrt(tzero + t))
            sampler = CorralFastIGW(eta=eta, gammamin=gammamin, gammamax=gammamax, nalgos=nalgos, device=Xs.device)
            start = time.time()
            
        opt.zero_grad()
        
        with torch.no_grad():
            fhatstar, ahatstar = model.argmax(Xs)
            sample, algo, invpalgo = sampler.sample(fhatstar, ahatstar, model, Xs)
            reward = 1 - l1_loss(sample, ys)
        
        score = model(Xs, sample)
        loss = log_loss(score, reward)
        loss.backward()
        opt.step()
        scheduler.step()
        
        with torch.no_grad():
            acc += 1 - torch.mean(l1_loss(ahatstar, ys)).item()
            accsincelast += 1 - torch.mean(l1_loss(ahatstar, ys)).item()
            avreward += torch.mean(reward).item()
            rewardsincelast += torch.mean(reward).item()
            avloss += loss
            sincelast += loss
            sampler.update(algo, invpalgo, reward)

        if bno & (bno - 1) == 0:
            now = time.time()
            print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
                    avloss.n, avloss.mean(), sincelast.mean(), acc.mean(),
                    accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
                    now - start),
                  flush=True)
            sincelast, accsincelast, rewardsincelast = [ EasyAcc() for _ in range(3) ]
            print(f'sampler.palgo = { 1/sampler.invpalgo }')

    now = time.time()
    print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
            avloss.n, avloss.mean(), sincelast.mean(), acc.mean(),
            accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
            now - start),
          flush=True)
    print(f'sampler.palgo = { 1/sampler.invpalgo }')

import sys
mydata = MyDataset(sys.argv[1])

def flass():
    import random
    for initlr, tzero, eta, gammamin, gammamax, nalgos in (
       (
	   10**(-2 + random.random()),
	   10 + 100 * random.random(),
	   10**(-1 + random.random()),
	   (1 << random.randint(0, 5)),
	   (1 << random.randint(6, 12)),
	   random.randint(8, 16),
       )
       for _ in range(59)
    ):
        print('hypers ', initlr, tzero, eta, gammamin, gammamax, nalgos)
        learnOnline(mydata, seed=4545, batch_size=64 if sys.argv[1] == 'zurich.dat' else 8,
                    initlr=initlr, tzero=tzero, eta=eta, gammamin=gammamin, gammamax=gammamax, nalgos=nalgos,
                    modelfactory=lambda x: ArgmaxPlusDispersion(argmaxblock=LinearArgmax(dobs=x.shape[1])))

flass()
