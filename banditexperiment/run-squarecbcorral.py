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

class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        import numpy
        
        self.data = numpy.load('652_contest.npy')
        self.ys = torch.Tensor(self.data)
        
    def __len__(self):
        return 10**5

    def __getitem__(self, index):
        # Select sample
        generator = torch.Generator().manual_seed(index)
        realized = torch.bernoulli(input=self.ys, generator=generator)
        return realized

class CountTable(object):
    def __init__(self, naction):
        self.table = [ [0, 0, 0.0] for _ in range(naction) ]
        
    def fhat(self):
        return torch.Tensor([ x[2] for x in self.table ])
        
    def update(self, action, reward):
        for a, r in map(lambda x: (x[0].item(), x[1].item()), zip(action, reward)):
            self.table[a][1] += 1
            self.table[a][0] += r
            self.table[a][2] = self.table[a][0] / self.table[a][1]

class CorralIGW(object):
    def __init__(self, *, eta, gammamin, gammamax, nalgos, device):
        import numpy
        
        super(CorralIGW, self).__init__()
        
        self.eta = eta / nalgos
        self.gammas = torch.Tensor(numpy.geomspace(gammamin, gammamax, nalgos), device=device)
        self.invpalgo = torch.Tensor([ self.gammas.shape[0] ] * self.gammas.shape[0], device=device)
        
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
 
    def sample(self, fhat):
        N, K = fhat.shape

        algosampler = torch.distributions.categorical.Categorical(probs=1.0/self.invpalgo, validate_args=False)
        algo = algosampler.sample((N,))
        invpalgo = torch.gather(input=self.invpalgo.unsqueeze(0).expand(N, -1),
                                dim=1,
                                index=algo.unsqueeze(1))
        gamma = torch.gather(input=self.gammas.unsqueeze(0).expand(N, -1),
                             dim=1,
                             index=algo.unsqueeze(1))
        
        fhatstar, ahatstar = torch.max(fhat, dim=1, keepdim=True)
        rando = torch.randint(high=K, size=(N, 1), device=fhat.device)
        fhatrando = torch.gather(input=fhat, dim=1, index=rando)
        probs = K / (K + gamma * (fhatstar - fhatrando))
        unif = torch.rand(size=(N, 1), device=fhat.device)
        shouldexplore = (unif <= probs).long()
        return (ahatstar + shouldexplore * (rando - ahatstar)).squeeze(1), algo, invpalgo        

def learnOnline(dataset, *, seed, eta, gammamin, gammamax, nalgos, batch_size):
    import time
    
    torch.manual_seed(seed)
    generator = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    log_loss = torch.nn.BCELoss()
    model = None
        
    print('{:<5s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}\t{:<10s}'.format(
            'n', 'loss', 'since last', 'acc', 'since last', 'reward', 'since last', 'dt (sec)'),
          flush=True)
    avloss, sincelast, acc, accsincelast, avreward, rewardsincelast = [ EasyAcc() for _ in range(6) ]
    
    for bno, ys in enumerate(generator):
        if model is None:
            import numpy as np
            model = CountTable(naction=ys.shape[1])
            sampler = CorralIGW(eta=eta, gammamin=gammamin, gammamax=gammamax, nalgos=nalgos, device=ys.device)
            start = time.time()
            
        with torch.no_grad():
            fhat = model.fhat().unsqueeze(0).expand(ys.shape[0], -1)
            sample, algo, invpalgo = sampler.sample(fhat)
            reward = torch.gather(input=ys, dim=1, index=sample.unsqueeze(1)).float()
            
        with torch.no_grad():
            samplefhat = torch.gather(input=fhat, index=sample.unsqueeze(1), dim=1)
            loss = log_loss(samplefhat, reward)
            model.update(sample, reward.squeeze(1))
        
        with torch.no_grad():
            pred = torch.argmax(fhat, dim=1)
            ypred = torch.gather(input=ys, dim=1, index=pred.unsqueeze(1))
            acc += torch.mean(ypred).float()
            accsincelast += torch.mean(ypred).float()
            avloss += loss
            sincelast += loss
            avreward += torch.mean(reward)
            rewardsincelast += torch.mean(reward)
            sampler.update(algo, invpalgo, reward)
        
        if bno % 1000 == 0:
            now = time.time()
            print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
                    avloss.n, avloss.mean(), sincelast.mean(), acc.mean(),
                    accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
                    now - start),
                  flush=True)
            sincelast, accsincelast, rewardsincelast = [ EasyAcc() for _ in range(3) ]
            #print(f'sampler.palgo = { 1/sampler.invpalgo }')

    now = time.time()
    print('{:<5d}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}\t{:<10.5f}'.format(
            avloss.n, avloss.mean(), sincelast.mean(), acc.mean(),
            accsincelast.mean(), avreward.mean(), rewardsincelast.mean(),
            now - start),
          flush=True)
    #print(f'sampler.palgo = { 1/sampler.invpalgo }')

mydata = MyDataset()

for myseed in range(45, 45+32):
    print('')
    print('=========================================================================================')
    print(f'learnOnline(mydata, seed={myseed}, batch_size=1, eta=1, gammamin=1000, gammamax=1000000, nalgos=8)')
    learnOnline(mydata, seed=myseed, batch_size=1, eta=1, gammamin=1000, gammamax=1000000, nalgos=8)
