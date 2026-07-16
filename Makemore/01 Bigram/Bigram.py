words = open('../data/names.txt','r').read().splitlines()
words[:10]

len(words) # These are the number of words in names.txt

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0)+1

sorted(b.items(), key= lambda kv: -kv[1])

import torch

N = torch.zeros((27,27),dtype=torch.int32)

chars = (sorted(list(set(''.join(words)))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1


import matplotlib.pyplot as plt
plt.figure(figsize=(18,18))
plt.imshow(N, cmap="Blues")

for i in range(27):
    for j in range(27):
        chstr = itos[i] + itos[j]
        plt.text(j, i, chstr,
                 ha="center", va="bottom",
                 fontsize=6, color="gray")
        plt.text(j, i, N[i, j].item(),
                 ha="center", va="top",
                 fontsize=6, color="gray")

plt.axis("off")

p = N[0].float()
p = p / p.sum()  


g = torch.Generator().manual_seed(2147483647)
ix = torch.multinomial(p,num_samples=1,replacement=True,generator=g).item()

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3,generator=g)
p = p/p.sum()

# print(torch.multinomial(p,num_samples=100,replacement=True,generator=g))

# print(p.shape)

P = (N+1).float()
# P = P/P.sum()
# print((P.sum(1,keepdim=True)))
P = P/P.sum(1,keepdim=True)


g = torch.Generator().manual_seed(2147483647)
for i in range(5):
    out = []
    ix = 0 
    while True:

        p = P[ix]

        ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(itos[ix])
        if ix == 0:
            break
    print(''.join(out))

print('\n')
# Now we find the efficiency of this model

# Loss Function 
log_likelihood = 0.0
n = 0

# GOAL: maximize likelihood of the data w.r.t. model parameters (statistical modeling)
# equivalent to maximizing the log likelihood (because log is monotonic)
# equivalent to minimizing the negative log likelihood
# equivalent to minimizing the average negative log likelihood

for w in words:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        N[ix1,ix2] += 1
        prob = P[ix1, ix2]
        logprob = torch.log(prob)
        log_likelihood += logprob
        n += 1
        # print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')

print(f'{log_likelihood=}')
nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n=}')   # This will be usually the loss function => That is the quality of this model, 
print('\n')

# create the training set of bigrams (x,y)
xs, ys = [], []

for w in words[:1]:
    chs = ['.'] + list(w) + ['.']
    for ch1, ch2 in zip(chs,chs[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        print(ch1,ch2)
        xs.append(ix1)
        ys.append(ix2)

xs = torch.tensor(xs)
ys = torch.tensor(ys)
print(xs)
print(ys)


# Feeding integer into the neural nets/ one hot encoding 
import torch.nn.functional as F
xenc = F.one_hot(xs,num_classes=27).float()
print(xenc)
plt.imshow(xenc)
# plt.show()


## One layer of neurons implemented with matrix multiplication
W = torch.randn((27,27))

logits = xenc @ W   # log counts 
counts = logits.exp() # equivalent N
probs = counts / counts.sum(1,keepdim=True)
print(probs)

# (5,27) @ (27,27) -> (5,27)