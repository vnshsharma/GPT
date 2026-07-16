words = open('../data/names.txt','r').read().splitlines()
words[:10]

len(words) # These are the number of words in names.txt

b = {}
for w in words:
    chs = ['<S>'] + list(w) + ['<E>']
    for ch1, ch2 in zip(chs,chs[1:]):
        bigram = (ch1,ch2)
        b[bigram] = b.get(bigram,0)+1

# print(sorted(b.items(), key= lambda kv: -kv[1]))

import torch
N = torch.zeros((28,28),dtype=torch.int32)
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

        b[bigram] = b.get(bigram,0)+1

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
# print(p)   # This is the probability of the sampling data

g = torch.Generator().manual_seed(2147483647)
p = torch.rand(3,generator=g)
p = p/p.sum()
print(p)