# First we read that input.txt file 
with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
print('length of dataset in characters: ',len(text))

print(text[0:1000])

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

# Let's make from string to integer 
stio = {ch:i for i,ch in enumerate(chars)}
# Now assign chars to an integer
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s : [stio[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

print(encode('hii there'))
print(decode(encode('hii there')))

import torch 
data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
print(data[:1000])
