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

# Split up the data into train and validation sets 
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# How GPT creates 
# it is only for understand and is not used 
# in actual batch generation or model training 

# block_size = 8
# sample_data = data[:block_size+1]
# print(train_data)

# # This block size is only used for predict the next target
# x = sample_data[:block_size]
# y = sample_data[1:block_size+1]
# for t in range(block_size):
#     context = x[:t+1]
#     target = y[t]
#     print(f'when input is {context} the target: {target}')

torch.manual_seed(1337)
batch_size = 4 # how many independent sequence will we process in parallel
block_size = 8 # what is the maximum context length for prediction

def get_batch(split):
    if split == 'train':
        data = train_data
    else:
        data = val_data
    ix = torch.randint(len(data)-block_size,(block_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x,y

xb,yb = get_batch('train')
print('Inputs:')
print(xb.shape)
print(xb)
print('targets:')
print(yb.shape)
print(yb)


for b in range(batch_size):
    for t in range(block_size):
        context = xb[b, :t+1]
        target = yb[b,t]
        print(f'when input is {context.tolist()} the target: {target}')
