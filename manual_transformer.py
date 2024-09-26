import random
import numpy as np
import torch
import math

vocab_x = '<SOS>,<EOS>,<PAD>,0,1,2,3,4,5,6,7,8,9,q,w,e,r,t,y,u,i,o,p,a,s,d,f,g,h,j,k,l,z,x,c,v,b,n,m'
vocab_x = {word: i for i, word in enumerate(vocab_x.split(','))}
vocab_xr = [k for k, _ in vocab_x.items()]
vocab_y = {k.upper(): v for k, v in vocab_x.items()}
vocab_yr = [k for k, _ in vocab_y.items()]


def get_data():
    words = ['0','1','2','3','4','5','6','7','8','9','q','w','e','r','t','y','u','i','o','p','a','s','d','f','g','h','j','k','l','z','x','c','v','b','n','m']
    p = np.array([i+1 for i in range(len(words))])
    p = p / p.sum()
    n = random.randint(30, 48)
    x = np.random.choice(words, size=n, replace=True, p=p)
    x = x.tolist()
    def f(i):
        i = i.upper()
        if not i.isdigit():
            return i
        i = 9 - int(i)
        return str(i)
    y = [f(i) for i in x]
    y = y[::-1]
    y = [y[0]] + y
    x = ['<SOS>'] + x + ['<EOS>']
    y = ['<SOS>'] + y + ['<EOS>']
    x = x + ['<PAD>'] * 50
    y = y + ['<PAD>'] * 51
    x = x[:50]
    y = y[:51]
    x = [vocab_x[i] for i in x]
    y = [vocab_y[i] for i in y]
    x = torch.LongTensor(x)
    y = torch.LongTensor(y)
    return x, y

class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        return get_data()

loader = torch.utils.data.DataLoader(dataset=Dataset(), 
                                     batch_size=8,
                                     drop_last=True,
                                     shuffle=True,
                                     collate_fn=None)

for x, y in loader:
    print(x.shape)
    print(y.shape)
    break

def mask_pad(data:torch.Tensor):
    # data:输入序列 (batch, length)->8x50
    mask = data == vocab_x['<PAD>']
    mask = mask.reshape(-1,1,1,50) # (batch_size x head_nums x 1 x embed_dims)
    mask = mask.expand(-1,1,50,50)
    return mask

def mask_tril(data:torch.Tensor):
    # data:目标序列
    tril = 1 - torch.tril(torch.ones(1, 50,50,dtype=torch.long))
    # print(torch.tril(torch.ones(1, 5,5,dtype=torch.long)))
    mask = data == vocab_y['<PAD>']
    mask = mask.unsqueeze(1).long()
    mask = mask + tril
    mask = mask > 0
    mask = (mask == 1).unsqueeze(dim=1)

    return mask

def attention(Q, K, V, mask):
    score = torch.matmul(Q, K.permute(0,1,3,2))
    score /= 8 ** 0.5
    score = torch.masked_fill(score, mask, -float('inf'))
    score = torch.softmax(score, dim=-1)
    score = torch.matmul(score, V)
    score = score.permute(0,2,1,3).reshape(-1, 50, 32)
    return score

# mask_tril(x[:1])

# mask_tril(x)

# score = torch.tensor([[[[0.1, 0.2, 0.3, 0.4],    # 头1，句子1
#                         [0.5, 0.6, 0.7, 0.8],    # 头1，句子2
#                         [0.9, 1.0, 1.1, 1.2]],   # 头1，句子3
#                        [[1.1, 1.2, 1.3, 1.4],    # 头2，句子1
#                         [1.5, 1.6, 1.7, 1.8],    # 头2，句子2
#                         [1.9, 2.0, 0.0, 0.0]]]]) # 头2，句子3，补两个0
# reshaped = score.reshape(-1, 3, 8)
# permuted = score.permute(0, 2, 1, 3).reshape(-1, 3, 8)
# print(reshaped)
# print('----------------------')
# print(permuted)

class MultiHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_Q = torch.nn.Linear(32, 32)
        self.fc_K = torch.nn.Linear(32, 32)
        self.fc_V = torch.nn.Linear(32, 32)
        self.out_fc = torch.nn.Linear(32, 32)
        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)
        self.dropout = torch.nn.Dropout(p=0.1)
    
    def forward(self, Q, K, V, mask):
        b = Q.shape[0]
        clone_Q = Q.clone()
        Q = self.norm(Q)
        K = self.norm(K)
        V = self.norm(V)
        Q = self.fc_Q(Q)
        K = self.fc_K(K)
        V = self.fc_V(V)
        Q = Q.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        K = K.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        V = V.reshape(b, 50, 4, 8).permute(0, 2, 1, 3)
        score = attention(Q, K, V, mask)
        score = self.dropout(self.out_fc(score))
        score = score + clone_Q
        return score

class PositionEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # pos:第几个词; i: 词向量的第几个维度; d_model:词向量总维度
        def get_pe(pos, i, d_model):
            d = 1e4**(i / d_model)
            pe = pos / d
            if i % 2 == 0:
                return math.sin(pe)
            return math.cos(pe)
        pe = torch.empty(50, 32)
        for i in range(50):
            for j in range(32):
                pe[i, j] = get_pe(i, j, 32)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.embed = torch.nn.Embedding(39, 32)
        self.embed.weight.data.normal_(0, 0.1)
    def forward(self, x):
        embed = self.embed(x)
        embed = embed + self.pe
        return embed

class FullyConnectedOutput(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=32, out_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=64, out_features=32),
            torch.nn.Dropout(p=0.1)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=32,
                                       elementwise_affine=True)
    
    def forward(self, x):
        clone_x = x.clone()
        x = self.norm(x)
        out = self.fc(x)
        out = out + clone_x
        return out

class EncoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh = MultiHead()
        self.fc = FullyConnectedOutput()
    def forward(self, x, mask):
        score = self.mh(x, x, x, mask)
        out = self.fc(score)
        return out

class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = EncoderLayer()
        self.layer2 = EncoderLayer()
        self.layer3 = EncoderLayer()
    def forward(self, x, mask):
        x = self.layer1(x, mask)
        x = self.layer2(x, mask)
        x = self.layer3(x, mask)
        return x


class DecoderLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mh1 = MultiHead()
        self.mh2 = MultiHead()
        self.fc = FullyConnectedOutput()
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.mh1(y, y, y, mask_tril_y)
        y = self.mh2(y, x, x, mask_pad_x)
        y = self.fc(y)
        return y

class Decoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = DecoderLayer()
        self.layer2 = DecoderLayer()
        self.layer3 = DecoderLayer()
    
    def forward(self, x, y, mask_pad_x, mask_tril_y):
        y = self.layer1(x, y, mask_pad_x, mask_tril_y)
        y = self.layer2(x, y, mask_pad_x, mask_tril_y)
        y = self.layer3(x, y, mask_pad_x, mask_tril_y)
        return y


class Transformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_x = PositionEmbedding()
        self.embed_y = PositionEmbedding()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.fc_out = torch.nn.Linear(32, 39)
    
    def forward(self, x, y):
        mask_pad_x = mask_pad(x)
        mask_tril_y = mask_tril(y)
        x, y = self.embed_x(x), self.embed_y(y)
        x = self.encoder(x, mask_pad_x)
        y = self.decoder(x, y, mask_pad_x, mask_tril_y)
        y = self.fc_out(y)
        return y

model = Transformer()

def predict(x):
    model.eval()
    mask_pad_x = mask_pad(x)
    target = [vocab_y['<SOS>']] + [vocab_y['<PAD>']] * (len(x) - 1)
    target = torch.LongTensor(target).unsqueeze(0)
    x = model.embed_x(x)
    x = model.encoder(x, mask_pad_x)
    for i in range(49):
        y = target
        mask_tril_y = mask_tril(y)
        y = model.embed_y(y)
        y = model.decoder(x, y, mask_pad_x, mask_tril_y)
        out = model.fc_out(y)
        out = out[:, i, :]
        out = torch.argmax(out, dim=1).detach()
        target[:, i + 1] = out
    return target


def train():
    loss_func = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    for epoch in range(3):
        for i, (x,y) in enumerate(loader):
            pred = model(x, y[:, :-1])
            pred = pred.reshape(-1, 39)
            y = y[:, 1:].reshape(-1)
            select = y != vocab_y['<PAD>']
            pred = pred[select]
            y = y[select]
            loss = loss_func(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            if i % 200 == 0:
                pred = pred.argmax(dim=1)
                correct = (pred == y).sum().item()
                accuracy = correct / len(pred)
                lr = optim.param_groups[0]['lr']
                print(epoch, i, loss.item(), lr, accuracy)
        sched.step()
train()