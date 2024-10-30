import math
import sys

sys.setrecursionlimit(1000000)
import numpy as np
import torch
from torch import nn, optim


def make_batch(sentences):
    input_batch = [[src_vocab[n] for n in sentences[0].split()]]
    output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
    target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
    return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)


def get_attn_pad_mask(seq_q, seq_k):
    """
    seq_q: [batch_size, seq_len]
    把pad的位置标记一下 0 就代表当前位置为空, 然后变成 batch_size x len_q x len_k 形状
    这里只要标记k就行了
    """
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k, one is masking
    return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k


def get_attn_subsequent_mask(seq):
    """
    seq: [batch_size, tgt_len]
    只关注 在当前单词之前出现的单词 其他都给标记一下, 后面都给置 +INF
    故: 会生成类似如下矩阵
    0 1 1 1
    0 0 1 1
    0 0 0 1
    0 0 0 0
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]


class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, d_model))
        position = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
        self.P[:, :, 0::2] = torch.sin(position / div_term)
        self.P[:, :, 1::2] = torch.cos(position / div_term)

    def forward(self, X):
        X = X + self.P[:, :X.shape[1], :].to(X.device)
        return self.dropout(X)


class PositionWiseFFN(nn.Module):
    """
    隐藏层大小为 d_ff 的全链接网络, 注意是带残缺层的
    """

    def __init__(self):
        super(PositionWiseFFN, self).__init__()
        self.dense1 = nn.Linear(d_model, d_ff)
        self.dense2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.layerNorm = nn.LayerNorm(d_model)

    def forward(self, X):
        """ X: [batch_size, seq_len, d_model] """
        return self.layerNorm(X + self.dense2(self.relu(self.dense1(X))))


class DotProductAttention(nn.Module):
    def __init__(self, dropout=0.1, **kwargs):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(d_k)
        # 然后关键词地方来了，下面这个就是用到了我们之前重点讲的attn_mask，把被mask的地方置为无限小，softmax之后基本就是0，对q的单词不起作用
        scores.masked_fill_(attn_mask, -1e9)  # Fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        return torch.matmul(self.dropout(attn), V)


class MultiHeadAttention(nn.Module):
    """
    实现多头注意力机制
    解释:
    由于 X的形状为 [batch_size, seq_len, d_model]
    目的为了简化运算, 若有n_heads个头, 则分成 n_heads个[batch_size, seq_len, d_model / n_heads] 个矩阵进行计算
    即变成了[batch_size, seq_len, n_heads, d_model / n_heads]
    为了一次就能够运算完自注意力分数, 就把 n_heads 与 batch_size 固定在前两维一起进行计算,
    就变成了[batch_size, n_heads, seq_len, d_model / n_heads], 这样计算一次即可

    简单讲: 就是把词向量的长度 d_models 拆成 n_heads 个小的进行计算
    不会破坏其的语义关系

    注意, 一般 seq_len =  src_vocab_size = len_q = len_k
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        # 初始三个待训练的矩阵, 这里的多头
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear = nn.Linear(n_heads * d_v, d_model)

    def forward(self, Q, K, V, attn_mask):
        """
        输入进来的数据形状：
        Q: [batch_size x len_q x d_model],
        K: [batch_size x len_k x d_model],
        V: [batch_size x len_k x d_model]
        """
        residual, batch_size = Q, Q.size(0)

        Q = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # Q: [batch_size x n_heads x len_q x d_k]
        K = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # K: [batch_size x n_heads x len_k x d_k]
        V = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # V: [batch_size x n_heads x len_k x d_v]

        """
        输入进行的attn_mask形状是 batch_size x len_q x len_k，
        然后经过下面这个代码得到 新的attn_mask : [batch_size x n_heads x len_q x len_k]，
        就是把pad信息重复了n个头上
        """
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)
        """
        得到的结果有两个：context: [batch_size x n_heads x len_q x d_v],
        attn: [batch_size x n_heads x len_q x len_k]
        """
        context = DotProductAttention()(Q, K, V, attn_mask)
        # context: [batch_size x len_q x n_heads * d_v]
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v)
        """
        这一步就是把 [batch_size, n_heads , len_q , len_v]
        变回 [batch_size, len_q, n_heads x len_v] 
        这样就与原先的输入的X的形状保持一样,
        """
        output = self.linear(context)  # Q.shape == output.shape
        return self.layer_norm(output + residual)  # output: [batch_size x len_q x d_model]


class EncoderLayer(nn.Module):
    """
    Encoder layer
    每个层包含两部分
    1. 多头注意力机制
    2. 前馈神经网络
    """

    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFFN()

    def forward(self, X, attn_mask):
        """
        X: [batch_size, seq_len, d_model]
        attn_mask: [batch_size, seq_q, seq_k]
        """
        # MultiHeadAttention 的forward 所需要的参数 Q K V attion_mask
        X = self.enc_self_attn(X, X, X, attn_mask)
        X = self.pos_ffn(X)
        return X


class Encoder(nn.Module):
    """
     Encoder
     总共包含三部分,
     1. 词向量embedding
     2. 位置编码部分
     3. n_layers, 就是n个重复的层
    """

    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)  # 位置赢
        # 进行重复 n_layers 层的模块
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, X):
        """ X: [batch_size, seq_len] 还没有转化成词向量, 还单纯只是个词表, 目的是为了先获取pad
        get_attn_pad_mask只是是为了得到句子中 pad 的位置信息
        forward 一般是调用上面定义的函数, 这样想就知道下一步干啥
        """
        print(X)
        enc_self_attn_mask = get_attn_pad_mask(X, X)
        X = self.src_emb(X)
        X += self.pos_emb(X)
        for layer in self.layers:
            X = layer(X, enc_self_attn_mask)
        return X


# Decoder 部分
class DecoderLayer(nn.Module):
    """
    一个 DecoderLayer 层包含
    1. 首先是 带 Mask 的 多头注意力机制
    2. 正常的多头注意力机制, 这里的Q来自这里, K, V 来自Encoder的输出
    3. 带位置信息的全连接网络
    """

    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PositionWiseFFN()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        当前层需要知道 Decoder 的输入X, 以及 Encoder 的输出 enc_X,
        decoder 自注意的pad屏蔽, 以及 与 encoder 的交互mask屏蔽
        """
        dec_outputs = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs, dec_self_attn_mask)
        "进行交互是 decoder 提供 Q, encoder 提供 K, V"
        dec_outputs = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs, dec_enc_attn_mask)
        dec_outputs = self.pos_ffn(dec_outputs)
        return dec_outputs


class Decoder(nn.Module):
    """
    Decoder 一般包含三部分
    1. 词向量embedding
    2. 位置编码部分
    3. n_layers 个 DecoderLayer 重复的层
    """

    def __init__(self):
        super(Decoder, self).__init__()
        self.tag_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """ ec_inputs : [batch_size x target_len] """
        dec_outputs = self.tag_emb(dec_inputs)
        dec_outputs = self.pos_emb(dec_outputs)

        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs)
        dec_self_attn_subsequent_mask = get_attn_subsequent_mask(dec_inputs)

        """这一步就是把 decoder 的输入的mask 和 屏蔽单词后面的信息(即右上三角矩阵) 相加做,并起来 因为pad效果可以叠加的 """
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequent_mask), 0)

        """
        这个做的是交互注意力机制中的mask矩阵，enc的输入是k，我去看这个k里面哪些是pad符号，
        给到后面的模型；注意哦，我q肯定也是有pad符号，但是这里我不在意的，之前说了好多次了哈
        """
        dec_enc_attn_mask = get_attn_pad_mask(dec_inputs, enc_inputs)

        for layer in self.layers:
            dec_outputs = layer(dec_outputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask)
        return dec_outputs


class Transformer(nn.Module):
    """
    Transformer 包含三部分
    1. encoder
    2. decoder
    3. linear softmax 预测输出什么
    """
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.projection = nn.Linear(d_model, tgt_vocab_size, bias=False)

    def forward(self, enc_inputs, dec_inputs):
        enc_outputs = self.encoder(enc_inputs)
        dec_outputs = self.decoder(dec_inputs, enc_inputs, enc_outputs)
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1))


if __name__ == '__main__':

    sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

    # Transformer Parameters
    # Padding Should be Zero
    src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
    src_vocab_size = len(src_vocab)

    tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
    tgt_vocab_size = len(tgt_vocab)

    src_len = 5  # length of source
    tgt_len = 5  # length of target

    """模型的参数"""
    d_model = 512  # Embedding Size
    d_ff = 2048  # FeedForward dimension
    d_k = d_v = 64  # dimension of K(=Q), V
    n_layers = 6  # number of Encoder of Decoder Layer
    n_heads = 8  # number of heads in Multi-Head Attention

    model = Transformer()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    EncInputs, DecInputs, target_batch = make_batch(sentences)

    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(EncInputs, DecInputs)
        loss = criterion(outputs, target_batch.contiguous().view(-1))
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
        loss.backward()
        optimizer.step()
