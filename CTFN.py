import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, to_networkx
import networkx as nx
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
from einops import rearrange, repeat
from torch_geometric.data import Data

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        # mask value: -inf
        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # cat all output -> [b, n, head_num*head_dim]
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out

class Attention2(nn.Module):
    def __init__(self, dim, heads, dim_head, dropout):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x:[b,n,dim]
        b, n, _, h = *x.shape, self.heads

        # get qkv tuple:([b,n,head_num*head_dim],[...],[...])
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        # split q,k,v from [b,n,head_num*head_dim] -> [b,head_num,n,head_dim]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # transpose(k) * q / sqrt(head_dim) -> [b,head_num,n,n]
        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        # softmax normalization -> attention matrix
        attn = dots.softmax(dim=-1)
        # value * attention matrix -> output
        attn=attn.squeeze(1)
        # out = torch.einsum('bhij,bhjd->bhid', attn, v)
        # out = rearrange(out, 'b h n d -> b n (h d)')
        # out = self.to_out(out)
        return attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_head, dropout=dropout)))
            ]))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel, num_channel, [1, 2], 1, 0))

    def forward(self, x,  mask=None):
        if self.mode == 'ViT':
            for attn, ff in self.layers:
                x = attn(x,  mask=mask)
                x = ff(x)
        elif self.mode == 'CAF':
            last_output = []
            nl = 0
            for attn, ff in self.layers:
                last_output.append(x)
                if nl > 1:
                    MMM=torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)
                    MMM2 = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3))
                    x = self.skipcat[nl - 2](
                        torch.cat([x.unsqueeze(3), last_output[nl - 2].unsqueeze(3)], dim=3)).squeeze(3)
                x = attn(x,  mask=mask)
                x = ff(x)
                nl += 1

        return x

class Transformer2(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_head, dropout, num_channel, mode):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(Attention2(dim, heads=heads, dim_head=dim_head, dropout=dropout))

        self.mode = mode
        self.skipcat = nn.ModuleList([])
        for _ in range(depth - 2):
            self.skipcat.append(nn.Conv2d(num_channel, num_channel, [1, 2], 1, 0))

    def forward(self, x):
        for attn in self.layers:
            x = attn(x)
            # x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, n_gcn, near_band, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()

        patch_dim = n_gcn * near_band

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 1,201,64
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)  # [b,n,dim] 64*200*1*1*64=64*200*64
        b, n, _ = x.shape

        # add position embedding
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim] self.cls_token=1*1*64 -> 64*1*64
        x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim] 拼接操作 64*(200+1)*64->64*201*64
        pos = self.pos_embedding[:, :(n + 1)]  # [1,201,64]
        x += pos  # 64*201*64
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask)  # 64*201*64

        # classification: using cls_token output
        x = self.to_latent(x[:, 0])  # 64*64
        x = self.mlp_head(x)  # 64*16
        # MLP classification layer
        return x

class ViT2(nn.Module):
    def __init__(self, patch_dim, num_patches, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=1,
                 dim_head=16, dropout=0., emb_dropout=0., mode='ViT'):
        super().__init__()


        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 1,201,64
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer2(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, mode)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, mask=None):
        # patchs[batch, patch_num, patch_size*patch_size*c]  [batch,200,145*145]
        # x = rearrange(x, 'b c h w -> b c (h w)')

        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)  # [b,n,dim] 64*200*1*1*64=64*200*64
        b, n, _ = x.shape

        # add position embedding
        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)  # [b,1,dim] self.cls_token=1*1*64 -> 64*1*64
        # x = torch.cat((cls_tokens, x), dim=1)  # [b,n+1,dim] 拼接操作 64*(200+1)*64->64*201*64
        pos = self.pos_embedding[:, :(n)]  # [1,201,64]
        x += pos  # 64*201*64
        x = self.dropout(x)

        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x)  # 64*201*64

        # classification: using cls_token output
        # x = self.to_latent(x[:, 0])  # 64*64
        # x = self.mlp_head(x)  # 64*16
        # MLP classification layer
        return x

class SSConv(nn.Module):
    '''
    Spectral-Spatial Convolution
    '''
    def __init__(self, in_ch, out_ch,kernel_size=3):
        super(SSConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=out_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size//2,
            groups=out_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=False
        )
        self.Act1 = nn.LeakyReLU()
        self.Act2 = nn.LeakyReLU()
        self.BN=nn.BatchNorm2d(in_ch)


    def forward(self, input):
        out = self.point_conv(self.BN(input))
        out = self.Act1(out)
        out = self.depth_conv(out)
        out = self.Act2(out)
        return out

class MultiHeadBlockCNN(nn.Module):  # Multihead Attention Fusion Module for GCN
    def __init__(self, num_heads, input_size):
        super(MultiHeadBlockCNN, self).__init__()
        if input_size % num_heads != 0:
            raise ValueError(
                "The CNN input size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (input_size, num_heads))
        self.num_heasds = num_heads
        self.head_size = int(input_size / num_heads)
        self.attn_pool = []
        self.attn_conv = []

        for i in range(num_heads):
            conv = nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2)
            setattr(self, 'attn_conv%i' % i, conv)  # 将该层添加到这个Module中，setattr函数用来设置属性，其中第一个参数为继承的类别，第二个为名称，第三个是数值
            self.attn_conv.append(conv)

    def cut_to_heads(self, x):
        # x.shape [b, c, h, w] -> [n, b, c/n, h, w]
        (b, c, h, w) = x.shape
        x = x.reshape([b, self.num_heasds, self.head_size, h, w])
        return x.permute(1, 0, 2, 3, 4)

    def forward(self, input_tensor):
        multi_tensor = self.cut_to_heads(input_tensor)
        output_list = []
        for i in range(self.num_heasds):
            avg_out = torch.mean(multi_tensor[i], dim=1, keepdim=True)
            x = self.attn_conv[i](avg_out)
            out = torch.sigmoid(x).mul(multi_tensor[i])
            output_list.append(out)
        output_tensor = torch.cat(output_list, dim=1)
        return output_tensor

class CNNConvBlock(nn.Module):
    def __init__(self, ch_in, ch_out, k, h, w):
        super(CNNConvBlock, self).__init__()
        self.BN = nn.BatchNorm2d(ch_in)
        self.conv_in = nn.Conv2d(ch_in, ch_out, kernel_size=1, padding=0, stride=1, groups=1)
        self.conv_out = nn.Conv2d(ch_out, ch_out, kernel_size=k, padding=k // 2, stride=1, groups=ch_out)
        self.pool = nn.AvgPool2d(3, padding=1, stride=1)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.BN(x)
        x = self.act(self.conv_in(x))
        x = self.pool(x)
        x = self.act(self.conv_out(x))

        return x

class CTFN(nn.Module):
    def __init__(self, height: int, width: int, changel: int, class_count: int, Q: torch.Tensor, A: torch.Tensor, S: torch.Tensor, Edge_index, Edge_atter, SP_size:int, CNN_nhid):
        super(CTFN, self).__init__()
        # 类别数,即网络最终输出通道数
        self.height=height
        self.width=width
        self.class_count = class_count  # 类别数
        # 网络输入数据大小
        self.channel = changel#200
        self.Q = Q#20125*196
        self.S = S  # 20125*196
        self.Edge_index = Edge_index
        self.Edge_atter = Edge_atter
        self.num_node = SP_size
        self.norm_col_Q = Q / (torch.sum(Q, 0, keepdim=True))  # 列归一化Q20125*196
        layers_count=2#ip 2 sa 4
        
        # Spectra Transformation Sub-Network
        self.CNN_denoise = nn.Sequential()
        for i in range(layers_count):
            if i == 0:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(self.channel))
                self.CNN_denoise.add_module('CNN_denoise_Conv'+str(i),nn.Conv2d(self.channel, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act'+str(i),nn.LeakyReLU())
            else:
                self.CNN_denoise.add_module('CNN_denoise_BN'+str(i),nn.BatchNorm2d(128),)
                self.CNN_denoise.add_module('CNN_denoise_Conv' + str(i), nn.Conv2d(128, 128, kernel_size=(1, 1)))
                self.CNN_denoise.add_module('CNN_denoise_Act' + str(i), nn.LeakyReLU())
        
        # Pixel-level Convolutional Sub-Network
        self.CNN_Branch = nn.Sequential()
        for i in range(layers_count):
            if i<layers_count-1:
                self.CNN_Branch.add_module('CNN_Branch'+str(i),SSConv(128, 128,kernel_size=5))
            else:
                self.CNN_Branch.add_module('CNN_Branch' + str(i), SSConv(128, 64, kernel_size=5))
############################################################

        num_patches=SP_size
        dim=32 #64
        depth=5
        heads=4
        mlp_dim=8
        dropout=0.1
        emb_dropout=0.1
        dim_head = 16
        patch_dim=128
################################################################
        self.pos_embedding = nn.Parameter(torch.randn(num_patches , dim))  # 1,201,64
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, num_patches, 'ViT')

        self.pool = 'cls'
        self.to_latent = nn.Identity()



        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim)
        )

        self.Cat = nn.Sequential(nn.Conv2d(256, 256, [1, 2], 1, 0))
########################################################################
        # Softmax layer
        # self.Softmax_linear =nn.Sequential(nn.Linear(dim, self.class_count))
        self.Softmax_linear = nn.Sequential(nn.Linear(192+dim, self.class_count))
##########################################################################
        # self.Cat=nn.Sequential(nn.Conv2d(256, 256, [1, 2], 1, 0))
        # self.skipcat = nn.ModuleList([])
        # self.skipcat.append(nn.Conv2d(256, 256, [1, 2], 1, 0))

        # CNN Conv
        self.CNN_nhid = CNN_nhid
        self.CNNlayerA1 = CNNConvBlock(self.channel, CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA2 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)
        self.CNNlayerA3 = CNNConvBlock(CNN_nhid, CNN_nhid, 7, self.height, self.width)

        self.CNNlayerB1 = CNNConvBlock(self.channel, CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB2 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)
        self.CNNlayerB3 = CNNConvBlock(CNN_nhid, CNN_nhid, 5, self.height, self.width)

        self.CNNlayerC1 = CNNConvBlock(self.channel, CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC2 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)
        self.CNNlayerC3 = CNNConvBlock(CNN_nhid, CNN_nhid, 3, self.height, self.width)

        CNN_nhead = 6
        self.CNN_hidden_size = 3 * CNN_nhid

        self.CNN_Multihead = MultiHeadBlockCNN(CNN_nhead, self.CNN_hidden_size)


        self.Tr_net= ViT2(
            patch_dim=1,  # 周围像素总数
            num_patches=192,  # 通道数
            num_classes=64,
            dim=100,  # 自定义64 200
            depth=1,
            heads=1,
            mlp_dim=8,
            dropout=0.1,
            emb_dropout=0.1
        )

    def forward(self, x: torch.Tensor,y_flatten):
        '''
        :param x: H*W*C
        :return: probability_map
        '''
        x_origin=x
        (h, w, c) = x.shape
        
        # 先去除噪声
        noise = self.CNN_denoise(torch.unsqueeze(x.permute([2, 0, 1]), 0))#permute是维度交换https://zhuanlan.zhihu.com/p/76583143
        noise =torch.squeeze(noise, 0).permute([1, 2, 0])
        clean_x=noise  #直连
        ##########STsN层，得到145*145*128的张量 clean_x #########

        clean_x_flatten=clean_x.reshape([h * w, -1])
        superpixels_flatten = torch.mm(self.norm_col_Q.t(), clean_x_flatten)  # 低频部分 196*21025 * 21025*128


        x = superpixels_flatten
        ## embedding every patch vector to embedding size: [batch, patch_num, embedding_size]
        x = x.to(torch.float32)
        x = self.patch_to_embedding(x)  # [b,n,dim] 64*200*1*1*64=64*200*64
        b, n = x.shape

        # add position embedding
        pos = self.pos_embedding[:, :(n)]  # [1,201,64]
        x += pos  # 64*201*64
        x = self.dropout(x)
        ########################################

        x = x.unsqueeze(0)
        # transformer: x[b,n + 1,dim] -> x[b,n + 1,dim]
        x = self.transformer(x, mask=None)  # 64*201*64
        x = self.mlp_head(x)  # 64*16
        Transformer_result = torch.matmul(self.Q, x.squeeze(0))
        Transformer_result = Transformer_result[y_flatten]
######################################### CNN ###############################################################

        CNNin = torch.unsqueeze(x_origin.permute([2, 0, 1]), 0)

        CNNmid1_A = self.CNNlayerA1(CNNin)
        CNNmid1_B = self.CNNlayerB1(CNNin)
        CNNmid1_C = self.CNNlayerC1(CNNin)

        CNNin = CNNmid1_A + CNNmid1_B + CNNmid1_C

        CNNmid2_A = self.CNNlayerA2(CNNin)
        CNNmid2_B = self.CNNlayerB2(CNNin)
        CNNmid2_C = self.CNNlayerC2(CNNin)

        CNNin = CNNmid2_A + CNNmid2_B + CNNmid2_C

        CNNout_A = self.CNNlayerA3(CNNin)
        CNNout_B = self.CNNlayerB3(CNNin)
        CNNout_C = self.CNNlayerC3(CNNin)

        CNNout = torch.cat([CNNout_A, CNNout_B, CNNout_C], dim=1)

        CNNout = torch.squeeze(CNNout, 0).permute([1, 2, 0]).reshape([self.height * self.width, -1])

        CNN_attention= CNNout[y_flatten].unsqueeze(2)
        attention_x = self.Tr_net(CNN_attention)

        out = torch.einsum('bij,bjd->bid', attention_x, CNN_attention).squeeze(2)
        out =0.98* CNN_attention.squeeze(2)+0.02*out
        Y1 = torch.cat([out, Transformer_result], dim=-1)
        Y = self.Softmax_linear(Y1)
        return Y

def data_process(S, Edge_index, Edge_atter,y,num_node):
    data = Data(x=S, edge_index=Edge_index, edge_attr=Edge_atter, y=y)
    data.num_nodes = num_node
    data.num_edges = Edge_index.shape[0]
    data.batch = torch.zeros(num_node, dtype=torch.int64).to(device)
    data = graphormer_pre_processing(
        data,
        20
    )
    return data


def graphormer_pre_processing(data, distance):

    graph: nx.DiGraph = to_networkx(data)

    data.in_degrees = torch.tensor([d for _, d in graph.in_degree()]).to(device)
    data.out_degrees = torch.tensor([d for _, d in graph.out_degree()]).to(device)

    max_in_degree = torch.max(data.in_degrees)
    max_out_degree = torch.max(data.out_degrees)
    if max_in_degree >= 64:#cfg.posenc_GraphormerBias.num_in_degrees
        raise ValueError(
            f"Encountered in_degree: {max_in_degree}, set posenc_"
            f"GraphormerBias.num_in_degrees to at least {max_in_degree + 1}"
        )
    if max_out_degree >= 64:#cfg.posenc_GraphormerBias.num_out_degrees
        raise ValueError(
            f"Encountered out_degree: {max_out_degree}, set posenc_"
            f"GraphormerBias.num_out_degrees to at least {max_out_degree + 1}"
        )

    N = len(graph.nodes)
    shortest_paths = nx.shortest_path(graph)

    spatial_types = torch.empty(N ** 2, dtype=torch.long).fill_(distance).to(device)
    graph_index = torch.empty(2, N ** 2, dtype=torch.long).to(device)

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        shortest_path_types = torch.zeros(N ** 2, distance, dtype=torch.long).to(device)
        edge_attr = torch.zeros(N, N, dtype=torch.long).to(device)
        edge_attr[data.edge_index[0], data.edge_index[1]] = data.edge_attr

    for i in range(N):
        for j in range(N):
            graph_index[0, i * N + j] = i
            graph_index[1, i * N + j] = j

    for i, paths in shortest_paths.items():
        for j, path in paths.items():
            if len(path) > distance:
                path = path[:distance]

            assert len(path) >= 1
            spatial_types[i * N + j] = len(path) - 1

            if len(path) > 1 and hasattr(data, "edge_attr") and data.edge_attr is not None:
                path_attr = [
                    edge_attr[path[k], path[k + 1]] for k in
                    range(len(path) - 1)  # len(path) * (num_edge_types)
                ]

                # We map each edge-encoding-distance pair to a distinct value
                # and so obtain dist * num_edge_features many encodings
                shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(
                    path_attr, dtype=torch.long)

    data.spatial_types = spatial_types
    data.graph_index = graph_index

    if hasattr(data, "edge_attr") and data.edge_attr is not None:
        data.shortest_path_types = shortest_path_types
    return data


# Permutes from (batch, node, node, head) to (batch, head, node, node)
BATCH_HEAD_NODE_NODE = (0, 3, 1, 2)

# Inserts a leading 0 row and a leading 0 column with F.pad
INSERT_GRAPH_TOKEN = (1, 0, 1, 0)

class BiasEncoder(torch.nn.Module):
    def __init__(self, num_heads: int, num_spatial_types: int,
                 num_edge_types: int, use_graph_token: bool = True):
        super().__init__()
        self.num_heads = num_heads

        # Takes into account disconnected nodes
        self.spatial_encoder = torch.nn.Embedding(
            num_spatial_types + 1, num_heads)
        self.spatial_encoder.to(device)
        self.edge_dis_encoder = torch.nn.Embedding(
            num_spatial_types * num_heads * num_heads, 1)
        self.edge_dis_encoder.to(device)
        self.edge_encoder = torch.nn.Embedding(num_edge_types, num_heads)
        self.edge_encoder.to(device)
        self.use_graph_token = use_graph_token
        if self.use_graph_token:
            self.graph_token = torch.nn.Parameter(torch.zeros(1, num_heads, 1))
        self.reset_parameters()

    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        self.edge_encoder.weight.data.normal_(std=0.02)
        self.edge_dis_encoder.weight.data.normal_(std=0.02)
        if self.use_graph_token:
            self.graph_token.data.normal_(std=0.02)

    def forward(self, data):
        spatial_types= self.spatial_encoder(data.spatial_types)
        spatial_encodings = to_dense_adj(data.graph_index,
                                         data.batch,
                                         spatial_types)
        bias = spatial_encodings.permute(BATCH_HEAD_NODE_NODE)

        if hasattr(data, "shortest_path_types"):
            edge_types: torch.Tensor = self.edge_encoder(
                data.shortest_path_types)
            edge_encodings = to_dense_adj(data.graph_index,
                                          data.batch,
                                          edge_types)

            spatial_distances = to_dense_adj(data.graph_index,
                                             data.batch,
                                             data.spatial_types)
            spatial_distances = spatial_distances.float().clamp(min=1.0).unsqueeze(1)

            B, N, _, max_dist, H = edge_encodings.shape

            edge_encodings = edge_encodings.permute(3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
            edge_encodings = torch.bmm(edge_encodings, self.edge_dis_encoder.weight.reshape(-1, self.num_heads, self.num_heads))
            edge_encodings = edge_encodings.reshape(max_dist, B, N, N, self.num_heads).permute(1, 2, 3, 0, 4)
            edge_encodings = edge_encodings.sum(-2).permute(BATCH_HEAD_NODE_NODE) / spatial_distances
            bias += edge_encodings

        if self.use_graph_token:
            bias = F.pad(bias, INSERT_GRAPH_TOKEN)
            bias[:, :, 1:, 0] = self.graph_token
            bias[:, :, 0, :] = self.graph_token

        B, H, N, _ = bias.shape
        data.attn_bias = bias.reshape(B * H, N, N)
        return data


class NodeEncoder(torch.nn.Module):
    def __init__(self, embed_dim, num_in_degree, num_out_degree,
                 input_dropout=0.0, use_graph_token: bool = True):
        super().__init__()
        self.in_degree_encoder = torch.nn.Embedding(num_in_degree, embed_dim).to(device)
        self.out_degree_encoder = torch.nn.Embedding(num_out_degree, embed_dim).to(device)

    def forward(self,data):
        in_degree_encoding = self.in_degree_encoder(data.in_degrees.data)
        out_degree_encoding = self.out_degree_encoder(data.out_degrees.data)
        data.degree_encoding = in_degree_encoding + out_degree_encoding

        return data


def add_graph_token(data, token):
    B = len(data.batch.unique())
    tokens = torch.repeat_interleave(token, B, 0)
    data.x = torch.cat([tokens, data.x], 0)
    data.batch = torch.cat(
        [torch.arange(0, B, device=data.x.device, dtype=torch.long), data.batch]
    )
    data.batch, sort_idx = torch.sort(data.batch)
    data.x = data.x[sort_idx]
    return data

class GraphormerEncoder(torch.nn.Sequential):
    def __init__(self, dim_emb ,*args, **kwargs):
        encoders = [
            BiasEncoder(
                4,#MHSA的頭的個數head
                20,#空間種類上限
                4,#邊的種類上限
                False
            ),
            NodeEncoder(
                dim_emb,#輸出編碼維度
                64,#出度和入度上下限 隨便設個較大值
                64,
                input_dropout=0.0,
            ),
        ]
        super().__init__(*encoders)

