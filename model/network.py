from torch import nn
import torch
import math
from torch.nn import functional as F
from model.base_function import init_net


def define_g(init_type='normal', gpu_ids=[]):
    net = Generator(patchsizes=[1,4,8,16])
    return init_net(net, init_type, gpu_ids)


def define_d(init_type= 'normal', gpu_ids=[]):
    net = Discriminator(in_channels=3)
    return init_net(net, init_type, gpu_ids)



class Generator(nn.Module):
    def __init__(self, patchsizes, ngf=64, num_ent=4, num_det=4, max_ngf=256):
        super().__init__()
        self.down = Encoder(ngf)
        self.up = Decoder(ngf)
        self.num_ent = num_ent
        self.num_det = num_det
        self.transformerEnc = nn.ModuleList([
         TransformerEncoder(patchsizes, max_ngf) for i in range(num_ent)
        ])
        self.transformerDec = nn.ModuleList([
            TransformerDecoder(patchsizes, max_ngf) for i in range(num_ent)
        ])

    def forward(self, x, mask, r_mask):
        feature = torch.cat([x, mask], dim=1)
        feature = self.down(feature)
        m = F.interpolate(mask, size=feature.size()[-2:], mode='nearest')
        rm = F.interpolate(r_mask, size=feature.size()[-2:], mode='nearest')
        for enc_block in self.transformerEnc:
            feature = enc_block(feature, m)
        enc_fea = feature
        for dec_block in self.transformerDec:
            feature = dec_block(feature, enc_fea, m, rm)
        out = self.up(feature)
        return out


class Discriminator(nn.Module):
    def __init__(self, in_channels, use_sigmoid=True, use_spectral_norm=True, init_weights=True):
        super(Discriminator, self).__init__()
        self.use_sigmoid = use_sigmoid

        self.conv1 = self.features = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv2 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv3 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv4 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.conv5 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=1, bias=not use_spectral_norm), use_spectral_norm),
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)

        outputs = conv5
        if self.use_sigmoid:
            outputs = torch.sigmoid(conv5)

        return outputs, [conv1, conv2, conv3, conv4, conv5]


class Encoder(nn.Module):
    def __init__(self, ngf=64):
        super().__init__()
        self.encoder1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=4, out_channels=ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        )

        self.encoder2 = nn.Sequential(
            nn.Conv2d(in_channels=ngf, out_channels=ngf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2, track_running_stats=False),
            nn.LeakyReLU(0.2, True))

        self.encoder22 = ResnetBlock(ngf*2)
        self.encoder32 = ResnetBlock(ngf*4)
        self.encoder3 = nn.Sequential(
            nn.Conv2d(in_channels=ngf*2, out_channels=ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4, track_running_stats=False),
            nn.LeakyReLU(0.2, True))



    def forward(self, img_m):
        x = self.encoder1(img_m)
        x = self.encoder2(x)
        x = self.encoder22(x)
        x = self.encoder3(x)
        x = self.encoder32(x)
        return x


class Decoder(nn.Module):
    def __init__(self, ngf):
        super().__init__()
        self.decoder1 = nn.Sequential(
            nn.Conv2d(in_channels=ngf*4, out_channels=ngf*2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 2, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf, track_running_stats=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ngf, out_channels=ngf, kernel_size=3, padding=1),
            nn.InstanceNorm2d(ngf, track_running_stats=False),
            nn.LeakyReLU(0.2, True)
        )
        self.decoder3 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels=ngf, out_channels=3, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder1(x)
        x = torch.nn.functional.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        x = self.decoder2(x)
        x = self.decoder3(x)
        return x



class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1, use_spectral_norm=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(dim, track_running_stats=False),
        )

    def forward(self, x):
        out = x + self.conv_block(x)
        # Remove ReLU at the end of the residual block
        # http://torch.ch/blog/2016/02/04/resnets.html
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256):
        super().__init__()
        self.attn = MultiPatchMultiAttention(patchsizes, num_hidden)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, x, mask=None):
        x = self.attn(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256):
        super().__init__()

        self.cross_attn = MultiPatchMultiAttention(patchsizes, num_hidden)
        self.self_attn = MultiPatchMultiAttention(patchsizes, num_hidden)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, query, enc, mask=None, r_mask=None):
        x = self.self_attn(query, query, query, r_mask)
        x = self.cross_attn(x, enc, enc,  mask, cross=True, r_mask=r_mask)
        x = self.feed_forward(x)
        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    q, k, v  = B * N (h*w) * C
    """

    def forward(self, query, key, value, mask=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if mask is not None:
            scores.masked_fill(mask, float('-inf'))
            #scores = scores * mask
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiAttn(nn.Module):
    """
    Attention Network
    """

    def __init__(self, head=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        key, value, query B*C*H*W
        """
        super().__init__()
        self.h = head

        #self.output_linear = nn.Sequential(nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1), nn.LeakyReLU(0.2, inplace=True))

        self.attn = Attention()

    def forward(self, query, key, value, mask=None):

        B,N,C = key.size()
        num_hidden_per_attn = C // self.h
        k = key.view(B, N, self.h, num_hidden_per_attn)
        v = value.view(B, N, self.h, num_hidden_per_attn)
        q = query.view(B, N, self.h, num_hidden_per_attn)

        k = k.permute(2,0,1,3).contiguous() # view(-1, N, num_hidden_per_attn)
        v = v.permute(2,0,1,3).contiguous()
        q = q.permute(2,0,1,3).contiguous()

        if mask is not None:
            mask = mask.unsqueeze(0)
            out, attn = self.attn(q, k, v, mask)
        else:
            out, attn = self.attn(q, k, v)
        out = out.view(self.h, B, N, num_hidden_per_attn)
        out = out.permute(1, 2, 0, 3).contiguous().view(B, N, C)
        return out, attn



# Standard 2 layerd FFN of transformer
class FeedForward(nn.Module):
    def __init__(self, num_hidden):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x




class StandardMultiAtten(nn.Module):
    """
    Attention Network
    """

    def __init__(self, num_hidden, head=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        key, value, query B*C*H*W
        """
        super().__init__()

        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // head
        self.h = head

        self.key = nn.Conv2d(num_hidden, num_hidden, kernel_size=1)
        self.value = nn.Conv2d(num_hidden, num_hidden, kernel_size=1)
        self.query = nn.Conv2d(num_hidden, num_hidden, kernel_size=1)
        self.output_linear = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))

        self.attn = Attention()
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, query, key, value, mask):
        residual = query
        B,C,H,W = key.size()
        k = self.key(key).view(B, C, H*W).permute(0,2,1).contiguous().view(B, H*W, self.h, self.num_hidden_per_attn)
        v = self.value(value).view(B, C, -1).permute(0,2,1).contiguous().view(B, H*W, self.h, self.num_hidden_per_attn)
        q = self.query(query).view(B, C, -1).permute(0,2,1).contiguous().view(B, H*W, self.h, self.num_hidden_per_attn)

        k = k.permute(2,0,1,3).contiguous().view(-1, H*W, self.num_hidden_per_attn)
        v = v.permute(2,0,1,3).contiguous().view(-1, H*W, self.num_hidden_per_attn)
        q = q.permute(2,0,1,3).contiguous().view(-1, H*W, self.num_hidden_per_attn)

        out, _ = self.attn(q, k, v, mask)
        out = out.view(self.h, B, H*W, self.num_hidden_per_attn)
        out = out.permute(1, 2, 0, 3).contiguous().view(B, H*W, C)
        out = out.permute(0, 2, 1).contiguous().view(B, C, H, W)
        out = self.output_linear(out)
        out = out + residual
        return out


class MultiPatchMultiAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, patchsizes, num_hidden):
        super().__init__()
        self.patchsize = patchsizes
        self.num_head = len(patchsizes)
        self.num_hidden_per_attn = num_hidden // self.num_head
        self.query_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.output_linear = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))
        for i in patchsizes:
            if i == 1:
                attention = Attention()
                setattr(self, 'attention' + str(i), attention)
            else:
                attention = MultiAttn(head=i)
                setattr(self, 'attention' + str(i), attention)

    def forward(self, query, key, value, mask, cross=False, r_mask=None):
        residual = query
        B, C, H, W = query.size()
        output = []
        q_group = self.query_embedding(query)
        k_group = self.key_embedding(key)
        v_group = self.value_embedding(value)
        if cross:
            k_group = k_group * mask
            v_group = v_group * mask
            if r_mask is not None:
                q_group = q_group * r_mask

        for s, q, k, v in zip(self.patchsize,
                            torch.chunk(q_group, self.num_head, dim=1),
                            torch.chunk(k_group, self.num_head, dim=1),
                            torch.chunk(v_group, self.num_head, dim=1)):
            num_w = W // s
            num_h = H // s
            # 1) embedding and reshape

            q = q.view(B, self.num_hidden_per_attn, num_h, s, num_w, s)
            k = k.view(B, self.num_hidden_per_attn, num_h, s, num_w, s)
            v = v.view(B, self.num_hidden_per_attn, num_h, s, num_w, s)
            q = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                B,  num_h*num_w, self.num_hidden_per_attn*s*s)
            k = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                B,  num_h*num_w, self.num_hidden_per_attn*s*s)
            v = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                B,  num_h*num_w, self.num_hidden_per_attn*s*s)

            attnblock = getattr(self, 'attention'+ str(s))
            if cross:
                result, _ = attnblock(q, k, v)
            else:
                m = mask.view(B, 1, num_h, s, num_w, s)
                m = m.permute(0, 2, 4, 1, 3, 5).contiguous().view(
                    B, num_h * num_w, s * s)
                m = (m.mean(-1) < 0.5).unsqueeze(1).repeat(1, num_w * num_h, 1)
                result, _ = attnblock(q, k, v, m)
            # 3) "Concat" using a view and apply a final linear.
            result = result.view(B, num_h, num_w, self.num_hidden_per_attn, s, s)
            result = result.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, self.num_hidden_per_attn, H, W)
            output.append(result)
        output = torch.cat(output, 1)
        x = self.output_linear(output)
        x = x + residual
        return x


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module