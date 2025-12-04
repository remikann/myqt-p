import copy
import numpy as np
import torch
from mmcv.cnn import ConvModule, build_conv_layer, kaiming_init
from mmcv.runner import force_fp32
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import Linear
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_

from mmdet3d.core import (circle_nms, draw_heatmap_gaussian, gaussian_radius,
                          xywhr2xyxyr, limit_period, PseudoSampler)
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.core import Box3DMode, LiDARInstance3DBoxes
from mmdet3d.models import builder
from mmdet3d.models.builder import HEADS, build_loss
from mmdet3d.models.utils import clip_sigmoid
from mmdet3d.models.fusion_layers import apply_3d_transformation
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
from mmdet.core import build_bbox_coder, multi_apply, build_assigner, build_sampler, AssignResult
from mmdet3d.ops.roiaware_pool3d import points_in_boxes_batch
# --- debug utils: safe rank & logging on rank0 only ---
import os, logging
try:
    from mmcv.runner import get_dist_info
except Exception:
    get_dist_info = None

def _is_main_process() -> bool:
    # 在分布式未初始化时也返回 True，避免“rank 未定义”
    try:
        if get_dist_info is None:
            return True
        rank, _world = get_dist_info()
        return (rank == 0)
    except Exception:
        return True

def _log_rank0(msg: str):
    if _is_main_process():
        print(msg)
# ------------------------------------------------------

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, input_channel, num_pos_feats=288):
        super().__init__()
        self.position_embedding_head = nn.Sequential(
            nn.Conv1d(input_channel, num_pos_feats, kernel_size=1),
            nn.BatchNorm1d(num_pos_feats),
            nn.ReLU(inplace=True),
            nn.Conv1d(num_pos_feats, num_pos_feats, kernel_size=1))

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()
        position_embedding = self.position_embedding_head(xyz)
        return position_embedding


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 self_posembed=None, cross_posembed=None, cross_only=False):
        super().__init__()
        self.cross_only = cross_only
        if not self.cross_only:
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            """Return an activation function given a string"""
            if activation == "relu":
                return F.relu
            if activation == "gelu":
                return F.gelu
            if activation == "glu":
                return F.glu
            raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

        self.activation = _get_activation_fn(activation)

        self.self_posembed = self_posembed
        self.cross_posembed = cross_posembed

    def with_pos_embed(self, tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, query, key, query_pos, key_pos, attn_mask=None):
        """
        :param query: B C Pq
        :param key: B C Pk
        :param query_pos: B Pq 3/6
        :param key_pos: B Pk 3/6
        :param value_pos: [B Pq 3/6]
        :return:
        """
        # NxCxP to PxNxC
        if self.self_posembed is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        if self.cross_posembed is not None:
            key_pos_embed = self.cross_posembed(key_pos).permute(2, 0, 1)
        else:
            key_pos_embed = None

        query = query.permute(2, 0, 1)
        key = key.permute(2, 0, 1)

        if not self.cross_only:
            q = k = v = self.with_pos_embed(query, query_pos_embed)
            query2 = self.self_attn(q, k, value=v)[0]
            query = query + self.dropout1(query2)
            query = self.norm1(query)

        query2 = self.multihead_attn(query=self.with_pos_embed(query, query_pos_embed),
                                     key=self.with_pos_embed(key, key_pos_embed),
                                     value=self.with_pos_embed(key, key_pos_embed), attn_mask=attn_mask)[0]
        query = query + self.dropout2(query2)
        query = self.norm2(query)

        query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
        query = query + self.dropout3(query2)
        query = self.norm3(query)

        # NxCxP to PxNxC
        query = query.permute(1, 2, 0)
        return query


class MultiheadAttention(nn.Module):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need
    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.
        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.
    Examples::
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))
            self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
            self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty(1, 1, embed_dim))
            self.bias_v = Parameter(torch.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()
        self._debug_printed = False
        
    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)


def multi_head_attention_forward(query,  # type: Tensor
                                 key,  # type: Tensor
                                 value,  # type: Tensor
                                 embed_dim_to_check,  # type: int
                                 num_heads,  # type: int
                                 in_proj_weight,  # type: Tensor
                                 in_proj_bias,  # type: Tensor
                                 bias_k,  # type: Optional[Tensor]
                                 bias_v,  # type: Optional[Tensor]
                                 add_zero_attn,  # type: bool
                                 dropout_p,  # type: float
                                 out_proj_weight,  # type: Tensor
                                 out_proj_bias,  # type: Tensor
                                 training=True,  # type: bool
                                 key_padding_mask=None,  # type: Optional[Tensor]
                                 need_weights=True,  # type: bool
                                 attn_mask=None,  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,  # type: Optional[Tensor]
                                 k_proj_weight=None,  # type: Optional[Tensor]
                                 v_proj_weight=None,  # type: Optional[Tensor]
                                 static_k=None,  # type: Optional[Tensor]
                                 static_v=None,  # type: Optional[Tensor]
                                 ):
    # type: (...) -> Tuple[Tensor, Optional[Tensor]]
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.
    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    qkv_same = torch.equal(query, key) and torch.equal(key, value)
    kv_same = torch.equal(key, value)

    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert list(query.size()) == [tgt_len, bsz, embed_dim]
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            q, k, v = F.linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                k, v = F.linear(key, _w, _b).chunk(2, dim=-1)

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = F.linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = F.linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = F.linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = torch.jit._unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == query.size(-1)

        k_proj_weight_non_opt = torch.jit._unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == key.size(-1)

        v_proj_weight_non_opt = torch.jit._unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.size()
        assert len1 == embed_dim and len2 == value.size(-1)

        if in_proj_bias is not None:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = F.linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = F.linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = F.linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = torch.cat([k, bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask,
                                       torch.zeros((attn_mask.size(0), 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        assert static_k.size(0) == bsz * num_heads
        assert static_k.size(2) == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.size(0) == bsz * num_heads
        assert static_v.size(2) == head_dim
        v = static_v

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        src_len += 1
        k = torch.cat([k, torch.zeros((k.size(0), 1) + k.size()[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = torch.cat([v, torch.zeros((v.size(0), 1) + v.size()[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = torch.cat([attn_mask, torch.zeros((attn_mask.size(0), 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = torch.cat(
                [key_padding_mask, torch.zeros((key_padding_mask.size(0), 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    attn_output_weights = torch.bmm(q, k.transpose(1, 2))
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        attn_mask = attn_mask.unsqueeze(0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    attn_output = torch.bmm(attn_output_weights, v)
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = F.linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 heads,
                 head_conv=64,
                 final_kernel=1,
                 init_bias=-2.19,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 **kwargs):
        super(FFN, self).__init__()

        self.heads = heads
        self.init_bias = init_bias
        for head in self.heads:
            classes, num_conv = self.heads[head]

            conv_layers = []
            c_in = in_channels
            for i in range(num_conv - 1):
                conv_layers.append(
                    ConvModule(
                        c_in,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=bias,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg))
                c_in = head_conv

            conv_layers.append(
                build_conv_layer(
                    conv_cfg,
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True))
            conv_layers = nn.Sequential(*conv_layers)

            self.__setattr__(head, conv_layers)

    def init_weights(self):
        """Initialize weights."""
        for head in self.heads:
            if head == 'heatmap':
                self.__getattr__(head)[-1].bias.data.fill_(self.init_bias)
            else:
                for m in self.__getattr__(head).modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

    def forward(self, x):
        """Forward function for SepHead.

        Args:
            x (torch.Tensor): Input feature map with the shape of
                [B, 512, 128, 128].

        Returns:
            dict[str: torch.Tensor]: contains the following keys:

                -reg （torch.Tensor): 2D regression value with the \
                    shape of [B, 2, H, W].
                -height (torch.Tensor): Height value with the \
                    shape of [B, 1, H, W].
                -dim (torch.Tensor): Size value with the shape \
                    of [B, 3, H, W].
                -rot (torch.Tensor): Rotation value with the \
                    shape of [B, 1, H, W].
                -vel (torch.Tensor): Velocity value with the \
                    shape of [B, 2, H, W].
                -heatmap (torch.Tensor): Heatmap with the shape of \
                    [B, N, H, W].
        """
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict


@HEADS.register_module()
class TransFusionHead(nn.Module):
    def __init__(self,
                 fuse_img=False,
                 num_views=0,
                 in_channels_img=64,
                 out_size_factor_img=4,
                 num_proposals=128,
                 auxiliary=True,
                 in_channels=128 * 3,
                 hidden_channel=128,
                 num_classes=4,
                 # config for Transformer
                 num_decoder_layers=3,
                 num_heads=8,
                 learnable_query_pos=False,
                 initialize_by_heatmap=False,
                 nms_kernel_size=1,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 # config for FFN
                 common_heads=dict(),
                 num_heatmap_convs=2,
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 bias='auto',
                 # loss
                 loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_iou=dict(type='VarifocalLoss', use_sigmoid=True, iou_weighted=True, reduction='mean'),
                 loss_bbox=dict(type='L1Loss', reduction='mean'),
                 loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean'),
                 with_aux_attr=False, num_vis=4, num_cause=6,
                 loss_vis=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
                 loss_cause=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2),
                 # others
                 train_cfg=None,
                 test_cfg=None,
                 bbox_coder=None,
                 ):
        super(TransFusionHead, self).__init__()

        self.num_classes = num_classes
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.learnable_query_pos = learnable_query_pos
        self.initialize_by_heatmap = initialize_by_heatmap
        self.nms_kernel_size = nms_kernel_size
        if self.initialize_by_heatmap is True:
            assert self.learnable_query_pos is False, "initialized by heatmap is conflicting with learnable query position"
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_heatmap = build_loss(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        if self.initialize_by_heatmap:
            layers = []
            layers.append(ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
            layers.append(build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
            self.heatmap_head = nn.Sequential(*layers)
            self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)
            ####################################### mikann添加的初始化 ########################################
            nn.init.xavier_normal_(self.class_encoding.weight)  # 添加Xavier初始化
            nn.init.constant_(self.class_encoding.bias, 0)
        else:
            # query feature
            self.query_feat = nn.Parameter(torch.randn(1, hidden_channel, self.num_proposals))
            self.query_pos = nn.Parameter(torch.rand([1, self.num_proposals, 2]), requires_grad=learnable_query_pos)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))
        # auxiliary attribute prediction 
        self.with_aux_attr = with_aux_attr
        if self.with_aux_attr:
            self.num_vis = num_vis
            self.num_cause = num_cause
            # 用 1x1 conv 在 proposal 维上出 logits（也可用 Linear）
            self.vis_head   = nn.Conv1d(hidden_channel, self.num_vis, kernel_size=1)
            self.cause_head = nn.Conv1d(hidden_channel, self.num_cause, kernel_size=1)
            self.loss_vis   = build_loss(loss_vis)
            self.loss_cause = build_loss(loss_cause)
            self.attr_ignore_index = -1
        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.fuse_img = fuse_img
        if self.fuse_img:
            self.num_views = num_views
            self.out_size_factor_img = out_size_factor_img
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,  # channel of img feature map
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            self.heatmap_head_img = copy.deepcopy(self.heatmap_head)
            # transformer decoder layers for img fusion
            self.decoder.append(
                TransformerDecoderLayer(
                    hidden_channel, num_heads, ffn_channel, dropout, activation,
                    self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                    cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                ))
            # cross-attention only layers for projecting img feature onto BEV
            for i in range(num_views):
                self.decoder.append(
                    TransformerDecoderLayer(
                        hidden_channel, num_heads, ffn_channel, dropout, activation,
                        self_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_posembed=PositionEmbeddingLearned(2, hidden_channel),
                        cross_only=True,
                    ))
            self.fc = nn.Sequential(*[nn.Conv1d(hidden_channel, hidden_channel, kernel_size=1)])

            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(FFN(hidden_channel * 2, heads, conv_cfg=conv_cfg, norm_cfg=norm_cfg, bias=bias))

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg['out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg['out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

        self.img_feat_pos = None
        self.img_feat_collapsed_pos = None
    def _feat_hw_from_cfg(self):
        """
        Compute (H, W) of BEV feature map from config, independent of forward() outputs.
        Uses train_cfg first, then fall back to test_cfg.
        """
        cfg_src = self.train_cfg if self.train_cfg is not None else self.test_cfg
        assert cfg_src is not None, "Both train_cfg and test_cfg are None; cannot compute feature map size."
        assert 'grid_size' in cfg_src and 'out_size_factor' in cfg_src, \
            "Missing 'grid_size' or 'out_size_factor' in cfg; please set them in config."

        # grid_size order is [x_len, y_len, z_len]; BEV feature uses x,y.
        x_len, y_len = cfg_src['grid_size'][0], cfg_src['grid_size'][1]
        osf = cfg_src['out_size_factor']
        # W 对应 x 方向，H 对应 y 方向（和常见 [B, C, H, W] 一致）
        W = int(np.floor(x_len / osf))
        H = int(np.floor(y_len / osf))
        return H, W

    def create_2D_grid(self, x_size, y_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_y, batch_x = torch.meshgrid(*[torch.linspace(it[0], it[1], it[2]) for it in meshgrid])
        batch_x = batch_x + 0.5
        batch_y = batch_y + 0.5
        coord_base = torch.cat([batch_x[None], batch_y[None]], dim=0)[None]
        coord_base = coord_base.view(1, 2, -1).permute(0, 2, 1)
        return coord_base

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]
    # 在 class TransFusionHead(nn.Module): 内部，加上
    def _as_list_of_dicts(self, outs):
        """保证前向输出统一成 list[dict] 的形式（即使只有一层解码器）"""
        if isinstance(outs, dict):
            return [outs]
        elif isinstance(outs, (list, tuple)):
            # 容错：有时返回的是 tuple
            return list(outs)
        else:
            raise TypeError(f'Unexpected preds type: {type(outs)}; expect dict or list[dict].')

    def forward_single(self, inputs, img_inputs, img_metas):
        """Forward function for CenterPoint.

        Args:
            inputs (torch.Tensor): Input feature map with the shape of
                [B, 512, 128(H), 128(W)]. (consistent with L748)

        Returns:
            list[dict]: Output results for tasks.
        """
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)

        #################################
        # image to BEV
        #################################
        lidar_feat_flatten = lidar_feat.view(batch_size, lidar_feat.shape[1], -1)  # [BS, C, H*W]
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)

        if self.fuse_img:
            img_feat = self.shared_conv_img(img_inputs)  # [BS * n_views, C, H, W]

            img_h, img_w, num_channel = img_inputs.shape[-2], img_inputs.shape[-1], img_feat.shape[1]
            raw_img_feat = img_feat.view(batch_size, self.num_views, num_channel, img_h, img_w).permute(0, 2, 3, 1, 4) # [BS, C, H, n_views, W]
            img_feat = raw_img_feat.reshape(batch_size, num_channel, img_h, img_w * self.num_views)  # [BS, C, H, n_views*W]
            img_feat_collapsed = img_feat.max(2).values
            img_feat_collapsed = self.fc(img_feat_collapsed).view(batch_size, num_channel, img_w * self.num_views)

            # positional encoding for image guided query initialization
            if self.img_feat_collapsed_pos is None:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos = self.create_2D_grid(1, img_feat_collapsed.shape[-1]).to(img_feat.device)
            else:
                img_feat_collapsed_pos = self.img_feat_collapsed_pos

            bev_feat = lidar_feat_flatten
            for idx_view in range(self.num_views):
                bev_feat = self.decoder[2 + idx_view](bev_feat, img_feat_collapsed[..., img_w * idx_view:img_w * (idx_view + 1)], bev_pos, img_feat_collapsed_pos[:, img_w * idx_view:img_w * (idx_view + 1)])

        #################################
        # image guided query initialization
        #################################
        if self.initialize_by_heatmap:
            dense_heatmap = self.heatmap_head(lidar_feat)
            dense_heatmap_img = None
            if self.fuse_img:
                dense_heatmap_img = self.heatmap_head_img(bev_feat.view(lidar_feat.shape))  # [BS, num_classes, H, W]
                heatmap = (dense_heatmap.detach().sigmoid() + dense_heatmap_img.detach().sigmoid()) / 2
            else:
                heatmap = dense_heatmap.detach().sigmoid()
            padding = self.nms_kernel_size // 2
            local_max = torch.zeros_like(heatmap)
            # equals to nms radius = voxel_size * out_size_factor * kenel_size
            local_max_inner = F.max_pool2d(heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
            local_max[:, :, padding:(-padding), padding:(-padding)] = local_max_inner
            ## for Pedestrian & Traffic_cone in nuScenes
            if self.test_cfg['dataset'] == 'nuScenes':
                # local_max[:, 8, ] = F.max_pool2d(heatmap[:, 8], kernel_size=1, stride=1, padding=0)
                # local_max[:, 9, ] = F.max_pool2d(heatmap[:, 9], kernel_size=1, stride=1, padding=0)

                local_max = F.max_pool2d(
                    heatmap, kernel_size=self.nms_kernel_size, stride=1,
                    padding=self.nms_kernel_size // 2
                )
                # 仅当这些通道存在时才做“1x1 特例”
                for k in (8, 9):
                    if heatmap.shape[1] > k:
                        # 注意保持维度：使用切片 k:k+1，再 squeeze 回来
                        local_max[:, k] = F.max_pool2d(
                            heatmap[:, k:k+1], kernel_size=1, stride=1, padding=0
                        ).squeeze(1)
            elif self.test_cfg['dataset'] == 'Waymo':  # for Pedestrian & Cyclist in Waymo
                local_max[:, 1, ] = F.max_pool2d(heatmap[:, 1], kernel_size=1, stride=1, padding=0)
                local_max[:, 2, ] = F.max_pool2d(heatmap[:, 2], kernel_size=1, stride=1, padding=0)
            # heatmap = heatmap * (heatmap == local_max)
            # heatmap = heatmap.view(batch_size, heatmap.shape[1], -1)

            # # top #num_proposals among all classes
            # top_proposals = heatmap.view(batch_size, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
            # top_proposals_class = top_proposals // heatmap.shape[-1]
            # top_proposals_index = top_proposals % heatmap.shape[-1]
            heatmap = heatmap * (heatmap == local_max)
            # NOTE: 先保留 H、W，再展平成 [B, C, H*W]，这样类别与平面索引的拆分最清晰
            B, C, H, W = heatmap.shape
            # after: B, C, H, W = heatmap.shape
            if _is_main_process():
                if C != self.num_classes:
                    _log_rank0(f"[TransFusionHead] heatmap C({C}) != num_classes({self.num_classes})")
                else:
                    _log_rank0(f"[TransFusionHead] heatmap OK: C={C}, num_classes={self.num_classes}, H={H}, W={W}, K={self.num_proposals}")
            assert C == self.num_classes, \
                f"Heatmap channel ({C}) must equal num_classes ({self.num_classes})."
            # K 合法性
            assert self.num_proposals <= (C * H * W), \
                f"num_proposals({self.num_proposals}) > C*H*W({C*H*W})"

            # nms kernel 合法（必须是正奇数；否则 padding/slice 会出错）
            if hasattr(self, 'nms_kernel_size'):
                k = int(self.nms_kernel_size)
                assert k >= 1 and (k % 2 == 1), \
                    f"nms_kernel_size({self.nms_kernel_size}) must be an odd positive int"    
                    
            heatmap = heatmap.view(B, C, H * W)

            # top #num_proposals among all classes（在 C*H*W 里选 K）
            top_proposals = heatmap.view(B, -1).argsort(dim=-1, descending=True)[..., :self.num_proposals]
            # 类别 = // (H*W)；用 torch.div 消除 __floordiv__ 警告（行为与原来一致，等价于“向 0 取整”）
            top_proposals_class = torch.div(top_proposals, H * W, rounding_mode='trunc')
            # 平面内索引 = % (H*W)
            top_proposals_index = top_proposals % (H * W)
            # 防呆：类别索引应在 [0, C)
            assert top_proposals_class.max() < C and top_proposals_class.min() >= 0
            query_feat = lidar_feat_flatten.gather(index=top_proposals_index[:, None, :].expand(-1, lidar_feat_flatten.shape[1], -1), dim=-1)
            self.query_labels = top_proposals_class
            
            
            # add category embedding
            one_hot = F.one_hot(top_proposals_class, num_classes=self.num_classes).permute(0, 2, 1)
            query_cat_encoding = self.class_encoding(one_hot.float())
            query_feat += query_cat_encoding

            query_pos = bev_pos.gather(index=top_proposals_index[:, None, :].permute(0, 2, 1).expand(-1, -1, bev_pos.shape[-1]), dim=1)
        else:
            query_feat = self.query_feat.repeat(batch_size, 1, 1)  # [BS, C, num_proposals]
            base_xyz = self.query_pos.repeat(batch_size, 1, 1).to(lidar_feat.device)  # [BS, num_proposals, 2]
            # ★ 关键：A 方案下显式给 query_pos，避免后面 decoder 使用时报 UnboundLocalError
            query_pos = base_xyz
        #################################
        # transformer decoder layer (LiDAR feature as K,V)
        #################################
        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Transformer Decoder Layer
            # :param query: B C Pq    :param query_pos: B Pq 3/6
            query_feat = self.decoder[i](query_feat, lidar_feat_flatten, query_pos, bev_pos)
            
            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            if self.with_aux_attr and i == self.num_decoder_layers - 1:
            # query_feat: [B, C, num_proposals]
                vis_logits   = self.vis_head(query_feat)      # [B, 4, K]
                cause_logits = self.cause_head(query_feat)    # [B, 6, K]
                res_layer['vis_logits']   = vis_logits
                res_layer['cause_logits'] = cause_logits            
            first_res_layer = res_layer
            if not self.fuse_img:
                ret_dicts.append(res_layer)

            # for next level positional embedding
            query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        #################################
        # transformer decoder layer (img feature as K,V)
        #################################
        if self.fuse_img:
            # positional encoding for image fusion
            img_feat = raw_img_feat.permute(0, 3, 1, 2, 4) # [BS, n_views, C, H, W]
            img_feat_flatten = img_feat.view(batch_size, self.num_views, num_channel, -1)  # [BS, n_views, C, H*W]
            if self.img_feat_pos is None:
                (h, w) = img_inputs.shape[-2], img_inputs.shape[-1]
                img_feat_pos = self.img_feat_pos = self.create_2D_grid(h, w).to(img_feat_flatten.device)
            else:
                img_feat_pos = self.img_feat_pos

            prev_query_feat = query_feat.detach().clone()
            query_feat = torch.zeros_like(query_feat)  # create new container for img query feature
            query_pos_realmetric = query_pos.permute(0, 2, 1) * self.test_cfg['out_size_factor'] * self.test_cfg['voxel_size'][0] + self.test_cfg['pc_range'][0]
            query_pos_3d = torch.cat([query_pos_realmetric, res_layer['height']], dim=1).detach().clone()
            if 'vel' in res_layer:
                vel = copy.deepcopy(res_layer['vel'].detach())
            else:
                vel = None
            pred_boxes = self.bbox_coder.decode(
                copy.deepcopy(res_layer['heatmap'].detach()),
                copy.deepcopy(res_layer['rot'].detach()),
                copy.deepcopy(res_layer['dim'].detach()),
                copy.deepcopy(res_layer['center'].detach()),
                copy.deepcopy(res_layer['height'].detach()),
                vel,
            )

            on_the_image_mask = torch.ones([batch_size, self.num_proposals]).to(query_pos_3d.device) * -1

            for sample_idx in range(batch_size if self.fuse_img else 0):
                lidar2img_rt = query_pos_3d.new_tensor(img_metas[sample_idx]['lidar2img'])
                img_scale_factor = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['scale_factor'][:2]
                                            if 'scale_factor' in img_metas[sample_idx].keys() else [1.0, 1.0]))
                img_flip = img_metas[sample_idx]['flip'] if 'flip' in img_metas[sample_idx].keys() else False
                img_crop_offset = (
                    query_pos_3d.new_tensor(img_metas[sample_idx]['img_crop_offset'])
                    if 'img_crop_offset' in img_metas[sample_idx].keys() else 0)
                img_shape = img_metas[sample_idx]['img_shape'][:2]
                img_pad_shape = img_metas[sample_idx]['input_shape'][:2]
                boxes = LiDARInstance3DBoxes(pred_boxes[sample_idx]['bboxes'][:, :7], box_dim=7)
                query_pos_3d_with_corners = torch.cat([query_pos_3d[sample_idx], boxes.corners.permute(2, 0, 1).view(3, -1)], dim=-1)  # [3, num_proposals] + [3, num_proposals*8]
                # transform point clouds back to original coordinate system by reverting the data augmentation
                if batch_size == 1:  # skip during inference to save time
                    points = query_pos_3d_with_corners.T
                else:
                    points = apply_3d_transformation(query_pos_3d_with_corners.T, 'LIDAR', img_metas[sample_idx], reverse=True).detach()
                num_points = points.shape[0]

                for view_idx in range(self.num_views):
                    pts_4d = torch.cat([points, points.new_ones(size=(num_points, 1))], dim=-1)
                    pts_2d = pts_4d @ lidar2img_rt[view_idx].t()

                    pts_2d[:, 2] = torch.clamp(pts_2d[:, 2], min=1e-5)
                    pts_2d[:, 0] /= pts_2d[:, 2]
                    pts_2d[:, 1] /= pts_2d[:, 2]

                    # img transformation: scale -> crop -> flip
                    # the image is resized by img_scale_factor
                    img_coors = pts_2d[:, 0:2] * img_scale_factor  # Nx2
                    img_coors -= img_crop_offset

                    # grid sample, the valid grid range should be in [-1,1]
                    coor_x, coor_y = torch.split(img_coors, 1, dim=1)  # each is Nx1

                    if img_flip:
                        # by default we take it as horizontal flip
                        # use img_shape before padding for flip
                        orig_h, orig_w = img_shape
                        coor_x = orig_w - coor_x

                    coor_x, coor_corner_x = coor_x[0:self.num_proposals, :], coor_x[self.num_proposals:, :]
                    coor_y, coor_corner_y = coor_y[0:self.num_proposals, :], coor_y[self.num_proposals:, :]
                    coor_corner_x = coor_corner_x.reshape(self.num_proposals, 8, 1)
                    coor_corner_y = coor_corner_y.reshape(self.num_proposals, 8, 1)
                    coor_corner_xy = torch.cat([coor_corner_x, coor_corner_y], dim=-1)

                    h, w = img_pad_shape
                    on_the_image = (coor_x > 0) * (coor_x < w) * (coor_y > 0) * (coor_y < h)
                    on_the_image = on_the_image.squeeze()
                    # skip the following computation if no object query fall on current image
                    if on_the_image.sum() <= 1:
                        continue
                    on_the_image_mask[sample_idx, on_the_image] = view_idx

                    # add spatial constraint
                    center_ys = (coor_y[on_the_image] / self.out_size_factor_img)
                    center_xs = (coor_x[on_the_image] / self.out_size_factor_img)
                    centers = torch.cat([center_xs, center_ys], dim=-1).int()  # center on the feature map
                    corners = (coor_corner_xy[on_the_image].max(1).values - coor_corner_xy[on_the_image].min(1).values) / self.out_size_factor_img
                    radius = torch.ceil(corners.norm(dim=-1, p=2) / 2).int()  # radius of the minimum circumscribed circle of the wireframe
                    sigma = (radius * 2 + 1) / 6.0
                    distance = (centers[:, None, :] - (img_feat_pos - 0.5)).norm(dim=-1) ** 2
                    gaussian_mask = (-distance / (2 * sigma[:, None] ** 2)).exp()
                    gaussian_mask[gaussian_mask < torch.finfo(torch.float32).eps] = 0
                    attn_mask = gaussian_mask

                    query_feat_view = prev_query_feat[sample_idx, :, on_the_image]
                    query_pos_view = torch.cat([center_xs, center_ys], dim=-1)
                    query_feat_view = self.decoder[self.num_decoder_layers](query_feat_view[None], img_feat_flatten[sample_idx:sample_idx + 1, view_idx], query_pos_view[None], img_feat_pos, attn_mask=attn_mask.log())
                    query_feat[sample_idx, :, on_the_image] = query_feat_view.clone()

            self.on_the_image_mask = (on_the_image_mask != -1)
            res_layer = self.prediction_heads[self.num_decoder_layers](torch.cat([query_feat, prev_query_feat], dim=1))
            res_layer['center'] = res_layer['center'] + query_pos.permute(0, 2, 1)
            for key, value in res_layer.items():
                pred_dim = value.shape[1]
                res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)] = first_res_layer[key][~self.on_the_image_mask.unsqueeze(1).repeat(1, pred_dim, 1)]
            ret_dicts.append(res_layer)

        if self.initialize_by_heatmap:
            ret_dicts[0]['query_heatmap_score'] = heatmap.gather(index=top_proposals_index[:, None, :].expand(-1, self.num_classes, -1), dim=-1)  # [bs, num_classes, num_proposals]
            if self.fuse_img:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap_img
            else:
                ret_dicts[0]['dense_heatmap'] = dense_heatmap

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        # new_res = {}
        # for key in ret_dicts[0].keys():
        #     if key not in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
        #         new_res[key] = torch.cat([ret_dict[key] for ret_dict in ret_dicts], dim=-1)
        #     else:
        #         new_res[key] = ret_dicts[0][key]
        # return [new_res]
        keys = set().union(*[d.keys() for d in ret_dicts])   # 用所有层的并集
        new_res = {}
        for k in keys:
            if k in ['dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score']:
                new_res[k] = ret_dicts[0][k]
            else:
                new_res[k] = torch.cat([d[k] for d in ret_dicts if k in d], dim=-1)
        return [new_res]
    def forward(self, feats, img_feats, img_metas):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple(list[dict]): Output results. first index by level, second index by layer
        """
        if img_feats is None:
            img_feats = [None]
        res = multi_apply(self.forward_single, feats, img_feats, [img_metas])
        assert len(res) == 1, "only support one level features."
        return res[0]

    def get_targets(self, gt_bboxes_3d, gt_labels_3d, preds_dict):
        """Generate training targets.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dicts (tuple of dict): first index by layer (default 1)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)  [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        
        list_of_pred_dict = []
        for batch_idx in range(len(gt_bboxes_3d)):
            pred_dict = {}
            for key in preds_dict[0].keys():
                pred_dict[key] = preds_dict[0][key][batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(gt_bboxes_3d) == len(list_of_pred_dict)

        res_tuple = multi_apply(self.get_targets_single, gt_bboxes_3d, gt_labels_3d, list_of_pred_dict, np.arange(len(gt_labels_3d)))
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        if self.initialize_by_heatmap:
            heatmap = torch.cat(res_tuple[7], dim=0)
            assigned_gt_inds_full = torch.cat(res_tuple[8], dim=0)  # ← 新增
            return (labels, label_weights, bbox_targets, bbox_weights,ious, num_pos, matched_ious, heatmap, assigned_gt_inds_full)
        else:
            assigned_gt_inds_full = torch.cat(res_tuple[7], dim=0)  # ← 新增
            return (labels, label_weights, bbox_targets, bbox_weights,ious, num_pos, matched_ious, assigned_gt_inds_full)

    def get_targets_single(self, gt_bboxes_3d, gt_labels_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.

        Args:
            gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): Ground truth gt boxes.
            gt_labels_3d (torch.Tensor): Labels of boxes.
            preds_dict (dict): dict of prediction result for a single sample
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.

                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask)  [1, num_proposals]
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
        """
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! donot change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(score, rot, dim, center, height, vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign seperately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1), :]
            score_layer = score[..., self.num_proposals * idx_layer:self.num_proposals * (idx_layer + 1)]

            if self.train_cfg.assigner.type == 'HungarianAssigner3D':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, gt_labels_3d, score_layer, self.train_cfg)
            elif self.train_cfg.assigner.type == 'HeuristicAssigner':
                assign_result = self.bbox_assigner.assign(bboxes_tensor_layer, gt_bboxes_tensor, None, gt_labels_3d, self.query_labels[batch_idx])
            else:
                raise NotImplementedError
            assign_result_list.append(assign_result)

        # # combine assign result of each layer
        # assign_result_ensemble = AssignResult(
        #     num_gts=sum([res.num_gts for res in assign_result_list]),a
        #     gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
        #     max_overlaps=torch.cat([res.max_overlaps for res in assign_result_list]),
        #     labels=torch.cat([res.labels for res in assign_result_list]),
        # )
        # ---- 鲁棒合并：过滤 None，避免 torch.cat 遇到 NoneType
        valid = [res for res in assign_result_list
                 if (res is not None and getattr(res, 'max_overlaps', None) is not None)]
        if len(valid) == 0:
            # 没有任何有效分配：返回一组“全负样本”占位，保证训练继续
            device = center.device
            labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long) + self.num_classes
            label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
            bbox_targets = bboxes_tensor.new_zeros(num_proposals, self.bbox_coder.code_size)
            bbox_weights = bboxes_tensor.new_zeros_like(bbox_targets)
            ious = bboxes_tensor.new_zeros(num_proposals)
            if self.initialize_by_heatmap:
                grid_size = torch.as_tensor(self.train_cfg['grid_size'])
                feature_hw = torch.div(grid_size[:2], self.train_cfg['out_size_factor'], rounding_mode='trunc')
                heatmap = bbox_targets.new_zeros(self.num_classes, feature_hw[1], feature_hw[0])
                return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None],
                        ious[None], 0, 0.0, heatmap[None])
            else:
                return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None],
                        ious[None], 0, 0.0)

        # 正常合并有效的层
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in valid]),
            gt_inds=torch.cat([res.gt_inds for res in valid]),
            max_overlaps=torch.cat([res.max_overlaps for res in valid]),
            labels=torch.cat([res.labels for res in valid]),
        )
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble, bboxes_tensor, gt_bboxes_tensor)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals
        # === 新增：为每个 proposal 记录它匹配到的 gt 索引（负样本为 -1） ===
        assigned_gt_inds_full = torch.full(
            (num_proposals,), -1, dtype=torch.long, device=bboxes_tensor.device
        )
        if len(pos_inds) > 0:
            assigned_gt_inds_full[pos_inds] = sampling_result.pos_assigned_gt_inds
        # create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        if self.initialize_by_heatmap:
            device = labels.device
            # [B, 7] -> [x, y, z, dx, dy, dz, yaw]，这里与原逻辑一致
            gt_bboxes_3d = torch.cat(
                [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
                dim=1
            ).to(device)

            pc_range   = torch.as_tensor(self.train_cfg['point_cloud_range'], device=device)
            voxel_size = torch.as_tensor(self.train_cfg['voxel_size'], device=device)
            osf        = int(self.train_cfg['out_size_factor'])

            # 1) 优先从当前样本的 preds_dict 里推断 (H, W)
            #    你的 forward 会把 dense_heatmap 放到 ret_dicts[0]['dense_heatmap']，进入 get_targets_single 时是单样本的 dict
            if isinstance(preds_dict, dict) and ('dense_heatmap' in preds_dict) and (preds_dict['dense_heatmap'] is not None):
                # shape: [1, C, H, W] 或 [C, H, W]
                dhm = preds_dict['dense_heatmap']
                H, W = int(dhm.shape[-2]), int(dhm.shape[-1])
            else:
                # 2) 兜底：从配置推断 (H, W) = grid_size[:2] // out_size_factor
                grid_size_xy = torch.as_tensor(self.train_cfg['grid_size'][:2], device=device)
                W = int(grid_size_xy[0].item() // osf)   # x_len
                H = int(grid_size_xy[1].item() // osf)   # y_len

            # 注意：feature_map_size 在你原注释里是 [x_len, y_len] = [W, H]
            feature_map_size = torch.tensor([W, H], device=device, dtype=torch.long)

            # 可选一次性打印，便于核对
            if (not hasattr(self, '_dbg_printed')) or (self._dbg_printed is False):
                print(f"[TransFusionHead][DEBUG] feature_map_size(x,y)=({W},{H})")
                self._dbg_printed = True

            # 构建密集热力图标签：形状 [num_classes, H, W]
            heatmap = gt_bboxes_3d.new_zeros(self.num_classes, H, W)

            for idx in range(len(gt_bboxes_3d)):
                # 你的宽长定义与原实现保持一致：先按真实尺寸，再换算到特征图单位
                w = ((gt_bboxes_3d[idx][3] / voxel_size[0]) / osf).to(device=device, dtype=torch.float32)
                l = ((gt_bboxes_3d[idx][4] / voxel_size[1]) / osf).to(device=device, dtype=torch.float32)

                # 注意：0 维 tensor 不能直接用于 if，先 .item()
                if (w.item() > 0.0) and (l.item() > 0.0):
                    mo = torch.tensor(float(self.train_cfg['gaussian_overlap']),
                                    device=device, dtype=torch.float32)
                    # 传入 (tensor, tensor) 而不是 float
                    radius_t = gaussian_radius((l, w), min_overlap=mo)  # 返回 tensor
                    # 转成 Python int，并与 min_radius 取最大值
                    radius = max(int(self.train_cfg['min_radius']), int(radius_t.item()))

                    # 中心点从真实坐标 -> 网格坐标 -> 特征图坐标
                    x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]
                    coor_x = (x - pc_range[0]) / voxel_size[0] / osf
                    coor_y = (y - pc_range[1]) / voxel_size[1] / osf

                    center     = torch.tensor([coor_x, coor_y], dtype=torch.float32, device=device)
                    center_int = center.to(torch.int32)

                    # 画二维高斯
                    draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius)
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            # return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None]
            return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None],ious[None], int(pos_inds.shape[0]), float(mean_iou), heatmap[None], assigned_gt_inds_full[None])
        else:
            mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
            # return labels[None], label_weights[None], bbox_targets[None], bbox_weights[None], ious[None], int(pos_inds.shape[0]), float(mean_iou)
            return (labels[None], label_weights[None], bbox_targets[None], bbox_weights[None],ious[None], int(pos_inds.shape[0]), float(mean_iou), assigned_gt_inds_full[None])

    
    @force_fp32(apply_to=('preds_dicts',))
    def loss(self, gt_bboxes_3d, gt_labels_3d, preds_dicts,
            gt_vis_labels_list=None, gt_cause_labels_list=None,
            img_metas=None, **kwargs):
        """Loss function for TransFusion head (+ optional visibility / cause)."""

        assert isinstance(preds_dicts, (list, tuple)) and len(preds_dicts) >= 1
        preds_dict = preds_dicts[0]  # merged dict after forward()

        loss_dict = dict()

        # -------- 1) 取 targets（兼容：是否带 heatmap / 是否带匹配索引） --------
        if self.initialize_by_heatmap:
            ret = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts)
            if len(ret) == 9:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap, assigned_gt_inds_full = ret
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, heatmap = ret
                assigned_gt_inds_full = None
        else:
            ret = self.get_targets(gt_bboxes_3d, gt_labels_3d, preds_dicts)
            if len(ret) == 8:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious, assigned_gt_inds_full = ret
            else:
                labels, label_weights, bbox_targets, bbox_weights, ious, num_pos, matched_ious = ret
                assigned_gt_inds_full = None

        # if hasattr(self, 'on_the_image_mask'):
        #     label_weights = label_weights * self.on_the_image_mask
        #     bbox_weights  = bbox_weights * self.on_the_image_mask[:, :, None]
            num_pos = bbox_weights.max(-1).values.sum()

        # -------- 2) heatmap loss（可选） --------
        if self.initialize_by_heatmap and 'dense_heatmap' in preds_dict:
            loss_heatmap = self.loss_heatmap(
                clip_sigmoid(preds_dict['dense_heatmap']),
                heatmap,
                avg_factor=max(heatmap.eq(1).float().sum().item(), 1)
            )
            loss_dict['loss_heatmap'] = loss_heatmap

        # -------- 3) 各层分类/回归损失 --------
        total_layers = self.num_decoder_layers if self.auxiliary else 1
        for idx_layer in range(total_layers):
            if idx_layer == self.num_decoder_layers - 1 or (idx_layer == 0 and not self.auxiliary):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            sl = slice(idx_layer * self.num_proposals, (idx_layer + 1) * self.num_proposals)

            # cls
            layer_labels        = labels[..., sl].reshape(-1)
            layer_label_weights = label_weights[..., sl].reshape(-1)
            layer_score         = preds_dict['heatmap'][..., sl]           # [B, C, K]
            layer_cls_score     = layer_score.permute(0, 2, 1).reshape(-1, self.num_classes)

            if layer_cls_score.numel() == 0 or layer_labels.numel() == 0:
                layer_loss_cls = layer_cls_score.sum() * 0.0
            else:
                layer_loss_cls = self.loss_cls(
                    layer_cls_score, layer_labels, layer_label_weights,
                    avg_factor=max(num_pos, 1)
                )

            # bbox
            layer_center = preds_dict['center'][..., sl]
            layer_height = preds_dict['height'][..., sl]
            layer_rot    = preds_dict['rot'][..., sl]
            layer_dim    = preds_dict['dim'][..., sl]

            preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot], dim=1).permute(0, 2, 1)
            if 'vel' in preds_dict:
                layer_vel = preds_dict['vel'][..., sl]
                preds = torch.cat([layer_center, layer_height, layer_dim, layer_rot, layer_vel], dim=1).permute(0, 2, 1)

            code_weights       = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[:, sl, :]
            layer_reg_weights  = layer_bbox_weights * layer_bbox_weights.new_tensor(code_weights)
            layer_bbox_targets = bbox_targets[:, sl, :]

            layer_loss_bbox = self.loss_bbox(preds, layer_bbox_targets, layer_reg_weights, avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls']  = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        # -------- 4) 属性（visibility / cause）loss（仅在最后一层监督） --------
        if self.with_aux_attr and ('vis_logits' in preds_dict) and ('cause_logits' in preds_dict):
            # 取最后一层的 K 个 proposal（auxiliary=False 时就是唯一一层）
            if self.auxiliary:
                vis_logits   = preds_dict['vis_logits'][..., -self.num_proposals:].permute(0, 2, 1)   # [B, K, 4]
                cause_logits = preds_dict['cause_logits'][..., -self.num_proposals:].permute(0, 2, 1) # [B, K, 6]
            else:
                vis_logits   = preds_dict['vis_logits'].permute(0, 2, 1)
                cause_logits = preds_dict['cause_logits'].permute(0, 2, 1)

            if assigned_gt_inds_full is None:
                # 还没把“proposal→gt 索引”加到 get_targets 返回：先安全跳过
                loss_dict['loss_vis']   = vis_logits.sum() * 0
                loss_dict['loss_cause'] = cause_logits.sum() * 0
            else:
                # 只监督正样本（assigned_gt_inds_full >= 0）
                last_slice = slice((total_layers - 1) * self.num_proposals, total_layers * self.num_proposals)
                assigned_last = assigned_gt_inds_full[:, last_slice]  # [B, K]
                pos_mask = assigned_last >= 0

                device = vis_logits.device
                B, K = assigned_last.shape
                vis_tgt   = assigned_last.new_full((B, K), -1, dtype=torch.long)
                cause_tgt = assigned_last.new_full((B, K), -1, dtype=torch.long)

                for b in range(B):
                    if pos_mask[b].any():
                        gt_idx = assigned_last[b, pos_mask[b]]
                        gv = torch.as_tensor(gt_vis_labels_list[b],   device=device, dtype=torch.long)
                        gc = torch.as_tensor(gt_cause_labels_list[b], device=device, dtype=torch.long)
                        vis_tgt[b, pos_mask[b]]   = gv[gt_idx]
                        cause_tgt[b, pos_mask[b]] = gc[gt_idx]

                keep_v = (vis_tgt >= 0)
                keep_c = (cause_tgt >= 0)

                if keep_v.any():
                    loss_dict['loss_vis'] = self.loss_vis(vis_logits[keep_v], vis_tgt[keep_v])
                else:
                    loss_dict['loss_vis'] = vis_logits.sum() * 0
                if keep_c.any():
                    loss_dict['loss_cause'] = self.loss_cause(cause_logits[keep_c], cause_tgt[keep_c])
                else:
                    loss_dict['loss_cause'] = cause_logits.sum() * 0
        # -------- 5) 记录正样本数 --------            
        loss_dict['num_pos'] = layer_loss_cls.new_tensor(float(num_pos))
        return loss_dict

    # transfusion_head.py
    @force_fp32(apply_to=('preds_dicts',))
    def get_bboxes(self, preds_dicts, img_metas, img=None, rescale=False, for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (list[list[dict]]): 
                形状大致是 [num_layers, num_branch(=1), dict]，
                dict 里有 heatmap/center/dim/rot/height/(vel)/vis_logits/cause_logits 等。
            img_metas (list[dict]): 当前 batch 的 meta。

        Returns:
            list[dict]: 每张图一个 dict，键包括：
                'boxes_3d', 'scores_3d', 'labels_3d', 'attr_vis', 'attr_cause'
        """
        # 只用最后一层、pts 分支
        last_layer = preds_dicts[-1]
        # preds_dicts 可能是：
        #   1) list[dict]         -> last_layer 是 dict
        #   2) tuple[list[dict]]  -> last_layer 是 list[dict]，需要取第 0 个 dict
        if isinstance(last_layer, (list, tuple)):
            # 这种情况每个 batch 一个 dict，这里只用第 0 个就够拿 shape 了
            last = last_layer[0]
        else:
            # 这种情况本身就是 batched dict
            last = last_layer

        B = last['heatmap'].shape[0]
        # --------- 1. 截取最后 K 个 proposal，和你原来的做法一致 ---------
        heatmap = last['heatmap'][..., -self.num_proposals:].sigmoid()
        center  = last['center'][..., -self.num_proposals:]
        height  = last['height'][..., -self.num_proposals:]
        dim     = last['dim'][..., -self.num_proposals:]
        rot     = last['rot'][..., -self.num_proposals:]
        vel     = last['vel'][..., -self.num_proposals:] if 'vel' in last else None

        # --------- 2. decode box（原来就有的逻辑） ---------
        decoded = self.bbox_coder.decode(
            heatmap, rot, dim, center, height, vel, filter=True
        )
        # decoded[b] 是 dict，至少有 'bboxes' / 'scores' / 'labels'

        # --------- 3. 处理 attr（vis / cause）---------
        vis_pred   = None  # [B, K]
        cause_pred = None  # [B, K]

        if self.with_aux_attr:
            if 'vis_logits' in last:
                # vis_logits: [B, num_vis_cls, num_proposals]
                vis_logits = last['vis_logits'][..., -self.num_proposals:]
                # 你如果做 loss 时用的是 CrossEntropyLoss，这里就直接 argmax 即可
                vis_pred = vis_logits.argmax(dim=1)  # [B, K]

            if 'cause_logits' in last:
                cause_logits = last['cause_logits'][..., -self.num_proposals:]
                cause_pred = cause_logits.argmax(dim=1)  # [B, K]
        # 统一 img_metas 形态：不管传进来是 dict 还是 list/tuple，最后都变成 list[dict]
        if isinstance(img_metas, dict):
            meta_list = [img_metas]
        elif isinstance(img_metas, (list, tuple)):
            # 有时候会是 list[list[dict]]（多相机），有时候是 list[dict]（单相机）
            # 我们只关心“batch 维度”，多相机的情况你原本的代码是拿 meta['box_type_3d']，
            # 不区分相机，所以这里不需要展开更深一层。
            meta_list = list(img_metas)
        else:
            raise TypeError(f'img_metas must be dict or list/tuple of dict, but got {type(img_metas)}')
        # --------- 4. 对每张图构造输出结构 ---------
        bbox_list = []
        for b in range(B):
            boxes3d = decoded[b]['bboxes']   # Tensor [N, box_dim]
            scores  = decoded[b]['scores']   # [N]
            labels  = decoded[b]['labels']   # [N]

            # 转成 box_type_3d 对象（你之前就是这么做的）
            # 这里用前面统一好的 meta_list，兼容 dict / list 两种情况
            if b < len(meta_list):
                meta = meta_list[b]
            else:
                # 极端情况下如果 batch_size > len(meta_list)，就复用最后一个 meta
                meta = meta_list[-1]
            box_type_3d = meta['box_type_3d']
            boxes_3d_obj = box_type_3d(boxes3d, box_dim=boxes3d.shape[-1])

            # attr 也截到 N（一般 N == K，如果 bbox_coder 里有 filter，N 可能小于 K）
            attr_vis_b   = None
            attr_cause_b = None
            if vis_pred is not None:
                attr_vis_b = vis_pred[b][: boxes3d.size(0)]   # [N]
            if cause_pred is not None:
                attr_cause_b = cause_pred[b][: boxes3d.size(0)]  # [N]

            bbox_list.append(dict(
                boxes_3d   = boxes_3d_obj,
                scores_3d  = scores,
                labels_3d  = labels.int(),
                attr_vis   = attr_vis_b,     # 可能为 None
                attr_cause = attr_cause_b,   # 可能为 None
            ))

        return bbox_list