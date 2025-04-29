import torch
from torch.functional import norm
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from collections import OrderedDict
import math
from itertools import combinations
from torch.nn.init import xavier_normal_ 
import numpy as np
# from torch.nn.modules.activation import MultiheadAttention
from copy import deepcopy
from torch.autograd import Variable
import torchvision.models as models
from ipdb import set_trace
from einops import rearrange
import os
from torch.autograd import Variable
from diffusers.models.attention_processor import Attention as CrossAttention
from utils.registry import Registry
from models.base.backbone import BACKBONE_REGISTRY
from models.base.base_blocks import HEAD_REGISTRY
# from collections import OrderedDict
from typing import Tuple, Union
import gzip
import html
import os
from functools import lru_cache, partial

import ftfy
import regex as re

import hashlib
import os
import urllib
import warnings
from typing import Any, Union, List
from pkg_resources import packaging

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from tqdm import tqdm
from utils.config import Config
from denoising_diffusion_pytorch import Unet1D, GaussianDiffusion1D
from denoising_diffusion_pytorch.denoising_diffusion_pytorch_1d import GaussianDiffusionDiT
from denoising_diffusion_pytorch.karras_unet_1d import DiT
# from .model import build_model
# from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

# torch.autograd.set_detect_anomaly(True)
if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out

__all__ = ["available_models", "load", "tokenize"]
_tokenizer = SimpleTokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
}


def _download(url: str, root: str):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target, "rb").read()).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match")

    return download_target


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load(name: str, cfg, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, spatial=False):
    """Load a CLIP model
    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).
    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")

    with open(model_path, 'rb') as opened_file:
        try:
            # loading JIT archive
            model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
            state_dict = None
        except RuntimeError:
            # loading saved state dict
            if jit:
                warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                jit = False
            state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(state_dict or model.state_dict(), cfg=cfg).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

def load_DiT(device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None, spatial=False):
    """Load a DiT model
    """
    model_path = download_root
    state_dict = torch.load(model_path, map_location="cpu")
    return state_dict


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)
    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize
    context_length : int
        The context length to use; all CLIP models use 77 as the context length
    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length
    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, spatial=False):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
        self.spatial = spatial
        self.embed_dim = embed_dim
        self.ml_proj = nn.Linear(embed_dim*2, embed_dim)

    def forward(self, x):
        # print('x.size',x.size())
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        if x.size()[-1] > self.embed_dim:
            x = self.ml_proj(x)
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC

        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        # if self.spatial:
        #     x_copy = x[1:]
        # x, _ = F.multi_head_attention_forward(
        #     query=x[:1], key=x, value=x,
        #     embed_dim_to_check=x.shape[-1],
        #     num_heads=self.num_heads,
        #     q_proj_weight=self.q_proj.weight,
        #     k_proj_weight=self.k_proj.weight,
        #     v_proj_weight=self.v_proj.weight,
        #     in_proj_weight=None,
        #     in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
        #     bias_k=None,
        #     bias_v=None,
        #     add_zero_attn=False,
        #     dropout_p=0,
        #     out_proj_weight=self.c_proj.weight,
        #     out_proj_bias=self.c_proj.bias,
        #     use_separate_proj_weight=True,
        #     training=self.training,
        #     need_weights=False
        # )
        if self.spatial:
            if self.spatial=="v2":
                cls_token, _ = F.multi_head_attention_forward(
                            query=x[:1], key=x, value=x,
                            embed_dim_to_check=x.shape[-1],
                            num_heads=self.num_heads,
                            q_proj_weight=self.q_proj.weight,
                            k_proj_weight=self.k_proj.weight,
                            v_proj_weight=self.v_proj.weight,
                            in_proj_weight=None,
                            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                            bias_k=None,
                            bias_v=None,
                            add_zero_attn=False,
                            dropout_p=0,
                            out_proj_weight=self.c_proj.weight,
                            out_proj_bias=self.c_proj.bias,
                            use_separate_proj_weight=True,
                            training=self.training,
                            need_weights=False
                            )
                x_mid = self.v_proj(x[1:])
                x_mid = self.c_proj(x_mid)
                x = torch.cat([cls_token, x_mid], dim=0)
                return x.squeeze(0)
            else:

                x, _ = F.multi_head_attention_forward(
                query=x, key=x, value=x,
                embed_dim_to_check=x.shape[-1],
                num_heads=self.num_heads,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
                in_proj_weight=None,
                in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
                bias_k=None,
                bias_v=None,
                add_zero_attn=False,
                dropout_p=0,
                out_proj_weight=self.c_proj.weight,
                out_proj_bias=self.c_proj.bias,
                use_separate_proj_weight=True,
                training=self.training,
                need_weights=False
                )
                # return torch.cat([x, self.c_proj(x_copy)], dim=0)
                return x.squeeze(0)
        else:
            x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
            )
            return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64, spatial=False, multilayer=False):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        self.flag = 'ft' #'simplevision'
        if self.flag == 'simplevision':
            self.simple_adapter = SimpleAdapter(output_dim)
        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.ml = multilayer
        if multilayer:
            self.lateral_conv1 = nn.Conv2d(width*4, width*8, kernel_size=1)
            self.lateral_conv2 = nn.Conv2d(width*8, width*16, kernel_size=1)
            self.lateral_conv3 = nn.Conv2d(width*16, width*32, kernel_size=1)
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, spatial)
            self.ml_attnpool = None
        else:
            self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim, spatial)
            self.ml_attnpool = None
    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        if not self.ml:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.attnpool(x)
        else:
            x1 = self.layer1(x)
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)

            # s2 = self.avgpool(self.lateral_conv1(x1)) + x2  #layer1+layer2
            # s3 = self.avgpool(self.lateral_conv2(s2)) + x3  #layer2+layer3
            s4 = self.avgpool(self.lateral_conv3(x3)) + x4  #layer3+layer4

            x = self.attnpool(torch.cat([x4, s4], dim=1))
        if self.flag == 'simplevision':
            x = self.simple_adapter(x)

        return x, x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16DATASET."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

# class ClassInstantier(OrderedDict):
#     def __getitem__(self, key):
#         content = super().__getitem__(key)
#         cls, kwargs = content if isinstance(content, tuple) else (content, {})
#         return cls(**kwargs)
#
#
# ACT2CLS = {
#     "gelu": GELUActivation,
#     "gelu_10": (ClippedGELUActivation, {"min": -10, "max": 10}),
#     "gelu_fast": FastGELUActivation,
#     "gelu_new": NewGELUActivation,
#     "gelu_python": (GELUActivation, {"use_gelu_python": True}),
#     "linear": LinearActivation,
#     "mish": MishActivation,
#     "quick_gelu": QuickGELUActivation,
#     "relu": nn.ReLU,
#     "relu6": nn.ReLU6,
#     "sigmoid": nn.Sigmoid,
#     "silu": SiLUActivation,
#     "swish": SiLUActivation,
#     "tanh": nn.Tanh,
# }
# ACT2FN = ClassInstantier(ACT2CLS)
#
# def get_activation(activation_string):
#     if activation_string in ACT2FN:
#         return ACT2FN[activation_string]
#     else:
#         raise KeyError(f"function {activation_string} not found in ACT2FN mapping {list(ACT2FN.keys())}")

class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            raise ValueError("Unknown nonlinear function type.")

    def forward(self, x):
        return self.f(x)

class Adapter(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.use_gating = False
        self.adapter_residual_before_ln = True
        self.add_layer_norm_after = True
        self.input_size = input_size
        self.add_layer_norm_before = False

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = None
        if self.down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class('LeakyRelU'.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        # if config["phm_layer"]:
        #     # Linear down projection of the input
        #     self.adapter_up = PHMLayer(adapter_name, self.down_sample, self.input_size, "up", config)
        # else:
        self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if self.use_gating:
            self.adapter_gate = nn.Linear(self.input_size, 1)

        # Additional scaling factor (from He et al. (2021))
        self.adapter_scaling = nn.Parameter(torch.ones(1))

        self.adapter_down.apply(self.init_bert_weights)
        self.adapter_up.apply(self.init_bert_weights)
        if self.use_gating:
            self.adapter_gate.apply(self.init_bert_weights)


    def forward(self, x, residual_input, output_gating=False):
        # print('x.shape', x.shape)
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.adapter_scaling
        output = up

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.adapter_gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)


        if self.use_gating and output_gating:
            return output, down, up, gate
        return output


    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class SimpleAdapter(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        emb_dim = input_size
        self.adapter_dim = '_dim_x2'
        if self.adapter_dim == '_dim_x2':
            hidden_dim = emb_dim * 2
        else:
            hidden_dim = emb_dim // 2

        self.linear1 = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(hidden_dim, emb_dim, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.FloatTensor([0.5]))


    def forward(self, x):
        # print('x.shape', x.shape)
        out = self.relu(self.linear1(x))
        out = self.linear2(out)
        out = self.alpha * out + (torch.FloatTensor([1.0]).cuda() - self.alpha) * x
        out = self.relu(out)
        # print('alpha: ', self.alpha)

        return out

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, flag=None, layer_id=0):
        super().__init__()
        self.layer_id = layer_id
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        self.flag = flag
        if self.flag == 'vision':
            self.attention_adapter = Adapter(d_model)
            self.output_adapter = Adapter(d_model)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        if self.flag == 'vision':
            if self.layer_id in [3,7,11]:
                residual = x
                x = self.attention(self.ln_1(x))
                ##Attention Adapter
                x = self.attention_adapter(x, residual)

                residual = x
                x = self.mlp(self.ln_2(x))
                ##Output Adapter
                x = self.output_adapter(x, residual)
            else:
                x = x + self.attention(self.ln_1(x))
                x = x + self.mlp(self.ln_2(x))

        elif self.flag == 'text' or self.flag == 'simplevision' or self.flag == 'ft':
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            raise RuntimeError(f"{self.flag} of Transformer is not supported.")

        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, flag=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, flag, k) for k in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, multilayer: bool):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)
        self.extract_layer = [7,11] #[0, 3, 7, 11]
        self.flag = 'ft' #'simplevision'
        if not multilayer:
            self.transformer = Transformer(width, layers, heads, flag=self.flag)
        else:
            self.transformer = MultiLevel_Transformer(Transformer(width, layers, heads, flag=self.flag), self.extract_layer)

        if self.flag == 'simplevision':
            self.simple_adapter = SimpleAdapter(width)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim)) #[768,512]
        if multilayer:
            self.ml_visu_proj = nn.Linear(width * len(self.extract_layer), output_dim) #[n*768,512]
            self.ml_ln_post = LayerNorm(width * len(self.extract_layer))
            self.ml_visu_proj.weight = nn.Parameter(torch.cat([self.proj for i in range(len(self.extract_layer))], dim=0).permute(1,0))
            self.ml_ln_post.weight = nn.Parameter(torch.cat([self.ln_post.weight for i in range(len(self.extract_layer))], dim=-1))

        # nn.init.normal_(self.proj)
        self.ml = multilayer

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)

        if not self.ml:
            x = x.permute(1, 0, 2)  # LND -> NLD
            x_pre = self.ln_post(x[:, 0, :])
            if self.flag == 'simplevision':
                x = self.simple_adapter(x_pre)

            if self.proj is not None:
                x = x_pre @ self.proj
        else:
            x = torch.cat(x, dim=2)  # LN4D
            x = x.permute(1, 0, 2)  # LN4D -> NL4D
            # print('NL4D: ', x.size())   #torch.Size([40, 197, 3072])
            x_pre = self.ml_ln_post(x[:, 0, :])   #[40, 1536]

            # multilayer_perceiver

            x = self.ml_visu_proj(x_pre.float())  # NL4D-->NLD
            # print('NLD: ', x.size())

            if self.flag == 'simplevision':
                x = self.simple_adapter(x)

        return x, x_pre

class MultiLevel_Transformer(nn.Module):
    def __init__(self, clip_vit, extract_layer):
        super().__init__()
        heads = clip_vit.width // 64
        self.width = clip_vit.width
        self.layers = clip_vit.layers
        self.resblocks = clip_vit.resblocks
        self.extract_layer = extract_layer

    def forward(self, x: torch.Tensor):
        ml_feature = []
        for i in range(max(self.extract_layer)+1):
            x = self.resblocks[i](x)
            if i in self.extract_layer:
                ml_feature.append(x)
        return ml_feature


class CLIP(nn.Module):
    def __init__(self,
                 cfg,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 spatial=False,
                 ):
        super().__init__()
        self.cfg = cfg
        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,  # (3, 4, 6, 3)
                output_dim=embed_dim,  # 1024
                heads=vision_heads,  # 32
                input_resolution=image_resolution,  # 224
                width=vision_width,  # 64
                spatial=spatial,
                multilayer = False #self.cfg.ML_Vision
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                multilayer=False #self.cfg.ML_Vision
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            flag = 'text'
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        # self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)
            if self.visual.ml_attnpool is not None:
                std = self.visual.ml_attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.ml_attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.ml_attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.ml_attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.ml_attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        token_embd = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = token_embd.type(self.dtype) + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # print(index)
        # full_x = x[:, :,:] @ self.text_projection #x[:, :index,:] @ self.text_projection  #TODO
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1), :] @ self.text_projection

        return x


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text


def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, spatial=False, cfg=None):
    vit = "visual.proj" in state_dict
    # print('state_dict.keys：', state_dict.keys())
    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(cfg,
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, spatial=spatial,
    )
    # print(model.state_dict()['visual.transformer.resblocks.0.attn.in_proj_weight'])
    # print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
    # print('model.keys:',model.state_dict().keys())
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # state_dict_keys = list(state_dict.keys())
    # for i in range(len(state_dict_keys)):
    #     if 'visual.transformer' in state_dict_keys[i]:
    #         new_key = state_dict_keys[i].replace('visual.transformer', 'visual.ml_transformer')
    #         state_dict[new_key] = state_dict.pop(state_dict_keys[i], None)

    # convert_weights(model)
    model.load_state_dict(state_dict, strict=False)
    # print(model.state_dict()['visual.transformer.resblocks.0.attn.in_proj_weight'])
    return model.eval()


class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=2):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=kernel_size, mode='bilinear', align_corners=True)
            self.conv = DoubleConv2(in_channels, out_channels, in_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=kernel_size, stride=kernel_size, groups=1)
            self.conv = DoubleConv2(in_channels, out_channels)

    def forward(self, x):
        x = self.up(x)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=8),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv2(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(mid_channels),
            # nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False, groups=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# MODEL_REGISTRY = Registry("Model")
# STEM_REGISTRY = Registry("Stem")
# BRANCH_REGISTRY = Registry("Branch")
# HEAD_REGISTRY = Registry("Head")
# HEAD_BACKBONE_REGISTRY = Registry("HeadBackbone")

class PreNormattention_qkv(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, q, k, v, **kwargs):
        q_norm = self.norm(q)
        k_norm = q_norm
        v_norm = k_norm
        output = self.fn(q_norm, k_norm, v_norm, **kwargs) + q
        return output

class Transformer_v1(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention_qkv(dim, Attention_qkv(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, q, k, v):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(q, k, v)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x, x, x)
                x = ff(x) + x
        return x

class Transformer_v2(nn.Module):
    def __init__(self, heads=8, dim=2048, dim_head_k=256, dim_head_v=256, dropout_atte = 0.05, mlp_dim=2048, dropout_ffn = 0.05, depth=1):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.depth = depth
        for _ in range(depth):
            self.layers.append(nn.ModuleList([  # PreNormattention(2048, Attention(2048, heads = 8, dim_head = 256, dropout = 0.2))
                # PreNormattention(heads, dim, dim_head_k, dim_head_v, dropout=dropout_atte),
                PreNormattention(dim, Attention(dim, heads = heads, dim_head = dim_head_k, dropout = dropout_atte)),
                FeedForward(dim, mlp_dim, dropout = dropout_ffn),
            ]))
    def forward(self, x):
        # if self.depth
        for attn, ff in self.layers[:1]:
            x = attn(x)
            x = ff(x) + x
        if self.depth > 1:
            for attn, ff in self.layers[1:]:
                x = attn(x)
                x = ff(x) + x
        return x


class PreNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x




class Attention_qkv(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_k = nn.Linear(dim, inner_dim, bias = False)
        self.to_v = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, q, k, v):
        b, n, _, h = *q.shape, self.heads
        bk = k.shape[0]
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        q = self.to_q(q)
        k = self.to_k(k)
        v = self.to_v(v)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', b=bk, h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', b=bk, h = h)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)    # [30, 8, 8, 5]

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class PostNormattention(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


def cos_sim(x, y, epsilon=0.01):
    """
    Calculates the cosine similarity between the last dimension of two tensors.
    """
    numerator = torch.matmul(x, y.transpose(-1,-2))
    xnorm = torch.norm(x, dim=-1).unsqueeze(-1)
    ynorm = torch.norm(y, dim=-1).unsqueeze(-1)
    denominator = torch.matmul(xnorm, ynorm.transpose(-1,-2)) + epsilon
    dists = torch.div(numerator, denominator)
    return dists


def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask, as_tuple=False)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector



class CNN_FSHead(nn.Module):
    """
    Base class which handles a few-shot method. Contains a resnet backbone which computes features.
    """
    def __init__(self, cfg):
        super(CNN_FSHead, self).__init__()
        args = cfg
        self.train()
        self.args = args

        last_layer_idx = -1
        
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet18":
            backbone = models.resnet18(pretrained=True) 
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet34":
            backbone = models.resnet34(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

        elif self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            backbone = models.resnet50(pretrained=True)
            self.backbone = nn.Sequential(*list(backbone.children())[:last_layer_idx])

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()
        target_features = self.backbone(target_images).squeeze()

        dim = int(support_features.shape[1])

        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)

        return support_features, target_features

    def forward(self, support_images, support_labels, target_images):
        """
        Should return a dict containing logits which are required for computing accuracy. Dict can also contain
        other info needed to compute the loss. E.g. inter class distances.
        """
        raise NotImplementedError

    def distribute_model(self):
        """
        Use to split the backbone evenly over all GPUs. Modify if you have other components
        """
        if self.args.TRAIN.DDP_GPU > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)])
    
    def loss(self, task_dict, model_dict):
        """
        Takes in a the task dict containing labels etc.
        Takes in the model output dict, which contains "logits", as well as any other info needed to compute the loss.
        Default is cross entropy loss.
        """
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
        
        


class PositionalEncoding(nn.Module):
    """
    Positional encoding from the Transformer paper.
    """
    def __init__(self, d_model, dropout, max_len=5000, pe_scale_factor=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.pe_scale_factor = pe_scale_factor
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term) * self.pe_scale_factor
        pe[:, 1::2] = torch.cos(position * div_term) * self.pe_scale_factor
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
                          
    def forward(self, x):
       x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
       return self.dropout(x)


@HEAD_REGISTRY.register()
class TemporalCrossTransformer(nn.Module):
    """
    A temporal cross transformer for a single tuple cardinality. E.g. pairs or triples.
    """
    def __init__(self, cfg, temporal_set_size=3):
        super(TemporalCrossTransformer, self).__init__()
        # temporal_set_size=3
        args = cfg
        
        self.args = args
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.trans_linear_in_dim = 2048
        else:
            self.trans_linear_in_dim = 512
        # self.trans_linear_in_dim = 2048
        self.temporal_set_size = temporal_set_size

        max_len = int(self.args.DATA.NUM_INPUT_FRAMES * 1.5)
        self.pe = PositionalEncoding(self.trans_linear_in_dim, self.args.trans_dropout, max_len=max_len)

        self.k_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()
        self.v_linear = nn.Linear(self.trans_linear_in_dim * temporal_set_size, self.args.trans_linear_out_dim)#.cuda()

        self.norm_k = nn.LayerNorm(self.args.trans_linear_out_dim)
        self.norm_v = nn.LayerNorm(self.args.trans_linear_out_dim)
        
        self.class_softmax = torch.nn.Softmax(dim=1)
        
        # generate all tuples
        frame_idxs = [i for i in range(self.args.DATA.NUM_INPUT_FRAMES)]
        frame_combinations = combinations(frame_idxs, temporal_set_size)
        self.tuples = nn.ParameterList([nn.Parameter(torch.tensor(comb), requires_grad=False) for comb in frame_combinations])
        self.tuples_len = len(self.tuples) 
    
    
    def forward(self, support_set, support_labels, queries):
        n_queries = queries.shape[0]   # [35, 8, 2048]
        n_support = support_set.shape[0]   # [5, 8, 2048]
        
        # static pe
        support_set = self.pe(support_set)   # [5, 8, 2048]
        queries = self.pe(queries)      # queries

        # construct new queries and support set made of tuples of images after pe
        s = [torch.index_select(support_set, -2, p).reshape(n_support, -1) for p in self.tuples]
        q = [torch.index_select(queries, -2, p).reshape(n_queries, -1) for p in self.tuples]
        support_set = torch.stack(s, dim=-2)   # [5, 28, 4096]
        queries = torch.stack(q, dim=-2)    # [35, 28, 4096]

        # apply linear maps
        support_set_ks = self.k_linear(support_set)   # [5, 28, 1152]
        queries_ks = self.k_linear(queries)           # [35, 28, 1152]
        support_set_vs = self.v_linear(support_set)   # [5, 28, 1152]
        queries_vs = self.v_linear(queries)           # [35, 28, 1152]
        
        # apply norms where necessary
        mh_support_set_ks = self.norm_k(support_set_ks)
        mh_queries_ks = self.norm_k(queries_ks)
        mh_support_set_vs = support_set_vs
        mh_queries_vs = queries_vs
        
        unique_labels = torch.unique(support_labels)   # [0., 1., 2., 3., 4.]

        # init tensor to hold distances between every support tuple and every target tuple
        all_distances_tensor = torch.zeros(n_queries, self.args.TRAIN.WAY, device=queries.device)   # [35, 5]

        for label_idx, c in enumerate(unique_labels):
        
            # select keys and values for just this class
            class_k = torch.index_select(mh_support_set_ks, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            class_v = torch.index_select(mh_support_set_vs, 0, extract_class_indices(support_labels, c))   # [1, 28, 1152]
            k_bs = class_k.shape[0]

            class_scores = torch.matmul(mh_queries_ks.unsqueeze(1), class_k.transpose(-2,-1)) / math.sqrt(self.args.trans_linear_out_dim)   # ([35, 1, 28, 1152], [1, 1152, 28]) --> [35, 1, 28, 28]

            # reshape etc. to apply a softmax for each query tuple
            class_scores = class_scores.permute(0,2,1,3)     # [35, 28, 1, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1)     # [35, 28, 28]
            class_scores = [self.class_softmax(class_scores[i]) for i in range(n_queries)]
            class_scores = torch.cat(class_scores)      # [980, 28]
            class_scores = class_scores.reshape(n_queries, self.tuples_len, -1, self.tuples_len)    # [35, 28, 1, 28]
            class_scores = class_scores.permute(0,2,1,3)       # [35, 1, 28, 28]
            
            # get query specific class prototype         
            query_prototype = torch.matmul(class_scores, class_v)    # [35, 1, 28, 1152]
            query_prototype = torch.sum(query_prototype, dim=1)      # [35, 1, 28, 1152]
            
            # calculate distances from queries to query-specific class prototypes
            diff = mh_queries_vs - query_prototype   # [35, 28, 1152]
            norm_sq = torch.norm(diff, dim=[-2,-1])**2    # [35]
            distance = torch.div(norm_sq, self.tuples_len)
            
            # multiply by -1 to get logits
            distance = distance * -1
            c_idx = c.long()
            all_distances_tensor[:,c_idx] = distance
        
        return_dict = {'logits': all_distances_tensor}
        
        return return_dict


@HEAD_REGISTRY.register()
class CNN_TRX(CNN_FSHead):
    """
    Backbone connected to Temporal Cross Transformers of multiple cardinalities.
    """
    def __init__(self, cfg):
        super(CNN_TRX, self).__init__(cfg)
        args = cfg
        #fill default args
        self.args.trans_linear_out_dim = 1152
        self.args.temp_set = [2,3]
        self.args.trans_dropout = 0.1

        self.transformers = nn.ModuleList([TemporalCrossTransformer(args, s) for s in args.temp_set]) 

    def forward(self, inputs):
        # support_images, support_labels, target_images = inputs
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)
        all_logits = [t(support_features, support_labels, target_features)['logits'] for t in self.transformers]
        all_logits = torch.stack(all_logits, dim=-1)
        sample_logits = all_logits 
        sample_logits = torch.mean(sample_logits, dim=[-1])

        return_dict = {'logits': sample_logits}
        return return_dict

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs. Leaves TRX on GPU 0.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.backbone.cuda(0)
            self.backbone = torch.nn.DataParallel(self.backbone, device_ids=[i for i in range(0, self.args.num_gpus)])

            self.transformers.cuda(0)





def OTAM_cum_dist(dists, lbda=0.1):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


@HEAD_REGISTRY.register()
class CNN_OTAM(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM, self).__init__(cfg)
        args = cfg
        self.args = cfg

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]

        support_features, target_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        cum_dists = OTAM_cum_dist(dists) + OTAM_cum_dist(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))


        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_CrossTransformer(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_CrossTransformer, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.dim = 2048
        # self.hidden_dim = 512  # v0 
        # self.hidden_dim = 2048   # v1
        self.hidden_dim = 1024   # v2
        self.way = cfg.TRAIN.WAY
        self.shot = cfg.TRAIN.SHOT
        self.key_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)
        self.query_head = self.key_head
        self.value_head = nn.Conv1d(self.dim, self.hidden_dim, 1, bias=False)

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        # support_images, support_labels, target_images = inputs

        support_features, query_image_features = self.get_feats(support_images, target_images)
        # [25, 8, 2048] [25, 8, 2048]

        unique_labels = torch.unique(support_labels)
        support_features = [torch.index_select(support_features, 0, extract_class_indices(support_labels, c)) for c in unique_labels]
        support_features = torch.cat(support_features, 0)  # [25, 8, 2048]
     
        query = self.query_head(query_image_features.permute(0,2,1))   # [25, 512, 8]
        support_key = self.key_head(support_features.permute(0,2,1))
        support_value = self.value_head(support_features.permute(0,2,1))

        ## flatten pixels & k-shot in support (j & m in the paper respectively)
        support_key = support_key.view(self.way, self.shot, support_key.shape[1], -1)
        support_value = support_value.view(self.way, self.shot, support_value.shape[1], -1)

        support_key = support_key.permute(0, 2, 3, 1)
        support_value = support_value.permute(0, 2, 3, 1)

        support_key = support_key.contiguous().view(self.way, support_key.shape[1], -1)   # [5, 512, 40]
        support_value = support_value.contiguous().view(self.way, support_value.shape[1], -1)

		## v is j images' m pixels, ie k-shot*h*w
        attn_weights = torch.einsum('bdp,ndv->bnpv', query, support_key) * (self.hidden_dim ** -0.5)   # [15, 5, 8, 40]
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
		
        ## get weighted sum of support values
        support_value = support_value.unsqueeze(0).expand(attn_weights.shape[0], -1, -1, -1)   # [15, 5, 512, 40]
        query_aligned_prototype = torch.einsum('bnpv,bndv->bnpd', attn_weights, support_value)   # [15, 5, 8, 512]

		### Step 3: Calculate query value
        query_value = self.value_head(query_image_features.permute(0,2,1)).permute(0,2,1)   # [25, 8, 512]
		# query_value = query_value.view(query_value.shape[0], -1, query_value.shape[1]) ##bpd
		
		### Step 4: Calculate distance between queries and supports
        distances = []
        for classid in range(query_aligned_prototype.shape[1]):
            support_features = rearrange(F.normalize(query_aligned_prototype[:, classid], dim=2), 'b s d -> b (s d)')     # [15, 4096]
            target_features = rearrange(F.normalize(query_value, dim=2), 'b s d -> b (s d)')      # [15, 4096]
            # dxc = torch.matmul(target_features, support_features.transpose(0,1))
            dxc = (target_features*support_features).sum(1)/8   # vo is no/8

			# dxc = torch.cdist(query_aligned_prototype[:, classid], 
			# 								query_value, p=2)
			# dxc = dxc**2
			# B,P,R = dxc.shape
			# dxc = dxc.sum(dim=(1,2)) / (P*R)
            distances.append(dxc)
		
        # class_dists = rearrange(class_dists, 'c q -> q c')
        class_dists = torch.stack(distances, dim=1)
        return_dict = {'logits': class_dists}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())
    


@HEAD_REGISTRY.register()
class CNN_TSN(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_TSN, self).__init__(cfg)
        args = cfg
        self.norm_sq_dist = False


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        support_features, target_features = self.get_feats(support_images, target_images)   # [25, 8, 2048] [25, 8, 2048]
        unique_labels = torch.unique(support_labels)

        support_features = torch.mean(support_features, dim=1)
        target_features = torch.mean(target_features, dim=1)

        if self.norm_sq_dist:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)
            
            diffs = [target_features - class_prototypes[i] for i in unique_labels]
            diffs = torch.stack(diffs)

            norm_sq = torch.norm(diffs, dim=[-1])**2
            distance = - rearrange(norm_sq, 'c q -> q c')
            return_dict = {'logits': distance}

        else:
            class_sim = cos_sim(target_features, support_features)
            class_sim = [torch.mean(torch.index_select(class_sim, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
            class_sim = torch.stack(class_sim)
            class_sim = rearrange(class_sim, 'c q -> q c')
            return_dict = {'logits': class_sim}

        return return_dict


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        log_attn = F.log_softmax(attn, 2)
        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)
        return output, attn, log_attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, q, k, v):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        
        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k) # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k) # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v) # (n*b) x lv x dv

        output, attn, log_attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1) # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
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



class PositionalEncoder(nn.Module):
    def __init__(self, d_model=2048, max_seq_len = 20, dropout = 0.1, A_scale=10., B_scale=1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.A_scale = A_scale
        self.B_scale = B_scale
        self.register_buffer('pe', pe)
 
    
    def forward(self, x):
        
        x = x * math.sqrt(self.d_model/self.A_scale)
        #add constant to embedding
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + self.B_scale * pe
        return self.dropout(x)


@HEAD_REGISTRY.register()
class CNN_HyRSM_1shot(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, cfg):
        super(CNN_HyRSM_1shot, self).__init__(cfg)
        last_layer_idx = -1
        self.args = cfg
        
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                   
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [35, 5, 8, 2048]  V1
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        support_features_ext = support_features.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
 
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists
        
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())





@HEAD_REGISTRY.register()
class CNN_HyRSM_5shot(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_5shot, self).__init__(cfg)
        args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        # self.relu1 = nn.ReLU(inplace=True)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY

        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        # feature_in = self.temporal_atte(feature_in, feature_in, feature_in)  # .view(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)) [35, 6, 2048]  45%
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        # frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        # calculate query -> support and support -> query
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 
        
        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits}
        return return_dict


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_1shot(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """

    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_1shot, self).__init__(cfg)
        last_layer_idx = -1
        self.args = cfg
        
        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
                
        self.temproal_regular_label = temproal_regular_label.cuda()
                

    def get_feats(self, support_images, target_images):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        # h_dim = int(support_features.shape[2])
        # w_dim = int(support_features.shape[3])

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [35, 5, 8, 2048]  V1
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        # class_logits = self.classification_layer(torch.cat([support_features.mean(1), target_features.mean(1)], 0))
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        support_features_ext = support_features.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        dim = support_features.shape[-1]
        # F.normalize(support_features, dim=2)

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        # frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query
        
        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(support_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_5shot(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_5shot, self).__init__(cfg)
        args = cfg
        # args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3":
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)

        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:            
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        

        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
                
        self.temproal_regular_label = temproal_regular_label.cuda()

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY

        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)
        dim = support_features.shape[-1]

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        # target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]   [280, 2048]
 
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query

        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        return return_dict


@HEAD_REGISTRY.register()
class CNN_HyRSM_plusplus_semi(CNN_FSHead):
    """
    TSN with a CNN backbone.
    Either cosine similarity or negative norm squared distance. 
    Use mean distance from query to class videos.
    """
    def __init__(self, cfg):
        super(CNN_HyRSM_plusplus_semi, self).__init__(cfg)
        args = cfg
        # args = cfg
        self.args = cfg
        self.norm_sq_dist = False
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50" or self.args.VIDEO.HEAD.BACKBONE_NAME == "inception_v3" :
            self.mid_dim = 2048
        else:
            self.mid_dim = 512
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        elif hasattr(self.args.TRAIN,"NO_POSITION"):
            self.pe = nn.Sequential()
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        
        last_layer_idx = -1
        
        self.relu = nn.ReLU(inplace=True)
        
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head = self.mid_dim//self.args.TRAIN.HEAD, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(self.args.TRAIN.HEAD, self.mid_dim, self.mid_dim//self.args.TRAIN.HEAD, self.mid_dim//self.args.TRAIN.HEAD, dropout=0.05)
        else:
            self.temporal_atte_before = PreNormattention(self.mid_dim, Attention(self.mid_dim, heads = 8, dim_head = self.mid_dim//8, dropout = 0.2))
            self.temporal_atte = MultiHeadAttention(8, self.mid_dim, self.mid_dim//8, self.mid_dim//8, dropout=0.05)
        
        self.layer2 = nn.Sequential(nn.Conv1d(self.mid_dim*2, self.mid_dim, kernel_size=1, padding=0),)
                                    
        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
        else:
            self.classification_layer = nn.Linear(self.mid_dim, 64)
        

        max_seq_len = self.args.DATA.NUM_INPUT_FRAMES
        temproal_regular = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular[i, j] = 1./(pow(i-j,2)+1.0)
                else:
                    temproal_regular[i, j] = 1.-torch.exp(torch.tensor(-pow(abs(i-j)-self.args.TRAIN.WINDOW_SIZE,2)/self.args.TRAIN.TEMPORAL_BALANCE))
        self.temproal_regular = temproal_regular.cuda()

        temproal_regular_label = torch.zeros(max_seq_len, max_seq_len)
        for i in range(max_seq_len):
            for j in range(max_seq_len):
                if abs(i-j)<=self.args.TRAIN.WINDOW_SIZE:
                    temproal_regular_label[i, j] = 1.0
            
        self.temproal_regular_label = temproal_regular_label.cuda()

    def get_feats(self, support_images, target_images, support_labels, support_images_unlabel=None, use_unlabel=False):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.backbone(support_images).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.backbone(target_images).squeeze()   # [200, 2048, 7, 7]
        if use_unlabel:
            support_unlabel_features = self.backbone(support_images_unlabel).squeeze()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        # Temporal
        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        if use_unlabel:
            Support_unlabel_num = support_unlabel_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES

        support_features = self.relu(self.temporal_atte_before(self.pe(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [25, 8, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))   # [15, 8, 2048]
        if use_unlabel:
            support_unlabel_features = self.relu(self.temporal_atte_before(self.pe(support_unlabel_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))))

        if hasattr(self.args.TRAIN, "NUM_CLASS"):
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
        else:
            class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)

        unique_labels = torch.unique(support_labels)
        QUERY_PER_CLASS = target_features.shape[0]//self.args.TRAIN.WAY
        

        # pseudo labeling for unlabeled data
        if use_unlabel:
            class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
            class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

            support_features_ext = class_prototypes.unsqueeze(0).repeat(Support_unlabel_num,1,1,1)
            support_unlabel_features_ext = support_unlabel_features.unsqueeze(1)
            
            feature_in = torch.cat([support_features_ext.mean(2), support_unlabel_features_ext.mean(2)], 1)
            # feature_in = self.temporal_atte(feature_in, feature_in, feature_in)  # .view(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)) [35, 6, 2048]  45%
            feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)).detach() 
            support_features_a = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
            support_features_a= self.layer2(support_features_a.permute(0,2,1)).permute(0,2,1).reshape(Support_unlabel_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_unlabel_features_a = self.layer2(torch.cat([support_unlabel_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)
            # unique_labels = torch.unique(support_labels)
            n_queries = support_unlabel_features_a.shape[0]
            n_support = support_features_a.shape[1]
            frame_num = support_features_a.shape[2]
            dim = support_features_a.shape[-1]
            support_features_a = rearrange(support_features_a, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
            frame_sim = torch.matmul(F.normalize(support_features_a, dim=2), F.normalize(support_unlabel_features_a, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
            frame_dists = 1 - frame_sim
            dists = frame_dists
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 
            class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
            class_dists = - torch.stack(class_dists).detach()   # [5, 35]
            class_dists = rearrange(class_dists, 'c q -> q c')
            pseudo_label_class = torch.softmax(class_dists/self.args.TRAIN.SEMI_TEMPORAL, dim=-1)
            max_probs_class, targets_u_class = torch.max(pseudo_label_class,dim=-1)
            mask_class = max_probs_class.ge(self.args.TRAIN.SEMI_THRESHOLD).float()
            index_class = torch.where(mask_class==1.)
            if torch.any(index_class[0].bool()):
                targets_u_class_refine = targets_u_class[index_class].float()
                support_unlabel_features_refine = support_unlabel_features[index_class]
                support_features = torch.cat([support_features, support_unlabel_features_refine], dim=0)
                support_labels = torch.cat([support_labels, targets_u_class_refine], dim=0)



        # update prototype
        class_prototypes = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        class_prototypes = torch.stack(class_prototypes)   # [5, 2048, 8]

        support_features_ext = class_prototypes.unsqueeze(0).repeat(Query_num,1,1,1)
        target_features_ext = target_features.unsqueeze(1)
        
        feature_in = torch.cat([support_features_ext.mean(2), target_features_ext.mean(2)], 1)
        feature_in = self.relu(self.temporal_atte(feature_in, feature_in, feature_in)) 
        support_features = torch.cat([support_features_ext, feature_in[:,:-1,:].unsqueeze(2).repeat(1,1,self.args.DATA.NUM_INPUT_FRAMES,1)], 3).reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim*2)
        support_features= self.layer2(support_features.permute(0,2,1)).permute(0,2,1).reshape(Query_num, -1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = self.layer2(torch.cat([target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), feature_in[:,-1,:].unsqueeze(1).repeat(1,self.args.DATA.NUM_INPUT_FRAMES,1)],2).permute(0,2,1)).permute(0,2,1)

        return support_features, target_features, class_logits


    def forward(self, inputs):
        # [200, 3, 224, 224] [4., 4., 0., 2., 0., 3., 3., 4., 3., 1., 4., 2., 2., 1., 1., 0., 2., 1., 1., 0., 3., 2., 0., 3., 4.]  [200, 3, 224, 224]
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        if 'target_set_weakly' in inputs:
            support_images_unlabel = inputs["target_set_weakly"]
            use_unlabel=True
        else:
            support_images_unlabel = None
            use_unlabel = False
        
        support_features, target_features, class_logits = self.get_feats(support_images, target_images, support_labels, support_images_unlabel, use_unlabel)
        # [35, 5, 8, 2048] [35, 8, 2048] [40, 64]
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[1]
        frame_num = support_features.shape[2]
        # F.normalize(support_features, dim=2)
        dim = support_features.shape[-1]

        support_features = rearrange(support_features, 'b h s d -> b (h s) d')  # [200, 2048] [35, 40, 2048]
        
        frame_sim = torch.matmul(F.normalize(support_features, dim=2), F.normalize(target_features, dim=2).permute(0,2,1)).reshape(n_queries, n_support, frame_num, frame_num)
        frame_dists = 1 - frame_sim
        dists = frame_dists

        frame_sim_support = torch.matmul(F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_support*n_queries, frame_num, frame_num)
        frame_dists_support = 1 - frame_sim_support

        frame_sim_query = torch.matmul(F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2), F.normalize(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), dim=2).permute(0,2,1)).reshape(n_queries, frame_num, frame_num)
        frame_dists_query = 1 - frame_sim_query

        if hasattr(self.args.TRAIN, "BALANCE_COEFFICIENT") and self.args.TRAIN.BALANCE_COEFFICIENT:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + self.args.TRAIN.BALANCE_COEFFICIENT*(1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        else:
            loss_temporal_regular = torch.mean(frame_dists_support*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_support)) + torch.mean(frame_dists_query*self.temproal_regular_label*self.temproal_regular + (1-self.temproal_regular_label)*F.relu(self.temproal_regular-frame_dists_query))
        
        cum_dists_regular = dists
        cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2) 

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)   # [5, 35]
        class_dists = rearrange(class_dists, 'c q -> q c')
        return_dict = {'logits': - class_dists, 'class_logits': class_logits, "loss_temporal_regular": loss_temporal_regular}
        return return_dict




@HEAD_REGISTRY.register()
class CNN_BiMHM_MoLo(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_BiMHM_MoLo, self).__init__(cfg)
        args = cfg
        self.args = cfg
        last_layer_idx = -1
        self.backbone = nn.Sequential(*list(self.backbone.children())[:last_layer_idx])
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            if hasattr(self.args.TRAIN,"TEMP_COFF") and self.args.TRAIN.TEMP_COFF:
                self.scale = self.args.TRAIN.TEMP_COFF
                self.scale_motion = self.args.TRAIN.TEMP_COFF
            else:
                self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale.data.fill_(1.0)

                self.scale_motion = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
                self.scale_motion.data.fill_(1.0)

        self.relu = nn.ReLU(inplace=True)
        self.relu1 = nn.ReLU(inplace=True)
        if self.args.VIDEO.HEAD.BACKBONE_NAME == "resnet50":
            self.mid_dim = 2048
            # self.mid_dim = 256
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(2048, 256, kernel_size=1, padding=0, groups=4)
        else:
            self.mid_dim = 512
            self.pre_reduce = nn.Sequential()
            # self.pre_reduce = nn.Conv2d(512, 512, kernel_size=1, padding=0)  # nn.Sequential()
        if hasattr(self.args.TRAIN,"POSITION_A") and hasattr(self.args.TRAIN,"POSITION_B"):
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=self.args.TRAIN.POSITION_A, B_scale=self.args.TRAIN.POSITION_B)
        else:
            self.pe = PositionalEncoder(d_model=self.mid_dim, dropout=0.1, A_scale=10., B_scale=1.)
        self.class_token = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        self.class_token_motion = nn.Parameter(torch.randn(1, 1, self.mid_dim))
        if hasattr(self.args.TRAIN,"HEAD") and self.args.TRAIN.HEAD:
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head_k = self.mid_dim//self.args.TRAIN.HEAD, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = self.args.TRAIN.HEAD, dim_head_k = self.mid_dim//self.args.TRAIN.HEAD, dropout_atte = 0.2)
            
        else:
            
            self.temporal_atte_before = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            self.temporal_atte_before_motion = Transformer_v2(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
            
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.factor = 8
        self.motion_reduce = nn.Conv3d(self.mid_dim, self.mid_dim//self.factor, kernel_size=(3,3,3), padding=(1,1,1), groups=1)
        self.motion_conv = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim//self.factor, kernel_size=3, padding=1, groups=1)
        self.motion_up = nn.Conv2d(self.mid_dim//self.factor, self.mid_dim, kernel_size=1, padding=0, groups=1)
        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                self.classification_layer = nn.Linear(self.mid_dim, int(self.args.TRAIN.NUM_CLASS))
            else:
                self.classification_layer = nn.Linear(self.mid_dim, 64)
        
        bilinear = True
        # factor = 2 if bilinear else 1
        factor = 1
        n_classes = 3
        self.up1 = Up2(self.mid_dim//self.factor, 128 // factor, bilinear, kernel_size=2)
        self.up2 = Up2(128, 32 // factor, bilinear, kernel_size=4)
        self.up3 = Up2(32, 16, bilinear, kernel_size=4)
        self.outc = OutConv(16, n_classes)
        # set_trace()
        
    

    def get_feats(self, support_images, target_images, support_labels):
        """
        Takes in images from the support set and query video and returns CNN features.
        """
        support_features = self.pre_reduce(self.backbone(support_images)).squeeze()  # [40, 2048, 7, 7] (5 way - 1 shot - 5 query)
        target_features = self.pre_reduce(self.backbone(target_images)).squeeze()   # [200, 2048, 7, 7]
        # set_trace()
        batch_s = int(support_features.shape[0])
        batch_t = int(target_features.shape[0])

        dim = int(support_features.shape[1])
        
        support_features_motion = self.motion_reduce(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)   # [40, 128, 7, 7]
        target_features_motion = self.motion_reduce(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim, 7, 7).permute(0,2,1,3,4)).permute(0,2,1,3,4).reshape(-1, dim//self.factor, 7, 7)
        support_features_motion_conv = self.motion_conv(support_features_motion)   # [40, 128, 7, 7]
        target_features_motion_conv = self.motion_conv(target_features_motion)
        support_features_motion = support_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - support_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        support_features_motion = support_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # support_features_motion = self.relu(self.motion_up(support_features_motion))

        target_features_motion = target_features_motion_conv.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,1:] - target_features_motion.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim//self.factor, 7, 7)[:,:-1]
        target_features_motion = target_features_motion.reshape(-1, dim//self.factor, 7, 7)
        # target_features_motion = self.relu(self.motion_up(target_features_motion))
        feature_motion_recons = torch.cat([support_features_motion, target_features_motion], dim=0)
        feature_motion_recons = self.up1(feature_motion_recons)
        feature_motion_recons = self.up2(feature_motion_recons)
        feature_motion_recons = self.up3(feature_motion_recons)
        
        feature_motion_recons = self.outc(feature_motion_recons)
        support_features_motion = self.relu(self.motion_up(support_features_motion))
        target_features_motion = self.relu(self.motion_up(target_features_motion))
        support_features_motion = self.avg_pool(support_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        target_features_motion = self.avg_pool(target_features_motion).squeeze().reshape(-1, self.args.DATA.NUM_INPUT_FRAMES-1, dim)
        support_bs = int(support_features_motion.shape[0])
        target_bs = int(target_features_motion.shape[0])
        support_features_motion = torch.cat((self.class_token_motion.expand(support_bs, -1, -1), support_features_motion), dim=1)
        target_features_motion = torch.cat((self.class_token_motion.expand(target_bs, -1, -1), target_features_motion), dim=1)
        target_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(target_features_motion)))   # [5, 9, 2048]
        support_features_motion = self.relu(self.temporal_atte_before_motion(self.pe(support_features_motion)))

        


        support_features = self.avg_pool(support_features).squeeze()
        target_features = self.avg_pool(target_features).squeeze()

        Query_num = target_features.shape[0]//self.args.DATA.NUM_INPUT_FRAMES
        support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
        # support_features = self.temporal_atte(support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)).unsqueeze(0).repeat(Query_num,1,1,1)   # [35, 5, 8, 2048]   V0
        # target_features = self.temporal_atte(target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim), target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim))# .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]
        support_bs = int(support_features.shape[0])
        target_bs = int(target_features.shape[0])
        support_features = torch.cat((self.class_token.expand(support_bs, -1, -1), support_features), dim=1)
        target_features = torch.cat((self.class_token.expand(target_bs, -1, -1), target_features), dim=1)
        support_features = self.relu(self.temporal_atte_before(self.pe(support_features)))   # [5, 9, 2048]
        target_features = self.relu(self.temporal_atte_before(self.pe(target_features)))   # .repeat(1,self.args.TRAIN.WAY,1,1)  # [35, 1, 8, 2048]

        if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:
            if hasattr(self.args.TRAIN, "NUM_CLASS"):
                if hasattr(self.args.TRAIN, "USE_LOCAL") and self.args.TRAIN.USE_LOCAL:
                    class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, int(self.args.TRAIN.NUM_CLASS))
                else:
                    class_logits = self.classification_layer(torch.cat([support_features.mean(1)+support_features_motion.mean(1), target_features.mean(1)+target_features_motion.mean(1)], 0))
            else:
                class_logits = self.classification_layer(torch.cat([support_features, target_features], 0)).reshape(-1, 64)
        else:
            class_logits = None
        

        unique_labels = torch.unique(support_labels)
        support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features = torch.stack(support_features)

        support_features_motion = [torch.mean(torch.index_select(support_features_motion, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
        support_features_motion = torch.stack(support_features_motion)
        
        return support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons

    def forward(self, inputs):
        support_images, support_labels, target_images = inputs['support_set'], inputs['support_labels'], inputs['target_set'] # [200, 3, 224, 224]
        # [200, 3, 84, 84]
        if self.training and hasattr(self.args.TRAIN, "USE_FLOW"):
            support_images_re = inputs["support_set_flow"].reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)[:,:(self.args.DATA.NUM_INPUT_FRAMES-1),:,:,:]
            target_images_re = inputs["target_set_flow"].reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)[:,:(self.args.DATA.NUM_INPUT_FRAMES-1),:,:,:]
            input_recons = torch.cat([support_images_re, target_images_re], dim=0).reshape(-1, 3, 224, 224)
        else:
            support_images_re = support_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            target_images_re = target_images.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES,3, 224, 224)
            # support_images, support_labels, target_images = inputs
            input_recons = torch.cat([support_images_re[:,1:,:]-support_images_re[:,:-1,:], target_images_re[:,1:,:]- target_images_re[:,:-1,:]], dim=0).reshape(-1, 3, 224, 224)

        support_features, target_features, class_logits, support_features_motion, target_features_motion, feature_motion_recons = self.get_feats(support_images, target_images, support_labels)
        
        # 
        unique_labels = torch.unique(support_labels)

        n_queries = target_features.shape[0]
        n_support = support_features.shape[0]

        # global
        support_features_g = support_features[:,0,:]
        target_features_g = target_features[:,0,:]
        support_features = support_features[:,1:,:]
        target_features = target_features[:,1:,:]

        # support to query
        class_sim_s2q = cos_sim(support_features, target_features_g)  # [5, 8, 35]
        class_dists_s2q = 1 - class_sim_s2q
        class_dists_s2q = [torch.sum(torch.index_select(class_dists_s2q, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q = torch.stack(class_dists_s2q).squeeze(1)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_s2q = rearrange(class_dists_s2q * self.scale, 'c q -> q c')

        # query to support 
        class_sim_q2s = cos_sim(target_features, support_features_g)  # [35, 8, 5]
        class_dists_q2s = 1 - class_sim_q2s   
        class_dists_q2s = [torch.sum(torch.index_select(class_dists_q2s, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s = torch.stack(class_dists_q2s).squeeze(2)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_q2s = rearrange(class_dists_q2s * self.scale, 'c q -> q c')

        # global
        support_features_motion_g = support_features_motion[:,0,:]
        target_features_motion_g = target_features_motion[:,0,:]
        support_features_motion = support_features_motion[:,1:,:]
        target_features_motion = target_features_motion[:,1:,:]

        # support to query
        class_sim_s2q_motion = cos_sim(support_features_motion, target_features_motion_g)  # [5, 8, 35]
        class_dists_s2q_motion = 1 - class_sim_s2q_motion
        class_dists_s2q_motion = [torch.sum(torch.index_select(class_dists_s2q_motion, 0, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_s2q_motion = torch.stack(class_dists_s2q_motion).squeeze(1)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_s2q_motion = rearrange(class_dists_s2q_motion * self.scale_motion, 'c q -> q c')

        # query to support 
        class_sim_q2s_motion = cos_sim(target_features_motion, support_features_motion_g)  # [35, 8, 5]
        class_dists_q2s_motion = 1 - class_sim_q2s_motion   
        class_dists_q2s_motion = [torch.sum(torch.index_select(class_dists_q2s_motion, 2, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_q2s_motion = torch.stack(class_dists_q2s_motion).squeeze(2)
        if hasattr(self.args.TRAIN,"USE_CONTRASTIVE") and self.args.TRAIN.USE_CONTRASTIVE:
            class_dists_q2s_motion = rearrange(class_dists_q2s_motion * self.scale_motion, 'c q -> q c')

        support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
        target_features = rearrange(target_features, 'b s d -> (b s) d')    # [200, 2048]

        frame_sim = cos_sim(target_features, support_features)    # [200, 200]
        frame_dists = 1 - frame_sim
        
        dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

        # calculate query -> support and support -> query
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            cum_dists = dists.min(3)[0].sum(2)
        else:
            cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
            

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')

        support_features_motion = rearrange(support_features_motion, 'b s d -> (b s) d')  # [200, 2048]
        target_features_motion = rearrange(target_features_motion, 'b s d -> (b s) d')    # [200, 2048]
        frame_sim_motion = cos_sim(target_features_motion, support_features_motion)    # [200, 200]
        frame_dists_motion = 1 - frame_sim_motion   
        dists_motion = rearrange(frame_dists_motion, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]
        # calculate query -> support and support -> query
        if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
            # cum_dists_motion = OTAM_cum_dist(dists_motion)
            cum_dists_motion = dists_motion.min(3)[0].sum(2)
        else:
            cum_dists_motion = dists_motion.min(3)[0].sum(2) + dists_motion.min(2)[0].sum(2)
        class_dists_motion = [torch.mean(torch.index_select(cum_dists_motion, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists_motion = torch.stack(class_dists_motion)
        class_dists_motion = rearrange(class_dists_motion, 'c q -> q c')
        
        if hasattr(self.args.TRAIN, "LOGIT_BALANCE_COFF") and self.args.TRAIN.LOGIT_BALANCE_COFF:
            class_dists = class_dists + self.args.TRAIN.LOGIT_BALANCE_COFF*class_dists_motion
        else:
            class_dists = class_dists + 0.3*class_dists_motion
        
        if self.training:
            loss_recons = (feature_motion_recons - input_recons) ** 2   # [280, 3, 224, 224]
            loss_recons = loss_recons.mean()  # [N, L], mean loss per patch
        else:
            loss_recons = torch.tensor(0.)

        
        return_dict = {'logits': - class_dists , 'class_logits': class_logits, "logits_s2q": -class_dists_s2q, "logits_q2s": -class_dists_q2s, "logits_s2q_motion": -class_dists_s2q_motion, "logits_q2s_motion": -class_dists_q2s_motion, "loss_recons": loss_recons,}
        return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


def OTAM_cum_dist_v2(dists, lbda=0.5):
    """
    Calculates the OTAM distances for sequences in one direction (e.g. query to support).
    :input: Tensor with frame similarity scores of shape [n_queries, n_support, query_seq_len, support_seq_len] 
    TODO: clearn up if possible - currently messy to work with pt1.8. Possibly due to stack operation?
    """
    dists = F.pad(dists, (1,1), 'constant', 0)  # [25, 25, 8, 10]

    cum_dists = torch.zeros(dists.shape, device=dists.device)

    # top row
    for m in range(1, dists.shape[3]):
        # cum_dists[:,:,0,m] = dists[:,:,0,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,0,m-1]))
        # paper does continuous relaxation of the cum_dists entry, but it trains faster without, so using the simpler version for now:
        cum_dists[:,:,0,m] = dists[:,:,0,m] + cum_dists[:,:,0,m-1] 


    # remaining rows
    for l in range(1,dists.shape[2]):
        #first non-zero column
        cum_dists[:,:,l,1] = dists[:,:,l,1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,0] / lbda) + torch.exp(- cum_dists[:,:,l-1,1] / lbda) + torch.exp(- cum_dists[:,:,l,0] / lbda) )
        
        #middle columns
        for m in range(2,dists.shape[3]-1):
            cum_dists[:,:,l,m] = dists[:,:,l,m] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,m-1] / lbda) + torch.exp(- cum_dists[:,:,l,m-1] / lbda ) )
            
        #last column
        #cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
        cum_dists[:,:,l,-1] = dists[:,:,l,-1] - lbda * torch.log( torch.exp(- cum_dists[:,:,l-1,-2] / lbda) + torch.exp(- cum_dists[:,:,l-1,-1] / lbda) + torch.exp(- cum_dists[:,:,l,-2] / lbda) )
    
    return cum_dists[:,:,-1,-1]


# cfg = Config(load=True)
# backbone=None
# preprocess=None
# if cfg.TRAIN.BACKBONE_FROZEN:
#     torch.cuda.empty_cache()
#     if cfg.VIDEO.HEAD.BACKBONE_NAME == "RN50":
#         backbone, preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda:0", cfg=cfg, jit=False,
#                                          download_root='/mnt/hdd/fpeng/CLIP/models/')  # ViT-B/16
#         if cfg.NUM_GPUS > 1:
#             backbone = torch.nn.DataParallel(backbone, device_ids=[i for i in range(0, cfg.NUM_GPUS)])
#
#     elif cfg.VIDEO.HEAD.BACKBONE_NAME=="ViT-B/16":
#         backbone, preprocess = load(cfg.VIDEO.HEAD.BACKBONE_NAME, device="cuda:0", cfg=cfg, jit=False, download_root='/mnt/hdd/fpeng/CLIP/models/')   # ViT-B/16
#         if cfg.NUM_GPUS > 1:
#             backbone = torch.nn.DataParallel(backbone, device_ids=[i for i in range(0, cfg.NUM_GPUS)])


    #
    # def distribute_model(self):
    #     """
    #     Use to split the backbone evenly over all GPUs. Modify if you have other components
    #     """
    #     if self.args.TRAIN.DDP_GPU > 1:
    #         self.backbone.cuda(0)
    #         self.backbone = torch.nn.DataParallel(self.backbone,
    #                                               device_ids=[i for i in range(0, self.args.TRAIN.DDP_GPU)])


# class PromptLearner(nn.Module):
#     def __init__(self, cfg, classnames, clip_model):
#         super().__init__()
#
#     def forward(self):

ENSEMBLE_TEMPLATES = [
    "a photo of",
    "there is somebody",
    "an action of",
]

IMAGENET_TEMPLATES_SELECT = [
    "itap of a",
    "a bad photo of the",
    "a origami",
    "a photo of the large",
    "art of the",
    "a photo of the small",
]


def schedule(time):
    t = 0.3175604292915215 + time * 1.2332345639299847
    return t.sin(), t.cos()

class Combine(torch.nn.Module):
    def __init__(self):
        super().__init__()
        t = torch.linspace(0.0, 6.907755278982137, 4).exp()
        t *= 2
        t *= 3.141592653589793
        self.register_buffer('t', t)
        self.upsample = torch.nn.Upsample(size=(512), mode='nearest')  #torch.nn.UpsamplingNearest2d(size=(64, 64))
        # self.cnn = torch.nn.Conv2d(3, 32, kernel_size=1, stride=1, padding=0)

    def get_var(self, var):
        var = self.t * var
        var = torch.cat((var.sin(), var.cos()), dim=1)
        var = var.unsqueeze(dim=-1)
        var = self.upsample(var)
        return var

    def forward(self, var):
        var = var.squeeze(dim=-1).squeeze(dim=-1)
        var = self.get_var(var)
        # combine = torch.cat((image, var), dim=1)
        return var

class MLPDiffusion(nn.Module):
    def __init__(self, decod_ln, dec, t_embedding_layer): #——n_steps = 20,100
        super().__init__()
        self.decod_ln = decod_ln
        self.dec = dec
        self.t_embedding_layer = t_embedding_layer

    def forward(self, x, t):
        x = self.decod_ln(x)
        x = self.dec[0](x)
        x += self.t_embedding_layer(t)
        x = self.dec[1](x)
        x = self.dec[2](x)
        return x

num_steps = 1000
ddim_steps = 20
betas = torch.linspace(-6,6,num_steps)
betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

#计算alpha、alpha_prod、alpha_prod_previous、alpha_bar_sqrt等变量的值
alphas = 1-betas
# alpha连乘
alphas_prod = torch.cumprod(alphas,0)
#从第一项开始，第0项另乘1
alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
# alphas_prod开根号
alphas_bar_sqrt = torch.sqrt(alphas_prod) ##TODO
#之后公式中要用的
one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
###
sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_prod)
sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_prod - 1)
# 大小都一样，常数不需要训练
assert alphas.shape==alphas_prod.shape==alphas_prod_p.shape==\
alphas_bar_sqrt.shape==one_minus_alphas_bar_log.shape\
==one_minus_alphas_bar_sqrt.shape
def diffusion_loss_fn(x_0, alphas_bar_sqrt, one_minus_alphas_bar_sqrt, n_steps):
    """对任意时刻t进行采样计算loss"""
    batch_size = x_0.shape[0]
    # 对一个batchsize样本生成随机的时刻t，t变得随机分散一些，一个batch size里面覆盖更多的t
    t = torch.randint(0, n_steps, size=(batch_size//2,))
    if batch_size%2 != 0:
        t0 = torch.randint(0, n_steps, size=(1,))
    t = torch.cat([t, n_steps - 1 - t, t0], dim=0)# t的形状（bz）
    t = t.unsqueeze(-1)# t的形状（bz,1）
    # x0的系数，根号下(alpha_bar_t)
    a = alphas_bar_sqrt[t]
    # eps的系数,根号下(1-alpha_bar_t)
    aml = one_minus_alphas_bar_sqrt[t]
    # 生成随机噪音eps
    e = torch.randn_like(x_0)
    # 构造模型的输入
    x = x_0 * a.unsqueeze(-1).cuda() + e * aml.unsqueeze(-1).cuda()
    t = t.squeeze(-1).cuda()
    e = e.cuda()
    return e, x, t
    # 送入模型，得到t时刻的随机噪声预测值
    # output = model(x, t.squeeze(-1))
    # 与真实噪声一起计算误差，求平均值
    # return (e - output).square().mean()

def predict_start_from_noise(x_t, t, noise):
    t = torch.tensor([t]).unsqueeze(-1)
    output = sqrt_recip_alphas_cumprod[t].unsqueeze(-1).cuda() * x_t - \
             sqrt_recipm1_alphas_cumprod[t].unsqueeze(-1).cuda() * noise
    return output

def identity(t, *args, **kwargs):
    return t

def q_x(x_0, t):
    """可以基于x[0]得到任意时刻t的x[t]"""
    #生成正态分布采样
    noise = torch.randn_like(x_0)
    #得到均值方差，根据时间步选择alphas_bar_sqrt值和one_minus_alphas_bar_sqrt值
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    #根据x0求xt
    return (alphas_t * x_0 + alphas_1_m_t * noise)  # 在x[0]的基础上添加噪声

def ddim_sample(model, shape, y_cond= None):
    batch, total_timesteps, sampling_timesteps, eta, objective = shape[0],  num_steps, 20, 0., "pred_noise"
    self_condition = False
    times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
    times = list(reversed(times.int().tolist()))
    time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

    cur_x = torch.randn(shape).cuda()
    x_start = None
    clip_x_start=False
    for time, time_next in time_pairs:
        time_cond = torch.full((batch,), time, dtype=torch.long)
        self_cond = x_start if self_condition else None
        _, pred_noise = p_sample(model, cur_x, time_cond, betas, one_minus_alphas_bar_sqrt, y_cond)

        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        x_start = predict_start_from_noise(cur_x, time, pred_noise)
        x_start = maybe_clip(x_start)

        if time_next < 0:
            cur_x = x_start
            continue

        alpha = alphas_prod[time].cuda()
        alpha_next = alphas_prod[time_next].cuda()

        sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()

        noise = torch.randn_like(cur_x).cuda()

        cur_x = x_start * alpha_next.sqrt() + \
              c * pred_noise + \
              sigma * noise

    return cur_x

def p_sample(model, x, t, betas, one_minus_alphas_bar_sqrt, y_cond=None):
    """从x[T]采样t时刻的重构值"""
    if ddim_steps == 0:
        t = torch.tensor([t])
    t = t.unsqueeze(-1)
    coeff = betas[t] / one_minus_alphas_bar_sqrt[t]
    eps_theta = model(x, t.squeeze(-1).cuda(), x_self_cond=False, y=y_cond)
    #得到均值
    mean = (1 / (1 - betas[t]).sqrt()).cuda().unsqueeze(-1) * (x - (coeff.cuda().unsqueeze(-1) * eps_theta))
    z = torch.randn_like(x)
    sigma_t = betas[t].sqrt().cuda()
    #得到sample的分布
    sample = mean + sigma_t.unsqueeze(-1) * z
    return sample, eps_theta

def p_sample_loop(model, shape, n_steps, betas, one_minus_alphas_bar_sqrt, y_cond=None):
    """从x[T]恢复x[T-1]、x[T-2]|...x[0]"""
    cur_x = torch.randn(shape).cuda()
    for i in reversed(range(n_steps)):
        cur_x, _ = p_sample(model, cur_x, i, betas, one_minus_alphas_bar_sqrt, y_cond)
        # torch.cuda.empty_cache()
    return cur_x


@HEAD_REGISTRY.register()
class CNN_OTAM_CLIPFSAR(CNN_FSHead):
    """
    OTAM with a CNN backbone.
    """
    def __init__(self, cfg):
        super(CNN_OTAM_CLIPFSAR, self).__init__(cfg)
        args = cfg
        self.args = cfg
        if cfg.VIDEO.HEAD.BACKBONE_NAME=="RN50":
            # if self.args.TRAIN.BACKBONE_FROZEN:
            #     self.preprocess = preprocess
            # else:
            backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False,
                                            download_root='/mnt/hdd/fpeng/CLIP/models/')  # ViT-B/16

            self.backbone = backbone.visual    # model.load_state_dict(state_dict)
            self.backbone_proj = self.backbone.proj
            if cfg.TRAIN.Frozen_Cliploss:
                self.initial_backbone = None #backbone.visual #backbone.visual
            else:
                self.initial_backbone = None #backbone.visual
            self.transformer = backbone.transformer
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            if self.args.TRAIN.FeaturePreproj:
                self.mid_dim = 1024
            else:
                self.mid_dim = 1024
            self.text_dim = self.transformer.width

        elif cfg.VIDEO.HEAD.BACKBONE_NAME=="ViT-B/16":
            # if self.args.TRAIN.BACKBONE_FROZEN:
            #     self.preprocess = preprocess
            # else:
            backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False,
                                        download_root='/mnt/hdd/fpeng/CLIP/models/')  # ViT-B/16

            self.backbone = backbone.visual   # model.load_state_dict(state_dict)
            self.backbone_proj = self.backbone.proj
            if cfg.TRAIN.Frozen_Cliploss:
                self.initial_backbone = None #backbone.visual
            else:
                self.initial_backbone = None #backbone.vsual
            self.transformer = backbone.transformer
            self.class_real_train = cfg.TRAIN.CLASS_NAME
            self.class_real_test = cfg.TEST.CLASS_NAME
            # backbone, self.preprocess = load("RN50", device="cuda", cfg=cfg, jit=False)
            # self.backbone = backbone.visual model.load_state_dict(state_dict)
            # self.backbone = CLIP
            if self.args.TRAIN.FeaturePreproj:
                self.mid_dim = 512
            else:
                self.mid_dim = 512

            self.text_dim = self.transformer.width
            # print('self.text_dim',self.text_dim)


        self.dtype = backbone.dtype
        self.prompt_ln_final = backbone.ln_final
        self.prompt_projection = backbone.text_projection
        self.prompt_positional_embedding = backbone.positional_embedding

        #TODO self.prompt_learner = PromptLearner(cfg, backbone)
        with torch.no_grad():
            if hasattr(self.args.TRAIN, "Ensemble") and self.args.TRAIN.Ensemble:
                templates = ENSEMBLE_TEMPLATES
            elif hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                templates = ["a photo of"]
            else:
                templates = ["a photo of"]

            num_temp = len(templates)
            print(f"Prompt ensembling (n={num_temp})")
            n_ctx = len(max(templates, key=len, default='').split(" "))
            print('n_ctx', n_ctx)
            mean_embedding_train = 0
            mean_embedding_test = 0
            ind_tr = []
            ind_te = []

            if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                import json
                f = open(self.args.TEST.PROMPT, 'r')
                actions_detail = json.load(f)
                f.close()

            for i, temp in enumerate(templates):
                if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
                    text_templete_train = ["a photo of {}: ".format(self.class_real_train[int(ii)]) + actions_detail[
                        self.class_real_train[int(ii)].replace('_', ' ').replace('A person is ','').replace('a person is ', '')] for ii in range(len(self.class_real_train))]
                    text_templete_test = ["a photo of {}: ".format(self.class_real_test[int(ii)]) + actions_detail[
                        self.class_real_test[int(ii)].replace('_', ' ').replace('A person is ','').replace('a person is ', '')] for ii in range(len(self.class_real_test))]
                else:
                    text_templete_train = [temp.replace("_", " ")+ " " + self.class_real_train[int(ii)].replace("_", " ") for ii in range(len(self.class_real_train))]
                    text_templete_test= [temp.replace("_", " ") + " " + self.class_real_test[int(ii)].replace("_", " ") for ii in
                                           range(len(self.class_real_test))]
                #
                text_templete_train = tokenize(text_templete_train).cuda()
                embedding_x = backbone.token_embedding(text_templete_train)
                mean_embedding_train = mean_embedding_train + embedding_x
                ind_tr.append(text_templete_train.argmax(dim=-1).unsqueeze(0)) #[n]
                #
                text_templete_test = tokenize(text_templete_test).cuda()
                embedding_y = backbone.token_embedding(text_templete_test)
                mean_embedding_test = mean_embedding_test + embedding_y
                ind_te.append(text_templete_test.argmax(dim=-1).unsqueeze(0))

            mean_x_embedding = mean_embedding_train / num_temp
            mean_y_embedding = mean_embedding_test / num_temp
            ind_tr = torch.max(torch.cat(ind_tr, dim=0),dim=0)[0]  #[i,n] -->[n]
            ind_te = torch.max(torch.cat(ind_te, dim=0),dim=0)[0]  #[i,n] -->[n]

            # print('ind_tr.shape()',ind_tr.shape)

        self.register_buffer("token_prefix_x", mean_x_embedding[:, :1, :])  # SOS
        self.register_buffer("token_prefix_y", mean_y_embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix_x", mean_x_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        self.register_buffer("token_suffix_y", mean_y_embedding[:, 1 + n_ctx:, :])  # CLS, EOS
        if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
            classnames_x = [self.class_real_train[int(ii)].replace("_", " ") + ": " + actions_detail[self.class_real_train[int(ii)].replace('_', ' ').replace('A person is ','').replace('a person is ', '')] for ii in
                            range(len(self.class_real_train))]
            classnames_y = [self.class_real_test[int(ii)].replace("_", " ") + ": " + actions_detail[
                self.class_real_test[int(ii)].replace('_', ' ').replace('A person is ', '').replace('a person is ', '')]
                            for ii in range(len(self.class_real_test))]
        else:
            classnames_x = [self.class_real_train[int(ii)].replace("_", " ") for ii in range(len(self.class_real_train))]
            classnames_y = [self.class_real_test[int(ii)].replace("_", " ") for ii in
                            range(len(self.class_real_test))]
        name_lens_x = [len(tokenize(name)) for name in classnames_x]
        name_lens_y = [len(tokenize(name)) for name in classnames_y]
        self.register_buffer("token_eos_y", mean_y_embedding[0, ind_te[0], :])
        self.register_buffer("token_eos_x", mean_x_embedding[0, ind_tr[0], :])

        if not hasattr(self.args.TRAIN, "TePrompt") or not self.args.TRAIN.TePrompt:
            with torch.no_grad():
                if isinstance(backbone,torch.nn.DataParallel):
                    back_bone =backbone.module
                    back_bone=back_bone.cuda()
                    self.text_features_train, self.text_features_train_full, _ = self.text_encoder(mean_x_embedding, ind_tr)
                    self.text_features_test, self.text_features_test_full, _ = self.text_encoder(
                        mean_y_embedding, ind_te)  # [CLASS,D]

                else:
                    self.text_features_train, self.text_features_train_full, _ = self.text_encoder(mean_x_embedding, ind_tr)
                    self.text_features_test, self.text_features_test_full, _ = self.text_encoder(mean_y_embedding, ind_te)  # [CLASS,D]

            # TODO self.prompt_learner

        else:
            ctx_vectors = mean_x_embedding[0, 1: 1 + n_ctx, :]  #[n_ctx, dim]
            self.ctx = nn.Parameter(ctx_vectors)

            prompts_x = []
            prompts_y = []
            prefix_x = self.token_prefix_x
            prefix_y = self.token_prefix_y
            suffix_x = self.token_suffix_x
            suffix_y = self.token_suffix_y
            ctx = self.ctx
            for i in range(len(classnames_x)):
                name_len = name_lens_x[i]
                prefix_i = prefix_x[i: i + 1, :, :]
                ctx_i = ctx.unsqueeze(0)
                class_i = suffix_x[i: i + 1, :name_len, :]
                suffix_i = suffix_x[i: i + 1, name_len:, :]

                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        class_i,  # (1, name_len, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts_x.append(prompt)
            prompts_x = torch.cat(prompts_x, dim=0)   ##[n_cls, textlen, dim]

            for i in range(len(classnames_y)):
                name_len = name_lens_y[i]
                prefix_i = prefix_y[i: i + 1, :, :]
                ctx_i = ctx.unsqueeze(0)
                class_i = suffix_y[i: i + 1, :name_len, :]
                suffix_i = suffix_y[i: i + 1, name_len:, :]

                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        ctx_i,  # (1, n_ctx, dim)
                        class_i,  # (1, name_len, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts_y.append(prompt)
            prompts_y = torch.cat(prompts_y, dim=0)   ##[n_cls, textlen, dim]

            self.prompts_y = prompts_y
            self.prompts_x =prompts_x
            self.ind_tr = ind_tr
            self.ind_te = ind_te

            # #sub actions
            # SUBACTS_SUM = 3
            # if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
            #     import json
            #     f = open(self.args.TEST.PROMPT, 'r')
            #     subactions = json.load(f)
            #     f.close()
            #     if not self.args.SUBACTS: #子动作文本合并成一条
            #         text_templete = ["a photo of {}.".format(self.class_real_train[int(ii)]) +'\n\n' +subactions[self.class_real_train[int(ii)].replace('_',' ')] for ii in range(len(self.class_real_train))]
            #         print(text_templete)
            #     else:    #子动作文本分开成3条
            #         text_templete=[]
            #         text_templete.append(["a photo of {}.".format(self.class_real_train[int(ii)]) for ii in
            #                               range(len(self.class_real_train))])
            #         for i in range(SUBACTS_SUM):
            #             text_templete.append([subactions[self.class_real_train[int(ii)].replace('_', ' ')].split('\n\n')[i] for ii in
            #                          range(len(self.class_real_train))])  #[4, CLASS]
            #         print(text_templete)
            # else:  #原版文本，只有动作名
            #     text_templete = ["a photo of {}.".format(self.class_real_train[int(ii)]) for ii in range(len(self.class_real_train))]
            #
            # if not self.args.SUBACTS:  #text特征[CLASS, D]
            #     text_templete = tokenize(text_templete, truncate=True).cuda()
            #     self.text_features_train = backbone.encode_text(text_templete)
            # else:   #text特征
            #     text_features_train = []
            #     for i in range(SUBACTS_SUM+1):
            #         text_temple = tokenize(text_templete[i], truncate=True).cuda()
            #         text_features_train.append(backbone.encode_text(text_temple))
            #     self.text_features_train = torch.cat(text_features_train, dim=-1)  #[CLASS, (D * 4)]
            #
            #
            # if hasattr(self.args.TEST, "PROMPT") and self.args.TEST.PROMPT:
            #     import json
            #     f = open(self.args.TEST.PROMPT, 'r')
            #     subactions = json.load(f)
            #     f.close()
            #     if not self.args.SUBACTS:
            #         text_templete = ["a photo of {}.".format(self.class_real_test[int(ii)]) + '\n\n' + subactions[self.class_real_test[int(ii)].replace('_', ' ')] for ii in
            #                          range(len(self.class_real_test))]
            #         print(text_templete)
            #     else:
            #         text_templete=[]
            #         text_templete.append(["a photo of {}.".format(self.class_real_test[int(ii)]) for ii in
            #                               range(len(self.class_real_test))])
            #         for i in range(SUBACTS_SUM):
            #             text_templete.append([subactions[self.class_real_test[int(ii)].replace('_', ' ')].split('\n\n')[i] for ii in
            #                          range(len(self.class_real_test))])
            #         print(text_templete)
            # else:
            #     text_templete = ["a photo of {}.".format(self.class_real_test[int(ii)]) for ii in
            #                          range(len(self.class_real_test))]
            #
            # if not self.args.SUBACTS:
            #     text_templete = tokenize(text_templete, truncate=True).cuda()
            #     self.text_features_test = backbone.encode_text(text_templete)
            # else:
            #     text_features_test = []
            #     for i in range(SUBACTS_SUM+1):
            #         text_temple = tokenize(text_templete[i], truncate=True).cuda()
            #         text_features_test.append(backbone.encode_text(text_temple))
            #     self.text_features_test = torch.cat(text_features_test, dim=-1)  #[CLASS,(D*SUBACTS+1)]

        self.mid_layer = nn.Sequential() 
        self.classification_layer = nn.Sequential() 
        self.scale = nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.scale.data.fill_(1.0)
        self.n_steps = 1000  #100

        
        if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
        else:
            self.context2 = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
        # set_trace()
        if self.args.TRAIN.TemPrompt:
            # if hasattr(self.args.TRAIN, "TRANSFORMER_DEPTH") and self.args.TRAIN.TRANSFORMER_DEPTH:
            #     self.tlm = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2, depth=int(self.args.TRAIN.TRANSFORMER_DEPTH))
            # else:
            self.tlm = Transformer_v1(dim=self.mid_dim, heads = 8, dim_head_k = self.mid_dim//8, dropout_atte = 0.2)
        if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:
            self.decod_ln = LayerNorm(self.mid_dim)
            self.dec = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(self.mid_dim, self.mid_dim * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(self.mid_dim * 4, self.prompt_positional_embedding.shape[-1]))
        ]))
            if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                self.cross_attn1 = CrossAttention(
                    query_dim=self.mid_dim,
                    cross_attention_dim=self.mid_dim,
                    heads=8,
                    dim_head=64,
                    bias=True,
                    upcast_softmax=True,
                    cross_attention_norm='layer_norm',
                )
            elif hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                self.cross_attn1 = CrossAttention(
                    query_dim=self.mid_dim,
                    cross_attention_dim=self.mid_dim,
                    heads=8,
                    dim_head=64,
                    bias=True,
                    upcast_softmax=True,
                    cross_attention_norm='layer_norm',
                )
            elif hasattr(self.args.TRAIN, "CrossAttn_Post") and self.args.TRAIN.CrossAttn_Post:
                self.cross_attn1 = CrossAttention(
                    query_dim=self.mid_dim,
                    cross_attention_dim=self.mid_dim,
                    heads=8,
                    dim_head=64,
                    bias=True,
                    upcast_softmax=True,
                    cross_attention_norm='layer_norm',
                )
            if hasattr(self.args.TRAIN, "Diffusion") and self.args.TRAIN.Diffusion:
                self.combine = Combine()
                self.t_embedding_layer = nn.Embedding(self.n_steps, self.mid_dim * 4)
                self.MLPDiffusion = MLPDiffusion(self.decod_ln, self.dec, self.t_embedding_layer)
                self.cross_attn = CrossAttention(
                    query_dim=self.mid_dim,
                    cross_attention_dim=self.mid_dim,
                    heads=8,
                    dim_head=64,
                    bias=True,
                    upcast_softmax=True,
                    cross_attention_norm='layer_norm',
                )
                if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                    self.DiTmodel = DiT(depth=4,
                                    hidden_size=768,
                                    num_heads=12,
                                    in_channels = self.args.DATA.NUM_INPUT_FRAMES,
                                    input_dim = self.text_dim,
                                    num_steps= self.n_steps,
                                    pre_train= False
                                    )

                    Dit_load = load_DiT(device="cuda", jit=False,
                                        download_root='/mnt/hdd/fpeng/CLIP-FSAR/DiT_B_depth4_gpu4.pth')  # ViT-B/16
                    for k in Dit_load.keys():
                        print(k)
                    self.DiTmodel.load_state_dict(Dit_load, strict=False)
                    # print(model.state_dict()['visual.transformer.resblocks.0.attn.in_proj_weight'])

                    self.DiT = GaussianDiffusionDiT(
                        self.DiTmodel,
                        seq_length=self.text_dim,
                        timesteps=self.n_steps,
                        sampling_timesteps=20,
                        objective='pred_noise'
                    )
                else:
                    self.Unet1D = Unet1D(
                        dim=64,
                        dim_mults=(1, 2, 4, 8),
                        channels=self.args.DATA.NUM_INPUT_FRAMES,
                        crossmodal_condition = True,
                        crossmodal_conv = False,
                        self_condition = False
                    )
                    # Unet1d_load = load_DiT(device="cuda", jit=False,
                    #                     download_root='/mnt/hdd/fpeng/CLIP-FSAR/Unet1D_gpu4.pth')  # ViT-B/16
                    # self.Unet1D.load_state_dict(Unet1d_load, strict=False)
                    self.Unet1Ddiffusion = GaussianDiffusion1D(
                        self.Unet1D,
                        seq_length=self.text_dim,
                        timesteps=self.n_steps,
                        sampling_timesteps=20,
                        objective='pred_noise'
                    )
                    if self.args.TRAIN.Diffusion_Scale:
                        self.Diffusion_Scale = 1.0


    def text_encoder(self, prompt_embed, index):
        # token_embd = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = prompt_embed.type(self.dtype) + self.prompt_positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.prompt_ln_final(x).type(self.dtype)

        # print(index)
        full_x = x[:, :,:] @ self.prompt_projection #x[:, :index,:] @ self.text_projection  #TODO
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_preproj = x[torch.arange(x.shape[0]), index, :]
        x = x_preproj @ self.prompt_projection
        if self.args.TRAIN.FeaturePreproj:
            # print('x_preproj.shape', x_preproj.shape)
            return x_preproj, full_x, prompt_embed
        else:
            # print('x.shape', x.shape)
            return x, full_x, prompt_embed
    def get_feats(self, support_images, target_images, support_real_class=False, support_labels=False):
        """
        Takes in images from the support set and query video and returns CNN or ViT features.
        """
        # if self.args.VIDEO.HEAD.BACKBONE_NAME == "RN50":
        #     backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False,
        #                                      download_root='/mnt/hdd/fpeng/CLIP/models/')  # ViT-B/16
        # elif self.args.VIDEO.HEAD.BACKBONE_NAME == "ViT-B/16":
        #     backbone, self.preprocess = load(self.args.VIDEO.HEAD.BACKBONE_NAME, device="cuda", cfg=self.args, jit=False,
        #                                      download_root='/mnt/hdd/fpeng/CLIP/models/')  # ViT-B/16

        if self.training:
            #image(用proj之前的特征)
            # if self.args.TRAIN.BACKBONE_FROZEN:
            #     with torch.no_grad():
            #         if isinstance(backbone, torch.nn.DataParallel):
            #             back_bone = backbone.module
            #             back_bone = back_bone.cuda()
            #             support_features, support_features_pre = back_bone.visual(support_images) #[B*T,D]
            #         else:
            #             support_features, support_features_pre = backbone.visual(support_images)
            # else:
            support_features, support_features_pre = self.backbone(support_images) #[B*T,D]
            # with torch.no_grad():
            support_features_ori, support_features_ori_preproj = support_features.clone(), support_features_pre.clone() #self.backbone(support_images) #self.initial_backbone(support_images)

            support_features = support_features.squeeze()
            support_features_ori = support_features_ori.squeeze()
            support_features_ori_preproj = support_features_ori_preproj.squeeze()
            support_features_pre = support_features_pre.squeeze()
            # os.system("nvidia-smi")
            # if self.args.TRAIN.BACKBONE_FROZEN:
            #     with torch.no_grad():
            #         if isinstance(backbone, torch.nn.DataParallel):
            #             target_features, target_features_pre = back_bone.visual(target_images)
            #         else:
            #             target_features, target_features_pre = backbone.visual(target_images)
            # else:
            target_features, target_features_pre = self.backbone(target_images)
            # with torch.no_grad():
            target_features_ori, target_features_ori_preproj = target_features.clone(), target_features_pre.clone()#self.backbone(target_images)#self.initial_backbone(target_images)
            target_features = target_features.squeeze()
            target_features_ori = target_features_ori.squeeze()
            target_features_ori_preproj = target_features_ori_preproj.squeeze()
            target_features_pre = target_features_pre.squeeze()
            # os.system("nvidia-smi")

            dim = int(support_features.shape[1])
            dim2 = int(support_features_pre.shape[1])

            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim) #[BTD]
            support_features_ori = support_features_ori.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim) #[BTD]
            support_features_ori_preproj = support_features_ori_preproj.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2) #[BTD]
            support_features_pre = support_features_pre.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2) #[BTD]
            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features_ori = target_features_ori.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features_ori_preproj = target_features_ori_preproj.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            target_features_pre = target_features_pre.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            support_features_text = None
            
        else:
            # if self.args.TRAIN.BACKBONE_FROZEN:
            #     with torch.no_grad():
            #         if isinstance(backbone, torch.nn.DataParallel):
            #             back_bone = backbone.module
            #             back_bone = back_bone.cuda()
            #             support_features, support_features_pre = back_bone.visual(support_images)
            #             target_features, target_features_pre = back_bone.visual(target_images)
            #         else:
            #             support_features, support_features_pre = backbone.visual(support_images)
            #             target_features, target_features_pre = backbone.visual(target_images)
            # else:
            with torch.no_grad():
                support_features, support_features_pre = self.backbone(support_images)
                target_features, target_features_pre = self.backbone(target_images)
            # with torch.no_grad():
            support_features_ori, support_features_ori_preproj = support_features.clone(), support_features_pre.clone() #self.backbone(support_images) #self.initial_backbone(support_images)
            target_features_ori, target_features_ori_preproj = target_features.clone(), target_features_pre.clone() #self.backbone(target_images) #self.initial_backbone(target_images)


            support_features = support_features.squeeze()
            support_features_ori = support_features_ori.squeeze()
            support_features_ori_preproj = support_features_ori_preproj.squeeze()
            support_features_pre = support_features_pre.squeeze()
            target_features = target_features.squeeze()
            target_features_ori = target_features_ori.squeeze()
            target_features_ori_preproj = target_features_ori_preproj.squeeze()
            target_features_pre = target_features_pre.squeeze()
            dim = int(target_features.shape[1])
            dim2 = int(target_features_pre.shape[1])

            target_features = target_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features_ori = target_features_ori.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            target_features_ori_preproj = target_features_ori_preproj.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            target_features_pre = target_features_pre.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            support_features = support_features.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_ori = support_features_ori.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim)
            support_features_ori_preproj = support_features_ori_preproj.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            support_features_pre = support_features_pre.reshape(-1, self.args.DATA.NUM_INPUT_FRAMES, dim2)
            # support_real_class = torch.unique(support_real_class)
            support_features_text = self.text_features_test[support_real_class.long()]

        if self.args.TRAIN.FeaturePreproj:
            return support_features_pre, target_features_pre, support_features_text, support_features_ori_preproj, target_features_ori_preproj
        else:
            return support_features, target_features, support_features_text, support_features_ori, target_features_ori

    def forward(self, inputs):
        support_images, support_labels, target_images, support_real_class, target_real_class = inputs['support_set'], inputs['support_labels'], inputs['target_set'], inputs['real_support_labels'], inputs['real_target_labels'] # [200, 3, 224, 224] inputs["real_support_labels"]
        # [T, 3, 84, 84]
        # set_trace()
        if hasattr(self.args.TRAIN, "TePrompt") and self.args.TRAIN.TePrompt:
            self.text_features_train, self.text_features_train_full, _ = self.text_encoder(self.prompts_x,
                                                                                              self.ind_tr)
            self.text_features_test, self.text_features_test_full, _ = self.text_encoder(self.prompts_y,
                                                                                        self.ind_te)  # [CLASS,D]

        if self.training:
            support_features, target_features, _, support_features_ori, target_features_ori = self.get_feats(support_images, target_images, support_labels)
            support_bs = support_features.shape[0]
            target_bs = target_features.shape[0]
            # print('support_features.shape',support_features.shape)
            # if self.args.TRAIN.Frozen_Cliploss:
            #     support_features_ori_frozen = support_features_ori.clone()
            #     target_features_ori_frozen = target_features_ori.clone()
            #     support_features_ori = support_features.clone()
            #     target_features_ori = target_features.clone()

            if hasattr(self.args.TRAIN, "USE_CLASSIFICATION") and self.args.TRAIN.USE_CLASSIFICATION:  #"True"
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                if self.args.TRAIN.FeaturePreproj:
                    feature_classification_in = feature_classification_in @ self.backbone_proj
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                if not self.args.SUBACTS:
                    if self.args.TRAIN.FeaturePreproj:
                        class_text_logits = cos_sim(feature_classification, self.text_features_train@self.prompt_projection) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

                    else:
                        class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale  #相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？
                else: #TODO: 对子动作文本和每一帧计算class_text_logits,即根据相似度分数赋予文本后缀
                    dims = self.text_features_train.shape[-1]
                    dim = int(dims/4)
                    for i in range(4):
                        class_text_logits = cos_sim(feature_classification, self.text_features_train[..., dim*i:dim*(i+1)]) * self.scale  # 相似度数字（support和targetcat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

            else:
                class_text_logits = None

            if self.args.TRAIN.TemPrompt:
                if self.training:
                    # print('support_real_class.long().shape', support_real_class.shape)
                    # print('self.text_features_train_full.shape', self.text_features_train_full.shape)
                    #context_support = self.text_features_train_full[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1, 1)  #[B, T, n_cxt, D]
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)  #[B, T, n_cxt, D]
                    context_target = self.text_features_train[target_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)  #[B, T, n_cxt, D]

                else:
                    #context_support = self.text_features_test_full[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1, 1) # .repeat(support_bs+target_bs, 1, 1)
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
            else:
                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1) #.repeat(1,self.args.DATA.NUM_INPUT_FRAMES, 1)
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1) #.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)


            context_support = self.mid_layer(context_support)   #文本的support
            unique_labels = torch.unique(support_labels)
            context_support_pre = context_support.clone()
            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE: #False
                support_features_pre = support_features.clone()
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)
                support_features_ori_pre = support_features_ori.clone()
                support_features_ori = [
                    torch.mean(torch.index_select(support_features_ori, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features_ori = torch.stack(support_features_ori)
                context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                context_support = torch.stack(context_support)
            if self.args.TRAIN.TemPrompt:
                # print('support_features.shape',support_features.shape)
                # print('context_support.shape', context_support.shape)#TODO [5,8, 512]
                dim = int(support_features.shape[-1])
                support_bs = int(support_features.shape[0])
                support_bs_pre = int(support_features_ori_pre.shape[0])
                if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:#TODO
                    if not hasattr(self.args.TRAIN, "Diffusion") or not self.args.TRAIN.Diffusion:
                        if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                            De_v = self.cross_attn1(context_support, encoder_hidden_states=support_features_ori)
                        else:
                            De_v = self.dec(self.decod_ln(support_features_ori.reshape(-1, dim))).reshape(support_bs, self.args.DATA.NUM_INPUT_FRAMES, dim) #[C,T,D]
                            if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                De_v = self.cross_attn1(De_v, encoder_hidden_states=context_support)
                            elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                De_v = De_v + support_features_ori
                    else:
                        if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                            ###DiT_DIffusion
                            noise, pred_noise, mseloss, pred_x0 = self.DiT(context_support, crossmodal_cond=support_features_ori,
                                                                  alphas_bar_sqrt=alphas_bar_sqrt, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
                                                                sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod, n_steps=num_steps)
                        else:
                            ##UNet1Ddiffusion
                            if self.args.TRAIN.FeaturePreproj:
                                support_features_ori = support_features_ori@self.backbone_proj
                                support_features_ori_pre = support_features_ori_pre@self.backbone_proj
                                support_features = support_features @self.backbone_proj
                                dim = int(support_features.shape[-1])
                                # print('context_support.shape',context_support.shape)
                                # print('support_features_ori.shape', support_features_ori.shape)
                            # print('context_support.mean(-1)', context_support.mean(-1))
                            # print('context_support.mean()', context_support.mean())
                            if self.args.TRAIN.Diffusion_Scale:
                                noise, pred_noise, mseloss = self.Unet1Ddiffusion(context_support*self.Diffusion_Scale, crossmodal_cond=support_features_ori)
                            else:
                                noise, pred_noise, mseloss = self.Unet1Ddiffusion(context_support, crossmodal_cond=support_features_ori)
                            # print('train_support_psample_loop')
                            # print('pred_noise', pred_noise.shape)
                            # print('support_features_ori', support_features_ori.shape)
                        if self.training and not self.args.TRAIN.TrainingDiffusion:
                            if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                                De_v = self.cross_attn1(context_support, encoder_hidden_states=support_features_ori)
                            else:
                                De_v = self.dec(self.decod_ln(support_features_ori.reshape(-1, dim))).reshape(
                                    support_bs_pre, self.args.DATA.NUM_INPUT_FRAMES, dim)  # [C,T,D]
                                if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                    De_v = self.cross_attn1(De_v, encoder_hidden_states=context_support)
                                elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                    De_v = De_v + support_features_ori
                        else:
                            # De_v = torch.randn(support_bs_pre, self.args.DATA.NUM_INPUT_FRAMES, dim).cuda()
                            # N = 20
                            # # print('N = 20')
                            # for i in range(N):
                            #     time = torch.full(size=(support_bs_pre, 1, 1, 1), fill_value=(N - i) / N, dtype=torch.float32).cuda()
                            #     noise_r, latent_r = schedule(time)
                            #     t_embedding = self.combine( noise_r ** 2)
                            #     latent_r = latent_r.squeeze(1)
                            #     noise_r = noise_r.squeeze(1)
                            #     pred_noise_n = self.dec(self.decod_ln((De_v+t_embedding).reshape(-1, dim))).reshape(
                            #         support_bs_pre, self.args.DATA.NUM_INPUT_FRAMES, dim)
                            #     pred_noise_n = self.cross_attn(pred_noise_n, encoder_hidden_states=support_features_ori)
                            #     De_v = (De_v - noise_r * pred_noise_n) / latent_r
                            #     time = time - (1 / N)
                            #     noise_r, latent_r = schedule(time)
                            #     latent_r = latent_r.squeeze(1)
                            #     noise_r = noise_r.squeeze(1)
                            #     De_v = latent_r * De_v + noise_r * pred_noise_n
                            if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                                ###DiT_DIffusion
                                # De_v = p_sample_loop(self.MLPDiffusion, support_features_ori.shape, self.n_steps, betas, one_minus_alphas_bar_sqrt, y_cond=support_features_ori)
                                if ddim_steps>0:
                                    with torch.no_grad():
                                        De_v = self.DiT.ddim_sample(support_features_ori.shape, y_cond=support_features_ori, alphas_prod=alphas_prod,
                                                                        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
                                                                        betas=betas, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt)
                                else:
                                    with torch.no_grad():
                                        De_v = self.DiT.p_sample_loop(support_features_ori.shape, num_steps, betas, one_minus_alphas_bar_sqrt, y_cond=support_features_ori)

                                # De_v = De_v.detach().requires_grad_()  # 重新开始计算图
                                # torch.cuda.empty_cache()  # 清理 CUDA 缓存
                            else:
                                ###UNet1Ddiffusion
                                if self.args.TRAIN.Diffusion_Scale:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=support_bs, crossmodal_cond=support_features_ori)/self.Diffusion_Scale
                                else:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=support_bs, crossmodal_cond=support_features_ori)
                                # print('De_v.mean(-1)', De_v.mean(-1))
                                # print('De_v.mean()', De_v.mean())
                                # print('train_support_psample_loop')
                    # with torch.no_grad():
                    prefix_x = self.token_prefix_x
                    eos_x = self.token_eos_x
                    suffix_x = self.token_suffix_x
                    prompts_x=[]
                    for i in range(support_bs):
                        prefix_i = prefix_x[0, :, :].unsqueeze(0)
                        ctx_i = De_v[i:i+1, :, :]
                        eos_i = eos_x.unsqueeze(0).unsqueeze(1)
                        suffix_i = suffix_x[0, -1, :].unsqueeze(0).unsqueeze(1).repeat(1,67,1)

                        prompt = torch.cat(
                            [
                                prefix_i,  # (1, 1, dim)
                                ctx_i,  # (1, n_ctx, dim)
                                eos_i,  # (1, 1, dim)
                                suffix_i,
                            ],
                            dim=1,
                        )
                        prompts_x.append(prompt)
                    prompts_x = torch.cat(prompts_x, dim=0)  ##[n_cls, textlen, dim]
                    # _, Et_Dev, _ = self.text_encoder(prompts_x, self.args.DATA.NUM_INPUT_FRAMES + 1)
                    # Et_Dev = Et_Dev[:,1:,:]# [B, T, D]
                    # print('Et_Dev.shape', Et_Dev.shape) #([5, 8, 512])
                    Et_Dev_pre = De_v.clone()

                    # Et_Dev = [torch.mean(torch.index_select(De_v, 0, extract_class_indices(support_labels, c)),
                    #                dim=0) for c in unique_labels]
                    # Et_Dev = torch.stack(Et_Dev)  # [Class, T, D]
                    Et_Dev = De_v
                    # print('context_support.mean()',context_support.mean())
                    # print('Et_Dev.mean()',Et_Dev.mean())
                    mse_loss = (Et_Dev.detach().cpu().reshape(-1, dim) - context_support.detach().cpu().reshape(-1,dim)).square().mean()
                    print('mse_loss: ',mse_loss)
                    mse_loss = (Et_Dev.reshape(-1, dim) - context_support.reshape(-1,dim)).square().mean()

                    context_support = (context_support + Et_Dev)/2

                    if self.args.TRAIN.Visualplus:
                        support_features = support_features + Et_Dev    ###
                        support_features_pre = support_features_pre + Et_Dev_pre    ###
                if hasattr(self.args.TRAIN, "CrossAttn_Post") and self.args.TRAIN.CrossAttn_Post:
                    support_features = self.cross_attn1(support_features, encoder_hidden_states=context_support)
                    support_features = self.tlm(support_features, support_features, support_features)
                else:
                    support_features = torch.cat([support_features.unsqueeze(2), context_support.unsqueeze(2)], dim=2)  #对图文拼接的特征做时序建模,support是[B,T,D], #context_support是[B, T, n_cxt, D]-->[B,T,1+n_ctx,D]
                    # print('support_bs',support_bs)
                    support_features = support_features.reshape(self.args.DATA.NUM_INPUT_FRAMES*support_bs, -1, dim)
                    support_features = self.context2(support_features, support_features, support_features)[:, 0, :]  #[B*T, 1, D]
                    support_features = support_features.squeeze().reshape(support_bs, self.args.DATA.NUM_INPUT_FRAMES, dim)
                    support_features = self.tlm(support_features, support_features, support_features)
            else:
                support_features = torch.cat([support_features, context_support], dim=1)  # 对图文拼接的特征做时序建模,support是[B,T,D], #context_support是[B, 1, D]-->[B,T+1,D]
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES, :]

            if self.args.TRAIN.TemPrompt:
                if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:
                    dim = int(target_features.shape[-1])
                    target_bs = int(target_features.shape[0])
                    if hasattr(self.args.TRAIN, "Diffusion") and self.args.TRAIN.Diffusion:
                        # noise_target = torch.randn(target_bs, self.args.DATA.NUM_INPUT_FRAMES, dim).cuda()
                        # noise_r, latent_r = schedule(torch.rand(target_bs, 1, 1, 1).cuda())
                        # t_embedding = self.combine(noise_r ** 2)
                        # latent_r = latent_r.squeeze(1)
                        # noise_r = noise_r.squeeze(1)
                        # latents = context_target * latent_r + noise_target * noise_r
                        # pred_noise_target = self.dec(self.decod_ln((latents+t_embedding).reshape(-1, dim))).reshape(target_bs,
                        #                                                                       self.args.DATA.NUM_INPUT_FRAMES,
                        #                                                                       dim)
                        # pred_noise_target = self.cross_attn(pred_noise_target, encoder_hidden_states=target_features_ori)
                        if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                            ###DiT_DIffusion
                            noise_target, pred_noise_target, mseloss_target, pred_x0_target =  self.DiT(context_target, crossmodal_cond=target_features_ori,
                                                                                        alphas_bar_sqrt=alphas_bar_sqrt, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod,
                                                                sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod, n_steps = num_steps)

                        else:
                            ###UNet1Ddiffusion
                            if self.args.TRAIN.FeaturePreproj:
                                target_features_ori = target_features_ori@self.backbone_proj
                                target_features = target_features@self.backbone_proj
                                dim = int(target_features.shape[-1])
                            if self.args.TRAIN.Diffusion_Scale:
                                noise_target, pred_noise_target, mseloss_target = self.Unet1Ddiffusion(context_target*self.Diffusion_Scale, crossmodal_cond=target_features_ori)
                            else:
                                noise_target, pred_noise_target, mseloss_target = self.Unet1Ddiffusion(context_target, crossmodal_cond=target_features_ori)

                        if not self.training or self.args.TRAIN.TrainingDiffusion:
                            if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                                ##DiT_Diffusion
                                if ddim_steps>0:
                                    with torch.no_grad():
                                        De_v_target = self.DiT.ddim_sample(target_features_ori.shape, y_cond=target_features_ori, alphas_prod=alphas_prod,
                                                                        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
                                                                        betas=betas, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt)
                                else:
                                    with torch.no_grad():
                                        De_v_target = self.DiT.p_sample_loop(target_features_ori.shape, num_steps, betas, one_minus_alphas_bar_sqrt, y_cond=target_features_ori)
                                # De_v_target = De_v_target.detach().requires_grad_()  # 重新开始计算图
                                # torch.cuda.empty_cache()  # 清理 CUDA 缓存
                            else:
                                ###UNet1Ddiffusion
                                if self.args.TRAIN.Diffusion_Scale:
                                    with torch.no_grad():
                                        De_v_target = self.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)/self.Diffusion_Scale
                                else:
                                    with torch.no_grad():
                                        De_v_target = self.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)
                                    # print('train_target_psample_loop')

                        else:
                            if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                                De_v_target_target = self.cross_attn1(context_target, encoder_hidden_states=target_features_ori)
                            else:
                                De_v_target = self.dec(self.decod_ln(target_features_ori.reshape(-1, dim))).reshape(target_bs,
                                                                                                             self.args.DATA.NUM_INPUT_FRAMES,
                                                                                                             dim)  # [B,T,D]
                                if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                    De_v_target = self.cross_attn1(De_v_target, encoder_hidden_states=context_target)
                                elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                    De_v_target = De_v_target + target_features_ori
                    else:
                        if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                            De_v_target = self.cross_attn1(context_target, encoder_hidden_states=target_features_ori)
                        else:
                            De_v_target = self.dec(self.decod_ln(target_features_ori.reshape(-1, dim))).reshape(target_bs,
                                                                                                     self.args.DATA.NUM_INPUT_FRAMES,
                                                                                                     dim)  # [B,T,D]
                            if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                De_v_target = self.cross_attn1(De_v_target, encoder_hidden_states=context_target)
                            elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                De_v_target = De_v_target + target_features_ori

                    # with torch.no_grad():
                    prefix_x = self.token_prefix_x
                    eos_x = self.token_eos_x
                    suffix_x = self.token_suffix_x
                    prompts_x = []
                    for i in range(target_bs):
                        prefix_i = prefix_x[0, :, :].unsqueeze(0)
                        ctx_i = De_v_target[i:i + 1, :, :]
                        eos_i = eos_x.unsqueeze(0).unsqueeze(1)
                        suffix_i = suffix_x[0, -1, :].unsqueeze(0).unsqueeze(1).repeat(1, 67, 1)
                        prompt = torch.cat(
                            [
                                prefix_i,  # (1, 1, dim)
                                ctx_i,  # (1, n_ctx, dim)
                                eos_i,  # (1, 1, dim)
                                suffix_i,
                            ],
                            dim=1,
                        )
                        prompts_x.append(prompt)
                    prompts_x = torch.cat(prompts_x, dim=0)  ##[n_cls, textlen, dim]
                    # _, Et_Dev_target, _ = self.text_encoder(prompts_x, self.args.DATA.NUM_INPUT_FRAMES + 1)
                    # Et_Dev_target = Et_Dev_target[:, 1:, :]  # [B, T, D]
                    Et_Dev_target = De_v_target.clone()

                    ####Diffusion_CLIP_ft
                    if self.args.TRAIN.Frozen_Cliploss:
                        target_features_ori_frozen = target_features_ori
                        target_features_ori_frozen_unique = [torch.mean(torch.index_select(target_features_ori_frozen, 0, extract_class_indices(inputs["target_labels"], c)),
                                       dim=0) for c in unique_labels]
                        target_features_ori_frozen_unique = torch.stack(target_features_ori_frozen_unique)
 
                        cos_mseloss = cos_sim(self.classification_layer(Et_Dev_target).reshape(-1, dim), target_features_ori_frozen_unique.reshape(-1, dim))*self.scale
                        # *(B*T,25)
                        ####
                        target_features_ori_frozen_mean = target_features_ori_frozen/ target_features_ori_frozen.norm(dim=-1, keepdim=True)
                        Et_Dev_target_mean = Et_Dev_target/Et_Dev_target.norm(dim=-1, keepdim=True)
                        context_target_mean = context_target/context_target .norm(dim=-1, keepdim=True)
                        logits_per_image = self.scale * target_features_ori_frozen_mean.reshape(-1, dim)@ Et_Dev_target_mean.reshape(-1, dim).t()
                        global_cliploss = -torch.log((2 -(1- logits_per_image/100).mean())/2)
                        ###
                        edit_direction = (Et_Dev_target_mean - target_features_ori_frozen_mean)
                        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
                        target_direction = (context_target_mean - target_features_ori_frozen_mean)
                        target_direction /= (target_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)
                        cos_func = torch.nn.CosineSimilarity()
                        direction_lossv1 = (-torch.log((2.-(1. -  cos_func(edit_direction, target_direction)))/2)).mean()
                        print('direction_lossv1', direction_lossv1)
                        ###
                        logits_per_image_src = self.scale * target_features_ori_frozen_mean.reshape(-1, dim)@ context_target_mean.reshape(-1, dim).t()
                        l1_loss_fn = nn.L1Loss()
                        direction_lossv2 = l1_loss_fn(logits_per_image_src, logits_per_image).mean()
                        ###
                        l1_loss = l1_loss_fn(context_target_mean, Et_Dev_target_mean).mean()

                    if self.args.TRAIN.Visualplus:
                        target_features = target_features + Et_Dev_target   ###
                    if hasattr(self.args.TRAIN, "CrossAttn_Post") and self.args.TRAIN.CrossAttn_Post:
                        target_features = self.cross_attn1(target_features, encoder_hidden_states=Et_Dev_target)
                        target_features = self.tlm(target_features, target_features, target_features)
                    else:
                        target_features = torch.cat([target_features.unsqueeze(2), Et_Dev_target.unsqueeze(2)],
                                                    dim=2)  # 对图文拼接的特征做时序建模,support是[B,T,D], #context_support是[B, T, n_cxt, D]-->[B,T,1+n_ctx,D]
                        target_features = target_features.reshape(self.args.DATA.NUM_INPUT_FRAMES * target_bs, -1, dim)
                        target_features = self.context2(target_features, target_features, target_features)[:, 0,
                                          :]  # [B*T, 1, D]
                        target_features = target_features.squeeze().reshape(target_bs, self.args.DATA.NUM_INPUT_FRAMES, dim)
                        target_features = self.tlm(target_features, target_features, target_features)

                    ###v1
                    #every T frame cos_sim of image and imaginary_text
                    Et_Dev_target_unique = [torch.mean(torch.index_select(De_v_target, 0, extract_class_indices(inputs["target_labels"], c)),
                                   dim=0) for c in unique_labels]
                    Et_Dev_target_unique = torch.stack(Et_Dev_target_unique)  # [Class, T, D]
                    if self.args.TRAIN.Frozen_Cliploss:
                        support_features_ori_frozen = support_features_ori_pre
                        feature_classification_in_ori = torch.cat([support_features_ori_frozen, target_features_ori_frozen], dim=0)  # [B,T,D]
                    else:
                        feature_classification_in_ori = torch.cat([support_features_ori_pre, target_features_ori], dim=0)  # [B,T,D]
                    feature_classification_ori = self.classification_layer(feature_classification_in_ori).reshape(-1, dim) #[240, 512]

                    if inputs['Et_Dev_lastepo'] == None:
                        Et_Dev_mean = self.text_features_train.clone()  # [64,512]
                    else:
                        Et_Dev_mean = inputs['Et_Dev_lastepo']
                        # print('Et_Dev_lastepo')
                    Et_Dev_cat = self.classification_layer(torch.cat([Et_Dev, De_v_target], dim=0)).reshape(-1, dim)
                    # Et_Dev_x0 = self.classification_layer(torch.cat([pred_x0, pred_x0_target], dim=0)).reshape(-1, dim)  ##[30*8, 512]
                    Et_Dev_all = torch.mean((Et_Dev + Et_Dev_target_unique) / 2, dim=1).squeeze()
                    # print('support_real_class.long()', support_real_class.long().detach().cpu())
                    unique_labels_support_real = torch.unique(support_real_class.long())
                    # print('unique_labels_support_real', unique_labels_support_real.detach().cpu())
                    # support_real_class.long()
                    # tensor([22, 56, 18, 51, 51, 22, 45, 56, 18, 22, 45, 18, 18, 56, 45, 45, 51, 51,
                    #         18, 56, 22, 51, 56, 22, 45])
                    # unique_labels_support_real
                    # tensor([18, 22, 45, 51, 56])
                    # Et_Dev_mean = self.text_features_train.clone()  # [64,512]
                    Et_Dev_mean[unique_labels_support_real] = Et_Dev_all.float()   #[64, 512]

                    if self.args.TRAIN.FeaturePreproj:
                        class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_mean.reshape(-1,
                                                                                               dim)@self.prompt_projection) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？
                    else:
                        if self.args.TRAIN.Frozen_Cliploss:
                            # feature_classification_ori = feature_classification_ori / feature_classification_ori.norm(dim=-1, keepdim=True)
                            # Et_Dev_mean_norm = Et_Dev_mean.reshape(-1, dim) / Et_Dev_mean.reshape(-1, dim).norm(dim=-1, keepdim=True)
                            # Et_Dev_cat_norm = Et_Dev_cat / Et_Dev_cat.norm(dim=-1, keepdim=True)
                            # # Et_Dev_x0_norm = Et_Dev_x0 / Et_Dev_x0.norm(dim=-1, keepdim=True)
                            # text_features_train_norm = self.text_features_train/self.text_features_train.norm(dim=-1, keepdim=True)
                            # # class_text_logits_ori = self.scale * feature_classification_ori @ Et_Dev_mean_norm.t()

                            class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_mean.reshape(-1,
                                                                                                            dim)) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

                            # class_text_logits_ori = self.scale * Et_Dev_cat_norm @ text_features_train_norm.t()

                            #
                            # mask = torch.eye(Et_Dev_x0_norm.shape[0]).cuda()
                            # gen_cosim_loss = (1 - (feature_classification_ori @ Et_Dev_x0_norm.t())*mask).mean() ##[30*8,30*8]


                        else:
                            class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_mean.reshape(-1,
                                                                                dim)) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

                    if hasattr(self.args.TRAIN, "Diffusion") and self.args.TRAIN.Diffusion:
                        noise = torch.cat([noise, noise_target], dim=0)
                        pred_noise = torch.cat([pred_noise, pred_noise_target], dim=0)


                    # ###v2
                    # if self.args.TRAIN.Visualplus:
                    #     feature_classification_in_ori = torch.cat([support_features_pre, target_features],
                    #                                               dim=0)  # [B,T,D]
                    # else:
                    #     feature_classification_in_ori = torch.cat([Et_Dev_pre, Et_Dev_target],
                    #                                           dim=0)  # [B,T,D]
                    # feature_classification_ori = self.classification_layer(feature_classification_in_ori).reshape(-1, dim)  # [240, 512]
                    # class_text_logits_ori = cos_sim(feature_classification_ori, self.text_features_train) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

                    # print('support_features_ori_pre', support_features_ori_pre.shape) #[5, 8, 512]
                    # print('target_features_ori', target_features_ori.shape)  #[25, 8, 512]
                    # print('feature_classification_in_ori', feature_classification_in_ori.shape) #[30, 8, 512]
                    # print('feature_classification_ori', feature_classification_ori.shape) #[240, 512]
                    # print('class_text_logits_ori.shape', class_text_logits_ori.shape) #([240, 240]
                    # print('Et_Dev_all.shape', Et_Dev_all.shape) #([5, 512]
                    # print('Et_Dev_mean', Et_Dev_mean.shape)  #([64, 512]
                else:
                    target_features = self.context2(target_features, target_features, target_features)
                    target_features = self.tlm(target_features, target_features, target_features)
            else:
                target_features = self.context2(target_features, target_features, target_features)  # 时序建模后的target特征

            if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                pass
            else:
                unique_labels = torch.unique(support_labels)
                support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                support_features = torch.stack(support_features)

            unique_labels = torch.unique(support_labels)

            n_queries = target_features.shape[0]
            n_support = support_features.shape[0]
            frame_num = self.args.DATA.NUM_INPUT_FRAMES

            if hasattr(self.args.TRAIN, "HyRSMDist") and self.args.TRAIN.HyRSMDist:
                # target_shape:([25, 8, 512])
                # support_shape:torch.Size([5, 8, 512])
                # n_queries: 25
                # n_support: 5
                # print('n_queries', n_queries)
                # print('n_support', n_support)
                # print('target_shape', target_features.shape)
                # print('support_shape', support_features.shape)
                support_features = support_features.repeat(n_queries, 1, 1, 1)
                support_features = rearrange(support_features, 'b h s d -> b (h s) d')
                frame_sim = torch.matmul(F.normalize(support_features, dim=2),
                                         F.normalize(target_features, dim=2).permute(0, 2, 1)).reshape(n_queries,
                                                                                                       n_support,
                                                                                                       frame_num,
                                                                                                       frame_num)
                frame_dists = 1 - frame_sim
                dists = frame_dists

                cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
            else:
                support_features = rearrange(support_features, 'b s d -> (b s) d')
                target_features = rearrange(target_features, 'b s d -> (b s) d')

                frame_sim = cos_sim(target_features, support_features)   #每一帧的相似度
                frame_dists = 1 - frame_sim  #每一帧的距离

                dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                # calculate query -> support and support -> query
                if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                    cum_dists = OTAM_cum_dist_v2(dists)
                else:
                    cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
        else: #test
            if hasattr(self.args.TRAIN, "EVAL_TEXT") and self.args.TRAIN.EVAL_TEXT:  #False 图文分数
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features, support_features_ori, target_features_ori = self.get_feats(support_images, target_images, support_real_class)
                # if self.args.TRAIN.Frozen_Cliploss:
                #     support_features_ori_frozen = support_features_ori.clone()
                #     target_features_ori_frozen = target_features_ori.clone()
                #     support_features_ori = support_features
                #     target_features_ori = target_features
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)  #每一帧的相似度评分
                
                cum_dists = -logits_per_image # 
                class_text_logits = None

            elif hasattr(self.args.TRAIN, "COMBINE") and self.args.TRAIN.COMBINE: #False
                # text_features = self.text_features_test[support_real_class.long()]
                unique_labels = torch.unique(support_labels)
                support_features, target_features, text_features, support_features_ori, target_features_ori = self.get_feats(support_images, target_images, support_real_class)
                # if self.args.TRAIN.Frozen_Cliploss:
                #     support_features_ori_frozen = support_features_ori.clone()
                #     target_features_ori_frozen = target_features_ori.clone()
                #     support_features_ori = support_features
                #     target_features_ori = target_features
                text_features = [torch.mean(torch.index_select(text_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                text_features = torch.stack(text_features)
                # unique_labels = torch.unique(support_labels)
                image_features = self.classification_layer(target_features.mean(1))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)

                # cosine similarity as logits
                logit_scale = self.scale # 1. # self.backbone.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()
                logits_per_image = F.softmax(logits_per_image, dim=1)
                
                class_text_logits = None

                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)

                if self.args.TRAIN.FeaturePreproj:
                    class_text_logits = cos_sim(feature_classification@self.backbone_proj, self.text_features_train@self.prompt_projection)*self.scale
                else:
                    class_text_logits = cos_sim(feature_classification, self.text_features_train)*self.scale

                if self.training:
                    context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
                
                else:
                    context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)#.repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
                
                target_features = self.context2(target_features, target_features, target_features)
                context_support = self.mid_layer(context_support)  # F.relu(self.mid_layer(context_support))
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    support_features_ori_pre = support_features_ori.clone()
                    support_features_ori = [torch.mean(torch.index_select(support_features_ori, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features_ori = torch.stack(support_features_ori)
                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                support_features = torch.cat([support_features, context_support], dim=1)
                support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)

                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]
                frame_num = self.args.DATA.NUM_INPUT_FRAMES

                if hasattr(self.args.TRAIN, "HyRSMDist") and self.args.TRAIN.HyRSMDist:

                    # print('n_queries', n_queries)
                    # print('n_support', n_support)
                    # print('target_shape', target_features.shape)
                    # print('support_shape', support_features.shape)
                    support_features = support_features.repeat(n_queries,1,1,1)
                    support_features = rearrange(support_features, 'b h s d -> b (h s) d')
                    frame_sim = torch.matmul(F.normalize(support_features, dim=2),
                                             F.normalize(target_features, dim=2).permute(0, 2, 1)).reshape(n_queries, n_support, frame_num, frame_num)
                    frame_dists = 1 - frame_sim
                    dists = frame_dists

                    cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)
                else:
                    support_features = rearrange(support_features, 'b s d -> (b s) d')
                    target_features = rearrange(target_features, 'b s d -> (b s) d')

                    frame_sim = cos_sim(target_features, support_features)
                    frame_dists = 1 - frame_sim

                    dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                    # calculate query -> support and support -> query
                    if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                        cum_dists_visual = OTAM_cum_dist_v2(dists)
                    else:
                        cum_dists_visual = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))
                    cum_dists_visual_soft = F.softmax((8-cum_dists_visual)/8., dim=1)
                    if hasattr(self.args.TRAIN, "TEXT_COFF") and self.args.TRAIN.TEXT_COFF: #False
                        cum_dists = -(logits_per_image.pow(self.args.TRAIN.TEXT_COFF)*cum_dists_visual_soft.pow(1.0-self.args.TRAIN.TEXT_COFF))
                    else:
                        cum_dists = -(logits_per_image.pow(0.9)*cum_dists_visual_soft.pow(0.1))
                
                class_text_logits = None #注释掉

            else:
                support_features, target_features, _ , support_features_ori, target_features_ori = self.get_feats(support_images, target_images, support_labels)
                # if self.args.TRAIN.Frozen_Cliploss:
                #     support_features_ori_frozen = support_features_ori.clone()
                #     target_features_ori_frozen = target_features_ori.clone()
                #     support_features_ori = support_features.clone()
                #     target_features_ori = target_features.clone()
                support_bs = support_features.shape[0]
                target_bs = target_features.shape[0]
                
                feature_classification_in = torch.cat([support_features,target_features], dim=0)
                feature_classification = self.classification_layer(feature_classification_in).mean(1)
                if self.args.TRAIN.FeaturePreproj:
                    class_text_logits = cos_sim(feature_classification@self.backbone_proj, self.text_features_test@self.prompt_projection)*self.scale
                else:
                    class_text_logits = cos_sim(feature_classification, self.text_features_test)*self.scale

                if self.args.TRAIN.TemPrompt:
                    if self.training:
                        #context_support = self.text_features_train_full[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1, 1)
                        context_support = self.text_features_train[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
                    else:
                        #context_support = self.text_features_test_full[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1, 1) # .repeat(support_bs+target_bs, 1, 1)
                        context_support = self.text_features_test[support_real_class.long()].unsqueeze(1).repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)
                else:
                    if self.training:
                        context_support = self.text_features_train[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1)
                    else:
                        context_support = self.text_features_test[support_real_class.long()].unsqueeze(1)  # .repeat(1, self.args.DATA.NUM_INPUT_FRAMES, 1) # .repeat(support_bs+target_bs, 1, 1)

                context_support_pre = context_support.clone()
                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    unique_labels = torch.unique(support_labels)
                    support_features_pre = support_features.clone()
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)
                    support_features_ori_pre = support_features_ori.clone()
                    support_features_ori = [torch.mean(torch.index_select(support_features_ori, 0, extract_class_indices(support_labels, c)),
                                   dim=0) for c in unique_labels]
                    support_features_ori = torch.stack(support_features_ori)

                    context_support = [torch.mean(torch.index_select(context_support, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    context_support = torch.stack(context_support)
                if self.args.TRAIN.TemPrompt:
                    dim = int(support_features.shape[-1])
                    support_bs = int(support_features.shape[0])
                    support_bs_pre = int(support_features_ori_pre.shape[0])
                    if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:  # TODO
                        if not hasattr(self.args.TRAIN, "Diffusion") or not self.args.TRAIN.Diffusion:
                            if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                                De_v = self.cross_attn1(context_support, encoder_hidden_states=support_features_ori)
                            else:
                                De_v = self.dec(self.decod_ln(support_features_ori.reshape(-1, dim))).reshape(
                                    support_bs,
                                    self.args.DATA.NUM_INPUT_FRAMES,
                                    dim)  # [C,T,D]
                                if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                    De_v = self.cross_attn1(De_v, encoder_hidden_states=context_support)
                                elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                    De_v = De_v + support_features_ori
                        else:
                            if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                                ###DiT_Diffusion
                                if ddim_steps>0:
                                    with torch.no_grad():
                                        De_v = self.DiT.ddim_sample(support_features_ori.shape, y_cond=support_features_ori, alphas_prod=alphas_prod,
                                                                        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
                                                                        betas=betas, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt)
                                else:
                                    with torch.no_grad():
                                        De_v = self.DiT.p_sample_loop(support_features_ori.shape, num_steps, betas, one_minus_alphas_bar_sqrt, y_cond=support_features_ori)
                            else:
                                ###UNet1Ddiffusion
                                if self.args.TRAIN.FeaturePreproj:
                                    support_features_ori = support_features_ori @ self.backbone_proj
                                    support_features_ori_pre = support_features_ori_pre @ self.backbone_proj
                                    support_features = support_features @ self.backbone_proj
                                    dim = int(support_features.shape[-1])
                                if self.args.TRAIN.Diffusion_Scale:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=support_bs, crossmodal_cond=support_features_ori)/self.Diffusion_Scale
                                else:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=support_bs, crossmodal_cond=support_features_ori)
                                # print('test_support_psample_loop')

                        # with torch.no_grad():
                        prefix_y = self.token_prefix_y
                        eos_y = self.token_eos_y
                        suffix_y = self.token_suffix_y
                        prompts_y = []
                        for i in range(support_bs):
                            prefix_i = prefix_y[0, :, :].unsqueeze(0)
                            ctx_i = De_v[i:i + 1, :, :]
                            eos_i = eos_y.unsqueeze(0).unsqueeze(1)
                            suffix_i = suffix_y[0, -1, :].unsqueeze(0).unsqueeze(1).repeat(1, 67, 1)
                            prompt = torch.cat(
                                [
                                    prefix_i,  # (1, 1, dim)
                                    ctx_i,  # (1, n_ctx, dim)
                                    eos_i,  # (1, 1, dim)
                                    suffix_i,
                                ],
                                dim=1,
                            )
                            prompts_y.append(prompt)
                        prompts_y = torch.cat(prompts_y, dim=0)  ##[n_cls, textlen, dim]
                        # _, Et_Dev, _ = self.text_encoder(prompts_y, self.args.DATA.NUM_INPUT_FRAMES + 1)
                        # Et_Dev = Et_Dev[:, 1:, :]# [CLASS, T, D]
                        Et_Dev_pre = De_v.clone()
                        # Et_Dev = [torch.mean(torch.index_select(De_v, 0, extract_class_indices(support_labels, c)),
                        #                      dim=0) for c in unique_labels]
                        # Et_Dev = torch.stack(Et_Dev)  # [Class, T, D]
                        Et_Dev = De_v
                        context_support = (context_support + Et_Dev) / 2
                        if self.args.TRAIN.Visualplus:
                            support_features = support_features + Et_Dev   ###
                            support_features_pre = support_features_pre + Et_Dev

                    if hasattr(self.args.TRAIN, "CrossAttn_Post") and self.args.TRAIN.CrossAttn_Post:
                        support_features = self.cross_attn1(support_features, encoder_hidden_states=context_support)
                        support_features = self.tlm(support_features, support_features, support_features)
                    else:
                        support_features = torch.cat([support_features.unsqueeze(2), context_support.unsqueeze(2)], dim=2)  # 对图文拼接的特征做时序建模,support是[B,T,D], #context_support是[B, T, n_cxt, D]-->[B,T,1+n_ctx,D]

                        support_features = support_features.reshape(self.args.DATA.NUM_INPUT_FRAMES * support_bs, -1, dim)
                        support_features = self.context2(support_features, support_features, support_features)[:, 0, :]  # [B*T, 1, D]
                        support_features = support_features.squeeze().reshape(support_bs, self.args.DATA.NUM_INPUT_FRAMES, dim)
                        support_features = self.tlm(support_features, support_features, support_features)
                else:
                    support_features = torch.cat([support_features, context_support], dim=1)
                    support_features = self.context2(support_features, support_features, support_features)[:,:self.args.DATA.NUM_INPUT_FRAMES,:]

                if self.args.TRAIN.TemPrompt:
                    if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:
                        dim = int(target_features.shape[-1])
                        target_bs = int(target_features.shape[0])
                        # print('target_bs', target_bs)
                        target_bs = int(target_features_ori.shape[0])
                        # print('target_bs_ori', target_bs)

                        if hasattr(self.args.TRAIN, "Diffusion") and self.args.TRAIN.Diffusion:
                            if hasattr(self.args.TRAIN, "DiffusionTransformer") and self.args.TRAIN.DiffusionTransformer:
                                ###DiT_Diffusion
                                if ddim_steps>0:
                                    with torch.no_grad():
                                        De_v = self.DiT.ddim_sample(target_features_ori.shape, y_cond=target_features_ori, alphas_prod=alphas_prod,
                                                                        sqrt_recip_alphas_cumprod=sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod=sqrt_recipm1_alphas_cumprod,
                                                                        betas=betas, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt)
                                else:
                                    with torch.no_grad():
                                        De_v = self.DiT.p_sample_loop(target_features_ori.shape, num_steps, betas,
                                                                        one_minus_alphas_bar_sqrt, y_cond=target_features_ori)

                            else:
                                ###UNet1Ddiffusion
                                if self.args.TRAIN.FeaturePreproj:
                                    target_features_ori = target_features_ori @ self.backbone_proj
                                    target_features = target_features @ self.backbone_proj
                                    dim = int(target_features.shape[-1])
                                if self.args.TRAIN.Diffusion_Scale:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)/self.Diffusion_Scale
                                else:
                                    with torch.no_grad():
                                        De_v = self.Unet1Ddiffusion.sample(batch_size=target_bs, crossmodal_cond=target_features_ori)
                                # print('test_target_psample_loop')

                        else:
                            if hasattr(self.args.TRAIN, "CrossAttn_Pre") and self.args.TRAIN.CrossAttn_Pre:
                                De_v = self.cross_attn1(target_features_ori, encoder_hidden_states=target_features_ori)
                            else:
                                De_v = self.dec(self.decod_ln(target_features_ori.reshape(-1, dim))).reshape(target_bs,
                                                                                                         self.args.DATA.NUM_INPUT_FRAMES,
                                                                                                         dim)  # [B,T,D]
                                if hasattr(self.args.TRAIN, "CrossAttn") and self.args.TRAIN.CrossAttn:
                                    De_v = self.cross_attn1(De_v, encoder_hidden_states=target_features_ori)
                                elif hasattr(self.args.TRAIN, "Residual") and self.args.TRAIN.Residual:
                                    De_v = De_v + target_features_ori
                        # with torch.no_grad():
                        prefix_y = self.token_prefix_y
                        eos_y = self.token_eos_y
                        suffix_y = self.token_suffix_y
                        prompts_y = []
                        for i in range(target_bs):
                            prefix_i = prefix_y[0, :, :].unsqueeze(0)
                            ctx_i = De_v[i:i + 1, :, :]
                            eos_i = eos_y.unsqueeze(0).unsqueeze(1)
                            suffix_i = suffix_y[0, -1, :].unsqueeze(0).unsqueeze(1).repeat(1, 67, 1)
                            prompt = torch.cat(
                                [
                                    prefix_i,  # (1, 1, dim)
                                    ctx_i,  # (1, n_ctx, dim)
                                    eos_i,  # (1, 1, dim)
                                    suffix_i,
                                ],
                                dim=1,
                            )
                            prompts_y.append(prompt)
                        prompts_y = torch.cat(prompts_y, dim=0)  ##[n_cls, textlen, dim]
                        # _, Et_Dev_target, _ = self.text_encoder(prompts_y, self.args.DATA.NUM_INPUT_FRAMES + 1)
                        # Et_Dev_target = Et_Dev_target[:, 1:, :]  # [B, T, D]
                        Et_Dev_target = De_v.clone()
                        if self.args.TRAIN.Visualplus:
                            target_features = target_features + Et_Dev_target ###

                        if hasattr(self.args.TRAIN, "CrossAttn_Post") and self.args.TRAIN.CrossAttn_Post:
                            target_features = self.cross_attn1(target_features, encoder_hidden_states=Et_Dev_target)
                            target_features = self.tlm(target_features, target_features, target_features)
                        else:
                            # print('target_features.shape', target_features.shape)
                            # print('Et_Dev_target.shape', Et_Dev_target.shape)
                            target_features = torch.cat([target_features.unsqueeze(2), Et_Dev_target.unsqueeze(2)],
                                                        dim=2)  # 对图文拼接的特征做时序建模,support是[B,T,D], #context_support是[B, T, n_cxt, D]-->[B,T,1+n_ctx,D]
                            target_features = target_features.reshape(self.args.DATA.NUM_INPUT_FRAMES * target_bs, -1, dim)
                            target_features = self.context2(target_features, target_features, target_features)[:, 0,
                                              :]  # [B*T, 1, D]
                            target_features = target_features.squeeze().reshape(target_bs, self.args.DATA.NUM_INPUT_FRAMES,
                                                                                dim)
                            target_features = self.tlm(target_features, target_features, target_features)

                        ####V1
                        ##every T frame cos_sim of image and imaginary_text
                        if self.args.TRAIN.Frozen_Cliploss:
                            support_features_ori_frozen = support_features_ori
                            target_features_ori_frozen = target_features_ori
                            feature_classification_in_ori = torch.cat([support_features_ori_frozen, target_features_ori_frozen],
                                                                      dim=0)  # [B,T,D]
                        else:
                            feature_classification_in_ori = torch.cat([support_features_ori_pre, target_features_ori],
                                                                      dim=0)  # [B,T,D]
                        feature_classification_ori = self.classification_layer(feature_classification_in_ori).reshape(
                            -1, dim)
                        Et_Dev_all = torch.cat([Et_Dev, Et_Dev_target], dim=0)
                        if self.args.TRAIN.FeaturePreproj:
                            class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_all.reshape(-1, dim)@self.prompt_projection) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？
                        else:
                            if self.args.TRAIN.Frozen_Cliploss:
                                class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_all.reshape(-1,
                                                                                                               dim)) * self.scale
                            else:
                                class_text_logits_ori = cos_sim(feature_classification_ori, Et_Dev_all.reshape(-1,
                                                                                                       dim))* self.scale
                        # noise = torch.cat([noise, noise_target], dim=0)
                        # noise_pred = torch.cat([pred_noise, pred_noise_target], dim=0)
                        # ###v2
                        # if self.args.TRAIN.Visualplus:
                        #     feature_classification_in_ori = torch.cat([support_features_pre, target_features],
                        #                                               dim=0)  # [B,T,D]
                        # else:
                        #     feature_classification_in_ori = torch.cat([Et_Dev_pre, Et_Dev_target],
                        #                                           dim=0)  # [B,T,D]
                        # feature_classification_ori = self.classification_layer(feature_classification_in_ori).reshape(
                        #     -1, dim)  # [240, 512]
                        # class_text_logits_ori = cos_sim(feature_classification_ori,
                        #                                 self.text_features_test) * self.scale  # 相似度数字（support和target cat后的图像特征和train文本特征的相似度）--这个是子动作的判断标准？

                    else:
                        target_features = self.context2(target_features, target_features, target_features)
                        target_features = self.tlm(target_features, target_features, target_features)
                else:
                    target_features = self.context2(target_features, target_features, target_features)  # 时序建模后的target特征

                if hasattr(self.args.TRAIN, "MERGE_BEFORE") and self.args.TRAIN.MERGE_BEFORE:
                    pass
                else:
                    unique_labels = torch.unique(support_labels)
                    support_features = [torch.mean(torch.index_select(support_features, 0, extract_class_indices(support_labels, c)), dim=0) for c in unique_labels]
                    support_features = torch.stack(support_features)


                unique_labels = torch.unique(support_labels)

                n_queries = target_features.shape[0]
                n_support = support_features.shape[0]
                frame_num = self.args.DATA.NUM_INPUT_FRAMES

                if hasattr(self.args.TRAIN, "HyRSMDist") and self.args.TRAIN.HyRSMDist:
                    # n_queries 5
                    # n_support 5
                    # target_shape torch.Size([5, 8, 512])
                    # support_shape torch.Size([5, 8, 512])
                    # print('n_queries', n_queries)
                    # print('n_support', n_support)
                    # print('target_shape', target_features.shape)
                    # print('support_shape', support_features.shape)
                    support_features = support_features.repeat(n_queries,1,1,1)
                    support_features = rearrange(support_features, 'b h s d -> b (h s) d')
                    frame_sim = torch.matmul(F.normalize(support_features, dim=2),
                                             F.normalize(target_features, dim=2).permute(0, 2, 1)).reshape(n_queries, n_support, frame_num, frame_num)
                    frame_dists = 1 - frame_sim
                    dists = frame_dists

                    cum_dists = dists.min(3)[0].sum(2) + dists.min(2)[0].sum(2)

                else:
                    support_features = rearrange(support_features, 'b s d -> (b s) d')  # [200, 2048]
                    target_features = rearrange(target_features, 'b s d -> (b s) d')  # [200, 2048]

                    frame_sim = cos_sim(target_features, support_features)    # [200, 200]
                    frame_dists = 1 - frame_sim
                
                    dists = rearrange(frame_dists, '(tb ts) (sb ss) -> tb sb ts ss', tb = n_queries, sb = n_support)  # [25, 25, 8, 8]

                    # calculate query -> support and support -> query
                    if hasattr(self.args.TRAIN, "SINGLE_DIRECT") and self.args.TRAIN.SINGLE_DIRECT:
                        cum_dists = OTAM_cum_dist_v2(dists)
                    else:
                        cum_dists = OTAM_cum_dist_v2(dists) + OTAM_cum_dist_v2(rearrange(dists, 'tb sb ts ss -> tb sb ss ts'))

        class_dists = [torch.mean(torch.index_select(cum_dists, 1, extract_class_indices(unique_labels, c)), dim=1) for c in unique_labels]
        class_dists = torch.stack(class_dists)
        class_dists = rearrange(class_dists, 'c q -> q c')
        if hasattr(self.args.TRAIN, "Decoder") and self.args.TRAIN.Decoder:
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits, "class_logits_ori":class_text_logits_ori, "vis_feat":feature_classification_in, "vis_feat_ori":feature_classification_in_ori}
            if hasattr(self.args.TRAIN, "Diffusion") and self.args.TRAIN.Diffusion and self.training:
                return_dict['noise'] = noise
                return_dict['pred_noise'] = pred_noise
                # mseloss_mean = (mseloss+mseloss_target)/2
                # return_dict['mseloss'] = mseloss_mean
                return_dict['gen_cosim_loss'] = 0 #gen_cosim_loss
                return_dict['Et_Dev_mean'] = Et_Dev_mean
                return_dict ['mse_loss'] = mse_loss
                return_dict['cos_mseloss'] = cos_mseloss
                return_dict['global_cliploss'] = global_cliploss
                return_dict['direction_lossv1'] = direction_lossv1
                return_dict['direction_lossv2'] = direction_lossv2
                return_dict['l1_loss'] = l1_loss
                target_features_ori = torch.cat([support_features_ori,target_features_ori],dim=0)
                target_features_ori_frozen = torch.cat([support_features_ori,target_features_ori_frozen],dim=0)
                # print('context_support_pre', context_support_pre.detach().cpu())
                context_target = torch.cat([context_support,context_target],dim=0)
                return_dict['target_features_ori_frozen'] = target_features_ori_frozen.detach().clone()
                return_dict['target_features_ori'] = target_features_ori.detach().clone()
                return_dict['context_target'] = context_target.detach().clone()
                return_dict['cofficients'] = [alphas_prod, sqrt_recip_alphas_cumprod, sqrt_recipm1_alphas_cumprod, betas, one_minus_alphas_bar_sqrt]
                return_dict['support_features'] = support_features
                return_dict['target_features'] = target_features
            return return_dict
        else:
            return_dict = {'logits': - class_dists, "class_logits": class_text_logits}

            return return_dict

    def loss(self, task_dict, model_dict):
        return F.cross_entropy(model_dict["logits"], task_dict["target_labels"].long())


