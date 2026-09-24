# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared loader for RAM++ (transformers 5.x compat shims).

Used by the build/validate scripts. Loads the vendored model with
weights; conversion scripts then monkeypatch GPU-clean forwards.
"""
import os
import sys
from types import ModuleType
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "recognize-anything")
WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "weights", "ram_plus_swin_large_14m.pth")

def _install_shims():
    """Installs fairscale stubs + transformers 5.x compat monkeypatches."""
    sys.path.insert(0, REPO)
    # stub fairscale (imported by ram/vit.py; unused for swin_l)
    def _stub(name):
        m = ModuleType(name)
        sys.modules[name] = m
        return m
    _stub("fairscale")
    _stub("fairscale.nn")
    _stub("fairscale.nn.checkpoint")
    _cp = _stub("fairscale.nn.checkpoint.checkpoint_activations")
    _cp.checkpoint_wrapper = lambda m=None, *a, **k: m
    # transformers 5.x moved/removed names the vendored bert.py imports
    import transformers.modeling_utils as _mu
    import transformers.pytorch_utils as _pu
    _mu.apply_chunking_to_forward = _pu.apply_chunking_to_forward
    _mu.prune_linear_layer = _pu.prune_linear_layer
    _mu.find_pruneable_heads_and_indices = getattr(
        _pu, "find_pruneable_heads_and_indices", lambda *a, **k: (set(), None))
    from transformers.modeling_utils import PreTrainedModel as _PTM
    _PTM.init_weights = lambda self, *a, **k: None          # ckpt overwrites
    _PTM.post_init = lambda self, *a, **k: None
    if not hasattr(_PTM, "all_tied_weights_keys"):
        _PTM.all_tied_weights_keys = property(lambda self: {})
    if not hasattr(_PTM, "get_head_mask"):
        _PTM.get_head_mask = (
            lambda self, hm, n, is_attention_chunked=False: [None] * n)
    def _ext(self, attention_mask, input_shape, device=None, dtype=None):
        dtype = dtype or torch.float32
        ext = attention_mask[:, None, None, :].to(dtype)
        return (1.0 - ext) * torch.finfo(dtype).min
    _PTM.get_extended_attention_mask = _ext
    from transformers import BertTokenizer as _BT
    _orig = _BT.from_pretrained.__func__
    # patch enc_token_id path handled in utils.py source already

def load_ram_plus(image_size=384):
    """Loads the vendored RAM++ swin_l model with pretrained weights.

    Args:
        image_size: Input image size the model is configured for.

    Returns:
        The eval-mode RAM++ model.
    """
    _install_shims()
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        from ram.models import ram_plus
        model = ram_plus(pretrained=WEIGHTS, image_size=image_size,
                         vit="swin_l")
        model.eval()
    finally:
        os.chdir(cwd)
    return model
