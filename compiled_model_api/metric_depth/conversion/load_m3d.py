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

"""Import-time stubs that let Metric3D's torch-hub hubconf load without mmcv.

Patches inspect.getsourcefile (probed with builtins during hub load) and forces
hubconf's try-mmcv Config/DictAction import to fall back to mmengine.
"""
import sys
import types
import inspect
_orig = inspect.getsourcefile
def _safe_gsf(o):
    """inspect.getsourcefile variant that returns None instead of raising.

    Args:
        o: Object whose source file is looked up.

    Returns:
        The source file path, or None when the lookup raises.
    """
    try:
        return _orig(o)
    except Exception:
        return None
inspect.getsourcefile = _safe_gsf

# force hubconf's try-mmcv to fall back to mmengine
_REJECT = {"Config", "DictAction"}
class _Stub(types.ModuleType):
    __file__ = "<stub>"
    __spec__ = None
    __path__ = []
    def __getattr__(self, n):
        if n.startswith("__") or n in _REJECT:
            raise AttributeError(n)
        return lambda *a, **k: None
for name in ["mmcv", "mmcv.utils", "mmcv.cnn", "mmcv.ops", "mmcv.runner"]:
    sys.modules[name] = _Stub(name)
sys.modules["mmcv.utils"].collect_env = lambda *a, **k: {}
import torch

def load():
    """Loads the pretrained metric3d_vit_small model from torch hub.

    Returns:
        The Metric3D v2 ViT-S model in eval mode.
    """
    return torch.hub.load('yvanyin/metric3d', 'metric3d_vit_small',
                          pretrain=True, trust_repo=True).eval()

if __name__ == "__main__":
    m = load()
    print("loaded:", type(m).__name__, "| params",
          round(sum(p.numel() for p in m.parameters())/1e6,1), "M")
    for n,c in m.named_children():
        print("  ", n, ":", type(c).__name__, "|",
              [nn for nn,_ in c.named_children()][:10])
