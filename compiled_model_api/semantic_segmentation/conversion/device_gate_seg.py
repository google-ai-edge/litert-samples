import build_lraspp as B, numpy as np, torch, os, glob
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__))
HUB=os.path.expanduser("~/.cache/torch/hub/yvanyin_metric3d_main")
p=sorted(glob.glob(HUB+"/data/wild_demo/*.jpg"))[2]   # indoor/person-ish scene
MEAN=np.array([0.485,0.456,0.406],np.float32); STD=np.array([0.229,0.224,0.225],np.float32)
im=Image.open(p).convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((B.SIZE,B.SIZE),Image.BILINEAR)
arr=np.asarray(im).astype(np.float32)/255.0
img_np=((arr-MEAN)/STD).transpose(2,0,1)[None].copy()
m=B.build()
with torch.no_grad(): ref=m(torch.from_numpy(img_np)).numpy()  # [1,512,512,21]
np.save(HERE+"/seg_ref.npy",ref); img_np.tofile(HERE+"/seg_input.bin")
im.save(HERE+"/seg_rgb.png")
print(f"input {img_np.shape}; ref {ref.shape} argmax-classes {np.unique(ref[0].argmax(-1))[:8]}")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=HERE+"/lraspp_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],img_np.astype(d["dtype"])); it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),ref.ravel())[0,1]:.6f}; argmax agree {(o[0].argmax(-1)==ref[0].argmax(-1)).mean()*100:.1f}%")
