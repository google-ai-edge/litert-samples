import build_cpga as B, numpy as np, torch, os, sys
from PIL import Image
HERE=os.path.dirname(os.path.abspath(__file__)); S=B.SIZE
p=sys.argv[1] if len(sys.argv)>1 else "low.jpg"
im=Image.open(p).convert("RGB"); s=min(im.size); im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=(np.asarray(im).astype(np.float32)/255.0).transpose(2,0,1)[None].copy()
m=B.build()
with torch.no_grad(): out=m(torch.from_numpy(x)).numpy()
np.save(f"{HERE}/cp_ref.npy",out); x.tofile(f"{HERE}/cp_input.bin"); im.save(f"{HERE}/cp_in.png")
Image.fromarray((out[0].transpose(1,2,0)*255).clip(0,255).astype(np.uint8)).save(f"{HERE}/cp_torch.png")
print(f"input mean {x.mean():.3f} (dark); torch out mean {out.mean():.3f} (enhanced)")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/cpga_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],x.astype(d["dtype"])); it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),out.ravel())[0,1]:.6f}")
