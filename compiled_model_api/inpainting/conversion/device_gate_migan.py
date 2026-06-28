import build_migan as B, numpy as np, torch, os
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__)); R=B.RES
im=Image.open(f"{HERE}/scene.jpg").convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((R,R),Image.BILINEAR)
arr=np.asarray(im).astype(np.float32)/255.0
img=(arr-0.5)*2  # [-1,1] HWC
# mask: 1=keep, 0=fill. Remove an elliptical region (lower-center, where a subject often sits)
m=Image.new("L",(R,R),255); d=ImageDraw.Draw(m)
cx,cy=int(R*0.5),int(R*0.62); rw,rh=int(R*0.17),int(R*0.30)
d.ellipse([cx-rw,cy-rh,cx+rw,cy+rh],fill=0)
mask=(np.asarray(m).astype(np.float32)/255.0)[...,None]  # HWC 1ch, 1 keep/0 fill
imgc=img.transpose(2,0,1)[None]; maskc=mask.transpose(2,0,1)[None]
x=np.concatenate([maskc-0.5, imgc*maskc],axis=1).astype(np.float32)  # [1,4,R,R]
m_=B.build()
with torch.no_grad(): out=m_(torch.from_numpy(x)).numpy()  # [-1,1]
np.save(f"{HERE}/mg_ref.npy",out); x.tofile(f"{HERE}/mg_input.bin"); np.save(f"{HERE}/mg_mask.npy",maskc); np.save(f"{HERE}/mg_img.npy",imgc)
def comp(o): 
    c=imgc*maskc+o*(1-maskc); c=((c[0].transpose(1,2,0)*0.5+0.5)*255).clip(0,255).astype(np.uint8); return Image.fromarray(c)
comp(out).save(f"{HERE}/mg_torch.png")
masked=((imgc*maskc)[0].transpose(1,2,0)*0.5+0.5)*255; Image.fromarray(masked.clip(0,255).astype(np.uint8)).save(f"{HERE}/mg_masked.png"); im.save(f"{HERE}/mg_in.png")
print(f"input {x.shape}; torch out range [{out.min():.2f},{out.max():.2f}]")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/migan_fp16.tflite"); it.allocate_tensors()
dd=it.get_input_details()[0]; it.set_tensor(dd["index"],x.astype(dd["dtype"])); it.invoke()
o=it.get_tensor(it.get_output_details()[0]["index"])
print(f"desktop-fp16 vs torch corr {np.corrcoef(o.ravel(),out.ravel())[0,1]:.6f}")
