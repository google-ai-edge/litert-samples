import build_rtm_face as B, numpy as np, torch, os
from PIL import Image, ImageDraw
HERE=os.path.dirname(os.path.abspath(__file__)); S=256
MEAN=np.array([123.675,116.28,103.53],np.float32); STD=np.array([58.395,57.12,57.375],np.float32)
# WFLW 98-landmark groups (polylines / loops)
GROUPS=[("contour",list(range(0,33)),False),("browL",list(range(33,42)),False),("browR",list(range(42,51)),False),
        ("noseB",list(range(51,55)),False),("noseT",list(range(55,60)),False),
        ("eyeL",list(range(60,68)),True),("eyeR",list(range(68,76)),True),
        ("mouthO",list(range(76,88)),True),("mouthI",list(range(88,96)),True)]
im=Image.open(f"{HERE}/face.jpg").convert("RGB"); s=min(im.size)
im=im.crop(((im.width-s)//2,(im.height-s)//2,(im.width+s)//2,(im.height+s)//2)).resize((S,S),Image.BILINEAR)
x=((np.asarray(im).astype(np.float32)-MEAN)/STD).transpose(2,0,1)[None].copy()
from mmpose.apis import init_model
m=init_model(B.cfg,B.ckpt,device="cpu").eval(); w=B.Wrap(m).eval()
with torch.no_grad(): sx,sy=w(torch.from_numpy(x)); sx,sy=sx.numpy()[0],sy.numpy()[0]
np.save(f"{HERE}/fc_sx.npy",sx); np.save(f"{HERE}/fc_sy.npy",sy); x.tofile(f"{HERE}/fc_input.bin")
def decode(sx,sy): return np.stack([sx.argmax(1)/2.0, sy.argmax(1)/2.0],1)
def draw(kp,name):
    d=im.copy(); dr=ImageDraw.Draw(d)
    for _,ids,closed in GROUPS:
        pts=[(kp[i,0],kp[i,1]) for i in ids]
        dr.line(pts+([pts[0]] if closed else []),fill=(0,230,0),width=2)
    for i in range(98): dr.ellipse([kp[i,0]-1.5,kp[i,1]-1.5,kp[i,0]+1.5,kp[i,1]+1.5],fill=(255,40,40))
    d.save(name)
draw(decode(sx,sy),f"{HERE}/fc_torch.png"); im.save(f"{HERE}/fc_in.png")
print(f"input {x.shape}; 98 landmarks decoded")
from ai_edge_litert.interpreter import Interpreter
it=Interpreter(model_path=f"{HERE}/rtm_face_fp16.tflite"); it.allocate_tensors()
d=it.get_input_details()[0]; it.set_tensor(d["index"],x.astype(d["dtype"])); it.invoke()
od=it.get_output_details(); o0=it.get_tensor(od[0]["index"])[0]; o1=it.get_tensor(od[1]["index"])[0]
csx=np.corrcoef(o0.ravel(),sx.ravel())[0,1]; ox,oy=(o0,o1) if csx>0.9 else (o1,o0)
print(f"desktop-fp16 corr sx {np.corrcoef(ox.ravel(),sx.ravel())[0,1]:.5f} sy {np.corrcoef(oy.ravel(),sy.ravel())[0,1]:.5f}; out0_is_sx {csx>0.9}")
