"""
gsam_video_gui.py  ─────────────────────────────────────────────
Grounded-SAM-2  video annotator (mask-dict + persist + delete + single-preview)
------------------------------------------------------------------
1. key → LOAD  (automatically read outputs/<key>/masks/mask_dict.pkl)
2. DETECT / positive/negative click → CONFIRM write into mask_dict
3. SAVE mask_dict  /  DELETE Object (by oid)
4. PROPAGATE: use current frame as seed for Sam-2 video propagation, keep global oid
   - remove duplicates if same name & IoU>0.95  - save colored visualization into annotated_images
------------------------------------------------------------------
Outputs:
    - outputs/<key>/scene/scene.pkl  (update with "mask")
Note:
    - "mask" : a list N frames of [
        {
            "oid": {
                "name": # object name,
                "bbox": # [x1,y1,x2,y2],
                "mask": # [H,W] boolean array,
            },
            ...
        }
    ]
"""

from __future__ import annotations
import re, cv2, sys, torch, pickle, gradio as gr, numpy as np
from pathlib import Path
from typing import Dict, Any
import supervision as sv

# ────────── Model loading ──────────
ROOT  = Path.cwd()
THIRD = ROOT / "third_party/Grounded-SAM-2"; 
sys.path.append(str(THIRD))

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

DEV  = "cuda" if torch.cuda.is_available() else "cpu"
CFG  = "configs/sam2.1/sam2.1_hiera_l.yaml"
CKPT = "third_party/Grounded-SAM-2/checkpoints/sam2.1_hiera_large.pt"

video_pred = build_sam2_video_predictor(CFG, CKPT)
img_pred   = SAM2ImagePredictor(build_sam2(CFG, CKPT))
dino_proc  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                 "IDEA-Research/grounding-dino-base").to(DEV)

OUT_ROOT = ROOT / "outputs"; OUT_ROOT.mkdir(exist_ok=True)

IMGBOX_H = 500  # display height in pixels, same as gr.Image(height=500)

# ────────── utils ──────────
def natural(p: Path):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', p.stem)]

def load_frames(folder: Path):
    pics = sorted(sum((list(folder.glob(f"*.{e}")) for e in ["jpg","jpeg","png"]), []), key=natural)
    frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in pics]
    return pics, np.stack(frames, 0) if pics else ([], np.empty((0,)))

def overlay(img, mask, color, a=.45):
    out = img.copy()
    out[mask] = out[mask]*(1-a)+np.array(color)*a
    return out.astype(np.uint8)

def to_original_xy(x_disp: int, y_disp: int, H: int, W: int) -> tuple[int, int]:
    """
    Map display coordinates (from Gradio Image.select) back to original pixel coordinates.
    Assumes the image is rendered at fixed display height IMGBOX_H with preserved aspect ratio.
    """
    if H <= 0 or W <= 0:
        return x_disp, y_disp
    scale = IMGBOX_H / float(H)  # browser uses same scale on both axes
    x = int(round(x_disp / scale))
    y = int(round(y_disp / scale))
    # clamp to bounds
    x = max(0, min(W - 1, x))
    y = max(0, min(H - 1, y))
    return x, y

# ────────── supervision annotators (reuse) ──────────
MASK_ANN  = sv.MaskAnnotator()
BOX_ANN   = sv.BoxAnnotator()
LABEL_ANN = sv.LabelAnnotator()

def draw_objects(img: np.ndarray, objs: Dict[int, Dict]) -> np.ndarray:
    """Draw all masks/bboxes/text in objs onto img at once"""
    if not objs:
        return img

    masks = np.stack([o["mask"] for o in objs.values()])
    xyxy  = np.stack([o["bbox"] for o in objs.values()])
    class_ids = np.array(list(objs.keys()))
    det = sv.Detections(xyxy=xyxy, mask=masks, class_id=class_ids)

    img = MASK_ANN.annotate(img, det)
    img = BOX_ANN .annotate(img, det)
    labels = [f"{oid}_{objs[oid]['name']}" for oid in det.class_id]
    img = LABEL_ANN.annotate(img, det, labels=labels)
    return img

# ────────── mask_dict I/O ──────────
def save_mask_dict(out_dir: Path, mdict: Dict[int,Dict]):
    scene_path = out_dir / "scene/scene.pkl"
    with open(scene_path, "rb") as f:
        scene_dict = pickle.load(f)
    scene_dict["mask"] = mdict
    with open(out_dir / "scene/scene.pkl", "wb") as f:
        pickle.dump(scene_dict, f)
    
    vis_dir=OUT_ROOT/S["key"]/ "annotated_images"; vis_dir.mkdir(parents=True,exist_ok=True)
    T=len(S["frames"])
    for i in range(T):
        cv2.imwrite(str(vis_dir/f"{i:06d}.jpg"),cv2.cvtColor(render(i),cv2.COLOR_RGB2BGR))


def load_mask_dict(pkl: Path):
    with open(pkl,"rb") as f: return pickle.load(f)

# ────────── IoU & insertion utils ──────────
def mask_iou(a: np.ndarray, b: np.ndarray)->float:
    inter=np.logical_and(a,b).sum(); union=np.logical_or(a,b).sum()
    return 0. if union==0 else inter/union

def add_mask_unique(fid:int,name:str,bbox,mask,*,iou_thr=.95,oid:int|None=None):
    S["mask_dict"].setdefault(fid,{})
    for ex_oid,o in S["mask_dict"][fid].items():
        if o["name"]==name and mask_iou(o["mask"],mask)>iou_thr:
            o["bbox"],o["mask"]=bbox,mask; return
    if oid is None:
        oid=S["next_oid"]; S["next_oid"]+=1
    S["mask_dict"][fid][oid]={"name":name,"bbox":bbox,"mask":mask}

# ────────── DINO / SAM / video (same as before, slightly modified) ──────────
def run_dino(img,txt):
    batch=dino_proc(images=[img],text=txt,return_tensors="pt")
    batch={k:v.to(DEV) for k,v in batch.items()}
    with torch.no_grad():
        out=dino_model(pixel_values=batch["pixel_values"],
                       input_ids=batch["input_ids"],
                       attention_mask=batch.get("attention_mask"))
    res=dino_proc.post_process_grounded_object_detection(out,batch["input_ids"],
                                                         .25,.3,target_sizes=[img.shape[:2]])
    return res[0]["boxes"].cpu().numpy()

def mask_from_boxes(img,boxes):
    img_pred.set_image(img)
    m,*_=img_pred.predict(box=boxes,multimask_output=False)
    return (m.squeeze(1) if m.ndim==4 else m)>.5

def mask_from_points(img,pts):
    if not pts: return None
    img_pred.set_image(img)
    pc=np.array([[x,y] for x,y,_ in pts]); pl=np.array([l for *_,l in pts])
    m,*_=img_pred.predict(point_coords=pc,point_labels=pl,multimask_output=False)
    return (m.squeeze(1) if m.ndim==4 else m)>.5

def propagate(frames,start,seeds,resized_dir:Path):
    st=video_pred.init_state(video_path=str(resized_dir))
    for oid,m in seeds: video_pred.add_new_mask(st,start,oid,m)
    T,H,W=frames.shape[:3]; O=len(seeds)
    allm=np.zeros((T,O,H,W),bool)
    for f,ids,logits in video_pred.propagate_in_video(st):
        for i,oid in enumerate(ids): allm[f,oid-1]=(logits[i]>0).cpu().numpy()
    return allm

# ────────── Global state ──────────
S: Dict[str,Any]={"mask_dict":{}, "next_oid":1}

# ────────── Rendering ──────────
def render(idx:int):
    if "frames" not in S or idx>=len(S["frames"]): return np.zeros((60,60,3),np.uint8)
    img=draw_objects(S["frames"][idx].copy(),S["mask_dict"].get(idx,{}))
    if idx==S.get("cur",-1) and S.get("pending") is not None:
        img=overlay(img,S["pending"].any(0),(255,0,0))
    for x,y,l in S.get("click",[]): cv2.drawMarker(img,(x,y),(0,255,0) if l else (255,0,0),8,2)
    return img

def prepare_data(key:str):
    scene_path = ROOT / Path(f"outputs/{key}/scene/scene.pkl")
    if not scene_path.is_file():
        return None, False, f"⚠️ scene.pkl not found in outputs/{key}/scene/"
    with open(scene_path, "rb") as f:
        scene_data = pickle.load(f)
    images = scene_data["images"]
    for i in range(len(images)):
        cv2.imwrite(str(OUT_ROOT/key/"resized_images"/f"{i:06d}.jpg"), cv2.cvtColor(images[i], cv2.COLOR_RGB2BGR))
    return scene_data, True, f"✅ Prepared {len(images)} resized images from scene.pkl"

# ────────── Callbacks ──────────
def cb_load_key(k:str):
    k=k.strip()
    
    resized=OUT_ROOT/k/"resized_images"
    resized.mkdir(parents=True,exist_ok=True)
    scene_dict, success, info = prepare_data(k)  # prepare resized images from scene.pkl if not exist    
    if not success: return gr.update(),gr.update(),info
    if not k or not resized.exists(): return gr.update(),gr.update(),"⚠️ key/resized_images does not exist"
    _, frames=load_frames(resized);  n=len(frames)
    if n==0: return gr.update(),gr.update(),"⚠️ No frame images"
    
    mdict=scene_dict["mask"] if "mask" in scene_dict else {}

    all_ids=[oid for frame in mdict.values() for oid in frame.keys()]
    next_oid = max(all_ids)+1 if all_ids else 1

    S.clear(); S.update(key=k,resized_dir=resized,frames=frames,
                        cur=0,click=[],pending=None,mask_dict=mdict,next_oid=next_oid)
    slider_state=gr.update(minimum=0,maximum=n-1,value=0,step=1,interactive=n>1)
    return render(0),slider_state,f"✅ Loaded {n} frames (mask:{bool(mdict)})"

def cb_slider(i:int):
    S["cur"]=int(i); S["pending"]=None; S.pop("click",None)
    return render(S["cur"])

def cb_click(evt: gr.SelectData, typ: str):
    if "frames" not in S:
        return gr.update()
    x, y = int(evt.index[0]), int(evt.index[1])  # display coords == original coords when no scaling
    S.setdefault("click", []).append((x, y, 1 if typ == "positive" else 0))
    S["pending"] = mask_from_points(S["frames"][S["cur"]], S["click"])
    return render(S["cur"])

def cb_undo():
    if S.get("click"): S["click"].pop()
    S["pending"]=mask_from_points(S["frames"][S["cur"]],S["click"])
    return render(S["cur"])

def cb_detect(img,txt):
    txt=txt.strip()
    if not txt: return "⚠️ Empty prompt", render(S["cur"])
    boxes=run_dino(img,txt)
    if len(boxes)==0: return "⚠️ No objects detected",render(S["cur"])
    S["pending"]=mask_from_boxes(img,boxes)
    S["pending_info"]=[{"name":txt,"bbox":b} for b in boxes]
    return f"🔵 Preview {len(boxes)} mask(s), Confirm→Save", render(S["cur"])

def cb_confirm(name:str):
    if S.get("pending") is None: return "⚠️ No pending mask",render(S["cur"])
    idx,masks=S["cur"],S["pending"]
    infos=S.pop("pending_info",None) or [{"name":name or "pc_obj","bbox":sv.mask_to_xyxy(masks[0:1])[0]}]
    for mi,info in enumerate(infos):
        m=masks[mi] if masks.ndim==3 else masks
        add_mask_unique(idx,info["name"],info["bbox"],m)
    S["pending"]=None; S.pop("click",None)
    return "✅ Confirmed", render(idx)

def cb_save_dict():
    if "key" not in S: return "⚠️ Not loaded", render(S.get("cur",0))
    save_mask_dict(OUT_ROOT/S["key"],S["mask_dict"])
    return "💾 mask_dict saved", render(S["cur"])

def cb_delete(oid_txt:str):
    if not oid_txt.strip().isdigit(): return "⚠️ oid must be an integer", render(S["cur"])
    oid=int(oid_txt.strip()); cnt=0
    for objs in S["mask_dict"].values():
        if oid in objs: objs.pop(oid); cnt+=1
    if cnt==0: return f"ℹ️ oid {oid} not found", render(S["cur"])
    save_mask_dict(OUT_ROOT/S["key"],S["mask_dict"])
    return f"🗑️ Deleted oid {oid} (appeared in {cnt} frames)", render(S["cur"])

def cb_prop():
    cur_idx=S.get("cur",0); cur_objs=S["mask_dict"].get(cur_idx,{})
    if not cur_objs: return "⚠️ No confirmed objects in current frame", render(cur_idx)
    seeds=[(oid,o["mask"]) for oid,o in cur_objs.items()]
    state=video_pred.init_state(video_path=str(S["resized_dir"]))
    for oid,m in seeds: video_pred.add_new_mask(state,cur_idx,oid,m)
    T=len(S["frames"]); allm={fi:{} for fi in range(T)}
    for fi,ids,log in video_pred.propagate_in_video(state):
        for ii,oid in enumerate(ids): allm[fi][oid]=(log[ii]>0).cpu().numpy()
    for fi,objs in allm.items():
        for oid,m in objs.items():
            bbox=sv.mask_to_xyxy(np.squeeze(m)[None])[0]
            name=S["mask_dict"][cur_idx][oid]["name"]
            add_mask_unique(fi,name,bbox,np.squeeze(m),iou_thr=.99,oid=oid)
    save_mask_dict(OUT_ROOT/S["key"],S["mask_dict"])

    return "✅ Propagation finished and saved", render(cur_idx)

def cb_reset(confirm_check):
    """Reset all annotations: clear mask_dict, reset object ID counter, and clean up state"""
    if "key" not in S: return "⚠️ Not loaded", render(S.get("cur",0)), gr.update(value=False)

    if not confirm_check:
        return "⚠️ Please check the confirmation box to reset", render(S.get("cur",0)), gr.update(value=False)

    # Clear all mask data
    S["mask_dict"] = {}
    S["next_oid"] = 1
    S["pending"] = None
    S.pop("click", None)
    S.pop("pending_info", None)

    # Save the empty mask_dict to persist the reset
    save_mask_dict(OUT_ROOT/S["key"], S["mask_dict"])

    return "🔄 All annotations reset", render(S.get("cur", 0)), gr.update(value=False)

# ────────── GUI ──────────
css_prevent_image_drag = """
#sam-image-box img {
    user-drag: none;
    -webkit-user-drag: none;
}
"""


with gr.Blocks(title="Grounded-SAM-2 Annotator", css=css_prevent_image_drag) as demo:
    gr.Markdown("1️⃣ LOAD → 2️⃣ DETECT/click→CONFIRM → 3️⃣ SAVE / DELETE / PROPAGATE")

    key_in, load_btn = gr.Text(label="Output-key"), gr.Button("LOAD")

    with gr.Row():
        prompt_in = gr.Text(label="Prompt")
        click_mode = gr.Radio(["positive","negative"], value="positive", label="Click type")
        pc_name = gr.Text(label="Point-click name", value="pc_obj")

    with gr.Row():
        det_btn, undo_btn = gr.Button("Detect"), gr.Button("Undo click")
        confirm_btn, save_btn = gr.Button("Confirm mask"), gr.Button("Save mask_dict")

    with gr.Row():
        del_oid = gr.Text(label="Delete oid")
        del_btn = gr.Button("Delete Object")

    gr.Markdown("---")
    gr.Markdown("⚠️ **DANGER ZONE**: Reset will permanently delete all annotations and cannot be undone!")
    with gr.Row():
        reset_confirm_check = gr.Checkbox(label="I understand this will delete all masks and reset object IDs", value=False)
        reset_btn = gr.Button("⚠️ Reset All Annotations", variant="stop")

    slider = gr.Slider(0,1,1,label="Frame",interactive=False)
    imgbox = gr.Image(type="numpy",label="Preview", elem_id="sam-image-box")

    prop_btn, logbox = gr.Button("PROPAGATE & SAVE"), gr.Textbox(label="Log")

    # Event binding
    load_btn.click(cb_load_key, key_in, [imgbox,slider,logbox])
    slider.change(cb_slider, slider, imgbox)

    imgbox.select(cb_click,[click_mode],imgbox)
    undo_btn.click(cb_undo,None,imgbox)

    det_btn.click(cb_detect,[imgbox,prompt_in],[logbox,imgbox])
    confirm_btn.click(cb_confirm, pc_name,[logbox,imgbox])
    save_btn.click(cb_save_dict, None,[logbox,imgbox])
    del_btn.click(cb_delete, del_oid,[logbox,imgbox])
    prop_btn.click(cb_prop, None,[logbox,imgbox])

    # Reset button event
    reset_btn.click(cb_reset, reset_confirm_check, [logbox, imgbox, reset_confirm_check])


# ────────── CLI ──────────
if __name__=="__main__":
    import argparse, os
    ap=argparse.ArgumentParser()
    ap.add_argument("--key",type=str,default="")
    ap.add_argument("--share",type=bool,default=False)
    ap.add_argument("--port",type=int,default=7860)
    args=ap.parse_args()

    if args.key: key_in.value=args.key
    demo.launch(share=args.share,server_port=args.port,server_name="0.0.0.0")