from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import shutil
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parent / "runtime" / "recoil_profiles"
HOST = "127.0.0.1"
PORT = 8766
DATA_URL = re.compile(r"^data:(?P<mime>[-\w.+/]+);base64,(?P<data>.+)$")


class ProfileError(ValueError):
    pass


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "recoil-profile"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def fnum(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"Invalid number: {value!r}") from exc


def fint(value: Any) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ProfileError(f"Invalid integer: {value!r}") from exc


def safe_join(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    root = root.resolve()
    if root not in path.parents and path != root:
        raise ProfileError("Asset path escapes the recoil directory.")
    return path


def decode_data_url(value: str) -> tuple[bytes, str]:
    match = DATA_URL.match(value)
    if not match:
        raise ProfileError("Image data must be a base64 data URL.")
    try:
        raw = base64.b64decode(match.group("data"), validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ProfileError("Image data is not valid base64.") from exc
    ext = mimetypes.guess_extension(match.group("mime")) or ".png"
    return raw, ".jpg" if ext == ".jpe" else ext


def parse_profile_id_path(path: str) -> str:
    parts = [part for part in path.split("?", 1)[0].split("/") if part]
    if len(parts) < 4 or parts[:3] != ["api", "recoil", "profiles"]:
        raise ProfileError("Profile id is required.")
    return slugify(unquote(parts[-1]))


class Repository:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.assets = root / "assets"
        self.root.mkdir(parents=True, exist_ok=True)
        self.assets.mkdir(parents=True, exist_ok=True)

    def profile_path(self, profile_id: str) -> Path:
        return self.root / f"{profile_id}.json"

    def asset_dir(self, profile_id: str) -> Path:
        return self.assets / profile_id

    def list_profiles(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                payload = self._read(path.stem)
                items.append(
                    {
                        "id": payload["id"],
                        "name": payload["name"],
                        "updated_at": payload.get("updated_at", ""),
                        "shot_count": len(payload["steps"]),
                        "scale_factor": payload["scale_factor"],
                        "horizontal_scale_factor": payload.get("horizontal_scale_factor", payload["scale_factor"]),
                        "fire_interval_ms": payload["fire_interval_ms"],
                        "valid": True,
                        "error": "",
                    }
                )
            except ProfileError as exc:
                items.append(
                    {
                        "id": path.stem,
                        "name": path.stem,
                        "updated_at": "",
                        "shot_count": 0,
                        "scale_factor": 0,
                        "horizontal_scale_factor": 0,
                        "fire_interval_ms": 0,
                        "valid": False,
                        "error": str(exc),
                    }
                )
        items.sort(key=lambda x: (not x["valid"], x["name"], x["id"]))
        return items

    def get(self, profile_id: str) -> dict[str, Any]:
        payload = self._read(profile_id)
        for image in payload["images"]:
            image["asset_url"] = "/" + image["asset_path"].replace("\\", "/")
        return payload

    def save(self, payload: dict[str, Any], profile_id: str | None = None) -> dict[str, Any]:
        current_id = slugify(profile_id) if profile_id else ""
        existing = self.get(current_id) if current_id and self.profile_path(current_id).exists() else None
        data = self._normalize(payload, current_id or None, existing, writing=True)
        final_id = data["id"]
        data["created_at"] = (existing or {}).get("created_at") or data.get("created_at") or now_iso()
        data["updated_at"] = now_iso()

        blobs: list[tuple[dict[str, Any], bytes, str]] = []
        for image in data["images"]:
            if image.get("data_url"):
                blob, ext = decode_data_url(str(image["data_url"]))
            else:
                source = safe_join(self.root, str(image.get("asset_path") or ""))
                if not source.exists():
                    raise ProfileError(f"Missing asset for image '{image['id']}'.")
                blob, ext = source.read_bytes(), source.suffix or ".png"
            blobs.append((image, blob, ext))

        asset_dir = self.asset_dir(final_id)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)
        asset_dir.mkdir(parents=True, exist_ok=True)
        clean_images: list[dict[str, Any]] = []
        for index, (image, blob, ext) in enumerate(blobs):
            filename = f"{index + 1:02d}-{slugify(image.get('name') or image['id'])}{ext}"
            (asset_dir / filename).write_bytes(blob)
            clean_images.append(
                {
                    "id": image["id"],
                    "name": image["name"],
                    "asset_path": f"assets/{final_id}/{filename}",
                    "width": image["width"],
                    "height": image["height"],
                    "offset_x": image["offset_x"],
                    "offset_y": image["offset_y"],
                    "markers": image["markers"],
                }
            )
        data["images"] = clean_images
        self.profile_path(final_id).write_text(json.dumps(data, indent=2), encoding="utf-8")
        return self.get(final_id)

    def delete(self, profile_id: str) -> None:
        path = self.profile_path(profile_id)
        if not path.exists():
            raise ProfileError(f"Recoil profile '{profile_id}' does not exist.")
        path.unlink()
        asset_dir = self.asset_dir(profile_id)
        if asset_dir.exists():
            shutil.rmtree(asset_dir)

    def _read(self, profile_id: str) -> dict[str, Any]:
        path = self.profile_path(profile_id)
        if not path.exists():
            raise ProfileError(f"Recoil profile '{profile_id}' was not found.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ProfileError(f"Invalid JSON in '{path.name}'.") from exc
        return self._normalize(payload, profile_id, payload, writing=False)

    def _normalize(self, payload: dict[str, Any], current_id: str | None, existing: dict[str, Any] | None, *, writing: bool) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ProfileError("Profile payload must be a JSON object.")
        name = str(payload.get("name") or "Untitled Profile").strip() or "Untitled Profile"
        profile_id = slugify(current_id or str(payload.get("id") or name))
        if writing and not current_id:
            suffix = 2
            base_id = profile_id
            while self.profile_path(profile_id).exists():
                profile_id = f"{base_id}-{suffix}"
                suffix += 1

        scale = fnum(payload.get("scale_factor", 0))
        horizontal_scale = fnum(payload.get("horizontal_scale_factor", payload.get("scale_factor", 0)))
        interval = fint(payload.get("fire_interval_ms", 0))
        if scale <= 0:
            raise ProfileError("scale_factor must be greater than 0.")
        if horizontal_scale <= 0:
            raise ProfileError("horizontal_scale_factor must be greater than 0.")
        if interval <= 0:
            raise ProfileError("fire_interval_ms must be greater than 0.")

        raw_images = payload.get("images")
        if not isinstance(raw_images, list) or not raw_images:
            raise ProfileError("At least one image is required.")
        images: list[dict[str, Any]] = []
        markers: dict[tuple[str, str], tuple[float, float]] = {}
        for image_index, image in enumerate(raw_images):
            if not isinstance(image, dict):
                raise ProfileError("Image entries must be objects.")
            image_id = str(image.get("id") or f"image_{image_index + 1}").strip() or f"image_{image_index + 1}"
            offset_x = fnum(image.get("offset_x", 0))
            offset_y = fnum(image.get("offset_y", 0))
            raw_marker_list = image.get("markers", [])
            if not isinstance(raw_marker_list, list):
                raise ProfileError("Image markers must be a list.")
            marker_list: list[dict[str, Any]] = []
            for marker_index, marker in enumerate(raw_marker_list):
                if not isinstance(marker, dict):
                    raise ProfileError("Marker entries must be objects.")
                marker_id = str(marker.get("id") or f"marker_{marker_index + 1}").strip() or f"marker_{marker_index + 1}"
                point = {"id": marker_id, "x": fnum(marker.get("x", 0)), "y": fnum(marker.get("y", 0))}
                marker_list.append(point)
                markers[(image_id, marker_id)] = (point["x"] + offset_x, point["y"] + offset_y)
            images.append(
                {
                    "id": image_id,
                    "name": str(image.get("name") or image_id).strip() or image_id,
                    "width": max(0, fint(image.get("width", 0))),
                    "height": max(0, fint(image.get("height", 0))),
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "markers": marker_list,
                    "asset_path": str(image.get("asset_path") or ""),
                    "data_url": image.get("data_url") or "",
                }
            )

        raw_steps = payload.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            raise ProfileError("At least one step is required.")
        steps: list[dict[str, Any]] = []
        for index, step in enumerate(raw_steps):
            if not isinstance(step, dict):
                raise ProfileError("Step entries must be objects.")
            source_image_id = str(step.get("source_image_id") or "").strip()
            source_marker_id = str(step.get("source_marker_id") or "").strip()
            px = fnum(step.get("pattern_x", 0))
            py = fnum(step.get("pattern_y", 0))
            if source_image_id and source_marker_id and (source_image_id, source_marker_id) in markers:
                px, py = markers[(source_image_id, source_marker_id)]
            steps.append(
                {
                    "index": index,
                    "pattern_x": px,
                    "pattern_y": py,
                    "duration_ms": max(1, fint(step.get("duration_ms", interval) or interval)),
                    "source_image_id": source_image_id,
                    "source_marker_id": source_marker_id,
                }
            )
        origin_x, origin_y = steps[0]["pattern_x"], steps[0]["pattern_y"]
        for index, step in enumerate(steps):
            step["index"] = index
            step["pattern_x"] = round(step["pattern_x"] - origin_x, 3)
            step["pattern_y"] = round(step["pattern_y"] - origin_y, 3)
        steps[0]["pattern_x"] = 0.0
        steps[0]["pattern_y"] = 0.0

        return {
            "schema_version": 1,
            "id": profile_id,
            "name": name,
            "created_at": str(payload.get("created_at") or (existing or {}).get("created_at") or ""),
            "updated_at": str(payload.get("updated_at") or (existing or {}).get("updated_at") or ""),
            "scale_factor": scale,
            "horizontal_scale_factor": horizontal_scale,
            "fire_interval_ms": interval,
            "images": images,
            "steps": steps,
        }


def html() -> str:
    return """<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Delta Recoil Editor</title><style>
body{margin:0;background:#11151a;color:#eef6fa;font:14px Segoe UI,sans-serif}main{max-width:1480px;margin:auto;padding:16px;display:grid;gap:14px}.box{background:#192028;border:1px solid #33444f;border-radius:16px;padding:14px;min-width:0}.top{display:flex;flex-wrap:wrap;gap:8px;align-items:center}button,input,select{background:#0f151a;color:#eef6fa;border:1px solid #425560;border-radius:10px;padding:8px 10px}button{cursor:pointer}.primary{background:linear-gradient(135deg,#d7a25b,#ff8c55);border:0;color:#15100d;font-weight:700}.split{display:grid;grid-template-columns:var(--left-pane,62%) 12px minmax(340px,1fr);gap:10px;align-items:start}.pane{min-width:0}.meta{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:8px}.metafield{display:grid;gap:6px}.metafield span{color:#93a8b1;font-size:12px}.images,.shots{display:grid;gap:8px;min-width:0}.shots{max-height:calc(100vh - 240px);overflow:auto;padding-right:4px}.row{display:grid;gap:6px;align-items:center;background:#10171d;border:1px solid #32424d;border-radius:12px;padding:8px}.imgrow{grid-template-columns:1fr auto auto auto}.shotrow{grid-template-columns:52px minmax(72px,1fr) minmax(72px,1fr) minmax(72px,1fr) minmax(120px,1.2fr) auto auto auto auto}.splitter{border-radius:999px;background:#32424d;cursor:col-resize;min-height:260px;align-self:stretch}.splitter:hover{background:#d7a25b}canvas{width:100%;height:520px;display:block;background:#0c1014;border-radius:14px}.muted{color:#93a8b1}.toolbar{display:flex;flex-wrap:wrap;gap:8px;margin:10px 0}.active{outline:1px solid #d7a25b}@media(max-width:1100px){.split{grid-template-columns:1fr}.splitter{display:none}.shots{max-height:none}.shotrow{grid-template-columns:repeat(2,minmax(0,1fr))}}</style></head><body><main>
<section class='box'><h1 style='margin:0 0 8px'>Delta Recoil Profile Editor</h1><div class='top'><select id='profiles'></select><button id='newp'>New</button><button id='refreshp'>Refresh</button><button id='savep' class='primary'>Save</button><button id='deletep'>Delete</button></div><p id='status' class='muted'>Loading...</p></section>
<section id='split' class='split'><section id='authorpane' class='box pane'><h2>Authoring Surface</h2><div class='meta'><label class='metafield'><span>Profile Name</span><input id='name' placeholder='Profile name'></label><label class='metafield'><span>Vertical Recoil Scale (Y)</span><input id='vscale' type='number' step='0.001' min='0.001' placeholder='Vertical scale'></label><label class='metafield'><span>Horizontal Recoil Scale (X)</span><input id='hscale' type='number' step='0.001' min='0.001' placeholder='Defaults to vertical scale'></label><label class='metafield'><span>Fire Interval (ms)</span><input id='interval' type='number' step='1' min='1' placeholder='Fire interval ms'></label><label class='metafield'><span>Canvas Tool</span><select id='tool'><option value='add'>Add Marker</option><option value='drag'>Drag Marker / Image</option><option value='delete'>Delete Marker</option></select></label></div><div class='toolbar'><button id='addimg'>Add Images</button><button id='addstep'>Add Manual Step</button><button id='sync'>Sync Linked Steps</button><button id='resetzoom'>Reset Zoom</button><input id='files' type='file' accept='image/*' multiple hidden></div><div id='images' class='images'></div><canvas id='canvas' width='900' height='520'></canvas><p class='muted'>Left click uses the selected tool. In drag mode, dragging a marker moves that marker, and dragging empty image space moves that image layer. Right drag pans. Mouse wheel zooms. Zoom only changes the view; saved marker coordinates stay in editor space. Higher image rows draw on top of lower ones.</p></section><div id='splitter' class='splitter' title='Drag to resize panes'></div><section id='shotpane' class='box pane'><h2>Shot List</h2><div id='shots' class='shots'></div></section></section>
<script>
const S={profiles:[],p:null,active:"",cache:new Map(),zoom:1,panX:20,panY:20,drag:null,layerDrag:null,pan:null,move:false},$=id=>document.getElementById(id),status=t=>$("status").textContent=t,enc=encodeURIComponent,VW=900,VH=520,PAD=20;
const req=(u,o={})=>fetch(u,o).then(async r=>{const j=await r.json();if(!r.ok) throw new Error(j.error||`HTTP ${r.status}`);return j;});
const empty=()=>({schema_version:1,id:"",name:"",created_at:"",updated_at:"",scale_factor:1,horizontal_scale_factor:1,fire_interval_ms:100,images:[],steps:[]}),uid=p=>`${p}_${Math.random().toString(36).slice(2,10)}`;
const layers=()=>S.p?.images||[], imageOffset=i=>({x:Number(i?.offset_x||0),y:Number(i?.offset_y||0)}), imageGlobalPoint=(i,x,y)=>({x:imageOffset(i).x+Number(x||0),y:imageOffset(i).y+Number(y||0)}), localPointForImage=(i,x,y)=>({x:Number(x||0)-imageOffset(i).x,y:Number(y||0)-imageOffset(i).y}), pointInImage=(i,x,y)=>{const o=imageOffset(i);return x>=o.x&&y>=o.y&&x<=o.x+Number(i.width||0)&&y<=o.y+Number(i.height||0);}, topImageAt=(x,y)=>{for(const img of layers()){if(pointInImage(img,x,y)) return img;} return null;}, active=()=>layers().find(i=>i.id===S.active)||null, guideBounds=()=>{const imgs=layers(); if(!imgs.length) return null; let minX=0,minY=0,maxX=0,maxY=0,seed=false; imgs.forEach(img=>{const o=imageOffset(img),w=Number(img.width||0),h=Number(img.height||0),l=o.x,t=o.y,r=o.x+w,b=o.y+h; if(!seed){minX=l;minY=t;maxX=r;maxY=b;seed=true;} else {minX=Math.min(minX,l);minY=Math.min(minY,t);maxX=Math.max(maxX,r);maxY=Math.max(maxY,b);}}); return {minX,minY,maxX,maxY,width:maxX-minX,height:maxY-minY};}, markerFor=s=>{const i=S.p.images.find(x=>x.id===s.source_image_id);const m=i?i.markers.find(v=>v.id===s.source_marker_id)||null:null;return i&&m?imageGlobalPoint(i,m.x,m.y):null};
const setSplit=v=>{const split=$("split"),x=Math.max(38,Math.min(78,Number(v)||62)); split.style.setProperty("--left-pane",`${x}%`); try{localStorage.setItem("delta_recoil_split",String(x));}catch(_){}}; 
function initSplit(){setSplit((()=>{try{return Number(localStorage.getItem("delta_recoil_split")||62);}catch(_){return 62;}})()); const split=$("split"),bar=$("splitter"); let drag=false; const move=e=>{if(!drag||window.matchMedia("(max-width:1100px)").matches) return; const r=split.getBoundingClientRect(),p=((e.clientX-r.left)/Math.max(r.width,1))*100; setSplit(p);}; const stop=()=>{drag=false; window.removeEventListener("mousemove",move); window.removeEventListener("mouseup",stop);}; bar.addEventListener("mousedown",e=>{if(window.matchMedia("(max-width:1100px)").matches) return; e.preventDefault(); drag=true; window.addEventListener("mousemove",move); window.addEventListener("mouseup",stop);});}
async function ensure(image){if(!image) return null; if(S.cache.has(image.id)) return S.cache.get(image.id); const src=image.data_url||image.asset_url; if(!src) return null; const img=new Image(); await new Promise((ok,err)=>{img.onload=ok;img.onerror=()=>err(new Error(`Failed to load ${image.name}`));img.src=src;}); S.cache.set(image.id,img); return img;}
function syncMeta(){if(!S.p) return; S.p.name=$("name").value.trim(); S.p.scale_factor=Number($("vscale").value||0); S.p.horizontal_scale_factor=Number($("hscale").value||$("vscale").value||0); S.p.fire_interval_ms=Math.max(1,Number($("interval").value||1));}
function renderSelect(sel=S.p?.id||""){const d=$("profiles");d.innerHTML="";let o=document.createElement("option");o.value="";o.textContent="Unsaved / New";d.appendChild(o); for(const p of S.profiles){o=document.createElement("option");o.value=p.id;o.textContent=p.valid?`${p.name} (${p.shot_count} shots)`:`${p.name} [invalid]`;o.disabled=!p.valid;d.appendChild(o);} d.value=sel; if(d.value!==sel) d.value="";}
function renderMeta(){$("name").value=S.p?.name||"";$("vscale").value=String(S.p?.scale_factor??1);$("hscale").value=String(S.p?.horizontal_scale_factor??S.p?.scale_factor??1);$("interval").value=String(S.p?.fire_interval_ms??100);}
function renderImages(){const wrap=$("images");wrap.innerHTML=""; if(!S.p?.images.length){wrap.textContent="No images yet.";return;} S.p.images.forEach((img,idx)=>{const row=document.createElement("div");row.className=`row imgrow ${img.id===S.active?"active":""}`; const b=document.createElement("button");const o=imageOffset(img);b.textContent=`${idx+1}. ${img.name} (${img.markers.length}) @ ${o.x.toFixed(1)}, ${o.y.toFixed(1)}`; b.onclick=async()=>{S.active=img.id;await ensure(img);renderImages();draw();}; row.appendChild(b); [["Up",()=>moveImage(idx,-1),idx===0],["Down",()=>moveImage(idx,1),idx===S.p.images.length-1],["Remove",()=>removeImage(img.id),false]].forEach(([t,fn,dis])=>{const x=document.createElement("button");x.textContent=t;x.disabled=dis;x.onclick=fn;row.appendChild(x);}); wrap.appendChild(row);});}
function renderShots(){const wrap=$("shots");wrap.innerHTML=""; if(!S.p?.steps.length){wrap.textContent="No shots yet.";return;} S.p.steps.forEach((s,i)=>{const row=document.createElement("div");row.className="row shotrow"; const num=document.createElement("strong");num.textContent=`#${i+1}`; row.appendChild(num); const mk=(v,step,cb)=>{const x=document.createElement("input");x.type="number";x.step=step;x.value=String(v);x.onchange=()=>cb(Number(x.value||0));return x;}; row.appendChild(mk(s.pattern_x,"0.001",v=>s.pattern_x=v)); row.appendChild(mk(s.pattern_y,"0.001",v=>s.pattern_y=v)); row.appendChild(mk(s.duration_ms,"1",v=>s.duration_ms=Math.max(1,v||S.p.fire_interval_ms||1))); const src=document.createElement("span");src.className="muted";src.textContent=s.source_image_id&&s.source_marker_id?`${s.source_image_id}:${s.source_marker_id}`:"manual";row.appendChild(src); [["Up",()=>moveStep(i,-1),i===0],["Down",()=>moveStep(i,1),i===S.p.steps.length-1],["Insert",()=>insertStep(i+1),false],["Remove",()=>removeStep(i),false]].forEach(([t,fn,dis])=>{const x=document.createElement("button");x.textContent=t;x.disabled=dis;x.onclick=fn;row.appendChild(x);}); wrap.appendChild(row);});}
function syncSteps(){if(!S.p?.steps.length) return; const pts=S.p.steps.map(s=>{const m=markerFor(s); return m?{x:m.x,y:m.y}:{x:Number(s.pattern_x||0),y:Number(s.pattern_y||0)}}), ox=pts[0].x, oy=pts[0].y; S.p.steps.forEach((s,i)=>{const m=markerFor(s); if(m){s.pattern_x=Number((pts[i].x-ox).toFixed(3)); s.pattern_y=Number((pts[i].y-oy).toFixed(3));} s.index=i; s.duration_ms=Math.max(1,Number(s.duration_ms||S.p.fire_interval_ms||1));}); S.p.steps[0].pattern_x=0; S.p.steps[0].pattern_y=0; renderShots();}
const fitGuide=(w,h)=>{const mw=Math.max(1,VW-PAD*2),mh=Math.max(1,VH-PAD*2),sw=Math.max(1,Number(w||1)),sh=Math.max(1,Number(h||1)),z=Math.min(mw/sw,mh/sh,1);return {width:Number((sw*z).toFixed(3)),height:Number((sh*z).toFixed(3))};};
const canvasPoint=e=>{const c=$("canvas"),r=c.getBoundingClientRect(),sx=c.width/r.width,sy=c.height/r.height;return {x:(e.clientX-r.left)*sx,y:(e.clientY-r.top)*sy};};
const centerPan=(zoom=1)=>{const b=guideBounds(); return b?{x:(VW-b.width*zoom)/2-b.minX*zoom,y:(VH-b.height*zoom)/2-b.minY*zoom}:{x:PAD,y:PAD};};
const itoc=(x,y)=>({x:S.panX+x*S.zoom,y:S.panY+y*S.zoom}), ctoi=(x,y)=>({x:(x-S.panX)/S.zoom,y:(y-S.panY)/S.zoom});
function hit(x,y){for(const img of layers()){for(let i=img.markers.length-1;i>=0;i--){const m=img.markers[i],gp=imageGlobalPoint(img,m.x,m.y),p=itoc(gp.x,gp.y),dx=p.x-x,dy=p.y-y; if(dx*dx+dy*dy<=100) return {image:img,marker:m};}} return null;}
function draw(){const c=$("canvas"),g=c.getContext("2d"); g.clearRect(0,0,c.width,c.height); const imgs=layers(); if(!imgs.length){g.fillStyle="#93a8b1"; g.font="16px Segoe UI"; g.fillText("Select or upload image layers.",24,36); return;} const ready=imgs.some(img=>S.cache.get(img.id)); if(!ready){g.fillStyle="#93a8b1"; g.font="16px Segoe UI"; g.fillText("Loading image layers...",24,36); return;} g.save(); g.translate(S.panX,S.panY); g.scale(S.zoom,S.zoom); for(let i=imgs.length-1;i>=0;i--){const img=imgs[i],bmp=S.cache.get(img.id),o=imageOffset(img); if(!bmp) continue; g.globalAlpha=img.id===S.active?0.94:0.30; g.drawImage(bmp,o.x,o.y,img.width||bmp.width,img.height||bmp.height); if(img.id===S.active){g.globalAlpha=1; g.strokeStyle="rgba(255,241,217,.85)"; g.lineWidth=2/Math.max(S.zoom,0.2); g.strokeRect(o.x,o.y,img.width||bmp.width,img.height||bmp.height);}} g.restore(); imgs.forEach((img,layerIdx)=>{const activeLayer=img.id===S.active; img.markers.forEach((m,i)=>{const gp=imageGlobalPoint(img,m.x,m.y),p=itoc(gp.x,gp.y); g.beginPath(); g.fillStyle=activeLayer?"#ff8c55":"#5eb4c7"; g.strokeStyle=activeLayer?"#fff1d9":"#d8f6ff"; g.lineWidth=2; g.arc(p.x,p.y,activeLayer?7:6,0,Math.PI*2); g.fill(); g.stroke(); g.fillStyle=activeLayer?"#eef6fa":"#d8eef5"; g.font=activeLayer?"12px Segoe UI":"11px Segoe UI"; g.fillText(`${layerIdx+1}:${i+1}`,p.x+10,p.y-10);});});}
async function render(){renderSelect(); renderMeta(); renderImages(); renderShots(); if(active()) await ensure(active()); draw();}
async function loadProfiles(sel=S.p?.id||""){const r=await req('/api/recoil/profiles'); S.profiles=r.profiles||[]; renderSelect(sel);}
async function loadProfile(id){if(!id){S.p=empty(); S.active=""; S.cache.clear(); await render(); renderSelect(""); status("Started a new profile."); return;} const r=await req(`/api/recoil/profiles/${enc(id)}`); S.p=r.profile; S.active=S.p.images[0]?.id||""; S.cache.clear(); for(const i of S.p.images) await ensure(i); resetZoom(); await render(); status(`Loaded '${S.p.name}'.`);}
function moveImage(i,d){const n=i+d; if(!S.p||n<0||n>=S.p.images.length) return; const [x]=S.p.images.splice(i,1); S.p.images.splice(n,0,x); renderImages(); draw();}
function removeImage(id){if(!S.p) return; S.p.images=S.p.images.filter(i=>i.id!==id); S.p.steps=S.p.steps.filter(s=>s.source_image_id!==id); S.p.steps.forEach((s,i)=>s.index=i); if(S.active===id) S.active=S.p.images[0]?.id||""; if(S.p.steps.length) syncSteps(); else renderShots(); renderImages(); draw();}
function insertStep(i){if(!S.p) return; syncMeta(); S.p.steps.splice(i,0,{index:i,pattern_x:0,pattern_y:0,duration_ms:Math.max(1,Number(S.p.fire_interval_ms||1)),source_image_id:"",source_marker_id:""}); S.p.steps.forEach((s,x)=>s.index=x); renderShots();}
function moveStep(i,d){const n=i+d; if(!S.p||n<0||n>=S.p.steps.length) return; const [x]=S.p.steps.splice(i,1); S.p.steps.splice(n,0,x); S.p.steps.forEach((s,idx)=>s.index=idx); syncSteps();}
function removeStep(i){if(!S.p) return; S.p.steps.splice(i,1); if(!S.p.steps.length){renderShots(); return;} S.p.steps.forEach((s,idx)=>s.index=idx); syncSteps();}
function addMarker(x,y){const img=active(); if(!img||!S.p) return; syncMeta(); const p=localPointForImage(img,x,y),m={id:uid('marker'),x:Number(p.x.toFixed(3)),y:Number(p.y.toFixed(3))}; img.markers.push(m); S.p.steps.push({index:S.p.steps.length,pattern_x:x,pattern_y:y,duration_ms:Math.max(1,Number(S.p.fire_interval_ms||1)),source_image_id:img.id,source_marker_id:m.id}); syncSteps(); renderImages(); draw();}
function removeMarker(imageId,id){const img=layers().find(v=>v.id===imageId); if(!img||!S.p) return; img.markers=img.markers.filter(m=>m.id!==id); S.p.steps=S.p.steps.filter(s=>!(s.source_image_id===imageId&&s.source_marker_id===id)); S.p.steps.forEach((s,i)=>s.index=i); if(S.p.steps.length) syncSteps(); else renderShots(); renderImages(); draw();}
async function readFiles(files){const out=[]; for(const file of files){const data=await new Promise((ok,err)=>{const r=new FileReader();r.onload=()=>ok(r.result);r.onerror=()=>err(new Error(`Failed to read ${file.name}`));r.readAsDataURL(file);}); const img=await new Promise((ok,err)=>{const x=new Image();x.onload=()=>{const g=fitGuide(x.width,x.height);ok({id:uid('image'),name:file.name,width:g.width,height:g.height,offset_x:0,offset_y:0,markers:[],asset_path:'',asset_url:'',data_url:data});};x.onerror=()=>err(new Error(`Failed to decode ${file.name}`));x.src=data;}); out.push(img);} return out;}
function resetZoom(){S.zoom=1; const p=centerPan(1); S.panX=p.x; S.panY=p.y; draw();}
async function saveProfile(){syncMeta(); syncSteps(); const method=S.p.id?'PUT':'POST', path=S.p.id?`/api/recoil/profiles/${enc(S.p.id)}`:'/api/recoil/profiles'; const r=await req(path,{method,headers:{'Content-Type':'application/json'},body:JSON.stringify(S.p)}); S.p=r.profile; S.active=S.p.images[0]?.id||''; S.cache.clear(); for(const i of S.p.images) await ensure(i); await loadProfiles(S.p.id); await render(); status(`Saved '${S.p.name}'.`);}
async function deleteProfile(){if(!S.p?.id){S.p=empty(); S.active=''; S.cache.clear(); await render(); renderSelect(''); status('Cleared unsaved profile.'); return;} await req(`/api/recoil/profiles/${enc(S.p.id)}`,{method:'DELETE'}); S.p=empty(); S.active=''; S.cache.clear(); await loadProfiles(''); await render(); status('Deleted profile.');}
$('canvas').addEventListener('contextmenu',e=>e.preventDefault()); $('canvas').addEventListener('mousedown',e=>{const q=canvasPoint(e),gp=ctoi(q.x,q.y);S.move=false; if(e.button===2){S.pan={x:q.x,y:q.y,px:S.panX,py:S.panY}; return;} const hitMarker=hit(q.x,q.y),hitImage=topImageAt(gp.x,gp.y); if($('tool').value==='delete'&&hitMarker){S.active=hitMarker.image.id; removeMarker(hitMarker.image.id,hitMarker.marker.id); return;} if($('tool').value==='drag'&&hitMarker){S.active=hitMarker.image.id; S.drag={imageId:hitMarker.image.id,markerId:hitMarker.marker.id}; renderImages(); draw(); return;} if($('tool').value==='drag'&&hitImage){S.active=hitImage.id; const o=imageOffset(hitImage); S.layerDrag={imageId:hitImage.id,startX:gp.x,startY:gp.y,offsetX:o.x,offsetY:o.y}; renderImages(); draw();}}); $('canvas').addEventListener('mousemove',e=>{const q=canvasPoint(e),gp=ctoi(q.x,q.y);if(S.pan){S.move=true; S.panX=S.pan.px+(q.x-S.pan.x); S.panY=S.pan.py+(q.y-S.pan.y); draw(); return;} if(S.layerDrag){S.move=true; const img=layers().find(x=>x.id===S.layerDrag.imageId); if(!img) return; img.offset_x=Number((S.layerDrag.offsetX+(gp.x-S.layerDrag.startX)).toFixed(3)); img.offset_y=Number((S.layerDrag.offsetY+(gp.y-S.layerDrag.startY)).toFixed(3)); syncSteps(); renderImages(); draw(); return;} if(!S.drag) return; S.move=true; const img=layers().find(x=>x.id===S.drag.imageId),m=img?.markers.find(x=>x.id===S.drag.markerId); if(!img||!m) return; const p=localPointForImage(img,gp.x,gp.y); m.x=Number(Math.max(0,Math.min(img.width,p.x)).toFixed(3)); m.y=Number(Math.max(0,Math.min(img.height,p.y)).toFixed(3)); syncSteps(); draw();}); window.addEventListener('mouseup',()=>{S.drag=null; S.layerDrag=null; S.pan=null;}); $('canvas').addEventListener('click',e=>{const q=canvasPoint(e),gp=ctoi(q.x,q.y);if(e.button!==0||S.move||$('tool').value!=='add') return; const img=topImageAt(gp.x,gp.y)||active(); if(!img) return; S.active=img.id; const p=localPointForImage(img,gp.x,gp.y); if(p.x<0||p.y<0||p.x>img.width||p.y>img.height) return; addMarker(gp.x,gp.y);}); $('canvas').addEventListener('wheel',e=>{e.preventDefault(); const q=canvasPoint(e),z=Math.max(.2,Math.min(6,S.zoom*(e.deltaY<0?1.1:.9))),p=ctoi(q.x,q.y); S.zoom=z; S.panX=q.x-p.x*S.zoom; S.panY=q.y-p.y*S.zoom; draw();},{passive:false});
$('newp').onclick=async()=>{S.p=empty(); S.active=''; S.cache.clear(); await render(); renderSelect(''); status('Started a new profile.');}; $('refreshp').onclick=()=>loadProfiles(S.p?.id||'').then(()=>status('Reloaded profiles.')).catch(e=>status(e.message)); $('savep').onclick=()=>saveProfile().catch(e=>status(e.message)); $('deletep').onclick=()=>deleteProfile().catch(e=>status(e.message)); $('addimg').onclick=()=>$('files').click(); $('addstep').onclick=()=>insertStep(S.p?.steps.length||0); $('sync').onclick=syncSteps; $('resetzoom').onclick=resetZoom;
$('files').addEventListener('change',async()=>{if(!$('files').files?.length) return; const items=await readFiles(Array.from($('files').files)); if(!S.p) S.p=empty(); S.p.images.push(...items); S.active=S.active||items[0]?.id||''; for(const i of items) await ensure(i); $('files').value=''; resetZoom(); await render(); status(`Added ${items.length} image(s).`);}); $('profiles').addEventListener('change',()=>loadProfile($('profiles').value).catch(e=>status(e.message))); $('name').addEventListener('input',syncMeta); $('vscale').addEventListener('change',syncMeta); $('hscale').addEventListener('change',syncMeta); $('interval').addEventListener('change',()=>{syncMeta(); renderShots();});
initSplit();
(async()=>{S.p=empty(); await loadProfiles(''); await render(); status('Ready.');})().catch(e=>status(e.message));
</script></body></html>"""


def handler(repo: Repository, page: str):
    class EditorHandler(BaseHTTPRequestHandler):
        def send_json(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length else b"{}"
            try:
                payload = json.loads(raw.decode("utf-8") or "{}")
            except json.JSONDecodeError as exc:
                raise ProfileError("Request body must be valid JSON.") from exc
            if not isinstance(payload, dict):
                raise ProfileError("Request body must be a JSON object.")
            return payload

        def profile_id(self) -> str:
            return parse_profile_id_path(self.path)

        def do_GET(self) -> None:  # noqa: N802
            path = self.path.split("?", 1)[0]
            try:
                if path == "/":
                    body = page.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif path == "/api/recoil/profiles":
                    self.send_json({"profiles": repo.list_profiles()})
                elif path.startswith("/api/recoil/profiles/"):
                    self.send_json({"profile": repo.get(self.profile_id())})
                elif path.startswith("/assets/"):
                    asset = safe_join(repo.root, path.lstrip("/"))
                    if not asset.exists() or not asset.is_file():
                        self.send_json({"error": "Asset not found."}, HTTPStatus.NOT_FOUND)
                        return
                    body = asset.read_bytes()
                    self.send_response(HTTPStatus.OK)
                    self.send_header("Content-Type", mimetypes.guess_type(asset.name)[0] or "application/octet-stream")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
            except ProfileError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_POST(self) -> None:  # noqa: N802
            try:
                if self.path.split("?", 1)[0] != "/api/recoil/profiles":
                    self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                    return
                saved = repo.save(self.read_json())
                self.send_json({"profile": saved, "profiles": repo.list_profiles()})
            except ProfileError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_PUT(self) -> None:  # noqa: N802
            try:
                if not self.path.split("?", 1)[0].startswith("/api/recoil/profiles/"):
                    self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                    return
                saved = repo.save(self.read_json(), self.profile_id())
                self.send_json({"profile": saved, "profiles": repo.list_profiles()})
            except ProfileError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def do_DELETE(self) -> None:  # noqa: N802
            try:
                if not self.path.split("?", 1)[0].startswith("/api/recoil/profiles/"):
                    self.send_json({"error": "Not found."}, HTTPStatus.NOT_FOUND)
                    return
                repo.delete(self.profile_id())
                self.send_json({"ok": True, "profiles": repo.list_profiles()})
            except ProfileError as exc:
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
            return

    return EditorHandler


def run(host: str, port: int, root: Path) -> None:
    repo = Repository(root)
    server = ThreadingHTTPServer((host, port), handler(repo, html()))
    print(f"[editor] Recoil profile editor running at http://{host}:{port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[editor] Stopping recoil profile editor.")
    finally:
        server.server_close()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta recoil profile editor")
    parser.add_argument("--host", default=HOST)
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--root", type=Path, default=ROOT)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    run(args.host, args.port, args.root)

