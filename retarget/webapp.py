# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retarget-only web app: Kimodo motion (NPZ/BVH) + Mixamo character -> baked GLB.

Runs on CPU with numpy + pygltflib only (no torch) — sized for Kaggle free tier.
Endpoints:
  GET  /                     single-page UI (Three.js viewport)
  GET  /api/health           liveness
  POST /api/character        upload character .glb  -> skeleton/mapping report
  POST /api/motion           upload motion .npz/.bvh -> frame/fps info
  POST /api/retarget         start async retarget job (character + motion + options)
  GET  /api/job/<id>         poll job status + log
  GET  /download/<job_id>    the finished GLB (character + baked animation)

Launch:  python webapp.py            (serves on 0.0.0.0:8000)
"""

import io
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from glb import add_animation, build_target_skeleton, detect_prefix, load_gltf, save_gltf  # noqa: E402
from kimodo_to_mixamo import load_kimodo_npz  # noqa: E402
from mapping import SOMA_TO_MIXAMO  # noqa: E402
from retarget import build_pairs  # noqa: E402
from soma import SOMA77_NAMES  # noqa: E402
from soma_bvh import parse_bvh_to_kimodo_rows  # noqa: E402

from fastapi import FastAPI, File, Form, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse  # noqa: E402

MAX_UPLOAD = 200 * 1024 * 1024  # 200 MB
WORK_DIR = Path(os.environ.get("KIMODO_WEB_DIR", Path(tempfile.gettempdir()) / "kimodo_web"))
WORK_DIR.mkdir(parents=True, exist_ok=True)

CHARS: dict[str, dict] = {}  # id -> {name, path, prefix, report}
MOTIONS: dict[str, dict] = {}  # id -> {name, kind, fps, posed, rots, contacts}
JOBS: dict[str, dict] = {}  # id -> {status, log, error, path, out_name}
STATE_LOCK = threading.Lock()


def _log(job: dict, msg: str) -> None:
    with STATE_LOCK:
        job["log"].append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _clip_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", stem) or "motion"
    return "kimodo_" + stem


app = FastAPI(title="kimodo retarget web")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "numpy": np.__version__}


@app.post("/api/character")
async def upload_character(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    if not file.filename or not file.filename.lower().endswith(".glb"):
        raise HTTPException(400, "please upload a .glb character file")
    cid = uuid.uuid4().hex[:12]
    path = WORK_DIR / f"char_{cid}.glb"
    path.write_bytes(raw)
    try:
        gltf = load_gltf(str(path))
        prefix = detect_prefix(gltf)
        sk, warns = build_target_skeleton(gltf)
        pairs, missing = build_pairs(sk, prefix)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not parse GLB: {exc}") from exc

    essential = ["Hips", "Spine", "Head", "LeftArm", "RightArm", "LeftUpLeg", "RightUpLeg",
                 "LeftLeg", "RightLeg", "LeftFoot", "RightFoot", "LeftToeBase"]
    miss_essential = [m for m in essential if any(d.endswith("-> " + m) for d in missing)]
    parts = [f"{len(sk.names)} nodes", f"prefix '{prefix or '-'}'", f"mapped {len(pairs)}/{len(SOMA_TO_MIXAMO)}"]
    if miss_essential:
        parts.append("missing: " + ", ".join(miss_essential))
        parts.append("this is usually not a Mixamo-named rig")
    report = {
        "nodes": len(sk.names),
        "prefix": prefix,
        "mapped": len(pairs),
        "total": len(SOMA_TO_MIXAMO),
        "missing": missing,
        "missing_essential": miss_essential,
        "warnings": warns + ([f"could not resolve {len(missing)} bones: {', '.join(missing[:8])}"] if missing else []),
        "message": " · ".join(parts),
    }
    report["ok"] = len(pairs) >= 10 and "Hips" not in miss_essential
    CHARS[cid] = {"name": Path(file.filename).name, "path": str(path), "prefix": prefix, "report": report}
    return JSONResponse({"id": cid, "name": Path(file.filename).name, "report": report})


@app.post("/api/motion")
async def upload_motion(file: UploadFile = File(...)) -> JSONResponse:
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "file too large")
    fname = file.filename or "motion.bin"
    ext = Path(fname).suffix.lower()
    mid = uuid.uuid4().hex[:12]
    tmp = WORK_DIR / f"motion_{mid}{ext}"
    try:
        if ext == ".npz":
            tmp.write_bytes(raw)
            data = load_kimodo_npz(str(tmp))
            kind, fps = "npz", 30.0  # Kimodo default; NPZ carries no fps value
        elif ext == ".bvh":
            tmp.write_bytes(raw)
            from soma import SOMA77_INDEX

            rows = parse_bvh_to_kimodo_rows(str(tmp), SOMA77_INDEX, SOMA77_NAMES)
            kind = "bvh"
            fps = float(rows["fps"])
            data = {"posed_joints": rows["posed_joints"], "global_rot_mats": rows["global_rot_mats"], "foot_contacts": None}
        else:
            raise HTTPException(400, "unsupported motion type — upload .npz (kimodo_gen output) or .bvh (demo export)")

        posed = data["posed_joints"]
        rots = data["global_rot_mats"]
        contacts = data["foot_contacts"]
        T = int(posed.shape[0])
        if T < 2:
            raise HTTPException(400, "motion has fewer than 2 frames")
        if not np.isfinite(posed).all() or not np.isfinite(rots).all():
            raise HTTPException(400, "motion contains NaN/Inf values")
        MOTIONS[mid] = {
            "name": Path(fname).name,
            "kind": kind,
            "fps": float(fps),
            "posed": np.ascontiguousarray(posed),
            "rots": np.ascontiguousarray(rots),
            "contacts": contacts if contacts is None else np.ascontiguousarray(contacts),
        }
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, f"could not parse motion: {exc}") from exc
    return JSONResponse({
        "id": mid,
        "name": Path(fname).name,
        "kind": MOTIONS[mid]["kind"],
        "frames": T,
        "fps": float(fps),
        "duration": round(T / fps, 2),
        "has_contacts": MOTIONS[mid]["contacts"] is not None,
    })


def _run_job(job_id: str, char_id: str, motion_id: str, mode: str, limb_fix: bool, sticky_feet: bool, clip: str) -> None:
    job = JOBS[job_id]
    try:
        _log(job, "loading character GLB")
        char = CHARS[char_id]
        gltf = load_gltf(char["path"])
        prefix = detect_prefix(gltf)
        sk, warns = build_target_skeleton(gltf)
        pairs, missing = build_pairs(sk, prefix)
        for w in warns:
            _log(job, f"warn: {w}")
        _log(job, f"prefix '{prefix}', {len(sk.names)} nodes, mapped {len(pairs)}/{len(SOMA_TO_MIXAMO)}")

        motion = MOTIONS[motion_id]
        _log(job, f"motion '{motion['name']}' ({motion['kind']}), {motion['posed'].shape[0]} frames @ {motion['fps']:.0f}fps")
        if sticky_feet and motion["contacts"] is None:
            _log(job, "no foot contacts in this input (BVH) — sticky feet disabled")
            sticky_feet = False

        _log(job, f"retargeting (mode={mode}, limb_fix={limb_fix}, sticky_feet={sticky_feet}) ...")
        t0 = time.time()
        result = retarget_core(
            motion["rots"], motion["posed"], motion["contacts"], sk,
            prefix=prefix, mode=mode, limb_fix=limb_fix, sticky_feet=sticky_feet, fps=motion["fps"],
        )
        _log(job, f"retarget done in {time.time() - t0:.1f}s — {len(result.rotations)} bone tracks")

        _log(job, "baking animation into GLB")
        add_animation(gltf, result, clip)
        out = WORK_DIR / f"job_{job_id}.glb"
        save_gltf(gltf, str(out))
        job["path"] = str(out)
        job["out_name"] = clip + ".glb"
        with STATE_LOCK:
            job["status"] = "done"
        _log(job, f"saved {out.name} — ready")
    except Exception as exc:  # noqa: BLE001
        with STATE_LOCK:
            job["status"] = "error"
            job["error"] = str(exc)
        _log(job, "ERROR: " + str(exc))
        _log(job, traceback.format_exc(limit=6))


@app.post("/api/retarget")
async def start_retarget(
    character: str = Form(...),
    motion: str = Form(...),
    mode: str = Form("inplace"),
    limb_fix: str = Form("1"),
    sticky_feet: str = Form("1"),
) -> JSONResponse:
    if character not in CHARS:
        raise HTTPException(404, "unknown character id — upload it first")
    if motion not in MOTIONS:
        raise HTTPException(404, "unknown motion id — upload it first")
    if mode not in ("inplace", "travel"):
        raise HTTPException(400, "mode must be inplace or travel")
    job_id = uuid.uuid4().hex[:12]
    clip = _clip_name(MOTIONS[motion]["name"])
    with STATE_LOCK:
        JOBS[job_id] = {"status": "running", "log": [], "error": None, "path": None, "out_name": clip + ".glb"}
    job = JOBS[job_id]
    _log(job, f"job {job_id}: character '{CHARS[character]['name']}' + motion '{MOTIONS[motion]['name']}' -> clip '{clip}'")
    th = threading.Thread(
        target=_run_job,
        args=(job_id, character, motion, mode, limb_fix == "1", sticky_feet == "1", clip),
        daemon=True,
    )
    th.start()
    return JSONResponse({"id": job_id})


@app.get("/api/job/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return JSONResponse({"id": job_id, "status": job["status"], "log": job["log"], "error": job["error"]})


@app.get("/download/{job_id}")
def download(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    if job["status"] != "done" or not job["path"]:
        raise HTTPException(409, "job not finished yet")
    return FileResponse(job["path"], media_type="model/gltf-binary", filename=job["out_name"])


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Kimodo &rarr; Mixamo retargeter</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin: 0; font: 14px/1.45 system-ui, sans-serif; background: #0f1115; color: #d7dce3;
         display: flex; flex-direction: column; height: 100vh; }
  header { padding: 10px 18px; background: #161a22; border-bottom: 1px solid #23293a;
           display: flex; align-items: baseline; gap: 14px; }
  header h1 { font-size: 16px; margin: 0; color: #7fd1ff; }
  header span { color: #8b93a7; font-size: 12px; }
  main { flex: 1; display: grid; grid-template-columns: 380px 1fr; min-height: 0; }
  #panel { padding: 14px; overflow-y: auto; border-right: 1px solid #23293a; }
  .drop { border: 1.5px dashed #39415a; border-radius: 10px; padding: 12px; margin-bottom: 12px;
          text-align: center; cursor: pointer; transition: border-color .15s, background .15s; }
  .drop:hover, .drop.over { border-color: #7fd1ff; background: #1a2030; }
  .drop input { display: none; }
  .drop b { color: #aeb6c6; }
  .drop .info { display: block; margin-top: 6px; font-size: 12px; color: #7fd1ff; word-break: break-all; }
  .drop .bad { color: #ff8f8f; }
  .row { display: flex; gap: 10px; align-items: center; margin: 10px 0; flex-wrap: wrap; }
  label.opt { display: flex; gap: 6px; align-items: center; color: #aeb6c6; }
  select { background: #1a2030; color: #d7dce3; border: 1px solid #39415a; border-radius: 6px; padding: 4px 8px; }
  button { background: #2563a8; color: #fff; border: 0; border-radius: 8px; padding: 9px 16px;
           font: inherit; cursor: pointer; }
  button:disabled { background: #39415a; color: #8b93a7; cursor: default; }
  button.go { width: 100%; font-size: 15px; background: #1a7f5a; }
  button.go:disabled { background: #2c3648; }
  #dl { display: inline-block; text-decoration: none; background: #1a7f5a; color: #fff;
        border-radius: 8px; padding: 9px 16px; }
  #dl:disabled { background: #39415a; color: #8b93a7; }
  #log { background: #0a0d12; border: 1px solid #23293a; border-radius: 8px; height: 170px;
         overflow-y: auto; padding: 8px 10px; font: 11.5px/1.5 ui-monospace, monospace; color: #9aa4b8;
         white-space: pre-wrap; }
  #viewer { position: relative; min-width: 0; }
  #viewer canvas { display: block; }
  #viewerbar { position: absolute; left: 10px; bottom: 10px; display: flex; gap: 8px; align-items: center;
               background: #161a22cc; border: 1px solid #23293a; padding: 8px 10px; border-radius: 10px; }
  #viewerbar button { padding: 5px 10px; }
  input[type=range] { accent-color: #7fd1ff; width: 100px; }
  #clipinfo { color: #8b93a7; font-size: 12px; }
  #hintbar { position: absolute; right: 10px; top: 10px; color: #6b7488; font-size: 11px; }
</style>
</head>
<body>
<header>
  <h1>Kimodo &rarr; Mixamo retargeter</h1>
  <span>CPU only &middot; uploads your character once, retargets Kimodo motions onto it, bakes a GLB</span>
</header>
<main>
  <div id="panel">
    <div class="drop" id="charDrop"><b>Character GLB</b> (Mixamo rig)<br>click or drop here
      <input type="file" id="charFile" accept=".glb">
      <span class="info" id="charInfo"></span></div>
    <div class="drop" id="motionDrop"><b>Motion file</b> (.npz from kimodo_gen, or .bvh from the demo)<br>click or drop here
      <input type="file" id="motionFile" accept=".npz,.bvh">
      <span class="info" id="motionInfo"></span></div>
    <div class="row">
      <label class="opt">mode
        <select id="mode">
          <option value="inplace" selected>in-place</option>
          <option value="travel">travel</option>
        </select>
      </label>
      <label class="opt"><input type="checkbox" id="limbfix" checked> limb-length fix</label>
      <label class="opt"><input type="checkbox" id="sticky" checked> sticky feet</label>
    </div>
    <button class="go" id="go" disabled>Retarget</button>
    <div class="row" style="justify-content: space-between">
      <span id="clipinfo"></span>
      <a id="dl" class="dl" download disabled>Download GLB</a>
    </div>
    <div id="log"></div>
  </div>
  <div id="viewer">
    <div id="hintbar">drag to orbit &middot; scroll to zoom</div>
    <div id="viewerbar">
      <button id="play">&#10074;&#10074;</button>
      <label class="opt">speed <input type="range" id="speed" min="0.1" max="2" step="0.1" value="1"></label>
      <span id="clipinfo2"></span>
    </div>
  </div>
</main>
<script type="importmap">
{ "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
} }
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const $ = id => document.getElementById(id);
const logBox = $('log');
function log(msg, bad) {
  const line = document.createElement('div');
  line.textContent = msg;
  if (bad) line.style.color = '#ff8f8f';
  logBox.appendChild(line);
  logBox.scrollTop = logBox.scrollHeight;
}

// ---- viewer ------------------------------------------------------------------
const viewer = $('viewer');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
viewer.appendChild(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0f1115);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(0, 1.3, 3.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 1.0, 0);
controls.update();
const hemiLight = new THREE.HemisphereLight(0xbfd4ff, 0x30281c, 1.1);
scene.add(hemiLight);
const grid = new THREE.GridHelper(20, 40, 0x3b4560, 0x232a3d);
scene.add(grid);
const sun = new THREE.DirectionalLight(0xffffff, 1.6);
sun.position.set(-2, 4, 3);
scene.add(sun);
const STATIC = [hemiLight, grid, sun];

function resize() {
  const w = viewer.clientWidth, h = viewer.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);

const clock = new THREE.Clock();
let mixer = null, action = null, playing = true, speed = 1;

function tick() {
  requestAnimationFrame(tick);
  const dt = clock.getDelta();
  if (mixer && playing) mixer.update(dt * speed);
  controls.update();
  renderer.render(scene, camera);
}
tick();

function showResult(arrayBuffer) {
  [...scene.children].forEach(c => { if (!STATIC.includes(c)) scene.remove(c); });
  new GLTFLoader().parse(arrayBuffer, '', (gltf) => {
    const obj = gltf.scene;
    scene.add(obj);
    obj.updateMatrixWorld(true);
    // frame the model: walk-in-place clip at hips height
    const bbox = new THREE.Box3().setFromObject(obj);
    const size = bbox.getSize(new THREE.Vector3());
    const center = bbox.getCenter(new THREE.Vector3());
    const h = Math.max(size.y, 0.8);
    controls.target.copy(center);
    camera.position.set(center.x, center.y + h * 0.25, center.z + Math.max(h * 1.6, 1.2));
    camera.near = Math.max(h / 100, 0.01);
    camera.far = h * 20;
    camera.updateProjectionMatrix();
    controls.update();
    mixer = new THREE.AnimationMixer(obj);
    const clip = gltf.animations[0];
    if (clip) {
      action = mixer.clipAction(clip);
      action.play();
      $('clipinfo').textContent = 'clip: ' + clip.name + ' (' + clip.duration.toFixed(1) + 's)';
      $('clipinfo2').textContent = clip.name;
    } else {
      log('warning: no animation in result', true);
    }
    playing = true;
    $('play').innerHTML = '&#10074;&#10074;';
  }, (err) => log('GLB load failed: ' + err, true));
}

$('play').onclick = () => { playing = !playing; $('play').innerHTML = playing ? '&#10074;&#10074;' : '&#9654;'; };
$('speed').oninput = e => speed = parseFloat(e.target.value);

// resize via ResizeObserver (column layout can change)
new ResizeObserver(resize).observe(viewer);
resize();

// ---- uploads ------------------------------------------------------------------
let charId = null, motionId = null, hasContacts = false, motionName = '';

async function upload(url, file, infoEl) {
  infoEl.className = 'info';
  infoEl.textContent = 'uploading ' + file.name + ' ...';
  const fd = new FormData();
  fd.append('file', file, file.name);
  const r = await fetch(url, { method: 'POST', body: fd });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) {
    infoEl.className = 'info bad';
    infoEl.textContent = j.detail || ('HTTP ' + r.status);
    throw new Error(j.detail || r.status);
  }
  return j;
}

function wireDrop(dropId, inputId, infoId, handler) {
  const drop = $(dropId), input = $(inputId);
  drop.onclick = () => input.click();
  input.onchange = () => input.files[0] && handler(input.files[0], $(infoId));
  drop.ondragover = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop = e => {
    e.preventDefault(); drop.classList.remove('over');
    const f = e.dataTransfer.files[0];
    if (f) handler(f, $(infoId));
  };
}

async function handleChar(file, info) {
  try {
    const j = await upload('/api/character', file, info);
    charId = j.id;
    info.textContent = j.name + ' — ' + j.report.message;
    log('character: ' + j.name + ' (' + j.report.nodes + ' nodes, prefix "' + j.report.prefix + '")');
    (j.report.warnings || []).forEach(w => log('  warn: ' + w));
    updateGo();
  } catch (e) { log('character upload failed: ' + e.message, true); }
}

async function handleMotion(file, info) {
  try {
    const j = await upload('/api/motion', file, info);
    motionId = j.id; hasContacts = j.has_contacts; motionName = j.name;
    info.textContent = j.name + ' — ' + j.kind + ', ' + j.frames + ' frames @ ' + j.fps.toFixed(1) + 'fps (' + j.duration + 's)'
        + (j.has_contacts ? '' : ' — no foot contacts (sticky feet off)');
    log('motion: ' + j.name + ' — ' + j.frames + ' frames @ ' + j.fps.toFixed(1) + 'fps, contacts: ' + (j.has_contacts ? 'yes' : 'NO'));
    updateGo();
  } catch (e) { log('motion upload failed: ' + e.message, true); }
}

wireDrop('charDrop', 'charFile', 'charInfo', handleChar);
wireDrop('motionDrop', 'motionFile', 'motionInfo', handleMotion);

function updateGo() {
  $('go').disabled = !(charId && motionId);
}

// ---- retarget job --------------------------------------------------------------
let jobId = null;

$('go').onclick = async () => {
  if (!charId || !motionId || jobId) return;
  jobId = null;
  $('dl').disabled = true;
  $('dl').removeAttribute('href');
  $('go').disabled = true;
  $('go').textContent = 'Retargeting...';
  log('starting retarget job...');
  const fd = new FormData();
  fd.append('character', charId);
  fd.append('motion', motionId);
  fd.append('mode', $('mode').value);
  fd.append('limb_fix', $('limbfix').checked ? '1' : '0');
  fd.append('sticky_feet', ($('sticky').checked && hasContacts) ? '1' : '0');
  const r = await fetch('/api/retarget', { method: 'POST', body: fd });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) { log('failed to start: ' + (j.detail || r.status), true); $('go').disabled = false; return; }
  jobId = j.id;
  const poll = async () => {
    const jr = await fetch('/api/job/' + jobId);
    const jj = await jr.json();
    const seen = new Set(logBox.dataset.seen ? JSON.parse(logBox.dataset.seen) : []);
    for (const line of jj.log) if (!seen.has(line)) { seen.add(line); log(line); }
    logBox.dataset.seen = JSON.stringify([...seen].slice(-400));
    if (jj.status === 'done') {
      jobId = null;
      $('go').disabled = false;
      $('go').textContent = 'Retarget';
      const url = '/download/' + jj.id;
      $('dl').disabled = false;
      $('dl').href = url;
      $('dl').setAttribute('download', motionName.replace(/\.(npz|bvh)$/i, '') + '_mixamo.glb');
      log('done — downloading for preview...');
      const buf = await (await fetch(url)).arrayBuffer();
      showResult(buf);
      log('ready. use Download GLB to keep it.');
      return;
    }
    if (jj.status === 'error') {
      jobId = null;
      $('go').disabled = false;
      $('go').textContent = 'Retarget';
      log('job FAILED: ' + jj.error, true);
      return;
    }
    setTimeout(poll, 800);
  };
  poll();
};
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(PAGE)


def retarget_core(global_rot_src, posed_joints_src, foot_contacts, sk, **kw):
    from retarget import retarget as _retarget  # deferred: sibling module import

    return _retarget(global_rot_src, posed_joints_src, foot_contacts, sk, **kw)


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"serving on http://{host}:{port}  (public URL comes from the tunnel)")
    uvicorn.run(app, host=host, port=port, log_level="info")
