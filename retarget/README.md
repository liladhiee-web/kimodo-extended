# Kimodo → Mixamo GLB Retargeter

Bakes Kimodo-generated motion onto **your own Mixamo-rigged characters** as standard glTF
animation clips, ready for Three.js / Babylon.js / any game engine.

```text
Kimodo (SOMA-77 NPZ) ──► kimodo_to_mixamo.py ──► your_character.glb + baked clips
                                 │
                                 ▼
                    THREE.AnimationMixer plays it directly
```

## What it does

1. **Rest-aligned rotation transfer** — SOMA-77 global rotations are conjugated into your
   character's bind frame (exact bind-pose reproduction, handles any bone axis conventions).
2. **Limb length correction** — arm/leg bones swing-fix so Mixamo proportions follow the
   motion (SOMA limbs are usually longer/shorter than Mixamo's template).
3. **Foot-contact lock (sticky feet)** — uses Kimodo's `foot_contacts` labels (L heel,
   L toe, R heel, R toe) with analytic two-bone leg IK to flatten planted feet; kills the
   classic mocap foot-slide.
4. **In-place root** — hips translation is written *relative to frame 0*, so your game
   logic drives the character transform (requested default). Use `--mode travel` to bake
   absolute root motion instead.

## Requirements

- Python 3.10+
- `pip install numpy pygltflib`

## Usage

```bash
# inspect what maps (no conversion)
python retarget/kimodo_to_mixamo.py --check --character mychar.glb

# bake one or many clips
python retarget/kimodo_to_mixamo.py ^
    --motion walk.npz jump.npz dance.npz ^
    --character mychar.glb ^
    --out mychar_kimodo.glb ^
    --fps 30 --mode inplace

# self-test (synthetic skeleton + synthetic gait, no downloads needed)
python retarget/run_selftest.py
```

Options: `--no-limb-fix`, `--no-sticky-feet`, `--clip-prefix myname_`.

Input motions: Kimodo NPZ from `kimodo_gen` on any **SOMA-77** model
(`posed_joints`, `global_rot_mats`, `foot_contacts`), **or a Kimodo SOMA-77 `.bvh`**
(demo export — has no foot contacts, so sticky feet is auto-disabled; numpy-only parser,
`retarget/soma_bvh.py`, rotation convention verified against Kimodo's own
`scipy from_euler("ZYX")` loader). For SMPL-X/G1 formats convert first
with Kimodo's own `kimodo_convert`.

## Web app (upload + 3D preview, no CLI)

`retarget/webapp.py` is a small **CPU-only** FastAPI app — no torch — for Kaggle or any host:

```bash
pip install fastapi uvicorn python-multipart numpy pygltflib
python retarget/webapp.py          # serves http://localhost:8000
```

Open `/`, upload your character `.glb` and a motion `.npz`/`.bvh`, press **Retarget**,
preview the baked animation in the built-in Three.js viewport (play/pause/speed),
then **Download GLB** — the downloaded file IS the result (character + clip).

Endpoints: `POST /api/character`, `POST /api/motion`, `POST /api/retarget` (async job),
`GET /api/job/<id>`, `GET /download/<job_id>`, `GET /api/health`.

Kaggle notebook: `kaggle_retarget_web.ipynb` (CPU accelerator; clones the repo, installs
(~1 min), starts the app, exposes it through a Cloudflare quick-tunnel).

## Three.js integration

```js
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const loader = new GLTFLoader();
loader.load('/chars/mychar_kimodo.glb', (gltf) => {
  const group = gltf.scene;
  scene.add(group);

  const mixer = new THREE.AnimationMixer(group);
  const clips = gltf.animations;                 // [{ name: 'kimodo_walk' }, ...]
  const action = mixer.clipAction(THREE.AnimationClip.findByName(clips, 'kimodo_walk'));
  action.play();

  // clips are in-place: move the character with your own game logic
  // e.g. controller.position.add(forwardDir * speed * dt);
});
```

Notes for in-place clips:

- The **hips translation track is relative** — your movement code decides where the
  character goes. Kimodo's `root_positions` / `global_root_heading` in the NPZ are the
  source heading/trajectory data if you want to drive the controller from it.
- Clip length in seconds = frames / fps (e.g. 45 frames @ 30 fps = 1.5 s).
- Cross-fade between clips with `action.crossFadeTo(nextAction, 0.3, false)` or
  `THREE.AnimationUtils` — they all animate the same skeleton, so blending just works.

## Character requirements / caveats

- Skeleton style: **Mixamo** bone layout (optionally `mixamorig:`-prefixed; the prefix is
  auto-detected, bone names matched case-insensitively). Other rigs can be supported by
  editing `mapping.py` (it is a plain dict: SOMA name → target bone name).
- The animation modifies node rotations on the skin joints; glTF bind TRS of all other
  nodes is untouched, so your meshes/skinning keep working.
- Non-structural SOMA joints are dropped intentionally (Neck2 → Neck, jaw, eyes, finger
  tips beyond Mixamo's 3 phalanges).
- Bind scale: warned if a bone node has non-unit bind scale (Mixamo exports generally do not).
- The retargeter assumes both rigs are ~Y-up, meters, facing the same direction. Kimodo's
  world matches glTF conventions by default.
