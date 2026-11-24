// 스티커 탭: SAM 3D로 만든 다꾸용 볼록 스티커(유광)를 판에 붙이고, 판을 기울여 본다
//   스티커 = assets/<id>/sticker.glb (pipeline/sticker.py) — 높이를 준 메시 + 유광 코팅(clearcoat) 재질
//   스티커 끌기 = 옮기기 · 모서리 손잡이 = 돌리기와 크기 · 빈 곳 끌기 = 판 기울이기 · 오른쪽 끌기(두 손가락) = 판 옮기기 · 휠(벌리기) = 확대
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const LONG = 0.3; // 판 긴 변 · 짧은 변 · 두께 (m) — 스티커는 긴 변 6cm. 세로로 긴 화면(휴대폰)에서는 판도 세로로
const SHORT = 0.21;
const THICK = 0.004;
let BW = LONG;
let BH = SHORT;
const LIFT = 0.0003; // 스티커 종이 두께
const STACK = 0.00025; // 나중에 붙인 스티커일수록 이만큼 위
const TILT0 = -0.55; // 처음 각도: 책상 위 다이어리를 내려다보듯 (뒤로 젖힘)
const PITCH = [-1.25, 0.6]; // 판을 기울이는 범위 (뒤집히지 않게)
const YAW = [-1.0, 1.0];
const FOV = 30;

let ctx = null;

/** 스티커 탭 열기 (처음 한 번 장면을 만든다) */
export function openStickers(list) {
  ctx ??= setup(list);
  ctx.start();
}

export function closeStickers() {
  ctx?.stop();
}

function roundedRect(w, h, r) {
  const s = new THREE.Shape();
  const x = -w / 2;
  const y = -h / 2;
  s.moveTo(x + r, y);
  s.lineTo(x + w - r, y);
  s.quadraticCurveTo(x + w, y, x + w, y + r);
  s.lineTo(x + w, y + h - r);
  s.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  s.lineTo(x + r, y + h);
  s.quadraticCurveTo(x, y + h, x, y + h - r);
  s.lineTo(x, y + r);
  s.quadraticCurveTo(x, y, x + r, y);
  return s;
}

/** 다이어리 모눈 종이 (5mm 칸) */
function paperTexture(renderer) {
  const W = 2048;
  const H = Math.round((W * BH) / BW);
  const c = document.createElement('canvas');
  c.width = W;
  c.height = H;
  const g = c.getContext('2d');
  g.fillStyle = '#fbf2e2';
  g.fillRect(0, 0, W, H);
  const step = (W / (BW * 1000)) * 5;
  g.strokeStyle = 'rgba(110, 140, 185, 0.3)';
  g.lineWidth = 2;
  g.beginPath();
  for (let x = step; x < W; x += step) {
    g.moveTo(x, 0);
    g.lineTo(x, H);
  }
  for (let y = step; y < H; y += step) {
    g.moveTo(0, y);
    g.lineTo(W, y);
  }
  g.stroke();
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  t.repeat.set(1 / BW, 1 / BH); // ExtrudeGeometry 앞면 UV = (x, y) m
  t.offset.set(0.5, 0.5);
  return t;
}

/** 스티커 칼선 모양을 흐리게 → 판에 닿는 곳의 그림자 */
function shadowTexture(img) {
  const s = 256 / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * s));
  const h = Math.max(1, Math.round(img.height * s));
  const pad = 16;
  const c = document.createElement('canvas');
  c.width = w + 2 * pad;
  c.height = h + 2 * pad;
  const g = c.getContext('2d');
  g.filter = 'blur(5px)';
  g.drawImage(img, pad, pad, w, h);
  return { tex: new THREE.CanvasTexture(c), sx: c.width / w, sy: c.height / h };
}

/** 스티커 텍스처 알파 → 작은 마스크 (누른 곳이 스티커인지 칼선대로 판단) */
function alphaMask(img) {
  const s = 256 / Math.max(img.width, img.height);
  const w = Math.max(1, Math.round(img.width * s));
  const h = Math.max(1, Math.round(img.height * s));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0, w, h);
  const d = g.getImageData(0, 0, w, h).data;
  const a = new Uint8Array(w * h);
  for (let i = 0; i < a.length; i++) a[i] = d[i * 4 + 3];
  return { w, h, a };
}

function setup(list) {
  const frame = $('#boardFrame');
  const canvas = $('#boardCanvas');
  const portrait = frame.clientWidth < frame.clientHeight;
  [BW, BH] = portrait ? [SHORT, LONG] : [LONG, SHORT];
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setClearColor(0x000000, 0);
  renderer.toneMapping = THREE.NeutralToneMapping;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;

  const scene = new THREE.Scene();
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture; // 유광 면에 비치는 방 (판을 기울이면 반사가 움직인다)
  scene.environmentIntensity = 0.9;
  const camera = new THREE.PerspectiveCamera(FOV, 1.5, 0.01, 10);

  const sun = new THREE.DirectionalLight(0xffffff, 1.2);
  sun.position.set(-0.55, 0.65, 0.3); // 비스듬히: 볼록한 스티커가 판에 그림자를 드리우게
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, { left: -0.4, right: 0.4, top: 0.4, bottom: -0.4, near: 0.2, far: 2 });
  sun.shadow.bias = -0.0004;
  sun.shadow.radius = 3;
  scene.add(sun);

  // 판 (이 그룹째로 기울이고 옮긴다)
  const board = new THREE.Group();
  scene.add(board);
  const pageGeo = new THREE.ExtrudeGeometry(roundedRect(BW, BH, 0.012), { depth: THICK, bevelEnabled: false, curveSegments: 8 });
  pageGeo.translate(0, 0, -THICK);
  const page = new THREE.Mesh(pageGeo, [
    new THREE.MeshStandardMaterial({ map: paperTexture(renderer), roughness: 0.92 }),
    new THREE.MeshStandardMaterial({ color: 0xe6dccb, roughness: 0.9 }),
  ]);
  page.receiveShadow = true;
  board.add(page);
  const layer = new THREE.Group();
  board.add(layer);

  // 고른 스티커 테두리
  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.5, -0.5, 0),
      new THREE.Vector3(0.5, -0.5, 0),
      new THREE.Vector3(0.5, 0.5, 0),
      new THREE.Vector3(-0.5, 0.5, 0),
    ]),
    new THREE.LineBasicMaterial({ color: 0x13294b, transparent: true, opacity: 0.7, depthTest: false }),
  );
  outline.renderOrder = 10;

  // ---------- 스티커 불러오기 ----------
  const loader = new GLTFLoader();
  const kinds = new Map();
  function kind(s) {
    if (!kinds.has(s.id)) {
      kinds.set(
        s.id,
        loader.loadAsync(s.glb).then((gltf) => {
          let mesh = null;
          gltf.scene.traverse((o) => {
            if (o.isMesh && !mesh) mesh = o;
          });
          const mat = mesh.material;
          mat.alphaToCoverage = true; // 칼선 가장자리 계단 줄이기 (MSAA)
          const geo = mesh.geometry;
          geo.computeBoundingBox();
          const b = geo.boundingBox;
          // 위치 ↔ 텍스처 좌표는 선형: 누른 곳의 알파를 보려고
          const uv = geo.attributes.uv;
          let u0 = Infinity;
          let u1 = -Infinity;
          let v0 = Infinity;
          let v1 = -Infinity;
          for (let i = 0; i < uv.count; i++) {
            u0 = Math.min(u0, uv.getX(i));
            u1 = Math.max(u1, uv.getX(i));
            v0 = Math.min(v0, uv.getY(i));
            v1 = Math.max(v1, uv.getY(i));
          }
          return {
            geo,
            mat,
            box: b,
            hw: (b.max.x - b.min.x) / 2,
            hh: (b.max.y - b.min.y) / 2,
            toUV: (x, y) => [u0 + ((x - b.min.x) / (b.max.x - b.min.x)) * (u1 - u0), v0 + ((b.max.y - y) / (b.max.y - b.min.y)) * (v1 - v0)],
            mask: alphaMask(mat.map.image),
            shadow: (() => {
              // 텍스처 그림 전체가 차지하는 자리 (위치 ↔ UV 선형 관계에서)
              const iw = (b.max.x - b.min.x) / (u1 - u0);
              const ih = (b.max.y - b.min.y) / (v1 - v0);
              const sh = shadowTexture(mat.map.image);
              const geo = new THREE.PlaneGeometry(iw * sh.sx, ih * sh.sy);
              geo.translate(b.min.x - u0 * iw + iw / 2, b.max.y + v0 * ih - ih / 2, 0);
              const m = new THREE.MeshBasicMaterial({ color: 0x000000, map: sh.tex, transparent: true, opacity: 0.3, depthWrite: false });
              return { geo, mat: m };
            })(),
          };
        }),
      );
    }
    return kinds.get(s.id);
  }

  // ---------- 붙인 스티커 ----------
  const placed = [];
  let selected = null;

  function apply(p) {
    const z = LIFT + placed.indexOf(p) * STACK;
    p.mesh.position.set(p.x, p.y, z);
    p.mesh.rotation.set(0, 0, p.rot);
    p.mesh.scale.setScalar(p.scale);
    // 그림자는 판 바로 위, 빛 반대쪽(오른쪽 아래)으로 조금
    const c = Math.cos(p.rot);
    const s = Math.sin(p.rot);
    const ox = 0.0008;
    const oy = -0.001;
    p.shadow.position.set((ox * c + oy * s) / p.scale, (-ox * s + oy * c) / p.scale, (0.00006 - z) / p.scale);
  }
  function restack() {
    for (const p of placed) apply(p);
  }

  async function add(s, x, y, rot = 0) {
    const k = await kind(s);
    const mesh = new THREE.Mesh(k.geo, k.mat);
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    const shadow = new THREE.Mesh(k.shadow.geo, k.shadow.mat);
    mesh.add(shadow);
    const p = { s, k, mesh, shadow, x, y, rot, scale: 1 };
    placed.push(p);
    layer.add(mesh);
    apply(p);
    status();
    return p;
  }

  function remove(p) {
    if (!p) return;
    if (selected === p) select(null);
    layer.remove(p.mesh);
    placed.splice(placed.indexOf(p), 1);
    restack();
    status();
  }

  function toFront(p) {
    placed.splice(placed.indexOf(p), 1);
    placed.push(p);
    restack();
  }

  function select(p) {
    selected = p;
    outline.removeFromParent();
    if (p) {
      outline.scale.set(p.k.hw * 2.06, p.k.hh * 2.06, 1);
      outline.position.set((p.k.box.min.x + p.k.box.max.x) / 2, (p.k.box.min.y + p.k.box.max.y) / 2, 0);
      p.mesh.add(outline);
    }
    handles.hidden = !p;
  }

  /** 판 위 점(판 좌표)에 있는 맨 위 스티커 (칼선 모양 그대로) */
  function pick(pt) {
    for (let i = placed.length - 1; i >= 0; i--) {
      const p = placed[i];
      const dx = pt.x - p.x;
      const dy = pt.y - p.y;
      const c = Math.cos(p.rot);
      const s = Math.sin(p.rot);
      const lx = (dx * c + dy * s) / p.scale;
      const ly = (-dx * s + dy * c) / p.scale;
      const b = p.k.box;
      if (lx < b.min.x || lx > b.max.x || ly < b.min.y || ly > b.max.y) continue;
      const [u, v] = p.k.toUV(lx, ly);
      const m = p.k.mask;
      const px = Math.min(m.w - 1, Math.max(0, Math.floor(u * m.w)));
      const py = Math.min(m.h - 1, Math.max(0, Math.floor(v * m.h)));
      if (m.a[py * m.w + px] > 127) return p;
    }
    return null;
  }

  // ---------- 화면 ↔ 판 ----------
  const ray = new THREE.Raycaster();
  const ndc = new THREE.Vector2();
  const plane = new THREE.Plane();
  const tmp = new THREE.Vector3();
  const nrm = new THREE.Vector3();

  function boardPoint(clientX, clientY) {
    const r = canvas.getBoundingClientRect();
    ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    ray.setFromCamera(ndc, camera);
    board.updateMatrixWorld();
    nrm.set(0, 0, 1).applyQuaternion(board.quaternion);
    plane.setFromNormalAndCoplanarPoint(nrm, board.position);
    if (!ray.ray.intersectPlane(plane, tmp)) return null;
    return board.worldToLocal(tmp.clone());
  }
  const inBoard = (v) => ({ x: THREE.MathUtils.clamp(v.x, -BW / 2, BW / 2), y: THREE.MathUtils.clamp(v.y, -BH / 2, BH / 2) });

  // ---------- 판 움직이기 ----------
  const view = { zoom: 1, fit: 1 };
  const ang = { pitch: TILT0, yaw: 0 };
  const eul = new THREE.Euler(0, 0, 0, 'YXZ');
  const vel = { x: 0, y: 0, t: 0 };

  /** 끈 만큼 판을 젖히고(위아래) 돌린다(좌우) — 범위 끝에 닿으면 그 방향 관성은 멈춤 */
  function tilt(dx, dy) {
    const yaw = THREE.MathUtils.clamp(ang.yaw + dx * 0.006, ...YAW);
    const pitch = THREE.MathUtils.clamp(ang.pitch + dy * 0.006, ...PITCH);
    if (yaw === ang.yaw) vel.x = 0;
    if (pitch === ang.pitch) vel.y = 0;
    ang.yaw = yaw;
    ang.pitch = pitch;
    board.quaternion.setFromEuler(eul.set(pitch, yaw, 0, 'YXZ'));
  }
  tilt(0, 0);
  function pan(dx, dy) {
    const h = canvas.clientHeight || 1;
    const k = (2 * camera.position.z * Math.tan(THREE.MathUtils.degToRad(FOV) / 2)) / h;
    board.position.x = THREE.MathUtils.clamp(board.position.x + dx * k, -0.25, 0.25);
    board.position.y = THREE.MathUtils.clamp(board.position.y - dy * k, -0.2, 0.2);
  }
  function setZoom(z) {
    view.zoom = THREE.MathUtils.clamp(z, 0.6, 3);
    camera.position.set(0, 0, view.fit / view.zoom);
    camera.lookAt(0, 0, 0);
  }
  function resetBoard() {
    ang.pitch = TILT0;
    ang.yaw = 0;
    tilt(0, 0);
    board.position.set(0, 0, 0);
    vel.x = vel.y = 0;
    setZoom(1);
  }

  function resize() {
    const w = frame.clientWidth;
    const h = frame.clientHeight;
    if (!w || !h) return;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    const t = Math.tan(THREE.MathUtils.degToRad(FOV) / 2);
    view.fit = Math.max((BH / 2) * 1.12 / t, (BW / 2) * 1.08 / (t * camera.aspect));
    setZoom(view.zoom);
  }
  new ResizeObserver(resize).observe(frame);

  // ---------- 손잡이 (돌리기 · 크기, 떼기) ----------
  const handles = $('#handles');
  const knob = $('#handleRotate');
  const del = $('#handleDelete');
  const corner = new THREE.Vector3();

  function place(el, lx, ly) {
    corner.set(lx, ly, 0);
    selected.mesh.localToWorld(corner);
    corner.project(camera);
    el.style.left = `${((corner.x + 1) / 2) * canvas.clientWidth}px`;
    el.style.top = `${((1 - corner.y) / 2) * canvas.clientHeight}px`;
  }
  function placeHandles() {
    if (!selected) return;
    const b = selected.k.box;
    const cx = (b.min.x + b.max.x) / 2;
    const cy = (b.min.y + b.max.y) / 2;
    place(knob, cx + selected.k.hw * 1.03, cy - selected.k.hh * 1.03);
    place(del, cx - selected.k.hw * 1.03, cy + selected.k.hh * 1.03);
  }

  let knobDrag = null;
  knob.addEventListener('pointerdown', (e) => {
    e.preventDefault();
    e.stopPropagation();
    knob.setPointerCapture(e.pointerId);
    const pt = boardPoint(e.clientX, e.clientY);
    if (!pt || !selected) return;
    knobDrag = { p: selected, a0: Math.atan2(pt.y - selected.y, pt.x - selected.x), d0: Math.hypot(pt.x - selected.x, pt.y - selected.y), rot0: selected.rot, s0: selected.scale };
  });
  knob.addEventListener('pointermove', (e) => {
    if (!knobDrag) return;
    const pt = boardPoint(e.clientX, e.clientY);
    if (!pt) return;
    const p = knobDrag.p;
    p.rot = knobDrag.rot0 + Math.atan2(pt.y - p.y, pt.x - p.x) - knobDrag.a0;
    p.scale = THREE.MathUtils.clamp((knobDrag.s0 * Math.hypot(pt.x - p.x, pt.y - p.y)) / Math.max(knobDrag.d0, 1e-4), 0.4, 3);
    apply(p);
  });
  const knobEnd = () => {
    knobDrag = null;
  };
  knob.addEventListener('pointerup', knobEnd);
  knob.addEventListener('pointercancel', knobEnd);
  del.addEventListener('click', () => remove(selected));

  // ---------- 끌기 ----------
  const pointers = new Map();
  let drag = null;

  function startPinch() {
    const [a, b] = [...pointers.values()];
    drag = { type: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), z0: view.zoom, mx: (a.x + b.x) / 2, my: (a.y + b.y) / 2 };
  }

  canvas.addEventListener('pointerdown', (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    vel.x = vel.y = 0;
    if (pointers.size === 2) {
      if (drag?.type === 'sticker') toFront(drag.p);
      startPinch();
      return;
    }
    if (pointers.size > 2) return;
    if (e.button === 2 || e.shiftKey) {
      drag = { type: 'pan', x: e.clientX, y: e.clientY };
      return;
    }
    const pt = boardPoint(e.clientX, e.clientY);
    const hit = pt && pick(pt);
    if (hit) {
      select(hit);
      toFront(hit);
      drag = { type: 'sticker', p: hit, ox: pt.x - hit.x, oy: pt.y - hit.y };
      canvas.style.cursor = 'grabbing';
    } else {
      select(null);
      drag = { type: 'tilt', x: e.clientX, y: e.clientY };
    }
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!pointers.has(e.pointerId)) {
      const pt = boardPoint(e.clientX, e.clientY);
      canvas.style.cursor = pt && pick(pt) ? 'grab' : 'default';
      return;
    }
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    if (!drag) return;
    if (drag.type === 'pinch' && pointers.size >= 2) {
      const [a, b] = [...pointers.values()];
      setZoom((drag.z0 * Math.hypot(a.x - b.x, a.y - b.y)) / Math.max(drag.d0, 1));
      const mx = (a.x + b.x) / 2;
      const my = (a.y + b.y) / 2;
      pan(mx - drag.mx, my - drag.my);
      drag.mx = mx;
      drag.my = my;
    } else if (drag.type === 'sticker') {
      const pt = boardPoint(e.clientX, e.clientY);
      if (!pt) return;
      const q = inBoard({ x: pt.x - drag.ox, y: pt.y - drag.oy });
      drag.p.x = q.x;
      drag.p.y = q.y;
      apply(drag.p);
    } else if (drag.type === 'tilt') {
      const dx = e.clientX - drag.x;
      const dy = e.clientY - drag.y;
      tilt(dx, dy);
      Object.assign(vel, { x: dx, y: dy, t: performance.now() });
      drag.x = e.clientX;
      drag.y = e.clientY;
    } else if (drag.type === 'pan') {
      pan(e.clientX - drag.x, e.clientY - drag.y);
      drag.x = e.clientX;
      drag.y = e.clientY;
    }
  });

  function end(e) {
    pointers.delete(e.pointerId);
    if (drag?.type === 'tilt' && performance.now() - vel.t > 80) vel.x = vel.y = 0; // 멈췄다 놓으면 관성 없음
    if (drag?.type !== 'tilt') vel.x = vel.y = 0;
    drag = null;
    canvas.style.cursor = 'default';
  }
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
  canvas.addEventListener(
    'wheel',
    (e) => {
      e.preventDefault();
      setZoom(view.zoom * Math.exp(-e.deltaY * 0.0015));
    },
    { passive: false },
  );
  window.addEventListener('keydown', (e) => {
    if (!active || !selected || e.target.closest?.('input, select, textarea')) return;
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      remove(selected);
    }
  });

  // ---------- 스티커 목록 (누르면 가운데에, 끌면 놓은 자리에) ----------
  const sheet = $('#sheet');
  sheet.innerHTML = list
    .map(
      (s) => `<button type="button" class="sheet-item" data-id="${esc(s.id)}" aria-label="${esc(s.name)} 스티커 붙이기">
      <img src="${esc(s.thumb)}" alt="" draggable="false"><span><b>${esc(s.name)}</b><small>${esc(s.tab)} · SAM 3D</small></span></button>`,
    )
    .join('');
  const overCanvas = (x, y) => {
    const r = canvas.getBoundingClientRect();
    return x >= r.left && x <= r.right && y >= r.top && y <= r.bottom;
  };
  for (const btn of sheet.querySelectorAll('.sheet-item')) {
    const s = list.find((t) => t.id === btn.dataset.id);
    let d = null;
    btn.addEventListener('pointerdown', (e) => {
      e.preventDefault();
      btn.setPointerCapture(e.pointerId);
      d = { x: e.clientX, y: e.clientY, moved: false, p: null, busy: false };
      kind(s);
    });
    btn.addEventListener('pointermove', async (e) => {
      if (!d) return;
      if (Math.hypot(e.clientX - d.x, e.clientY - d.y) > 6) d.moved = true;
      if (!d.moved || !overCanvas(e.clientX, e.clientY)) return;
      const pt = boardPoint(e.clientX, e.clientY);
      if (!pt) return;
      const q = inBoard(pt);
      if (!d.p && !d.busy) {
        d.busy = true;
        d.p = await add(s, q.x, q.y);
        select(d.p);
      } else if (d.p) {
        d.p.x = q.x;
        d.p.y = q.y;
        apply(d.p);
      }
    });
    const up = async () => {
      if (d && !d.moved) {
        const n = placed.length;
        const p = await add(s, ((n % 5) - 2) * 0.012, (((n + 2) % 3) - 1) * 0.01);
        select(p);
      }
      d = null;
    };
    btn.addEventListener('pointerup', up);
    btn.addEventListener('pointercancel', () => {
      d = null;
    });
    btn.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        d = { moved: false };
        up();
      }
    });
  }

  $('#boardReset').addEventListener('click', resetBoard);
  $('#boardClear').addEventListener('click', () => {
    for (const p of [...placed]) remove(p);
  });

  function status() {
    $('#boardStatus').textContent = placed.length ? `붙인 스티커 ${placed.length}장` : '판이 비었어요. 왼쪽 목록에서 끌어다 붙이세요.';
  }

  // 처음: 스티커를 하나씩 나란히 붙여 둔다
  const loading = $('#boardLoading');
  const first = Promise.all(list.map((s) => kind(s)))
    .then(async () => {
      const n = list.length;
      for (let i = 0; i < n; i++) {
        const k = (i + 1) / (n + 1) - 0.5;
        const tilt0 = (i % 2 ? 1 : -1) * 0.12;
        if (portrait) await add(list[i], (i % 2 ? 0.025 : -0.025), -k * BH, tilt0);
        else await add(list[i], k * BW, 0.005, tilt0);
      }
      loading.hidden = true;
    })
    .catch((e) => {
      loading.querySelector('p').textContent = '스티커를 불러오지 못했어요.';
      loading.querySelector('p').classList.add('error');
      console.error(e);
    });

  let active = false;
  function frameLoop() {
    if (!drag && (Math.abs(vel.x) > 0.05 || Math.abs(vel.y) > 0.05)) {
      tilt(vel.x, vel.y);
      vel.x *= 0.88;
      vel.y *= 0.88;
    }
    placeHandles();
    renderer.render(scene, camera);
  }
  resize();
  status();
  // 확인용 (헤드리스 브라우저 테스트)
  window.__stickers = { placed, board, camera, add, select, remove, tilt, setZoom, resetBoard, ready: first, list };
  return {
    start() {
      active = true;
      resize();
      renderer.setAnimationLoop(frameLoop);
    },
    stop() {
      active = false;
      renderer.setAnimationLoop(null);
    },
  };
}
