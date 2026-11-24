// 원본 영상 ↔ 3D 결과 대조 뷰어
//   data/index.json(상단 탭: 동영상 1 · 2 · 3 …)과 data/<id>/manifest.json을 읽는다 (pipeline/run_all.py · export_viewer.py가 만든다).
//   '영상 카메라 따라가기': 영상 시간에 맞춰 3D 카메라를 SfM이 구한 그 프레임의 카메라 자리로 옮긴다 (사이 프레임은 보간).
//   SAM 3D는 입력 사진 한 장의 시점, Gausify는 카메라 정보가 없어 자유 회전만.
//   마지막 탭 '스티커'는 sticker.js (SAM 3D로 만든 볼록 스티커를 판에 붙인다).
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const REPO = 'https://github.com/bloodmoon3929/goodmorning-3d';
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

const video = $('#video');
const canvas = $('#canvas');
const stage = $('#stage');
const loading = $('#loading');

// want: 사용자가 고른 카메라 모드 (카메라 정보가 없는 방법을 보는 동안에는 잠시 자유 회전)
const state = { manifest: null, method: null, mesh: null, mode: 'follow', want: 'follow', layout: 'side', mix: 0.6, token: 0, seeking: false, sticker: false };
const STICKER = 'sticker'; // 스티커 탭 id

// ---------- three.js · Spark ----------
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false, alpha: true });
renderer.setClearColor(0x000000, 0);
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
const scene = new THREE.Scene();
scene.add(new SparkRenderer({ renderer }));
const camera = new THREE.PerspectiveCamera(50, 9 / 16, 0.01, 1000);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;

function resize() {
  const box = canvas.parentElement.getBoundingClientRect();
  if (!box.width || !box.height) return;
  renderer.setSize(box.width, box.height, false);
  camera.aspect = box.width / box.height;
  camera.updateProjectionMatrix();
}
const ro = new ResizeObserver(resize);
ro.observe($('#threeFrame'));
ro.observe($('#videoFrame'));

// ---------- 카메라 ----------
const qa = new THREE.Quaternion();
const qb = new THREE.Quaternion();

/** 영상 시간 t에서의 카메라: 앞뒤 SfM 카메라를 보간 */
function poseAt(cams, t) {
  if (t <= cams[0].t) return { ...cams[0], dt: cams[0].t - t };
  const last = cams[cams.length - 1];
  if (t >= last.t) return { ...last, dt: t - last.t };
  let lo = 0;
  let hi = cams.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (cams[mid].t <= t) lo = mid;
    else hi = mid;
  }
  const a = cams[lo];
  const b = cams[hi];
  const k = (t - a.t) / (b.t - a.t);
  qa.set(...a.q);
  qb.set(...b.q);
  qa.slerp(qb, k);
  return {
    p: a.p.map((v, i) => v + (b.p[i] - v) * k),
    q: [qa.x, qa.y, qa.z, qa.w],
    frame: (k < 0.5 ? a : b).frame,
    dt: Math.min(t - a.t, b.t - t),
  };
}

function setFov(follow) {
  camera.fov = follow ? follow.fovY : 50;
  const [px, py] = follow?.principal ?? [0.5, 0.5];
  if (Math.abs(px - 0.5) > 1e-3 || Math.abs(py - 0.5) > 1e-3) {
    const w = 1000;
    const h = w / camera.aspect;
    camera.setViewOffset(w, h, (0.5 - px) * w, (0.5 - py) * h, w, h);
  } else {
    camera.clearViewOffset();
  }
  camera.updateProjectionMatrix();
}

function applyFollow() {
  const f = state.method?.follow;
  if (!f) return null;
  const c = f.kind === 'video' ? poseAt(f.cameras, video.currentTime) : f.cameras[0];
  camera.position.set(...c.p);
  camera.quaternion.set(...c.q);
  return c;
}

function freeStart() {
  const tanV = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2);
  const d = 1 / (Math.min(tanV, tanV * camera.aspect) * 0.8);
  const e = THREE.MathUtils.degToRad(12);
  camera.position.set(0, d * Math.sin(e), d * Math.cos(e));
  controls.target.set(0, 0, 0);
  camera.lookAt(0, 0, 0);
}

// ---------- 모드 · 배치 ----------
function followLabel(m) {
  if (m?.failed) return '만들지 못함';
  if (!m?.follow) return '카메라 정보 없음';
  return m.follow.kind === 'video' ? '영상 카메라 따라가기' : '입력 사진 시점';
}

function setMode(mode) {
  const m = state.method;
  if (mode === 'follow' && !m?.follow) mode = 'free';
  const entering = mode !== state.mode;
  state.mode = mode;
  controls.enabled = mode === 'free';
  setFov(m?.follow ?? null);
  if (mode === 'follow' && m.follow.kind === 'input') {
    video.pause();
    video.currentTime = m.follow.cameras[0].t;
  }
  if (mode === 'free' && entering) {
    controls.target.set(0, 0, 0);
    const r = camera.position.length();
    if (r < 1e-3 || r > 50) freeStart();
    controls.update();
  }
  const follow = $('#modeFollow');
  follow.textContent = followLabel(m);
  follow.disabled = !m?.follow;
  for (const b of document.querySelectorAll('[data-mode]')) b.setAttribute('aria-pressed', String(b.dataset.mode === state.mode));
  updateNote();
}

function setLayout(layout) {
  state.layout = layout;
  stage.classList.toggle('overlay', layout === 'overlay');
  (layout === 'overlay' ? $('#videoFrame') : $('#threeFrame')).append(canvas, loading);
  canvas.style.opacity = layout === 'overlay' ? String(state.mix) : '1';
  $('#mixLabel').classList.toggle('off', layout !== 'overlay');
  $('#mix').disabled = layout !== 'overlay';
  for (const b of document.querySelectorAll('[data-layout]')) b.setAttribute('aria-pressed', String(b.dataset.layout === layout));
  resize();
  updateNote();
}

const man = (n) => (n >= 10000 ? `${(n / 10000).toFixed(1)}만 개` : `${n.toLocaleString('ko-KR')}개`);
const mb = (bytes) => `${(bytes / 1e6).toFixed(1)}MB`;

// ---------- 방법 ----------
async function selectMethod(m) {
  const token = ++state.token;
  state.method = m;
  for (const b of document.querySelectorAll('.method')) b.setAttribute('aria-pressed', String(b.dataset.id === m.id));
  $('#threeCap').textContent = `· ${m.label} (${m.sub})`;
  renderFacts();
  if (state.mesh) {
    scene.remove(state.mesh);
    state.mesh.dispose?.();
    state.mesh = null;
  }
  const img = $('#loadingImg');
  if (m.failed) {
    img.hidden = true;
    $('#loadingText').textContent = `만들지 못함 — ${m.failed}`;
    $('#loadingText').classList.add('error');
    loading.hidden = false;
    setMode('free');
    return;
  }
  img.hidden = !m.turntable;
  if (m.turntable) img.src = `${m.turntable.url}00.webp`;
  $('#loadingText').textContent = `3D 불러오는 중 · ${mb(m.bytes)}`;
  $('#loadingText').classList.remove('error');
  loading.hidden = false;
  const was = state.mode;
  setMode(state.want);
  if (state.mode === 'free' && was === 'free') freeStart();

  const mesh = new SplatMesh({ url: m.url });
  new THREE.Matrix4().set(...m.matrix).decompose(mesh.position, mesh.quaternion, mesh.scale);
  try {
    await mesh.initialized;
  } catch (e) {
    if (token === state.token) {
      $('#loadingText').textContent = '3D를 불러오지 못했어요. 아래 비교 그림을 보세요.';
      $('#loadingText').classList.add('error');
    }
    console.error(e);
    return;
  }
  if (token !== state.token) {
    mesh.dispose?.();
    return;
  }
  scene.add(mesh);
  state.mesh = mesh;
  loading.hidden = true;
}

function updateNote() {
  const m = state.method;
  if (!m) return;
  let text;
  if (m.failed) {
    text = `이 방법은 이 영상으로 3D를 만들지 못했다: ${m.failed}.`;
  } else if (state.mode === 'free') {
    text = '끌어서 돌리고, 휠이나 두 손가락으로 확대한다.';
    if (!m.follow) text = `이 결과에는 카메라 정보가 없어 영상과 시점을 맞출 수 없다. ${text}`;
  } else if (m.follow.kind === 'video') {
    text = '영상의 프레임마다 SfM이 구한 카메라 자리에서 3D를 그린다 (카메라가 없는 프레임은 앞뒤를 보간). 3D는 굿즈만 학습해서 손 · 배경은 없다. 겹쳐 보기로 원본과 어긋난 곳을 볼 수 있다.';
  } else {
    const fit = m.follow.fit;
    text = `SAM 3D는 사진 한 장(${m.follow.cameras[0].frame})으로 만들어서 그 사진 시점(${m.follow.cameras[0].t.toFixed(2)}초)에서만 원본과 맞춰 볼 수 있다. 초점거리는 입력 마스크에 투영해 가장 잘 겹치는 값(${fit.focal_px}px, IoU ${fit.iou}).`;
  }
  $('#note').textContent = text;
}

function renderFacts() {
  const ms = state.manifest.methods;
  const rows = [
    ['방식', (m) => m.facts.how],
    ['입력', (m) => m.facts.input],
    ['카메라 자세', (m) => m.facts.poses],
    ['가우시안', (m) => (m.failed ? '—' : `${man(m.gaussians)} · 색 SH ${m.sh}차`)],
    ['확인용 PSNR', (m) => m.facts.psnr],
    ['걸린 시간', (m) => m.facts.time],
  ];
  if (ms.some((m) => m.facts.note)) rows.push(['결과', (m) => m.facts.note || '—']);
  const on = (m) => (m.id === state.method?.id ? ' class="on"' : '');
  let html = `<thead><tr><th scope="col"></th>${ms
    .map((m, i) => `<th scope="col"${on(m)}>${i + 1}. ${esc(m.label)}<br><small>${esc(m.sub)}</small></th>`)
    .join('')}</tr></thead><tbody>`;
  for (const [name, f] of rows) html += `<tr><th scope="row">${name}</th>${ms.map((m) => `<td${on(m)}>${esc(f(m))}</td>`).join('')}</tr>`;
  html += `<tr><th scope="row">파일</th>${ms
    .map((m) => `<td${on(m)}>${m.failed ? '—' : `<a href="${REPO}/blob/main/${encodeURI(m.download)}">${esc(m.download.split('/').pop())}</a>`}</td>`)
    .join('')}</tr>`;
  $('#facts').innerHTML = `${html}</tbody>`;
}

// ---------- 영상(탭) ----------
function selectTab(id) {
  for (const b of document.querySelectorAll('.tab')) {
    b.setAttribute('aria-selected', String(b.dataset.id === id));
    if (b.dataset.id === id) b.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }
  const url = new URL(location.href);
  url.searchParams.set('id', id);
  history.replaceState(null, '', url);
}

/** 스티커 탭: 영상 화면을 감추고 스티커 판을 연다 */
async function showStickers() {
  selectTab(STICKER);
  state.sticker = true;
  video.pause();
  $('#videoView').hidden = true;
  $('#stickerView').hidden = false;
  (await import('./sticker.js')).openStickers(state.stickers);
}

async function loadItem(id) {
  if (id === STICKER && state.stickers.length) return showStickers();
  if (state.sticker) {
    state.sticker = false;
    (await import('./sticker.js')).closeStickers();
    $('#stickerView').hidden = true;
    $('#videoView').hidden = false;
  }
  const tab = state.index.find((t) => t.id === id) ?? state.index[0];
  selectTab(tab.id);

  const mf = await (await fetch(`data/${tab.id}/manifest.json`)).json();
  state.manifest = mf;
  const ar = mf.video.width / mf.video.height;
  document.documentElement.style.setProperty('--ar', String(ar));
  video.pause();
  video.src = mf.video.url;
  video.load();
  video.playbackRate = Number($('#rate').value);
  $('#videoCap').textContent = `· ${tab.tab} ${mf.name}`;
  $('#itemAbout').textContent = `${tab.tab} · ${mf.name} — ${tab.source} (${mf.video.duration.toFixed(1)}초). 이 ${tab.tab.startsWith('사진') ? '사진들' : '영상'}으로 만든 3D ${mf.methods.length}가지:`;
  $('#sourceLink').href = `${REPO}/blob/main/${encodeURI(mf.video.source)}`;
  $('#compareImg').src = `../assets/${tab.id}/compare.jpg`;
  $('#methods').innerHTML = mf.methods
    .map(
      (m, i) => `<button type="button" class="method" data-id="${esc(m.id)}" aria-pressed="false">
      <b><span class="no">${i + 1}</span>${esc(m.label)}</b><span>${esc(m.sub)}${m.failed ? '' : ` · ${man(m.gaussians)}`}</span>
      <small>${esc(m.brief)}</small>
      <em class="tag${m.follow ? '' : m.failed ? ' fail' : ' off'}">${followLabel(m)}</em></button>`,
    )
    .join('');
  for (const b of document.querySelectorAll('.method')) {
    b.addEventListener('click', () => selectMethod(mf.methods.find((m) => m.id === b.dataset.id)));
  }
  const keep = mf.methods.find((m) => m.id === state.method?.id);
  resize();
  await selectMethod(keep ?? mf.methods[0]);
}

let lastTimeText = '';
function updateTime(c) {
  const t = video.currentTime;
  const dur = video.duration || state.manifest?.video.duration || 0;
  let s = `${t.toFixed(2)}초 / ${dur.toFixed(2)}초`;
  if (state.mode === 'follow' && c) {
    s += state.method.follow.kind === 'video' ? ` · 카메라 ${c.frame}${c.dt > 0.02 ? ' 근처 (보간)' : ''}` : ` · 입력 사진 ${c.frame}`;
  }
  if (s !== lastTimeText) {
    $('#time').textContent = s;
    lastTimeText = s;
  }
  if (!state.seeking) $('#seek').value = String(t);
}

// ---------- 시작 ----------
const idx = await (await fetch('data/index.json')).json();
state.index = idx.items;
state.stickers = idx.stickers ?? [];
$('#tabs').innerHTML =
  state.index
    .map((t) => `<button type="button" class="tab" role="tab" data-id="${esc(t.id)}" aria-selected="false"><b>${esc(t.tab)}</b><span>${esc(t.name)}</span></button>`)
    .join('') +
  (state.stickers.length
    ? `<button type="button" class="tab tab-sticker" role="tab" data-id="${STICKER}" aria-selected="false"><b>스티커</b><span>SAM 3D 볼록 스티커 ${state.stickers.length}장</span></button>`
    : '');
for (const b of document.querySelectorAll('.tab')) b.addEventListener('click', () => loadItem(b.dataset.id));

for (const b of document.querySelectorAll('[data-mode]')) {
  b.addEventListener('click', () => {
    state.want = b.dataset.mode;
    setMode(b.dataset.mode);
  });
}
for (const b of document.querySelectorAll('[data-layout]')) b.addEventListener('click', () => setLayout(b.dataset.layout));
$('#mix').addEventListener('input', (e) => {
  state.mix = Number(e.target.value);
  if (state.layout === 'overlay') canvas.style.opacity = String(state.mix);
});
$('#rate').addEventListener('change', (e) => {
  video.playbackRate = Number(e.target.value);
});
$('#play').addEventListener('click', () => {
  if (video.paused) video.play();
  else video.pause();
});
video.addEventListener('play', () => {
  $('#play').textContent = '멈춤';
  $('#play').setAttribute('aria-label', '멈춤');
});
video.addEventListener('pause', () => {
  $('#play').textContent = '재생';
  $('#play').setAttribute('aria-label', '재생');
});
video.addEventListener('loadedmetadata', () => {
  $('#seek').max = String(video.duration);
});
$('#seek').addEventListener('input', (e) => {
  video.currentTime = Number(e.target.value);
});
$('#seek').addEventListener('pointerdown', () => {
  state.seeking = true;
});
window.addEventListener('pointerup', () => {
  state.seeking = false;
});
video.loop = true;

setLayout('side');
renderer.setAnimationLoop(() => {
  if (state.sticker) return; // 스티커 탭에서는 영상 쪽을 그리지 않는다
  let c = null;
  if (state.mode === 'follow') c = applyFollow();
  else controls.update();
  renderer.render(scene, camera);
  updateTime(c);
});
// 확인용 (헤드리스 브라우저 테스트)
window.__viewer = { state, video, camera, selectMethod, setMode, setLayout, loadItem };
await loadItem(new URLSearchParams(location.search).get('id') || state.index[0].id);
