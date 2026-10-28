#!/usr/bin/env python3
# Modified for HIL-UMI.
"""Interactive four-stage boundary annotation for a LeRobot UMI dataset.

The browser displays the front and wrist videos for one episode, synchronized
to a frame-level timeline. Three confirmed boundaries divide the episode into
four half-open ranges. Completed annotations are written atomically in the
``augmentation_metadata.json`` format already consumed by advantage.
"""

# ruff: noqa: RUF001

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading
from typing import Any

from flask import Flask
from flask import jsonify
from flask import render_template_string
from flask import request
from flask import send_file
import pyarrow.parquet as pq

FRONT_VIDEO_KEY = "extra_view_image"
WRIST_VIDEO_KEY = "image"


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, path)


def validate_boundaries(boundaries: Any, num_frames: int) -> list[int]:
    if not isinstance(boundaries, list) or len(boundaries) != 3:
        raise ValueError("boundaries must contain exactly three frame indices")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in boundaries):
        raise ValueError("boundary frame indices must be integers")
    if not 0 < boundaries[0] < boundaries[1] < boundaries[2] < num_frames:
        raise ValueError(f"boundaries must satisfy 0 < b1 < b2 < b3 < {num_frames}")
    return boundaries


class AnnotationStore:
    def __init__(self, dataset_dir: Path, annotation_path: Path):
        self.dataset_dir = dataset_dir.expanduser().resolve()
        self.annotation_path = annotation_path.expanduser().resolve()
        self._lock = threading.Lock()
        self.info = self._read_info()
        self.episodes = self._read_episodes()
        self.annotations = self._read_annotations()

    def _read_info(self) -> dict[str, Any]:
        info_path = self.dataset_dir / "meta/info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
        return json.loads(info_path.read_text())

    def _episode_paths(self, episode_index: int) -> tuple[Path, Path, Path]:
        chunk_size = int(self.info["chunks_size"])
        chunk_index = episode_index // chunk_size
        parquet_path = self.dataset_dir / self.info["data_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        front_path = self.dataset_dir / self.info["video_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=FRONT_VIDEO_KEY,
        )
        wrist_path = self.dataset_dir / self.info["video_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
            video_key=WRIST_VIDEO_KEY,
        )
        return parquet_path, front_path, wrist_path

    def _read_episodes(self) -> dict[int, dict[str, Any]]:
        episode_records: dict[int, dict[str, Any]] = {}
        episodes_path = self.dataset_dir / "meta/episodes.jsonl"
        if episodes_path.is_file():
            for line in episodes_path.read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                episode_records[int(record["episode_index"])] = record

        total_episodes = int(self.info["total_episodes"])
        fps = float(self.info["fps"])
        episodes = {}
        for episode_index in range(total_episodes):
            parquet_path, front_path, wrist_path = self._episode_paths(episode_index)
            if not parquet_path.is_file():
                raise FileNotFoundError(f"Missing parquet for episode {episode_index}: {parquet_path}")
            if not front_path.is_file() or not wrist_path.is_file():
                raise FileNotFoundError(f"Missing UMI video for episode {episode_index}")
            num_frames = pq.read_metadata(parquet_path).num_rows
            record = episode_records.get(episode_index, {})
            tasks = record.get("tasks", [])
            episodes[episode_index] = {
                "episode_index": episode_index,
                "num_frames": num_frames,
                "fps": fps,
                "duration_seconds": num_frames / fps,
                "task": tasks[0] if tasks else "",
                "parquet_path": parquet_path,
                "front_path": front_path,
                "wrist_path": wrist_path,
            }
        return episodes

    def _read_annotations(self) -> dict[str, Any]:
        if not self.annotation_path.exists():
            return {}
        payload = json.loads(self.annotation_path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Annotation file must contain a JSON object: {self.annotation_path}")
        for episode_key, annotation in payload.items():
            episode_index = int(episode_key)
            if episode_index not in self.episodes:
                continue
            validate_boundaries(
                annotation.get("subtask_completion_indices"),
                self.episodes[episode_index]["num_frames"],
            )
        return payload

    def episode_summary(self, episode_index: int) -> dict[str, Any]:
        if episode_index not in self.episodes:
            raise KeyError(episode_index)
        episode = self.episodes[episode_index]
        annotation = self.annotations.get(str(episode_index), {})
        boundaries = annotation.get("subtask_completion_indices", [])
        return {
            "episode_index": episode_index,
            "num_frames": episode["num_frames"],
            "fps": episode["fps"],
            "duration_seconds": episode["duration_seconds"],
            "task": episode["task"],
            "boundaries": boundaries,
            "completed": len(boundaries) == 3,
            "front_video_url": f"media/{episode_index}/front",
            "wrist_video_url": f"media/{episode_index}/wrist",
        }

    def all_summaries(self) -> list[dict[str, Any]]:
        return [self.episode_summary(index) for index in sorted(self.episodes)]

    def save(self, episode_index: int, boundaries: Any) -> dict[str, Any]:
        if episode_index not in self.episodes:
            raise KeyError(episode_index)
        checked = validate_boundaries(boundaries, self.episodes[episode_index]["num_frames"])
        with self._lock:
            existing = dict(self.annotations.get(str(episode_index), {}))
            existing["subtask_completion_indices"] = checked
            existing.setdefault("segments", [])
            self.annotations[str(episode_index)] = existing
            ordered = {key: self.annotations[key] for key in sorted(self.annotations, key=int)}
            atomic_write_json(self.annotation_path, ordered)
            self.annotations = ordered
        return self.episode_summary(episode_index)

    def delete(self, episode_index: int) -> None:
        if episode_index not in self.episodes:
            raise KeyError(episode_index)
        with self._lock:
            self.annotations.pop(str(episode_index), None)
            ordered = {key: self.annotations[key] for key in sorted(self.annotations, key=int)}
            atomic_write_json(self.annotation_path, ordered)
            self.annotations = ordered

    def video_path(self, episode_index: int, camera: str) -> Path:
        if episode_index not in self.episodes:
            raise KeyError(episode_index)
        if camera == "front":
            return self.episodes[episode_index]["front_path"]
        if camera == "wrist":
            return self.episodes[episode_index]["wrist_path"]
        raise KeyError(camera)


PAGE = r"""
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>HIL-UMI 阶段轨迹标注</title>
  <style>
    :root { color-scheme: dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --text:#e6edf3;
      --muted:#8b949e; --accent:#58a6ff; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
      --s0:#1f6feb; --s1:#8957e5; --s2:#bf8700; --s3:#238636; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.45 system-ui,-apple-system,sans-serif; }
    header { position:sticky; top:0; z-index:10; display:flex; gap:14px; align-items:center; padding:12px 20px;
      background:rgba(13,17,23,.96); border-bottom:1px solid var(--line); }
    h1 { margin:0; font-size:19px; white-space:nowrap; }
    select,button { background:#21262d; color:var(--text); border:1px solid #444c56; border-radius:7px;
      padding:8px 11px; font:inherit; }
    button { cursor:pointer; } button:hover { border-color:var(--accent); } button.primary { background:#1f6feb; }
    button.success { background:#238636; } button.danger { color:#ffb3ae; }
    #progress { margin-left:auto; color:var(--muted); white-space:nowrap; }
    main { max-width:1500px; margin:auto; padding:18px; }
    .panel { background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px; margin-bottom:14px; }
    .videos { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .video-box { position:relative; background:#000; border-radius:8px; overflow:hidden; }
    video { display:block; width:100%; max-height:55vh; background:#000; }
    .tag { position:absolute; top:8px; left:8px; z-index:2; padding:3px 8px; border-radius:5px;
      background:rgba(0,0,0,.7); }
    .status-row,.actions,.nav { display:flex; align-items:center; gap:9px; flex-wrap:wrap; }
    .status-row { justify-content:space-between; margin-bottom:9px; }
    #timeline { width:100%; accent-color:var(--accent); }
    .track { position:relative; height:18px; margin:3px 2px 12px; border-radius:5px; overflow:hidden; background:#30363d; }
    .segment { position:absolute; top:0; bottom:0; opacity:.75; }
    .marker { position:absolute; top:-2px; bottom:-2px; width:3px; background:white; box-shadow:0 0 5px #000; }
    .stages { display:grid; grid-template-columns:repeat(4,1fr); gap:8px; margin:10px 0 14px; }
    .stage { border:1px solid var(--line); border-left-width:5px; border-radius:7px; padding:8px; background:#0d1117; }
    .stage strong { display:block; } .stage span { color:var(--muted); font-variant-numeric:tabular-nums; }
    .hint { color:var(--muted); } #message { min-height:22px; margin-top:8px; }
    .error { color:#ff7b72; } .saved { color:#56d364; }
    @media (max-width:900px) { .videos { grid-template-columns:1fr; } .stages { grid-template-columns:1fr 1fr; }
      header { flex-wrap:wrap; } #progress { margin-left:0; } }
  </style>
</head>
<body>
<header>
  <h1>χ₀ · 叠毛巾四阶段标注</h1>
  <select id="episodeSelect"></select>
  <button id="prevBtn">← 上一条</button>
  <button id="nextBtn">下一条 →</button>
  <button id="nextOpenBtn">下一条未标注</button>
  <span id="progress"></span>
</header>
<main>
  <section class="panel">
    <div class="status-row">
      <div><strong id="episodeTitle"></strong> · <span id="taskText" class="hint"></span></div>
      <div id="frameText"></div>
    </div>
    <div class="videos">
      <div class="video-box"><span class="tag">Front · extra_view_image</span><video id="frontVideo" controls muted></video></div>
      <div class="video-box"><span class="tag">Wrist · image</span><video id="wristVideo" muted></video></div>
    </div>
  </section>
  <section class="panel">
    <input id="timeline" type="range" min="0" max="1" value="0" step="1">
    <div id="track" class="track"></div>
    <div id="stages" class="stages"></div>
    <div class="actions">
      <button data-boundary="0">确认边界 1</button>
      <button data-boundary="1">确认边界 2</button>
      <button data-boundary="2">确认边界 3</button>
      <button id="clearDraftBtn" class="danger">清空当前标记</button>
      <button id="saveBtn" class="success">保存并进入下一条未标注</button>
    </div>
    <div id="message" class="hint">拖动时间轴到下一阶段的第一帧，然后依次确认三个边界。</div>
  </section>
  <section class="panel nav">
    <span class="hint">快捷键：空格播放/暂停；←/→ 逐帧；数字 1/2/3 确认边界。</span>
  </section>
</main>
<script>
const state = { episodes: [], current: 0, boundaries: [null,null,null], currentFrame: 0, syncing: false };
const el = id => document.getElementById(id);
const front = el('frontVideo'), wrist = el('wristVideo'), timeline = el('timeline');

async function api(url, options={}) {
  const response = await fetch(url, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
function episode() { return state.episodes[state.current]; }
function formatTime(frame) { const ep=episode(); return `${(frame/ep.fps).toFixed(2)}s`; }
function validBoundaries() {
  const [a,b,c]=state.boundaries, n=episode().num_frames;
  return Number.isInteger(a) && Number.isInteger(b) && Number.isInteger(c) && 0<a && a<b && b<c && c<n;
}
function setMessage(text, kind='') { const box=el('message'); box.textContent=text; box.className=kind; }
function updateHeader() {
  const completed=state.episodes.filter(item=>item.completed).length;
  el('progress').textContent=`已完成 ${completed} / ${state.episodes.length}`;
  el('episodeTitle').textContent=`Episode ${String(episode().episode_index).padStart(6,'0')}`;
  el('taskText').textContent=episode().task;
  state.episodes.forEach((item,index) => { if (el('episodeSelect').options[index]) {
    el('episodeSelect').options[index].textContent=`${item.completed?'✓':'○'} Episode ${String(item.episode_index).padStart(6,'0')}`;
  }});
  el('episodeSelect').value=String(state.current);
}
function updateFrame() {
  const ep=episode();
  el('frameText').textContent=`Frame ${state.currentFrame} / ${ep.num_frames-1} · ${formatTime(state.currentFrame)}`;
  timeline.value=String(state.currentFrame);
}
function updateStages() {
  const ep=episode(), n=ep.num_frames, b=state.boundaries;
  const cuts=[0,b[0],b[1],b[2],n];
  const colors=['var(--s0)','var(--s1)','var(--s2)','var(--s3)'];
  const track=el('track'); track.innerHTML='';
  for (let i=0;i<4;i++) {
    if (cuts[i] == null) continue;
    const end=cuts[i+1] == null ? cuts[i] : cuts[i+1];
    const seg=document.createElement('div'); seg.className='segment'; seg.style.background=colors[i];
    seg.style.left=`${100*cuts[i]/n}%`; seg.style.width=`${100*Math.max(0,end-cuts[i])/n}%`; track.appendChild(seg);
  }
  b.forEach(value => { if (value == null) return; const marker=document.createElement('div'); marker.className='marker';
    marker.style.left=`calc(${100*value/n}% - 1px)`; track.appendChild(marker); });
  const stages=el('stages'); stages.innerHTML='';
  for (let i=0;i<4;i++) { const card=document.createElement('div'); card.className='stage'; card.style.borderLeftColor=colors[i];
    const start=cuts[i], end=cuts[i+1]; const range=(start==null||end==null) ? '未完成' : `[${start}, ${end}) · ${end-start} frames`;
    card.innerHTML=`<strong>Stage ${i}</strong><span>${range}</span>`; stages.appendChild(card); }
  el('saveBtn').disabled=!validBoundaries();
}
function seek(frame) {
  const ep=episode(); state.currentFrame=Math.max(0,Math.min(ep.num_frames-1,Math.round(frame)));
  const time=state.currentFrame/ep.fps; state.syncing=true; front.currentTime=time; wrist.currentTime=time; state.syncing=false;
  updateFrame();
}
async function loadEpisode(index) {
  state.current=Math.max(0,Math.min(state.episodes.length-1,index));
  const ep=state.episodes[state.current]; state.boundaries=ep.boundaries.length===3 ? [...ep.boundaries] : [null,null,null];
  state.currentFrame=0; timeline.max=String(ep.num_frames-1); front.src=ep.front_video_url; wrist.src=ep.wrist_video_url;
  front.load(); wrist.load(); updateHeader(); updateFrame(); updateStages(); setMessage(ep.completed?'该 episode 已标注，可直接编辑后重新保存。':'拖动时间轴并确认三个边界。');
}
function nextUnannotated() {
  for (let offset=1; offset<=state.episodes.length; offset++) { const index=(state.current+offset)%state.episodes.length;
    if (!state.episodes[index].completed) { loadEpisode(index); return; } }
  setMessage('全部 episode 已完成标注。','saved');
}
async function save() {
  if (!validBoundaries()) { setMessage('三个边界必须严格递增且位于 episode 内。','error'); return; }
  try { const ep=episode(); const saved=await api(`api/episodes/${ep.episode_index}/boundaries`, {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({boundaries:state.boundaries}) });
    Object.assign(ep,saved); updateHeader(); setMessage('已保存。','saved'); nextUnannotated();
  } catch (error) { setMessage(error.message,'error'); }
}
timeline.addEventListener('input', () => seek(Number(timeline.value)));
front.addEventListener('timeupdate', () => { if (state.syncing||!episode()) return; const frame=Math.round(front.currentTime*episode().fps);
  if (frame!==state.currentFrame) { state.currentFrame=Math.min(frame,episode().num_frames-1); updateFrame(); }
  if (Math.abs(wrist.currentTime-front.currentTime)>.06) wrist.currentTime=front.currentTime; });
front.addEventListener('play', () => wrist.play().catch(()=>{})); front.addEventListener('pause', () => wrist.pause());
front.addEventListener('seeking', () => { if (!state.syncing) wrist.currentTime=front.currentTime; });
document.querySelectorAll('[data-boundary]').forEach(button => button.addEventListener('click', () => {
  const index=Number(button.dataset.boundary); state.boundaries[index]=state.currentFrame; updateStages();
  setMessage(`边界 ${index+1} = frame ${state.currentFrame}（下一阶段首帧）`); }));
el('clearDraftBtn').addEventListener('click', async () => { state.boundaries=[null,null,null]; updateStages();
  if (episode().completed && confirm('同时删除这个 episode 已保存的标注吗？')) { await api(`api/episodes/${episode().episode_index}/boundaries`,{method:'DELETE'});
    episode().completed=false; episode().boundaries=[]; updateHeader(); } });
el('saveBtn').addEventListener('click', save); el('prevBtn').addEventListener('click',()=>loadEpisode(state.current-1));
el('nextBtn').addEventListener('click',()=>loadEpisode(state.current+1)); el('nextOpenBtn').addEventListener('click',nextUnannotated);
el('episodeSelect').addEventListener('change',event=>loadEpisode(Number(event.target.value)));
document.addEventListener('keydown', event => { if (['INPUT','SELECT'].includes(event.target.tagName)) return;
  if (event.code==='Space') { event.preventDefault(); front.paused?front.play():front.pause(); }
  else if (event.key==='ArrowLeft') seek(state.currentFrame-1); else if (event.key==='ArrowRight') seek(state.currentFrame+1);
  else if (['1','2','3'].includes(event.key)) { state.boundaries[Number(event.key)-1]=state.currentFrame; updateStages(); } });

async function init() {
  try { state.episodes=await api('api/episodes'); const select=el('episodeSelect');
    state.episodes.forEach((ep,index)=>{ const option=document.createElement('option'); option.value=String(index);
      option.textContent=`${ep.completed?'✓':'○'} Episode ${String(ep.episode_index).padStart(6,'0')}`; select.appendChild(option); });
    const firstOpen=state.episodes.findIndex(ep=>!ep.completed); await loadEpisode(firstOpen>=0?firstOpen:0);
  } catch (error) { setMessage(error.message,'error'); }
}
init();
</script>
</body>
</html>
"""


def create_app(dataset_dir: Path, annotation_path: Path | None = None) -> Flask:
    dataset_dir = dataset_dir.expanduser().resolve()
    annotation_path = annotation_path or dataset_dir / "augmentation_metadata.json"
    store = AnnotationStore(dataset_dir, annotation_path)
    app = Flask(__name__)
    app.config["ANNOTATION_STORE"] = store

    @app.get("/")
    def index():
        return render_template_string(PAGE)

    @app.get("/api/episodes")
    def list_episodes():
        return jsonify(store.all_summaries())

    @app.get("/api/episodes/<int:episode_index>")
    def get_episode(episode_index: int):
        try:
            return jsonify(store.episode_summary(episode_index))
        except KeyError:
            return jsonify({"error": f"Unknown episode: {episode_index}"}), 404

    @app.put("/api/episodes/<int:episode_index>/boundaries")
    def save_boundaries(episode_index: int):
        try:
            payload = request.get_json(force=True)
            return jsonify(store.save(episode_index, payload.get("boundaries")))
        except KeyError:
            return jsonify({"error": f"Unknown episode: {episode_index}"}), 404
        except (AttributeError, TypeError, ValueError) as error:
            return jsonify({"error": str(error)}), 400

    @app.delete("/api/episodes/<int:episode_index>/boundaries")
    def delete_boundaries(episode_index: int):
        try:
            store.delete(episode_index)
            return jsonify({"ok": True})
        except KeyError:
            return jsonify({"error": f"Unknown episode: {episode_index}"}), 404

    @app.get("/media/<int:episode_index>/<camera>")
    def media(episode_index: int, camera: str):
        try:
            path = store.video_path(episode_index, camera)
        except KeyError:
            return jsonify({"error": "Unknown episode or camera"}), 404
        return send_file(path, mimetype="video/mp4", conditional=True)

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="LeRobot dataset root")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Output JSON path (default: <dataset>/augmentation_metadata.json)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(args.dataset, args.annotations)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
