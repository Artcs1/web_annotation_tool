import os
import json
import webbrowser
from flask import Flask, jsonify, render_template_string, send_from_directory, request
from threading import Timer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
ANNOTATION_DIR = os.path.join(BASE_DIR, "annotations")
HOST = "127.0.0.1"
PORT = 5000

app = Flask(__name__, static_folder=BASE_DIR)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Annotation Viewer</title>
<style>
body { font-family: sans-serif; text-align:center; background:#f0f2f5; margin:0; padding:0; }
.container { padding:20px; display:inline-block; background:white; margin-top:20px; border-radius:10px; box-shadow:0 0 10px rgba(0,0,0,0.2); }
#canvas-container { position:relative; display:inline-block; border:1px solid #ccc; margin:auto; }
#frame { display:block; max-width:100%; height:auto; }
#overlay { position:absolute; left:0; top:0; width:100%; height:100%; pointer-events:none; }
.box { position:absolute; border:2px solid red; box-sizing:border-box; }
.label { position:absolute; color:white; font-size:12px; padding:2px 4px; border-radius:3px; white-space:nowrap; }
button { padding:6px 12px; margin:5px; font-size:14px; cursor:pointer; }
</style>
</head>
<body>
<div class="container">
<h2>Video Annotation Viewer</h2>
<div id="canvas-container">
<img id="frame" />
<div id="overlay"></div>
</div>
<div>
<button id="next-clip">Next Video (n)</button>
<button id="next-annot">Next Annotator (m)</button>
</div>
<div id="info" style="margin-top:10px;"></div>
</div>

<script>
const frame = document.getElementById('frame');
const overlay = document.getElementById('overlay');
const info = document.getElementById('info');
let state = { clip:0, annot:0, clips:[], annots:[], annotations:[] };

const CONF_COLOR = {
    5: "red",
    4: "orange",
    3: "green",
    2: "blue",
    1: "purple"
};

function drawBoxes(){
    overlay.innerHTML = '';
    if (!state.annotations.length) return;

    const imgWidth = frame.clientWidth;
    const imgHeight = frame.clientHeight;

    const naturalWidth = frame.naturalWidth || 1920;
    const naturalHeight = frame.naturalHeight || 1080;

    const scaleX = imgWidth / naturalWidth;
    const scaleY = imgHeight / naturalHeight;

    state.annotations.forEach(b=>{
        const [x1,y1,x2,y2] = b.box2d;
        const left = x1 * scaleX;
        const top = y1 * scaleY;
        const width = (x2 - x1) * scaleX;
        const height = (y2 - y1) * scaleY;
        const color = CONF_COLOR[b.confidence] || "gray";

        const div = document.createElement('div');
        div.className = 'box';
        div.style.left = left + 'px';
        div.style.top = top + 'px';
        div.style.width = width + 'px';
        div.style.height = height + 'px';
        div.style.border = `2px solid ${color}`;
        overlay.appendChild(div);

        if (b.label) {
            const lbl = document.createElement('div');
            lbl.className = 'label';
            lbl.textContent = b.label + " (" + b.confidence + ")";
            lbl.style.left = left + 'px';
            lbl.style.top = (top - 18) + 'px';
            lbl.style.background = color;
            overlay.appendChild(lbl);
        }
    });
}

async function loadData(){
    try{
        const res = await fetch(`/api/data?clip_index=${state.clip}&annotator_index=${state.annot}`);
        const data = await res.json();
        if(data.error){ info.textContent=data.error; return; }

        state.clips = data.clips;
        state.annots = data.annotators;
        state.annotations = data.annotations;

        frame.src = data.image_url;
        info.innerHTML = `<b>Clip:</b> ${state.clips[state.clip]} | <b>Annotator:</b> ${state.annots[state.annot]}`;
        frame.onload = drawBoxes;
    }catch(err){
        console.error(err);
        info.textContent='Failed to load data.';
    }
}

document.getElementById('next-clip').onclick = ()=>{
    state.clip = (state.clip + 1) % state.clips.length;
    loadData();
};
document.getElementById('next-annot').onclick = ()=>{
    state.annot = (state.annot + 1) % state.annots.length;
    loadData();
};
document.addEventListener('keydown', e=>{
    if(e.key==='n'){ state.clip=(state.clip+1)%state.clips.length; loadData(); }
    if(e.key==='m'){ state.annot=(state.annot+1)%state.annots.length; loadData(); }
});

loadData();
</script>
</body>
</html>
"""

def ensure_dirs():
    if not os.path.exists(VIDEO_DIR):
        os.makedirs(VIDEO_DIR, exist_ok=True)
    if not os.path.exists(ANNOTATION_DIR):
        os.makedirs(ANNOTATION_DIR, exist_ok=True)

@app.route('/api/data')
def api_data():
    clip_idx = int(request.args.get('clip_index',0))
    annot_idx = int(request.args.get('annotator_index',0))

    clips = sorted([d for d in os.listdir(VIDEO_DIR) if os.path.isdir(os.path.join(VIDEO_DIR,d))])
    annots = sorted([f for f in os.listdir(ANNOTATION_DIR) if f.endswith('.json')])
    if not clips or not annots:
        return jsonify({"error":"No clips or annotators found."})

    clip_idx %= len(clips)
    annot_idx %= len(annots)
    current_clip = clips[clip_idx]
    current_annot_file = annots[annot_idx]
    img_url = f"/videos/{current_clip}/00001.jpeg"

    with open(os.path.join(ANNOTATION_DIR, current_annot_file), 'r', encoding='utf-8') as f:
        data = json.load(f).get("annotations", [])

    annotations = []
    for video_ann in data:
        folder_name = video_ann.get("videoFolder", "").strip('/').split('/')[-1]
        if folder_name == current_clip:
            for g in video_ann.get("groups", []):
                annotations.append({
                    "label": f"Group {g.get('groupId','N/A')}",
                    "box2d": g.get("bbox"),
                    "confidence": g.get("confidence", 1)
                })
            break

    return jsonify({
        "clips": clips,
        "annotators": [a.replace('.json','') for a in annots],
        "image_url": img_url,
        "annotations": annotations
    })

@app.route('/videos/<clip>/<frame>')
def serve_frame(clip, frame):
    return send_from_directory(os.path.join(VIDEO_DIR, clip), frame)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

def open_browser():
    webbrowser.open_new(f"http://{HOST}:{PORT}")

if __name__ == '__main__':
    ensure_dirs()
    Timer(1, open_browser).start()
    app.run(host=HOST, port=PORT, debug=True)
