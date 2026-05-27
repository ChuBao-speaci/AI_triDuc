/* =========================================
   EmotionScan v2 — script.js
   ========================================= */

const video        = document.getElementById('video');
const canvas       = document.getElementById('overlay');
const ctx          = canvas.getContext('2d');
const emotionPill  = document.getElementById('emotionPill');
const emotionIcon  = document.getElementById('emotionIcon');
const emotionLabel = document.getElementById('emotionLabel');
const emotionConf  = document.getElementById('emotionConf');
const faceCount    = document.getElementById('faceCount');
const fpsDisplay   = document.getElementById('fpsDisplay');
const logBody      = document.getElementById('logBody');
const emptyState   = document.getElementById('emptyState');
const logCount     = document.getElementById('logCount');
const statIn       = document.getElementById('statIn');
const statOut      = document.getElementById('statOut');
const statHappy    = document.getElementById('statHappy');
const toast        = document.getElementById('toast');

const tempCanvas = document.createElement('canvas');
const tCtx       = tempCanvas.getContext('2d');

// App state
let currentData = { emotion: 'Neutral', confidence: 0, allScores: {} };
let logs        = [];
let countIn     = 0;
let countOut    = 0;
let countHappy  = 0;
let moodCountsIn  = {};
let moodCountsOut = {};
let running     = false;

// FPS tracking
let frameTs     = 0;
let frameCount  = 0;
let fps         = 0;

const CLASSES = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'];

const EMO_META = {
    'Angry':    { icon: '😠', cls: 'angry'    },
    'Disgust':  { icon: '🤢', cls: 'disgust'  },
    'Fear':     { icon: '😨', cls: 'fear'     },
    'Happy':    { icon: '😄', cls: 'happy'    },
    'Neutral':  { icon: '😐', cls: 'neutral'  },
    'Sad':      { icon: '😢', cls: 'sad'      },
    'Surprise': { icon: '😲', cls: 'surprise' },
};

const MOOD_COLORS = {
    'Happy':    '#22c55e',
    'Neutral':  '#64748b',
    'Sad':      '#38bdf8',
    'Angry':    '#f43f5e',
    'Surprise': '#facc15',
    'Fear':     '#fb923c',
    'Disgust':  '#a78bfa',
};

// ── Camera ──────────────────────────────────────────────────────────────────

async function startCamera() {
    // Kiểm tra HTTPS — browser chặn camera trên HTTP thuần
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        emotionLabel.textContent = 'Cần HTTPS để dùng camera!';
        showToast('⚠ Truy cập bằng https:// thay vì http://');
        // Tự động chuyển hướng sang HTTPS nếu đang ở HTTP
        if (location.protocol === 'http:' && location.hostname !== 'localhost') {
            setTimeout(() => {
                location.href = location.href.replace('http://', 'https://');
            }, 1500);
        }
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: { ideal: 1280 }, height: { ideal: 960 }, facingMode: 'user' }
        });
        video.srcObject = stream;
        video.onloadedmetadata = () => {
            canvas.width  = video.videoWidth;
            canvas.height = video.videoHeight;
            tempCanvas.width  = video.videoWidth;
            tempCanvas.height = video.videoHeight;
            running = true;
            loop();
        };
    } catch (err) {
        console.error('Camera error:', err);
        if (err.name === 'NotAllowedError') {
            emotionLabel.textContent = 'Bị chặn — hãy cho phép camera!';
            showToast('⚠ Cho phép camera trong cài đặt trình duyệt');
        } else if (err.name === 'NotFoundError') {
            emotionLabel.textContent = 'Không tìm thấy camera!';
            showToast('⚠ Không tìm thấy thiết bị camera');
        } else {
            emotionLabel.textContent = 'Lỗi camera: ' + err.message;
        }
    }
}

// ── Main inference loop ──────────────────────────────────────────────────────

function loop() {
    if (!running) return;
    tCtx.drawImage(video, 0, 0);

    // FPS
    frameCount++;
    const now = performance.now();
    if (now - frameTs >= 1000) {
        fps = frameCount;
        frameCount = 0;
        frameTs = now;
        fpsDisplay.textContent = `${fps} fps`;
    }

    tempCanvas.toBlob(async (blob) => {
        const fd = new FormData();
        fd.append('file', blob, 'frame.jpg');

        try {
            const res  = await fetch('/predict', { method: 'POST', body: fd });
            const data = await res.json();
            updateUI(data.results);
        } catch (_) {
            // silent fail — server may be restarting
        }

        setTimeout(loop, 120); // ~8 fps inference
    }, 'image/jpeg', 0.75);
}

// ── UI Updates ───────────────────────────────────────────────────────────────

function updateUI(results) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!results || results.length === 0) {
        faceCount.textContent = '0 khuôn mặt';
        setNeutralPill();
        clearBars();
        return;
    }

    faceCount.textContent = `${results.length} khuôn mặt`;

    results.sort((a, b) => a.bbox[0] - b.bbox[0]);

    results.forEach((p, i) => {
        const [x1, y1, x2, y2] = p.bbox;
        const personLabel = `Người ${i + 1}`;
        const meta  = EMO_META[p.label] || EMO_META['Neutral'];
        const isHappy = p.label === 'Happy';

        // Box
        ctx.strokeStyle = isHappy ? '#22c55e' : '#5b7cf6';
        ctx.lineWidth   = 2.5;
        ctx.shadowColor = isHappy ? '#22c55e' : '#5b7cf6';
        ctx.shadowBlur  = 8;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.shadowBlur = 0;

        // Label tag
        const tagText  = `${personLabel} • ${meta.icon} ${p.label} ${p.score.toFixed(0)}%`;
        ctx.font       = 'bold 13px monospace';
        const tw       = ctx.measureText(tagText).width + 16;
        const tagY     = Math.max(y1 - 30, 0);

        ctx.fillStyle  = isHappy ? '#22c55e' : '#5b7cf6';
        roundRect(ctx, x1, tagY, tw, 24, 5);
        ctx.fill();

        ctx.fillStyle  = '#fff';
        ctx.fillText(tagText, x1 + 8, tagY + 16);
    });

    // Update pill from first face
    const primary = results[0];
    const meta    = EMO_META[primary.label] || EMO_META['Neutral'];

    currentData = {
        emotion:    primary.label,
        confidence: primary.score.toFixed(1),
        allScores:  primary.all_scores || {},
    };

    emotionIcon.textContent  = meta.icon;
    emotionLabel.textContent = primary.label.toUpperCase();
    emotionConf.textContent  = `${primary.score.toFixed(1)}%`;

    // Update pill color class
    Object.values(EMO_META).forEach(m => emotionPill.classList.remove(m.cls));
    emotionPill.classList.add(meta.cls);

    // Update bars
    updateBars(primary.all_scores || {}, primary.label, primary.score);
}

function setNeutralPill() {
    Object.values(EMO_META).forEach(m => emotionPill.classList.remove(m.cls));
    emotionIcon.textContent  = '😐';
    emotionLabel.textContent = 'Chờ nhận diện...';
    emotionConf.textContent  = '';
}

function clearBars() {
    document.querySelectorAll('.emo-bar-fill').forEach(b => b.style.width = '0%');
    document.querySelectorAll('.emo-bar-val').forEach(v => v.textContent = '0%');
}

function updateMoodChart() {
    renderMoodGroup('moodChartIn', moodCountsIn);
    renderMoodGroup('moodChartOut', moodCountsOut);
}

function renderMoodGroup(containerId, counts) {
    const donut = document.getElementById(containerId);
    const legendId = containerId === 'moodChartIn' ? 'moodLegendIn' : 'moodLegendOut';
    const legend = document.getElementById(legendId);
    const labels = ['Happy','Neutral','Sad','Angry','Surprise','Fear','Disgust'];
    const items  = labels.map(label => ({
        label,
        value: counts[label] || 0,
        color: MOOD_COLORS[label] || '#c8cde8',
    })).filter(item => item.value > 0);
    const total = items.reduce((sum, item) => sum + item.value, 0);

    if (total === 0) {
        donut.innerHTML = '<div class="chart-empty">Chưa có dữ liệu</div>';
        legend.innerHTML = '';
        return;
    }

    let currentDeg = 0;
    const segments = items.map(item => {
        const deg = Math.round((item.value / total) * 360);
        const from = currentDeg;
        const to   = currentDeg + deg;
        currentDeg = to;
        return `${item.color} ${from}deg ${to}deg`;
    });

    donut.innerHTML = `
        <div class="chart-donut" style="background: radial-gradient(circle at center, var(--panel) 58%, transparent 59%), conic-gradient(${segments.join(', ')});">
            <span>${total} lượt</span>
        </div>
    `;

    legend.innerHTML = items.map(item => `
        <div class="chart-legend-item">
            <span class="legend-color" style="background:${item.color}"></span>
            <span>${item.label}: ${item.value}</span>
        </div>
    `).join('');
}

function updateBars(allScores, topLabel, topScore) {
    document.querySelectorAll('.emo-bar-row').forEach(row => {
        const emo  = row.dataset.emo;
        const fill = row.querySelector('.emo-bar-fill');
        const val  = row.querySelector('.emo-bar-val');
        let score  = 0;

        if (allScores[emo] !== undefined) {
            score = allScores[emo];
        } else if (emo === topLabel) {
            score = topScore;
        }

        fill.style.width    = `${score.toFixed(1)}%`;
        val.textContent     = `${score.toFixed(0)}%`;
        val.style.color     = emo === topLabel ? '#fff' : '';
        fill.style.opacity  = emo === topLabel ? '1' : '0.55';
    });
}

function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
    ctx.closePath();
}

// ── Logging ──────────────────────────────────────────────────────────────────

function logEntry(action) {
    const now  = new Date().toLocaleTimeString('vi-VN');
    const emo  = currentData.emotion;
    const conf = currentData.confidence;

    const isIn  = action === 'CHECK-IN';
    const entry = { time: now, action, emotion: emo, confidence: conf, user: LOGGED_IN_USER };
    logs.push(entry);

    // Update stats
    if (isIn) {
        countIn++;
        statIn.textContent = countIn;
        moodCountsIn[emo] = (moodCountsIn[emo] || 0) + 1;
    } else {
        countOut++;
        statOut.textContent = countOut;
        moodCountsOut[emo] = (moodCountsOut[emo] || 0) + 1;
    }
    if (emo === 'Happy') { countHappy++; statHappy.textContent = countHappy; }

    logCount.textContent = `${logs.length} sự kiện`;
    updateMoodChart();

    // Prepend row
    emptyState.style.display = 'none';
    const tr = document.createElement('tr');
    tr.innerHTML = `
        <td style="font-family:monospace;font-size:12px;color:var(--muted)">${now}</td>
        <td>
            <span class="badge ${isIn ? 'badge-in' : 'badge-out'}">
                ${isIn ? '→ VÀO' : '← RA'}
            </span>
        </td>
        <td>
            <span class="badge-emo">${EMO_META[emo]?.icon || ''} ${emo} ${conf}%</span>
        </td>
    `;
    tr.style.animation = 'fadeIn 0.2s ease';
    logBody.prepend(tr);

    showToast(`${isIn ? '✓ Check-in' : '✓ Check-out'}: ${emo}`);
}

// ── Export ───────────────────────────────────────────────────────────────────

document.getElementById('btnExport').onclick = () => {
    if (logs.length === 0) { showToast('⚠ Chưa có dữ liệu!'); return; }

    const bom = '\uFEFF';
    const header = 'Nhân viên,Giờ,Hành động,Cảm xúc,Độ tin cậy\n';
    const rows   = logs.map(e => `${e.user},${e.time},${e.action},${e.emotion},${e.confidence}%`).join('\n');
    const blob   = new Blob([bom + header + rows], { type: 'text/csv;charset=utf-8;' });
    const url    = URL.createObjectURL(blob);
    const a      = document.createElement('a');
    a.href       = url;
    a.download   = `EmotionLog_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);

    // Reset
    logs = []; countIn = 0; countOut = 0; countHappy = 0;
    moodCountsIn = {}; moodCountsOut = {};
    statIn.textContent = statOut.textContent = statHappy.textContent = '0';
    logBody.innerHTML  = '';
    logCount.textContent = '0 sự kiện';
    emptyState.style.display = '';
    updateMoodChart();
    showToast('✓ Đã xuất & reset!');
};

// ── Buttons ──────────────────────────────────────────────────────────────────

document.getElementById('btnIn').onclick  = () => logEntry('CHECK-IN');
document.getElementById('btnOut').onclick = () => logEntry('CHECK-OUT');

// ── Toast ────────────────────────────────────────────────────────────────────

let toastTimer;
function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2200);
}

// ── Add CSS keyframe for row fade ────────────────────────────────────────────
const style = document.createElement('style');
style.textContent = '@keyframes fadeIn { from { opacity:0; transform:translateY(-6px) } to { opacity:1; transform:none } }';
document.head.appendChild(style);

// ── Start ─────────────────────────────────────────────────────────────────────
updateMoodChart();
startCamera();