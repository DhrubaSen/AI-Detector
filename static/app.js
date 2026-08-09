const LABELS = {
  ai_generated: "🤖 AI Generated",
  ai_assisted: "🔀 AI Assisted / Mixed",
  human_written: "✍️ Human Written",
  human_captured: "📷 Human Captured (Photograph)",
  human_created: "🖊️ Human Created (Screenshot/Diagram)",
  uncertain: "❓ Uncertain",
  unknown: "❓ Unknown",
  insufficient_text: "⚠️ Text Too Short",
  error: "❌ Error"
};

let docFile = null;
let imgFile = null;
let vidFile = null;

function switchTab(tab, btnEl) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  btnEl.classList.add('active');
  document.getElementById('panel-' + tab).classList.add('active');
}

function dragOver(e, id) {
  e.preventDefault();
  document.getElementById(id).classList.add('drag-over');
}
function dragLeave(id) {
  document.getElementById(id).classList.remove('drag-over');
}
function dropFile(e, type) {
  e.preventDefault();
  dragLeave(type + '-drop');
  const file = e.dataTransfer.files[0];
  if (file) setFile(file, type);
}
function fileSelected(e, type) {
  const file = e.target.files[0];
  if (file) setFile(file, type);
}
function setFile(file, type) {
  if (type === 'vid') {
    vidFile = file;
    document.getElementById('vid-file-name').textContent = file.name;
    document.getElementById('vid-file-size').textContent = formatSize(file.size);
    document.getElementById('vid-file-info').classList.add('show');
    document.getElementById('vid-btn').disabled = false;
  } else if (type === 'doc') {
    docFile = file;
    document.getElementById('doc-file-name').textContent = file.name;
    document.getElementById('doc-file-size').textContent = formatSize(file.size);
    document.getElementById('doc-file-info').classList.add('show');
    document.getElementById('doc-btn').disabled = false;
  } else {
    imgFile = file;
    document.getElementById('img-file-name').textContent = file.name;
    document.getElementById('img-file-size').textContent = formatSize(file.size);
    document.getElementById('img-file-info').classList.add('show');
    document.getElementById('img-btn').disabled = false;
  }
}
function formatSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

async function analyseText() {
  const text = document.getElementById('text-input').value.trim();
  if (!text) return;
  const btn = document.getElementById('text-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Analysing...';
  try {
    const res = await fetch('/api/check-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    renderResult(data, 'text-result');
  } catch(e) {
    document.getElementById('text-result').innerHTML = `<div class="error-box">Error: ${e.message}</div>`;
    document.getElementById('text-result').classList.add('show');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🔍 Analyse Text';
  }
}

async function analyseDocument() {
  if (!docFile) return;
  const btn = document.getElementById('doc-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Analysing...';
  const form = new FormData();
  form.append('file', docFile);
  try {
    const res = await fetch('/api/check-document', { method: 'POST', body: form });
    const data = await res.json();
    renderResult(data, 'doc-result');
  } catch(e) {
    document.getElementById('doc-result').innerHTML = `<div class="error-box">Error: ${e.message}</div>`;
    document.getElementById('doc-result').classList.add('show');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🔍 Analyse Document';
  }
}

async function analyseImage() {
  if (!imgFile) return;
  const btn = document.getElementById('img-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Analysing...';
  const form = new FormData();
  form.append('file', imgFile);
  try {
    const res = await fetch('/api/check-image', { method: 'POST', body: form });
    const data = await res.json();
    renderImageResult(data, 'img-result');
  } catch(e) {
    document.getElementById('img-result').innerHTML = `<div class="error-box">Error: ${e.message}</div>`;
    document.getElementById('img-result').classList.add('show');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🔍 Analyse Image';
  }
}

function renderResult(data, targetId) {
  const el = document.getElementById(targetId);
  const label = data.label || 'unknown';
  const conf = data.confidence || 0;
  const sig = data.signals || {};

  const verdictLabel = LABELS[label] || label;
  const confLabel = label === 'insufficient_text' ? '' : `${conf}% confidence`;

  let signalsHtml = '';
  if (Object.keys(sig).length > 0) {
    const sigItems = Object.entries(sig).map(([k, v]) => `
      <div class="signal-item">
        <div class="signal-name">${k.replace(/_/g,' ')}</div>
        <div class="signal-value">${typeof v === 'number' ? v.toFixed(3) : v}</div>
      </div>`).join('');
    signalsHtml = `
      <div class="card">
        <h2>Signal Details</h2>
        <div class="signals-grid">${sigItems}</div>
      </div>`;
  }

  let metaHtml = '';
  if (data.word_count) metaHtml += `${data.word_count} words · `;
  if (data.sentence_count) metaHtml += `${data.sentence_count} sentences · `;
  if (data.document_type) metaHtml += `${data.document_type} · `;
  if (data.filename) metaHtml += data.filename;

  // Store text for SciSpace link
  const textForSciSpace = document.getElementById('text-input') ? 
    encodeURIComponent(document.getElementById('text-input').value.trim().substring(0, 500)) : '';

  // Build signal contributions HTML
  let contribHtml = '';
  if (data.signal_contributions && Object.keys(data.signal_contributions).length > 0) {
    const contribs = Object.entries(data.signal_contributions)
      .filter(([k,v]) => parseFloat(v) > 0)
      .sort((a,b) => parseFloat(b[1]) - parseFloat(a[1]));
    if (contribs.length > 0) {
      const rows = contribs.map(([k,v]) => {
        const pct = parseFloat(v);
        const desc = v.split('% — ')[1] || '';
        return `<div style="margin-bottom:6px;">
          <div style="display:flex;justify-content:space-between;font-size:12px;color:#94D2BD;margin-bottom:2px;">
            <span>${desc}</span><span>${pct.toFixed(1)}%</span>
          </div>
          <div style="height:4px;background:#1e3a5f;border-radius:2px;">
            <div style="height:4px;background:#EE9B00;border-radius:2px;width:${Math.min(100,pct*3)}%;"></div>
          </div>
        </div>`;
      }).join('');
      contribHtml = `<div class="card"><h2>What Drove the AI Score</h2>${rows}
        <p style="font-size:11px;color:#64748b;margin-top:8px;">Each bar shows how much that signal contributed to the AI probability score.</p>
      </div>`;
    }
  }

  el.innerHTML = `
    <div class="card">
      ${metaHtml ? `<div class="meta">${metaHtml}</div>` : ''}
      <div class="verdict ${label}">
        <div class="verdict-icon">${verdictLabel.split(' ')[0]}</div>
        <div>
          <div class="verdict-label">${verdictLabel.substring(verdictLabel.indexOf(' ')+1)}</div>
          <div class="verdict-conf">${confLabel}</div>
        </div>
      </div>
      <div class="confidence-bar ${label}">
        <div class="confidence-fill" style="width:${conf}%"></div>
      </div>
      <div class="explanation">${data.explanation || ''}</div>
    </div>
    ${signalsHtml}
    <div class="card" style="padding:16px;">
      <div id="second-opinion-header" style="font-size:14px;font-weight:600;color:#94D2BD;margin-bottom:10px;cursor:pointer;">
        💡 How to Get a Second Opinion <span id="second-opinion-toggle" style="font-size:11px;color:#64748b;">▼ show</span>
      </div>
      <div id="second-opinion-panel" style="display:none;">
        <p style="font-size:13px;color:#e2e8f0;margin-bottom:10px;">
          For an independent second opinion, copy your upload and paste it into one of these tools directly.
        </p>
        <div style="font-size:13px;color:#94D2BD;line-height:1.9;">
          <strong>For text/documents:</strong> For example, <a href="https://scispace.com/ai-detector" target="_blank" style="color:#0A9396;">SciSpace</a> is one option for technical and scientific content. Other detectors are available — search "AI content detector" to find options that suit your needs.<br><br>
          <strong>For video:</strong> For example, <a href="https://tsdetect.com/ai-video-detector.html" target="_blank" style="color:#0A9396;">TrueSight</a> performs actual frame-level analysis and supports YouTube URLs directly. Other video detectors are available — search "AI video detector" to find options that suit your needs.<br>
        </div>

      </div>
    </div>`;
  el.classList.add('show');
  const secondOpinionHeader = document.getElementById('second-opinion-header');
  if (secondOpinionHeader) {
    secondOpinionHeader.addEventListener('click', toggleSecondOpinion);
  }
}

function renderImageResult(data, targetId) {
  const el = document.getElementById(targetId);
  const label = data.label || 'unknown';
  const conf = data.confidence || 0;
  const verdictLabel = LABELS[label] || label;

  let indicatorsHtml = '';
  if (data.ai_indicators && data.ai_indicators.length > 0) {
    const items = data.ai_indicators.map(i => `
      <div class="indicator-item">
        <div class="indicator-dot ai"></div>
        <span>${i}</span>
      </div>`).join('');
    indicatorsHtml += `<div class="indicator-group">
      <div class="indicator-title ai">🤖 AI Indicators</div>${items}</div>`;
  }
  if (data.human_indicators && data.human_indicators.length > 0) {
    const items = data.human_indicators.map(i => `
      <div class="indicator-item">
        <div class="indicator-dot human"></div>
        <span>${i}</span>
      </div>`).join('');
    indicatorsHtml += `<div class="indicator-group">
      <div class="indicator-title human">📷 Human/Photo Indicators</div>${items}</div>`;
  }

  const sig = data.signals || {};
  let sigHtml = '';
  if (sig.dimensions) sigHtml += `<div class="signal-item"><div class="signal-name">Dimensions</div><div class="signal-value" style="font-size:14px">${sig.dimensions}</div></div>`;
  // Watermark info
  const wm = sig.watermark_check || {};
  if (wm.watermark_detected) {
    sigHtml += `<div class="signal-item" style="border:1px solid rgba(238,155,0,0.4);background:rgba(238,155,0,0.05);">
      <div class="signal-name" style="color:#EE9B00;">⚠️ Watermark</div>
      <div class="signal-value" style="font-size:12px;color:#EE9B00;">Detected</div>
    </div>`;
  }
  if (sig.format) sigHtml += `<div class="signal-item"><div class="signal-name">Format</div><div class="signal-value" style="font-size:14px">${sig.format}</div></div>`;
  if (sig.avg_channel_std) sigHtml += `<div class="signal-item"><div class="signal-name">Color Variance</div><div class="signal-value">${sig.avg_channel_std}</div></div>`;
  if (sig.has_camera_exif !== undefined) sigHtml += `<div class="signal-item"><div class="signal-name">Camera EXIF</div><div class="signal-value" style="font-size:14px">${sig.has_camera_exif ? '✅ Yes' : '❌ No'}</div></div>`;

  // C2PA compliance block
  let complianceHtml = '';
  const c2pa = data.c2pa || {};
  const eu = data.eu_compliance || {};

  if (eu.status === 'NON_COMPLIANT') {
    complianceHtml = `
      <div class="card" style="border-color:rgba(220,38,38,0.4);background:rgba(220,38,38,0.05);">
        <h2 style="color:#f87171;">⚠️ EU AI Act Article 50 — Non-Compliant</h2>
        <p style="font-size:14px;color:#f87171;margin-bottom:8px;">${eu.issue}</p>
        <p style="font-size:13px;color:#64748b;">Enforcement deadline: <strong style="color:#EE9B00;">${eu.deadline}</strong></p>
        <p style="font-size:13px;color:#64748b;margin-top:6px;">Action required: ${eu.action_required}</p>
        <div style="margin-top:12px;padding:10px;background:#0a2c1a;border-radius:6px;">
          <p style="font-size:12px;color:#94D2BD;">C2PA (Coalition for Content Provenance and Authenticity) metadata: <strong>NOT FOUND</strong></p>
          <p style="font-size:12px;color:#64748b;margin-top:4px;">AI-generated content must embed machine-readable provenance data per EU AI Act Art.50 and California SB 942.</p>
        </div>
      </div>`;
  } else if (eu.status === 'COMPLIANT') {
    complianceHtml = `
      <div class="card" style="border-color:rgba(16,185,129,0.4);background:rgba(16,185,129,0.05);">
        <h2 style="color:#10b981;">✅ EU AI Act Article 50 — Compliant</h2>
        <p style="font-size:14px;color:#94D2BD;">${eu.note}</p>
        ${c2pa.creator_tool ? `<p style="font-size:13px;color:#64748b;margin-top:6px;">Creator tool: ${c2pa.creator_tool}</p>` : ''}
        ${c2pa.ai_assertions && c2pa.ai_assertions.length > 0 ? `<p style="font-size:13px;color:#64748b;">AI assertions: ${c2pa.ai_assertions.join(', ')}</p>` : ''}
      </div>`;
  } else if (eu.status === 'NOT_APPLICABLE') {
    complianceHtml = `
      <div class="card" style="border-color:rgba(100,116,139,0.3);">
        <h2 style="color:#64748b;">EU AI Act Article 50 — Not Applicable</h2>
        <p style="font-size:13px;color:#64748b;">${eu.note}</p>
        <p style="font-size:12px;color:#64748b;margin-top:4px;">C2PA watermark check: ${c2pa.has_c2pa ? '✅ Present' : '❌ Not found'}</p>
      </div>`;
  }

  el.innerHTML = `
    <div class="card">
      <div class="meta">${data.filename || ''}</div>
      <div class="verdict ${label}">
        <div class="verdict-icon">${verdictLabel.split(' ')[0]}</div>
        <div>
          <div class="verdict-label">${verdictLabel.substring(verdictLabel.indexOf(' ')+1)}</div>
          <div class="verdict-conf">${conf}% confidence</div>
        </div>
      </div>
      <div class="confidence-bar ${label}">
        <div class="confidence-fill" style="width:${conf}%"></div>
      </div>
      <div class="explanation">${data.explanation || ''}</div>
      ${indicatorsHtml ? `<div class="indicators">${indicatorsHtml}</div>` : ''}
      <div style="font-size:11px;color:#64748b;margin-top:8px;padding:8px;background:#0f1117;border-radius:6px;">
        ℹ️ Whole-image analysis only — cannot determine which elements within the image are AI vs human. Camera metadata reliability varies by how a file was shared: WhatsApp and similar apps strip metadata when a photo/video is sent normally, but preserve it completely when sent as a Document/File attachment — so metadata presence or absence isn't fully reliable evidence on its own.
      </div>
    </div>
    ${complianceHtml}
    ${sigHtml ? `<div class="card"><h2>Image Signals</h2><div class="signals-grid">${sigHtml}</div></div>` : ''}`;
  el.classList.add('show');
}

function switchVideoTab(tab) {
  document.getElementById('vid-upload-panel').style.display = tab === 'upload' ? 'block' : 'none';
  document.getElementById('vid-url-panel').style.display = tab === 'url' ? 'block' : 'none';
  document.getElementById('vid-tab-upload').classList.toggle('active', tab === 'upload');
  document.getElementById('vid-tab-url').classList.toggle('active', tab === 'url');
}

async function analyseVideoUrl() {
  const url = document.getElementById('vid-url-input').value.trim();
  if (!url) return;
  const btn = document.getElementById('vid-url-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Analysing...';
  try {
    const res = await fetch('/api/check-video-url', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    });
    const data = await res.json();
    renderVideoResult(data, 'vid-result');
  } catch(e) {
    document.getElementById('vid-result').innerHTML = `<div class="error-box">Error: ${e.message}</div>`;
    document.getElementById('vid-result').classList.add('show');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🔍 Analyse URL';
  }
}

function toggleSecondOpinion() {
  const panel = document.getElementById('second-opinion-panel');
  const toggle = document.getElementById('second-opinion-toggle');
  if (panel.style.display === 'none') {
    panel.style.display = 'block';
    toggle.textContent = '▲ hide';
  } else {
    panel.style.display = 'none';
    toggle.textContent = '▼ show';
  }
}

async function analyseVideo() {
  if (!vidFile) return;
  const btn = document.getElementById('vid-btn');
  btn.disabled = true;
  btn.classList.add('loading');
  btn.textContent = 'Analysing...';
  const form = new FormData();
  form.append('file', vidFile);
  try {
    const res = await fetch('/api/check-video', { method: 'POST', body: form });
    const data = await res.json();
    renderVideoResult(data, 'vid-result');
  } catch(e) {
    document.getElementById('vid-result').innerHTML = `<div class="error-box">Error: ${e.message}</div>`;
    document.getElementById('vid-result').classList.add('show');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
    btn.textContent = '🔍 Analyse Video';
  }
}

function renderVideoResult(data, targetId) {
  const el = document.getElementById(targetId);
  const label = data.label || 'unknown';
  const conf = data.confidence || 0;
  const verdictLabel = LABELS[label] || label;

  let indicatorsHtml = '';
  if (data.ai_indicators && data.ai_indicators.length > 0) {
    const items = data.ai_indicators.map(i => `
      <div class="indicator-item">
        <div class="indicator-dot ai"></div>
        <span>${i}</span>
      </div>`).join('');
    indicatorsHtml += `<div class="indicator-group"><div class="indicator-title ai">🤖 AI Indicators</div>${items}</div>`;
  }
  if (data.human_indicators && data.human_indicators.length > 0) {
    const items = data.human_indicators.map(i => `
      <div class="indicator-item">
        <div class="indicator-dot human"></div>
        <span>${i}</span>
      </div>`).join('');
    indicatorsHtml += `<div class="indicator-group"><div class="indicator-title human">📷 Human/Camera Indicators</div>${items}</div>`;
  }

  const sig = data.signals || {};
  let sigHtml = '';
  if (sig.format) sigHtml += `<div class="signal-item"><div class="signal-name">Format</div><div class="signal-value" style="font-size:14px">${sig.format}</div></div>`;
  if (sig.file_size_mb) sigHtml += `<div class="signal-item"><div class="signal-name">File Size</div><div class="signal-value" style="font-size:14px">${sig.file_size_mb}MB</div></div>`;
  if (sig.ai_tools_found) sigHtml += `<div class="signal-item"><div class="signal-name">AI Tools Found</div><div class="signal-value" style="font-size:12px">${sig.ai_tools_found.join(', ')}</div></div>`;

  el.innerHTML = `
    <div class="card">
      <div class="meta">${data.filename || ''}</div>
      <div class="verdict ${label}">
        <div class="verdict-icon">${verdictLabel.split(' ')[0]}</div>
        <div>
          <div class="verdict-label">${verdictLabel.substring(verdictLabel.indexOf(' ')+1)}</div>
          <div class="verdict-conf">${conf}% confidence</div>
        </div>
      </div>
      <div class="confidence-bar ${label}">
        <div class="confidence-fill" style="width:${conf}%"></div>
      </div>
      <div class="explanation">${data.explanation || ''}</div>
      ${indicatorsHtml ? `<div class="indicators">${indicatorsHtml}</div>` : ''}
      ${data.disclaimer ? `<div style="margin-top:12px;padding:10px 14px;background:#1e3a5f;border-radius:8px;font-size:12px;color:#94D2BD;">${data.disclaimer}</div>` : ''}
    </div>
    ${sigHtml ? `<div class="card"><h2>Video Signals</h2><div class="signals-grid">${sigHtml}</div></div>` : ''}`;
  el.classList.add('show');
}

// ── Event wiring (was inline onclick/onchange/ondrop/etc attributes) ────────
// Moved out of index.html so the CSP script-src directive can drop
// 'unsafe-inline' without breaking every button/drop-zone in the app.
document.addEventListener('DOMContentLoaded', () => {
  // Main tabs
  const mainTabs = [
    ['tab-text', 'text'],
    ['tab-document', 'document'],
    ['tab-image', 'image'],
    ['tab-video', 'video'],
  ];
  mainTabs.forEach(([id, tab]) => {
    const el = document.getElementById(id);
    if (el) el.addEventListener('click', () => switchTab(tab, el));
  });

  // Video sub-tabs
  const vidTabUpload = document.getElementById('vid-tab-upload');
  const vidTabUrl = document.getElementById('vid-tab-url');
  if (vidTabUpload) vidTabUpload.addEventListener('click', () => switchVideoTab('upload'));
  if (vidTabUrl) vidTabUrl.addEventListener('click', () => switchVideoTab('url'));

  // Analyse buttons
  const textBtn = document.getElementById('text-btn');
  if (textBtn) textBtn.addEventListener('click', analyseText);
  const docBtn = document.getElementById('doc-btn');
  if (docBtn) docBtn.addEventListener('click', analyseDocument);
  const imgBtn = document.getElementById('img-btn');
  if (imgBtn) imgBtn.addEventListener('click', analyseImage);
  const vidBtn = document.getElementById('vid-btn');
  if (vidBtn) vidBtn.addEventListener('click', analyseVideo);
  const vidUrlBtn = document.getElementById('vid-url-btn');
  if (vidUrlBtn) vidUrlBtn.addEventListener('click', analyseVideoUrl);

  // Drop zones + file inputs (doc, img, vid)
  const dropTypes = [
    ['doc-drop', 'doc-input', 'doc'],
    ['img-drop', 'img-input', 'img'],
    ['vid-drop', 'vid-input', 'vid'],
  ];
  dropTypes.forEach(([dropId, inputId, type]) => {
    const dropEl = document.getElementById(dropId);
    const inputEl = document.getElementById(inputId);
    if (dropEl && inputEl) {
      dropEl.addEventListener('click', () => inputEl.click());
      dropEl.addEventListener('dragover', (e) => dragOver(e, dropId));
      dropEl.addEventListener('dragleave', () => dragLeave(dropId));
      dropEl.addEventListener('drop', (e) => dropFile(e, type));
    }
    if (inputEl) {
      inputEl.addEventListener('change', (e) => fileSelected(e, type));
    }
  });
});
