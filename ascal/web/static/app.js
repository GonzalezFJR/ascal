/* ascal web app.
 *
 * Form -> POST api/jobs -> poll api/jobs/<id> -> api/jobs/<id>/result.json + display.jpg -> layered viewer,
 * tables and ECharts diagnostics. All URLs are relative, so the page works under any path prefix.
 *
 * CameraModel replicates ascal/model.py (rigid rotation + Kannala-Brandt + image rotation + optional decentering +
 * mirror) to read (alt, az) under the cursor without asking the server. Keep it in sync with model.py.
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const DEG = Math.PI / 180;
  const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(+v) ? "—" : Number(v).toFixed(d));
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  const CONST_NAMES = {And:"Andromeda",Ant:"Antlia",Aps:"Apus",Aqr:"Aquarius",Aql:"Aquila",Ara:"Ara",Ari:"Aries",Aur:"Auriga",Boo:"Boötes",
    Cae:"Caelum",Cam:"Camelopardalis",Cnc:"Cancer",CVn:"Canes Venatici",CMa:"Canis Major",CMi:"Canis Minor",Cap:"Capricornus",Car:"Carina",
    Cas:"Cassiopeia",Cen:"Centaurus",Cep:"Cepheus",Cet:"Cetus",Cha:"Chamaeleon",Cir:"Circinus",Col:"Columba",Com:"Coma Berenices",
    CrA:"Corona Australis",CrB:"Corona Borealis",Crv:"Corvus",Crt:"Crater",Cru:"Crux",Cyg:"Cygnus",Del:"Delphinus",Dor:"Dorado",Dra:"Draco",
    Equ:"Equuleus",Eri:"Eridanus",For:"Fornax",Gem:"Gemini",Gru:"Grus",Her:"Hercules",Hor:"Horologium",Hya:"Hydra",Hyi:"Hydrus",Ind:"Indus",
    Lac:"Lacerta",Leo:"Leo",LMi:"Leo Minor",Lep:"Lepus",Lib:"Libra",Lup:"Lupus",Lyn:"Lynx",Lyr:"Lyra",Men:"Mensa",Mic:"Microscopium",
    Mon:"Monoceros",Mus:"Musca",Nor:"Norma",Oct:"Octans",Oph:"Ophiuchus",Ori:"Orion",Pav:"Pavo",Peg:"Pegasus",Per:"Perseus",Phe:"Phoenix",
    Pic:"Pictor",Psc:"Pisces",PsA:"Piscis Austrinus",Pup:"Puppis",Pyx:"Pyxis",Ret:"Reticulum",Sge:"Sagitta",Sgr:"Sagittarius",Sco:"Scorpius",
    Scl:"Sculptor",Sct:"Scutum",Ser:"Serpens",Ser1:"Serpens",Ser2:"Serpens",Sex:"Sextans",Tau:"Taurus",Tel:"Telescopium",Tri:"Triangulum",
    TrA:"Triangulum Australe",Tuc:"Tucana",UMa:"Ursa Major",UMi:"Ursa Minor",Vel:"Vela",Vir:"Virgo",Vol:"Volans",Vul:"Vulpecula"};

  // ---------------------------------------------------------------- camera model (mirror of ascal/model.py)
  class CameraModel {
    constructor(c) {
      Object.assign(this, { cx: c.cx, cy: c.cy, f: c.f, psi: c.psi, width: c.width, height: c.height });
      this.k = (c.k || []).slice();
      this.p = c.p && c.p.length === 2 ? c.p.slice() : null;
      this.mirror = !!c.mirror;
      const ax = (c.tau_x || 0) * DEG, ay = (c.tau_y || 0) * DEG;
      const cx = Math.cos(ax), sx = Math.sin(ax), cy = Math.cos(ay), sy = Math.sin(ay);
      this.R = [[cy, -sy * sx, sy * cx], [0, cx, sx], [-sy, -cy * sx, cy * cx]];
    }
    static vec(alt, az) { const a = alt * DEG, z = az * DEG; return [Math.cos(a) * Math.sin(z), Math.cos(a) * Math.cos(z), Math.sin(a)]; }
    static altaz(v) { let az = Math.atan2(v[0], v[1]) / DEG; if (az < 0) az += 360; return [Math.asin(Math.max(-1, Math.min(1, v[2]))) / DEG, az]; }
    toCam(alt, az) { const v = CameraModel.vec(alt, az), R = this.R; return CameraModel.altaz([0, 1, 2].map((i) => R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2])); }
    toSky(alt, az) { const v = CameraModel.vec(alt, az), R = this.R; return CameraModel.altaz([0, 1, 2].map((i) => R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2])); }
    radius(th) { let o = th; this.k.forEach((k, i) => { o += k * Math.pow(th, 2 * i + 3); }); return this.f * o; }
    dradius(th) { let o = 1; this.k.forEach((k, i) => { o += (2 * i + 3) * k * Math.pow(th, 2 * i + 2); }); return this.f * o; }
    theta(r) {
      let th = Math.max(0, Math.min(Math.PI, r / this.f));
      for (let i = 0; i < 30; i++) { const d = this.dradius(th); th = Math.max(0, Math.min(Math.PI, th - (this.radius(th) - r) / (Math.abs(d) > 1e-9 ? d : 1e-9))); }
      return th;
    }
    dec(u, v) {
      const [p1, p2] = this.p, un = u / this.f, vn = v / this.f, r2 = un * un + vn * vn;
      return [(p1 * (r2 + 2 * un * un) + 2 * p2 * un * vn) * this.f, (2 * p1 * un * vn + p2 * (r2 + 2 * vn * vn)) * this.f];
    }
    unproject(x, y) {
      if (this.mirror) x = this.width - 1 - x;
      let u = x - this.cx, v = y - this.cy;
      if (this.p) { const u0 = u, v0 = v; for (let i = 0; i < 10; i++) { const [du, dv] = this.dec(u, v); u = u0 - du; v = v0 - dv; } }
      const altC = 90 - this.theta(Math.hypot(u, v)) / DEG;
      let azC = (this.psi - Math.atan2(u, -v) / DEG) % 360; if (azC < 0) azC += 360;
      return [this.toSky(altC, azC), 90 - altC];
    }
  }

  // ---------------------------------------------------------------- state
  const S = { cfg: null, file: null, example: null, job: null, poll: null, R: null, model: null, img: null,
              W: 0, H: 0, scale: 1, ox: 0, oy: 0, fit: 1, focus: null, charts: [], pointers: new Map(), pinch: null };
  const LAYERS = [
    { id: "image", label: "Image", color: "#666", on: true },
    { id: "pairs", label: "Matched stars", color: "#4da3ff", on: true },
    { id: "vectors", label: "Residual arrows", color: "#ff6bd6", on: false },
    { id: "detections", label: "All detections", color: "#22d3ee", on: false },
    { id: "catalog", label: "Catalogue stars", color: "#4ade80", on: false },
    { id: "names", label: "Star names", color: "#ffe08a", on: true },
    { id: "grid", label: "Alt/az grid", color: "#9fb4c8", on: true },
    { id: "cardinal", label: "Cardinal points", color: "#f4a948", on: true },
    { id: "constellations", label: "Constellations", color: "#ff9f1c", on: true },
    { id: "constnames", label: "Constellation names", color: "#ffc477", on: false },
    { id: "markers", label: "Zenith and optical centre", color: "#ff5c5c", on: false },
  ];
  const ON = Object.fromEntries(LAYERS.map((l) => [l.id, l.on]));

  // ---------------------------------------------------------------- form
  const status = (t, err) => { const el = $("status"); el.textContent = t || ""; el.className = "status" + (err ? " error" : ""); };

  async function init() {
    try {
      S.cfg = await (await fetch("api/config")).json();
      $("version").textContent = "v" + S.cfg.version;
      $("ttl").textContent = Math.round(S.cfg.ttl_h);
      $("max-time").max = S.cfg.max_time;
      $("formats").textContent = "JPEG, PNG, TIFF, FITS, camera raw · up to " + Math.round(S.cfg.max_mb) + " MB";
      $("elev").min = S.cfg.elev_range[0]; $("elev").max = S.cfg.elev_range[1];
      buildExamples(S.cfg.examples || []);
    } catch (_) { status("Cannot reach the server.", true); }
    const m = location.hash.match(/job=([A-Za-z0-9_-]+)/), x = location.hash.match(/example=([A-Za-z0-9_-]+)/);
    if (m) { showProgress(); follow(m[1]); }
    else if (x) { const e = (S.cfg?.examples || []).find((q) => q.id === x[1]); if (e) openExample(e); }
  }

  // ---------------------------------------------------------------- precomputed examples
  function buildExamples(list) {
    if (!list.length) return;
    $("n-examples").textContent = list.length;
    $("examples-toggle").hidden = false;
    $("example-cards").innerHTML = list.map((e) => `<button type="button" class="ex-card" data-id="${esc(e.id)}">
        <img src="api/examples/${encodeURIComponent(e.id)}/thumb.jpg" alt="" loading="lazy" width="360" height="360">
        <span class="ex-body"><strong>${esc(e.title)}</strong><span>${esc(e.place)}</span>
        <span>${esc(e.camera)} · ${esc(e.format)}</span><span>${esc(e.operator)}</span></span></button>`).join("");
    $("example-cards").querySelectorAll(".ex-card").forEach((b) => b.addEventListener("click", () => openExample(list.find((e) => e.id === b.dataset.id))));
  }
  $("examples-toggle").addEventListener("click", () => {
    const open = $("examples").hidden;
    $("examples").hidden = !open; $("examples-toggle").setAttribute("aria-expanded", String(open));
    if (open) $("examples").scrollIntoView({ behavior: "smooth", block: "nearest" });
  });

  async function openExample(e) {
    showProgress();
    $("progress-title").textContent = "Loading…"; $("progress-sub").textContent = `${e.title} · precomputed example`;
    $("log").hidden = true;
    history.replaceState(null, "", "#example=" + e.id);
    const base = `api/examples/${encodeURIComponent(e.id)}`;
    const wait = new Promise((r) => setTimeout(r, 2200));
    let log = "";
    try { log = await (await fetch(`${base}/log.txt`)).text(); } catch (_) { /* optional */ }
    await loadResult(base, { log: log.split("\n").filter(Boolean), example: e }, wait);
  }

  function setFile(f) {
    S.file = f;
    $("drop").classList.add("set");
    $("drop-title").textContent = f.name;
    $("drop-sub").textContent = (f.size / 1e6).toFixed(1) + " MB · click to change";
    const fd = new FormData(); fd.append("image", f);
    $("time-hint").className = "hint"; $("time-hint").textContent = "Reading the time stamp of the file…";
    fetch("api/exif", { method: "POST", body: fd }).then((r) => r.json()).then((x) => {
      if (x.exposure_s && !$("exposure").value) $("exposure").value = x.exposure_s;
      if (x.datetime) {
        const iso = x.datetime.slice(0, 19);
        const fits = (x.source || "").startsWith("FITS");
        if (fits || !$("utc").value) $("utc").value = iso.replace("T", " ");
        $("time-hint").className = "hint found";
        $("time-hint").textContent = fits ? `Time taken from the FITS header (${iso.replace("T", " ")} UTC).`
          : `Found ${iso.replace("T", " ")} in the ${x.source}. Check it is UTC and the start of the exposure.`;
      } else {
        $("time-hint").textContent = "No time stamp in the file: type the UTC start of the exposure.";
      }
    }).catch(() => { $("time-hint").textContent = ""; });
  }

  const drop = $("drop"), fileIn = $("file");
  drop.addEventListener("click", () => fileIn.click());
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileIn.click(); } });
  fileIn.addEventListener("change", () => fileIn.files[0] && setFile(fileIn.files[0]));
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, () => drop.classList.remove("over")));
  drop.addEventListener("drop", (e) => { e.preventDefault(); if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]); });
  $("geo").addEventListener("click", () => {
    if (!navigator.geolocation) return status("Geolocation is not available in this browser.", true);
    navigator.geolocation.getCurrentPosition((p) => {
      $("lat").value = p.coords.latitude.toFixed(5); $("lon").value = p.coords.longitude.toFixed(5);
      if (p.coords.altitude) $("elev").value = Math.round(p.coords.altitude);
    }, () => status("Could not read your location.", true));
  });

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    if (!S.file) return status("Choose an image first.", true);
    for (const id of ["utc", "lat", "lon"]) if (!$(id).value) { $(id).focus(); return status("Fill in the UTC time, latitude and longitude.", true); }
    const tm = $("utc").value.trim().replace("Z", "").match(/^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2})(:\d{2}(\.\d+)?)?$/);
    if (!tm || Number.isNaN(Date.parse(`${tm[1]}T${tm[2]}Z`))) { $("utc").focus(); return status("Write the UTC time as YYYY-MM-DD hh:mm:ss.", true); }
    const el = +($("elev").value || 0), [e0, e1] = S.cfg?.elev_range || [-450, 6000];
    if (!(el >= e0 && el <= e1)) { $("elev").focus(); return status(`The elevation must be between ${e0} and ${e1} m.`, true); }
    const lat = +$("lat").value, lon = +$("lon").value;
    if (!(lat >= -90 && lat <= 90) || !(lon >= -180 && lon <= 360)) return status("Latitude must be within ±90° and longitude within ±180°.", true);
    const kw = ($("fwhm").value || "auto").trim().toLowerCase(), [k0, k1] = S.cfg?.fwhm_range || [1.5, 15];
    if (kw !== "auto" && !(+kw >= k0 && +kw <= k1)) { $("fwhm").focus(); return status(`The detection kernel must be "auto" or between ${k0} and ${k1} px.`, true); }
    const mt = +($("max-time").value || 40), mtx = S.cfg?.max_time || 180;
    if (!(mt >= 10 && mt <= mtx)) { $("max-time").focus(); return status(`The search time budget must be between 10 and ${mtx} s.`, true); }
    const fd = new FormData();
    fd.append("image", S.file);
    fd.append("utc", `${tm[1]}T${tm[2]}${tm[3] || ":00"}`);
    fd.append("lat", $("lat").value); fd.append("lon", $("lon").value); fd.append("elev", $("elev").value || 0);
    if ($("exposure").value) fd.append("exposure", $("exposure").value);
    fd.append("max_time", $("max-time").value || 40); fd.append("parity", $("parity").value); fd.append("fwhm", kw);
    $("submit").disabled = true; status("Uploading…");
    try {
      const r = await fetch("api/jobs", { method: "POST", body: fd });
      const j = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(j.detail ? (typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail)) : "HTTP " + r.status);
      status(""); showProgress(); history.replaceState(null, "", "#job=" + j.id); follow(j.id);
    } catch (err) { status(err.message, true); } finally { $("submit").disabled = false; }
  });

  // ---------------------------------------------------------------- progress
  function showProgress() {
    $("progress").hidden = false; $("spinner").className = "spinner"; $("new-1").hidden = true;
    $("progress-title").textContent = "Waiting…"; $("progress-sub").textContent = ""; $("log").textContent = "";
    $("progress").querySelector(".error-box")?.remove(); $("log").hidden = false;
    $("form-card").hidden = true; $("viewer-wrap").hidden = true; $("results").hidden = true;
    window.scrollTo({ top: 0 });
  }

  function follow(id) {
    S.job = id; clearTimeout(S.poll);
    const tick = async () => {
      let s;
      try {
        const r = await fetch(`api/jobs/${id}`);
        if (r.status === 404) return failed({ message: "This result has expired or does not exist." });
        s = await r.json();
      } catch (_) { S.poll = setTimeout(tick, 2500); return; }
      $("log").textContent = (s.log || []).join("\n"); $("log").scrollTop = 1e9;
      if (s.state === "queued") { $("progress-title").textContent = `In the queue (position ${s.position})`; $("progress-sub").textContent = s.name; }
      if (s.state === "running") { $("progress-title").textContent = `Calibrating ${s.name}`; $("progress-sub").textContent = `${fmt(s.elapsed_s, 0)} s`; }
      if (s.state === "done") return loadResult(`api/jobs/${id}`, s);
      if (s.state === "error") return failed(s.error || {}, s);
      S.poll = setTimeout(tick, 1000);
    };
    tick();
  }

  function failed(err, s) {
    $("spinner").className = "spinner stop"; $("new-1").hidden = false;
    $("progress-title").textContent = err.kind === "calibration" ? "The image could not be calibrated" : "Error";
    $("progress-sub").textContent = s ? s.name : "";
    const box = document.createElement("div"); box.className = "error-box";
    box.innerHTML = `<b>${esc(err.message || "unknown error")}</b>` + (err.kind === "calibration" ? `<ul>
      <li>Check the UTC time (not local time) and the coordinates (longitude east positive).</li>
      <li>The frame must be a clear night image of most of the sky, with stars down to magnitude 4–5.</li>
      <li>For mirrored images try the parity option; for blurred or bloated stars a wider detection kernel (e.g. 6 px).</li>
      <li>Very large or star-poor images may need a longer time budget.</li></ul>` : "");
    $("progress").appendChild(box);
  }

  const newCalibration = () => {
    clearTimeout(S.poll); history.replaceState(null, "", location.pathname + location.search);
    $("form-card").hidden = false; $("progress").hidden = true; $("viewer-wrap").hidden = true; $("results").hidden = true;
    S.charts.forEach((c) => c.dispose()); S.charts = [];
  };
  $("new-1").addEventListener("click", newCalibration);
  $("new-2").addEventListener("click", newCalibration);
  $("share").addEventListener("click", () => {
    navigator.clipboard?.writeText(location.href).then(() => { $("share").textContent = "Link copied"; setTimeout(() => ($("share").textContent = "Copy link"), 1800); });
  });

  // ---------------------------------------------------------------- result
  async function loadResult(base, s, wait) {
    if (!s.example) $("progress-title").textContent = "Loading the result…";
    const R = await (await fetch(`${base}/result.json`)).json();
    if (wait) await wait;
    S.R = R; S.model = new CameraModel(R.model); S.W = R.frame.width; S.H = R.frame.height;
    const img = new Image();
    img.onload = () => {
      S.img = img;
      $("progress").hidden = true; $("viewer-wrap").hidden = false; $("results").hidden = false;
      $("result-title").textContent = R.original_name;
      const sm = R.summary;
      const e = s.example;
      $("result-title").textContent = e ? `${e.title} — ${e.place}` : R.original_name;
      $("result-sub").innerHTML = esc(`${R.frame.utc_mid.replace("T", " ").slice(0, 19)} UTC (mid-exposure) · ${fmt(R.site.lat, 4)}°, ${fmt(R.site.lon, 4)}° · `
        + `${sm.n_pairs} stars, median ${fmt(sm.median_px)} px · `) + (e ? "precomputed example" : esc(`result kept for ${fmt(s.expires_in_h, 1)} h`));
      $("ex-meta").hidden = !e;
      if (e) $("ex-meta").innerHTML = `<b>${esc(e.operator)}</b> · ${esc(e.site)} · ${esc(e.camera)}, ${esc(e.lens)} · ${esc(e.format)}, `
        + `${esc(String(e.exposure))} s · ${esc(e.description)} Data: <a href="${esc(e.link)}" target="_blank" rel="noopener">${esc(e.source)}</a> (${esc(e.credit)}).`;
      $("log-final").textContent = (s.log || []).join("\n");
      $("exp-cal").href = `${base}/calibration.json`;
      $("exp-csv").href = `${base}/pairs.csv`;
      $("exp-panel").href = `${base}/panel.png`;
      buildLayers(); resize(); fitView(); renderTables(); renderCharts();
    };
    img.src = `${base}/display.jpg`;
  }

  // ---------------------------------------------------------------- viewer
  const canvas = $("canvas"), stage = $("stage");
  let raf = 0;
  const redraw = () => { if (!raf) raf = requestAnimationFrame(() => { raf = 0; draw(); }); };

  function buildLayers() {
    const n = { pairs: S.R.layers.pairs.filter((p) => p.inlier).length, detections: S.R.frame.n_detections, catalog: S.R.layers.catalog.length,
                constellations: S.R.layers.constellations.length };
    $("layer-list").innerHTML = LAYERS.map((l) => `<label class="layer"><input type="checkbox" data-l="${l.id}" ${ON[l.id] ? "checked" : ""}>`
      + `<span class="sw" style="background:${l.color}"></span>${l.label}${n[l.id] !== undefined ? `<small>${n[l.id]}</small>` : ""}</label>`).join("");
    $("layer-list").querySelectorAll("input").forEach((el) => el.addEventListener("change", () => { ON[el.dataset.l] = el.checked; redraw(); }));
  }
  ["bright", "contrast"].forEach((id) => $(id).addEventListener("input", redraw));
  $("vec").addEventListener("input", () => { $("vec-k").textContent = $("vec").value; if (!ON.vectors) { ON.vectors = true; $("layer-list").querySelector('[data-l="vectors"]').checked = true; } redraw(); });

  function resize() {
    const r = stage.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
    redraw();
  }
  window.addEventListener("resize", () => { if (S.img) { resize(); } });

  function fitView() {
    const r = stage.getBoundingClientRect();
    S.fit = Math.min(r.width / S.W, r.height / S.H) * 0.98;
    S.scale = S.fit; S.ox = (r.width - S.W * S.scale) / 2; S.oy = (r.height - S.H * S.scale) / 2; redraw();
  }
  function zoomAt(f, sx, sy) {
    const r = stage.getBoundingClientRect();
    if (sx === undefined) { sx = r.width / 2; sy = r.height / 2; }
    const ns = Math.min(Math.max(S.scale * f, S.fit * 0.5), 16);
    S.ox = sx - (sx - S.ox) * (ns / S.scale); S.oy = sy - (sy - S.oy) * (ns / S.scale); S.scale = ns; redraw();
  }
  function centreOn(x, y, scale) {
    const r = stage.getBoundingClientRect();
    S.scale = Math.max(scale || S.scale, S.fit); S.ox = r.width / 2 - x * S.scale; S.oy = r.height / 2 - y * S.scale; redraw();
  }
  $("zin").addEventListener("click", () => zoomAt(1.6)); $("zout").addEventListener("click", () => zoomAt(1 / 1.6));
  $("zfit").addEventListener("click", fitView);
  $("z1").addEventListener("click", () => { const r = stage.getBoundingClientRect(); zoomAt(1 / S.scale, r.width / 2, r.height / 2); });

  /** Draw everything on ctx with the transform image px -> (x * sc + ox, y * sc + oy); u = size unit (CSS px). */
  function paint(ctx, sc, ox, oy, u, viewW, viewH, forExport) {
    const R = S.R, L = R.layers;
    const T = (x, y) => [x * sc + ox, y * sc + oy];
    const vis = (x, y, m = 40) => { const [a, b] = T(x, y); return a > -m && b > -m && a < viewW + m && b < viewH + m; };
    ctx.fillStyle = "#000"; ctx.fillRect(0, 0, viewW, viewH);
    if (ON.image) {
      ctx.save();
      const b = +$("bright").value, c = +$("contrast").value;
      if (b !== 1 || c !== 1) ctx.filter = `brightness(${b}) contrast(${c})`;
      ctx.imageSmoothingEnabled = sc * (S.W / S.img.naturalWidth) < 2.5;
      ctx.drawImage(S.img, ox, oy, S.W * sc, S.H * sc);
      ctx.restore();
    }
    const poly = (pts) => {
      let pen = false;
      for (const p of pts) {
        if (!p) { pen = false; continue; }
        const [a, b] = T(p[0], p[1]);
        if (pen) ctx.lineTo(a, b); else { ctx.moveTo(a, b); pen = true; }
      }
    };
    const text = (s, x, y, color, size, align = "center") => {
      ctx.font = `600 ${size}px system-ui, sans-serif`; ctx.textAlign = align; ctx.textBaseline = "middle";
      ctx.lineWidth = 3 * u; ctx.strokeStyle = "rgba(0,0,0,0.75)"; ctx.strokeText(s, x, y); ctx.fillStyle = color; ctx.fillText(s, x, y);
    };
    if (ON.grid) {
      ctx.lineWidth = 1 * u;
      for (const g of L.grid.alt) {
        ctx.beginPath(); ctx.setLineDash(g.alt === 0 ? [6 * u, 4 * u] : []);
        ctx.strokeStyle = g.alt === 0 ? "rgba(244,169,72,0.8)" : g.alt % 30 === 0 ? "rgba(190,210,230,0.62)" : "rgba(160,180,200,0.32)";
        poly(g.line); ctx.stroke();
      }
      ctx.setLineDash([]);
      for (const g of L.grid.az) {
        ctx.beginPath(); ctx.strokeStyle = g.az % 90 === 0 ? "rgba(190,210,230,0.62)" : "rgba(160,180,200,0.32)"; poly(g.line); ctx.stroke();
      }
      for (const l of L.grid.labels) { const [a, b] = T(l.x, l.y); text(l.text, a, b, "rgba(210,225,240,0.9)", 11 * u); }
    }
    if (ON.constellations) {
      ctx.strokeStyle = "rgba(255,159,28,0.55)"; ctx.lineWidth = 1.1 * u; ctx.beginPath();
      for (const c of L.constellations) for (const ln of c.lines) poly(ln);
      ctx.stroke();
    }
    if (ON.constnames) for (const c of L.constellations) if (c.label) { const [a, b] = T(c.label[0], c.label[1]); text(CONST_NAMES[c.abbr] || c.abbr, a, b, "rgba(255,196,119,0.95)", 12 * u); }
    if (ON.detections) {
      const d = L.detections; ctx.strokeStyle = "rgba(34,211,238,0.75)"; ctx.lineWidth = 1 * u; ctx.beginPath();
      const rr = Math.max(2.5 * u, 3 * sc);
      for (let i = 0; i < d.x.length; i++) { if (!vis(d.x[i], d.y[i])) continue; const [a, b] = T(d.x[i], d.y[i]); ctx.moveTo(a + rr, b); ctx.arc(a, b, rr, 0, 2 * Math.PI); }
      ctx.stroke();
    }
    if (ON.catalog) {
      ctx.strokeStyle = "rgba(74,222,128,0.85)"; ctx.lineWidth = 1.1 * u; ctx.beginPath();
      for (const s of L.catalog) {
        if (!vis(s.x, s.y)) continue;
        const [a, b] = T(s.x, s.y), h = Math.max(2.5, 7 - s.mag) * u;
        ctx.moveTo(a - h, b); ctx.lineTo(a - h * 0.35, b); ctx.moveTo(a + h * 0.35, b); ctx.lineTo(a + h, b);
        ctx.moveTo(a, b - h); ctx.lineTo(a, b - h * 0.35); ctx.moveTo(a, b + h * 0.35); ctx.lineTo(a, b + h);
      }
      ctx.stroke();
    }
    if (ON.pairs) {
      ctx.strokeStyle = "rgba(77,163,255,0.85)"; ctx.lineWidth = 1.3 * u; ctx.beginPath();
      const rr = Math.max((viewW < 700 ? 2.5 : 5) * u, 4 * sc);
      for (const p of L.pairs) { if (!p.inlier || !vis(p.x, p.y)) continue; const [a, b] = T(p.x, p.y); ctx.moveTo(a + rr, b); ctx.arc(a, b, rr, 0, 2 * Math.PI); }
      ctx.stroke();
    }
    if (ON.vectors) {
      const k = +$("vec").value; ctx.strokeStyle = "rgba(255,107,214,0.9)"; ctx.lineWidth = 1.2 * u; ctx.beginPath();
      for (const p of L.pairs) {
        if (!p.inlier || !vis(p.xp, p.yp)) continue;
        const [a, b] = T(p.xp, p.yp), ex = a + p.dx * k * sc, ey = b + p.dy * k * sc;
        ctx.moveTo(a, b); ctx.lineTo(ex, ey);
        const ang = Math.atan2(ey - b, ex - a), h = 4 * u;
        if (Math.hypot(ex - a, ey - b) > 2 * h) { ctx.moveTo(ex, ey); ctx.lineTo(ex - h * Math.cos(ang - 0.5), ey - h * Math.sin(ang - 0.5)); ctx.moveTo(ex, ey); ctx.lineTo(ex - h * Math.cos(ang + 0.5), ey - h * Math.sin(ang + 0.5)); }
      }
      ctx.stroke();
    }
    if (ON.names) {
      const lim = sc > S.fit * 2.5 ? 3.5 : sc > S.fit * 1.4 ? 2.5 : 1.8;
      for (const s of L.catalog) {
        if (!s.name || s.mag > lim || !vis(s.x, s.y)) continue;
        const [a, b] = T(s.x, s.y); text(s.name, a + 7 * u, b - 7 * u, "rgba(255,224,138,0.95)", 11 * u, "left");
      }
    }
    if (ON.cardinal) for (const c of L.cardinal) {
      const [a, b] = T(c.x, c.y); text(c.text, a, b, c.text.length === 1 ? "#f4a948" : "rgba(244,169,72,0.7)", (c.text.length === 1 ? 16 : 11) * u);
    }
    if (ON.markers) {
      const [zx, zy] = T(...R.markers.zenith), [cx, cy] = T(...R.markers.centre);
      ctx.strokeStyle = "#ff5c5c"; ctx.lineWidth = 1.6 * u; ctx.beginPath();
      ctx.moveTo(zx - 8 * u, zy); ctx.lineTo(zx + 8 * u, zy); ctx.moveTo(zx, zy - 8 * u); ctx.lineTo(zx, zy + 8 * u); ctx.stroke();
      text("zenith", zx + 10 * u, zy - 10 * u, "#ff8a8a", 11 * u, "left");
      ctx.beginPath(); ctx.arc(cx, cy, 6 * u, 0, 2 * Math.PI); ctx.stroke();
      text("optical centre", cx + 10 * u, cy + 12 * u, "#ff8a8a", 11 * u, "left");
    }
    if (S.focus && !forExport) {
      const [a, b] = T(S.focus[0], S.focus[1]);
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * u; ctx.beginPath(); ctx.arc(a, b, 14 * u, 0, 2 * Math.PI); ctx.stroke();
    }
  }

  function draw() {
    if (!S.img) return;
    const dpr = window.devicePixelRatio || 1, ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    paint(ctx, S.scale, S.ox, S.oy, 1, canvas.width / dpr, canvas.height / dpr, false);
  }

  // pointer interaction: pan, pinch, hover
  const toImg = (e) => { const r = canvas.getBoundingClientRect(); const sx = e.clientX - r.left, sy = e.clientY - r.top; return [sx, sy, (sx - S.ox) / S.scale, (sy - S.oy) / S.scale]; };
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); const [sx, sy] = toImg(e); zoomAt(Math.exp(-e.deltaY * 0.0015), sx, sy); }, { passive: false });
  canvas.addEventListener("dblclick", (e) => { const [sx, sy] = toImg(e); zoomAt(2, sx, sy); });
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId); S.pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (S.pointers.size === 2) { const [p, q] = [...S.pointers.values()]; S.pinch = { d: Math.hypot(p[0] - q[0], p[1] - q[1]), s: S.scale }; }
    canvas.classList.add("dragging");
  });
  canvas.addEventListener("pointermove", (e) => {
    const prev = S.pointers.get(e.pointerId);
    if (prev) {
      if (S.pointers.size === 2 && S.pinch) {
        S.pointers.set(e.pointerId, [e.clientX, e.clientY]);
        const [p, q] = [...S.pointers.values()], r = canvas.getBoundingClientRect();
        const d = Math.hypot(p[0] - q[0], p[1] - q[1]);
        zoomAt((S.pinch.s * d / S.pinch.d) / S.scale, (p[0] + q[0]) / 2 - r.left, (p[1] + q[1]) / 2 - r.top);
      } else { S.ox += e.clientX - prev[0]; S.oy += e.clientY - prev[1]; S.pointers.set(e.pointerId, [e.clientX, e.clientY]); redraw(); }
    }
    hover(e);
  });
  const up = (e) => { S.pointers.delete(e.pointerId); if (S.pointers.size < 2) S.pinch = null; if (!S.pointers.size) canvas.classList.remove("dragging"); };
  canvas.addEventListener("pointerup", up); canvas.addEventListener("pointercancel", up);
  canvas.addEventListener("pointerleave", () => { $("tip").hidden = true; });

  function hover(e) {
    if (!S.R) return;
    const [sx, sy, x, y] = toImg(e);
    if (x < 0 || y < 0 || x >= S.W || y >= S.H) { $("readout").textContent = "outside the image"; $("tip").hidden = true; return; }
    const [[alt, az], zd] = S.model.unproject(x, y);
    $("readout").textContent = `x ${x.toFixed(1)}  y ${y.toFixed(1)}   alt ${alt.toFixed(2)}°  az ${az.toFixed(2)}°`;
    let best = null, bd = 12;
    for (const p of S.R.layers.pairs) { const d = Math.hypot(p.x * S.scale + S.ox - sx, p.y * S.scale + S.oy - sy); if (d < bd) { bd = d; best = { pair: p }; } }
    if (!best) for (const s of S.R.layers.catalog) { const d = Math.hypot(s.x * S.scale + S.ox - sx, s.y * S.scale + S.oy - sy); if (d < bd) { bd = d; best = { star: s }; } }
    const tip = $("tip");
    if (!best) { tip.hidden = true; return; }
    if (best.pair) {
      const p = best.pair;
      tip.innerHTML = `<b>${esc(p.name || "HIP " + p.hip)}</b>${p.name ? ` · HIP ${p.hip}` : ""}<br>mag ${fmt(p.mag, 2)} · alt ${fmt(p.alt)}° · az ${fmt(p.az)}°<br>`
        + `residual ${fmt(p.res)} px (Δx ${fmt(p.dx)}, Δy ${fmt(p.dy)})${p.inlier ? "" : "<br><i>rejected by the robust fit</i>"}`;
    } else {
      const s = best.star;
      tip.innerHTML = `<b>${esc(s.name || "HIP " + s.hip)}</b>${s.name ? ` · HIP ${s.hip}` : ""}<br>mag ${fmt(s.mag, 2)} · alt ${fmt(s.alt)}° · az ${fmt(s.az)}°<br><i>catalogue position, not matched</i>`;
    }
    tip.hidden = false;
    const r = stage.getBoundingClientRect();
    tip.style.left = Math.min(sx + 14, r.width - 270) + "px"; tip.style.top = Math.min(sy + 14, r.height - 90) + "px";
  }

  // exports
  $("exp-png").addEventListener("click", () => {
    const k = S.img.naturalWidth / S.W, w = S.img.naturalWidth, h = S.img.naturalHeight;
    const c = document.createElement("canvas"); c.width = w; c.height = h;
    paint(c.getContext("2d"), k, 0, 0, Math.max(1, w / 1400), w, h, true);
    c.toBlob((b) => download(b, stem() + "_layers.png"), "image/png");
  });
  $("exp-all").addEventListener("click", () => {
    const out = { ...S.R }; delete out.layers;
    out.pairs = S.R.layers.pairs; out.software = "ascal " + (S.cfg?.version || "");
    download(new Blob([JSON.stringify(out, null, 1)], { type: "application/json" }), stem() + "_results.json");
  });
  const stem = () => (S.R.original_name || "frame").replace(/\.[^.]+$/, "");
  function download(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 4000); }

  // ---------------------------------------------------------------- tables
  function renderTables() {
    const R = S.R, sm = R.summary, m = R.model, d = m.derived || {}, c = R.cascade || {}, a = c.accepted || {}, fr = R.frame;
    const scale = d.plate_scale_axis_arcmin_per_px;
    $("kpis").innerHTML = [
      [sm.n_pairs, "matched stars"], [fmt(sm.median_px) + " px", `median residual (${fmt(sm.median_px * scale, 1)}′)`],
      [fmt(sm.rms_px) + " px", "rms residual"], [Math.round(100 * sm.within_1px) + " %", "stars within 1 px"],
      [fmt(d.total_tilt_deg, 2) + "°", "tilt of the camera"], [fmt(R.timings.total_s, 1) + " s", "processing time"],
    ].map(([v, l]) => `<div class="kpi"><b>${v}</b><span>${l}</span></div>`).join("");

    const row = (k, v, note) => `<tr><th>${k}</th><td class="num">${v}</td><td><small>${note || ""}</small></td></tr>`;
    const prior = { "-0.03,0": "equisolid-like", "0.04,0": "stereographic-like", "0,0": "equidistant" }[(a.first_prior || []).join(",")] || (a.first_prior || []).join(", ");
    const disc = { sky_disc: "detected sky disc", circumscribed: "circle circumscribed to the sensor", inscribed: "circle inscribed in the sensor", hough1: "Hough circle", hough2: "Hough circle" }[a.disc] || a.disc;
    const tables = [
      ["Camera model", [
        row("Optical centre (x, y)", `${fmt(m.cx, 2)}, ${fmt(m.cy, 2)}`, "px" + (m.mirror ? ", in the un-mirrored frame" : "")),
        row("Focal length f", fmt(m.f, 2), `px/rad (${fmt(d.focal_length_px_per_deg, 3)} px/°)`),
        row("k₃", m.k && m.k.length ? m.k[0].toFixed(5) : "—", "r = f (θ + k₃θ³ + k₅θ⁵)"),
        row("k₅", m.k && m.k.length > 1 ? m.k[1].toFixed(5) : "—", ""),
        row("Image rotation ψ", fmt(m.psi, 3), "°, azimuth of the image up direction"),
        row("Tilt τx, τy", `${fmt(m.tau_x, 3)}, ${fmt(m.tau_y, 3)}`, "°"),
        row("Total tilt", fmt(d.total_tilt_deg, 3), "° between optical axis and zenith"),
        row("Decentering p₁, p₂", m.p ? `${m.p[0].toExponential(2)}, ${m.p[1].toExponential(2)}` : "—", m.p ? "" : "not fitted"),
        row("Mirrored image", m.mirror ? "yes" : "no", ""),
        row("Zenith at pixel", d.zenith_pixel ? `${fmt(d.zenith_pixel[0], 1)}, ${fmt(d.zenith_pixel[1], 1)}` : "—", ""),
        row("Horizon radius", fmt(d.horizon_radius_px, 1), "px from the optical centre"),
        row("Plate scale", `${fmt(scale, 3)} / ${fmt(d.plate_scale_horizon_arcmin_per_px, 3)}`, "′ per px, on the axis / at the horizon"),
      ]],
      ["Fit quality", [
        row("Matched stars", sm.n_pairs, `of ${sm.n_candidates ?? "—"} candidate pairs, ${fr.n_detections} detections`),
        row("Median residual", fmt(sm.median_px, 3), `px = ${fmt(sm.median_px * scale * 60, 0)}″ on the axis`),
        row("RMS residual", fmt(sm.rms_px, 3), "px"),
        row("90th percentile", fmt(sm.p90_px, 3), "px"),
        row("Within 1 px", Math.round(100 * sm.within_1px) + " %", ""),
        ...(sm.bands || []).filter((b) => b.n && b.band !== "all").map((b) =>
          row(`Altitude ${b.band}°`, `${fmt(b.median, 2)} px`, `${b.n} stars, p90 ${fmt(b.p90, 2)} px`)),
      ]],
      ["Frame and processing", [
        row("Image", `${fr.width} × ${fr.height}`, "px"),
        row("Mid-exposure", fr.utc_mid.replace("T", " ").replace("Z", "").slice(0, 19), `UTC, exposure ${fmt(fr.exposure_s, 1)} s`),
        row("Site", `${fmt(R.site.lat, 5)}, ${fmt(R.site.lon, 5)}`, `° N, ° E · ${fmt(R.site.elev, 0)} m`),
        row("Detections", fr.n_detections, `star FWHM ${fmt(fr.fwhm_measured, 2)} px, kernel ${fmt(fr.fwhm_kernel, 1)} px`),
        row("Sky disc", fr.disc ? `${fr.disc[0]}, ${fr.disc[1]}; r ${fr.disc[2]}` : "—", "px"),
        row("Accepted hypothesis", esc(disc || "—"), `${a.parity || ""} parity, ${prior} prior${a.detection ? `, detection repeated with a ${["", "1.5", "2"][a.detection]}× kernel` : ""}`),
        row("Pose margin", a.margin ? "×" + fmt(a.margin, 1) : "—", "best pose against the next one"),
        row("Pre-smoothing", fr.presmooth_sigma ? `σ ${fmt(fr.presmooth_sigma, 2)} px` : "no", "undersampled stars are smoothed"),
        row("Tiles", fr.tiles && fr.tiles > 1 ? `${fr.tiles} × ${fr.tiles}` : "no", "parallel detection above 16 Mpx"),
        row("Time", `${fmt(R.timings.total_s, 1)} s`, `${fmt(R.timings.detection_s, 1)} s detection + ${fmt(R.timings.calibration_s, 1)} s calibration`),
        row("Refraction", "applied", "Saemundsson (1986), pressure from the elevation"),
      ]],
    ];
    $("tables").innerHTML = tables.map(([h, rows]) => `<div class="table-card"><h3>${h}</h3><table>${rows.join("")}</table></div>`).join("");
  }

  // ---------------------------------------------------------------- charts
  const AX = { axisLine: { lineStyle: { color: "#555c66" } }, splitLine: { lineStyle: { color: "rgba(255,255,255,0.07)" } },
               axisLabel: { color: "#aab0ba" }, nameTextStyle: { color: "#aab0ba" }, nameLocation: "middle" };
  const BASE = { backgroundColor: "transparent", textStyle: { color: "#d6d9de", fontFamily: "system-ui, sans-serif" },
                 grid: { left: 62, right: 22, top: 48, bottom: 52 },
                 legend: { type: "scroll", top: 2, left: 4, right: 118, itemGap: 12, textStyle: { color: "#c6cad1", fontSize: 11 },
                           pageIconColor: "#f4a948", pageTextStyle: { color: "#aab0ba" } },
                 toolbox: { right: 4, top: 0, itemSize: 14, itemGap: 8, feature: { dataZoom: { title: { zoom: "zoom", back: "reset" } }, restore: { title: "restore" }, saveAsImage: { title: "save", backgroundColor: "#161b22" } }, iconStyle: { borderColor: "#8b8d97" } },
                 dataZoom: [{ type: "inside", xAxisIndex: 0, filterMode: "none" }, { type: "inside", yAxisIndex: 0, filterMode: "none" }],
                 animation: false };
  const starTip = (p) => `<b>${esc(p.name || "HIP " + p.hip)}</b><br>mag ${fmt(p.mag)} · alt ${fmt(p.alt)}° · az ${fmt(p.az)}°<br>residual ${fmt(p.res)} px`;

  function chart(title, note, option, onClick) {
    const card = document.createElement("div"); card.className = "chart-card";
    card.innerHTML = `<h3>${title}</h3><p>${note}</p><div class="chart"></div>`;
    $("charts").appendChild(card);
    const ch = echarts.init(card.querySelector(".chart"), null, { renderer: "canvas" });
    ch.setOption(Object.assign({}, BASE, option));
    if (onClick) ch.on("click", onClick);
    S.charts.push(ch);
    if (window.ResizeObserver) new ResizeObserver(() => ch.resize()).observe(card.querySelector(".chart"));
  }
  const focusPair = (p) => { if (!p) return; S.focus = [p.x, p.y]; centreOn(p.x, p.y, Math.max(S.fit * 6, 1)); $("viewer-wrap").scrollIntoView({ behavior: "smooth" }); };
  const clickPair = (e) => focusPair(e.data && e.data.p);

  function renderCharts() {
    if (!window.echarts) { $("charts").innerHTML = '<p class="muted">The charts library could not be loaded.</p>'; return; }
    $("charts").innerHTML = ""; S.charts.forEach((c) => c.dispose()); S.charts = [];
    const R = S.R, P = R.layers.pairs, inl = P.filter((p) => p.inlier), out = P.filter((p) => !p.inlier), C = R.curves, m = R.model;
    const pt = (p, x, y) => ({ value: [x, y], p });
    const tipItem = { trigger: "item", formatter: (e) => (e.data && e.data.p ? starTip(e.data.p) : `${e.seriesName}: ${e.value.map((v) => fmt(v, 2)).join(", ")}`) };
    const curve = (name, ys, color, dash) => ({ name, type: "line", showSymbol: false, data: C.theta_deg.map((t, i) => (ys[i] == null ? null : [ys[i], 90 - t])).filter(Boolean),
      lineStyle: { color, width: name === "model" ? 2 : 1.2, type: dash ? "dashed" : "solid" }, itemStyle: { color }, z: name === "model" ? 5 : 2, tooltip: { show: false } });
    const rmax = m.derived.horizon_radius_px * 1.08;

    chart("Projection: altitude vs radius", "Altitude in the camera frame (tilt removed) against distance to the optical centre, for the fitted model, the ideal fisheye projections with the same focal length, and the matched stars.", {
      tooltip: tipItem,
      xAxis: { ...AX, type: "value", name: "distance to the optical centre (px)", nameGap: 30, min: 0, max: Math.ceil(rmax / 100) * 100 },
      yAxis: { ...AX, type: "value", name: "altitude in the camera frame (°)", nameGap: 40, min: 0, max: 90 },
      series: [
        { name: "matched stars", type: "scatter", symbolSize: 4, data: inl.map((p) => pt(p, p.r, p.alt_cam)), itemStyle: { color: "#4da3ff", opacity: 0.6 }, z: 3 },
        curve("model", C.model, "#ff6b4a"), curve("equidistant", C.equidistant, "#9aa3ad", true), curve("equisolid", C.equisolid, "#6ee7b7", true),
        curve("stereographic", C.stereographic, "#c4b5fd", true),
      ] }, clickPair);

    const bands = (R.summary.bands || []).filter((b) => b.n && b.band !== "all");
    const bandMid = bands.map((b) => { const [lo, hi] = b.band.split("-").map(Number); return [(lo + hi) / 2, b.median]; });
    const top = Math.max(2.5, percentile(inl.map((p) => p.res), 98) * 1.25);
    chart("Residual vs altitude", "Distance between detected and predicted position of each star. The lowest altitudes are limited by the detections (extinction, small plate scale, edge of the field).", {
      tooltip: tipItem,
      xAxis: { ...AX, type: "value", name: "altitude (°)", nameGap: 30, min: 0, max: 90 },
      yAxis: { ...AX, type: "value", name: "residual (px)", nameGap: 40, min: 0, max: +top.toFixed(1) },
      series: [
        { name: "matched", type: "scatter", symbolSize: 4, data: inl.map((p) => pt(p, p.alt, p.res)), itemStyle: { color: "#4da3ff", opacity: 0.65 } },
        { name: "rejected", type: "scatter", symbol: "diamond", symbolSize: 6, data: out.map((p) => pt(p, p.alt, Math.min(p.res, top))), itemStyle: { color: "#f87171", opacity: 0.8 } },
        { name: "median per band", type: "line", data: bandMid, lineStyle: { color: "#f4a948", width: 2 }, itemStyle: { color: "#f4a948" }, symbolSize: 7,
          tooltip: { formatter: (e) => `median ${fmt(e.value[1])} px at ${e.value[0]}°` } },
      ] }, clickPair);

    const k = 100;
    chart("Residual vectors on the sensor", `Each arrow goes from the predicted to the detected position, magnified ×${k} and coloured by the residual. Systematic patterns would reveal a missing term of the model.`, {
      tooltip: tipItem, grid: { left: 62, right: 70, top: 48, bottom: 52 },
      xAxis: { ...AX, type: "value", name: "x (px)", nameGap: 30, min: 0, max: R.frame.width },
      yAxis: { ...AX, type: "value", name: "y (px)", nameGap: 44, min: 0, max: R.frame.height, inverse: true },
      visualMap: { type: "continuous", min: 0, max: +top.toFixed(1), dimension: 2, seriesIndex: [1], right: 4, top: 50, itemHeight: 140, calculable: false,
                   inRange: { color: RAMP }, textStyle: { color: "#aab0ba" }, text: ["px", ""] },
      series: [
        { name: "residual", type: "lines", coordinateSystem: "cartesian2d", polyline: false, silent: true, symbol: ["none", "arrow"], symbolSize: 6,
          data: inl.map((p) => ({ coords: [[p.xp, p.yp], [p.xp + p.dx * k, p.yp + p.dy * k]], lineStyle: { color: ramp(p.res / top) } })), lineStyle: { width: 1.3, opacity: 0.95 } },
        { name: "stars", type: "scatter", symbolSize: 3, data: inl.map((p) => ({ value: [p.xp, p.yp, p.res], p })) },
      ] }, clickPair);

    const bins = []; const bw = 0.1, nb = Math.ceil(top / bw);
    for (let i = 0; i < nb; i++) bins.push([+(i * bw + bw / 2).toFixed(2), 0]);
    inl.forEach((p) => { const i = Math.min(nb - 1, Math.floor(p.res / bw)); bins[i][1]++; });
    chart("Residual distribution", `Histogram of the residuals of the matched stars (bins of ${bw} px); the dashed line is the median.`, {
      tooltip: { trigger: "axis", formatter: (e) => `${fmt(e[0].value[0] - bw / 2, 1)}–${fmt(e[0].value[0] + bw / 2, 1)} px: ${e[0].value[1]} stars` },
      xAxis: { ...AX, type: "value", name: "residual (px)", nameGap: 30, min: 0, max: +top.toFixed(1) },
      yAxis: { ...AX, type: "value", name: "stars", nameGap: 40 },
      series: [{ name: "stars", type: "bar", data: bins, barWidth: "90%", itemStyle: { color: "#4da3ff" },
                 markLine: { silent: true, symbol: "none", lineStyle: { color: "#f4a948", type: "dashed" }, label: { color: "#f4a948", formatter: `median ${fmt(R.summary.median_px)} px` },
                             data: [{ xAxis: R.summary.median_px }] } }] });

    chart("Residual components", "Δx and Δy of every matched star (detected minus predicted). The circle marks 1 px. A shifted cloud would mean a timing or position error.", {
      tooltip: tipItem,
      xAxis: { ...AX, type: "value", name: "Δx (px)", nameGap: 30, min: -2.5, max: 2.5 },
      yAxis: { ...AX, type: "value", name: "Δy (px)", nameGap: 40, min: -2.5, max: 2.5 },
      series: [
        { name: "matched", type: "scatter", symbolSize: 4, data: inl.map((p) => pt(p, p.dx, p.dy)), itemStyle: { color: "#4da3ff", opacity: 0.6 } },
        { name: "1 px", type: "line", showSymbol: false, silent: true, data: Array.from({ length: 73 }, (_, i) => [Math.cos(i * 5 * DEG), Math.sin(i * 5 * DEG)]), lineStyle: { color: "#f4a948", type: "dashed" }, tooltip: { show: false } },
      ] }, clickPair);

    chart("Residual vs magnitude", "Bright stars are often saturated and faint ones noisy; both show here.", {
      tooltip: tipItem,
      xAxis: { ...AX, type: "value", name: "Hipparcos magnitude", nameGap: 30, scale: true },
      yAxis: { ...AX, type: "value", name: "residual (px)", nameGap: 40, min: 0, max: +top.toFixed(1) },
      series: [{ name: "matched", type: "scatter", symbolSize: 4, data: inl.map((p) => pt(p, p.mag, p.res)), itemStyle: { color: "#4da3ff", opacity: 0.65 } }] }, clickPair);

    chart("Plate scale", "Arcminutes per pixel along the radius and across it, from the zenith of the camera to its horizon.", {
      tooltip: { trigger: "axis", valueFormatter: (v) => fmt(v, 3) + "′/px" },
      xAxis: { ...AX, type: "value", name: "zenith distance in the camera frame (°)", nameGap: 30, min: 0, max: 95 },
      yAxis: { ...AX, type: "value", name: "arcmin per pixel", nameGap: 40, scale: true },
      series: [
        { name: "radial", type: "line", showSymbol: false, data: C.theta_deg.map((t, i) => [t, C.scale_radial[i]]), lineStyle: { color: "#ff6b4a", width: 2 }, itemStyle: { color: "#ff6b4a" } },
        { name: "tangential", type: "line", showSymbol: false, data: C.theta_deg.map((t, i) => [t, C.scale_tangential[i]]), lineStyle: { color: "#4da3ff", width: 2 }, itemStyle: { color: "#4da3ff" } },
      ] });

    chart("Residuals by altitude band", "Median and 90th percentile of the residual in each altitude band; the number of stars is in the tooltip.", {
      tooltip: { trigger: "axis", formatter: (e) => { const b = bands[e[0].dataIndex]; return `${b.band}°: ${b.n} stars<br>median ${fmt(b.median)} px · p90 ${fmt(b.p90)} px · ${Math.round(100 * b.within_1px)} % within 1 px`; } },
      xAxis: { ...AX, type: "category", name: "altitude band (°)", nameGap: 30, data: bands.map((b) => b.band) },
      yAxis: { ...AX, type: "value", name: "residual (px)", nameGap: 40 },
      dataZoom: [],
      series: [
        { name: "median", type: "bar", data: bands.map((b) => b.median), itemStyle: { color: "#4da3ff" } },
        { name: "p90", type: "bar", data: bands.map((b) => b.p90), itemStyle: { color: "#f4a948" } },
      ] });
    requestAnimationFrame(() => S.charts.forEach((c) => c.resize()));
  }

  const RAMP = ["#2563eb", "#22d3ee", "#facc15", "#f43f5e"];
  function ramp(t) {
    t = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
    const i = Math.min(RAMP.length - 2, Math.floor(t)), f = t - i;
    const c = (h) => [1, 3, 5].map((j) => parseInt(h.slice(j, j + 2), 16));
    const a = c(RAMP[i]), b = c(RAMP[i + 1]);
    return `rgb(${a.map((v, j) => Math.round(v + (b[j] - v) * f)).join(",")})`;
  }

  function percentile(a, q) { if (!a.length) return 0; const s = a.slice().sort((x, y) => x - y); return s[Math.min(s.length - 1, Math.floor((q / 100) * s.length))]; }

  // ---------------------------------------------------------------- date-time picker (UTC)
  const DTP = { open: false, y: 0, m: 0, sel: null, hh: 0, mm: 0, ss: 0 };
  const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
  const pad = (n) => String(n).padStart(2, "0");

  function readField() {
    const t = $("utc").value.trim().replace("Z", "").match(/^(\d{4})-(\d{2})-(\d{2})(?:[ T](\d{2}):(\d{2})(?::(\d{2}))?)?/);
    if (t) return { y: +t[1], m: +t[2] - 1, d: +t[3], hh: +(t[4] || 0), mm: +(t[5] || 0), ss: +(t[6] || 0) };
    const n = new Date();
    return { y: n.getUTCFullYear(), m: n.getUTCMonth(), d: n.getUTCDate(), hh: 0, mm: 0, ss: 0, empty: true };
  }
  function writeField() {
    if (!DTP.sel) return;
    const keep = $("utc").value.match(/(:\d{2})(\.\d+)/);       // keep fractional seconds typed by hand
    $("utc").value = `${DTP.sel.y}-${pad(DTP.sel.m + 1)}-${pad(DTP.sel.d)} ${pad(DTP.hh)}:${pad(DTP.mm)}:${pad(DTP.ss)}`
      + (keep && +keep[1].slice(1) === DTP.ss ? keep[2] : "");
  }
  function dial(id, label, max) {
    return `<div class="dial"><button type="button" data-d="${id}" data-s="1" aria-label="${label} up">▲</button>`
      + `<input type="text" inputmode="numeric" maxlength="2" id="dtp-${id}" aria-label="${label}" value="${pad(DTP[id])}" data-max="${max}">`
      + `<button type="button" data-d="${id}" data-s="-1" aria-label="${label} down">▼</button><small>${label}</small></div>`;
  }
  function renderDtp() {
    const el = $("dtp"), first = new Date(Date.UTC(DTP.y, DTP.m, 1)), start = (first.getUTCDay() + 6) % 7;
    const today = new Date(), cells = [];
    for (let i = 0; i < 42; i++) {
      const d = new Date(Date.UTC(DTP.y, DTP.m, 1 - start + i));
      const cls = [d.getUTCMonth() !== DTP.m ? "other" : "",
        DTP.sel && d.getUTCFullYear() === DTP.sel.y && d.getUTCMonth() === DTP.sel.m && d.getUTCDate() === DTP.sel.d ? "sel" : "",
        d.toISOString().slice(0, 10) === today.toISOString().slice(0, 10) ? "today" : ""].join(" ");
      cells.push(`<button type="button" class="${cls}" data-ymd="${d.toISOString().slice(0, 10)}">${d.getUTCDate()}</button>`);
    }
    el.innerHTML = `<div class="dtp-head"><button type="button" data-nav="-1" aria-label="Previous month">‹</button>
        <select id="dtp-month" aria-label="Month">${MONTHS.map((n, i) => `<option value="${i}" ${i === DTP.m ? "selected" : ""}>${n}</option>`).join("")}</select>
        <input type="number" id="dtp-year" aria-label="Year" min="1990" max="2100" value="${DTP.y}">
        <button type="button" data-nav="1" aria-label="Next month">›</button></div>
      <div class="dtp-grid">${["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"].map((d) => `<span class="dow">${d}</span>`).join("")}${cells.join("")}</div>
      <div class="dtp-time">${dial("hh", "hour", 23)}<span class="sep">:</span>${dial("mm", "min", 59)}<span class="sep">:</span>${dial("ss", "sec", 59)}</div>
      <div class="dtp-foot"><button type="button" id="dtp-now">Now (UTC)</button><button type="button" id="dtp-done" class="primary">Done</button></div>`;
  }
  function openDtp() {
    const f = readField();
    Object.assign(DTP, { open: true, y: f.y, m: f.m, hh: f.hh, mm: f.mm, ss: f.ss, sel: f.empty ? null : { y: f.y, m: f.m, d: f.d } });
    renderDtp(); $("dtp").hidden = false;
  }
  const closeDtp = () => { DTP.open = false; $("dtp").hidden = true; };
  function setDial(id, v) {
    const max = id === "hh" ? 23 : 59;
    DTP[id] = ((Math.round(v) % (max + 1)) + max + 1) % (max + 1);
    const inp = $("dtp-" + id); if (inp) inp.value = pad(DTP[id]);
    if (!DTP.sel) DTP.sel = { y: DTP.y, m: DTP.m, d: 1 };
    writeField();
  }
  $("dt-open").addEventListener("click", () => (DTP.open ? closeDtp() : openDtp()));
  $("dtp").addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    if (b.dataset.nav) { DTP.m += +b.dataset.nav; if (DTP.m < 0) { DTP.m = 11; DTP.y--; } if (DTP.m > 11) { DTP.m = 0; DTP.y++; } renderDtp(); }
    else if (b.dataset.ymd) { const [y, m, d] = b.dataset.ymd.split("-").map(Number); DTP.sel = { y, m: m - 1, d }; DTP.y = y; DTP.m = m - 1; writeField(); renderDtp(); }
    else if (b.dataset.d) setDial(b.dataset.d, DTP[b.dataset.d] + +b.dataset.s);
    else if (b.id === "dtp-now") { const n = new Date(); DTP.sel = { y: n.getUTCFullYear(), m: n.getUTCMonth(), d: n.getUTCDate() }; DTP.y = DTP.sel.y; DTP.m = DTP.sel.m; DTP.hh = n.getUTCHours(); DTP.mm = n.getUTCMinutes(); DTP.ss = n.getUTCSeconds(); writeField(); renderDtp(); }
    else if (b.id === "dtp-done") closeDtp();
  });
  $("dtp").addEventListener("change", (e) => {
    if (e.target.id === "dtp-month") { DTP.m = +e.target.value; renderDtp(); }
    if (e.target.id === "dtp-year") { DTP.y = Math.max(1990, Math.min(2100, +e.target.value || DTP.y)); renderDtp(); }
  });
  $("dtp").addEventListener("input", (e) => {
    const id = e.target.id && e.target.id.startsWith("dtp-") ? e.target.id.slice(4) : null;
    if (["hh", "mm", "ss"].includes(id) && /^\d{1,2}$/.test(e.target.value)) {
      const v = +e.target.value, max = +e.target.dataset.max;
      if (v <= max) { DTP[id] = v; if (!DTP.sel) DTP.sel = { y: DTP.y, m: DTP.m, d: 1 }; writeField(); }
    }
  });
  $("dtp").addEventListener("wheel", (e) => {
    const inp = e.target.closest(".dial")?.querySelector("input"); if (!inp) return;
    e.preventDefault(); setDial(inp.id.slice(4), DTP[inp.id.slice(4)] + (e.deltaY < 0 ? 1 : -1));
  }, { passive: false });
  $("dtp").addEventListener("keydown", (e) => {
    if (e.key === "Escape") { closeDtp(); $("dt-open").focus(); }
    const id = e.target.id && e.target.id.slice(4);
    if (["hh", "mm", "ss"].includes(id) && (e.key === "ArrowUp" || e.key === "ArrowDown")) { e.preventDefault(); setDial(id, DTP[id] + (e.key === "ArrowUp" ? 1 : -1)); }
  });
  document.addEventListener("pointerdown", (e) => { if (DTP.open && !e.target.closest(".dt-field")) closeDtp(); });
  $("utc").addEventListener("input", () => { if (DTP.open) openDtp(); });

  init();
})();
