/* AI Engineering Roadmap 2026 - Netflix-style front end. No build step, no framework. */
(function () {
  "use strict";
  const D = window.ROADMAP;
  const C = Object.assign({
    brand: "AI ENGINEERING", year: "2026", storage: "aie26",
    repo: "https://github.com/vishalgwu/ultimate-ai-engineering-roadmap-2026",
    kicker: "ROADMAP SERIES", gateTitle: "Who's building?", gateSub: "Pick a profile and the catalogue reorders itself for your role.",
    top10Title: "Top 10 Chapters in AI Engineering Today", conceptsTitle: "Trending Concepts (Changes Daily)",
    newChapters: [12, 14, 17], newLabel: "RECENTLY EXPANDED", posterTag: "AI",
    episodeBlurb: "Every episode follows the same arc: Conceptual Deep Dive, Cross-Connection Scenarios, Production Anti-Patterns, and Minimal Working Code.",
    episodeSub: "Concept deep dive, scenarios, anti-patterns and code", search: "Titles, concepts, tools", title: "AI Engineering Roadmap 2026",
  }, D.config || {});
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => Array.from(el.querySelectorAll(s));
  const byNum = Object.fromEntries(D.chapters.map((c) => [c.num, c]));
  const GH = C.repo;

  /* ---------- storage (never throw) ---------- */
  const store = {
    get(k, d) { try { const v = localStorage.getItem(C.storage + ":" + k); return v ? JSON.parse(v) : d; } catch (e) { return d; } },
    set(k, v) { try { localStorage.setItem(C.storage + ":" + k, JSON.stringify(v)); } catch (e) { /* private mode */ } },
  };
  let profile = store.get("profile", null);
  let mylist = new Set(store.get("mylist", []));
  let progress = store.get("progress", {});      // {num: {pct, top}}
  let seen = store.get("seen", {});              // {"9.9": true}

  const PROFILES = (D.profiles && D.profiles.length) ? D.profiles : [
    { id: "app", name: "Application", role: "Application Engineer", initial: "A", color: "#E50914", rec: [6, 7, 9, 10, 14, 17] },
    { id: "platform", name: "Platform", role: "ML Platform Engineer", initial: "P", color: "#B20710", rec: [4, 5, 12, 13, 15, 14] },
    { id: "safety", name: "Safety & Evals", role: "Evaluation & Safety", initial: "S", color: "#831010", rec: [10, 14, 15, 6, 11] },
    { id: "switch", name: "Career Switcher", role: "New to AI", initial: "N", color: "#5b5b5b", rec: [1, 2, 3, 18, 16, 19] },
  ];

  /* ---------- helpers ---------- */
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  function rng(seed) { let a = seed >>> 0; return () => { a |= 0; a = (a + 0x6d2b79f5) | 0; let t = Math.imul(a ^ (a >>> 15), 1 | a); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
  const match = (c) => 84 + ((c.num * 7) % 15);
  const toast = (m) => { const t = $("#toast"); t.textContent = m; t.classList.remove("hidden"); clearTimeout(toast.h); toast.h = setTimeout(() => t.classList.add("hidden"), 2400); };
  const ico = {
    play: '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 4.5v15l13-7.5z"/></svg>',
    plus: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6"><path d="M12 5v14M5 12h14"/></svg>',
    check: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6"><path d="m4 12.5 5 5L20 6.5"/></svg>',
    down: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.6"><path d="m5 9 7 7 7-7"/></svg>',
    close: '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2.4"><path d="M5 5l14 14M19 5 5 19"/></svg>',
    chevL: '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6"><path d="m15 5-7 7 7 7"/></svg>',
    chevR: '<svg viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2.6"><path d="m9 5 7 7-7 7"/></svg>',
  };

  /* ---------- generative poster art ---------- */
  const PALETTES = (C.palettes && C.palettes.length) ? C.palettes : [["#E50914", "#2a0709"], ["#B20710", "#120a0b"], ["#ff3040", "#3a0a0f"], ["#8f0b12", "#0e0e0e"], ["#d40f1c", "#1d0b0d"], ["#6e0a10", "#161616"]];
  function poster(ch, kind) {
    const tall = kind === "tall";
    const W = tall ? 200 : 320, H = tall ? 300 : 180;
    const r = rng(ch.num * 7919 + 13);
    const [c1, c2] = PALETTES[ch.num % PALETTES.length];
    const pts = Array.from({ length: tall ? 14 : 20 }, () => [r() * W, r() * H * 0.82, 1.2 + r() * 3.4]);
    let lines = "";
    for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
      const d = Math.hypot(pts[i][0] - pts[j][0], pts[i][1] - pts[j][1]);
      if (d < (tall ? 80 : 90)) lines += `<line x1="${pts[i][0].toFixed(1)}" y1="${pts[i][1].toFixed(1)}" x2="${pts[j][0].toFixed(1)}" y2="${pts[j][1].toFixed(1)}" stroke="#fff" stroke-opacity="${(0.28 - d / 400).toFixed(2)}" stroke-width=".7"/>`;
    }
    const dots = pts.map((p) => `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="${p[2].toFixed(1)}" fill="#fff" fill-opacity="${(0.25 + r() * 0.55).toFixed(2)}"/>`).join("");
    const words = ch.title.split(" "); const rows = []; let cur = "";
    words.forEach((w) => { if ((cur + " " + w).trim().length > (tall ? 12 : 17)) { rows.push(cur.trim()); cur = w; } else cur += " " + w; });
    rows.push(cur.trim());
    const longest = Math.max(...rows.map((t) => t.length));
    const fs = Math.min(tall ? 26 : 25, Math.floor((W - (tall ? 28 : 32)) / (longest * 0.52)));
    const lh = Math.round(fs * 0.92);
    const ty = H - 14 - (rows.length - 1) * lh;
    const text = rows.slice(0, 3).map((t, i) => `<text x="${tall ? 14 : 16}" y="${ty + i * lh}" font-family="Bebas Neue,Impact,sans-serif" font-size="${fs}" fill="#fff" letter-spacing="1">${esc(t.toUpperCase())}</text>`).join("");
    const id = `g${ch.num}${tall ? "t" : "w"}`;
    return `<svg class="poster-svg" viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid slice" role="img" aria-label="${esc(ch.title)} poster">
      <defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient>
      <linearGradient id="${id}s" x1="0" y1="0" x2="0" y2="1"><stop offset=".35" stop-color="#000" stop-opacity="0"/><stop offset="1" stop-color="#000" stop-opacity=".85"/></linearGradient></defs>
      <rect width="${W}" height="${H}" fill="url(#${id})"/>
      <text x="${W - 8}" y="${tall ? 170 : 128}" text-anchor="end" font-family="Bebas Neue,Impact,sans-serif" font-size="${tall ? 190 : 150}" fill="#fff" fill-opacity=".09">${String(ch.num).padStart(2, "0")}</text>
      ${lines}${dots}<rect width="${W}" height="${H}" fill="url(#${id}s)"/>${text}</svg>`;
  }

  /* ---------- cards ---------- */
  function actions(ch) {
    const on = mylist.has(ch.num);
    return `<div class="btns">
      <button class="rbtn play" data-act="play" data-ch="${ch.num}" aria-label="Read ${esc(ch.title)}" title="Start reading">${ico.play}</button>
      <button class="rbtn ${on ? "on" : ""}" data-act="list" data-ch="${ch.num}" aria-label="${on ? "Remove from" : "Add to"} My List" title="${on ? "Remove from" : "Add to"} My List">${on ? ico.check : ico.plus}</button>
      <button class="rbtn more" data-act="info" data-ch="${ch.num}" aria-label="More info" title="More info">${ico.down}</button></div>`;
  }
  function cardInfo(ch) {
    return `<div class="info">${actions(ch)}
      <div class="cmeta"><span class="match">${match(ch)}% Match</span><span class="chip">${esc(ch.level)}</span><span>${ch.minutes} min</span></div>
      <div class="gen">${ch.genres.map(esc).join('<i>&bull;</i>')}</div></div>`;
  }
  function chapterCard(ch, opts = {}) {
    const p = progress[ch.num]; const pct = p ? Math.min(100, Math.round(p.pct)) : 0;
    const isNew = (C.newChapters || []).includes(ch.num) && !opts.noTag;
    return `<div class="card" tabindex="0" data-ch="${ch.num}" role="button" aria-label="${esc(ch.title)}">
      <div class="poster">${poster(ch)}<span class="tag">${isNew ? `<span class="tag new" style="position:static">${esc(C.newLabel)}</span>` : esc(C.posterTag)}</span>
      ${pct > 1 ? `<div class="prog"><span style="width:${pct}%"></span></div>` : ""}</div>${cardInfo(ch)}</div>`;
  }
  function topCard(ch, rank) {
    return `<div class="card" tabindex="0" data-ch="${ch.num}" role="button" aria-label="Number ${rank}: ${esc(ch.title)}">
      <span class="rank" aria-hidden="true">${rank}</span><div class="cardbox"><div class="poster">${poster(ch, C.top10Tall === false ? undefined : "tall")}</div>${cardInfo(ch)}</div></div>`;
  }
  function conceptCard(c) {
    const ch = byNum[c.chapter];
    return `<div class="card concept" tabindex="0" data-ch="${c.chapter}" data-sec="${c.num}" role="button" aria-label="${esc(c.title)}">
      <div class="cbody"><div class="cnum">${c.num}</div><div><div class="ctitle">${esc(c.title)}</div><div class="csub">${esc(ch.title)}</div></div></div></div>`;
  }

  function rowEl(id, title, html, cls = "") {
    return `<section class="row ${cls}" id="row-${id}" aria-label="${esc(title)}">
      <h2 class="row-title">${esc(title)}<span class="explore">Explore all &rsaquo;</span></h2>
      <button class="arrow l" aria-label="Scroll left" tabindex="-1">${ico.chevL}</button>
      <div class="scroller">${html}</div>
      <button class="arrow r" aria-label="Scroll right" tabindex="-1">${ico.chevR}</button></section>`;
  }

  function renderRows() {
    const out = [];
    const cont = Object.keys(progress).map(Number).filter((n) => progress[n].pct > 1 && progress[n].pct < 98 && byNum[n]).sort((a, b) => (progress[b].t || 0) - (progress[a].t || 0));
    if (cont.length) out.push(rowEl("continue", "Continue Reading", cont.map((n) => chapterCard(byNum[n], { noTag: 1 })).join("")));
    if (profile && profile.rec) out.push(rowEl("picks", `Top Picks for ${profile.role}`, profile.rec.map((n) => chapterCard(byNum[n], { noTag: 1 })).join("")));
    out.push(rowEl("top10", C.top10Title, D.top10.map((n, i) => topCard(byNum[n], i + 1)).join(""), "top10"));
    D.rows.forEach((r) => out.push(rowEl(r.id, r.title, r.chapters.map((n) => chapterCard(byNum[n])).join(""))));
    const day = Math.floor(Date.now() / 864e5); const rr = rng(day);
    const picks = [...D.concepts].sort(() => rr() - 0.5).slice(0, 16);
    out.push(rowEl("concepts", C.conceptsTitle, picks.map(conceptCard).join("")));
    out.push(rowEl("mylist", "My List", mylist.size ? [...mylist].map((n) => chapterCard(byNum[n], { noTag: 1 })).join("") : '<div class="empty-row">Add chapters with the + button and they will appear here.</div>'));
    $("#rows").innerHTML = out.join("");
    bindArrows();
  }
  function bindArrows() {
    $$(".row").forEach((row) => {
      const sc = $(".scroller", row);
      $(".arrow.l", row).onclick = () => sc.scrollBy({ left: -sc.clientWidth * 0.85, behavior: "smooth" });
      $(".arrow.r", row).onclick = () => sc.scrollBy({ left: sc.clientWidth * 0.85, behavior: "smooth" });
    });
  }

  /* ---------- hero ---------- */
  const FEATURED = (D.featured && D.featured.length) ? D.featured : [9, 11, 13, 10, 12, 14];
  let heroIdx = 0, heroTimer = null;
  function setHero(i) {
    heroIdx = i; const ch = byNum[FEATURED[i]];
    const b = $(".hero-body"); b.style.animation = "none"; void b.offsetWidth; b.style.animation = "";
    $("#hero-title").textContent = ch.title;
    const art = $("#hero-art"); art.style.animation = "none"; void art.offsetWidth; art.style.animation = ""; art.innerHTML = poster(ch);
    $("#hero-meta").innerHTML = `<span class="match">${match(ch)}% Match</span><span>${esc(C.year)}</span><span class="chip">${esc(ch.level)}</span><span>${ch.sections.length} episodes</span><span>${ch.minutes} min read</span>`;
    $("#hero-desc").textContent = ch.tagline;
    $("#hero-play").onclick = () => openReader(ch.num);
    $("#hero-info").onclick = () => openModal(ch.num);
    $("#hero-dots").innerHTML = FEATURED.map((n, k) => `<button class="${k === i ? "on" : ""}" aria-label="Show ${esc(byNum[n].title)}" data-k="${k}"></button>`).join("");
    $$("#hero-dots button").forEach((bt) => (bt.onclick = () => { setHero(+bt.dataset.k); startHero(); }));
  }
  function startHero() { clearInterval(heroTimer); heroTimer = setInterval(() => { if (!document.hidden && $("#reader").classList.contains("hidden")) setHero((heroIdx + 1) % FEATURED.length); }, 10000); }

  function heroCanvas() {
    const cv = $("#hero-canvas"), ctx = cv.getContext("2d");
    const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let w, h, pts, raf, visible = true;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    function size() { w = cv.clientWidth; h = cv.clientHeight; cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0); const n = Math.round(Math.min(90, (w * h) / 16000)); const r = rng(42); pts = Array.from({ length: n }, () => ({ x: r() * w, y: r() * h, vx: (r() - 0.5) * 0.35, vy: (r() - 0.5) * 0.35, r: 1 + r() * 2 })); }
    function frame() {
      ctx.clearRect(0, 0, w, h);
      for (const p of pts) { if (!still) { p.x += p.vx; p.y += p.vy; if (p.x < 0 || p.x > w) p.vx *= -1; if (p.y < 0 || p.y > h) p.vy *= -1; } }
      for (let i = 0; i < pts.length; i++) for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot(pts[i].x - pts[j].x, pts[i].y - pts[j].y);
        if (d < 150) { ctx.strokeStyle = `rgba(229,9,20,${0.34 * (1 - d / 150)})`; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(pts[i].x, pts[i].y); ctx.lineTo(pts[j].x, pts[j].y); ctx.stroke(); }
      }
      for (const p of pts) { ctx.fillStyle = "rgba(255,70,80,.75)"; ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, 6.283); ctx.fill(); }
      if (!still && visible && !document.hidden) raf = requestAnimationFrame(frame);
    }
    size(); frame();
    addEventListener("resize", () => { size(); if (still) frame(); });
    new IntersectionObserver((e) => { visible = e[0].isIntersecting; if (visible && !still) { cancelAnimationFrame(raf); frame(); } }).observe(cv);
    document.addEventListener("visibilitychange", () => { if (!document.hidden && !still) { cancelAnimationFrame(raf); frame(); } });
  }

  /* ---------- modal ---------- */
  function openModal(num) {
    const ch = byNum[num]; if (!ch) return;
    const idx = D.chapters.findIndex((c) => c.num === num);
    const eps = ch.sections.map((s, i) => {
      const m = Math.max(1, Math.round((s.words || 200) / 220));
      return `<div class="ep ${seen[num + ":" + s.num] ? "seen" : ""}" data-ch="${num}" data-sec="${s.num}" role="button" tabindex="0"><div class="n">${seen[num + ":" + s.num] ? "&#10003;" : i + 1}</div><div><div class="t">${esc(s.num + "  " + s.title)}</div><div class="d">${esc(C.episodeSub)}</div></div><div class="min">${m}m</div></div>`;
    }).join("");
    const more = D.chapters.filter((c) => c.num !== num && c.genres.some((g) => ch.genres.includes(g)) || Math.abs(c.num - num) === 1).filter((c) => c.num !== num).slice(0, 6);
    const on = mylist.has(num);
    $("#modal-card").innerHTML = `
      <div class="m-hero">${poster(ch)}
        <button class="m-close" aria-label="Close">${ico.close}</button>
        <div class="m-head"><h2 id="m-title">${esc(ch.title)}</h2>
          <div class="hero-actions"><button class="btn btn-play" data-act="play" data-ch="${num}">${ico.play.replace("<svg", '<svg width="22" height="22"')}${progress[num] && progress[num].pct > 1 ? "Resume" : "Start reading"}</button>
          <button class="rbtn ${on ? "on" : ""}" data-act="list" data-ch="${num}" aria-label="Toggle My List" style="width:40px;height:40px">${on ? ico.check : ico.plus}</button></div></div></div>
      <div class="m-body">
        <p class="meta-line"><span class="match" style="color:var(--green);font-weight:700">${match(ch)}% Match</span><span>${esc(C.year)}</span><span class="chip">${esc(ch.level)}</span><span>${ch.sections.length} episodes</span><span>${ch.words.toLocaleString()} words</span><span class="chip">${ch.codeBlocks} code samples</span></p>
        <div class="m-desc"><p>${esc(ch.tagline)}</p><p style="margin-top:.8rem;color:var(--muted)">${esc(C.episodeBlurb)}</p>
          <p style="margin-top:1rem"><a class="pill" href="${ch.pdf}" download>Download PDF</a> <a class="pill" href="${GH}/blob/main/chapters/${ch.slug}.md" target="_blank" rel="noopener">View source</a></p></div>
        <div class="m-side"><div><b>Starring:</b> <span class="link">${ch.cast.map(esc).join(", ")}</span></div><div><b>Genres:</b> <span class="link">${ch.genres.map(esc).join(", ")}</span></div><div><b>Season:</b> <span class="link">Episode ${idx + 1} of ${D.chapters.length}</span></div></div>
        <div class="m-eps"><h3>Episodes</h3>${eps}</div>
        <div class="m-more"><h3>More Like This</h3><div class="grid">${more.map((c) => chapterCard(c, { noTag: 1 })).join("")}</div></div>
      </div>`;
    const m = $("#modal"); m.classList.remove("hidden"); m.scrollTop = 0; document.body.style.overflow = "hidden";
  }
  function closeModal() { $("#modal").classList.add("hidden"); if ($("#reader").classList.contains("hidden")) document.body.style.overflow = ""; }

  /* ---------- reader ---------- */
  // Programmatic jumps (TOC, deep links, skip-to-code) should be instant, not a 10,000px smooth scroll.
  function jumpTo(el, block) {
    const sc = $("#reader-article"); const prev = sc.style.scrollBehavior;
    sc.style.scrollBehavior = "auto"; el.scrollIntoView({ block: block || "start" }); sc.style.scrollBehavior = prev;
  }
  let mermaidP = null;
  function loadMermaid() {
    if (window.mermaid) return Promise.resolve(window.mermaid);
    if (!mermaidP) mermaidP = new Promise((res, rej) => {
      const sc = document.createElement("script");
      sc.src = "https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"; sc.onload = () => res(window.mermaid); sc.onerror = rej;
      document.head.appendChild(sc);
    });
    return mermaidP;
  }
  async function renderMermaid(wrap) {
    const blocks = $$("pre code.language-mermaid", wrap);
    if (!blocks.length) return;
    try {
      const mm = await loadMermaid();
      mm.initialize(C.mermaidTheme === "default" ? { startOnLoad: false, theme: "default", securityLevel: "strict" } : { startOnLoad: false, theme: "dark", securityLevel: "strict", themeVariables: { primaryColor: "#2a0709", primaryBorderColor: "#E50914", lineColor: "#b3b3b3", primaryTextColor: "#fff" } });
      for (const c of blocks) {
        const pre = c.closest("pre"); const d = document.createElement("div"); d.className = "mermaid"; d.textContent = c.textContent;
        pre.replaceWith(d);
        try { await mm.run({ nodes: [d] }); } catch (e) { d.replaceWith(pre); }   // keep the source visible if a diagram fails
      }
    } catch (e) { /* offline or blocked: leave diagrams as code */ }
  }
  const mdCache = {};
  let cur = null; // {num}
  async function loadMd(ch) { if (mdCache[ch.num]) return mdCache[ch.num]; const r = await fetch(ch.md); if (!r.ok) throw new Error("HTTP " + r.status); return (mdCache[ch.num] = await r.text()); }

  function highlightPy(code) {
    const re = /(#[^\n]*)|("""[\s\S]*?"""|'''[\s\S]*?'''|f?"(?:\\.|[^"\\\n])*"|f?'(?:\\.|[^'\\\n])*')|\b(def|class|return|if|elif|else|for|while|in|not|and|or|import|from|as|with|try|except|finally|raise|yield|async|await|lambda|None|True|False|pass|break|continue|is|global|assert)\b|\b(\d+(?:\.\d+)?)\b|\b([A-Za-z_]\w*)(?=\()/g;
    let out = "", last = 0, m;
    while ((m = re.exec(code))) {
      out += esc(code.slice(last, m.index)); last = re.lastIndex;
      const cls = m[1] ? "c" : m[2] ? "s" : m[3] ? "k" : m[4] ? "n" : "f";
      out += `<span class="tk-${cls}">${esc(m[0])}</span>`;
    }
    return out + esc(code.slice(last));
  }

  async function openReader(num, secNum, fromHash) {
    const ch = byNum[num]; if (!ch) return;
    closeModal();
    const rd = $("#reader"), art = $("#reader-article");
    cur = { num };
    rd.classList.remove("hidden"); document.body.style.overflow = "hidden";
    $("#reader-title").innerHTML = `<small>Episode ${D.chapters.findIndex((c) => c.num === num) + 1} / ${D.chapters.length}</small>${esc(ch.title)}`;
    $("#reader-pdf").href = ch.pdf; $("#reader-src").href = `${GH}/blob/main/chapters/${ch.slug}.md`;
    const i = D.chapters.findIndex((c) => c.num === num);
    $("#reader-prev").disabled = i === 0; $("#reader-next").disabled = i === D.chapters.length - 1;
    $("#reader-toc").innerHTML = `<div class="toc-h">Episodes</div><button class="toc-i on" data-top="1"><b>00</b>Overview</button>` + ch.sections.map((s) => `<button class="toc-i" data-s="${s.num}"><b>${s.num}</b>${esc(s.title)}</button>`).join("");
    art.innerHTML = '<div class="md"><p style="color:var(--muted)">Loading episode...</p></div>';
    if (!fromHash) { const h = `#/read/${num}${secNum ? "/" + secNum : ""}`; if (location.hash !== h) { suppress = true; location.hash = h; } }
    let text;
    try { text = await loadMd(ch); } catch (e) { art.innerHTML = `<div class="md"><h1>Could not load this chapter</h1><p>${esc(e.message)}. Read it on <a href="${GH}/blob/main/chapters/${ch.slug}.md">GitHub</a>.</p></div>`; return; }
    if (!cur || cur.num !== num) return;
    const wrap = document.createElement("div"); wrap.className = "md"; wrap.innerHTML = marked.parse(text, { gfm: true, breaks: false });
    $$("h2", wrap).forEach((h) => { const m = h.textContent.match(/^(\d+)(?:\.(\d+))?\.?\s+(.*)$/); if (m) { const sn = m[2] ? `${m[1]}.${m[2]}` : `${num}.${m[1]}`; h.id = "s-" + sn; h.innerHTML = `<span class="secn">${sn}</span>${esc(m[3])}`; } });
    $$("pre", wrap).forEach((pre) => {
      const code = $("code", pre); const raw = code.textContent;
      if (/language-(python|py)/.test(code.className)) code.innerHTML = highlightPy(raw);
      const b = document.createElement("button"); b.className = "copy"; b.textContent = "Copy";
      b.onclick = () => { (navigator.clipboard ? navigator.clipboard.writeText(raw) : Promise.reject()).then(() => { b.textContent = "Copied"; setTimeout(() => (b.textContent = "Copy"), 1400); }).catch(() => toast("Select the code and copy manually")); };
      pre.appendChild(b);
    });
    const nx = D.chapters[i + 1];
    if (nx) { const u = document.createElement("div"); u.className = "upnext"; u.dataset.ch = nx.num; u.innerHTML = `<div class="up-p">${poster(nx)}</div><div class="up-b"><small>UP NEXT</small><strong>${esc(nx.title)}</strong><span style="color:var(--muted);font-size:.9rem">${esc(nx.tagline)}</span></div>`; wrap.appendChild(u); }
    art.innerHTML = ""; art.appendChild(wrap);
    renderMermaid(wrap);
    const target = secNum ? $(`[id="s-${secNum}"]`, wrap) : null;   // attribute selector: ids contain dots (9.9)
    const saved = progress[num];
    if (target) jumpTo(target); else if (saved && saved.top > 200 && saved.pct < 97) { art.scrollTop = saved.top; toast("Resuming where you left off"); } else art.scrollTop = 0;
    onReaderScroll();
    $("#skip-code").classList.remove("hidden"); clearTimeout(openReader.skipT); openReader.skipT = setTimeout(() => $("#skip-code").classList.add("hidden"), 12000);
    art.focus({ preventScroll: true });
  }
  function onReaderScroll() {
    if (!cur) return; const art = $("#reader-article"); const max = art.scrollHeight - art.clientHeight;
    const pct = max > 0 ? (art.scrollTop / max) * 100 : 0;
    $("#reader-bar").style.width = pct + "%";
    const ch = byNum[cur.num]; let active = null; const hs = $$("h2[id^='s-']", art);
    const topLine = art.getBoundingClientRect().top + 90;
    hs.forEach((h) => { if (h.getBoundingClientRect().top <= topLine) active = h.id.slice(2); });
    $$(".toc-i").forEach((b) => b.classList.toggle("on", active ? b.dataset.s === active : !!b.dataset.top));
    if (active) { const idx = ch.sections.findIndex((s) => s.num === active); for (let k = 0; k < idx; k++) seen[ch.num + ":" + ch.sections[k].num] = true; }
    clearTimeout(onReaderScroll.t); onReaderScroll.t = setTimeout(() => { progress[cur.num] = { pct: Math.max(pct, (progress[cur.num] || {}).pct || 0) > 97 ? 100 : pct, top: art.scrollTop, t: Date.now() }; store.set("progress", progress); store.set("seen", seen); }, 350);
  }
  function closeReader(fromHash) {
    $("#reader").classList.add("hidden"); document.body.style.overflow = ""; cur = null; $("#skip-code").classList.add("hidden");
    store.set("progress", progress);
    if (!fromHash && location.hash.startsWith("#/read")) { suppress = true; history.pushState(null, "", location.pathname + location.search + "#/"); }
    renderRows();
  }
  function stepChapter(d) { if (!cur) return; const i = D.chapters.findIndex((c) => c.num === cur.num) + d; if (D.chapters[i]) openReader(D.chapters[i].num); }

  /* ---------- search ---------- */
  function runSearch(q) {
    q = q.trim().toLowerCase();
    const res = $("#results"), hero = $("#hero"), rows = $("#rows"), stats = $("#stats");
    if (!q) { res.classList.add("hidden"); hero.classList.remove("hidden"); rows.classList.remove("hidden"); stats.classList.remove("hidden"); return; }
    hero.classList.add("hidden"); rows.classList.add("hidden"); stats.classList.add("hidden"); res.classList.remove("hidden");
    const chs = D.chapters.filter((c) => [c.title, c.tagline, c.cast.join(" "), c.genres.join(" "), c.level, c.sections.map((s) => s.title).join(" ")].join(" ").toLowerCase().includes(q));
    const secs = D.concepts.filter((c) => c.title.toLowerCase().includes(q)).slice(0, 40);
    res.innerHTML = `<h2>Results for <b>${esc(q)}</b>: ${chs.length} titles, ${secs.length} concepts</h2>` +
      (chs.length ? `<div class="grid">${chs.map((c) => chapterCard(c, { noTag: 1 })).join("")}</div>` : "") +
      (secs.length ? `<div class="res-sec"><h3>Concept episodes</h3>${secs.map((s) => `<a class="sec-hit" data-ch="${s.chapter}" data-sec="${s.num}"><b>${s.num}</b>${esc(s.title)}<span>${esc(byNum[s.chapter].title)}</span></a>`).join("")}</div>` : "") +
      (!chs.length && !secs.length ? '<p style="color:var(--muted)">Your search did not match anything. Try "RAG", "LoRA", "vLLM", "agents" or "evaluation".</p>' : "");
  }

  /* ---------- profile gate ---------- */
  function showGate() {
    $("#profiles").innerHTML = PROFILES.map((p) => `<li class="profile" tabindex="0" role="button" data-p="${p.id}"><div class="pic" style="background:${p.color}">${p.initial}</div><div class="pname">${esc(p.name)}</div><div class="prole">${esc(p.role)}</div></li>`).join("");
    $("#gate").classList.remove("hidden"); document.body.style.overflow = "hidden";
  }
  function pick(id) {
    profile = PROFILES.find((p) => p.id === id) || null; store.set("profile", profile);
    $("#gate").classList.add("hidden"); document.body.style.overflow = "";
    $("#avatar-initial").textContent = profile ? profile.initial : "VF"; $(".avatar").style.background = profile ? profile.color : "";
    if (C.splash !== false && !sessionStorage.getItem(C.storage + ":splash") && !matchMedia("(prefers-reduced-motion: reduce)").matches) {
      try { sessionStorage.setItem(C.storage + ":splash", "1"); } catch (e) { /* ignore */ }
      const s = $("#splash"); s.classList.remove("hidden"); setTimeout(() => s.classList.add("hidden"), 2700);
    }
    renderRows();
  }

  /* ---------- events ---------- */
  let suppress = false;
  document.addEventListener("click", (e) => {
    const t = e.target;
    const act = t.closest("[data-act]");
    if (act) {
      e.stopPropagation(); const n = +act.dataset.ch;
      if (act.dataset.act === "play") return openReader(n);
      if (act.dataset.act === "info") return openModal(n);
      if (act.dataset.act === "list") {
        if (mylist.has(n)) { mylist.delete(n); toast("Removed from My List"); } else { mylist.add(n); toast("Added to My List"); }
        store.set("mylist", [...mylist]);
        $$(`[data-act="list"][data-ch="${n}"]`).forEach((b) => { const on = mylist.has(n); b.classList.toggle("on", on); b.innerHTML = on ? ico.check : ico.plus; });
        const ml = $("#row-mylist .scroller");
        if (ml) ml.innerHTML = mylist.size ? [...mylist].map((k) => chapterCard(byNum[k], { noTag: 1 })).join("") : '<div class="empty-row">Add chapters with the + button and they will appear here.</div>';
        return;
      }
    }
    if (t.closest(".m-close") || t === $("#modal")) return closeModal();
    const ep = t.closest(".ep, .sec-hit, .concept");
    if (ep && ep.dataset.sec) return openReader(+ep.dataset.ch, ep.dataset.sec);
    const card = t.closest(".card, .upnext");
    if (card && card.dataset.ch) { if (card.classList.contains("upnext")) return openReader(+card.dataset.ch); return openModal(+card.dataset.ch); }
    const toc = t.closest(".toc-i");
    if (toc) { const art = $("#reader-article"); if (toc.dataset.top) art.scrollTo({ top: 0 }); else { const h = $(`[id="s-${toc.dataset.s}"]`, art); if (h) jumpTo(h); } $("#reader-toc").classList.remove("open"); return; }
    const prof = t.closest(".profile"); if (prof) return pick(prof.dataset.p);
    const jump = t.closest("[data-jump]");
    if (jump) { e.preventDefault(); const r = $("#row-" + jump.dataset.jump); if (r) r.scrollIntoView({ behavior: "smooth", block: "center" }); $$(".nav-links a").forEach((a) => a.classList.toggle("active", a === jump)); }
  });
  document.addEventListener("keydown", (e) => {
    const typing = /INPUT|TEXTAREA/.test((document.activeElement || {}).tagName || "");
    if (e.key === "Escape") { if (!$("#reader").classList.contains("hidden")) return closeReader(); if (!$("#modal").classList.contains("hidden")) return closeModal(); if ($("#search").classList.contains("open")) { $("#search-input").value = ""; runSearch(""); $("#search").classList.remove("open"); } }
    if (e.key === "/" && !typing) { e.preventDefault(); $("#search").classList.add("open"); $("#search-input").focus(); }
    if (cur && !typing) { if (e.key === "n") stepChapter(1); if (e.key === "p") stepChapter(-1); }
    if ((e.key === "Enter" || e.key === " ") && document.activeElement && document.activeElement.matches(".card, .ep, .profile")) { e.preventDefault(); document.activeElement.click(); }
  });
  $("#search-btn").onclick = () => { if (C.searchAlways) { runSearch($("#search-input").value); $("#search-input").focus(); return; } const s = $("#search"); s.classList.toggle("open"); if (s.classList.contains("open")) $("#search-input").focus(); else { $("#search-input").value = ""; runSearch(""); } };
  let st; $("#search-input").addEventListener("input", (e) => { clearTimeout(st); st = setTimeout(() => runSearch(e.target.value), 120); });
  $("#search-input").addEventListener("blur", (e) => { if (!e.target.value) $("#search").classList.remove("open"); });
  $("#avatar").onclick = showGate; $("#gate-skip").onclick = () => pick("");
  $("#reader-back").onclick = () => closeReader(); $("#reader-prev").onclick = () => stepChapter(-1); $("#reader-next").onclick = () => stepChapter(1);
  $("#reader-toc-btn").onclick = () => $("#reader-toc").classList.toggle("open");
  $("#reader-article").addEventListener("scroll", onReaderScroll, { passive: true });
  $("#skip-code").onclick = () => { const p = $("#reader-article pre"); if (p) jumpTo(p, "center"); $("#skip-code").classList.add("hidden"); };
  addEventListener("scroll", () => $("#nav").classList.toggle("solid", scrollY > 30), { passive: true });
  function route() {
    if (suppress) { suppress = false; return; }
    const m = location.hash.match(/^#\/read\/(\d+)(?:\/([\d.]+))?/);
    if (m && byNum[+m[1]]) openReader(+m[1], m[2], true); else if (!$("#reader").classList.contains("hidden")) closeReader(true);
  }
  addEventListener("hashchange", route);

  /* ---------- boot ---------- */
  function stats() {
    const t = D.totals;
    $("#stats").innerHTML = `<span><b>${t.chapters}</b>chapters</span><span><b>${t.sections}</b>concept episodes</span><span><b>${t.code}</b>code samples</span><span><b>${(t.words / 1000).toFixed(0)}k</b>words</span><span><b>MIT</b>open source</span>`;
  }
  function applyBrand() {
    document.title = `${C.title} - ${C.titleSuffix || "Binge the Series"}`;
    if (C.theme) document.body.dataset.theme = C.theme;
    $$(".wordmark").forEach((w) => (w.textContent = C.brand));
    const sw = $(".splash-word"); if (sw) sw.textContent = C.brand;
    const sy = $(".splash-year"); if (sy) sy.textContent = C.year;
    const kk = $(".hero-kicker"); if (kk) kk.innerHTML = `<span class="series-n">A</span> ${esc(C.kicker)}`;
    const gh = $(".gate h1"); if (gh) gh.textContent = C.gateTitle;
    const gs = $(".gate-sub"); if (gs) gs.textContent = C.gateSub;
    const si = $("#search-input"); if (si) si.placeholder = C.search;
    const fnote = $(".foot-note"); if (fnote && C.footerNote) fnote.textContent = C.footerNote;
    $$('a[href*="ultimate-ai-engineering-roadmap-2026"]').forEach((a) => (a.href = a.href.replace("https://github.com/vishalgwu/ultimate-ai-engineering-roadmap-2026", GH)));
  }
  applyBrand();
  stats(); setHero(0); startHero(); if (C.canvas !== false) heroCanvas();
  if (profile) { $("#avatar-initial").textContent = profile.initial; $(".avatar").style.background = profile.color; renderRows(); } else { renderRows(); if (!/^#\/read\//.test(location.hash)) showGate(); }
  route();
})();
