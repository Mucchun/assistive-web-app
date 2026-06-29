import { useCallback, useEffect, useRef, useState } from "react";
import * as cocoSsd from "@tensorflow-models/coco-ssd";
import * as tf from "@tensorflow/tfjs";
import Anthropic from "@anthropic-ai/sdk";
import "./App.css";

// ── Types ─────────────────────────────────────────────────────────────────────
type Tab = "scan" | "history" | "stats" | "settings";
type TextSize = "normal" | "large" | "xlarge";
type LangCode = "en" | "es" | "fr" | "tr" | "ar";
type Pred = { class: string; score: number; bbox: number[] };
type ScanEntry = {
  id: string;
  ts: number;
  objects: Array<{ class: string; score: number; distance: string; direction: string }>;
  searchTarget?: string;
  found?: boolean;
};
type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((e: SpeechRecognitionEvent) => void) | null;
  onerror: ((e: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
};

// ── Language config ───────────────────────────────────────────────────────────
const LANGS: Array<{ code: LangCode; label: string; tts: string; rec: string }> = [
  { code: "en", label: "English",   tts: "en-US", rec: "en-US" },
  { code: "es", label: "Español",   tts: "es-ES", rec: "es-ES" },
  { code: "fr", label: "Français",  tts: "fr-FR", rec: "fr-FR" },
  { code: "tr", label: "Türkçe",    tts: "tr-TR", rec: "tr-TR" },
  { code: "ar", label: "العربية",   tts: "ar-SA", rec: "ar-SA" },
];

// ── TTS module (module-level state so settings survive re-renders) ─────────────
let ttsQueue: Array<{ text: string; urgent: boolean }> = [];
let ttsSpeaking = false;
let ttsLang = "en-US";
let ttsRate = 1.0;

function drainTts() {
  if (ttsSpeaking || ttsQueue.length === 0) return;
  const item = ttsQueue.shift()!;
  ttsSpeaking = true;
  const utt = new SpeechSynthesisUtterance(item.text);
  utt.lang = ttsLang;
  utt.rate = ttsRate;
  utt.volume = 1;
  utt.onend = () => { ttsSpeaking = false; drainTts(); };
  utt.onerror = () => { ttsSpeaking = false; drainTts(); };
  window.speechSynthesis.speak(utt);
}

function speak(text: string, urgent = false) {
  if (urgent) {
    window.speechSynthesis.cancel();
    ttsQueue = [];
    ttsSpeaking = false;
    ttsQueue.push({ text, urgent });
  } else {
    if (ttsQueue.filter(q => !q.urgent).length < 2) {
      ttsQueue.push({ text, urgent: false });
    }
  }
  drainTts();
}

function resetTts() {
  window.speechSynthesis.cancel();
  ttsQueue = [];
  ttsSpeaking = false;
}

// ── Spatial helpers ───────────────────────────────────────────────────────────
function getDirection(bbox: number[], w: number): string {
  const cx = bbox[0] + bbox[2] / 2;
  if (cx < w / 3) return "left";
  if (cx > (2 * w) / 3) return "right";
  return "ahead";
}

function getDistance(bbox: number[], w: number, h: number): string {
  const area = (bbox[2] * bbox[3]) / (w * h);
  if (area < 0.03) return "far";
  if (area < 0.12) return "nearby";
  if (area < 0.30) return "close";
  return "very close";
}

function distRank(d: string): number {
  return ["very close", "close", "nearby", "far"].indexOf(d);
}

// ── Vibration: graduated by proximity ────────────────────────────────────────
function vibrateFor(dist: string, approaching = false): void {
  if (!("vibrate" in navigator)) return;
  if (approaching)          { navigator.vibrate([100, 50, 200]); return; }
  if (dist === "very close"){ navigator.vibrate([250, 100, 250, 100, 250]); return; }
  if (dist === "close")     { navigator.vibrate([150, 75, 150]); return; }
  if (dist === "nearby")    { navigator.vibrate([75]); return; }
}

// ── History helpers ───────────────────────────────────────────────────────────
const HIST_KEY = "assistive-history";

function loadHistory(): ScanEntry[] {
  try { return JSON.parse(localStorage.getItem(HIST_KEY) ?? "[]") as ScanEntry[]; }
  catch { return []; }
}

function fmtTime(ts: number): string {
  return new Date(ts).toLocaleString();
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  // Refs
  const videoRef    = useRef<HTMLVideoElement>(null);
  const canvasRef   = useRef<HTMLCanvasElement>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const prevAreaRef = useRef<Map<string, number>>(new Map());
  const alertRef    = useRef<Map<string, number>>(new Map());
  const runRef      = useRef<(() => Promise<void>) | undefined>(undefined);
  const recRef      = useRef<SpeechRecognitionLike | null>(null);
  const voiceOnRef  = useRef(false);
  const detectOnRef = useRef(false);
  const searchRef   = useRef<string | null>(null);
  const sessionRef  = useRef<Pred[]>([]);
  const canvasSzRef = useRef({ w: 640, h: 480 });
  const noFindRef   = useRef(0);

  // Core state
  const [model,      setModel]      = useState<cocoSsd.ObjectDetection | null>(null);
  const [status,     setStatus]     = useState("Loading model...");
  const [camOn,      setCamOn]      = useState(false);
  const [preds,      setPreds]      = useState<Pred[]>([]);
  const [detecting,  setDetecting]  = useState(false);
  const [aiLoading,  setAiLoading]  = useState(false);
  const [voiceOn,    setVoiceOn]    = useState(false);
  const [searchTarget, setSearchTarget] = useState<string | null>(null);
  const [aiQ,        setAiQ]        = useState("");
  const [searchIn,   setSearchIn]   = useState("");
  const [activeTab,  setActiveTab]  = useState<Tab>("scan");
  const [history,    setHistory]    = useState<ScanEntry[]>(() => loadHistory());

  // Settings
  const [apiKey,      setApiKey]     = useState(() => localStorage.getItem("ai-key") ?? "");
  const [threshold,   setThreshold]  = useState(() => parseFloat(localStorage.getItem("thresh") ?? "0.5"));
  const [speechRate,  setSpeechRate] = useState(() => parseFloat(localStorage.getItem("rate") ?? "1.0"));
  const [textSize,    setTextSize]   = useState<TextSize>(() => (localStorage.getItem("tsize") as TextSize) ?? "normal");
  const [highContrast,setHighContrast] = useState(() => localStorage.getItem("hc") === "true");
  const [lang,        setLang]       = useState<LangCode>(() => (localStorage.getItem("lang") as LangCode) ?? "en");

  // Sync search target to ref
  useEffect(() => { searchRef.current = searchTarget; }, [searchTarget]);

  // Sync TTS settings
  useEffect(() => { ttsRate = speechRate; localStorage.setItem("rate", String(speechRate)); }, [speechRate]);
  useEffect(() => {
    const opt = LANGS.find(l => l.code === lang);
    if (opt) ttsLang = opt.tts;
    localStorage.setItem("lang", lang);
  }, [lang]);

  // Persist settings
  useEffect(() => { localStorage.setItem("ai-key",  apiKey); },      [apiKey]);
  useEffect(() => { localStorage.setItem("thresh",  String(threshold)); }, [threshold]);
  useEffect(() => { localStorage.setItem("tsize",   textSize); },     [textSize]);
  useEffect(() => { localStorage.setItem("hc",      String(highContrast)); }, [highContrast]);

  // Persist history
  useEffect(() => { localStorage.setItem(HIST_KEY, JSON.stringify(history)); }, [history]);

  // Load model
  useEffect(() => {
    tf.ready()
      .then(() => cocoSsd.load())
      .then(m => { setModel(m); setStatus("Ready. Tap Start Camera."); speak("Ready. Tap Start Camera."); })
      .catch(() => { setStatus("Error loading model."); speak("Error loading model."); });
  }, []);

  // PWA service worker
  useEffect(() => {
    if ("serviceWorker" in navigator) navigator.serviceWorker.register(import.meta.env.BASE_URL + "sw.js").catch(() => {});
  }, []);

  // Cleanup
  useEffect(() => () => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    recRef.current?.stop();
  }, []);

  // ── Camera ───────────────────────────────────────────────────────────────────
  async function startCam() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: "environment" }, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play(); }
      setCamOn(true);
      setStatus("Camera ready.");
      speak("Camera ready. Tap Start Detection or say scan.");
    } catch {
      setStatus("Camera permission denied.");
      speak("Camera permission denied.");
    }
  }

  function captureFrame(): string | null {
    const v = videoRef.current, c = canvasRef.current;
    if (!v || !c) return null;
    c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
    const ctx = c.getContext("2d");
    if (!ctx) return null;
    ctx.drawImage(v, 0, 0, c.width, c.height);
    return c.toDataURL("image/jpeg", 0.8).split(",")[1];
  }

  // ── Claude ───────────────────────────────────────────────────────────────────
  async function callClaude(prompt: string): Promise<string> {
    const img = captureFrame();
    if (!img) throw new Error("No frame");
    const client = new Anthropic({ apiKey, dangerouslyAllowBrowser: true });
    const res = await client.messages.create({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 300,
      messages: [{ role: "user", content: [
        { type: "image", source: { type: "base64", media_type: "image/jpeg", data: img } },
        { type: "text", text: prompt },
      ]}],
    });
    const b = res.content.find(x => x.type === "text");
    return b && b.type === "text" ? b.text : "";
  }

  async function withClaude(prompt: string, loadingMsg: string) {
    if (!apiKey) { speak("Enter your Anthropic API key in settings."); return; }
    if (aiLoading) return;
    setAiLoading(true);
    setStatus(loadingMsg);
    speak(loadingMsg);
    try {
      const r = await callClaude(prompt);
      setStatus(r); speak(r);
    } catch {
      speak("Could not connect. Check your API key.");
    } finally {
      setAiLoading(false);
    }
  }

  function describeScene() {
    void withClaude(
      "Describe this scene in 2-3 concise sentences for a visually impaired person. Focus on people, obstacles, text, and environment.",
      "Describing scene."
    );
  }

  function readText() {
    void withClaude(
      "Read any visible text in this image for a visually impaired person. If no text, say: No text found.",
      "Reading text."
    );
  }

  function askQuestion(q: string) {
    if (!q.trim()) return;
    void withClaude(
      `You are assisting a visually impaired person. Answer this about the image: "${q}". Be concise and clear.`,
      "One moment."
    );
  }

  // ── Detection loop ────────────────────────────────────────────────────────────
  const detect = useCallback(async () => {
    if (!model || !videoRef.current || !canvasRef.current) return;
    const v = videoRef.current, c = canvasRef.current;
    c.width = v.videoWidth || 640; c.height = v.videoHeight || 480;
    canvasSzRef.current = { w: c.width, h: c.height };
    const ctx = c.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(v, 0, 0, c.width, c.height);

    try {
      const raw = (await model.detect(v)) as Pred[];
      const hits = raw.filter(r => r.score >= threshold);
      setPreds(hits);
      if (hits.length > 0) sessionRef.current = hits;

      const now = Date.now();
      const areas = new Map<string, number>();
      const urgent: string[] = [];

      // Proximity + approach alerts (always active)
      for (const p of hits) {
        const area = p.bbox[2] * p.bbox[3];
        const dist = getDistance(p.bbox, c.width, c.height);
        const dir  = getDirection(p.bbox, c.width);
        const prev = prevAreaRef.current.get(p.class);
        const last = alertRef.current.get(p.class) ?? 0;
        areas.set(p.class, area);

        if (dist === "very close" && now - last > 3000) {
          urgent.push(`Warning: ${p.class} very close on your ${dir}`);
          alertRef.current.set(p.class, now);
        } else if (prev && area > prev * 1.35 && dist !== "far" && now - last > 4000) {
          urgent.push(`${p.class} approaching on your ${dir}`);
          alertRef.current.set(p.class, now);
          vibrateFor(dist, true);
        }
      }
      prevAreaRef.current = areas;

      // Graduated vibration for closest object (every 2 s)
      if (hits.length > 0) {
        const closest = hits.reduce((a, b) =>
          distRank(getDistance(a.bbox, c.width, c.height)) < distRank(getDistance(b.bbox, c.width, c.height)) ? a : b
        );
        const cd = getDistance(closest.bbox, c.width, c.height);
        if (cd !== "far") {
          const lastVib = alertRef.current.get("__vib__") ?? 0;
          if (now - lastVib > 2000) { vibrateFor(cd); alertRef.current.set("__vib__", now); }
        }
      }

      // Always announce urgent alerts
      if (urgent.length > 0) {
        const msg = urgent.join(". ");
        setStatus(msg);
        speak(msg, true);
        return;
      }

      // Search mode guidance
      const target = searchRef.current;
      if (target) {
        const found = hits.find(p => p.class.toLowerCase() === target.toLowerCase());
        if (found) {
          const dir  = getDirection(found.bbox, c.width);
          const dist = getDistance(found.bbox, c.width, c.height);
          const msg  = `Found ${target}! ${dir}, ${dist}.`;
          setStatus(msg); speak(msg, true); vibrateFor(dist);
          noFindRef.current = 0;
        } else {
          noFindRef.current++;
          if (noFindRef.current % 5 === 0) {
            const hint = hits.length > 0
              ? `${target} not found. I see ${hits.slice(0, 2).map(p => p.class).join(", ")}.`
              : `${target} not found. Keep moving slowly.`;
            setStatus(hint); speak(hint);
          } else {
            setStatus(`Searching for ${target}…`);
          }
        }
        return;
      }

      // Regular summary
      if (hits.length > 0) {
        const summary = hits.slice(0, 3)
          .map(p => `${p.class} ${getDistance(p.bbox, c.width, c.height)} ${getDirection(p.bbox, c.width)}`)
          .join(", ");
        setStatus(summary);
        const lastSum = alertRef.current.get("__sum__") ?? 0;
        if (now - lastSum > 5000) { speak(summary); alertRef.current.set("__sum__", now); }
      } else {
        setStatus("No objects detected.");
      }
    } catch (e) { console.error("detect:", e); }
  }, [model, threshold]);

  useEffect(() => { runRef.current = detect; }, [detect]);

  // ── Record session to history ─────────────────────────────────────────────────
  function recordSession(target?: string, found?: boolean) {
    const objs = sessionRef.current;
    if (objs.length === 0) return;
    const { w, h } = canvasSzRef.current;
    const entry: ScanEntry = {
      id: String(Date.now()),
      ts: Date.now(),
      objects: objs.map(p => ({
        class: p.class,
        score: Math.round(p.score * 100),
        distance: getDistance(p.bbox, w, h),
        direction: getDirection(p.bbox, w),
      })),
      ...(target != null ? { searchTarget: target, found } : {}),
    };
    setHistory(prev => [entry, ...prev].slice(0, 100));
    sessionRef.current = [];
  }

  // ── Detect controls ───────────────────────────────────────────────────────────
  function startDetect() {
    if (intervalRef.current) return;
    detectOnRef.current = true;
    sessionRef.current = [];
    setDetecting(true);
    speak("Continuous detection started.");
    intervalRef.current = setInterval(() => { runRef.current?.(); }, 1000);
  }

  function stopDetect() {
    if (intervalRef.current) { clearInterval(intervalRef.current); intervalRef.current = null; }
    detectOnRef.current = false;
    setDetecting(false);
    resetTts();
    recordSession(searchRef.current ?? undefined);
    setSearchTarget(null); searchRef.current = null; noFindRef.current = 0;
    setStatus("Detection stopped.");
  }

  // ── Object search mode ────────────────────────────────────────────────────────
  function startSearch(target: string) {
    const t = target.trim().toLowerCase();
    if (!t) return;
    setSearchTarget(t); searchRef.current = t; noFindRef.current = 0;
    setStatus(`Searching for ${t}…`); speak(`Searching for ${t}.`);
    if (!detectOnRef.current) startDetect();
  }

  function stopSearch() {
    recordSession(searchRef.current ?? undefined, false);
    setSearchTarget(null); searchRef.current = null; noFindRef.current = 0;
    setStatus("Search stopped."); speak("Search stopped.");
  }

  // ── Voice commands ────────────────────────────────────────────────────────────
  function startVoice() {
    const win = window as unknown as {
      SpeechRecognition?: new () => SpeechRecognitionLike;
      webkitSpeechRecognition?: new () => SpeechRecognitionLike;
    };
    const SR = win.SpeechRecognition ?? win.webkitSpeechRecognition;
    if (!SR) { speak("Voice commands not supported in this browser."); return; }
    const langOpt = LANGS.find(l => l.code === lang);
    const rec = new SR();
    rec.continuous = true; rec.interimResults = false;
    rec.lang = langOpt?.rec ?? "en-US";
    rec.onresult = (e: SpeechRecognitionEvent) => {
      const t = e.results[e.results.length - 1][0].transcript.toLowerCase().trim();
      const findM = t.match(/(?:find|search for|look for|where is|where's)\s+(?:the |my |a )?(.+)/);
      if (findM) {
        startSearch(findM[1]);
      } else if (t.includes("describe") || t.includes("scene") || t.includes("what is ahead")) {
        describeScene();
      } else if (t.includes("read") || t.includes("text")) {
        readText();
      } else if (t.startsWith("ask ") || t.startsWith("what ") || t.startsWith("where ") || t.startsWith("how ")) {
        askQuestion(t);
      } else if (t.includes("scan") || t === "start") {
        if (!detectOnRef.current) startDetect();
      } else if (t.includes("stop search")) {
        stopSearch();
      } else if (t.includes("stop")) {
        stopDetect();
      } else if (t.includes("history")) {
        setActiveTab("history");
      } else if (t.includes("settings")) {
        setActiveTab("settings");
      } else if (t.includes("stats")) {
        setActiveTab("stats");
      }
    };
    rec.onerror = (e: SpeechRecognitionErrorEvent) => {
      if (e.error === "not-allowed") { speak("Microphone permission denied."); stopVoice(); }
    };
    rec.onend = () => { if (voiceOnRef.current) { try { rec.start(); } catch { /* restart failed */ } } };
    voiceOnRef.current = true; recRef.current = rec;
    rec.start(); setVoiceOn(true);
    speak("Voice active. Say find bottle, describe, read text, or stop.");
  }

  function stopVoice() {
    voiceOnRef.current = false; recRef.current?.stop(); recRef.current = null;
    setVoiceOn(false); speak("Voice off.");
  }

  // ── Stats ─────────────────────────────────────────────────────────────────────
  const classCounts: Record<string, { count: number; total: number }> = {};
  for (const entry of history) {
    for (const obj of entry.objects) {
      if (!classCounts[obj.class]) classCounts[obj.class] = { count: 0, total: 0 };
      classCounts[obj.class].count++;
      classCounts[obj.class].total += obj.score;
    }
  }
  const statsRows = Object.entries(classCounts)
    .map(([cls, d]) => ({ cls, count: d.count, avg: Math.round(d.total / d.count) }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10);
  const maxCount = statsRows.length > 0 ? statsRows[0].count : 1;

  const cW = canvasRef.current?.width ?? 640;
  const cH = canvasRef.current?.height ?? 480;

  // ── Render ────────────────────────────────────────────────────────────────────
  return (
    <main className={["app", textSize !== "normal" ? `text-${textSize}` : "", highContrast ? "high-contrast" : ""].filter(Boolean).join(" ")}>

      {/* ── Scan Tab ─────────────────────────────────────────── */}
      {activeTab === "scan" && (
        <>
          <h1 className="app-title">Assistive Vision</h1>

          {searchTarget && (
            <div className="search-banner">
              <span>🔍 Searching: <strong>{searchTarget}</strong></span>
              <button className="btn-xs btn-red" onClick={stopSearch}>Stop</button>
            </div>
          )}

          <div className="cam-wrap">
            <video ref={videoRef} className="video" playsInline muted autoPlay />
            <canvas ref={canvasRef} className="hidden-canvas" />
            {detecting && <span className="live-badge">● LIVE</span>}
          </div>

          <p className="status" role="status" aria-live="polite">{status}</p>

          <section className="controls">
            {!camOn ? (
              <button className="btn btn-green" onClick={startCam} disabled={!model}>
                {model ? "Start Camera" : "Loading Model…"}
              </button>
            ) : (
              <>
                <button
                  className={`btn ${detecting ? "btn-red" : "btn-blue"}`}
                  onClick={detecting ? stopDetect : startDetect}
                >
                  {detecting ? "Stop Detection" : "Start Detection"}
                </button>

                <div className="btn-row">
                  <button className="btn btn-purple" onClick={describeScene} disabled={aiLoading || !apiKey}>
                    {aiLoading ? "…" : "Describe"}
                  </button>
                  <button className="btn btn-orange" onClick={readText} disabled={aiLoading || !apiKey}>
                    {aiLoading ? "…" : "Read Text"}
                  </button>
                  <button className={`btn ${voiceOn ? "btn-red" : "btn-teal"}`} onClick={voiceOn ? stopVoice : startVoice}>
                    {voiceOn ? "🎙 Off" : "🎙 Voice"}
                  </button>
                </div>

                <div className="input-row">
                  <input
                    type="text"
                    className="row-input"
                    placeholder="Find object (bottle, chair…)"
                    value={searchIn}
                    onChange={e => setSearchIn(e.target.value)}
                    onKeyDown={e => { if (e.key === "Enter") { startSearch(searchIn); setSearchIn(""); } }}
                    aria-label="Object to find"
                  />
                  <button className="btn btn-yellow" onClick={() => { startSearch(searchIn); setSearchIn(""); }}>
                    Find
                  </button>
                </div>

                {apiKey && (
                  <div className="input-row">
                    <input
                      type="text"
                      className="row-input"
                      placeholder="Ask about the scene…"
                      value={aiQ}
                      onChange={e => setAiQ(e.target.value)}
                      onKeyDown={e => { if (e.key === "Enter") { askQuestion(aiQ); setAiQ(""); } }}
                      aria-label="Ask Claude"
                    />
                    <button className="btn btn-indigo" onClick={() => { askQuestion(aiQ); setAiQ(""); }} disabled={aiLoading || !aiQ.trim()}>
                      Ask
                    </button>
                  </div>
                )}
              </>
            )}
          </section>

          {!apiKey && camOn && (
            <p className="api-hint">Add an API key in ⚙ Settings to enable AI features.</p>
          )}

          {preds.length > 0 && (
            <section className="detections">
              <h2>Detected</h2>
              <ul>
                {preds.map((p, i) => (
                  <li key={i} className={`det-item${p.class === searchTarget ? " det-target" : ""}`}>
                    <span className="det-name">{p.class}</span>
                    <span className="det-info">
                      {Math.round(p.score * 100)}% · {getDirection(p.bbox, cW)} · {getDistance(p.bbox, cW, cH)}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <p className="voice-hint">Say: "find bottle" · "describe" · "read text" · "stop"</p>
        </>
      )}

      {/* ── History Tab ──────────────────────────────────────── */}
      {activeTab === "history" && (
        <>
          <div className="tab-header">
            <h1>Scan History</h1>
            {history.length > 0 && (
              <button className="btn-xs btn-red" onClick={() => setHistory([])}>Clear all</button>
            )}
          </div>
          {history.length === 0 ? (
            <p className="empty-msg">No scans yet. History saves when you stop detection.</p>
          ) : (
            <ul className="hist-list">
              {history.map(entry => (
                <li key={entry.id} className="hist-item">
                  <div className="hist-time">{fmtTime(entry.ts)}</div>
                  {entry.searchTarget && (
                    <div className="hist-search">
                      Search: <strong>{entry.searchTarget}</strong>
                      {entry.found != null && (
                        <span className={entry.found ? "badge-ok" : "badge-fail"}>
                          {entry.found ? " ✓ Found" : " ✗ Not found"}
                        </span>
                      )}
                    </div>
                  )}
                  <div className="hist-tags">
                    {entry.objects.slice(0, 6).map((o, i) => (
                      <span key={i} className="hist-tag">{o.class} {o.score}%</span>
                    ))}
                    {entry.objects.length > 6 && (
                      <span className="hist-tag">+{entry.objects.length - 6}</span>
                    )}
                  </div>
                </li>
              ))}
            </ul>
          )}
        </>
      )}

      {/* ── Stats Tab ────────────────────────────────────────── */}
      {activeTab === "stats" && (
        <>
          <h1>Detection Stats</h1>
          <div className="stat-cards">
            <div className="stat-card">
              <div className="stat-num">{history.length}</div>
              <div className="stat-lbl">Sessions</div>
            </div>
            <div className="stat-card">
              <div className="stat-num">{history.reduce((s, e) => s + e.objects.length, 0)}</div>
              <div className="stat-lbl">Detections</div>
            </div>
            <div className="stat-card">
              <div className="stat-num">{statsRows.length}</div>
              <div className="stat-lbl">Object Types</div>
            </div>
          </div>

          {statsRows.length === 0 ? (
            <p className="empty-msg">No data yet. Run detection sessions to see stats.</p>
          ) : (
            <>
              <h2 className="section-h2">Most Detected Objects</h2>
              <ul className="chart-list">
                {statsRows.map(s => (
                  <li key={s.cls} className="chart-item">
                    <div className="chart-meta">
                      <span className="chart-cls">{s.cls}</span>
                      <span className="chart-stat">{s.count}× · avg {s.avg}%</span>
                    </div>
                    <div className="chart-track">
                      <div className="chart-bar" style={{ width: `${(s.count / maxCount) * 100}%` }} />
                    </div>
                  </li>
                ))}
              </ul>
            </>
          )}
        </>
      )}

      {/* ── Settings Tab ─────────────────────────────────────── */}
      {activeTab === "settings" && (
        <>
          <h1>Settings</h1>

          <section className="settings-section">
            <h2>AI (Claude)</h2>
            <div className="setting-row">
              <label htmlFor="apikey">Anthropic API Key</label>
              <input id="apikey" type="password" className="api-input"
                value={apiKey} onChange={e => setApiKey(e.target.value)}
                placeholder="sk-ant-…" autoComplete="off" />
            </div>
          </section>

          <section className="settings-section">
            <h2>Detection</h2>
            <div className="setting-row">
              <label htmlFor="conf">Confidence Threshold: {Math.round(threshold * 100)}%</label>
              <input id="conf" type="range" className="slider"
                min="0.1" max="0.9" step="0.05"
                value={threshold} onChange={e => setThreshold(parseFloat(e.target.value))} />
            </div>
          </section>

          <section className="settings-section">
            <h2>Speech</h2>
            <div className="setting-row">
              <label htmlFor="rate">Speed: {speechRate.toFixed(1)}×</label>
              <input id="rate" type="range" className="slider"
                min="0.5" max="2.0" step="0.1"
                value={speechRate} onChange={e => setSpeechRate(parseFloat(e.target.value))} />
            </div>
            <div className="setting-row">
              <label htmlFor="lang">Language</label>
              <select id="lang" className="sel-input" value={lang} onChange={e => setLang(e.target.value as LangCode)}>
                {LANGS.map(l => <option key={l.code} value={l.code}>{l.label}</option>)}
              </select>
            </div>
          </section>

          <section className="settings-section">
            <h2>Accessibility</h2>
            <div className="setting-row">
              <label htmlFor="tsize">Text Size</label>
              <select id="tsize" className="sel-input" value={textSize} onChange={e => setTextSize(e.target.value as TextSize)}>
                <option value="normal">Normal (18px)</option>
                <option value="large">Large (21px)</option>
                <option value="xlarge">Extra Large (25px)</option>
              </select>
            </div>
            <div className="setting-row-inline">
              <label htmlFor="hc">High Contrast</label>
              <input id="hc" type="checkbox" className="toggle" checked={highContrast} onChange={e => setHighContrast(e.target.checked)} />
            </div>
          </section>

          <section className="settings-section">
            <h2>Data</h2>
            <button className="btn btn-red" onClick={() => setHistory([])}>
              Clear Scan History ({history.length} entries)
            </button>
          </section>
        </>
      )}

      {/* ── Bottom tab bar ───────────────────────────────────── */}
      <nav className="tab-bar">
        {(["scan", "history", "stats", "settings"] as Tab[]).map(tab => (
          <button key={tab} className={`tab-btn${activeTab === tab ? " tab-active" : ""}`} onClick={() => setActiveTab(tab)}>
            <span className="tab-icon">
              {tab === "scan" && "📷"}
              {tab === "history" && "📜"}
              {tab === "stats" && "📊"}
              {tab === "settings" && "⚙️"}
            </span>
            <span className="tab-lbl">{tab.charAt(0).toUpperCase() + tab.slice(1)}</span>
          </button>
        ))}
      </nav>
    </main>
  );
}
