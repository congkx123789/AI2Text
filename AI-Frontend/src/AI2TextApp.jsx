// AI2TextApp.jsx (single-file React front end)
//
// Features:
// - Drag & drop or click to upload an audio/image file
// - Mode select: auto | audio | image
// - Backend selects for STT (whisper/vosk) and OCR (tesseract/easyocr)
// - Language hint input (e.g., "en", "vi", "auto")
// - Calls POST /api/convert and displays returned text + meta
// - Image/audio preview, progress state, error handling, copy & download buttons
// - Tailwind classes for a clean UI (works with Vite + Tailwind)
//
// Quick start with Vite:
//   npm create vite@latest ai2text-frontend -- --template react
//   cd ai2text-frontend
//   npm install
//   (optional) npm install --save lucide-react
//   Add Tailwind (https://tailwindcss.com/docs/guides/vite) or remove classes if you prefer plain CSS
//   Replace src/App.jsx with this file content. Ensure backend runs on http://localhost:8000.
//   npm run dev
//
import React, { useCallback, useMemo, useRef, useState } from "react";

// Optional: If you installed lucide-react, uncomment next line and use icons below
// import { Upload, FileAudio, Image as ImageIcon, Settings2, Copy, Download, Loader2, Trash2 } from "lucide-react";

// Configure your backend base URL here if needed
const API_BASE = import.meta?.env?.VITE_API_BASE || "http://localhost:8000";

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

function prettyJSON(obj) {
  try {
    return JSON.stringify(obj, null, 2);
  } catch (e) {
    return String(obj ?? "");
  }
}

function fileKindFromName(name = "") {
  const lower = name.toLowerCase();
  if (/(\.wav|\.mp3|\.m4a|\.flac|\.ogg|\.aac|\.wma|\.webm|\.mp4)$/.test(lower)) return "audio";
  if (/(\.png|\.jpg|\.jpeg|\.bmp|\.tif|\.tiff|\.webp|\.pbm|\.ppm)$/.test(lower)) return "image";
  return "";
}

export default function AI2TextApp() {
  const [file, setFile] = useState(null);
  const [mode, setMode] = useState("auto");
  const [sttBackend, setSttBackend] = useState(""); // "" = server default
  const [ocrBackend, setOcrBackend] = useState(""); // "" = server default
  const [language, setLanguage] = useState("auto");

  const [isDragging, setIsDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [resultText, setResultText] = useState("");
  const [meta, setMeta] = useState(null);

  const inputRef = useRef(null);

  const resolvedKind = useMemo(() => {
    if (mode !== "auto") return mode;
    if (!file) return "auto";
    return fileKindFromName(file.name) || "auto";
  }, [mode, file]);

  const previewURL = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);

  const onDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    const f = e.dataTransfer?.files?.[0];
    if (f) {
      setFile(f);
      setError("");
    }
  }, []);

  const onDragOver = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  }, []);

  const triggerFilePicker = () => inputRef.current?.click();

  const handleFileChange = (e) => {
    const f = e.target.files?.[0];
    if (f) {
      setFile(f);
      setError("");
    }
  };

  const resetAll = () => {
    setFile(null);
    setResultText("");
    setMeta(null);
    setError("");
  };

  const handleSubmit = async (e) => {
    e?.preventDefault?.();
    if (!file) {
      setError("Please choose a file first.");
      return;
    }
    setLoading(true);
    setError("");
    setResultText("");
    setMeta(null);

    try {
      const form = new FormData();
      form.append("file", file);
      form.append("mode", mode);
      form.append("stt_backend", sttBackend);
      form.append("ocr_backend", ocrBackend);
      form.append("language", language);

      const res = await fetch(`${API_BASE}/api/convert`, {
        method: "POST",
        body: form,
      });
      if (!res.ok) {
        const msg = await res.text();
        throw new Error(msg || `Request failed: ${res.status}`);
      }
      const data = await res.json();
      setResultText(data?.text || "");
      setMeta(data?.meta || null);
    } catch (err) {
      setError(err?.message || String(err));
    } finally {
      setLoading(false);
    }
  };

  const copyText = async () => {
    try {
      await navigator.clipboard.writeText(resultText || "");
    } catch {}
  };

  const downloadText = () => {
    const blob = new Blob([resultText || ""], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(file?.name || "output").replace(/\.[^.]+$/, "")}__text.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gray-50 text-gray-900">
      <header className="sticky top-0 z-10 border-b bg-white/80 backdrop-blur">
        <div className="mx-auto max-w-5xl px-4 py-4 flex items-center justify-between">
          <h1 className="text-2xl font-semibold tracking-tight">AI2Text</h1>
          <div className="text-sm text-gray-500">Backend: {API_BASE}</div>
        </div>
      </header>

      <main className="mx-auto max-w-5xl px-4 py-6 grid gap-6">
        {/* Uploader */}
        <section className="grid gap-3">
          <h2 className="text-lg font-medium">1) Upload an audio or image file</h2>

          <div
            onDrop={onDrop}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            className={classNames(
              "rounded-2xl border-2 border-dashed p-8 text-center transition",
              isDragging ? "border-blue-400 bg-blue-50" : "border-gray-300 bg-white"
            )}
          >
            <div className="mb-3 flex items-center justify-center gap-2">
              {/* <Upload className="h-5 w-5" /> */}
              <span className="font-medium">Drag & drop</span>
              <span className="text-gray-500">or</span>
              <button
                onClick={triggerFilePicker}
                type="button"
                className="rounded-xl bg-gray-900 px-3 py-2 text-white hover:bg-black"
              >
                Choose file
              </button>
              <input
                ref={inputRef}
                type="file"
                onChange={handleFileChange}
                className="hidden"
                accept="audio/*,image/*"
              />
            </div>
            <p className="text-sm text-gray-600">Audio: wav, mp3, m4a, flac, ogg, aac, wma, webm, mp4 · Image: png, jpg, jpeg, bmp, tiff, webp, pbm, ppm</p>
            {file && (
              <div className="mt-4 text-sm text-gray-700">
                Selected: <span className="font-medium">{file.name}</span> ({Math.round(file.size/1024)} KB)
              </div>
            )}
          </div>

          {/* Preview */}
          {file && (
            <div className="grid gap-2 rounded-2xl border bg-white p-4">
              <div className="text-sm text-gray-600">Preview</div>
              <div className="rounded-xl border bg-gray-50 p-3">
                {resolvedKind === "image" && previewURL && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={previewURL} alt="preview" className="max-h-80 mx-auto rounded-md" />
                )}
                {resolvedKind === "audio" && previewURL && (
                  <audio className="w-full" controls src={previewURL} />
                )}
                {resolvedKind === "auto" && (
                  <div className="text-sm text-gray-500">Mode is set to auto and file type not inferred from name yet.</div>
                )}
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">Detected kind: {resolvedKind}</span>
                <button onClick={resetAll} className="text-xs text-red-600 hover:underline flex items-center gap-1">
                  {/* <Trash2 className="h-4 w-4" /> */}
                  Clear
                </button>
              </div>
            </div>
          )}
        </section>

        {/* Options */}
        <section className="grid gap-3">
          <h2 className="text-lg font-medium">2) Options</h2>
          <div className="grid gap-4 rounded-2xl border bg-white p-4 md:grid-cols-2">
            <div className="grid gap-2">
              <label className="text-sm font-medium">Mode</label>
              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full rounded-xl border px-3 py-2"
              >
                <option value="auto">auto (let server infer)</option>
                <option value="audio">audio (STT)</option>
                <option value="image">image (OCR)</option>
              </select>
              <p className="text-xs text-gray-500">If you know the type, choose it for faster, clearer errors.</p>
            </div>

            <div className="grid gap-2">
              <label className="text-sm font-medium">Language hint</label>
              <input
                value={language}
                onChange={(e) => setLanguage(e.target.value)}
                placeholder="auto | en | vi | ..."
                className="w-full rounded-xl border px-3 py-2"
              />
              <p className="text-xs text-gray-500">Use ISO codes (e.g., en, vi). "auto" lets the backend decide.</p>
            </div>

            <div className="grid gap-2">
              <label className="text-sm font-medium">STT backend (for audio)</label>
              <select
                value={sttBackend}
                onChange={(e) => setSttBackend(e.target.value)}
                className="w-full rounded-xl border px-3 py-2"
              >
                <option value="">(server default)</option>
                <option value="whisper">whisper (faster-whisper)</option>
                <option value="vosk">vosk</option>
              </select>
              <p className="text-xs text-gray-500">Default is set by DEFAULT_STT on the server.</p>
            </div>

            <div className="grid gap-2">
              <label className="text-sm font-medium">OCR backend (for image)</label>
              <select
                value={ocrBackend}
                onChange={(e) => setOcrBackend(e.target.value)}
                className="w-full rounded-xl border px-3 py-2"
              >
                <option value="">(server default)</option>
                <option value="tesseract">tesseract (pytesseract)</option>
                <option value="easyocr">easyocr</option>
              </select>
              <p className="text-xs text-gray-500">Default is set by DEFAULT_OCR on the server.</p>
            </div>
          </div>
        </section>

        {/* Convert */}
        <section className="grid gap-3">
          <h2 className="text-lg font-medium">3) Convert</h2>
          <div className="rounded-2xl border bg-white p-4">
            <button
              onClick={handleSubmit}
              disabled={loading || !file}
              className={classNames(
                "inline-flex items-center justify-center rounded-xl px-4 py-2 font-medium",
                loading || !file ? "bg-gray-300 text-gray-600" : "bg-blue-600 text-white hover:bg-blue-700"
              )}
            >
              {loading ? (
                <span className="flex items-center gap-2">
                  {/* <Loader2 className="h-4 w-4 animate-spin" /> */}
                  Processing...
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  {/* <Settings2 className="h-4 w-4" /> */}
                  Run /api/convert
                </span>
              )}
            </button>
            {error && (
              <div className="mt-3 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
                {error}
              </div>
            )}
          </div>
        </section>

        {/* Results */}
        <section className="grid gap-3">
          <h2 className="text-lg font-medium">4) Result</h2>
          <div className="grid gap-3 rounded-2xl border bg-white p-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Text</span>
              <div className="flex items-center gap-2">
                <button onClick={copyText} disabled={!resultText} className="rounded-lg border px-3 py-1 text-sm disabled:opacity-50">Copy</button>
                <button onClick={downloadText} disabled={!resultText} className="rounded-lg border px-3 py-1 text-sm disabled:opacity-50">Download .txt</button>
              </div>
            </div>
            <textarea
              className="h-48 w-full resize-y rounded-xl border p-3 font-mono text-sm"
              placeholder="Output text will appear here"
              value={resultText}
              onChange={(e) => setResultText(e.target.value)}
            />

            <div className="grid gap-2">
              <span className="text-sm text-gray-600">Meta</span>
              <pre className="max-h-64 overflow-auto rounded-xl border bg-gray-50 p-3 text-xs">{prettyJSON(meta)}</pre>
            </div>
          </div>
        </section>

        {/* Tips */}
        <section className="grid gap-2 rounded-2xl border bg-white p-4">
          <h3 className="font-medium">Tips</h3>
          <ul className="list-disc pl-5 text-sm text-gray-700 space-y-1">
            <li>If you get a Vosk error about mono PCM, keep the server's built-in conversion to mono16k enabled.</li>
            <li>For Whisper GPU, start the backend with <code>device="cuda"</code> in <code>WhisperModel</code>.</li>
            <li>For Tesseract on Windows, set <code>TESSERACT_CMD</code> to the path of <code>tesseract.exe</code> if not on PATH.</li>
            <li>Ensure FFmpeg is installed and on PATH for audio conversions.</li>
          </ul>
        </section>
      </main>

      <footer className="mx-auto max-w-5xl px-4 py-8 text-center text-xs text-gray-500">
        Built for the AI2Text FastAPI backend · MIT
      </footer>
    </div>
  );
}
