"use client"

import { useState, useEffect, useRef, useCallback } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Camera, Upload, Video, Play, Square, Download, Cpu, ZapIcon, Eye,
  RefreshCw, AlertCircle, Crosshair, Wand2, X, ChevronDown, ChevronUp,
} from "lucide-react"

// ── Interfaces ────────────────────────────────────────────────────────────────

interface ModelInfo {
  id: string; name: string; path: string; model_type: string; status: string
}

interface CatalogModel {
  id: string; name: string; type: string
  status: "loaded" | "downloading" | "downloaded" | "not_downloaded"
  download_progress: number
  error?: string
  supports_text: boolean
  supports_point: boolean
}

interface Detection {
  class_name: string; confidence: number
  bbox?: [number, number, number, number]
  mask_points?: [number, number][]
  track_id?: number
}

interface VideoJob {
  id: string; status: string; progress: number; message: string; error?: string
}

interface InferenceViewProps { apiUrl?: string }

const TASK_OPTIONS = [
  { value: "detect",  label: "Object Detection" },
  { value: "segment", label: "Segmentation" },
  { value: "pose",    label: "Pose / Keypoints" },
  { value: "track",   label: "Multi-Object Tracking" },
]

const TRACKER_OPTIONS = [
  { value: "bytetrack",  label: "ByteTrack (fast, no ReID)" },
  { value: "botsort",    label: "BoTSORT + OSNet ReID" },
  { value: "deepocsort", label: "DeepOCSORT + ReID" },
  { value: "strongsort", label: "StrongSORT + ReID" },
  { value: "ocsort",     label: "OCSORT (no ReID)" },
]

const GROUP_LABELS: Record<string, string> = {
  sam: "SAM v1", sam2: "SAM 2 / 2.1", sam3: "SAM 3",
  yoloworld: "YOLO-World", groundingdino: "GroundingDINO", owlvit: "OWL-ViT",
}

const STATUS_COLORS: Record<string, string> = {
  loaded:         "text-green-400 border-green-400/40",
  downloading:    "text-blue-400  border-blue-400/40",
  downloaded:     "text-yellow-400 border-yellow-400/40",
  not_downloaded: "text-muted-foreground border-border/40",
}

// ── Component ─────────────────────────────────────────────────────────────────

export function InferenceView({ apiUrl = "http://localhost:8000" }: InferenceViewProps) {

  // ── Model source ─────────────────────────────────────────────────────────────
  const [modelSource, setModelSource] = useState<"yolo" | "smart">("yolo")

  // ── YOLO state ───────────────────────────────────────────────────────────────
  const [yoloModels, setYoloModels]       = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState("")
  const [manualPath, setManualPath]       = useState("")
  const [task, setTask]                   = useState("detect")
  const [tracker, setTracker]             = useState("bytetrack")
  const [iou, setIou]                     = useState(0.45)
  const [imgsz, setImgsz]                 = useState(640)

  // ── Smart model state ─────────────────────────────────────────────────────────
  const [catalog, setCatalog]           = useState<CatalogModel[]>([])
  const [selSmartId, setSelSmartId]     = useState("")   // selected from loaded models
  const [showLibrary, setShowLibrary]   = useState(false)
  const [catalogLoading, setCatalogLoading] = useState(false)
  const [textPrompt, setTextPrompt]     = useState("")
  const [promptPoint, setPromptPoint]   = useState<{ x: number; y: number } | null>(null)
  const [smartImgsz, setSmartImgsz]     = useState(640)
  const catalogPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Shared ────────────────────────────────────────────────────────────────────
  const [confidence, setConfidence] = useState(0.25)

  // Ref always holds live params for webcam interval
  const activeModelPath = manualPath.trim() || selectedModel
  const paramsRef = useRef({
    modelSource, task, confidence, iou, imgsz,
    textPrompt, promptPoint, selSmartId, activeModelPath, tracker, smartImgsz,
  })
  useEffect(() => {
    paramsRef.current = {
      modelSource, task, confidence, iou, imgsz,
      textPrompt, promptPoint, selSmartId, activeModelPath, tracker, smartImgsz,
    }
  })

  // ── Source tab ────────────────────────────────────────────────────────────────
  const [sourceTab, setSourceTab] = useState<"image" | "video" | "webcam">("image")

  // ── Image state ───────────────────────────────────────────────────────────────
  const [imageFile, setImageFile]       = useState<File | null>(null)
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [inferring, setInferring]       = useState(false)
  const [resultImg, setResultImg]       = useState<string | null>(null)
  const [detections, setDetections]     = useState<Detection[]>([])
  const [error, setError]               = useState<string | null>(null)

  // ── Video state ───────────────────────────────────────────────────────────────
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoJob, setVideoJob]   = useState<VideoJob | null>(null)
  const videoPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // ── Webcam state ──────────────────────────────────────────────────────────────
  const [webcamActive, setWebcamActive] = useState(false)
  const [fps, setFps]                   = useState(0)
  const [sessionId, setSessionId]       = useState("")
  const videoRef          = useRef<HTMLVideoElement>(null)
  const canvasRef         = useRef<HTMLCanvasElement>(null)
  const streamRef         = useRef<MediaStream | null>(null)
  const webcamIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const fpsCounterRef     = useRef<number[]>([])
  const inFlightRef       = useRef(false)   // frame-dropout: skip when previous still running

  // ── Derived ───────────────────────────────────────────────────────────────────
  const loadedModels  = catalog.filter(m => m.status === "loaded")
  const activeSmartModel = catalog.find(m => m.id === selSmartId) ?? null
  const anyDownloading   = catalog.some(m => m.status === "downloading")

  // ── Load YOLO models ──────────────────────────────────────────────────────────
  useEffect(() => {
    fetch(`${apiUrl}/api/infer/models`)
      .then(r => r.json())
      .then(d => setYoloModels(d.models || []))
      .catch(() => {})
  }, [apiUrl])

  // ── Load smart catalog ────────────────────────────────────────────────────────
  async function refreshCatalog(quiet = false) {
    if (!quiet) setCatalogLoading(true)
    try {
      const res = await fetch(`${apiUrl}/api/infer/smart/catalog`)
      const d   = await res.json()
      setCatalog(d.models || [])
    } catch {}
    finally { if (!quiet) setCatalogLoading(false) }
  }

  useEffect(() => { refreshCatalog() }, [apiUrl])

  // Poll while anything is downloading
  useEffect(() => {
    if (anyDownloading) {
      catalogPollRef.current = setInterval(() => refreshCatalog(true), 2000)
    } else {
      if (catalogPollRef.current) { clearInterval(catalogPollRef.current); catalogPollRef.current = null }
    }
    return () => { if (catalogPollRef.current) clearInterval(catalogPollRef.current) }
  }, [anyDownloading])

  // Auto-select first loaded smart model when catalog loads
  useEffect(() => {
    if (!selSmartId && loadedModels.length > 0) setSelSmartId(loadedModels[0].id)
  }, [loadedModels.length])

  // ── Load / download a smart model ─────────────────────────────────────────────
  async function loadSmartModel(m: CatalogModel) {
    if (m.status === "loaded" || m.status === "downloading") return
    const fd = new FormData()
    fd.append("model_id", m.id)
    fd.append("model_type", m.type)
    await fetch(`${apiUrl}/api/infer/smart/load`, { method: "POST", body: fd })
    refreshCatalog()
    setShowLibrary(true)
  }

  // ── Image inference ───────────────────────────────────────────────────────────
  function handleImageFile(f: File) {
    setImageFile(f); setResultImg(null); setDetections([]); setError(null)
    setPromptPoint(null); setImagePreview(URL.createObjectURL(f))
  }

  async function runImageInference() {
    if (!imageFile) return
    if (modelSource === "yolo" && !activeModelPath) return
    if (modelSource === "smart" && !selSmartId) return
    setInferring(true); setError(null)
    try {
      const fd = new FormData()
      fd.append("image", imageFile)
      let url: string
      if (modelSource === "yolo") {
        fd.append("model_path", activeModelPath)
        fd.append("task", task)
        fd.append("confidence", String(confidence))
        fd.append("iou", String(iou))
        fd.append("imgsz", String(imgsz))
        url = `${apiUrl}/api/infer/image`
      } else {
        fd.append("model_id", selSmartId)
        fd.append("confidence", String(confidence))
        if (textPrompt.trim()) fd.append("text_prompt", textPrompt.trim())
        if (promptPoint) { fd.append("point_x", String(promptPoint.x)); fd.append("point_y", String(promptPoint.y)) }
        url = `${apiUrl}/api/infer/smart/image`
      }
      const res = await fetch(url, { method: "POST", body: fd })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
      const data = await res.json()
      setResultImg(data.annotated_b64); setDetections(data.detections || [])
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally { setInferring(false) }
  }

  function downloadResult() {
    if (!resultImg) return
    const a = document.createElement("a")
    a.href = `data:image/jpeg;base64,${resultImg}`; a.download = "result.jpg"; a.click()
  }

  function handleInputImageClick(e: React.MouseEvent<HTMLImageElement>) {
    if (modelSource !== "smart" || !activeSmartModel?.supports_point) return
    const rect = e.currentTarget.getBoundingClientRect()
    setPromptPoint({ x: (e.clientX - rect.left) / rect.width, y: (e.clientY - rect.top) / rect.height })
  }

  // ── Video inference ────────────────────────────────────────────────────────────
  async function startVideoInference() {
    if (!videoFile || !activeModelPath) return
    setError(null)
    const fd = new FormData()
    fd.append("model_path", activeModelPath); fd.append("task", task)
    fd.append("confidence", String(confidence)); fd.append("iou", String(iou))
    fd.append("imgsz", String(imgsz)); fd.append("tracker", tracker)
    fd.append("video", videoFile)
    try {
      const res = await fetch(`${apiUrl}/api/infer/video`, { method: "POST", body: fd })
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText)
      const { job_id } = await res.json()
      setVideoJob({ id: job_id, status: "starting", progress: 0, message: "Starting…" })
      videoPollRef.current = setInterval(() => pollVideoJob(job_id), 1500)
    } catch (e) { setError(e instanceof Error ? e.message : String(e)) }
  }

  const pollVideoJob = useCallback(async (job_id: string) => {
    try {
      const res  = await fetch(`${apiUrl}/api/infer/video/${job_id}/status`)
      const data: VideoJob = await res.json()
      setVideoJob(data)
      if (data.status === "completed" || data.status === "failed") {
        if (videoPollRef.current) { clearInterval(videoPollRef.current); videoPollRef.current = null }
      }
    } catch {}
  }, [apiUrl])

  // ── Webcam (with frame dropout) ────────────────────────────────────────────────
  async function startWebcam() {
    const p = paramsRef.current
    if (p.modelSource === "yolo" && !p.activeModelPath)  { setError("Select a model first"); return }
    if (p.modelSource === "smart" && !p.selSmartId)      { setError("Select a smart model first"); return }
    setError(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
      streamRef.current = stream
      if (videoRef.current) { videoRef.current.srcObject = stream; await videoRef.current.play() }
      setWebcamActive(true)
      const sid = sessionId || str8(); setSessionId(sid)
      inFlightRef.current = false

      webcamIntervalRef.current = setInterval(async () => {
        // Frame dropout: skip this tick if previous request is still in flight
        if (inFlightRef.current) return
        if (!videoRef.current || !canvasRef.current) return
        const v = videoRef.current
        if (v.videoWidth === 0) return

        const cp = paramsRef.current
        const ctx = canvasRef.current.getContext("2d")!
        // For smart models, downscale frame before encoding to reduce latency
        let cw = v.videoWidth, ch = v.videoHeight
        if (cp.modelSource === "smart" && cp.smartImgsz < Math.max(cw, ch)) {
          const scale = cp.smartImgsz / Math.max(cw, ch)
          cw = Math.round(cw * scale); ch = Math.round(ch * scale)
        }
        canvasRef.current.width = cw; canvasRef.current.height = ch
        ctx.drawImage(v, 0, 0, cw, ch)
        const frame = canvasRef.current.toDataURL("image/jpeg", 0.7).split(",")[1]

        inFlightRef.current = true
        try {
          let res: Response
          if (cp.modelSource === "yolo") {
            res = await fetch(`${apiUrl}/api/infer/frame`, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_path: cp.activeModelPath, frame_b64: frame,
                task: cp.task, confidence: cp.confidence, iou: cp.iou, imgsz: cp.imgsz,
                tracker: cp.tracker, session_id: sid,
              }),
            })
          } else {
            res = await fetch(`${apiUrl}/api/infer/smart/frame`, {
              method: "POST", headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_id: cp.selSmartId, frame_b64: frame,
                confidence: cp.confidence,
                text_prompt: cp.textPrompt || null,
                point_x: cp.promptPoint?.x ?? null,
                point_y: cp.promptPoint?.y ?? null,
              }),
            })
          }
          if (!res.ok) return
          const data = await res.json()
          setResultImg(data.annotated_b64); setDetections(data.detections || [])
          const now = Date.now()
          fpsCounterRef.current.push(now)
          fpsCounterRef.current = fpsCounterRef.current.filter(t => now - t < 1000)
          setFps(fpsCounterRef.current.length)
        } catch {}
        finally { inFlightRef.current = false }
      }, 50) // fires every 50ms; frame dropout keeps it from piling up
    } catch (e) {
      setError(`Webcam error: ${e instanceof Error ? e.message : e}`)
    }
  }

  function stopWebcam() {
    if (webcamIntervalRef.current) { clearInterval(webcamIntervalRef.current); webcamIntervalRef.current = null }
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    if (videoRef.current) videoRef.current.srcObject = null
    fetch(`${apiUrl}/api/infer/session/${sessionId}`, { method: "DELETE" }).catch(() => {})
    setWebcamActive(false); setSessionId(""); setFps(0); inFlightRef.current = false
  }

  useEffect(() => () => { stopWebcam() }, [])

  function str8() { return Math.random().toString(36).slice(2, 10) }
  const taskColor = (t: string) =>
    t === "detect" ? "text-blue-400" : t === "segment" ? "text-purple-400" : "text-green-400"

  // ── Catalog helpers ─────────────────────────────────────────────────────────
  const catalogGroups = Object.entries(GROUP_LABELS).map(([key, label]) => ({
    key, label, models: catalog.filter(m => m.type === key),
  })).filter(g => g.models.length > 0)

  function CatalogRow({ m }: { m: CatalogModel }) {
    const isLoading = m.status === "downloading"
    return (
      <div className="flex items-center gap-2 py-1.5 px-1 rounded hover:bg-muted/20">
        <span className="flex-1 text-xs truncate">{m.name}</span>
        {isLoading ? (
          <div className="flex items-center gap-1.5 shrink-0">
            <Progress value={m.download_progress} className="w-16 h-1" />
            <span className="text-[10px] text-blue-400">{m.download_progress}%</span>
          </div>
        ) : m.status === "loaded" ? (
          <Badge variant="outline" className="text-[9px] text-green-400 border-green-400/40 shrink-0">Ready</Badge>
        ) : m.error ? (
          <span className="text-[10px] text-red-400 truncate max-w-[80px]" title={m.error}>Error</span>
        ) : (
          <button
            onClick={() => loadSmartModel(m)}
            className="text-[10px] px-1.5 py-0.5 rounded border border-border/50 hover:border-primary/60 text-muted-foreground hover:text-foreground transition-colors shrink-0">
            {m.status === "downloaded" ? "Load" : "Download"}
          </button>
        )}
      </div>
    )
  }

  // ── Render ─────────────────────────────────────────────────────────────────────
  return (
    <div className="p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-semibold text-foreground">Inference</h1>
        <p className="text-muted-foreground mt-1">
          YOLO models or smart models (SAM, YOLO-World, GroundingDINO, OWL-ViT) on images, video, or webcam.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* ── Config panel ─────────────────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Cpu className="w-5 h-5 text-primary" /> Model &amp; Settings
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">

            {/* Source toggle */}
            <div className="flex gap-1 bg-muted/30 p-1 rounded-lg">
              {(["yolo", "smart"] as const).map(src => (
                <button key={src} onClick={() => setModelSource(src)}
                  className={`flex-1 flex items-center justify-center gap-1.5 py-1.5 rounded-md text-xs font-medium transition-all ${
                    modelSource === src ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                  }`}>
                  {src === "yolo" ? <Cpu className="w-3.5 h-3.5" /> : <Wand2 className="w-3.5 h-3.5" />}
                  {src === "yolo" ? "YOLO" : "Smart Model"}
                </button>
              ))}
            </div>

            {/* ── YOLO ── */}
            {modelSource === "yolo" && (<>
              <div className="space-y-2">
                <label className="text-sm font-medium">Model (from training runs)</label>
                <Select value={selectedModel} onValueChange={v => { setSelectedModel(v); setManualPath("") }}>
                  <SelectTrigger><SelectValue placeholder="Select trained model…" /></SelectTrigger>
                  <SelectContent>
                    {yoloModels.map(m => (
                      <SelectItem key={m.path} value={m.path}>
                        <span className="truncate max-w-[200px]">{m.name}</span>
                        <Badge variant="outline" className="ml-1 text-[10px]">{m.model_type}</Badge>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <label className="text-sm font-medium text-muted-foreground">Or paste model path</label>
                <Input placeholder="/path/to/model.pt" value={manualPath}
                  onChange={e => { setManualPath(e.target.value); setSelectedModel("") }}
                  className="text-xs font-mono" />
              </div>
              {activeModelPath && <p className="text-[10px] font-mono text-primary truncate">{activeModelPath}</p>}
              <div className="space-y-2">
                <label className="text-sm font-medium">Task Type</label>
                <Select value={task} onValueChange={setTask}>
                  <SelectTrigger><SelectValue /></SelectTrigger>
                  <SelectContent>
                    {TASK_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
                  </SelectContent>
                </Select>
              </div>
              {task === "track" && (
                <div className="space-y-2">
                  <label className="text-sm font-medium">Tracker / ReID</label>
                  <Select value={tracker} onValueChange={setTracker}>
                    <SelectTrigger><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {TRACKER_OPTIONS.map(o => <SelectItem key={o.value} value={o.value}>{o.label}</SelectItem>)}
                    </SelectContent>
                  </Select>
                  <p className="text-[10px] text-muted-foreground">
                    DeepOCSORT / StrongSORT require <span className="font-mono">boxmot</span> package.
                  </p>
                </div>
              )}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">IoU Threshold</label>
                  <span className="text-sm text-muted-foreground">{iou.toFixed(2)}</span>
                </div>
                <Slider value={[iou]} onValueChange={([v]) => setIou(v)} min={0.1} max={0.95} step={0.05} />
              </div>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Image Size</label>
                  <span className="text-sm text-muted-foreground">{imgsz}px</span>
                </div>
                <Slider value={[imgsz]} onValueChange={([v]) => setImgsz(v)} min={320} max={1280} step={32} />
              </div>
            </>)}

            {/* ── Smart Model ── */}
            {modelSource === "smart" && (<>

              {/* Active model selector (loaded only) */}
              <div className="space-y-2">
                <label className="text-sm font-medium">Active Model</label>
                {loadedModels.length === 0 ? (
                  <p className="text-xs text-muted-foreground italic">No smart models loaded yet — use the library below.</p>
                ) : (
                  <Select value={selSmartId} onValueChange={setSelSmartId}>
                    <SelectTrigger><SelectValue placeholder="Select…" /></SelectTrigger>
                    <SelectContent>
                      {loadedModels.map(m => (
                        <SelectItem key={m.id} value={m.id}>
                          <span>{m.name}</span>
                          <Badge variant="outline" className="ml-1 text-[10px]">{m.type}</Badge>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                )}
              </div>

              {/* Capability badges */}
              {activeSmartModel && (
                <div className="flex flex-wrap gap-1">
                  {activeSmartModel.supports_text && (
                    <Badge variant="secondary" className="text-[10px] gap-1">
                      <Wand2 className="w-2.5 h-2.5" /> Text prompt
                    </Badge>
                  )}
                  {activeSmartModel.supports_point && (
                    <Badge variant="secondary" className="text-[10px] gap-1">
                      <Crosshair className="w-2.5 h-2.5" /> Point prompt
                    </Badge>
                  )}
                </div>
              )}

              {/* Text prompt */}
              {activeSmartModel?.supports_text && (
                <div className="space-y-1.5">
                  <label className="text-sm font-medium">Text Prompt</label>
                  <Input placeholder="car, person, dog" value={textPrompt}
                    onChange={e => setTextPrompt(e.target.value)} className="text-xs" />
                  <p className="text-[10px] text-muted-foreground">Comma-separated class names.</p>
                </div>
              )}

              {/* Point prompt info */}
              {activeSmartModel?.supports_point && (
                <div className="text-xs text-muted-foreground rounded-md bg-muted/30 p-2.5 space-y-1">
                  <div className="flex items-center gap-1.5 font-medium text-foreground">
                    <Crosshair className="w-3.5 h-3.5 text-primary" /> Point Prompt
                  </div>
                  {promptPoint ? (
                    <div className="flex items-center gap-2">
                      <span>({(promptPoint.x * 100).toFixed(0)}%, {(promptPoint.y * 100).toFixed(0)}%)</span>
                      <button onClick={() => setPromptPoint(null)} className="text-muted-foreground hover:text-red-400 ml-auto">
                        <X className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ) : <span>Click on the uploaded image to set a point</span>}
                </div>
              )}

              {/* Webcam frame size */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <label className="text-sm font-medium">Webcam Frame Size</label>
                  <span className="text-sm text-muted-foreground">{smartImgsz}px</span>
                </div>
                <Slider value={[smartImgsz]} onValueChange={([v]) => setSmartImgsz(v)} min={160} max={1280} step={32} />
                <p className="text-[10px] text-muted-foreground">Lower = faster webcam inference for smart models</p>
              </div>

              {/* ── Model Library ── */}
              <div className="border border-border/40 rounded-lg overflow-hidden">
                <div
                  onClick={() => setShowLibrary(v => !v)}
                  className="w-full flex items-center justify-between px-3 py-2 text-xs font-medium hover:bg-muted/30 transition-colors cursor-pointer select-none">
                  <span className="flex items-center gap-1.5">
                    Model Library
                    {anyDownloading && <RefreshCw className="w-3 h-3 animate-spin text-blue-400" />}
                    <Badge variant="secondary" className="text-[10px]">{loadedModels.length} loaded</Badge>
                  </span>
                  <div className="flex items-center gap-1.5">
                    <button onClick={e => { e.stopPropagation(); refreshCatalog() }}
                      disabled={catalogLoading}
                      className="text-muted-foreground hover:text-foreground p-0.5">
                      <RefreshCw className={`w-3 h-3 ${catalogLoading ? "animate-spin" : ""}`} />
                    </button>
                    {showLibrary ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                  </div>
                </div>

                {showLibrary && (
                  <ScrollArea className="h-64 border-t border-border/40">
                    <div className="p-2 space-y-3">
                      {catalogGroups.map(g => (
                        <div key={g.key}>
                          <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wide px-1 mb-1">{g.label}</p>
                          {g.models.map(m => <CatalogRow key={m.id} m={m} />)}
                        </div>
                      ))}
                    </div>
                  </ScrollArea>
                )}
              </div>
            </>)}

            {/* Confidence — shared */}
            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Confidence</label>
                <span className="text-sm text-muted-foreground">{confidence.toFixed(2)}</span>
              </div>
              <Slider value={[confidence]} onValueChange={([v]) => setConfidence(v)} min={0.01} max={0.99} step={0.01} />
            </div>

            {error && (
              <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-400 flex gap-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span className="break-all">{error}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Input / Output panel ─────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Source tabs */}
          <div className="flex gap-1 bg-muted/30 p-1 rounded-lg w-fit">
            {(["image", "video", "webcam"] as const).map(tab => (
              <button key={tab}
                onClick={() => { setSourceTab(tab); setResultImg(null); setDetections([]) }}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all ${
                  sourceTab === tab ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground"
                }`}>
                {tab === "image" ? <Upload className="w-3.5 h-3.5" /> :
                 tab === "video" ? <Video  className="w-3.5 h-3.5" /> :
                                   <Camera className="w-3.5 h-3.5" />}
                {tab.charAt(0).toUpperCase() + tab.slice(1)}
                {tab === "video" && modelSource === "smart" && (
                  <Badge variant="outline" className="text-[9px] text-muted-foreground ml-0.5">YOLO only</Badge>
                )}
              </button>
            ))}
          </div>

          {/* ── Image tab ─────────────────────────────────────────────────── */}
          {sourceTab === "image" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border/50 rounded-lg cursor-pointer hover:border-primary/50 transition-colors">
                  <Upload className="w-6 h-6 text-muted-foreground mb-2" />
                  <span className="text-sm text-muted-foreground">{imageFile ? imageFile.name : "Click to upload image"}</span>
                  <input type="file" accept="image/*" className="hidden"
                    onChange={e => e.target.files?.[0] && handleImageFile(e.target.files[0])} />
                </label>

                <div className="flex gap-2">
                  <Button onClick={runImageInference}
                    disabled={!imageFile || inferring || (modelSource === "yolo" ? !activeModelPath : !selSmartId)}
                    className="gap-2">
                    {inferring ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Eye className="w-4 h-4" />}
                    {inferring ? "Running…" : "Run Inference"}
                  </Button>
                  {resultImg && (
                    <Button variant="outline" onClick={downloadResult} className="gap-2">
                      <Download className="w-4 h-4" /> Save Result
                    </Button>
                  )}
                </div>

                {(imagePreview || resultImg) && (
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    {imagePreview && (
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">
                          Input
                          {modelSource === "smart" && activeSmartModel?.supports_point && (
                            <span className="ml-1 text-primary">(click to set point)</span>
                          )}
                        </p>
                        <div className="relative">
                          {/* eslint-disable-next-line @next/next/no-img-element */}
                          <img src={imagePreview} alt="input"
                            className={`w-full rounded border border-border/40 ${modelSource === "smart" && activeSmartModel?.supports_point ? "cursor-crosshair" : ""}`}
                            onClick={handleInputImageClick} />
                          {promptPoint && (
                            <div className="absolute w-4 h-4 rounded-full border-2 border-white bg-primary/80 -translate-x-1/2 -translate-y-1/2 pointer-events-none"
                              style={{ left: `${promptPoint.x * 100}%`, top: `${promptPoint.y * 100}%` }} />
                          )}
                        </div>
                      </div>
                    )}
                    {resultImg && (
                      <div>
                        <p className="text-xs text-muted-foreground mb-1">Result</p>
                        {/* eslint-disable-next-line @next/next/no-img-element */}
                        <img src={`data:image/jpeg;base64,${resultImg}`} alt="result"
                          className="w-full rounded border border-border/40" />
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ── Video tab ─────────────────────────────────────────────────── */}
          {sourceTab === "video" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                {modelSource === "smart" ? (
                  <div className="h-32 rounded-lg border-2 border-dashed border-border/30 flex flex-col items-center justify-center gap-2 text-muted-foreground">
                    <Video className="w-8 h-8 opacity-30" />
                    <p className="text-sm">Video inference is YOLO-only.</p>
                    <p className="text-xs">Switch to YOLO or use Image / Webcam.</p>
                  </div>
                ) : (<>
                  <label className="flex flex-col items-center justify-center h-32 border-2 border-dashed border-border/50 rounded-lg cursor-pointer hover:border-primary/50 transition-colors">
                    <Video className="w-6 h-6 text-muted-foreground mb-2" />
                    <span className="text-sm text-muted-foreground">{videoFile ? videoFile.name : "Click to upload video"}</span>
                    <input type="file" accept="video/*" className="hidden"
                      onChange={e => { e.target.files?.[0] && setVideoFile(e.target.files[0]); setVideoJob(null) }} />
                  </label>
                  <div className="flex gap-2">
                    <Button onClick={startVideoInference}
                      disabled={!videoFile || !activeModelPath || videoJob?.status === "running" || videoJob?.status === "starting"}
                      className="gap-2">
                      <Play className="w-4 h-4" />
                      {videoJob?.status === "running" ? "Processing…" : "Process Video"}
                    </Button>
                    {videoJob?.status === "completed" && (
                      <Button variant="outline" className="gap-2"
                        onClick={() => window.open(`${apiUrl}/api/infer/video/${videoJob.id}/result`, "_blank")}>
                        <Download className="w-4 h-4" /> Download Result
                      </Button>
                    )}
                  </div>
                  {videoJob && (
                    <div className="space-y-2">
                      <div className="flex justify-between text-xs text-muted-foreground">
                        <span>{videoJob.message}</span>
                        <Badge variant="outline" className={
                          videoJob.status === "completed" ? "text-green-400 border-green-400/30" :
                          videoJob.status === "failed"    ? "text-red-400  border-red-400/30"  :
                                                            "text-blue-400 border-blue-400/30"
                        }>{videoJob.status}</Badge>
                      </div>
                      <Progress value={videoJob.progress} className="h-1.5" />
                    </div>
                  )}
                </>)}
              </CardContent>
            </Card>
          )}

          {/* ── Webcam tab ────────────────────────────────────────────────── */}
          {sourceTab === "webcam" && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-4">
                <div className="flex items-center gap-3 flex-wrap">
                  {!webcamActive ? (
                    <Button onClick={startWebcam}
                      disabled={modelSource === "yolo" ? !activeModelPath : !selSmartId}
                      className="gap-2">
                      <Camera className="w-4 h-4" /> Start Webcam
                    </Button>
                  ) : (
                    <Button variant="destructive" onClick={stopWebcam} className="gap-2">
                      <Square className="w-4 h-4" /> Stop
                    </Button>
                  )}
                  {webcamActive && (
                    <Badge variant="outline" className="text-green-400 border-green-400/30 gap-1">
                      <ZapIcon className="w-3 h-3" /> {fps} FPS
                    </Badge>
                  )}
                  {modelSource === "smart" && webcamActive && activeSmartModel?.supports_point && (
                    <p className="text-xs text-muted-foreground">
                      {promptPoint
                        ? `Point: (${(promptPoint.x*100).toFixed(0)}%, ${(promptPoint.y*100).toFixed(0)}%)`
                        : "No point — SAM segments everything"}
                    </p>
                  )}
                  {resultImg && webcamActive && (
                    <Button variant="outline" size="sm" onClick={downloadResult} className="gap-1 h-7 text-xs ml-auto">
                      <Download className="w-3 h-3" /> Save Frame
                    </Button>
                  )}
                </div>

                <video ref={videoRef} className="hidden" muted playsInline />
                <canvas ref={canvasRef} className="hidden" />

                {resultImg ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={`data:image/jpeg;base64,${resultImg}`} alt="live inference"
                    className="w-full rounded border border-border/40" />
                ) : webcamActive ? (
                  <div className="h-60 rounded border border-border/40 bg-black/20 flex items-center justify-center">
                    <p className="text-sm text-muted-foreground">Waiting for first frame…</p>
                  </div>
                ) : (
                  <div className="h-60 rounded border-2 border-dashed border-border/30 flex flex-col items-center justify-center gap-2">
                    <Camera className="w-8 h-8 text-muted-foreground/40" />
                    <p className="text-sm text-muted-foreground">Start webcam to begin live inference</p>
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* ── Detections ────────────────────────────────────────────────── */}
          {detections.length > 0 && (
            <Card className="border-border/50 bg-card/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <Eye className="w-4 h-4 text-primary" /> Detections
                  <Badge variant="secondary" className="ml-auto">{detections.length}</Badge>
                </CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-48">
                  <div className="space-y-1">
                    {detections.map((d, i) => (
                      <div key={i} className="flex items-center gap-2 text-xs p-1.5 rounded hover:bg-muted/30">
                        {d.track_id !== undefined && (
                          <span className="font-mono text-[10px] bg-primary/20 text-primary px-1 py-0.5 rounded shrink-0">#{d.track_id}</span>
                        )}
                        <span className={`font-semibold ${taskColor(task)}`}>{d.class_name}</span>
                        <span className="text-muted-foreground">{(d.confidence * 100).toFixed(1)}%</span>
                        {d.mask_points && (
                          <Badge variant="outline" className="text-[9px] text-purple-400 border-purple-400/30">mask</Badge>
                        )}
                        {d.bbox && (
                          <span className="font-mono text-muted-foreground/60 ml-auto text-[10px]">
                            [{d.bbox.map(v => Math.round(v)).join(", ")}]
                          </span>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
