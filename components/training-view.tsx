"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible"
import { Input } from "@/components/ui/input"
import {
  Play,
  Pause,
  Square,
  Cpu,
  Settings2,
  Terminal,
  TrendingUp,
  TrendingDown,
  HardDrive,
  Activity,
  Download,
  ChevronDown,
  LineChart as LineChartIcon,
  FileDown,
  Package,
  Trash2,
  FolderOpen,
} from "lucide-react"
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from "recharts"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** One row of per-epoch training data from the backend */
interface EpochRecord {
  epoch: number
  train_box_loss: number
  train_cls_loss: number
  train_dfl_loss: number
  val_box_loss: number
  val_cls_loss: number
  val_dfl_loss: number
  mAP50: number
  mAP50_95: number
  precision: number
  recall: number
  speed_ms: number
  its?: number
  gpu_mem_gb?: number
  // segmentation extras
  train_seg_loss?: number
  val_seg_loss?: number
  mAP50_seg?: number
}

interface LogEntry {
  timestamp: string
  message: string
  type: "info" | "warning" | "error" | "success"
}

interface BatchStats {
  epoch: number
  batch: number
  total_batches: number
  its: number
  losses: Record<string, number>
}

/** Shape returned by GET /api/train/:id/status */
interface TrainingStatus {
  id: string
  name: string
  status: "starting" | "running" | "pausing" | "completed" | "failed" | "stopped" | "paused" | "interrupted"
  progress: number
  current_epoch: number
  total_epochs: number
  metrics: Record<string, number>
  epoch_history: EpochRecord[]
  device_info: string | null
  gpu_mem_gb: number | null
  started_at: string
  model_path: string | null
  batch_stats: BatchStats | null
  logs: string[]
  error?: string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const MODEL_ARCHITECTURES: Array<{ value: string; label: string; params: string; taskType: string; arch: string }> = [
  // ── YOLOv8 — detection ──────────────────────────────────────────────────────
  { value: "yolov8n",      label: "YOLOv8 Nano",       params: "3.2M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolov8s",      label: "YOLOv8 Small",      params: "11.2M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov8m",      label: "YOLOv8 Medium",     params: "25.9M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov8l",      label: "YOLOv8 Large",      params: "43.7M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov8x",      label: "YOLOv8 XLarge",     params: "68.2M", taskType: "detection",      arch: "YOLO" },
  // ── YOLOv8 — segmentation ───────────────────────────────────────────────────
  { value: "yolov8n-seg",  label: "YOLOv8n Seg",       params: "3.4M",  taskType: "segmentation",   arch: "YOLO" },
  { value: "yolov8s-seg",  label: "YOLOv8s Seg",       params: "11.8M", taskType: "segmentation",   arch: "YOLO" },
  { value: "yolov8m-seg",  label: "YOLOv8m Seg",       params: "27.3M", taskType: "segmentation",   arch: "YOLO" },
  // ── YOLOv8 — classification ──────────────────────────────────────────────────
  { value: "yolov8n-cls",  label: "YOLOv8n Cls",       params: "2.7M",  taskType: "classification", arch: "YOLO" },
  { value: "yolov8s-cls",  label: "YOLOv8s Cls",       params: "6.4M",  taskType: "classification", arch: "YOLO" },
  // ── YOLOv9 — detection ──────────────────────────────────────────────────────
  { value: "yolov9n",      label: "YOLOv9 Nano",       params: "2.0M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolov9s",      label: "YOLOv9 Small",      params: "7.2M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolov9m",      label: "YOLOv9 Medium",     params: "20.1M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov9c",      label: "YOLOv9 Compact",    params: "25.5M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov9e",      label: "YOLOv9 Extended",   params: "57.8M", taskType: "detection",      arch: "YOLO" },
  // ── YOLOv10 — detection ─────────────────────────────────────────────────────
  { value: "yolov10n",     label: "YOLOv10 Nano",      params: "2.3M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolov10s",     label: "YOLOv10 Small",     params: "7.2M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolov10m",     label: "YOLOv10 Medium",    params: "15.4M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov10b",     label: "YOLOv10 Balanced",  params: "19.1M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov10l",     label: "YOLOv10 Large",     params: "24.4M", taskType: "detection",      arch: "YOLO" },
  { value: "yolov10x",     label: "YOLOv10 XLarge",    params: "29.5M", taskType: "detection",      arch: "YOLO" },
  // ── YOLO11 — detection ──────────────────────────────────────────────────────
  { value: "yolo11n",      label: "YOLO11 Nano",       params: "2.6M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolo11s",      label: "YOLO11 Small",      params: "9.4M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolo11m",      label: "YOLO11 Medium",     params: "20.1M", taskType: "detection",      arch: "YOLO" },
  { value: "yolo11l",      label: "YOLO11 Large",      params: "25.3M", taskType: "detection",      arch: "YOLO" },
  { value: "yolo11x",      label: "YOLO11 XLarge",     params: "56.9M", taskType: "detection",      arch: "YOLO" },
  // ── YOLO11 — segmentation ───────────────────────────────────────────────────
  { value: "yolo11n-seg",  label: "YOLO11n Seg",       params: "2.9M",  taskType: "segmentation",   arch: "YOLO" },
  { value: "yolo11s-seg",  label: "YOLO11s Seg",       params: "10.1M", taskType: "segmentation",   arch: "YOLO" },
  // ── YOLO11 — classification ──────────────────────────────────────────────────
  { value: "yolo11n-cls",  label: "YOLO11n Cls",       params: "1.6M",  taskType: "classification", arch: "YOLO" },
  { value: "yolo11s-cls",  label: "YOLO11s Cls",       params: "5.6M",  taskType: "classification", arch: "YOLO" },
  // ── YOLO12 — detection ──────────────────────────────────────────────────────
  { value: "yolo12n",      label: "YOLO12 Nano",       params: "2.6M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolo12s",      label: "YOLO12 Small",      params: "9.3M",  taskType: "detection",      arch: "YOLO" },
  { value: "yolo12m",      label: "YOLO12 Medium",     params: "20.2M", taskType: "detection",      arch: "YOLO" },
  { value: "yolo12l",      label: "YOLO12 Large",      params: "26.4M", taskType: "detection",      arch: "YOLO" },
  { value: "yolo12x",      label: "YOLO12 XLarge",     params: "59.1M", taskType: "detection",      arch: "YOLO" },
  // ── RT-DETR v1 — detection (Ultralytics) ─────────────────────────────────────
  { value: "rtdetr-l",        label: "RT-DETR Large",          params: "32M",  taskType: "detection", arch: "RT-DETR" },
  { value: "rtdetr-x",        label: "RT-DETR XLarge",         params: "67M",  taskType: "detection", arch: "RT-DETR" },
  // ── RT-DETRv2 — detection (HuggingFace / PekingU) ────────────────────────────
  { value: "rtdetrv2_r18vd",  label: "RT-DETRv2 R18 (Small)",  params: "20M",  taskType: "detection", arch: "RT-DETRv2" },
  { value: "rtdetrv2_r34vd",  label: "RT-DETRv2 R34 (Medium)", params: "31M",  taskType: "detection", arch: "RT-DETRv2" },
  { value: "rtdetrv2_r50vd",  label: "RT-DETRv2 R50 (Large)",  params: "42M",  taskType: "detection", arch: "RT-DETRv2" },
  { value: "rtdetrv2_r101vd", label: "RT-DETRv2 R101 (XLarge)",params: "76M",  taskType: "detection", arch: "RT-DETRv2" },
  // ── RF-DETR — detection (Roboflow) ───────────────────────────────────────────
  { value: "rfdetr_base",  label: "RF-DETR Base",      params: "29M",   taskType: "detection",      arch: "RF-DETR" },
  { value: "rfdetr_large", label: "RF-DETR Large",     params: "128M",  taskType: "detection",      arch: "RF-DETR" },
]

/** Derive the backend model_type from the selected architecture value. */
function getModelType(archValue: string): string {
  if (archValue.startsWith("rtdetrv2")) return "rtdetrv2"
  if (archValue.startsWith("rtdetr"))   return "rtdetr"
  if (archValue.startsWith("rfdetr"))   return "rfdetr"
  if (archValue.includes("-seg"))     return "segmentation"
  if (archValue.includes("-cls"))     return "classification"
  return "yolo"
}

const TASK_TYPES = [
  { value: "detection",      label: "Object Detection" },
  { value: "segmentation",   label: "Instance Segmentation" },
  { value: "classification", label: "Classification" },
]

const POLL_INTERVAL_MS = 2000

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface DatasetInfo {
  id: string
  name: string
  path: string
  format?: string
  task_type?: string
}

interface TrainingViewProps {
  datasets?: DatasetInfo[]
  apiUrl?: string
}

/** Classify a dataset's task domain based on format and task_type fields. */
function datasetDomain(ds: DatasetInfo): "classification" | "segmentation" | "detection" {
  const fmt = (ds.format  ?? "").toLowerCase()
  const tt  = (ds.task_type ?? "").toLowerCase()
  if (fmt === "classification-folder" || fmt === "classification_folder" || tt === "classification") return "classification"
  if (fmt === "cityscapes" || fmt === "ade20k" || fmt === "yolo-seg" || tt === "segmentation") return "segmentation"
  return "detection"
}

/** Return null if compatible, or an object with error / warning strings. */
function checkCompatibility(
  dsDomain: "classification" | "segmentation" | "detection",
  modelTaskType: string,   // "detection" | "segmentation" | "classification"
): { error?: string; warning?: string } | null {
  if (dsDomain === "classification" && modelTaskType !== "classification")
    return { error: "Classification datasets can only train Classification models." }
  if (dsDomain === "segmentation" && modelTaskType === "classification")
    return { error: "Segmentation datasets cannot train Classification models." }
  if (dsDomain === "detection" && modelTaskType === "segmentation")
    return { error: "Detection datasets don't contain polygon masks. Use a segmentation-format dataset, or train an Object Detection model instead." }
  if (dsDomain === "detection" && modelTaskType === "classification")
    return { error: "Detection datasets cannot train Classification models. Use a classification-folder dataset." }
  if (dsDomain === "segmentation" && modelTaskType === "detection")
    return { warning: "This dataset has segmentation masks. Polygons will be automatically converted to bounding boxes for detection training." }
  return null
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function TrainingView({ datasets = [], apiUrl = "http://localhost:8000" }: TrainingViewProps) {
  // Config state
  const [selectedDatasetId, setSelectedDatasetId] = useState("")
  const [modelArch, setModelArch]     = useState("yolov8n")
  const [taskType, setTaskType]       = useState("detection")
  const [epochs, setEpochs]           = useState(100)
  const [batchSize, setBatchSize]     = useState(16)
  const [imgSize, setImgSize]         = useState(640)
  const [usePretrained, setUsePretrained] = useState(true)

  // Save-period
  const [savePeriod, setSavePeriod]       = useState(0)  // 0 = disabled

  // Advanced hyperparams
  const [lr0, setLr0]                     = useState(0.01)
  const [lrf, setLrf]                     = useState(0.01)
  const [optimizer, setOptimizer]         = useState("SGD")
  const [patience, setPatience]           = useState(50)
  const [cosLr, setCosLr]                 = useState(false)
  const [warmupEpochs, setWarmupEpochs]   = useState(3)
  const [weightDecay, setWeightDecay]     = useState(0.0005)
  const [mosaic, setMosaic]               = useState(1.0)
  const [hsvH, setHsvH]                   = useState(0.015)
  const [hsvS, setHsvS]                   = useState(0.7)
  const [hsvV, setHsvV]                   = useState(0.4)
  const [flipud, setFlipud]               = useState(0.0)
  const [fliplr, setFliplr]               = useState(0.5)
  const [amp, setAmp]                     = useState(true)
  const [dropout, setDropout]             = useState(0.0)
  const [advancedOpen, setAdvancedOpen]   = useState(false)

  // Export state
  const [isExporting, setIsExporting]     = useState(false)
  const [exportFormat, setExportFormat]   = useState("onnx")

  // GPU install state
  const [gpuWarning, setGpuWarning]       = useState<string | null>(null)
  const [installingCuda, setInstallingCuda] = useState(false)

  // Run name
  const [runName, setRunName]         = useState("")

  // Runtime state
  const [trainingId, setTrainingId]   = useState<string | null>(null)
  const [isTraining, setIsTraining]   = useState(false)
  const [isPaused, setIsPaused]       = useState(false)   // true when training is paused on backend
  const [isPausing, setIsPausing]     = useState(false)   // true while finishing current epoch before pause
  const [currentEpoch, setCurrentEpoch] = useState(0)
  const [totalEpochs, setTotalEpochs] = useState(100)
  const [progress, setProgress]       = useState(0)
  const [metrics, setMetrics]         = useState<EpochRecord[]>([])
  const [logs, setLogs]               = useState<LogEntry[]>([])
  const [statusLabel, setStatusLabel] = useState<string>("")
  const [modelPath, setModelPath]     = useState<string | null>(null)
  const [error, setError]             = useState<string | null>(null)

  // Past runs — all jobs from backend, refreshed on mount and after training ends
  interface RunSummary {
    id: string; name: string; status: string; progress: number
    current_epoch: number; total_epochs: number; model_type: string
    started_at: string; model_path: string | null; last_checkpoint?: string | null
  }
  interface CheckpointInfo { name: string; path: string; size_mb: number }
  const [pastRuns, setPastRuns]           = useState<RunSummary[]>([])
  const [checkpoints, setCheckpoints]         = useState<CheckpointInfo[]>([])
  const [checkpointRunId, setCheckpointRunId] = useState<string | null>(null)
  const [deviceInfo, setDeviceInfo]   = useState<string | null>(null)
  const [liveGpuMem, setLiveGpuMem]   = useState<number | null>(null)
  const [batchStats, setBatchStats]   = useState<BatchStats | null>(null)

  // Real system stats — polled from /api/system while training
  const [cpuUsage, setCpuUsage]         = useState(0)
  const [gpuUsage, setGpuUsage]         = useState<number | null>(null)
  const [memoryUsage, setMemoryUsage]   = useState(0)
  const [gpuName, setGpuName]           = useState<string | null>(null)
  const systemPollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  const pollRef          = useRef<ReturnType<typeof setInterval> | null>(null)
  const seenLogCount     = useRef(0)
  const logsEndRef       = useRef<HTMLDivElement>(null)
  const failedPollCount  = useRef(0)   // consecutive polls returning "failed" — stop after threshold

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  function inferLogType(message: string, fallback: LogEntry["type"]): LogEntry["type"] {
    const lc = message.toLowerCase()
    if (lc.includes("error") || lc.includes("failed") || lc.includes("exception")) return "error"
    if (lc.includes("warning") || lc.includes("warn")) return "warning"
    if (lc.includes("completed") || lc.includes("saved") || lc.includes("success")) return "success"
    return fallback
  }

  function addLog(message: string, type: LogEntry["type"]) {
    const timestamp = new Date().toLocaleTimeString()
    setLogs(prev => [...prev, { timestamp, message, type }])
  }

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  async function handleInstallCudaTorch() {
    setInstallingCuda(true)
    addLog("Installing CUDA-enabled PyTorch… this may take several minutes.", "info")
    try {
      const res = await fetch(`${apiUrl}/api/install-cuda-torch`, { method: "POST" })
      const data = await res.json()
      if (data.success) {
        addLog("✓ CUDA PyTorch installed. Restart the backend server, then retry training.", "success")
        setGpuWarning(null)
      } else {
        addLog(`Install failed: ${data.message}`, "error")
      }
    } catch (err) {
      addLog(`Install request failed: ${err instanceof Error ? err.message : err}`, "error")
    } finally {
      setInstallingCuda(false)
    }
  }

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [logs])

  // -------------------------------------------------------------------------
  // System stats polling — always active so CPU/RAM/GPU are shown even when idle
  // -------------------------------------------------------------------------
  useEffect(() => {
    const pollSystem = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/system`)
        if (!res.ok) return
        const data = await res.json()
        setCpuUsage(data.cpu_percent ?? 0)
        if (data.gpu_percent !== null && data.gpu_percent !== undefined) {
          setGpuUsage(data.gpu_percent)
          if (data.gpu_name) setGpuName(data.gpu_name)
        } else {
          setGpuUsage(null)
        }
        setMemoryUsage(data.gpu_memory_percent ?? data.ram_percent ?? 0)
      } catch {
        // Backend unreachable — leave stats at last known value
      }
    }

    pollSystem()
    // Poll every 3 s while training, every 10 s while idle
    const interval = isTraining ? 3000 : 10000
    systemPollRef.current = setInterval(pollSystem, interval)
    return () => {
      if (systemPollRef.current) {
        clearInterval(systemPollRef.current)
        systemPollRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isTraining, apiUrl])

  // -------------------------------------------------------------------------
  // Persist trainingId to localStorage so it survives page refreshes
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (trainingId) localStorage.setItem('cvdm_training_id', trainingId)
    else localStorage.removeItem('cvdm_training_id')
  }, [trainingId])

  // -------------------------------------------------------------------------
  // On mount: restore any active/paused/interrupted training job from backend
  // -------------------------------------------------------------------------
  useEffect(() => {
    const restore = async () => {
      try {
        const res = await fetch(`${apiUrl}/api/train/jobs`)
        if (!res.ok) return
        const data = await res.json()
        const jobs: RunSummary[] = data.jobs || []

        // Always populate the past-runs list
        setPastRuns(jobs)

        // Only auto-restore a job that is actively running — do NOT auto-continue
        // paused/interrupted runs; user must explicitly click "Load & Resume".
        const active = jobs.find(j =>
          j.status === 'running' || j.status === 'starting' || j.status === 'pausing'
        )
        if (active) {
          setTrainingId(active.id)
          setIsTraining(true)
          setIsPaused(false)
          setIsPausing(active.status === 'pausing')
          setProgress(active.progress ?? 0)
          setCurrentEpoch(active.current_epoch ?? 0)
          setTotalEpochs(active.total_epochs ?? 100)
          setStatusLabel(active.status === 'pausing' ? 'Finishing epoch…' : 'Running')
          addLog(`Restored running job "${active.name}" (id: ${active.id})`, 'info')
          return
        }
      } catch { /* backend not reachable yet */ }
    }
    restore()
  }, [apiUrl]) // eslint-disable-line react-hooks/exhaustive-deps

  // -------------------------------------------------------------------------
  // Polling — starts whenever trainingId is set, pauses if isPaused
  // -------------------------------------------------------------------------

  useEffect(() => {
    // Stop polling only when fully paused/stopped — NOT while still finishing an epoch
    if (!trainingId || isPaused) {
      stopPolling()
      return
    }

    async function poll() {
      try {
        const res = await fetch(`${apiUrl}/api/train/${trainingId}/status`)
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        const data: TrainingStatus = await res.json()

        setCurrentEpoch(data.current_epoch)
        setTotalEpochs(data.total_epochs)
        setProgress(data.progress)

        // Sync device info, live GPU memory, and batch stats
        if (data.device_info) setDeviceInfo(data.device_info)
        if (data.gpu_mem_gb != null) setLiveGpuMem(data.gpu_mem_gb)
        if (data.batch_stats) setBatchStats(data.batch_stats)

        // Replace epoch history wholesale (backend is authoritative)
        if (data.epoch_history && data.epoch_history.length > 0) {
          setMetrics(data.epoch_history)
        }

        // Append only new log lines, auto-detecting type from content
        const newLogs = data.logs.slice(seenLogCount.current)
        seenLogCount.current = data.logs.length
        newLogs.forEach(msg => {
          addLog(msg, inferLogType(msg, "info"))
          if (msg.includes("Install CUDA-enabled PyTorch") || msg.includes("CPU-only")) {
            setGpuWarning(msg)
          }
        })

        if (data.status === "pausing") {
          failedPollCount.current = 0
          setIsPausing(true)
          setStatusLabel("Finishing epoch…")
          // keep polling — backend will transition to "paused" when the epoch completes
        } else if (data.status === "completed") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("Training completed successfully! Model saved.", "success")
          setStatusLabel("Completed")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "failed") {
          // Poll a couple more times to flush any final log lines, then stop.
          failedPollCount.current += 1
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          setError(data.error ?? "Unknown error")
          setStatusLabel("Failed")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          if (failedPollCount.current >= 3) {
            stopPolling()
            fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
          }
        } else if (data.status === "stopped") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("Training was stopped.", "warning")
          setStatusLabel("Stopped")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "paused") {
          failedPollCount.current = 0
          setIsTraining(false); setIsPausing(false); setIsPaused(true); setBatchStats(null)
          if (data.model_path) setModelPath(data.model_path)
          addLog("Epoch complete — training paused. Click Resume to continue.", "info")
          setStatusLabel("Paused")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
          fetch(`${apiUrl}/api/train/jobs`).then(r => r.json()).then(d => setPastRuns(d.jobs || [])).catch(() => {})
        } else if (data.status === "interrupted") {
          failedPollCount.current = 0
          setIsTraining(false)
          if (data.model_path) setModelPath(data.model_path)
          setStatusLabel("Interrupted")
          setCpuUsage(0); setGpuUsage(null); setMemoryUsage(0)
          stopPolling()
        } else {
          // status is "running" or "starting" — re-sync isTraining in case the
          // component was remounted (navigation) or the state drifted
          failedPollCount.current = 0
          setIsTraining(true)
          if (data.status === "starting") {
            setStatusLabel("Preparing dataset…")
          } else {
            setStatusLabel(prev => (prev === "Preparing dataset…" || prev === "Starting…") ? "Running" : prev)
          }
        }
      } catch (err) {
        addLog(`Polling error: ${err instanceof Error ? err.message : err}`, "error")
      }
    }

    poll()  // fire immediately, then every POLL_INTERVAL_MS
    pollRef.current = setInterval(poll, POLL_INTERVAL_MS)
    return stopPolling
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trainingId, isPaused, apiUrl])

  // -------------------------------------------------------------------------
  // Actions
  // -------------------------------------------------------------------------

  function handleDownloadReport() {
    if (!trainingId) return
    window.open(`${apiUrl}/api/train/${trainingId}/report`, "_blank")
  }

  async function handleStartTraining() {
    if (!selectedDatasetId) {
      addLog("Please select a dataset before starting training.", "error")
      return
    }

    // Dataset / model compatibility check (client-side early warning)
    const selectedDs = datasets.find(d => d.id === selectedDatasetId)
    if (selectedDs) {
      const dsDomain  = datasetDomain(selectedDs)
      const modelTask = getModelType(modelArch)   // "yolo" | "rtdetr" | "rfdetr" | "segmentation" | "classification"
      // Map backend model_type to simplified task domain for check
      const modelDomain = modelTask === "segmentation" ? "segmentation"
                        : modelTask === "classification" ? "classification"
                        : "detection"
      const compat = checkCompatibility(dsDomain, modelDomain)
      if (compat?.error) {
        addLog(compat.error, "error")
        setError(compat.error)
        return
      }
    }

    // Clear trainingId FIRST so that when isPaused becomes false the polling
    // effect sees !trainingId and stops immediately, preventing a race where
    // the previous job is re-polled and restores its stale epoch/metrics.
    setTrainingId(null)
    setMetrics([])
    setLogs([])
    setError(null)
    setModelPath(null)
    setCurrentEpoch(0)
    setTotalEpochs(epochs)
    setProgress(0)
    setDeviceInfo(null)
    setLiveGpuMem(null)
    seenLogCount.current = 0
    failedPollCount.current = 0
    setIsPaused(false)
    setIsPausing(false)
    setIsTraining(false)
    setBatchStats(null)
    setStatusLabel("Starting…")

    const backendModelType = getModelType(modelArch)

    try {
      const res = await fetch(`${apiUrl}/api/train`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: runName.trim(),
          dataset_id: selectedDatasetId,
          model_type: backendModelType,
          model_arch: modelArch,
          epochs,
          batch_size: batchSize,
          image_size: imgSize,
          pretrained: usePretrained,
          device: "auto",
          save_period: savePeriod,
          // Advanced hyperparams
          lr0,
          lrf,
          optimizer,
          patience,
          cos_lr: cosLr,
          warmup_epochs: warmupEpochs,
          weight_decay: weightDecay,
          mosaic,
          hsv_h: hsvH,
          hsv_s: hsvS,
          hsv_v: hsvV,
          flipud,
          fliplr,
          amp,
          dropout,
        }),
      })

      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }

      const data = await res.json()
      const id: string = data.training_id

      addLog(`Training job started (id: ${id})`, "info")
      addLog(`Model: ${MODEL_ARCHITECTURES.find(m => m.value === modelArch)?.label ?? modelArch}`, "info")
      addLog(`Dataset: ${datasets.find(d => d.id === selectedDatasetId)?.name ?? selectedDatasetId}`, "info")

      setTrainingId(id)
      setIsTraining(true)
      setStatusLabel("Running")
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err)
      addLog(`Failed to start training: ${msg}`, "error")
      setError(msg)
    }
  }

  async function handleStopTraining() {
    if (!trainingId) return
    try {
      await fetch(`${apiUrl}/api/train/${trainingId}/stop`, { method: "POST" })
      addLog("Stop signal sent — training will halt after the current epoch.", "warning")
      setIsTraining(false)
      setIsPaused(false)
      setStatusLabel("Stopping…")
      stopPolling()
    } catch (err: unknown) {
      addLog(`Stop request failed: ${err instanceof Error ? err.message : err}`, "error")
    }
  }

  async function handlePauseResume() {
    if (!trainingId) return
    if (!isPaused) {
      // Pause: signal backend — keep polling so we see "pausing" → "paused" transition
      try {
        const res = await fetch(`${apiUrl}/api/train/${trainingId}/pause`, { method: "POST" })
        if (res.ok) {
          setIsPausing(true)
          setStatusLabel("Finishing epoch…")
          // Do NOT stop polling or set isPaused here — the poll handler does that
          // when the backend responds with status "paused"
        } else {
          addLog("Pause failed.", "error")
        }
      } catch (err: unknown) {
        addLog(`Pause request failed: ${err instanceof Error ? err.message : err}`, "error")
      }
    } else {
      await doResume(trainingId)
    }
  }

  async function doResume(id: string) {
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}/resume`, { method: "POST" })
      if (res.ok) {
        setTrainingId(id)
        setIsPaused(false)
        setIsPausing(false)
        setIsTraining(true)
        setStatusLabel("Running")
        seenLogCount.current = 0   // re-stream all logs from this job
        failedPollCount.current = 0
        addLog("Resuming training from checkpoint…", "info")
      } else {
        const err = await res.json().catch(() => ({ detail: "Resume failed" }))
        addLog(`Resume failed: ${err.detail}`, "error")
      }
    } catch (err: unknown) {
      addLog(`Resume request failed: ${err instanceof Error ? err.message : err}`, "error")
    }
  }

  async function handleDeleteRun(id: string) {
    if (!confirm("Delete this run from the list? (Model weight files are kept on disk.)")) return
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}`, { method: "DELETE" })
      if (res.ok) {
        setPastRuns(prev => prev.filter(r => r.id !== id))
        if (trainingId === id) {
          setTrainingId(null)
          setIsPaused(false)
          setStatusLabel("")
        }
      } else {
        const err = await res.json().catch(() => ({ detail: "Delete failed" }))
        addLog(`Delete failed: ${err.detail}`, "error")
      }
    } catch (e) {
      addLog(`Delete request failed: ${e instanceof Error ? e.message : e}`, "error")
    }
  }

  async function handleLoadCheckpoints(id: string) {
    if (checkpointRunId === id) { setCheckpointRunId(null); return }
    try {
      const res = await fetch(`${apiUrl}/api/train/${id}/checkpoints`)
      if (res.ok) {
        const data = await res.json()
        setCheckpoints(data.checkpoints || [])
        setCheckpointRunId(id)
      }
    } catch { /* ignore */ }
  }

  function handleExportModel() {
    if (!trainingId) return
    window.open(`${apiUrl}/api/train/${trainingId}/export`, "_blank")
  }

  async function handleExportFormat(format: string) {
    if (!trainingId) return
    setIsExporting(true)
    try {
      const res = await fetch(`${apiUrl}/api/train/${trainingId}/export-format?format=${format}`, { method: "POST" })
      if (res.ok) {
        const blob = await res.blob()
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = `model.${format === 'onnx' ? 'onnx' : format === 'tflite' ? 'tflite' : format === 'coreml' ? 'mlpackage' : 'engine'}`
        a.click()
        URL.revokeObjectURL(url)
      } else {
        // Fallback: direct download link
        window.open(`${apiUrl}/api/train/${trainingId}/export?format=${format}`, "_blank")
      }
    } catch {
      window.open(`${apiUrl}/api/train/${trainingId}/export`, "_blank")
    } finally {
      setIsExporting(false)
    }
  }

  // -------------------------------------------------------------------------
  // Derived display values
  // -------------------------------------------------------------------------

  const latestMetrics = metrics[metrics.length - 1]
  const prevMetrics   = metrics[metrics.length - 2]

  const etaDisplay = (() => {
    if (!isTraining || isPaused || isPausing || currentEpoch === 0) return null
    const remaining = totalEpochs - currentEpoch
    let secsPerEpoch = 30
    // Live: batch_stats has real it/s and total_batches → most accurate
    if (batchStats && batchStats.its > 0 && batchStats.total_batches > 0) {
      secsPerEpoch = batchStats.total_batches / batchStats.its
    } else if (latestMetrics?.its) {
      const nb = batchStats?.total_batches ?? 100
      secsPerEpoch = nb / latestMetrics.its
    }
    const total = remaining * secsPerEpoch
    const h = Math.floor(total / 3600)
    const m = Math.floor((total % 3600) / 60)
    return h > 0 ? `${h}h ${m}m` : `${m}m`
  })()

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------

  return (
    <div className="p-6 space-y-6">

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Training</h1>
          <p className="text-muted-foreground mt-1">Train YOLO, RT-DETR, and RF-DETR models locally on your datasets</p>
        </div>
        <div className="flex items-center gap-2">
          {isTraining ? (
            <>
              <Button
                variant="outline"
                onClick={handlePauseResume}
                disabled={isPausing}
                className="gap-2"
              >
                {isPaused ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
                {isPausing ? "Finishing epoch…" : isPaused ? "Resume" : "Pause"}
              </Button>
              <Button variant="destructive" onClick={handleStopTraining} className="gap-2">
                <Square className="w-4 h-4" />
                Stop
              </Button>
            </>
          ) : (
            <>
              {modelPath && (
              <>
                <Button variant="outline" onClick={handleExportModel} className="gap-2">
                  <Download className="w-4 h-4" />
                  Download .pt
                </Button>
                <Select value={exportFormat} onValueChange={setExportFormat}>
                  <SelectTrigger className="w-28 h-9">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="onnx">ONNX</SelectItem>
                    <SelectItem value="tflite">TFLite</SelectItem>
                    <SelectItem value="coreml">CoreML</SelectItem>
                    <SelectItem value="engine">TensorRT</SelectItem>
                  </SelectContent>
                </Select>
                <Button
                  variant="outline"
                  onClick={() => handleExportFormat(exportFormat)}
                  disabled={isExporting}
                  className="gap-2"
                >
                  <FileDown className="w-4 h-4" />
                  {isExporting ? "Exporting…" : `Export ${exportFormat.toUpperCase()}`}
                </Button>
              </>
            )}
            {trainingId && metrics.length > 0 && (
              <Button variant="outline" onClick={handleDownloadReport} className="gap-2">
                <Package className="w-4 h-4" />
                Report
              </Button>
            )}
              <Button onClick={handleStartTraining} className="gap-2">
                <Play className="w-4 h-4" />
                Start Training
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Config */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Settings2 className="w-5 h-5 text-primary" />
              Configuration
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">

            <div className="space-y-2">
              <label className="text-sm font-medium">Dataset</label>
              {datasets.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">
                  No datasets loaded. Import one from the Datasets tab first.
                </p>
              ) : (
                <Select value={selectedDatasetId} onValueChange={setSelectedDatasetId} disabled={isTraining}>
                  <SelectTrigger><SelectValue placeholder="Select a dataset…" /></SelectTrigger>
                  <SelectContent>
                    {datasets.map(ds => (
                      <SelectItem key={ds.id} value={ds.id}>
                        <span>{ds.name}</span>
                        {ds.format && (
                          <Badge variant="outline" className="ml-2 text-[10px]">{ds.format}</Badge>
                        )}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
              {/* Compatibility notice */}
              {(() => {
                const ds = datasets.find(d => d.id === selectedDatasetId)
                if (!ds) return null
                const dsDomain  = datasetDomain(ds)
                const modelTask = getModelType(modelArch)
                const modelDomain = modelTask === "segmentation" ? "segmentation"
                                  : modelTask === "classification" ? "classification"
                                  : "detection"
                const compat = checkCompatibility(dsDomain, modelDomain)
                if (!compat) return null
                if (compat.error) return (
                  <p className="text-[11px] text-red-400 leading-snug">{compat.error}</p>
                )
                if (compat.warning) return (
                  <p className="text-[11px] text-yellow-400 leading-snug">{compat.warning}</p>
                )
                return null
              })()}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Run Name</label>
              <Input
                placeholder="e.g. cityscapes-run-1"
                value={runName}
                onChange={e => setRunName(e.target.value)}
                disabled={isTraining}
                className="text-sm"
              />
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Task Type</label>
              <Select
                value={taskType}
                onValueChange={v => {
                  setTaskType(v)
                  // Auto-select the first arch for the chosen task
                  const first = MODEL_ARCHITECTURES.find(a => a.taskType === v)
                  if (first) setModelArch(first.value)
                }}
                disabled={isTraining}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {TASK_TYPES.map(t => (
                    <SelectItem key={t.value} value={t.value}>{t.label}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Model Architecture</label>
              <Select
                value={modelArch}
                onValueChange={v => {
                  setModelArch(v)
                  // Sync task type to match the chosen arch
                  const m = MODEL_ARCHITECTURES.find(a => a.value === v)
                  if (m && m.taskType !== taskType) setTaskType(m.taskType)
                }}
                disabled={isTraining}
              >
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  {MODEL_ARCHITECTURES.filter(m => m.taskType === taskType).map(model => (
                    <SelectItem key={model.value} value={model.value}>
                      <span>{model.label}</span>
                      <Badge variant="outline" className="ml-1 text-[10px]">{model.arch}</Badge>
                      <Badge variant="secondary" className="ml-1 text-[10px]">{model.params}</Badge>
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
              {modelArch.startsWith("rfdetr") && (
                <p className="text-[11px] text-muted-foreground leading-snug">
                  RF-DETR requires a <span className="text-primary font-medium">COCO-format</span> dataset.
                  Use Convert to transform your dataset first.
                </p>
              )}
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Epochs</label>
                <span className="text-sm text-muted-foreground">{epochs}</span>
              </div>
              <Slider value={[epochs]} onValueChange={([v]) => setEpochs(v)} min={10} max={500} step={10} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Batch Size</label>
                <span className="text-sm text-muted-foreground">{batchSize}</span>
              </div>
              <Slider value={[batchSize]} onValueChange={([v]) => setBatchSize(v)} min={1} max={64} step={1} disabled={isTraining} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Image Size</label>
                <span className="text-sm text-muted-foreground">{imgSize}px</span>
              </div>
              <Slider value={[imgSize]} onValueChange={([v]) => setImgSize(v)} min={320} max={1280} step={32} disabled={isTraining} />
            </div>

            <div className="flex items-center justify-between">
              <label className="text-sm font-medium">Use Pretrained Weights</label>
              <Switch checked={usePretrained} onCheckedChange={setUsePretrained} disabled={isTraining} />
            </div>

            {/* Advanced Hyperparameters */}
            <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
              <CollapsibleTrigger asChild>
                <Button variant="ghost" size="sm" className="w-full flex items-center justify-between px-0 h-8">
                  <span className="text-sm font-medium flex items-center gap-1.5">
                    <Settings2 className="w-3.5 h-3.5" /> Advanced Hyperparams
                  </span>
                  <ChevronDown className={`w-3.5 h-3.5 transition-transform ${advancedOpen ? 'rotate-180' : ''}`} />
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="space-y-3 pt-2">
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">lr0 (initial LR)</label>
                    <Input type="number" value={lr0} onChange={e => setLr0(+e.target.value)} step={0.001} min={0.0001} max={1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">lrf (final LR)</label>
                    <Input type="number" value={lrf} onChange={e => setLrf(+e.target.value)} step={0.001} min={0.0001} max={1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                </div>
                <div className="space-y-1">
                  <label className="text-xs text-muted-foreground">Optimizer</label>
                  <Select value={optimizer} onValueChange={setOptimizer} disabled={isTraining}>
                    <SelectTrigger className="h-7 text-xs"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      {['SGD','Adam','AdamW','NAdam','RAdam','RMSProp','auto'].map(o => <SelectItem key={o} value={o} className="text-xs">{o}</SelectItem>)}
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Patience (ES)</label>
                    <Input type="number" value={patience} onChange={e => setPatience(+e.target.value)} min={0} max={300} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Warmup Epochs</label>
                    <Input type="number" value={warmupEpochs} onChange={e => setWarmupEpochs(+e.target.value)} min={0} max={10} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Weight Decay</label>
                    <Input type="number" value={weightDecay} onChange={e => setWeightDecay(+e.target.value)} step={0.0001} min={0} max={0.1} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs text-muted-foreground">Dropout</label>
                    <Input type="number" value={dropout} onChange={e => setDropout(+e.target.value)} step={0.05} min={0} max={0.9} disabled={isTraining} className="h-7 text-xs" />
                  </div>
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between"><label className="text-xs text-muted-foreground">Mosaic</label><span className="text-xs">{mosaic.toFixed(1)}</span></div>
                  <Slider value={[mosaic]} onValueChange={([v]) => setMosaic(v)} min={0} max={1} step={0.1} disabled={isTraining} />
                </div>
                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground">HSV-H</label><span>{hsvH.toFixed(3)}</span></div>
                    <Slider value={[hsvH]} onValueChange={([v]) => setHsvH(v)} min={0} max={0.1} step={0.001} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground">HSV-S</label><span>{hsvS.toFixed(2)}</span></div>
                    <Slider value={[hsvS]} onValueChange={([v]) => setHsvS(v)} min={0} max={1} step={0.05} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground">HSV-V</label><span>{hsvV.toFixed(2)}</span></div>
                    <Slider value={[hsvV]} onValueChange={([v]) => setHsvV(v)} min={0} max={1} step={0.05} disabled={isTraining} />
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground">FlipLR</label><span>{fliplr.toFixed(1)}</span></div>
                    <Slider value={[fliplr]} onValueChange={([v]) => setFliplr(v)} min={0} max={1} step={0.1} disabled={isTraining} />
                  </div>
                  <div className="space-y-1">
                    <div className="flex justify-between"><label className="text-muted-foreground">FlipUD</label><span>{flipud.toFixed(1)}</span></div>
                    <Slider value={[flipud]} onValueChange={([v]) => setFlipud(v)} min={0} max={1} step={0.1} disabled={isTraining} />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">Cosine LR Schedule</label>
                  <Switch checked={cosLr} onCheckedChange={setCosLr} disabled={isTraining} />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs text-muted-foreground">Mixed Precision (AMP)</label>
                  <Switch checked={amp} onCheckedChange={setAmp} disabled={isTraining} />
                </div>
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <label className="text-xs text-muted-foreground">Save Checkpoint Every N Epochs</label>
                    <span className="text-xs">{savePeriod === 0 ? "off" : `every ${savePeriod}`}</span>
                  </div>
                  <Slider value={[savePeriod]} onValueChange={([v]) => setSavePeriod(v)} min={0} max={50} step={1} disabled={isTraining} />
                  <p className="text-[10px] text-muted-foreground/60">0 = only best &amp; last. Set to save epoch5.pt, epoch10.pt, etc.</p>
                </div>
              </CollapsibleContent>
            </Collapsible>

            {gpuWarning && (
              <div className="rounded-md bg-yellow-500/10 border border-yellow-500/30 p-3 text-xs text-yellow-300 space-y-2">
                <p className="font-semibold">GPU detected but CUDA PyTorch not installed</p>
                <p className="text-yellow-400/80">Training will run on CPU (slow). Install CUDA PyTorch to use your GPU.</p>
                <Button
                  size="sm"
                  variant="outline"
                  className="w-full h-7 text-xs border-yellow-500/40 text-yellow-300 hover:bg-yellow-500/10"
                  onClick={handleInstallCudaTorch}
                  disabled={installingCuda}
                >
                  {installingCuda ? "Installing… (may take a few minutes)" : "Install CUDA PyTorch Automatically"}
                </Button>
              </div>
            )}

            {error && (
              <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-400 break-all">
                {error}
              </div>
            )}

            {modelPath && (
              <div className={`rounded-md p-3 text-xs break-all space-y-1 ${
                statusLabel === "Failed"
                  ? "bg-yellow-500/10 border border-yellow-500/30 text-yellow-400"
                  : "bg-green-500/10 border border-green-500/30 text-green-400"
              }`}>
                <p className="font-semibold">
                  {statusLabel === "Failed" ? "Partial weights available" : "Model saved"}
                </p>
                <p className="font-mono opacity-80">{modelPath}</p>
                <p>Use the &ldquo;Download .pt&rdquo; button above to save the model file.</p>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Right column */}
        <div className="lg:col-span-2 space-y-6">

          {/* Progress */}
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Activity className="w-5 h-5 text-primary" />
                Training Progress
                <div className="ml-auto flex items-center gap-2">
                  {deviceInfo && (
                    <Badge variant="outline" className="text-xs font-mono gap-1">
                      <Cpu className="w-3 h-3" />{deviceInfo.length > 30 ? deviceInfo.slice(0, 30) + "…" : deviceInfo}
                    </Badge>
                  )}
                  {liveGpuMem != null && (
                    <Badge variant="outline" className="text-xs font-mono gap-1">
                      <HardDrive className="w-3 h-3" />{liveGpuMem.toFixed(2)} GB VRAM
                    </Badge>
                  )}
                  {statusLabel && (
                    <Badge variant="secondary" className="text-xs">{statusLabel}</Badge>
                  )}
                </div>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-muted-foreground">
                    Epoch {currentEpoch} / {totalEpochs}
                    {batchStats && isTraining && (
                      <span className="ml-2 text-xs text-muted-foreground/60">
                        · Batch {batchStats.batch} / {batchStats.total_batches}
                      </span>
                    )}
                  </span>
                  <span className="text-sm text-muted-foreground flex items-center gap-3">
                    {batchStats && isTraining && batchStats.its > 0 ? (
                      <span className="font-mono text-xs">⚡ {batchStats.its.toFixed(1)} it/s</span>
                    ) : latestMetrics?.its ? (
                      <span className="font-mono text-xs">⚡ {latestMetrics.its.toFixed(1)} it/s</span>
                    ) : null}
                    {etaDisplay && `ETA: ${etaDisplay}`}
                    {isPaused && "Training paused — checkpoint saved"}
                  </span>
                </div>
                <Progress value={progress} className="h-2" />

                {/* Live batch losses during training */}
                {batchStats && isTraining && Object.keys(batchStats.losses).length > 0 && (
                  <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs font-mono text-muted-foreground/80">
                    <span className="text-muted-foreground/40">Live:</span>
                    {Object.entries(batchStats.losses).map(([key, val]) => (
                      <span key={key}>
                        {key.replace(/_loss$/, "")}
                        <span className="ml-1 text-foreground/60">{val.toFixed(4)}</span>
                      </span>
                    ))}
                  </div>
                )}

                {/* Train losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Train Losses</p>
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Box Loss"  value={latestMetrics?.train_box_loss} prev={prevMetrics?.train_box_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Cls Loss"  value={latestMetrics?.train_cls_loss} prev={prevMetrics?.train_cls_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="DFL Loss"  value={latestMetrics?.train_dfl_loss} prev={prevMetrics?.train_dfl_loss} lowerIsBetter decimals={4} />
                  </div>
                </div>

                {/* Val losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Val Losses</p>
                  <div className="grid grid-cols-3 gap-3">
                    <MetricCard label="Val Box"  value={latestMetrics?.val_box_loss} prev={prevMetrics?.val_box_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Val Cls"  value={latestMetrics?.val_cls_loss} prev={prevMetrics?.val_cls_loss} lowerIsBetter decimals={4} />
                    <MetricCard label="Val DFL"  value={latestMetrics?.val_dfl_loss} prev={prevMetrics?.val_dfl_loss} lowerIsBetter decimals={4} />
                  </div>
                </div>

                {/* mAP + P/R */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Validation Metrics</p>
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <MetricCard label="mAP@50"    value={latestMetrics?.mAP50}     prev={prevMetrics?.mAP50}     lowerIsBetter={false} decimals={4} />
                    <MetricCard label="mAP@50-95" value={latestMetrics?.mAP50_95}  prev={prevMetrics?.mAP50_95}  lowerIsBetter={false} decimals={4} />
                    <MetricCard label="Precision" value={latestMetrics?.precision} prev={prevMetrics?.precision} lowerIsBetter={false} decimals={4} />
                    <MetricCard label="Recall"    value={latestMetrics?.recall}    prev={prevMetrics?.recall}    lowerIsBetter={false} decimals={4} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Training Curves */}
          {metrics.length > 1 && (
            <Card className="border-border/50 bg-card/50 backdrop-blur">
              <CardHeader className="pb-2">
                <CardTitle className="text-lg flex items-center gap-2">
                  <LineChartIcon className="w-5 h-5 text-primary" />
                  Training Curves
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-5">
                {/* Train losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Train Losses</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="train_box_loss" name="Box" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="train_cls_loss" name="Cls" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                      <Line type="monotone" dataKey="train_dfl_loss" name="DFL" stroke="#8b5cf6" strokeWidth={1.5} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Val losses */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Val Losses</p>
                  <ResponsiveContainer width="100%" height={150}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="val_box_loss" name="Val Box" stroke="#60a5fa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                      <Line type="monotone" dataKey="val_cls_loss" name="Val Cls" stroke="#fbbf24" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                      <Line type="monotone" dataKey="val_dfl_loss" name="Val DFL" stroke="#a78bfa" strokeWidth={1.5} dot={false} strokeDasharray="4 2" />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* mAP chart */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">mAP over epochs</p>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="mAP50"    name="mAP@50"    stroke="#10b981" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="mAP50_95" name="mAP@50-95" stroke="#06b6d4" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
                {/* Precision / Recall chart */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Precision &amp; Recall</p>
                  <ResponsiveContainer width="100%" height={140}>
                    <LineChart data={metrics} margin={{ top: 4, right: 8, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} domain={[0, 1]} />
                      <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 6, fontSize: 11 }} />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      <Line type="monotone" dataKey="precision" name="Precision" stroke="#f97316" strokeWidth={2} dot={false} />
                      <Line type="monotone" dataKey="recall"    name="Recall"    stroke="#ec4899" strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {/* System stats */}
          <div className="grid grid-cols-3 gap-4">
            <SystemStat
              icon={<Cpu className="w-5 h-5 text-primary" />}
              label="CPU"
              value={`${cpuUsage.toFixed(0)}%`}
              progress={cpuUsage}
            />
            <SystemStat
              icon={<HardDrive className="w-5 h-5 text-primary" />}
              label={liveGpuMem != null ? "VRAM Used" : gpuUsage !== null ? "GPU Memory" : "RAM"}
              value={liveGpuMem != null
                ? `${liveGpuMem.toFixed(2)} GB`
                : `${memoryUsage.toFixed(0)}%`}
              progress={liveGpuMem == null ? memoryUsage : undefined}
            />
            <SystemStat
              icon={<Activity className="w-5 h-5 text-primary" />}
              label={gpuUsage !== null ? `GPU${gpuName ? ` (${gpuName.split(" ").pop()})` : ""}` : "GPU"}
              value={gpuUsage !== null ? `${gpuUsage.toFixed(0)}%` : "—"}
              progress={gpuUsage ?? undefined}
            />
          </div>

          {/* Logs */}
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader className="pb-2">
              <CardTitle className="text-lg flex items-center gap-2">
                <Terminal className="w-5 h-5 text-primary" />
                Logs
                <span className="ml-auto text-xs font-normal text-muted-foreground">{logs.length} lines</span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <ScrollArea className="h-[280px] w-full rounded border border-border/50 bg-black/30 p-3">
                {logs.length === 0 ? (
                  <p className="text-sm text-muted-foreground font-mono">Training logs will appear here…</p>
                ) : (
                  <div className="space-y-0.5 font-mono text-[11px] leading-5">
                    {logs.map((log, i) => (
                      <div key={i} className={
                        log.type === "error"   ? "text-red-400"    :
                        log.type === "warning" ? "text-yellow-300" :
                        log.type === "success" ? "text-green-400"  : "text-slate-300"
                      }>
                        <span className="text-slate-500 select-none">[{log.timestamp}]</span>{" "}{log.message}
                      </div>
                    ))}
                    <div ref={logsEndRef} />
                  </div>
                )}
              </ScrollArea>
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Past Runs */}
      {pastRuns.length > 0 && (
        <Collapsible>
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CollapsibleTrigger asChild>
              <CardHeader className="cursor-pointer hover:bg-muted/20 rounded-t-lg pb-3">
                <CardTitle className="text-base flex items-center gap-2">
                  <Package className="w-4 h-4 text-primary" />
                  Past Runs
                  <Badge variant="outline" className="ml-1 text-xs">{pastRuns.length}</Badge>
                  <ChevronDown className="w-4 h-4 ml-auto text-muted-foreground" />
                </CardTitle>
              </CardHeader>
            </CollapsibleTrigger>
            <CollapsibleContent>
              <CardContent className="pt-0 space-y-2">
                {pastRuns.map(run => (
                  <div key={run.id} className="rounded-lg border border-border/40 bg-background/40 overflow-hidden">
                    <div className="flex items-center gap-3 p-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <span className="text-sm font-medium truncate">{run.name}</span>
                          <Badge
                            variant="outline"
                            className={`text-[10px] shrink-0 ${
                              run.status === "completed"   ? "text-green-400 border-green-400/30" :
                              run.status === "failed"      ? "text-red-400 border-red-400/30" :
                              run.status === "paused"      ? "text-yellow-400 border-yellow-400/30" :
                              run.status === "interrupted" ? "text-orange-400 border-orange-400/30" :
                              run.status === "running" || run.status === "pausing" ? "text-blue-400 border-blue-400/30" :
                              "text-muted-foreground"
                            }`}
                          >
                            {run.status}
                          </Badge>
                          <span className="text-[10px] text-muted-foreground shrink-0">{run.model_type}</span>
                        </div>
                        <div className="flex items-center gap-2 mt-1">
                          <div className="flex-1 h-1 rounded bg-muted overflow-hidden">
                            <div className="h-full bg-primary/60 rounded" style={{ width: `${run.progress}%` }} />
                          </div>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {run.current_epoch}/{run.total_epochs} ep
                          </span>
                          <span className="text-[10px] text-muted-foreground shrink-0">
                            {new Date(run.started_at).toLocaleDateString()}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-1 shrink-0">
                        {/* Download best model */}
                        {run.model_path && (
                          <Button size="sm" variant="ghost" className="h-7 w-7 p-0" title="Download best model"
                            onClick={() => window.open(`${apiUrl}/api/train/${run.id}/export`, "_blank")}>
                            <Download className="w-3.5 h-3.5" />
                          </Button>
                        )}
                        {/* Show checkpoints */}
                        {(run.status === "completed" || run.status === "paused" || run.status === "stopped") && (
                          <Button size="sm" variant="ghost"
                            className={`h-7 w-7 p-0 ${checkpointRunId === run.id ? "text-primary" : ""}`}
                            title="View checkpoints"
                            onClick={() => handleLoadCheckpoints(run.id)}>
                            <FolderOpen className="w-3.5 h-3.5" />
                          </Button>
                        )}
                        {/* Load & Resume */}
                        {(run.status === "paused" || run.status === "interrupted") && run.id !== trainingId && (
                          <Button size="sm" variant="outline" className="gap-1 h-7 text-xs"
                            onClick={() => {
                              setTrainingId(run.id)
                              setIsPaused(true)
                              setIsPausing(false)
                              setIsTraining(false)
                              setProgress(run.progress)
                              setCurrentEpoch(run.current_epoch)
                              setTotalEpochs(run.total_epochs)
                              setStatusLabel(run.status === "interrupted" ? "Interrupted" : "Paused")
                              setMetrics([])
                              setLogs([])
                              seenLogCount.current = 0
                              if (run.model_path) setModelPath(run.model_path)
                            }}>
                            <Play className="w-3 h-3" />
                            Resume
                          </Button>
                        )}
                        {/* Delete (only for finished runs) */}
                        {run.status !== "running" && run.status !== "starting" && run.status !== "pausing" && (
                          <Button size="sm" variant="ghost" className="h-7 w-7 p-0 text-muted-foreground hover:text-red-400"
                            title="Delete run" onClick={() => handleDeleteRun(run.id)}>
                            <Trash2 className="w-3.5 h-3.5" />
                          </Button>
                        )}
                      </div>
                    </div>
                    {/* Checkpoint list inline */}
                    {checkpointRunId === run.id && checkpoints.length > 0 && (
                      <div className="border-t border-border/30 px-3 py-2 bg-muted/20">
                        <p className="text-[10px] text-muted-foreground mb-1.5 font-medium">Checkpoints</p>
                        <div className="flex flex-wrap gap-1.5">
                          {checkpoints.map(cp => (
                            <button key={cp.name}
                              className="text-[10px] px-2 py-0.5 rounded bg-primary/10 text-primary hover:bg-primary/20 border border-primary/20 font-mono"
                              onClick={() => window.open(`${apiUrl}/api/train/${run.id}/checkpoint/${cp.name}`, "_blank")}
                              title={`${cp.size_mb} MB`}>
                              {cp.name}
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                    {checkpointRunId === run.id && checkpoints.length === 0 && (
                      <div className="border-t border-border/30 px-3 py-2 text-[10px] text-muted-foreground">
                        No epoch checkpoints found. Set "Save Every N Epochs" &gt; 0 before training to generate them.
                      </div>
                    )}
                  </div>
                ))}
              </CardContent>
            </CollapsibleContent>
          </Card>
        </Collapsible>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function MetricCard({
  label, value, prev, lowerIsBetter, decimals = 4,
}: {
  label: string
  value?: number
  prev?: number
  lowerIsBetter: boolean
  decimals?: number
}) {
  const improved =
    value != null && prev != null
      ? lowerIsBetter ? value < prev : value > prev
      : null

  return (
    <div className="text-center p-3 rounded-lg bg-muted/30">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className="text-lg font-semibold flex items-center justify-center gap-1">
        {value != null ? value.toFixed(decimals) : "—"}
        {improved === true  && <TrendingDown className="w-3 h-3 text-green-500" />}
        {improved === false && <TrendingUp   className="w-3 h-3 text-red-400"   />}
      </p>
    </div>
  )
}

function SystemStat({
  icon, label, value, progress,
}: {
  icon: React.ReactNode
  label: string
  value: string
  progress?: number
}) {
  return (
    <Card className="border-border/50 bg-card/50 backdrop-blur">
      <CardContent className="pt-4">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center">
            {icon}
          </div>
          <div>
            <p className="text-xs text-muted-foreground">{label}</p>
            <p className="text-lg font-semibold">{value}</p>
          </div>
        </div>
        {progress !== undefined && (
          <Progress value={progress} className="h-1 mt-3" />
        )}
      </CardContent>
    </Card>
  )
}