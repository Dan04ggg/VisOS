"use client"
// ── CHANGES vs PREVIOUS VERSION ───────────────────────────────────────────────
// • filteredLoaded filter: guarded m.name with `(m.name ?? "")` to prevent
//   TypeError crash when backend returns a model with name: null/undefined.
// ──────────────────────────────────────────────────────────────────────────────

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { 
  Upload, 
  Download,
  Trash2,
  Search,
  Box,
  Cpu,
  FileCode,
  CheckCircle2,
  HardDrive,
  MoreVertical,
  RefreshCw,
  AlertCircle
} from "lucide-react"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

// Shape returned by GET /api/models
interface BackendModel {
  id: string
  name: string
  type: string        // "yolo" | "sam" | "sam2" | "detr" | "unknown"
  loaded: boolean
  pretrained?: boolean
  downloaded?: boolean
  path?: string
  classes?: string[]
}

interface ModelsViewProps {
  apiUrl?: string
}

// Pretrained models available from the backend – mirrors model_integration.py
const PRETRAINED_CATALOG: Array<{
  id: string
  name: string
  type: string
  task: string
  sizeLabel: string
  requiresHfToken?: boolean
}> = [
  // ── YOLOv8 ──────────────────────────────────────────────────────────────────
  { id: "yolov8n",     name: "YOLOv8 Nano",      type: "yolo",  task: "detection",          sizeLabel: "6 MB"    },
  { id: "yolov8s",     name: "YOLOv8 Small",      type: "yolo",  task: "detection",          sizeLabel: "22 MB"   },
  { id: "yolov8m",     name: "YOLOv8 Medium",     type: "yolo",  task: "detection",          sizeLabel: "52 MB"   },
  { id: "yolov8l",     name: "YOLOv8 Large",      type: "yolo",  task: "detection",          sizeLabel: "87 MB"   },
  { id: "yolov8x",     name: "YOLOv8 XLarge",     type: "yolo",  task: "detection",          sizeLabel: "136 MB"  },
  // ── YOLOv5 ──────────────────────────────────────────────────────────────────
  { id: "yolov5n",     name: "YOLOv5 Nano",       type: "yolo",  task: "detection",          sizeLabel: "4 MB"    },
  { id: "yolov5s",     name: "YOLOv5 Small",       type: "yolo",  task: "detection",          sizeLabel: "14 MB"   },
  // ── YOLOv9 ──────────────────────────────────────────────────────────────────
  { id: "yolov9n",     name: "YOLOv9 Nano",       type: "yolo",  task: "detection",          sizeLabel: "4 MB"    },
  { id: "yolov9s",     name: "YOLOv9 Small",      type: "yolo",  task: "detection",          sizeLabel: "14 MB"   },
  { id: "yolov9m",     name: "YOLOv9 Medium",     type: "yolo",  task: "detection",          sizeLabel: "40 MB"   },
  { id: "yolov9c",     name: "YOLOv9 Compact",    type: "yolo",  task: "detection",          sizeLabel: "51 MB"   },
  { id: "yolov9e",     name: "YOLOv9 Extended",   type: "yolo",  task: "detection",          sizeLabel: "116 MB"  },
  // ── YOLOv10 ─────────────────────────────────────────────────────────────────
  { id: "yolov10n",    name: "YOLOv10 Nano",      type: "yolo",  task: "detection",          sizeLabel: "5 MB"    },
  { id: "yolov10s",    name: "YOLOv10 Small",     type: "yolo",  task: "detection",          sizeLabel: "15 MB"   },
  { id: "yolov10m",    name: "YOLOv10 Medium",    type: "yolo",  task: "detection",          sizeLabel: "32 MB"   },
  { id: "yolov10b",    name: "YOLOv10 Balanced",  type: "yolo",  task: "detection",          sizeLabel: "40 MB"   },
  { id: "yolov10l",    name: "YOLOv10 Large",     type: "yolo",  task: "detection",          sizeLabel: "50 MB"   },
  { id: "yolov10x",    name: "YOLOv10 XLarge",    type: "yolo",  task: "detection",          sizeLabel: "61 MB"   },
  // ── YOLO11 ──────────────────────────────────────────────────────────────────
  { id: "yolo11n",     name: "YOLO11 Nano",       type: "yolo",  task: "detection",          sizeLabel: "5 MB"    },
  { id: "yolo11s",     name: "YOLO11 Small",      type: "yolo",  task: "detection",          sizeLabel: "19 MB"   },
  { id: "yolo11m",     name: "YOLO11 Medium",     type: "yolo",  task: "detection",          sizeLabel: "40 MB"   },
  { id: "yolo11l",     name: "YOLO11 Large",      type: "yolo",  task: "detection",          sizeLabel: "51 MB"   },
  { id: "yolo11x",     name: "YOLO11 XLarge",     type: "yolo",  task: "detection",          sizeLabel: "114 MB"  },
  // ── YOLO12 ──────────────────────────────────────────────────────────────────
  { id: "yolo12n",     name: "YOLO12 Nano",       type: "yolo",  task: "detection",          sizeLabel: "5 MB"    },
  { id: "yolo12s",     name: "YOLO12 Small",      type: "yolo",  task: "detection",          sizeLabel: "19 MB"   },
  { id: "yolo12m",     name: "YOLO12 Medium",     type: "yolo",  task: "detection",          sizeLabel: "40 MB"   },
  { id: "yolo12l",     name: "YOLO12 Large",      type: "yolo",  task: "detection",          sizeLabel: "51 MB"   },
  { id: "yolo12x",     name: "YOLO12 XLarge",     type: "yolo",  task: "detection",          sizeLabel: "114 MB"  },
  // ── RT-DETR (Ultralytics) ────────────────────────────────────────────────────
  { id: "rtdetr-l",    name: "RT-DETR Large",     type: "rtdetr", task: "detection",         sizeLabel: "~65 MB"  },
  { id: "rtdetr-x",    name: "RT-DETR XLarge",    type: "rtdetr", task: "detection",         sizeLabel: "~135 MB" },
  // ── RF-DETR (Roboflow) ───────────────────────────────────────────────────────
  { id: "rfdetr_base",  name: "RF-DETR Base",     type: "rfdetr", task: "detection",         sizeLabel: "~160 MB" },
  { id: "rfdetr_large", name: "RF-DETR Large",    type: "rfdetr", task: "detection",         sizeLabel: "~500 MB" },
  // ── Zero-shot / Open-vocabulary ──────────────────────────────────────────────
  { id: "yoloworld_s",  name: "YOLO-World S",     type: "yoloworld",    task: "zero-shot detection", sizeLabel: "~28 MB"  },
  { id: "yoloworld_m",  name: "YOLO-World M",     type: "yoloworld",    task: "zero-shot detection", sizeLabel: "~86 MB"  },
  { id: "yoloworld_l",  name: "YOLO-World L",     type: "yoloworld",    task: "zero-shot detection", sizeLabel: "~141 MB" },
  { id: "groundingdino_t", name: "GroundingDINO Tiny", type: "groundingdino", task: "zero-shot detection", sizeLabel: "~694 MB" },
  { id: "groundingdino_b", name: "GroundingDINO Base", type: "groundingdino", task: "zero-shot detection", sizeLabel: "~914 MB" },
  { id: "owlvit_base_patch32",  name: "OWL-ViT Base (Patch32)",  type: "owlvit", task: "zero-shot detection", sizeLabel: "~590 MB" },
  { id: "owlvit_base_patch16",  name: "OWL-ViT Base (Patch16)",  type: "owlvit", task: "zero-shot detection", sizeLabel: "~590 MB" },
  { id: "owlvit_large_patch14", name: "OWL-ViT Large (Patch14)", type: "owlvit", task: "zero-shot detection", sizeLabel: "~1.6 GB" },
  // ── SAM 1 ────────────────────────────────────────────────────────────────────
  { id: "sam_vit_b",   name: "SAM ViT-B",         type: "sam",   task: "segmentation",       sizeLabel: "375 MB"  },
  { id: "sam_vit_l",   name: "SAM ViT-L",         type: "sam",   task: "segmentation",       sizeLabel: "1.2 GB"  },
  // ── SAM 2 ────────────────────────────────────────────────────────────────────
  { id: "sam2_tiny",   name: "SAM 2 Tiny",        type: "sam2",  task: "segmentation",       sizeLabel: "38 MB"   },
  { id: "sam2_small",  name: "SAM 2 Small",       type: "sam2",  task: "segmentation",       sizeLabel: "46 MB"   },
  { id: "sam2_base",   name: "SAM 2 Base+",       type: "sam2",  task: "segmentation",       sizeLabel: "80 MB"   },
  { id: "sam2_large",  name: "SAM 2 Large",       type: "sam2",  task: "segmentation",       sizeLabel: "224 MB"  },
  // ── SAM 2.1 ──────────────────────────────────────────────────────────────────
  { id: "sam21_tiny",  name: "SAM 2.1 Tiny",      type: "sam2",  task: "segmentation",       sizeLabel: "38 MB"   },
  { id: "sam21_small", name: "SAM 2.1 Small",     type: "sam2",  task: "segmentation",       sizeLabel: "46 MB"   },
  { id: "sam21_base",  name: "SAM 2.1 Base+",     type: "sam2",  task: "segmentation",       sizeLabel: "80 MB"   },
  { id: "sam21_large", name: "SAM 2.1 Large",     type: "sam2",  task: "segmentation",       sizeLabel: "224 MB"  },
  // ── SAM 3 (gated on HuggingFace) ────────────────────────────────────────────
  { id: "sam3",        name: "SAM 3",             type: "sam3",  task: "segmentation",       sizeLabel: "~3.5 GB", requiresHfToken: true },
]

export function ModelsView({ apiUrl = "http://localhost:8000" }: ModelsViewProps) {
  const [models, setModels] = useState<BackendModel[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingModelId, setLoadingModelId] = useState<string | null>(null)
  const [downloadingIds, setDownloadingIds] = useState<Set<string>>(new Set())
  const [downloadProgress, setDownloadProgress] = useState<Record<string, number>>({})
  const [error, setError] = useState<string | null>(null)
  const [hfToken, setHfToken] = useState("")
  const fileInputRef = useRef<HTMLInputElement>(null)

  // -------------------------------------------------------------------------
  // Fetch models from backend on mount
  // -------------------------------------------------------------------------
  const fetchModels = async () => {
    setIsLoading(true)
    setError(null)
    try {
      const res = await fetch(`${apiUrl}/api/models`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      const data = await res.json()
      setModels(data.models ?? [])
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setIsLoading(false)
    }
  }

  useEffect(() => { fetchModels() }, [apiUrl])

  // -------------------------------------------------------------------------
  // Load a model via the backend
  // -------------------------------------------------------------------------
  const handleLoadModel = async (model: BackendModel) => {
    setLoadingModelId(model.id)
    try {
      const formData = new FormData()
      formData.append("model_type", model.type)
      if (model.path) formData.append("model_name", model.name)
      if (model.pretrained) formData.append("pretrained", model.id)

      const res = await fetch(`${apiUrl}/api/models/load`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      await fetchModels()
    } catch (err) {
      setError(`Load failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setLoadingModelId(null)
    }
  }

  // Backend has no unload endpoint — mark locally and refresh
  // (The backend keeps models in memory; closing the process clears them.)
  const handleUnloadModel = async (model: BackendModel) => {
    // Optimistic local update — backend has no unload route
    setModels(prev => prev.map(m => m.id === model.id ? { ...m, loaded: false } : m))
  }

  // Backend has no delete endpoint for loaded models — remove from local list
  const handleDeleteModel = (modelId: string) => {
    setModels(prev => prev.filter(m => m.id !== modelId))
  }

  // -------------------------------------------------------------------------
  // Download (load pretrained) via the backend
  // -------------------------------------------------------------------------
  const handleDownloadPreset = async (preset: typeof PRETRAINED_CATALOG[0]) => {
    setDownloadingIds(prev => new Set([...prev, preset.id]))
    setDownloadProgress(prev => ({ ...prev, [preset.id]: 5 }))
    setError(null)

    try {
      // Start background download
      const formData = new FormData()
      formData.append("model_type", preset.type)
      if (preset.requiresHfToken && hfToken) formData.append("hf_token", hfToken)
      formData.append("pretrained", preset.id)
      const startRes = await fetch(`${apiUrl}/api/models/download`, { method: "POST", body: formData })
      if (!startRes.ok) {
        const err = await startRes.json().catch(() => ({ detail: startRes.statusText }))
        throw new Error(err.detail ?? startRes.statusText)
      }

      // Poll for real progress
      await new Promise<void>((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const statusRes = await fetch(`${apiUrl}/api/models/download-status/${preset.id}`)
            if (!statusRes.ok) return
            const status = await statusRes.json()
            if (status.status === "done") {
              setDownloadProgress(prev => ({ ...prev, [preset.id]: 100 }))
              clearInterval(poll)
              resolve()
            } else if (status.status === "error") {
              clearInterval(poll)
              reject(new Error(status.error ?? "Download failed"))
            } else {
              setDownloadProgress(prev => ({
                ...prev,
                [preset.id]: Math.min((prev[preset.id] ?? 5) + 3, 90)
              }))
            }
          } catch { /* network blip, keep polling */ }
        }, 1000)
      })

      await fetchModels()
    } catch (err) {
      setError(`Download failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingIds(prev => { const n = new Set(prev); n.delete(preset.id); return n })
      setDownloadProgress(prev => { const n = { ...prev }; delete n[preset.id]; return n })
    }
  }

  // -------------------------------------------------------------------------
  // Import custom model file
  // -------------------------------------------------------------------------
  const handleImportModel = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setLoadingModelId("import")
    setError(null)
    try {
      const formData = new FormData()
      formData.append("model_file", file)
      formData.append("model_type", "yolo")
      formData.append("model_name", file.name)

      const res = await fetch(`${apiUrl}/api/models/load`, {
        method: "POST",
        body: formData,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: res.statusText }))
        throw new Error(err.detail ?? res.statusText)
      }
      await fetchModels()
    } catch (err) {
      setError(`Import failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setLoadingModelId(null)
      // Reset input so same file can be re-selected
      if (fileInputRef.current) fileInputRef.current.value = ""
    }
  }

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------
  const getTypeColor = (type: string) => {
    switch (type) {
      case "yolo":   return "bg-blue-500/10 text-blue-500"
      case "sam":
      case "sam2":   return "bg-purple-500/10 text-purple-500"
      case "sam3":   return "bg-pink-500/10 text-pink-500"
      case "rfdetr": return "bg-emerald-500/10 text-emerald-500"
      default:       return "bg-orange-500/10 text-orange-500"
    }
  }

  const TypeIcon = ({ type }: { type: string }) => {
    if (type === "yolo") return <Box className="w-6 h-6" />
    if (type === "sam" || type === "sam2" || type === "sam3") return <FileCode className="w-6 h-6" />
    if (type === "rfdetr") return <Cpu className="w-6 h-6" />
    return <Cpu className="w-6 h-6" />
  }

  // "Your Models" = anything loaded in memory OR downloaded to disk (pretrained or not)
  const loadedModels = models.filter(m => m.loaded || (m as any).downloaded)

  const filteredLoaded = loadedModels.filter(m =>
    (m.name ?? "").toLowerCase().includes(searchQuery.toLowerCase())
  )

  // Which pretrained IDs are downloaded (on disk) or loaded in memory?
  const downloadedPretrainedIds = new Set(
    models.filter(m => m.pretrained && (m.loaded || (m as any).downloaded)).map(m => m.id)
  )

  return (
    <div className="p-6 space-y-6">
      {/* Hidden file input for import */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".pt,.pth,.onnx,.weights"
        className="hidden"
        onChange={handleImportModel}
      />

      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-foreground">Models</h1>
          <p className="text-muted-foreground mt-1">Manage and load models for auto-annotation</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" size="icon" onClick={fetchModels} disabled={isLoading}>
            <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          </Button>
          <Button className="gap-2" onClick={() => fileInputRef.current?.click()} disabled={loadingModelId === "import"}>
            <Upload className="w-4 h-4" />
            {loadingModelId === "import" ? "Importing…" : "Import Model"}
          </Button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-md bg-destructive/10 border border-destructive/30 p-3 text-sm text-destructive">
          <AlertCircle className="w-4 h-4 shrink-0" />
          {error}
          <button className="ml-auto text-xs underline" onClick={() => setError(null)}>Dismiss</button>
        </div>
      )}

      <div className="relative max-w-md">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search models..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Your Models */}
        <div className="lg:col-span-2">
          <Card className="border-border/50 bg-card/50 backdrop-blur">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Box className="w-5 h-5 text-primary" />
                Your Models
              </CardTitle>
              <CardDescription>Loaded and imported models available for auto-annotation</CardDescription>
            </CardHeader>
            <CardContent>
              {isLoading && filteredLoaded.length === 0 ? (
                <div className="flex items-center justify-center py-12 text-muted-foreground gap-2">
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Loading models…
                </div>
              ) : filteredLoaded.length === 0 ? (
                <div className="flex flex-col items-center justify-center py-12 text-center">
                  <div className="w-16 h-16 rounded-full bg-muted/50 flex items-center justify-center mb-4">
                    <Box className="w-8 h-8 text-muted-foreground" />
                  </div>
                  <p className="text-muted-foreground mb-4">No models loaded yet</p>
                  <p className="text-sm text-muted-foreground/70 mb-4">
                    Download a pretrained model or import a .pt file
                  </p>
                  <Button variant="outline" className="gap-2" onClick={() => fileInputRef.current?.click()}>
                    <Upload className="w-4 h-4" />
                    Import a Model
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {filteredLoaded.map(model => (
                    <div
                      key={model.id}
                      className={`flex items-center gap-4 p-4 rounded-lg border transition-all ${
                        model.loaded ? "border-primary/50 bg-primary/5" : "border-border/50 bg-muted/20"
                      }`}
                    >
                      <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${getTypeColor(model.type)}`}>
                        <TypeIcon type={model.type} />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="font-medium">{model.name}</span>
                          {model.loaded && (
                            <Badge className="text-xs bg-green-500/10 text-green-500">
                              <CheckCircle2 className="w-3 h-3 mr-1" />
                              Loaded
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-3 text-xs text-muted-foreground">
                          <Badge variant="outline" className={getTypeColor(model.type)}>
                            {model.type.toUpperCase()}
                          </Badge>
                          {model.path && (
                            <span className="flex items-center gap-1 truncate max-w-[200px]">
                              <HardDrive className="w-3 h-3 shrink-0" />
                              {model.path}
                            </span>
                          )}
                        </div>
                      </div>

                      <div className="flex items-center gap-2">
                        {model.loaded ? (
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={() => handleUnloadModel(model)}
                            disabled={loadingModelId === model.id}
                          >
                            Unload
                          </Button>
                        ) : (
                          <Button
                            size="sm"
                            onClick={() => handleLoadModel(model)}
                            disabled={loadingModelId === model.id}
                          >
                            {loadingModelId === model.id ? (
                              <RefreshCw className="w-3 h-3 animate-spin" />
                            ) : "Load"}
                          </Button>
                        )}

                        <DropdownMenu>
                          <DropdownMenuTrigger asChild>
                            <Button variant="ghost" size="icon">
                              <MoreVertical className="w-4 h-4" />
                            </Button>
                          </DropdownMenuTrigger>
                          <DropdownMenuContent align="end">
                            <DropdownMenuSeparator />
                            <DropdownMenuItem
                              className="text-destructive"
                              onClick={() => handleDeleteModel(model.id)}
                            >
                              <Trash2 className="w-4 h-4 mr-2" />
                              Remove
                            </DropdownMenuItem>
                          </DropdownMenuContent>
                        </DropdownMenu>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Download pretrained models */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Download className="w-5 h-5 text-primary" />
              Download Models
            </CardTitle>
            <CardDescription>Download and load pretrained models via the backend</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[520px] overflow-auto">
              {PRETRAINED_CATALOG.map(preset => {
                const isDownloading = downloadingIds.has(preset.id)
                const isDownloaded = downloadedPretrainedIds.has(preset.id)
                const isLoaded = models.some(m => m.id === preset.id && m.loaded)
                const progress = downloadProgress[preset.id] ?? 0

                const needsToken = preset.requiresHfToken && !isDownloaded && !isLoaded

                return (
                  <div
                    key={preset.id}
                    className="flex flex-col gap-2 p-3 rounded-lg border border-border/50 bg-muted/20 hover:bg-muted/30 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2">
                          <p className="font-medium text-sm">{preset.name}</p>
                          {preset.requiresHfToken && (
                            <Badge variant="outline" className="text-[10px] text-amber-500 border-amber-500/50 shrink-0">HF Auth</Badge>
                          )}
                        </div>
                        <div className="flex items-center gap-2 text-xs text-muted-foreground">
                          <span>{preset.sizeLabel}</span>
                          <span className="text-muted-foreground/50">|</span>
                          <span>{preset.task}</span>
                        </div>
                        {isDownloading && (
                          <Progress value={progress} className="h-1 mt-2" />
                        )}
                      </div>

                      {isLoaded ? (
                        <Badge variant="secondary" className="text-xs shrink-0">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Loaded
                        </Badge>
                      ) : isDownloading ? (
                        <RefreshCw className="w-4 h-4 animate-spin text-muted-foreground shrink-0" />
                      ) : isDownloaded ? (
                        <Badge variant="outline" className="text-xs shrink-0 text-emerald-600 border-emerald-600">
                          <CheckCircle2 className="w-3 h-3 mr-1" />
                          Downloaded
                        </Badge>
                      ) : (
                        <Button
                          variant="ghost"
                          size="icon"
                          className="h-8 w-8 shrink-0"
                          onClick={() => handleDownloadPreset(preset)}
                          disabled={needsToken && !hfToken.trim()}
                          title={needsToken && !hfToken.trim() ? "Paste your HuggingFace token below first" : "Download"}
                        >
                          <Download className="w-4 h-4" />
                        </Button>
                      )}
                    </div>

                    {/* HuggingFace token row — only shown for gated models not yet downloaded */}
                    {needsToken && (
                      <div className="flex items-center gap-2 mt-1">
                        <Input
                          value={hfToken}
                          onChange={e => setHfToken(e.target.value)}
                          placeholder="hf_… (HuggingFace token — get one at hf.co/settings/tokens)"
                          className="h-7 text-xs font-mono"
                          type="password"
                        />
                      </div>
                    )}
                  </div>
                )
              })}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}