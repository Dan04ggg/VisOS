"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import { Slider } from "@/components/ui/slider"
import { Input } from "@/components/ui/input"
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts"
import {
  Play, Plus, Trash2, BarChart2, CheckCircle2, AlertCircle, RefreshCw,
} from "lucide-react"

interface DatasetInfo {
  id: string
  name: string
  path: string
}

interface ModelEntry {
  key: string
  path: string
  label: string
}

interface EvalResult {
  model_path: string
  model_name: string
  mAP50: number
  mAP50_95: number
  precision: number
  recall: number
  fitness: number
  mAP50_seg?: number
  mAP50_95_seg?: number
  avg_latency_ms?: number
  fps?: number
}

interface EvalJob {
  id: string
  status: string
  progress: number
  message: string
  results?: EvalResult[]
  error?: string
}

interface EvaluationViewProps {
  datasets?: DatasetInfo[]
  apiUrl?: string
}

const METRIC_LABELS: Record<string, string> = {
  mAP50: "mAP@50",
  mAP50_95: "mAP@50-95",
  precision: "Precision",
  recall: "Recall",
  fitness: "Fitness",
}

const CHART_COLORS = ["#3b82f6","#10b981","#f59e0b","#ef4444","#8b5cf6","#06b6d4"]

function buildRanks(results: EvalResult[], key: keyof EvalResult, lowerIsBetter = false): Map<string, number> {
  const sorted = [...results].sort((a, b) => {
    const av = (a[key] as number) ?? (lowerIsBetter ? Infinity : -Infinity)
    const bv = (b[key] as number) ?? (lowerIsBetter ? Infinity : -Infinity)
    return lowerIsBetter ? av - bv : bv - av
  })
  const map = new Map<string, number>()
  sorted.forEach((r, i) => map.set(r.model_path, i + 1))
  return map
}

function rankClass(rank: number, total: number): string {
  if (total <= 1) return "text-green-400 font-bold"
  if (rank === 1) return "text-green-400 font-bold"
  if (rank === 2) return "text-green-600"
  if (rank === 3) return "text-green-800"
  if (rank === total) return "text-red-400 font-bold"
  return ""
}

export function EvaluationView({ datasets = [], apiUrl = "http://localhost:8000" }: EvaluationViewProps) {
  const [selectedDataset, setSelectedDataset] = useState("")
  const [split, setSplit]                     = useState("val")
  const [conf, setConf]                       = useState(0.001)
  const [iou, setIou]                         = useState(0.6)
  const [imgsz, setImgsz]                     = useState(640)

  const [models, setModels]                   = useState<ModelEntry[]>([])
  const [newModelPath, setNewModelPath]        = useState("")
  const [trainedModels, setTrainedModels]     = useState<{id:string;name:string;path:string;model_type:string}[]>([])
  const [modelsLoading, setModelsLoading]     = useState(true)
  const [selectedDropModel, setSelectedDropModel] = useState("")

  const [job, setJob]                         = useState<EvalJob | null>(null)
  const [running, setRunning]                 = useState(false)
  const [error, setError]                     = useState<string | null>(null)
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // Load trained models for quick-add
  useEffect(() => {
    setModelsLoading(true)
    fetch(`${apiUrl}/api/infer/models`)
      .then(r => r.json())
      .then(d => setTrainedModels(d.models || []))
      .catch(() => {})
      .finally(() => setModelsLoading(false))
  }, [apiUrl])

  function addModel(path: string, label?: string) {
    if (!path || models.find(m => m.path === path)) return
    setModels(prev => [...prev, { key: Math.random().toString(36).slice(2), path, label: label || path.split(/[\\/]/).pop() || path }])
  }

  function removeModel(key: string) {
    setModels(prev => prev.filter(m => m.key !== key))
  }

  async function startEvaluation() {
    if (!selectedDataset || models.length === 0) return
    setError(null)
    setJob(null)
    setRunning(true)
    try {
      const res = await fetch(`${apiUrl}/api/evaluate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model_paths: models.map(m => m.path),
          dataset_id: selectedDataset,
          split, conf, iou, imgsz,
        }),
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({}))
        throw new Error(err.detail || res.statusText)
      }
      const { job_id } = await res.json()
      setJob({ id: job_id, status: "starting", progress: 0, message: "Starting…" })
      pollRef.current = setInterval(() => pollJob(job_id), 2000)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setRunning(false)
    }
  }

  async function pollJob(job_id: string) {
    try {
      const res = await fetch(`${apiUrl}/api/evaluate/${job_id}/status`)
      const data: EvalJob = await res.json()
      setJob(data)
      if (data.status === "completed" || data.status === "failed") {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
        setRunning(false)
      }
    } catch { /* ignore */ }
  }

  const results = job?.results ?? []
  const total = results.length

  // Pre-compute ranks for every metric
  const metricRankMaps = {
    ...Object.fromEntries(
      (Object.keys(METRIC_LABELS) as (keyof EvalResult)[]).map(k => [k, buildRanks(results, k)])
    ),
    fps: buildRanks(results, "fps"),
    avg_latency_ms: buildRanks(results, "avg_latency_ms", true),
  } as Record<string, Map<string, number>>

  // Build chart data
  const chartData = Object.keys(METRIC_LABELS).map(metric => {
    const row: Record<string, string|number> = { metric: METRIC_LABELS[metric] }
    results.forEach(r => { row[r.model_name] = r[metric as keyof EvalResult] as number })
    return row
  })

  // Y-axis: scale to best model value with 15% headroom
  const chartMax = (() => {
    let max = 0
    results.forEach(r => {
      Object.keys(METRIC_LABELS).forEach(k => {
        const v = r[k as keyof EvalResult] as number
        if (v > max) max = v
      })
    })
    return max > 0 ? max * 1.15 : 1
  })()

  return (
    <div className="p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-semibold">Model Evaluation</h1>
          <p className="text-muted-foreground mt-1">Head-to-head comparison of multiple models using YOLO val metrics</p>
        </div>
        <Button onClick={startEvaluation}
          disabled={running || models.length === 0 || !selectedDataset}
          className="gap-2">
          {running ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
          {running ? "Evaluating…" : "Run Evaluation"}
        </Button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* ── Setup panel ──────────────────────────────────────────────────────── */}
        <Card className="border-border/50 bg-card/50 backdrop-blur">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-primary" />
              Setup
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Dataset</label>
              {datasets.length === 0 ? (
                <p className="text-xs text-muted-foreground italic">No datasets loaded.</p>
              ) : (
                <Select value={selectedDataset} onValueChange={setSelectedDataset} disabled={running}>
                  <SelectTrigger><SelectValue placeholder="Select dataset…" /></SelectTrigger>
                  <SelectContent>
                    {datasets.map(d => <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>)}
                  </SelectContent>
                </Select>
              )}
            </div>

            <div className="space-y-2">
              <label className="text-sm font-medium">Eval Split</label>
              <Select value={split} onValueChange={setSplit} disabled={running}>
                <SelectTrigger><SelectValue /></SelectTrigger>
                <SelectContent>
                  <SelectItem value="val">Validation</SelectItem>
                  <SelectItem value="test">Test</SelectItem>
                  <SelectItem value="train">Train</SelectItem>
                </SelectContent>
              </Select>
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Confidence</label>
                <span className="text-xs text-muted-foreground">{conf.toFixed(3)}</span>
              </div>
              <Slider value={[conf]} onValueChange={([v]) => setConf(v)} min={0.001} max={0.5} step={0.001} disabled={running} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">IoU</label>
                <span className="text-xs text-muted-foreground">{iou.toFixed(2)}</span>
              </div>
              <Slider value={[iou]} onValueChange={([v]) => setIou(v)} min={0.1} max={0.95} step={0.05} disabled={running} />
            </div>

            <div className="space-y-2">
              <div className="flex justify-between">
                <label className="text-sm font-medium">Image Size</label>
                <span className="text-xs text-muted-foreground">{imgsz}px</span>
              </div>
              <Slider value={[imgsz]} onValueChange={([v]) => setImgsz(v)} min={320} max={1280} step={32} disabled={running} />
            </div>

            {error && (
              <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-400 flex gap-2">
                <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
                <span className="break-all">{error}</span>
              </div>
            )}
          </CardContent>
        </Card>

        {/* ── Models + results panel ────────────────────────────────────────────── */}
        <div className="lg:col-span-2 space-y-4">

          {/* Add models */}
          <Card className="border-border/50 bg-card/50">
            <CardHeader className="pb-2">
              <CardTitle className="text-base flex items-center gap-2">
                <Plus className="w-4 h-4 text-primary" />
                Models to Compare
                <Badge variant="secondary" className="ml-auto">{models.length}</Badge>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              {/* Dropdown quick-add from training runs */}
              <div className="space-y-1">
                <label className="text-xs text-muted-foreground">
                  {modelsLoading ? "Loading trained models…" : "Add from training runs"}
                </label>
                <div className="flex gap-2">
                  <Select
                    value={selectedDropModel}
                    onValueChange={setSelectedDropModel}
                    disabled={running || modelsLoading}
                  >
                    <SelectTrigger className="flex-1 text-xs">
                      <SelectValue placeholder={modelsLoading ? "Loading…" : trainedModels.length === 0 ? "No trained models found" : "Select a model…"} />
                    </SelectTrigger>
                    <SelectContent>
                      {trainedModels.map(m => (
                        <SelectItem key={m.path} value={m.path} disabled={!!models.find(x => x.path === m.path)}>
                          <span className="truncate max-w-[220px]">{m.name}</span>
                          <Badge variant="outline" className="ml-1 text-[10px]">{m.model_type}</Badge>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button size="sm" variant="outline" disabled={running || !selectedDropModel}
                    onClick={() => {
                      const m = trainedModels.find(x => x.path === selectedDropModel)
                      if (m) { addModel(m.path, m.name); setSelectedDropModel("") }
                    }}>
                    <Plus className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>

              {/* Manual path */}
              <div className="flex gap-2">
                <Input
                  placeholder="/path/to/model.pt"
                  value={newModelPath}
                  onChange={e => setNewModelPath(e.target.value)}
                  className="text-xs font-mono flex-1"
                  disabled={running}
                  onKeyDown={e => { if (e.key === "Enter") { addModel(newModelPath); setNewModelPath("") } }}
                />
                <Button size="sm" variant="outline" disabled={running || !newModelPath}
                  onClick={() => { addModel(newModelPath); setNewModelPath("") }}>
                  <Plus className="w-3.5 h-3.5" />
                </Button>
              </div>

              {/* Model list */}
              {models.length > 0 && (
                <div className="space-y-1">
                  {models.map((m, i) => (
                    <div key={m.key} className="flex items-center gap-2 text-xs p-2 rounded bg-muted/20 border border-border/30">
                      <span className="w-3 h-3 rounded-full shrink-0" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                      <span className="font-medium shrink-0">{m.label}</span>
                      <span className="font-mono text-muted-foreground/60 truncate flex-1">{m.path}</span>
                      <button onClick={() => removeModel(m.key)} disabled={running}
                        className="text-muted-foreground hover:text-red-400 shrink-0">
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Progress */}
          {job && (job.status === "running" || job.status === "starting") && (
            <Card className="border-border/50 bg-card/50">
              <CardContent className="pt-4 space-y-2">
                <div className="flex justify-between text-xs text-muted-foreground">
                  <span>{job.message}</span>
                  <span>{job.progress}%</span>
                </div>
                <Progress value={job.progress} className="h-1.5" />
              </CardContent>
            </Card>
          )}

          {/* Results table */}
          {results.length > 0 && (
            <Card className="border-border/50 bg-card/50">
              <CardHeader className="pb-2">
                <CardTitle className="text-base flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-green-400" />
                  Results — {job?.message}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Metric table */}
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="border-b border-border/40">
                        <th className="text-left py-1.5 pr-3 font-medium text-muted-foreground">Model</th>
                        {Object.values(METRIC_LABELS).map(l => (
                          <th key={l} className="text-right py-1.5 px-2 font-medium text-muted-foreground">{l}</th>
                        ))}
                        <th className="text-right py-1.5 px-2 font-medium text-muted-foreground">Latency</th>
                        <th className="text-right py-1.5 px-2 font-medium text-muted-foreground">FPS</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r, i) => (
                        <tr key={r.model_path} className="border-b border-border/20 hover:bg-muted/10">
                          <td className="py-2 pr-3 font-medium flex items-center gap-2">
                            <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ background: CHART_COLORS[i % CHART_COLORS.length] }} />
                            {r.model_name}
                          </td>
                          {(Object.keys(METRIC_LABELS) as (keyof EvalResult)[]).map(metric => (
                            <td key={metric} className={`text-right py-2 px-2 font-mono ${rankClass(metricRankMaps[metric]?.get(r.model_path) ?? 0, total)}`}>
                              {((r[metric] as number) * 100).toFixed(1)}%
                            </td>
                          ))}
                          <td className={`text-right py-2 px-2 font-mono ${rankClass(metricRankMaps["avg_latency_ms"]?.get(r.model_path) ?? 0, total)}`}>
                            {r.avg_latency_ms != null ? `${r.avg_latency_ms.toFixed(1)}ms` : "—"}
                          </td>
                          <td className={`text-right py-2 px-2 font-mono ${rankClass(metricRankMaps["fps"]?.get(r.model_path) ?? 0, total)}`}>
                            {r.fps != null ? `${r.fps.toFixed(0)}` : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Bar chart */}
                <div>
                  <p className="text-xs text-muted-foreground mb-2 font-medium">Visual comparison</p>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={chartData} margin={{ top: 4, right: 8, left: -10, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                      <XAxis dataKey="metric" tick={{ fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} domain={[0, chartMax]} tickFormatter={v => `${(v*100).toFixed(0)}%`} />
                      <Tooltip
                        contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 6, fontSize: 11 }}
                        formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
                      />
                      <Legend wrapperStyle={{ fontSize: 11 }} />
                      {results.map((r, i) => (
                        <Bar key={r.model_path} dataKey={r.model_name} fill={CHART_COLORS[i % CHART_COLORS.length]} />
                      ))}
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </CardContent>
            </Card>
          )}

          {job?.status === "failed" && (
            <div className="rounded-md bg-red-500/10 border border-red-500/30 p-3 text-xs text-red-400 flex gap-2">
              <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
              <span>{job.error}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
