'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import {
  Layers, Play, Pause, StopCircle, Trash2, Loader2,
  CheckCircle2, AlertCircle, Clock, ImageIcon, Tag, RefreshCw
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset, Annotation } from '@/app/page'
import { toast } from 'sonner'

interface BatchJobState {
  job_id: string
  status: 'running' | 'paused' | 'done' | 'error' | 'cancelled' | 'interrupted' | 'undone'
  paused: boolean
  progress: number
  total: number
  processed: number
  annotated: number
  failed: number
  total_annotations: number
  error?: string
  text_prompt: string
  started_at: string | number
  dataset_id: string
  recent_images?: Array<{ filename: string; path: string; annotation_count: number }>
}

interface ProcessedImage {
  filename: string
  path: string
  image_id: string
  abs_path?: string
  annotation_count: number
  annotations?: Annotation[]
}

interface BatchJobsViewProps {
  datasets: Dataset[]
  selectedDataset: Dataset | null
  apiUrl: string
}

export function BatchJobsView({ datasets, selectedDataset, apiUrl }: BatchJobsViewProps) {
  const [batchJobs, setBatchJobs] = useState<Record<string, BatchJobState>>({})
  const [selectedJob, setSelectedJob] = useState<string | null>(null)
  const [processedImages, setProcessedImages] = useState<ProcessedImage[]>([])
  const [isLoadingImages, setIsLoadingImages] = useState(false)
  const [previewImage, setPreviewImage] = useState<ProcessedImage | null>(null)
  const [previewImageUrl, setPreviewImageUrl] = useState<string | null>(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const pollIntervalsRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})

  // Restore batch jobs from backend on mount
  useEffect(() => {
    const restore = async () => {
      try {
        const r = await fetch(`${apiUrl}/api/auto-annotate/jobs`)
        if (r.ok) {
          const data = await r.json()
          const backendJobs: Record<string, BatchJobState> = {}
          for (const job of (data.jobs || [])) {
            backendJobs[job.job_id] = job
          }
          if (Object.keys(backendJobs).length > 0) {
            setBatchJobs(backendJobs)
            // Poll any still-running jobs
            Object.values(backendJobs).forEach(job => {
              if (job.status === 'running' || job.status === 'paused') {
                _pollJob(job.job_id)
              }
            })
            return
          }
        }
      } catch {}
      // Fallback: localStorage
      try {
        const saved = localStorage.getItem('cvdm_batchJobs')
        if (!saved) return
        const jobs: Record<string, BatchJobState> = JSON.parse(saved)
        if (Object.keys(jobs).length === 0) return
        setBatchJobs(jobs)
        Object.values(jobs).forEach(job => {
          if (job.status === 'running' || job.status === 'paused') {
            _pollJob(job.job_id)
          }
        })
      } catch {}
    }
    restore()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Persist batch jobs to localStorage
  useEffect(() => {
    try { localStorage.setItem('cvdm_batchJobs', JSON.stringify(batchJobs)) } catch {}
  }, [batchJobs])

  // Clear all poll intervals on unmount
  useEffect(() => {
    return () => { Object.values(pollIntervalsRef.current).forEach(clearInterval) }
  }, [])

  const _pollJob = (job_id: string) => {
    if (pollIntervalsRef.current[job_id]) {
      clearInterval(pollIntervalsRef.current[job_id])
    }
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/status`)
        if (!r.ok) { clearInterval(iv); delete pollIntervalsRef.current[job_id]; return }
        const data = await r.json()
        setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], ...data } }))

        const done = data.status === 'done' || data.status === 'error' || data.status === 'cancelled'
        if (done) {
          clearInterval(iv)
          delete pollIntervalsRef.current[job_id]
          if (data.status === 'done') {
            toast.success(
              `Batch done - ${data.annotated} images annotated, ` +
              `${data.total_annotations ?? 0} annotations created`
            )
          } else if (data.status === 'error') {
            toast.error(`Batch failed: ${data.error}`)
          }
        }
      } catch { clearInterval(iv); delete pollIntervalsRef.current[job_id] }
    }, 1500)
    pollIntervalsRef.current[job_id] = iv
  }

  const pauseJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/pause`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'paused', paused: true } }))
    } catch { toast.error('Failed to pause job') }
  }

  const resumeJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/resume`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'running', paused: false } }))
      _pollJob(job_id)
    } catch { toast.error('Failed to resume job') }
  }

  const cancelJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/cancel`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'cancelled' } }))
    } catch { toast.error('Failed to cancel job') }
  }

  const deleteJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}`, { method: 'DELETE' })
    } catch {}
    setBatchJobs(prev => {
      const next = { ...prev }
      delete next[job_id]
      return next
    })
    if (selectedJob === job_id) {
      setSelectedJob(null)
      setProcessedImages([])
    }
  }

  const undoJob = async (job_id: string) => {
    try {
      const resp = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/undo`, { method: 'POST' })
      if (!resp.ok) throw new Error((await resp.json()).detail ?? 'Undo failed')
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'undone' } }))
      toast.success('Batch annotations undone — dataset restored to pre-job state')
    } catch (err) { toast.error(err instanceof Error ? err.message : 'Undo failed') }
  }

  const continueJob = async (job_id: string) => {
    try {
      const r = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/restart`, { method: 'POST' })
      if (!r.ok) {
        const err = await r.json()
        toast.error(err.detail || 'Failed to continue job')
        return
      }
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'running', paused: false } }))
      _pollJob(job_id)
    } catch { toast.error('Failed to continue job') }
  }

  // Load processed images for a job
  const loadProcessedImages = useCallback(async (job_id: string) => {
    setIsLoadingImages(true)
    setProcessedImages([])
    try {
      const r = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/processed-images`)
      if (r.ok) {
        const data = await r.json()
        setProcessedImages(data.images || [])
      }
    } catch {
      toast.error('Failed to load processed images')
    } finally {
      setIsLoadingImages(false)
    }
  }, [apiUrl])

  // When selecting a job, load its processed images
  useEffect(() => {
    if (selectedJob) {
      loadProcessedImages(selectedJob)
    }
  }, [selectedJob, loadProcessedImages])

  // Auto-refresh images for running jobs
  useEffect(() => {
    if (!selectedJob) return
    const job = batchJobs[selectedJob]
    if (!job || (job.status !== 'running' && job.status !== 'paused')) return
    
    const refreshInterval = setInterval(() => {
      loadProcessedImages(selectedJob)
    }, 3000)
    
    return () => clearInterval(refreshInterval)
  }, [selectedJob, batchJobs, loadProcessedImages])

  // Load preview image with annotations
  const loadImagePreview = useCallback((img: ProcessedImage, job_id: string) => {
    const job = batchJobs[job_id]
    if (!job) return

    setPreviewImage(img)
    setIsLoadingPreview(true)
    
    // Use the annotated endpoint which draws bounding boxes on the image
    const url = `${apiUrl}/api/auto-annotate/text-batch/${job_id}/annotated/${encodeURIComponent(img.image_id)}`
    setPreviewImageUrl(url)
    setIsLoadingPreview(false)
  }, [apiUrl, batchJobs])

  const fmtElapsed = (started: string | number) => {
    const start = typeof started === 'number' ? started : new Date(started).getTime()
    const elapsed = Date.now() - start
    const sec = Math.floor(elapsed / 1000)
    if (sec < 60) return `${sec}s ago`
    const min = Math.floor(sec / 60)
    if (min < 60) return `${min}m ago`
    const hr = Math.floor(min / 60)
    return `${hr}h ${min % 60}m ago`
  }

  const getStatusIcon = (status: BatchJobState['status']) => {
    switch (status) {
      case 'running': return <Loader2 className="w-4 h-4 animate-spin text-primary" />
      case 'paused': return <Pause className="w-4 h-4 text-warning" />
      case 'done': return <CheckCircle2 className="w-4 h-4 text-success" />
      case 'error': return <AlertCircle className="w-4 h-4 text-destructive" />
      case 'cancelled':
      case 'interrupted':
      case 'undone': return <StopCircle className="w-4 h-4 text-muted-foreground" />
      default: return <Clock className="w-4 h-4 text-muted-foreground" />
    }
  }

  const getStatusColor = (status: BatchJobState['status']) => {
    switch (status) {
      case 'running': return 'bg-primary/10 text-primary border-primary/20'
      case 'paused': return 'bg-warning/10 text-warning border-warning/20'
      case 'done': return 'bg-success/10 text-success border-success/20'
      case 'error': return 'bg-destructive/10 text-destructive border-destructive/20'
      default: return 'bg-muted text-muted-foreground border-border'
    }
  }

  const jobsList = Object.values(batchJobs).slice().reverse()
  const totalJobs = jobsList.length
  const runningJobs = jobsList.filter(j => j.status === 'running').length
  const selectedJobData = selectedJob ? batchJobs[selectedJob] : null

  return (
    <div className="flex h-full overflow-hidden">
      {/* Jobs List Panel */}
      <div className="w-80 border-r border-border flex flex-col bg-background shrink-0">
        {/* Header */}
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-1">
            <h1 className="text-lg font-semibold">Batch Jobs</h1>
            {runningJobs > 0 && (
              <Badge variant="default" className="bg-primary gap-1">
                <span className="w-1.5 h-1.5 bg-white rounded-full animate-pulse" />
                {runningJobs} running
              </Badge>
            )}
          </div>
          <p className="text-sm text-muted-foreground">
            {totalJobs} job{totalJobs !== 1 ? 's' : ''} total
          </p>
        </div>

        {/* Jobs list */}
        <div className="flex-1 overflow-y-auto">
          {totalJobs === 0 ? (
            <div className="flex flex-col items-center justify-center h-64 p-6 text-center">
              <Layers className="w-12 h-12 text-muted-foreground/30 mb-4" />
              <p className="text-sm font-medium text-muted-foreground">No batch jobs yet</p>
              <p className="text-xs text-muted-foreground/60 mt-1">
                Start a batch annotation job from the Annotate view
              </p>
            </div>
          ) : (
            <div className="p-2 space-y-1">
              {jobsList.map(job => {
                const isSelected = selectedJob === job.job_id
                const isActive = job.status === 'running' || job.status === 'paused'
                const processed = job.processed ?? (job.annotated + job.failed)
                
                return (
                  <button
                    key={job.job_id}
                    onClick={() => setSelectedJob(job.job_id)}
                    className={cn(
                      'w-full text-left p-3 rounded-lg transition-all',
                      isSelected
                        ? 'bg-primary/10 border border-primary/20'
                        : 'hover:bg-muted/50 border border-transparent'
                    )}
                  >
                    {/* Prompt tags */}
                    <div className="flex flex-wrap gap-1 mb-2">
                      {job.text_prompt.split(',').map(t => t.trim()).filter(Boolean).slice(0, 3).map(tag => (
                        <span
                          key={tag}
                          className="px-1.5 py-0.5 bg-primary/8 text-primary border border-primary/15 rounded-full text-[10px] font-medium"
                        >
                          {tag}
                        </span>
                      ))}
                      {job.text_prompt.split(',').length > 3 && (
                        <span className="text-[10px] text-muted-foreground">
                          +{job.text_prompt.split(',').length - 3} more
                        </span>
                      )}
                    </div>

                    {/* Status and time */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        {getStatusIcon(job.status)}
                        <span className={cn(
                          'text-xs px-1.5 py-0.5 rounded border capitalize',
                          getStatusColor(job.status)
                        )}>
                          {job.status}
                        </span>
                      </div>
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {fmtElapsed(job.started_at)}
                      </span>
                    </div>

                    {/* Progress bar */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-[10px] text-muted-foreground">
                        <span>{processed} / {job.total || '?'}</span>
                        <span>{job.progress}%</span>
                      </div>
                      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                        <div
                          className={cn(
                            'h-full rounded-full transition-all duration-500',
                            job.status === 'done' ? 'bg-success' :
                            job.status === 'error' ? 'bg-destructive' :
                            job.status === 'paused' ? 'bg-warning' : 'bg-primary'
                          )}
                          style={{ width: `${job.progress}%` }}
                        />
                      </div>
                    </div>

                    {/* Stats row */}
                    <div className="flex gap-3 mt-2 text-[10px]">
                      <span className="text-success">{job.annotated} annotated</span>
                      <span className="text-primary">{job.total_annotations} labels</span>
                      {job.failed > 0 && (
                        <span className="text-destructive">{job.failed} failed</span>
                      )}
                    </div>
                  </button>
                )
              })}
            </div>
          )}
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {selectedJobData ? (
          <>
            {/* Job details header */}
            <div className="p-4 border-b border-border bg-background">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-2">
                    {getStatusIcon(selectedJobData.status)}
                    <h2 className="text-lg font-semibold truncate">
                      {selectedJobData.text_prompt}
                    </h2>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
                    <span className="font-mono text-xs">{selectedJobData.job_id}</span>
                    <span>-</span>
                    <span>{fmtElapsed(selectedJobData.started_at)}</span>
                    {selectedJobData.dataset_id && (
                      <>
                        <span>-</span>
                        <span>Dataset: {datasets.find(d => d.id === selectedJobData.dataset_id)?.name || selectedJobData.dataset_id}</span>
                      </>
                    )}
                  </div>
                </div>

                {/* Action buttons */}
                <div className="flex items-center gap-2 shrink-0">
                  {selectedJobData.status === 'running' && (
                    <Button size="sm" variant="outline" onClick={() => pauseJob(selectedJobData.job_id)} className="gap-1.5">
                      <Pause className="w-3.5 h-3.5" />
                      Pause
                    </Button>
                  )}
                  {selectedJobData.status === 'paused' && (
                    <Button size="sm" variant="outline" onClick={() => resumeJob(selectedJobData.job_id)} className="gap-1.5">
                      <Play className="w-3.5 h-3.5" />
                      Resume
                    </Button>
                  )}
                  {(selectedJobData.status === 'interrupted' || selectedJobData.status === 'error') && (
                    <Button size="sm" variant="outline" onClick={() => continueJob(selectedJobData.job_id)} className="gap-1.5">
                      <Play className="w-3.5 h-3.5" />
                      Continue
                    </Button>
                  )}
                  {(selectedJobData.status === 'running' || selectedJobData.status === 'paused') && (
                    <Button size="sm" variant="destructive" onClick={() => cancelJob(selectedJobData.job_id)} className="gap-1.5">
                      <StopCircle className="w-3.5 h-3.5" />
                      Cancel
                    </Button>
                  )}
                  {selectedJobData.status === 'done' && (
                    <Button size="sm" variant="outline" onClick={() => undoJob(selectedJobData.job_id)} className="gap-1.5 border-destructive/30 text-destructive hover:bg-destructive/10">
                      <RefreshCw className="w-3.5 h-3.5" />
                      Undo Batch
                    </Button>
                  )}
                  {(selectedJobData.status === 'done' || selectedJobData.status === 'error' || selectedJobData.status === 'cancelled' || selectedJobData.status === 'interrupted' || selectedJobData.status === 'undone') && (
                    <Button size="sm" variant="ghost" onClick={() => deleteJob(selectedJobData.job_id)} className="gap-1.5 text-muted-foreground hover:text-destructive">
                      <Trash2 className="w-3.5 h-3.5" />
                      Remove
                    </Button>
                  )}
                </div>
              </div>

              {/* Stats cards */}
              <div className="grid grid-cols-4 gap-3 mt-4">
                <div className="p-3 rounded-lg bg-muted/50 border border-border">
                  <p className="text-2xl font-bold tabular-nums">{selectedJobData.processed ?? (selectedJobData.annotated + selectedJobData.failed)}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">Processed</p>
                </div>
                <div className="p-3 rounded-lg bg-success/5 border border-success/20">
                  <p className="text-2xl font-bold text-success tabular-nums">{selectedJobData.annotated}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">Annotated</p>
                </div>
                <div className="p-3 rounded-lg bg-primary/5 border border-primary/20">
                  <p className="text-2xl font-bold text-primary tabular-nums">{selectedJobData.total_annotations}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">Total Labels</p>
                </div>
                <div className={cn(
                  'p-3 rounded-lg border',
                  selectedJobData.failed > 0 ? 'bg-destructive/5 border-destructive/20' : 'bg-muted/50 border-border'
                )}>
                  <p className={cn('text-2xl font-bold tabular-nums', selectedJobData.failed > 0 ? 'text-destructive' : 'text-muted-foreground')}>
                    {selectedJobData.failed}
                  </p>
                  <p className="text-xs text-muted-foreground mt-0.5">Failed</p>
                </div>
              </div>

              {/* Progress bar */}
              <div className="mt-4">
                <div className="flex justify-between text-xs text-muted-foreground mb-1">
                  <span>Progress</span>
                  <span>{selectedJobData.progress}%</span>
                </div>
                <div className="h-2 bg-muted rounded-full overflow-hidden">
                  <div
                    className={cn(
                      'h-full rounded-full transition-all duration-500',
                      selectedJobData.status === 'done' ? 'bg-success' :
                      selectedJobData.status === 'error' ? 'bg-destructive' :
                      selectedJobData.status === 'paused' ? 'bg-warning' : 'bg-primary'
                    )}
                    style={{ width: `${selectedJobData.progress}%` }}
                  />
                </div>
              </div>
            </div>

            {/* Processed images grid */}
            <div className="flex-1 overflow-hidden flex flex-col min-h-0">
              <div className="px-4 py-3 border-b border-border flex items-center justify-between bg-background/50">
                <h3 className="text-sm font-medium">
                  Processed Images
                  <span className="ml-2 text-muted-foreground font-normal">
                    ({processedImages.length})
                  </span>
                </h3>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => loadProcessedImages(selectedJob!)}
                  disabled={isLoadingImages}
                  className="gap-1.5 text-xs"
                >
                  <RefreshCw className={cn("w-3.5 h-3.5", isLoadingImages && "animate-spin")} />
                  Refresh
                </Button>
              </div>

              <div className="flex-1 overflow-y-auto p-4">
                {isLoadingImages ? (
                  <div className="flex items-center justify-center h-48">
                    <Loader2 className="w-8 h-8 animate-spin text-primary" />
                  </div>
                ) : processedImages.length === 0 ? (
                  <div className="flex flex-col items-center justify-center h-48 text-center">
                    <ImageIcon className="w-12 h-12 text-muted-foreground/30 mb-4" />
                    <p className="text-sm text-muted-foreground">No processed images yet</p>
                    <p className="text-xs text-muted-foreground/60 mt-1">
                      Images will appear here as they are annotated
                    </p>
                  </div>
                ) : (
                  <div className="grid grid-cols-[repeat(auto-fill,minmax(140px,1fr))] gap-3">
                    {processedImages.map((img, idx) => (
                      <button
                        key={idx}
                        onClick={() => loadImagePreview(img, selectedJob!)}
                        className="group relative aspect-square rounded-lg overflow-hidden border border-border hover:border-primary/50 transition-all hover:shadow-lg"
                      >
                        <img
                          src={`${apiUrl}/api/auto-annotate/text-batch/${selectedJob}/image/${encodeURIComponent(img.image_id)}`}
                          alt={img.filename}
                          className="w-full h-full object-cover"
                          loading="lazy"
                          onError={(e) => {
                            // Hide broken images
                            (e.target as HTMLImageElement).style.display = 'none'
                          }}
                        />
                        
                        {/* Hover overlay */}
                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/30 transition-colors flex items-center justify-center">
                          <span className="opacity-0 group-hover:opacity-100 transition-opacity text-white text-xs font-medium px-2 py-1 bg-black/50 rounded">
                            View Annotations
                          </span>
                        </div>

                        {/* Annotation count badge */}
                        <div className="absolute top-1.5 right-1.5">
                          <span className={cn(
                            'inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium',
                            img.annotation_count > 0
                              ? 'bg-primary/90 text-white'
                              : 'bg-black/50 text-white/70'
                          )}>
                            <Tag className="w-2.5 h-2.5" />
                            {img.annotation_count}
                          </span>
                        </div>

                        {/* Filename */}
                        <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-2 pb-1.5 pt-4">
                          <p className="text-white text-[10px] truncate">{img.filename}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <Layers className="w-16 h-16 text-muted-foreground/20 mx-auto mb-4" />
              <p className="text-lg font-medium text-muted-foreground">Select a job to view details</p>
              <p className="text-sm text-muted-foreground/60 mt-1">
                Click on a job from the list to see processed images
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Image Preview Dialog */}
      <Dialog open={!!previewImage} onOpenChange={(open) => !open && setPreviewImage(null)}>
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden flex flex-col">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <span className="truncate">{previewImage?.filename}</span>
              {previewImage && (
                <Badge variant="secondary" className="shrink-0">
                  {previewImage.annotation_count} annotations
                </Badge>
              )}
            </DialogTitle>
          </DialogHeader>
          <div className="flex-1 overflow-auto flex items-center justify-center bg-muted/30 rounded-lg min-h-[400px]">
            {isLoadingPreview ? (
              <Loader2 className="w-8 h-8 animate-spin text-primary" />
            ) : previewImageUrl ? (
              <img
                src={previewImageUrl}
                alt={previewImage?.filename}
                className="max-w-full max-h-[70vh] object-contain"
                onError={() => {
                  // Fallback to raw image if annotated preview fails
                  if (selectedJob && previewImage) {
                    setPreviewImageUrl(`${apiUrl}/api/auto-annotate/text-batch/${selectedJob}/image/${encodeURIComponent(previewImage.image_id)}`)
                  }
                }}
              />
            ) : (
              <p className="text-muted-foreground">Failed to load preview</p>
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
