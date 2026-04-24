'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import {
  MousePointer2, Square, Pentagon, Wand2, Save, Trash2,
  ChevronLeft, ChevronRight, ZoomIn, ZoomOut, RotateCcw,
  AlertCircle, Loader2, Undo2, FolderOpen, Tag,
  Brush, Crosshair, ChevronsRight, Keyboard,
  Download, CheckCircle2, X, Plus, Pause, Play, StopCircle
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset, ImageData, Annotation, ImageCache } from '@/app/page'
import { useSettings } from '@/lib/settings-context'
import { toast } from 'sonner'

const PAGE_SIZE = 50
const PRELOAD_AHEAD = 3

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
  model_id?: string
  confidence_threshold?: number
  started_at: string | number
  dataset_id: string
  recent_images?: Array<{ filename: string; path: string; image_id: string; annotation_count: number }>
}

interface AnnotationViewProps {
  selectedDataset: Dataset | null
  apiUrl: string
  imageCache: ImageCache
  updateImageCache: (datasetId: string, images: ImageData[]) => void
  initialImageId?: string | null
  onInitialImageConsumed?: () => void
}

type Tool = 'select' | 'bbox' | 'polygon' | 'keypoint' | 'brush' | 'sam'

export function AnnotationView({ selectedDataset, apiUrl, imageCache, updateImageCache, initialImageId, onInitialImageConsumed }: AnnotationViewProps) {
  const { settings } = useSettings()
  // Image list state
  const [allImages, setAllImages] = useState<ImageData[]>([])
  const [filteredImages, setFilteredImages] = useState<ImageData[]>([])
  const [currentIndex, setCurrentIndex] = useState(0)
  const [isLoadingList, setIsLoadingList] = useState(false)
  const [availableSplits, setAvailableSplits] = useState<string[]>([])
  const [availableClasses, setAvailableClasses] = useState<string[]>([])
  const [selectedSplit, setSelectedSplit] = useState('all')
  const [selectedClass, setSelectedClass] = useState('all')

  // Current image state
  const [currentImage, setCurrentImage] = useState<ImageData | null>(null)
  const [annotations, setAnnotations] = useState<Annotation[]>([])
  const [isImageLoading, setIsImageLoading] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Tool state
  const [activeTool, setActiveTool] = useState<Tool>('select')
  const [activeClass, setActiveClass] = useState<string>('')
  const [selectedAnnotation, setSelectedAnnotation] = useState<number | null>(null)
  const [zoom, setZoom] = useState(1)
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawStart, setDrawStart] = useState<{ x: number; y: number } | null>(null)
  const [tempBox, setTempBox] = useState<{ x: number; y: number; w: number; h: number } | null>(null)
  const [polygonPoints, setPolygonPoints] = useState<{ x: number; y: number }[]>([])
  const [history, setHistory] = useState<Annotation[][]>([])
  const [historyIndex, setHistoryIndex] = useState(-1)

  // Brush state
  const [brushSize, setBrushSize] = useState(20)
  const [brushPoints, setBrushPoints] = useState<{ x: number; y: number }[]>([])
  const [isBrushing, setIsBrushing] = useState(false)

  // Drag state for moving selected annotation in select mode
  const [isDragging, setIsDragging] = useState(false)
  const [dragStart, setDragStart] = useState<{ x: number; y: number; origBbox: number[] } | null>(null)

  // Add new class state
  const [addingClass, setAddingClass] = useState(false)
  const [newClassName, setNewClassName] = useState('')
  // Local copy of dataset classes so we can add to it without mutating the prop
  const [localClasses, setLocalClasses] = useState<string[]>(selectedDataset?.classes || [])

  // SAM click queue state
  type SamClick = { id: string; nx: number; ny: number; status: 'pending' | 'processing' }
  const [samPendingClicks, setSamPendingClicks] = useState<SamClick[]>([])
  const samPendingClicksRef = useRef<SamClick[]>([])
  const samProcessingRef = useRef(false)
  const selectedModelRef = useRef('')
  const confidenceRef = useRef(0.5)

  // Keypoint state
  const [keypointList, setKeypointList] = useState<{ x: number; y: number; label: string }[]>([])

  // Batch ops state
  const [copyToNext, setCopyToNext] = useState(1)
  const [showBatchPanel, setShowBatchPanel] = useState(false)
  const [showShortcutHint, setShowShortcutHint] = useState(false)

  // Auto-annotate state
  const [models, setModels] = useState<{ id: string; name: string; type: string; downloaded: boolean; loaded: boolean }[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [confidence, setConfidence] = useState(0.5)
  const [isAutoAnnotating, setIsAutoAnnotating] = useState(false)
  const [downloadingModelId, setDownloadingModelId] = useState<string | null>(null)
  // SAM3 text prompt — tag-chip model
  const [promptTags, setPromptTags] = useState<string[]>([])
  const [promptInput, setPromptInput] = useState('')
  // Batch job tracking: job_id → BatchJobState
  const [batchJobs, setBatchJobs] = useState<Record<string, BatchJobState>>({})
  const [isStartingBatch, setIsStartingBatch] = useState(false)
  // Right sidebar tab: 'annotate' = auto-annotate settings, 'jobs' = batch job monitor
  const [sidebarTab, setSidebarTab] = useState<'annotate' | 'jobs'>('annotate')

  // Resume-progress banner state
  const [resumeBanner, setResumeBanner] = useState<{
    imageId: string
    imageName: string
    imageIndex: number
    split: string
    classFilter: string
  } | null>(null)

  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const imageRef = useRef<HTMLImageElement | null>(null)
  const scaleRef = useRef(1)
  // Always-current refs so polling intervals can read latest values without closure stale issues
  const currentImageRef = useRef<ImageData | null>(null)
  const selectedDatasetRef = useRef<Dataset | null>(null)
  const annotationsRef = useRef<Annotation[]>([])
  // Track active poll intervals so we never start two for the same job
  const pollIntervalsRef = useRef<Record<string, ReturnType<typeof setInterval>>>({})
  // When true, the allImages useEffect skips the reset-to-index-0 behaviour
  const silentRefreshRef = useRef(false)
  // When set, the allImages useEffect jumps to this image id instead of index 0
  const initialImageIdRef = useRef<string | null>(null)

  // ── Keep refs in sync with latest state ─────────────────────────────────────
  useEffect(() => { currentImageRef.current = currentImage }, [currentImage])
  useEffect(() => { selectedDatasetRef.current = selectedDataset }, [selectedDataset])
  useEffect(() => { annotationsRef.current = annotations }, [annotations])
  useEffect(() => { samPendingClicksRef.current = samPendingClicks }, [samPendingClicks])
  useEffect(() => { selectedModelRef.current = selectedModel }, [selectedModel])
  useEffect(() => { confidenceRef.current = confidence }, [confidence])
  // Sync localClasses when dataset changes
  useEffect(() => { setLocalClasses(selectedDataset?.classes || []) }, [selectedDataset?.id])
  // If the dataset YAML has no class names, populate localClasses from the
  // annotation data once images have loaded (availableClasses is derived from annotations)
  useEffect(() => {
    if (availableClasses.length > 0 && localClasses.length === 0) {
      setLocalClasses(availableClasses)
      setActiveClass(prev => prev || availableClasses[0] || '')
    }
  }, [availableClasses]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Clear all poll intervals on unmount ───────────────────────────────────────
  useEffect(() => {
    return () => { Object.values(pollIntervalsRef.current).forEach(clearInterval) }
  }, [])

  // ── SAM queue: redraw canvas when pending clicks change ───────────────────────
  useEffect(() => {
    if (imageRef.current) {
      // drawCanvasWithData reads samPendingClicksRef directly, so just trigger a redraw
      const img = imageRef.current
      drawCanvasWithData(img, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList)
    }
  }, [samPendingClicks]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── SAM queue: process clicks one at a time ───────────────────────────────────
  useEffect(() => {
    if (samPendingClicks.length === 0 || samProcessingRef.current) return
    const processNext = async () => {
      const queue = samPendingClicksRef.current
      if (queue.length === 0 || samProcessingRef.current) return
      samProcessingRef.current = true
      const click = queue[0]
      // Mark as processing (yellow dot)
      const updated = samPendingClicksRef.current.map(c =>
        c.id === click.id ? { ...c, status: 'processing' as const } : c
      )
      samPendingClicksRef.current = updated
      setSamPendingClicks(updated)

      const modelId = selectedModelRef.current
      const conf = confidenceRef.current
      const ds = selectedDatasetRef.current
      const img = currentImageRef.current
      if (modelId && ds && img) {
        try {
          const params = new URLSearchParams({
            model_id: modelId,
            confidence_threshold: String(conf),
            point_x: String(click.nx),
            point_y: String(click.ny),
          })
          if (img.path) params.set('image_path_hint', img.path)
          const resp = await fetch(
            `${apiUrl}/api/auto-annotate/${ds.id}/single/${img.id}?${params}`,
            { method: 'POST' }
          )
          if (resp.ok) {
            const data = await resp.json()
            if (data.annotations?.length) {
              const next = [...annotationsRef.current, ...data.annotations]
              setAnnotations(next)
              saveToHistory(next)
              await saveAnnotations(false, next)
            } else {
              // Fallback: standard auto-annotate (no point)
              const fbParams = new URLSearchParams({ model_id: modelId, confidence_threshold: String(conf) })
              if (img.path) fbParams.set('image_path_hint', img.path)
              const fbResp = await fetch(
                `${apiUrl}/api/auto-annotate/${ds.id}/single/${img.id}?${fbParams}`,
                { method: 'POST' }
              )
              if (fbResp.ok) {
                const fbData = await fbResp.json()
                if (fbData.annotations?.length) {
                  const next = [...annotationsRef.current, ...fbData.annotations]
                  setAnnotations(next)
                  saveToHistory(next)
                  await saveAnnotations(false, next)
                }
              }
            }
          }
        } catch { /* silently skip failed individual clicks */ }
      }

      // Remove processed click
      const remaining = samPendingClicksRef.current.filter(c => c.id !== click.id)
      samPendingClicksRef.current = remaining
      setSamPendingClicks(remaining)
      samProcessingRef.current = false
    }
    processNext()
  }, [samPendingClicks]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Persist batch jobs to localStorage ───────────────────────────────────────
  useEffect(() => {
    try { localStorage.setItem('cvdm_batchJobs', JSON.stringify(batchJobs)) } catch {}
  }, [batchJobs])

  // ── Restore batch jobs from backend on first mount ───────────────────────────
  // Backend persists jobs to disk; on restart they survive with updated statuses.
  // We also fall back to localStorage for jobs the backend doesn't know about.
  useEffect(() => {
    const restore = async () => {
      try {
        // Try backend first (authoritative after restart)
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

  // ── Save annotation progress to localStorage ─────────────────────────────────
  useEffect(() => {
    if (!selectedDataset || !currentImage) return
    try {
      localStorage.setItem(`cvdm_annotate_${selectedDataset.id}`, JSON.stringify({
        imageId: currentImage.id,
        imageName: currentImage.filename,
        imageIndex: currentIndex,
        split: selectedSplit,
        classFilter: selectedClass,
      }))
    } catch {}
  }, [currentIndex, currentImage?.id, selectedSplit, selectedClass]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Load image list ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (!selectedDataset) return
    // Check for saved progress (only when not externally directed to a specific image)
    setResumeBanner(null)
    if (!initialImageId) {
      try {
        const raw = localStorage.getItem(`cvdm_annotate_${selectedDataset.id}`)
        if (raw) {
          const saved = JSON.parse(raw)
          if (saved.imageId && saved.imageIndex > 0) {
            setResumeBanner(saved)
          }
        }
      } catch {}
    }
    const cached = imageCache[selectedDataset.id]
    if (cached && cached.length > 0) {
      initFromImages(cached)
    } else {
      loadImageList()
    }
    loadModels()
    if (selectedDataset.classes?.length) setActiveClass(selectedDataset.classes[0])
    setLocalClasses(selectedDataset.classes || [])
  }, [selectedDataset])

  useEffect(() => {
    let imgs = allImages
    if (selectedSplit !== 'all') imgs = imgs.filter(i => i.split === selectedSplit)
    if (selectedClass !== 'all') {
      imgs = imgs.filter(i =>
        i.class_name === selectedClass ||
        (i.annotations || []).some(a => a.class_name === selectedClass)
      )
    }
    setFilteredImages(imgs)
    // Silent refresh (from batch polling): don't reset position or reload the canvas
    if (silentRefreshRef.current) {
      silentRefreshRef.current = false
      return
    }
    if (imgs.length > 0) {
      const targetId = initialImageIdRef.current
      if (targetId) {
        initialImageIdRef.current = null
        onInitialImageConsumed?.()
        const idx = Math.max(0, imgs.findIndex(img => img.id === targetId))
        setCurrentIndex(idx)
        loadImageFile(imgs[idx])
      } else {
        setCurrentIndex(0)
        loadImageFile(imgs[0])
      }
    }
  }, [selectedSplit, selectedClass, allImages])

  const initFromImages = (images: ImageData[]) => {
    // Set ref BEFORE setAllImages so the allImages useEffect can consume it
    if (initialImageId) initialImageIdRef.current = initialImageId
    setAllImages(images)
    const splits = [...new Set(images.map(i => i.split).filter(Boolean) as string[])]
    const classes = [...new Set(
      images.flatMap(i => [
        i.class_name,
        ...(i.annotations || []).map(a => a.class_name)
      ]).filter(Boolean) as string[]
    )]
    setAvailableSplits(splits)
    setAvailableClasses(classes)
    setFilteredImages(images)
    // currentIndex + image loading are handled by the allImages useEffect below
  }

  const loadImageList = async () => {
    if (!selectedDataset) return
    setIsLoadingList(true)
    setError(null)
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images?limit=999999`)
      if (!resp.ok) throw new Error('Failed to load images')
      const data = await resp.json()
      const images: ImageData[] = data.images || []
      updateImageCache(selectedDataset.id, images)
      initFromImages(images)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load images')
    } finally {
      setIsLoadingList(false)
    }
  }

  /**
   * Fetch a fresh image list then navigate to a specific image by stem id.
   * Used by batch-job thumbnail clicks and "View Annotations" buttons.
   */
  const navigateToAnnotatedImage = async (imageId: string) => {
    if (!selectedDataset) return
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/images?limit=999999&bust_cache=true`)
      if (!resp.ok) return
      const data = await resp.json()
      const images: ImageData[] = data.images || []
      updateImageCache(selectedDataset.id, images)
      // Set target before initFromImages so the allImages useEffect navigates there
      initialImageIdRef.current = imageId
      setAllImages(images)
    } catch {}
  }

  /**
   * Refresh the image list and cache WITHOUT resetting the current canvas view.
   * Used by batch-job polling so annotations from the job become visible
   * in the list/gallery without kicking the user back to image #0.
   * Also updates annotations for the currently displayed image if new ones exist.
   */
  const refreshImageListSilent = async () => {
    const ds = selectedDatasetRef.current
    if (!ds) return
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${ds.id}/images?limit=999999&bust_cache=true`)
      if (!resp.ok) return
      const data = await resp.json()
      const images: ImageData[] = data.images || []
      updateImageCache(ds.id, images)
      // Set flag BEFORE setAllImages so the useEffect sees it synchronously
      silentRefreshRef.current = true
      setAllImages(images)
      // Sync current image annotations if they differ from what's on canvas
      const img = currentImageRef.current
      if (img) {
        const updated = images.find(i => i.id === img.id)
        if (updated && (updated.annotations?.length ?? 0) !== annotationsRef.current.length) {
          setAnnotations(updated.annotations || [])
        }
      }
    } catch {}
  }

  const loadModels = async () => {
    try {
      const r = await fetch(`${apiUrl}/api/models`)
      if (r.ok) {
        const d = await r.json()
        setModels((d.models || []).map((m: any) => ({
          id: m.id,
          name: m.name,
          type: m.type || 'unknown',
          downloaded: !!(m.downloaded || m.loaded),
          loaded: !!m.loaded,
        })))
      }
    } catch {}
  }

  const downloadModel = async (model: { id: string; name: string; type: string }) => {
    setDownloadingModelId(model.id)
    toast.info(`Downloading ${model.name}… this may take a minute`, { duration: 120000, id: `dl-${model.id}` })
    try {
      const fd = new FormData()
      fd.append('model_type', model.type)
      fd.append('pretrained', model.id)
      const r = await fetch(`${apiUrl}/api/models/download`, { method: 'POST', body: fd })
      if (!r.ok) throw new Error('Failed to start download')

      await new Promise<void>((resolve, reject) => {
        const poll = setInterval(async () => {
          try {
            const s = await fetch(`${apiUrl}/api/models/download-status/${model.id}`)
            if (!s.ok) return
            const status = await s.json()
            if (status.phase === 'loading_memory') {
              toast.info(`Loading ${model.name} into memory… please wait`, { duration: 120000, id: `dl-${model.id}` })
            }
            if (status.status === 'done') { clearInterval(poll); resolve() }
            else if (status.status === 'error') { clearInterval(poll); reject(new Error(status.error ?? 'Download failed')) }
          } catch {}
        }, 1000)
      })

      toast.dismiss(`dl-${model.id}`)
      await loadModels()
      setSelectedModel(model.id)
      toast.success(`${model.name} ready`)
    } catch (err) {
      toast.dismiss(`dl-${model.id}`)
      toast.error(`Download failed: ${err instanceof Error ? err.message : err}`)
    } finally {
      setDownloadingModelId(null)
    }
  }

  // Auto-switch to SAM wand when a SAM model is selected
  useEffect(() => {
    if (!selectedModel) return
    const m = models.find(m => m.id === selectedModel)
    if (m && (m.type === 'sam' || m.type === 'sam2' || m.type === 'sam3')) {
      setActiveTool('sam')
    }
  }, [selectedModel])

  // ── Load a single image file (fast — just one HTTP request) ─────────────────
  const loadImageFile = useCallback(async (imgData: ImageData) => {
    setIsImageLoading(true)
    setError(null)
    setCurrentImage(imgData)
    setAnnotations(imgData.annotations || [])
    setSelectedAnnotation(null)
    setPolygonPoints([])
    setTempBox(null)
    setHistory([])
    setHistoryIndex(-1)
    imageRef.current = null

    return new Promise<void>((resolve) => {
      const img = new Image()
      img.onload = () => {
        imageRef.current = img
        setIsImageLoading(false)
        requestAnimationFrame(() => drawCanvasWithData(img, imgData.annotations || [], null, null, []))
        resolve()
      }
      img.onerror = () => {
        setIsImageLoading(false)
        setError('Failed to load image')
        resolve()
      }
      img.src = `${apiUrl}/api/datasets/${selectedDataset?.id}/image-file/${imgData.path}`
    })
  }, [apiUrl, selectedDataset])

  const navigateImage = useCallback(async (dir: 'prev' | 'next') => {
    const newIdx = dir === 'prev'
      ? Math.max(0, currentIndex - 1)
      : Math.min(filteredImages.length - 1, currentIndex + 1)
    if (newIdx !== currentIndex && filteredImages[newIdx]) {
      // Auto-save current annotations before moving if the setting is on
      if (settings.autoSave && selectedDataset && currentImage && annotations.length > 0) {
        try {
          await fetch(
            `${apiUrl}/api/datasets/${selectedDataset.id}/image/${currentImage.id}/annotations`,
            { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: currentImage.id, annotations }) }
          )
          if (settings.notifications) toast.success('Annotations auto-saved')
        } catch {
          if (settings.notifications) toast.error('Auto-save failed')
        }
      }
      setCurrentIndex(newIdx)
      await loadImageFile(filteredImages[newIdx])
    }
  }, [currentIndex, filteredImages, loadImageFile, settings.autoSave, settings.notifications, selectedDataset, currentImage, annotations, apiUrl])

  // ── Canvas drawing ───────────────────────────────────────────────────────────
  const drawCanvasWithData = useCallback((
    img: HTMLImageElement,
    anns: Annotation[],
    selectedIdx: number | null,
    tBox: { x: number; y: number; w: number; h: number } | null,
    polyPts: { x: number; y: number }[],
    zoomLevel = zoom,
    kpList: { x: number; y: number; label: string }[] = []
  ) => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container || !img) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const cw = container.clientWidth - 32
    const ch = container.clientHeight - 32
    const scale = Math.min(cw / img.width, ch / img.height) * zoomLevel
    scaleRef.current = scale
    canvas.width = img.width * scale
    canvas.height = img.height * scale
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height)

    const COLORS = ['#00D4B4', '#22c55e', '#ab00ff', '#ab00ff', '#ab00ff', '#00D4B4']

    const hexToRgba = (hex: string, alpha: number) => {
      const r = parseInt(hex.slice(1, 3), 16)
      const g = parseInt(hex.slice(3, 5), 16)
      const b = parseInt(hex.slice(5, 7), 16)
      return `rgba(${r},${g},${b},${alpha})`
    }

    const drawLabel = (ctx: CanvasRenderingContext2D, text: string, x: number, y: number, color: string) => {
      ctx.font = 'bold 12px sans-serif'
      const tw = ctx.measureText(text).width + 10
      ctx.fillStyle = color
      ctx.fillRect(x, y - 20, tw, 20)
      ctx.fillStyle = '#fff'
      ctx.fillText(text, x + 5, y - 6)
    }

    anns.forEach((ann, idx) => {
      const isSel = idx === selectedIdx
      const color = isSel ? '#22c55e' : COLORS[(ann.class_id || 0) % COLORS.length]
      ctx.strokeStyle = color
      ctx.lineWidth = isSel ? 3 : 2
      ctx.fillStyle = hexToRgba(color, isSel ? 0.25 : 0.125)

      if (ann.bbox && ann.bbox.length === 4) {
        const [x, y, w, h] = ann.bbox
        ctx.fillRect(x * scale, y * scale, w * scale, h * scale)
        ctx.strokeRect(x * scale, y * scale, w * scale, h * scale)
        drawLabel(ctx, ann.class_name || 'unknown', x * scale, y * scale, color)
      } else if (ann.x_center !== undefined && ann.width !== undefined) {
        const cx = ann.x_center * img.width * scale
        const cy = (ann.y_center || 0) * img.height * scale
        const w = ann.width * img.width * scale
        const h = (ann.height || 0) * img.height * scale
        const x = cx - w / 2, y = cy - h / 2
        ctx.fillRect(x, y, w, h)
        ctx.strokeRect(x, y, w, h)
        drawLabel(ctx, ann.class_name || 'unknown', x, y, color)
      } else if (ann.points && ann.points.length >= 4) {
        const pts = ann.points
        // Points may be normalized (0-1) — detect by checking if all <= 1
        const isNorm = ann.normalized || pts.every(p => p >= 0 && p <= 1)
        const px = (v: number, dim: number) => isNorm ? v * dim * scale : v * scale
        ctx.beginPath()
        ctx.moveTo(px(pts[0], img.width), px(pts[1], img.height))
        for (let i = 2; i < pts.length; i += 2) {
          ctx.lineTo(px(pts[i], img.width), px(pts[i + 1], img.height))
        }
        ctx.closePath()
        ctx.fill()
        ctx.stroke()
        drawLabel(ctx, ann.class_name || 'unknown', px(pts[0], img.width), px(pts[1], img.height), color)
      }
    })

    if (tBox) {
      ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.strokeRect(tBox.x, tBox.y, tBox.w, tBox.h)
      ctx.setLineDash([])
    }

    if (polyPts.length > 0) {
      ctx.fillStyle = '#f59e0b'; ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2; ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(polyPts[0].x, polyPts[0].y)
      for (let i = 1; i < polyPts.length; i++) ctx.lineTo(polyPts[i].x, polyPts[i].y)
      ctx.stroke()
      polyPts.forEach((pt, i) => {
        ctx.beginPath()
        ctx.arc(pt.x, pt.y, i === 0 ? 7 : 5, 0, Math.PI * 2)
        ctx.fillStyle = i === 0 ? '#ef4444' : '#f59e0b'
        ctx.fill()
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke()
        ctx.strokeStyle = '#f59e0b'; ctx.lineWidth = 2
      })
      ctx.fillStyle = 'rgba(0,0,0,0.75)'
      ctx.fillRect(10, canvas.height - 34, 270, 28)
      ctx.fillStyle = '#fff'; ctx.font = '12px sans-serif'
      ctx.fillText(`${polyPts.length} pts — click to add · Enter/DblClick to finish`, 16, canvas.height - 14)
    }

    // Draw in-progress keypoints
    if (kpList.length > 0) {
      kpList.forEach((kp, i) => {
        const px = kp.x * img.width * scale
        const py = kp.y * img.height * scale
        ctx.beginPath()
        ctx.arc(px, py, 6, 0, Math.PI * 2)
        ctx.fillStyle = '#a855f7'
        ctx.fill()
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 2
        ctx.stroke()
        // Draw cross
        ctx.strokeStyle = '#fff'
        ctx.lineWidth = 1.5
        ctx.beginPath()
        ctx.moveTo(px - 9, py); ctx.lineTo(px + 9, py)
        ctx.moveTo(px, py - 9); ctx.lineTo(px, py + 9)
        ctx.stroke()
        // Label
        ctx.font = 'bold 10px sans-serif'
        ctx.fillStyle = '#a855f7'
        ctx.fillText(kp.label || String(i + 1), px + 8, py - 4)
      })
    }

    // Draw SAM pending click markers
    const samClicks = samPendingClicksRef.current
    samClicks.forEach(click => {
      const cx = click.nx * img.width * scale
      const cy = click.ny * img.height * scale
      const isProcessing = click.status === 'processing'
      const color = isProcessing ? '#f59e0b' : '#ef4444'
      // Crosshair lines
      ctx.strokeStyle = color
      ctx.lineWidth = 1.5
      ctx.setLineDash([])
      ctx.beginPath()
      ctx.moveTo(cx - 18, cy); ctx.lineTo(cx + 18, cy)
      ctx.moveTo(cx, cy - 18); ctx.lineTo(cx, cy + 18)
      ctx.stroke()
      // Outer ring
      ctx.beginPath()
      ctx.arc(cx, cy, 11, 0, Math.PI * 2)
      ctx.strokeStyle = color
      ctx.lineWidth = 2
      ctx.stroke()
      // Inner fill dot
      ctx.beginPath()
      ctx.arc(cx, cy, 5, 0, Math.PI * 2)
      ctx.fillStyle = color
      ctx.fill()
      // Status label
      ctx.font = 'bold 10px sans-serif'
      ctx.fillStyle = color
      ctx.fillText(isProcessing ? '…' : '+', cx + 13, cy - 6)
    })
  }, [zoom])

  const drawCanvas = useCallback(() => {
    if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList)
  }, [annotations, selectedAnnotation, tempBox, polygonPoints, zoom, keypointList, drawCanvasWithData])

  useEffect(() => {
    if (!isImageLoading && imageRef.current) drawCanvas()
  }, [annotations, selectedAnnotation, zoom, isImageLoading, keypointList])

  // ── Canvas interaction ───────────────────────────────────────────────────────
  const getCanvasCoords = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current
    if (!canvas) return { x: 0, y: 0 }
    const rect = canvas.getBoundingClientRect()
    return {
      x: (e.clientX - rect.left) * (canvas.width / rect.width),
      y: (e.clientY - rect.top) * (canvas.height / rect.height)
    }
  }

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const coords = getCanvasCoords(e)
    if (activeTool === 'bbox') {
      setIsDrawing(true); setDrawStart(coords)
    } else if (activeTool === 'polygon') {
      const newPts = [...polygonPoints, coords]
      setPolygonPoints(newPts)
      if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, newPts, zoom)
    } else if (activeTool === 'brush') {
      setIsBrushing(true)
      setBrushPoints([coords])
    } else if (activeTool === 'keypoint') {
      const scale = scaleRef.current
      const img = imageRef.current
      if (!img) return
      const kp = { x: coords.x / scale / img.width, y: coords.y / scale / img.height, label: activeClass || 'point' }
      const newKps = [...keypointList, kp]
      setKeypointList(newKps)
      // Redraw with keypoints shown
      if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, tempBox, polygonPoints, zoom, newKps)
    } else if (activeTool === 'sam') {
      enqueueSamClick(coords)
    } else {
      // In select mode: if clicking on already-selected bbox, start drag
      if (activeTool === 'select' && selectedAnnotation !== null) {
        const ann = annotations[selectedAnnotation]
        if (ann?.bbox?.length === 4) {
          const scale = scaleRef.current
          const [x, y, w, h] = ann.bbox
          const imgX = coords.x / scale, imgY = coords.y / scale
          if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) {
            setIsDragging(true)
            setDragStart({ x: coords.x, y: coords.y, origBbox: [...ann.bbox] })
            return
          }
        }
      }
      handleAnnotationClick(coords)
    }
  }

  const handleCanvasMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool === 'brush' && isBrushing) {
      const coords = getCanvasCoords(e)
      setBrushPoints(prev => [...prev, coords])
      // Draw brush stroke on canvas
      const canvas = canvasRef.current
      const ctx = canvas?.getContext('2d')
      if (ctx) {
        ctx.beginPath()
        ctx.arc(coords.x, coords.y, brushSize, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(59,130,246,0.35)'
        ctx.fill()
      }
      return
    }
    // Drag selected annotation
    if (isDragging && dragStart && selectedAnnotation !== null) {
      const coords = getCanvasCoords(e)
      const scale = scaleRef.current
      const dx = (coords.x - dragStart.x) / scale
      const dy = (coords.y - dragStart.y) / scale
      const [ox, oy, ow, oh] = dragStart.origBbox
      const next = annotations.map((ann, i) =>
        i === selectedAnnotation ? { ...ann, bbox: [ox + dx, oy + dy, ow, oh] } : ann
      )
      setAnnotations(next)
      return
    }
    if (!isDrawing || !drawStart || activeTool !== 'bbox') return
    const c = getCanvasCoords(e)
    const box = { x: Math.min(drawStart.x, c.x), y: Math.min(drawStart.y, c.y), w: Math.abs(c.x - drawStart.x), h: Math.abs(c.y - drawStart.y) }
    setTempBox(box)
    if (imageRef.current) drawCanvasWithData(imageRef.current, annotations, selectedAnnotation, box, polygonPoints, zoom, keypointList)
  }

  const handleCanvasMouseUp = () => {
    if (isDragging) {
      setIsDragging(false)
      setDragStart(null)
      saveToHistory(annotations)
      return
    }
    if (activeTool === 'brush' && isBrushing) {
      setIsBrushing(false)
      if (brushPoints.length > 3) {
        // Convert brush stroke to a polygon annotation
        const scale = scaleRef.current
        const img = imageRef.current
        if (!img) { setBrushPoints([]); return }
        // Simplify brush points to convex hull approx
        const pts = brushPoints.flatMap(p => [p.x / scale / img.width, p.y / scale / img.height])
        const ann: Annotation = {
          type: 'polygon',
          class_id: localClasses.indexOf(activeClass),
          class_name: activeClass || 'brush_mask',
          points: pts,
          normalized: true
        }
        const next = [...annotations, ann]
        setAnnotations(next); saveToHistory(next)
      }
      setBrushPoints([])
      return
    }
    if (!isDrawing || !tempBox || activeTool !== 'bbox') {
      setIsDrawing(false); setDrawStart(null); setTempBox(null); return
    }
    const scale = scaleRef.current
    const bbox = [tempBox.x / scale, tempBox.y / scale, tempBox.w / scale, tempBox.h / scale]
    if (bbox[2] > 5 && bbox[3] > 5) {
      const ann: Annotation = {
        type: 'bbox',
        class_id: localClasses.indexOf(activeClass),
        class_name: activeClass || 'unknown',
        bbox
      }
      const next = [...annotations, ann]
      setAnnotations(next)
      saveToHistory(next)
    }
    setIsDrawing(false); setDrawStart(null); setTempBox(null)
  }

  const enqueueSamClick = (coords: { x: number; y: number }) => {
    if (!selectedDataset || !currentImage || !selectedModel) {
      toast.error('Select a SAM-compatible model first')
      return
    }
    const scale = scaleRef.current
    const img = imageRef.current
    if (!img) return
    const nx = coords.x / scale / img.width
    const ny = coords.y / scale / img.height
    const id = Math.random().toString(36).slice(2)
    const newClick = { id, nx, ny, status: 'pending' as const }
    const updated = [...samPendingClicksRef.current, newClick]
    samPendingClicksRef.current = updated
    setSamPendingClicks(updated)
  }

  const commitKeypoints = () => {
    if (keypointList.length === 0) return
    const pts = keypointList.flatMap(kp => [kp.x, kp.y])
    const ann: Annotation = {
      type: 'keypoints',
      class_id: localClasses.indexOf(activeClass),
      class_name: activeClass || 'skeleton',
      points: pts,
      normalized: true
    }
    const next = [...annotations, ann]
    setAnnotations(next); saveToHistory(next)
    setKeypointList([])
    toast.success(`Added ${keypointList.length} keypoints`)
  }

  const copyAnnotationsToNext = async (count: number) => {
    if (!selectedDataset || !currentImage || annotations.length === 0) {
      toast.error('No annotations to copy')
      return
    }
    let copied = 0
    for (let i = 1; i <= count; i++) {
      const targetIdx = currentIndex + i
      if (targetIdx >= filteredImages.length) break
      const target = filteredImages[targetIdx]
      try {
        await fetch(
          `${apiUrl}/api/datasets/${selectedDataset.id}/image/${target.id}/annotations`,
          { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: target.id, annotations }) }
        )
        copied++
      } catch {}
    }
    toast.success(`Annotations copied to next ${copied} image(s)`)
  }

  const completePolygon = useCallback(() => {
    if (polygonPoints.length < 3) return
    const scale = scaleRef.current
    const points = polygonPoints.flatMap(pt => [pt.x / scale, pt.y / scale])
    const ann: Annotation = {
      type: 'polygon',
      class_id: localClasses.indexOf(activeClass),
      class_name: activeClass || 'unknown',
      points
    }
    const next = [...annotations, ann]
    setAnnotations(next)
    saveToHistory(next)
    setPolygonPoints([])
  }, [polygonPoints, annotations, activeClass, selectedDataset])

  const handleAnnotationClick = (coords: { x: number; y: number }) => {
    const scale = scaleRef.current
    const imgX = coords.x / scale, imgY = coords.y / scale
    for (let i = annotations.length - 1; i >= 0; i--) {
      const ann = annotations[i]
      if (ann.bbox && ann.bbox.length === 4) {
        const [x, y, w, h] = ann.bbox
        if (imgX >= x && imgX <= x + w && imgY >= y && imgY <= y + h) { setSelectedAnnotation(i); return }
      } else if (ann.x_center !== undefined) {
        const img = imageRef.current
        const iw = img?.width || 1, ih = img?.height || 1
        const cx = ann.x_center * iw, cy = (ann.y_center || 0) * ih
        const w = ann.width! * iw, h = (ann.height || 0) * ih
        if (imgX >= cx - w/2 && imgX <= cx + w/2 && imgY >= cy - h/2 && imgY <= cy + h/2) { setSelectedAnnotation(i); return }
      }
    }
    setSelectedAnnotation(null)
  }

  const saveToHistory = (anns: Annotation[]) => {
    const h = history.slice(0, historyIndex + 1)
    h.push([...anns])
    setHistory(h)
    setHistoryIndex(h.length - 1)
  }

  const undo = () => {
    if (historyIndex > 0) { setHistoryIndex(historyIndex - 1); setAnnotations([...history[historyIndex - 1]]) }
    else if (historyIndex === 0 && currentImage) { setAnnotations(currentImage.annotations || []); setHistoryIndex(-1) }
  }

  const deleteAnnotation = () => {
    if (selectedAnnotation === null) return
    const next = annotations.filter((_, i) => i !== selectedAnnotation)
    setAnnotations(next); saveToHistory(next); setSelectedAnnotation(null)
  }

  const saveAnnotations = async (andAdvance = false, explicitAnns?: Annotation[]) => {
    if (!selectedDataset || !currentImage) return
    const annsToSave = explicitAnns ?? annotations
    setIsSaving(true)
    try {
      const resp = await fetch(
        `${apiUrl}/api/datasets/${selectedDataset.id}/image/${currentImage.id}/annotations`,
        { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: currentImage.id, annotations: annsToSave }) }
      )
      if (!resp.ok) throw new Error('Failed to save')
      // Update global image cache entry — set silent flag so useEffect doesn't reset to image 0
      const updated = allImages.map(img => img.id === currentImage.id ? { ...img, annotations: annsToSave } : img)
      silentRefreshRef.current = true
      setAllImages(updated)
      updateImageCache(selectedDataset.id, updated)

      if (settings.notifications) toast.success('Annotations saved')
      if (andAdvance && currentIndex < filteredImages.length - 1) {
        await navigateImage('next')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Save failed')
      if (settings.notifications) toast.error('Save failed')
    } finally {
      setIsSaving(false)
    }
  }

  const [isModelLoading, setIsModelLoading] = useState(false)

  const autoAnnotate = async () => {
    if (!selectedDataset || !currentImage || !selectedModel) return
    // Check if model is downloaded but not loaded into memory — first call will load it
    const modelInfo = models.find(m => m.id === selectedModel)
    const needsLoad = modelInfo?.downloaded && !modelInfo?.loaded
    if (needsLoad) {
      setIsModelLoading(true)
      toast.info('Loading model into memory — first annotation may take 10–30 s…', { duration: 20000, id: 'model-load' })
    }
    setIsAutoAnnotating(true)
    try {
      const params = new URLSearchParams({
        model_id: selectedModel,
        confidence_threshold: String(confidence),
      })
      // Pass text prompt for zero-shot models (YOLO-World, GroundingDINO)
      if (combinedPrompt) params.set('text_prompt', combinedPrompt)
      if (currentImage.path) params.set('image_path_hint', currentImage.path)
      const resp = await fetch(
        `${apiUrl}/api/auto-annotate/${selectedDataset.id}/single/${currentImage.id}?${params}`,
        { method: 'POST' }
      )
      if (needsLoad) toast.dismiss('model-load')
      if (!resp.ok) throw new Error('Auto-annotation failed')
      const data = await resp.json()
      if (data.annotations?.length) {
        const next = [...annotations, ...data.annotations]
        setAnnotations(next)
        saveToHistory(next)
        // Save to dataset immediately so annotations persist
        await saveAnnotations(false, next)
      } else {
        toast.info('No objects detected')
      }
      // Refresh model list so the model shows as loaded now
      if (needsLoad) loadModels()
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Auto-annotation failed')
    } finally {
      setIsAutoAnnotating(false)
      setIsModelLoading(false)
    }
  }

  const submitNewClass = async () => {
    const name = newClassName.trim()
    if (!name || !selectedDataset) return
    setAddingClass(false)
    setNewClassName('')
    // Optimistic update — show the class immediately without waiting for the slow backend parse
    if (!localClasses.includes(name)) {
      setLocalClasses(prev => [...prev, name])
      setAvailableClasses(prev => [...prev, name])
    }
    setActiveClass(name)
    toast.success(`Class "${name}" added`)
    try {
      const resp = await fetch(
        `${apiUrl}/api/datasets/${selectedDataset.id}/add-classes`,
        { method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dataset_id: selectedDataset.id, new_classes: [name], use_model: false }) }
      )
      if (!resp.ok) throw new Error('Failed to add class')
      const data = await resp.json()
      const updatedClasses: string[] = data.updated_dataset?.classes || []
      if (updatedClasses.length > 0) {
        setLocalClasses(updatedClasses)
        setAvailableClasses(updatedClasses)
      }
    } catch {
      // Revert optimistic update on failure
      setLocalClasses(prev => prev.filter(c => c !== name))
      setAvailableClasses(prev => prev.filter(c => c !== name))
      toast.error('Failed to save class to backend')
    }
  }

  const combinedPrompt = promptTags.join(', ')

  const addPromptTag = (raw: string) => {
    const tags = raw.split(',').map(t => t.trim()).filter(Boolean)
    setPromptTags(prev => [...prev, ...tags.filter(t => !prev.includes(t))])
    setPromptInput('')
  }

  const removePromptTag = (tag: string) =>
    setPromptTags(prev => prev.filter(t => t !== tag))

  /**
   * Poll a single batch job until it finishes.
   * Uses refs so the interval stays valid even after component remounts.
   * Every 5 polls it re-fetches the current image's annotations so they
   * appear live while the job is running.
   * Deduplicates: calling this twice for the same job_id cancels the old interval.
   */
  const _pollJob = (job_id: string) => {
    // Cancel any existing interval for this job to avoid duplicates
    if (pollIntervalsRef.current[job_id]) {
      clearInterval(pollIntervalsRef.current[job_id])
    }
    let tick = 0
    const iv = setInterval(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/status`)
        if (!r.ok) { clearInterval(iv); delete pollIntervalsRef.current[job_id]; return }
        const data = await r.json()
        setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], ...data } }))

        // Every 5 ticks while running, refresh annotations for whichever image
        // is currently on screen so batch results appear without navigating away.
        tick++
        if (tick % 5 === 0 && (data.status === 'running' || data.status === 'paused')) {
          const ds = selectedDatasetRef.current
          const img = currentImageRef.current
          if (ds && img) {
            try {
              const ir = await fetch(`${apiUrl}/api/datasets/${ds.id}/image/${img.id}`)
              if (ir.ok) {
                const imgData = await ir.json()
                const fresh: Annotation[] = imgData.annotations || []
                // Only overwrite if the backend has MORE annotations (don't stomp user edits)
                if (fresh.length > annotationsRef.current.length) {
                  setAnnotations(fresh)
                }
              }
            } catch {}
          }
        }

        const done = data.status === 'done' || data.status === 'error' || data.status === 'cancelled'
        if (done) {
          clearInterval(iv)
          delete pollIntervalsRef.current[job_id]
          if (data.status === 'done') {
            toast.success(
              `Batch done — ${data.annotated} images annotated, ` +
              `${data.total_annotations ?? 0} annotations created`
            )
          } else if (data.status === 'error') {
            toast.error(`Batch failed: ${data.error}`)
          }
          // Silent refresh: update image list / annotations without resetting canvas view
          refreshImageListSilent()
        }
        // 'paused' keeps polling so stats stay live
      } catch { clearInterval(iv); delete pollIntervalsRef.current[job_id] }
    }, 1500)
    pollIntervalsRef.current[job_id] = iv
  }

  // When switching to annotate tab, refresh the current image's annotations
  // from the backend so batch results are immediately visible.
  useEffect(() => {
    if (sidebarTab !== 'annotate') return
    const ds = selectedDatasetRef.current
    const img = currentImageRef.current
    if (!ds || !img) return
    ;(async () => {
      try {
        const r = await fetch(`${apiUrl}/api/datasets/${ds.id}/image/${img.id}`)
        if (r.ok) {
          const data = await r.json()
          const fresh: Annotation[] = data.annotations || []
          if (fresh.length > annotationsRef.current.length) {
            setAnnotations(fresh)
          }
        }
      } catch {}
    })()
  }, [sidebarTab]) // eslint-disable-line react-hooks/exhaustive-deps

  const textAnnotateSingle = async () => {
    if (!selectedDataset || !currentImage || !selectedModel || !combinedPrompt) return
    setIsAutoAnnotating(true)
    try {
      const params = new URLSearchParams({
        model_id: selectedModel,
        confidence_threshold: String(confidence),
        text_prompt: combinedPrompt,
      })
      if (currentImage.path) params.set('image_path_hint', currentImage.path)
      const resp = await fetch(
        `${apiUrl}/api/auto-annotate/${selectedDataset.id}/single/${currentImage.id}?${params}`,
        { method: 'POST' }
      )
      if (!resp.ok) {
        let detail = 'Text annotation failed'
        try { detail = (await resp.json()).detail ?? detail } catch {}
        throw new Error(detail)
      }
      const data = await resp.json()
      if (data.annotations?.length) {
        const next = [...annotations, ...data.annotations]
        setAnnotations(next)
        saveToHistory(next)
        await saveAnnotations(false, next)
        toast.success(`Found ${data.annotations.length} annotation(s) for "${combinedPrompt}"`)
      } else {
        toast.info('No objects matched the text prompt on this image')
      }
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Text annotation failed')
    } finally {
      setIsAutoAnnotating(false)
    }
  }

  const textAnnotateBatch = async () => {
    if (!selectedDataset || !selectedModel || isStartingBatch) return
    const selModel = models.find(m => m.id === selectedModel)
    const isTextModel = selModel && ['sam3', 'yoloworld', 'groundingdino', 'owlvit'].includes(selModel.type)
    if (isTextModel && !combinedPrompt) return
    setIsStartingBatch(true)
    try {
      const params = new URLSearchParams({
        model_id: selectedModel,
        confidence_threshold: String(confidence),
      })
      if (combinedPrompt) params.set('text_prompt', combinedPrompt)
      const resp = await fetch(
        `${apiUrl}/api/auto-annotate/${selectedDataset.id}/text-batch?${params}`,
        { method: 'POST' }
      )
      if (!resp.ok) throw new Error('Failed to start batch annotation')
      const { job_id } = await resp.json()

      // Register job in state immediately so the monitor shows it right away
      setBatchJobs(prev => ({
        ...prev,
        [job_id]: {
          job_id,
          status: 'running',
          paused: false,
          progress: 0,
          total: 0,
          processed: 0,
          annotated: 0,
          failed: 0,
          total_annotations: 0,
          text_prompt: combinedPrompt || '',
          started_at: new Date().toISOString(),
          dataset_id: selectedDataset.id,
          recent_images: [],
        },
      }))
      toast.success(`Batch job started${combinedPrompt ? ` (${combinedPrompt})` : ''}`)
      setSidebarTab('jobs')
      _pollJob(job_id)
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Batch annotation failed')
    } finally {
      setIsStartingBatch(false)
    }
  }

  const deleteAllAnnotations = async () => {
    if (!selectedDataset) return
    if (!confirm(`Delete ALL annotations from "${selectedDataset.name}"? This cannot be undone.`)) return
    if (!confirm(`Are you sure? All annotation labels in this dataset will be permanently cleared.`)) return
    try {
      const resp = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/annotations`, { method: 'DELETE' })
      if (!resp.ok) throw new Error('Failed to delete annotations')
      setAnnotations([])
      // Invalidate image cache so annotation counts refresh
      updateImageCache(selectedDataset.id, [])
      toast.success('All annotations deleted')
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Failed to delete annotations')
    }
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
    } catch { toast.error('Failed to resume job') }
  }

  const cancelJob = async (job_id: string) => {
    try {
      await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/cancel`, { method: 'POST' })
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'cancelled' } }))
    } catch { toast.error('Failed to cancel job') }
  }

  const undoJob = async (job_id: string) => {
    try {
      const resp = await fetch(`${apiUrl}/api/auto-annotate/text-batch/${job_id}/undo`, { method: 'POST' })
      if (!resp.ok) throw new Error((await resp.json()).detail ?? 'Undo failed')
      setBatchJobs(prev => ({ ...prev, [job_id]: { ...prev[job_id], status: 'undone' } }))
      await refreshImageListSilent()
      toast.success('Batch annotations undone — dataset restored to pre-job state')
    } catch (err) { toast.error(err instanceof Error ? err.message : 'Undo failed') }
  }

  // ── Keyboard shortcuts ───────────────────────────────────────────────────────
  useEffect(() => {
    const onKey = async (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return
      if (e.key === 's' || e.key === 'S') { e.preventDefault(); await saveAnnotations(true) }
      if ((e.key === 'Delete' || e.key === 'Backspace') && selectedAnnotation !== null) { e.preventDefault(); deleteAnnotation() }
      if (e.key === 'ArrowLeft' && e.altKey) { e.preventDefault(); navigateImage('prev') }
      if (e.key === 'ArrowRight' && e.altKey) { e.preventDefault(); navigateImage('next') }
      if (e.key === 'Escape') { setSelectedAnnotation(null); setPolygonPoints([]); setTempBox(null); setIsDrawing(false); setKeypointList([]) }
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') { e.preventDefault(); undo() }
      if (e.key === 'Enter') {
        e.preventDefault()
        if (polygonPoints.length >= 3) completePolygon()
        else if (keypointList.length > 0) commitKeypoints()
      }
      // Tool shortcuts
      if (e.key === 'v') setActiveTool('select')
      if (e.key === 'b') setActiveTool('bbox')
      if (e.key === 'p') { setActiveTool('polygon'); setPolygonPoints([]) }
      if (e.key === 'k') setActiveTool('keypoint')
      if (e.key === 'r') setActiveTool('brush')
      if (e.key === 'q') setActiveTool('sam')
      // 1-9 class assignment shortcuts
      const num = parseInt(e.key)
      if (!isNaN(num) && num >= 1 && num <= 9 && localClasses.length > 0) {
        const cls = localClasses[num - 1]
        if (cls) {
          setActiveClass(cls)
          // Also update selected annotation's class if one is selected
          if (selectedAnnotation !== null) {
            const next = annotations.map((ann, i) => i === selectedAnnotation
              ? { ...ann, class_name: cls, class_id: num - 1 }
              : ann
            )
            setAnnotations(next); saveToHistory(next)
            toast.success(`Assigned class: ${cls}`)
          } else {
            toast.info(`Active class: ${cls}`)
          }
        }
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, filteredImages, selectedAnnotation, annotations, polygonPoints, keypointList, selectedDataset, navigateImage, localClasses])

  // ── Empty state ──────────────────────────────────────────────────────────────
  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground">No dataset selected</h3>
          <p className="text-sm text-muted-foreground/70 mt-1">Select a dataset from the Datasets view</p>
        </div>
      </div>
    )
  }

  if (isLoadingList) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <Loader2 className="w-12 h-12 animate-spin text-primary mx-auto mb-4" />
          <h3 className="text-lg font-medium">Loading Dataset</h3>
          <p className="text-sm text-muted-foreground mt-1">Loading images from {selectedDataset.name}...</p>
          <p className="text-xs text-muted-foreground/60 mt-1">Large datasets may take a moment</p>
        </div>
      </div>
    )
  }

  // ── No images found after loading ────────────────────────────────────────────
  if (!isLoadingList && filteredImages.length === 0 && allImages.length === 0) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center space-y-3">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto" />
          <h3 className="text-lg font-medium text-muted-foreground">No images found</h3>
          <p className="text-sm text-muted-foreground/70">
            Dataset "{selectedDataset.name}" has no images, or they couldn't be loaded.
          </p>
          <button
            onClick={loadImageList}
            className="text-sm text-primary underline underline-offset-2"
          >
            Retry loading images
          </button>
          {error && <p className="text-xs text-red-400 mt-1">{error}</p>}
        </div>
      </div>
    )
  }

  // ── Elapsed time helper ───────────────────────────────────────────────────────
  const fmtElapsed = (startedAt: string | number) => {
    const ms = typeof startedAt === 'number' ? startedAt : new Date(startedAt).getTime()
    const s = Math.floor((Date.now() - ms) / 1000)
    if (s < 0 || isNaN(s)) return '—'
    if (s < 60) return `${s}s`
    const m = Math.floor(s / 60)
    return `${m}m ${s % 60}s`
  }

  const datasetJobs = Object.values(batchJobs).filter(j => !selectedDataset || j.dataset_id === selectedDataset.id)
  const runningJobCount = datasetJobs.filter(j => j.status === 'running' || j.status === 'paused').length
  const totalJobCount = datasetJobs.length

  // ── Main UI ──────────────────────────────────────────────────────────────────
  return (
    <div className="h-full flex flex-col">
      {/* Resume-progress banner */}
      {resumeBanner && (
        <div className="flex items-center justify-between gap-3 px-4 py-2 bg-blue-500/10 border-b border-blue-500/20 text-sm">
          <span className="text-blue-400">
            You were on image <strong>{resumeBanner.imageIndex + 1}</strong> — <span className="font-mono text-xs">{resumeBanner.imageName}</span>. Continue from here?
          </span>
          <div className="flex gap-2 flex-shrink-0">
            <Button size="sm" variant="outline" className="h-7 text-xs" onClick={() => {
              setResumeBanner(null)
              try { localStorage.removeItem(`cvdm_annotate_${selectedDataset?.id}`) } catch {}
            }}>Start from beginning</Button>
            <Button size="sm" className="h-7 text-xs" onClick={() => {
              if (!resumeBanner) return
              setSelectedSplit(resumeBanner.split || 'all')
              setSelectedClass(resumeBanner.classFilter || 'all')
              initialImageIdRef.current = resumeBanner.imageId
              setResumeBanner(null)
            }}>Resume</Button>
          </div>
        </div>
      )}
      <div className="flex-1 flex min-h-0">
      {/* Canvas side — always visible */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Toolbar */}
        <div className="flex items-center gap-2 p-2 border-b border-border bg-card flex-wrap">
          {/* Drawing tools */}
          <div className="flex items-center gap-1">
            <Button size="sm" variant={activeTool === 'select' ? 'default' : 'outline'} onClick={() => setActiveTool('select')} title="Select (V)">
              <MousePointer2 className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'bbox' ? 'default' : 'outline'} onClick={() => setActiveTool('bbox')} title="Bounding Box (B)">
              <Square className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'polygon' ? 'default' : 'outline'}
              onClick={() => { setActiveTool('polygon'); setPolygonPoints([]) }} title="Polygon (P)">
              <Pentagon className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'keypoint' ? 'default' : 'outline'}
              onClick={() => { setActiveTool('keypoint'); setKeypointList([]) }} title="Keypoints (K)">
              <Crosshair className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'brush' ? 'default' : 'outline'}
              onClick={() => setActiveTool('brush')} title="Brush / Paint mask (R)">
              <Brush className="w-4 h-4" />
            </Button>
            <Button size="sm" variant={activeTool === 'sam' ? 'default' : 'outline'}
              onClick={() => setActiveTool('sam')} title="SAM click-to-segment (Q)">
              {samPendingClicks.some(c => c.status === 'processing') ? <Loader2 className="w-4 h-4 animate-spin" /> : <Wand2 className="w-4 h-4" />}
              {samPendingClicks.length > 0 && <span className="ml-1 text-xs">{samPendingClicks.length}</span>}
            </Button>
          </div>

          {/* Brush size (shown when brush active) */}
          {activeTool === 'brush' && (
            <div className="flex items-center gap-2">
              <span className="text-xs text-muted-foreground">Size:</span>
              <Slider value={[brushSize]} onValueChange={([v]) => setBrushSize(v)} min={5} max={80} step={5} className="w-20" />
              <span className="text-xs w-6">{brushSize}</span>
            </div>
          )}

          <div className="w-px h-6 bg-border" />

          {/* Active annotation class */}
          <Select value={activeClass} onValueChange={cls => {
            setActiveClass(cls)
            // If an annotation is selected, also change its class
            if (selectedAnnotation !== null) {
              const clsIdx = localClasses.indexOf(cls)
              const next = annotations.map((ann, i) =>
                i === selectedAnnotation ? { ...ann, class_name: cls, class_id: clsIdx } : ann
              )
              setAnnotations(next)
              saveToHistory(next)
            }
          }}>
            <SelectTrigger className="w-36 h-8 text-xs">
              <SelectValue placeholder="Class" />
            </SelectTrigger>
            <SelectContent>
              {localClasses.map((cls, i) => (
                <SelectItem key={cls} value={cls}>
                  <span className="flex items-center gap-2">
                    {i < 9 && <kbd className="text-[9px] px-1 py-0.5 bg-muted rounded font-mono">{i+1}</kbd>}
                    {cls}
                  </span>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Add new class button */}
          {addingClass ? (
            <form className="flex items-center gap-1" onSubmit={e => { e.preventDefault(); submitNewClass() }}>
              <Input
                autoFocus
                value={newClassName}
                onChange={e => setNewClassName(e.target.value)}
                placeholder="class name"
                className="h-8 w-24 text-xs bg-background border-border"
                onKeyDown={e => { if (e.key === 'Escape') { setAddingClass(false); setNewClassName('') } }}
              />
              <Button type="submit" size="sm" variant="outline" className="h-8 px-2 text-xs" disabled={!newClassName.trim()}>
                Add
              </Button>
              <Button type="button" size="sm" variant="ghost" className="h-8 px-1" onClick={() => { setAddingClass(false); setNewClassName('') }}>
                <X className="w-3 h-3" />
              </Button>
            </form>
          ) : (
            <Button size="sm" variant="outline" className="h-8 px-2" title="Add new class" onClick={() => setAddingClass(true)}>
              <Plus className="w-3.5 h-3.5" />
            </Button>
          )}

          <div className="w-px h-6 bg-border" />

          {/* Split filter */}
          {availableSplits.length > 0 && (
            <Select value={selectedSplit} onValueChange={setSelectedSplit}>
              <SelectTrigger className="w-32 h-8 text-xs">
                <FolderOpen className="w-3 h-3 mr-1" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All splits</SelectItem>
                {availableSplits.map(s => (
                  <SelectItem key={s} value={s}>{s} ({allImages.filter(i => i.split === s).length})</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          {/* Class filter */}
          {availableClasses.length > 0 && (
            <Select value={selectedClass} onValueChange={setSelectedClass}>
              <SelectTrigger className="w-32 h-8 text-xs">
                <Tag className="w-3 h-3 mr-1" />
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="all">All classes</SelectItem>
                {availableClasses.map(c => (
                  <SelectItem key={c} value={c}>{c}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          )}

          <div className="flex-1" />

          {/* Keyboard shortcut hint */}
          <Button size="sm" variant="ghost" onClick={() => setShowShortcutHint(h => !h)} title="Keyboard shortcuts">
            <Keyboard className="w-4 h-4" />
          </Button>

          {/* Undo + zoom */}
          <Button size="sm" variant="outline" onClick={undo} disabled={historyIndex < 0} title="Undo (Ctrl+Z)">
            <Undo2 className="w-4 h-4" />
          </Button>
          <div className="w-px h-6 bg-border" />
          <Button size="sm" variant="outline" onClick={() => setZoom(z => Math.max(0.25, z - 0.1))}><ZoomOut className="w-4 h-4" /></Button>
          <span className="text-[10px] font-mono w-12 text-center text-muted-foreground">{Math.round(zoom * 100)}%</span>
          <Button size="sm" variant="outline" onClick={() => setZoom(z => Math.min(4, z + 0.1))}><ZoomIn className="w-4 h-4" /></Button>
          <Button size="sm" variant="outline" onClick={() => setZoom(1)} title="Reset zoom"><RotateCcw className="w-4 h-4" /></Button>

          <div className="w-px h-6 bg-border" />

          {/* Navigation */}
          <Button size="sm" variant="outline" onClick={() => navigateImage('prev')} disabled={currentIndex === 0}><ChevronLeft className="w-4 h-4" /></Button>
          <span className="text-[10px] font-mono text-muted-foreground w-20 text-center">{currentIndex + 1} / {filteredImages.length}</span>
          <Button size="sm" variant="outline" onClick={() => navigateImage('next')} disabled={currentIndex >= filteredImages.length - 1}><ChevronRight className="w-4 h-4" /></Button>

          {/* Jobs indicator — clicking opens the Jobs sidebar tab */}
          {totalJobCount > 0 && (
            <>
              <div className="w-px h-6 bg-border" />
              <Button
                size="sm"
                variant={sidebarTab === 'jobs' ? 'default' : 'outline'}
                onClick={() => setSidebarTab('jobs')}
                className="gap-1.5 text-xs font-mono"
                title="View batch jobs"
              >
                {runningJobCount > 0 && <span className="live-dot text-primary" />}
                Jobs
                {runningJobCount > 0 && (
                  <span className="px-1.5 py-0 bg-primary/15 text-primary text-[9px] rounded font-mono leading-4">
                    {runningJobCount}
                  </span>
                )}
              </Button>
            </>
          )}
        </div>

        {/* Keyboard shortcut overlay */}
        {showShortcutHint && (
          <div className="absolute top-16 right-72 z-50 bg-card border border-border rounded-lg shadow-xl p-4 text-xs w-64">
            <div className="font-semibold mb-2 flex items-center justify-between">
              Keyboard Shortcuts
              <button onClick={() => setShowShortcutHint(false)} className="text-muted-foreground hover:text-foreground">✕</button>
            </div>
            <div className="space-y-1 text-muted-foreground">
              <div className="grid grid-cols-2 gap-x-4 gap-y-1">
                <span><kbd className="px-1 bg-muted rounded">V</kbd> Select</span>
                <span><kbd className="px-1 bg-muted rounded">B</kbd> BBox</span>
                <span><kbd className="px-1 bg-muted rounded">P</kbd> Polygon</span>
                <span><kbd className="px-1 bg-muted rounded">K</kbd> Keypoints</span>
                <span><kbd className="px-1 bg-muted rounded">R</kbd> Brush</span>
                <span><kbd className="px-1 bg-muted rounded">Q</kbd> SAM</span>
                <span><kbd className="px-1 bg-muted rounded">S</kbd> Save & Next</span>
                <span><kbd className="px-1 bg-muted rounded">Del</kbd> Delete</span>
                <span><kbd className="px-1 bg-muted rounded">Enter</kbd> Finish</span>
                <span><kbd className="px-1 bg-muted rounded">Ctrl+Z</kbd> Undo</span>
                <span><kbd className="px-1 bg-muted rounded">Esc</kbd> Cancel</span>
                <span><kbd className="px-1 bg-muted rounded">1-9</kbd> Class</span>
              </div>
              <div className="mt-2 pt-2 border-t border-border">
                Press <kbd className="px-1 bg-muted rounded">1</kbd>–<kbd className="px-1 bg-muted rounded">9</kbd> to assign the first 9 classes instantly.
              </div>
            </div>
          </div>
        )}

        {/* ── Annotation canvas view ──────────────────────────────────────── */}
        <>

        {/* Canvas — outer div is the scroll container; inner div centers content when smaller than viewport */}
        <div ref={containerRef} className="flex-1 overflow-auto bg-muted/30 relative">
          <div className="min-h-full flex items-center justify-center p-4">
          {isImageLoading ? (
            <div className="flex flex-col items-center gap-3">
              <Loader2 className="w-10 h-10 animate-spin text-primary" />
              <p className="text-sm font-medium">Loading image...</p>
            </div>
          ) : error && !imageRef.current ? (
            <div className="flex flex-col items-center gap-3 text-destructive">
              <AlertCircle className="w-10 h-10" />
              <p className="text-sm font-medium">{error}</p>
              <p className="text-xs text-muted-foreground">Image file could not be served — check dataset paths</p>
            </div>
          ) : currentImage ? (
            <>
              <canvas
                ref={canvasRef}
                className={cn(
                  "shadow-lg rounded",
                  activeTool !== 'select' && "cursor-crosshair",
                  activeTool === 'select' && "cursor-pointer",
                  activeTool === 'brush' && "cursor-cell",
                  activeTool === 'sam' && "cursor-cell"
                )}
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
                onMouseLeave={handleCanvasMouseUp}
                onDoubleClick={() => activeTool === 'polygon' && polygonPoints.length >= 3 && completePolygon()}
              />
              {activeTool === 'polygon' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-amber-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  Polygon mode — click to add points, double-click or Enter to finish
                </div>
              )}
              {activeTool === 'keypoint' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-purple-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  Keypoint mode — click to place points ({keypointList.length} placed). Press Enter to commit.
                </div>
              )}
              {activeTool === 'brush' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-blue-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  Brush mode — paint mask. Release mouse to commit polygon.
                </div>
              )}
              {activeTool === 'sam' && (
                <div className="absolute top-4 left-1/2 -translate-x-1/2 bg-green-500/90 text-white text-xs px-3 py-1.5 rounded-full font-medium shadow pointer-events-none">
                  SAM mode — click objects to segment
                  {samPendingClicks.length > 0 && ` · ${samPendingClicks.length} in queue`}
                </div>
              )}
              {keypointList.length > 0 && activeTool === 'keypoint' && (
                <div className="absolute bottom-16 right-2 flex gap-2">
                  <Button size="sm" onClick={commitKeypoints} className="bg-purple-600 hover:bg-purple-700 text-white">
                    Commit {keypointList.length} keypoints
                  </Button>
                  <Button size="sm" variant="outline" onClick={() => setKeypointList([])}>Clear</Button>
                </div>
              )}
            </>
          ) : (
            <div className="text-center space-y-2">
              <p className="text-muted-foreground text-sm">No image loaded</p>
              {filteredImages.length > 0 && (
                <button
                  onClick={() => loadImageFile(filteredImages[0])}
                  className="text-xs text-primary underline underline-offset-2"
                >
                  Load first image
                </button>
              )}
            </div>
          )}
          </div>{/* end inner centering wrapper */}
        </div>

        {/* Bottom bar */}
        <div className="flex items-center justify-between px-3 py-2 border-t border-border bg-card gap-3 shrink-0">
          <div className="flex items-center gap-2">
            <Button size="sm" variant="destructive" onClick={deleteAnnotation} disabled={selectedAnnotation === null}
              className="h-7 text-xs gap-1">
              <Trash2 className="w-3 h-3" strokeWidth={1.5} /> Delete
            </Button>
            <span className="text-[10px] text-muted-foreground/60 hidden lg:flex items-center gap-1.5">
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">S</kbd> Save+Next
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">Del</kbd> Delete
              <kbd className="px-1 py-0.5 bg-muted rounded font-mono text-[9px]">1–9</kbd> Class
            </span>
          </div>
          <span className="text-[10px] text-muted-foreground/60 font-mono truncate max-w-[180px]">
            {currentImage?.filename}
          </span>
          <div className="flex items-center gap-1.5">
            <Button size="sm" variant="outline" onClick={() => saveAnnotations(false)} disabled={isSaving}
              className="h-7 text-xs gap-1 border-border">
              {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" strokeWidth={1.5} />}
              Save
            </Button>
            <Button size="sm" onClick={() => saveAnnotations(true)}
              disabled={isSaving || currentIndex >= filteredImages.length - 1}
              className="h-7 text-xs gap-1">
              {isSaving ? <Loader2 className="w-3 h-3 animate-spin" /> : <Save className="w-3 h-3" strokeWidth={1.5} />}
              Save & Next
            </Button>
          </div>
        </div>
        </>
      </div>

      {/* ── Right panel ─────────────────────────────────────────────────── */}
      <div className="w-64 border-l border-border bg-card flex flex-col shrink-0">

        {/* Tab switcher */}
        <div className="flex border-b border-border shrink-0">
          <button
            onClick={() => setSidebarTab('annotate')}
            className={cn(
              'flex-1 py-2.5 text-[10px] font-mono font-semibold uppercase tracking-wider transition-colors',
              sidebarTab === 'annotate'
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-muted-foreground/60 hover:text-foreground'
            )}
          >
            Annotate
          </button>
          <button
            onClick={() => setSidebarTab('jobs')}
            className={cn(
              'flex-1 py-2.5 text-[10px] font-mono font-semibold uppercase tracking-wider transition-colors flex items-center justify-center gap-1.5',
              sidebarTab === 'jobs'
                ? 'text-primary border-b-2 border-primary bg-primary/5'
                : 'text-muted-foreground/60 hover:text-foreground'
            )}
          >
            {runningJobCount > 0 && <span className="live-dot" />}
            Jobs
            {totalJobCount > 0 && (
              <span className={cn(
                'px-1.5 py-px rounded font-mono text-[9px] leading-none',
                runningJobCount > 0 ? 'bg-primary/15 text-primary' : 'bg-muted text-muted-foreground'
              )}>
                {totalJobCount}
              </span>
            )}
          </button>
        </div>

        {/* ── Annotate tab ─────────────────────────────────────────────── */}
        {sidebarTab === 'annotate' && <>

        {/* Model + confidence */}
        <div className="p-3 space-y-3 border-b border-border">
          <Select
            value={selectedModel}
            onValueChange={v => {
              const m = models.find(m => m.id === v)
              if (m && !m.downloaded && downloadingModelId !== m.id) downloadModel(m)
              else if (m?.downloaded) setSelectedModel(v)
            }}
          >
            <SelectTrigger className="h-8 text-xs bg-background border-border">
              {downloadingModelId
                ? <span className="flex items-center gap-1.5 text-muted-foreground">
                    <Loader2 className="w-3 h-3 animate-spin" />
                    Downloading…
                  </span>
                : <SelectValue placeholder="Select model…" />
              }
            </SelectTrigger>
            <SelectContent>
              {models.map(m => (
                <SelectItem key={m.id} value={m.id} disabled={!m.downloaded && downloadingModelId !== m.id && downloadingModelId !== null}>
                  <div className="flex items-center gap-2 w-full">
                    <span className="flex-1 truncate">{m.name}</span>
                    {downloadingModelId === m.id
                      ? <span className="text-[9px] text-primary font-mono shrink-0 flex items-center gap-1"><Loader2 className="w-3 h-3 animate-spin" />Downloading</span>
                      : m.downloaded
                        ? <CheckCircle2 className="w-3 h-3 text-primary shrink-0" />
                        : <Download className="w-3 h-3 text-muted-foreground shrink-0" />
                    }
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <div>
            <div className="flex items-center justify-between mb-1.5">
              <label className="text-[10px] font-medium uppercase tracking-wider text-muted-foreground">Confidence</label>
              <span className="text-xs font-mono text-primary">{Math.round(confidence * 100)}%</span>
            </div>
            <Slider value={[confidence]} onValueChange={([v]) => setConfidence(v)} min={0.1} max={1} step={0.05} />
          </div>

          <Button
            className="w-full h-8 text-xs gap-1.5 font-medium"
            onClick={autoAnnotate}
            disabled={!selectedModel || isAutoAnnotating || runningJobCount > 0}
          >
            {isModelLoading
              ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Loading model…</>
              : isAutoAnnotating
                ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Annotating…</>
                : <><Wand2 className="w-3.5 h-3.5" strokeWidth={1.5} /> Auto-Annotate</>
            }
          </Button>
        </div>

        {/* Fixed-class batch annotate — YOLO, RT-DETR, RF-DETR (no text prompt needed) */}
        {(() => {
          const selModel = models.find(m => m.id === selectedModel)
          const isFixedClass = selModel && !['sam3', 'yoloworld', 'groundingdino', 'owlvit'].includes(selModel.type)
          if (!isFixedClass || !selModel) return null
          return (
            <div className="p-3 space-y-2.5 border-b border-border">
              <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                Batch Annotate
              </p>
              <Button size="sm"
                className="w-full h-8 text-xs gap-1.5 font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={textAnnotateBatch}
                disabled={isAutoAnnotating || isStartingBatch}>
                {isStartingBatch
                  ? <><Loader2 className="w-3 h-3 animate-spin" /> Starting…</>
                  : <><ChevronsRight className="w-3.5 h-3.5" /> Batch Annotate Dataset</>
                }
              </Button>
            </div>
          )
        })()}

        {/* Text prompt section — shown for SAM3, YOLO-World, and GroundingDINO */}
        {(() => {
          const selModel = models.find(m => m.id === selectedModel)
          const isTextModel = selModel && ['sam3', 'yoloworld', 'groundingdino', 'owlvit'].includes(selModel.type)
          if (!isTextModel) return null
          const sectionLabel = selModel.type === 'sam3' ? 'Text Segment' : 'Text Detect (Zero-Shot)'
          return (
            <div className="p-3 space-y-2.5 border-b border-border">
              {/* Section label */}
              <div className="flex items-center justify-between">
                <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-primary/70">
                  {sectionLabel}
                </p>
                {totalJobCount > 0 && (
                  <button onClick={() => setSidebarTab('jobs')}
                    className="text-[9px] font-mono text-muted-foreground hover:text-primary flex items-center gap-1.5 transition-colors">
                    {runningJobCount > 0 && <span className="live-dot text-primary" />}
                    {totalJobCount} job{totalJobCount !== 1 ? 's' : ''}
                  </button>
                )}
              </div>

              {/* Tag chips */}
              {promptTags.length > 0 && (
                <div className="flex flex-wrap gap-1">
                  {promptTags.map(tag => (
                    <span key={tag}
                      className="flex items-center gap-0.5 pl-2 pr-1 py-0.5 bg-primary/10 text-primary border border-primary/20 rounded-full text-[10px] font-medium">
                      {tag}
                      <button onClick={() => removePromptTag(tag)}
                        className="hover:text-primary/60 transition-colors ml-0.5">
                        <X className="w-2.5 h-2.5" />
                      </button>
                    </span>
                  ))}
                  <button onClick={() => setPromptTags([])}
                    className="text-[10px] text-muted-foreground/60 hover:text-muted-foreground px-1 transition-colors">
                    clear
                  </button>
                </div>
              )}

              {/* Label input */}
              <div className="flex gap-1">
                <Input
                  value={promptInput}
                  onChange={e => setPromptInput(e.target.value)}
                  onKeyDown={e => {
                    if ((e.key === 'Enter' || e.key === ',') && promptInput.trim()) {
                      e.preventDefault(); addPromptTag(promptInput)
                    }
                  }}
                  placeholder={promptTags.length ? 'Add label…' : 'car, person, dog…'}
                  className="h-7 text-xs flex-1 bg-background border-border"
                />
                <Button size="sm" variant="outline"
                  className="h-7 w-7 p-0 shrink-0 border-border"
                  onClick={() => promptInput.trim() && addPromptTag(promptInput)}
                  disabled={!promptInput.trim()}>
                  <Plus className="w-3 h-3" />
                </Button>
              </div>
              <p className="text-[10px] text-muted-foreground leading-relaxed">
                Enter or <kbd className="px-1 bg-muted rounded text-[9px]">,</kbd> adds a label.
              </p>

              {/* Annotate this image */}
              <Button size="sm" variant="outline"
                className="w-full h-7 text-xs gap-1.5 border-border hover:border-primary/40 hover:text-primary"
                onClick={textAnnotateSingle}
                disabled={promptTags.length === 0 || isAutoAnnotating}>
                {isAutoAnnotating
                  ? <Loader2 className="w-3 h-3 animate-spin" />
                  : <Wand2 className="w-3 h-3" strokeWidth={1.5} />
                }
                Annotate This Image
              </Button>

              {/* Batch annotate */}
              <Button size="sm"
                className="w-full h-8 text-xs gap-1.5 font-semibold bg-primary text-primary-foreground hover:bg-primary/90"
                onClick={textAnnotateBatch}
                disabled={promptTags.length === 0 || isAutoAnnotating || isStartingBatch}>
                {isStartingBatch
                  ? <><Loader2 className="w-3 h-3 animate-spin" /> Starting…</>
                  : <><ChevronsRight className="w-3.5 h-3.5" /> Batch Annotate Dataset</>
                }
              </Button>
            </div>
          )
        })()}

        {/* Annotations list */}
        <div className="flex-1 overflow-hidden flex flex-col min-h-0">
          <div className="px-4 py-2.5 flex items-center justify-between border-b border-border shrink-0">
            <p className="text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground/60">
              Annotations
            </p>
            <div className="flex items-center gap-2">
              <span className="text-[10px] font-mono text-primary">{annotations.length}</span>
              {selectedDataset && (
                <button
                  onClick={deleteAllAnnotations}
                  className="text-[9px] text-destructive/60 hover:text-destructive transition-colors"
                  title="Delete all annotations from dataset"
                >
                  clear all
                </button>
              )}
            </div>
          </div>
          <div className="flex-1 overflow-y-auto p-2">
            {annotations.length === 0 ? (
              <div className="py-6 text-center">
                <p className="text-xs text-muted-foreground/60">No annotations yet.</p>
                <p className="text-[10px] text-muted-foreground/40 mt-0.5">Use the tools above to add some.</p>
              </div>
            ) : (
              annotations.map((ann, idx) => (
                <div key={idx}
                  onClick={() => setSelectedAnnotation(idx === selectedAnnotation ? null : idx)}
                  className={cn(
                    'px-2.5 py-1.5 rounded-md mb-0.5 cursor-pointer transition-all text-xs',
                    'border',
                    selectedAnnotation === idx
                      ? 'bg-primary/10 border-primary/30 text-primary'
                      : 'border-transparent hover:bg-muted hover:border-border text-foreground'
                  )}>
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-medium truncate">{ann.class_name}</span>
                    <span className="text-[10px] text-muted-foreground capitalize shrink-0">{ann.type}</span>
                  </div>
                  {selectedAnnotation === idx && (
                    <div className="mt-1.5 flex items-center gap-1" onClick={e => e.stopPropagation()}>
                      <select
                        value={ann.class_name}
                        onChange={e => {
                          const cls = e.target.value
                          const clsIdx = localClasses.indexOf(cls)
                          const next = annotations.map((a, i) =>
                            i === idx ? { ...a, class_name: cls, class_id: clsIdx } : a
                          )
                          setAnnotations(next); saveToHistory(next); setActiveClass(cls)
                        }}
                        className="flex-1 h-6 text-[10px] rounded border border-border bg-background text-foreground px-1"
                      >
                        {localClasses.map(c => (
                          <option key={c} value={c}>{c}</option>
                        ))}
                      </select>
                      <button
                        onClick={() => {
                          const next = annotations.filter((_, i) => i !== idx)
                          setAnnotations(next); saveToHistory(next); setSelectedAnnotation(null)
                        }}
                        className="p-1 rounded hover:bg-destructive/10 hover:text-destructive transition-colors"
                        title="Delete annotation"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>

        {/* Copy forward */}
        <div className="border-t border-border p-3 space-y-2 shrink-0">
          <button
            onClick={() => setShowBatchPanel(b => !b)}
            className="w-full flex items-center justify-between text-[9px] font-mono font-semibold uppercase tracking-[0.18em] text-muted-foreground/50 hover:text-foreground transition-colors"
          >
            <span>Copy Forward</span>
            <span className="text-[8px]">{showBatchPanel ? '▲' : '▼'}</span>
          </button>
          {showBatchPanel && (
            <div className="space-y-2">
              <div className="flex items-center gap-2">
                <Input type="number" value={copyToNext}
                  onChange={e => setCopyToNext(Math.max(1, +e.target.value))}
                  min={1} max={100}
                  className="h-6 w-14 text-xs bg-background border-border font-mono" />
                <span className="text-xs text-muted-foreground">images</span>
              </div>
              <Button size="sm" variant="outline"
                className="w-full h-7 text-xs gap-1 border-border"
                onClick={() => copyAnnotationsToNext(copyToNext)}
                disabled={annotations.length === 0}>
                <ChevronsRight className="w-3 h-3" strokeWidth={1.5} />
                Propagate Annotations
              </Button>
            </div>
          )}
        </div>

        </> /* end sidebarTab === 'annotate' */}

        {/* ── Jobs tab ─────────────────────────────────────────────────── */}
        {sidebarTab === 'jobs' && (
          <div className="flex-1 overflow-y-auto flex flex-col min-h-0">
            {totalJobCount === 0 ? (
              <div className="flex flex-col items-center justify-center flex-1 gap-2 text-muted-foreground p-4">
                <p className="text-[10px] font-mono uppercase tracking-widest opacity-30">no jobs</p>
                <p className="text-xs text-center">Start a batch job from the Annotate tab</p>
                <button
                  onClick={() => setSidebarTab('annotate')}
                  className="text-xs text-primary underline underline-offset-2 mt-1"
                >
                  Go to Annotate
                </button>
              </div>
            ) : (
              <div className="p-2 space-y-2">
                {/* Summary row */}
                <div className="flex items-center gap-2 px-1 py-1.5 text-[9px] font-mono text-muted-foreground/50">
                  <span>{totalJobCount} job{totalJobCount !== 1 ? 's' : ''}</span>
                  {runningJobCount > 0 && (
                    <span className="flex items-center gap-1 text-primary">
                      <span className="live-dot" /> {runningJobCount} live
                    </span>
                  )}
                </div>
                {datasetJobs.slice().reverse().map(job => {
                  const processed = job.processed ?? 0
                  const isActive = job.status === 'running' || job.status === 'paused'
                  const isRunning = job.status === 'running'
                  return (
                    <div key={job.job_id}
                      className={cn(
                        'rounded-xl bg-background overflow-hidden',
                        isRunning ? 'gradient-border-active' : 'border border-border'
                      )}>

                      {/* Card header */}
                      <div className="flex items-start gap-2 px-3 pt-3 pb-2">
                        <div className="flex-1 min-w-0">
                          <div className="flex flex-wrap gap-1 mb-1.5">
                            {job.text_prompt.split(',').map(t => t.trim()).filter(Boolean).map(tag => (
                              <span key={tag}
                                className="px-1.5 py-px bg-primary/8 text-primary border border-primary/15 rounded-full text-[9px] font-semibold">
                                {tag}
                              </span>
                            ))}
                          </div>
                          <p className="text-[8px] text-muted-foreground/40 font-mono">
                            {job.job_id} · {fmtElapsed(job.started_at)}
                          </p>
                        </div>
                        <span className={cn(
                          'shrink-0 flex items-center gap-1 text-[8px] px-1.5 py-0.5 rounded-full font-mono uppercase tracking-wider',
                          isRunning                      && 'bg-primary/8 text-primary',
                          job.status === 'paused'        && 'bg-warning/10 text-warning',
                          job.status === 'done'          && 'bg-success/10 text-success',
                          job.status === 'error'         && 'bg-destructive/10 text-destructive',
                          job.status === 'undone'        && 'bg-muted text-muted-foreground',
                          (job.status === 'cancelled' || job.status === 'interrupted') && 'bg-muted text-muted-foreground',
                        )}>
                          {isRunning && <span className="live-dot" />}
                          {job.status}
                        </span>
                      </div>

                      {/* Progress */}
                      <div className="px-3 pb-3 space-y-2">
                        <div className="space-y-1">
                          <div className="flex items-center justify-between text-[8px] font-mono text-muted-foreground/50">
                            <span>{processed} / {job.total || '?'}</span>
                            <span>{job.progress}%</span>
                          </div>
                          <div className="relative h-[3px] bg-muted/60 rounded-full overflow-hidden">
                            <div
                              className={cn(
                                'absolute inset-y-0 left-0 rounded-full transition-all duration-700',
                                job.status === 'done'        ? 'bg-success' :
                                job.status === 'error'       ? 'bg-destructive' :
                                job.status === 'cancelled'   ? 'bg-muted-foreground' :
                                job.status === 'interrupted' ? 'bg-muted-foreground' :
                                job.status === 'paused'      ? 'bg-warning' : 'bg-primary'
                              )}
                              style={{ width: `${job.progress}%` }}
                            >
                              {isRunning && <span className="shimmer-overlay" />}
                            </div>
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="grid grid-cols-4 gap-0.5 pt-1 border-t border-border/60">
                          {[
                            { label: 'Done',    value: processed,             color: '' },
                            { label: 'Annot',   value: job.annotated,         color: 'text-success' },
                            { label: 'Labels',  value: job.total_annotations, color: 'text-primary' },
                            { label: 'Fail',    value: job.failed,            color: job.failed > 0 ? 'text-destructive' : 'text-muted-foreground/30' },
                          ].map(({ label, value, color }) => (
                            <div key={label} className="text-center pt-1.5">
                              <div className={cn('text-xs font-mono font-bold tabular-nums leading-none', color || 'text-foreground')}>
                                {value ?? '–'}
                              </div>
                              <div className="text-[7px] font-mono text-muted-foreground/40 uppercase tracking-widest mt-1">{label}</div>
                            </div>
                          ))}
                        </div>

                        {/* Recent images preview */}
                        {(job.recent_images?.length ?? 0) > 0 && (
                          <div className="pt-1 border-t border-border/60">
                            <div className="flex items-center justify-between mb-1.5">
                              <p className="text-[8px] font-mono text-muted-foreground/40 uppercase tracking-wider">
                                Recent · {job.recent_images!.length} annotated
                              </p>
                              {job.status === 'done' && job.dataset_id === selectedDataset?.id && (
                                <button
                                  onClick={() => {
                                    const first = job.recent_images![job.recent_images!.length - 1]
                                    if (first) navigateToAnnotatedImage(first.image_id)
                                  }}
                                  className="text-[8px] font-mono text-primary hover:underline"
                                >
                                  View Annotations →
                                </button>
                              )}
                              {job.status === 'done' && job.dataset_id !== selectedDataset?.id && (
                                <span className="text-[8px] font-mono text-muted-foreground/40">
                                  Switch to dataset to view
                                </span>
                              )}
                            </div>
                            <div className="flex gap-1 overflow-x-auto pb-1 scrollbar-thin">
                              {job.recent_images!.slice().reverse().map((img, idx) => (
                                <button
                                  key={idx}
                                  className="shrink-0 relative group cursor-pointer"
                                  disabled={job.dataset_id !== selectedDataset?.id}
                                  onClick={() => navigateToAnnotatedImage(img.image_id)}
                                  title={`${img.filename} · ${img.annotation_count} annotations`}
                                >
                                  <img
                                    src={`${apiUrl}/api/auto-annotate/text-batch/${job.job_id}/preview/${job.recent_images!.length - 1 - idx}`}
                                    alt={img.filename}
                                    className="w-14 h-14 object-cover rounded border border-border group-hover:border-primary/50 transition-colors"
                                    loading="lazy"
                                  />
                                  <div className="absolute bottom-0 left-0 right-0 bg-black/60 text-[7px] text-white font-mono text-center py-0.5 rounded-b opacity-0 group-hover:opacity-100 transition-opacity">
                                    {img.annotation_count} ann
                                  </div>
                                </button>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* Controls */}
                        {isActive && (
                          <div className="flex items-center gap-1.5 pt-0.5">
                            {isRunning ? (
                              <button onClick={() => pauseJob(job.job_id)}
                                className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-warning/20 bg-warning/5 text-warning hover:bg-warning/10 transition-colors">
                                <Pause className="w-2 h-2" strokeWidth={2} /> Pause
                              </button>
                            ) : (
                              <button onClick={() => resumeJob(job.job_id)}
                                className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-primary/20 bg-primary/5 text-primary hover:bg-primary/10 transition-colors">
                                <Play className="w-2 h-2" strokeWidth={2} /> Resume
                              </button>
                            )}
                            <button onClick={() => cancelJob(job.job_id)}
                              className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-destructive/20 bg-destructive/5 text-destructive hover:bg-destructive/10 transition-colors">
                              <StopCircle className="w-2 h-2" strokeWidth={2} /> Cancel
                            </button>
                          </div>
                        )}

                        {job.status === 'done' && (
                          <div className="flex items-center gap-1.5 pt-0.5 border-t border-border/60">
                            <button
                              onClick={() => undoJob(job.job_id)}
                              disabled={job.dataset_id !== selectedDataset?.id}
                              className="flex items-center gap-1 text-[9px] font-mono px-2 py-1 rounded border border-destructive/20 bg-destructive/5 text-destructive hover:bg-destructive/10 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
                              title={job.dataset_id !== selectedDataset?.id ? 'Switch to this dataset to undo' : 'Remove all annotations this job created'}
                            >
                              ↩ Undo Batch
                            </button>
                          </div>
                        )}

                        {job.status === 'error' && job.error && (
                          <div className="px-2 py-1.5 rounded bg-destructive/5 border border-destructive/15 text-[9px] font-mono text-destructive">
                            {job.error}
                          </div>
                        )}
                        {job.status === 'interrupted' && (
                          <div className="px-2 py-1.5 rounded bg-muted border border-border text-[9px] font-mono text-muted-foreground">
                            Server restarted — job interrupted at {job.progress}%
                          </div>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

      </div>

      {error && (
        <div className="absolute bottom-4 left-4 right-4 p-3 bg-destructive/10 border border-destructive/20 rounded-lg flex items-center gap-2 text-destructive text-sm">
          <AlertCircle className="w-4 h-4" />{error}
          <button onClick={() => setError(null)} className="ml-auto text-xs underline">Dismiss</button>
        </div>
      )}
      </div>
    </div>
  )
}