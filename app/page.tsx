'use client'

import { useState, useCallback, useEffect } from 'react'
import { Loader2, AlertTriangle, X, Database, Menu } from 'lucide-react'
import { Sidebar } from '@/components/sidebar'
import { DatasetsView } from '@/components/datasets-view'
import { DashboardView } from '@/components/dashboard-view'
import { GalleryView } from '@/components/gallery-view'
import { SortingView } from '@/components/sorting-view'
import { AnnotationView } from '@/components/annotation-view'
import { ConvertView } from '@/components/convert-view'
import { MergeView } from '@/components/merge-view'
import { TrainingView } from '@/components/training-view'
import { InferenceView } from '@/components/inference-view'
import { EvaluationView } from '@/components/evaluation-view'
import { ModelsView } from '@/components/models-view'
import { SettingsView } from '@/components/settings-view'
import { ClassManagementView } from '@/components/class-management-view'
import { AugmentationView } from '@/components/augmentation-view'
import { VideoExtractionView } from '@/components/video-extraction-view'
import { SplitView } from '@/components/split-view'
import { HealthView } from '@/components/health-view'
import { CompareView } from '@/components/compare-view'
import { SnapshotView } from '@/components/snapshot-view'
import { YamlWizardView } from '@/components/yaml-wizard-view'
import { DuplicateDetectionView } from '@/components/duplicate-detection-view'
import { BatchJobsView } from '@/components/batch-jobs-view'
import { useSettings } from '@/lib/settings-context'

export type ViewType = 'datasets' | 'dashboard' | 'gallery' | 'sorting' | 'annotate' | 'classes' | 'augmentation' | 'video-extraction' | 'split' | 'convert' | 'merge' | 'training' | 'inference' | 'evaluate' | 'models' | 'batch-jobs' | 'health' | 'compare' | 'snapshots' | 'yaml-wizard' | 'settings' | 'duplicate-detection'

export interface Dataset {
  id: string
  name: string
  path: string
  format: string
  task_type: string
  num_images: number
  num_annotations: number
  classes: string[]
  created_at: string
}

export interface ImageData {
  id: string
  filename: string
  path: string
  width?: number
  height?: number
  split?: string
  class_name?: string
  annotations: Annotation[]
  has_annotations: boolean
}

export interface Annotation {
  id?: string
  type: string
  class_id: number
  class_name: string
  bbox?: number[]
  x_center?: number
  y_center?: number
  width?: number
  height?: number
  points?: number[]
  normalized?: boolean
}

// Shared image cache so tab-switches don't reload
export type ImageCache = Record<string, ImageData[]>

// All valid view IDs — used to validate URL path segments
const VALID_VIEWS = new Set<ViewType>([
  'datasets', 'dashboard', 'gallery', 'sorting', 'annotate', 'classes',
  'augmentation', 'video-extraction', 'split', 'convert', 'merge',
  'training', 'inference', 'evaluate', 'models', 'batch-jobs', 'health',
  'compare', 'snapshots', 'yaml-wizard', 'settings', 'duplicate-detection',
])

// Views that require a dataset to be selected before they're useful.
// Note: 'video-extraction' is intentionally excluded — it creates a new
// dataset and should be accessible without one already loaded.
const DATASET_REQUIRED_VIEWS = new Set<ViewType>([
  'dashboard', 'gallery', 'sorting', 'annotate', 'classes',
  'augmentation', 'split', 'health',
  'snapshots', 'yaml-wizard', 'duplicate-detection',
])

// Inline dataset picker shown when a view needs a dataset but none is selected
function DatasetRequiredGuard({
  datasets,
  selectedDataset,
  onSelectDataset,
  children,
}: {
  datasets: Dataset[]
  selectedDataset: Dataset | null
  onSelectDataset: (d: Dataset) => void
  children: React.ReactNode
}) {
  if (selectedDataset) return <>{children}</>

  return (
    <div className="flex flex-col items-center justify-center h-full gap-6 p-8">
      <div className="flex flex-col items-center gap-2 text-center">
        <Database className="w-10 h-10 text-muted-foreground/40 mb-1" />
        <h2 className="text-base font-semibold text-foreground">No dataset selected</h2>
        <p className="text-sm text-muted-foreground max-w-xs">
          {datasets.length === 0
            ? 'No datasets loaded yet. Go to Datasets to upload images, extract video frames, or load a formatted dataset.'
            : 'Pick a dataset below to continue.'}
        </p>
      </div>
      {datasets.length > 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 w-full max-w-2xl">
          {datasets.map((d) => (
            <button
              key={d.id}
              onClick={() => onSelectDataset(d)}
              className="text-left p-3.5 rounded-xl border border-border bg-card hover:bg-accent hover:border-primary/40 transition-all duration-150 group"
            >
              <p className="text-sm font-semibold text-foreground truncate group-hover:text-primary transition-colors">
                {d.name}
              </p>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="text-[10px] px-1.5 py-0.5 bg-muted text-muted-foreground rounded font-mono">
                  {d.format?.toUpperCase()}
                </span>
                <span className="text-[11px] text-muted-foreground">
                  {d.num_images?.toLocaleString()} images
                </span>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default function Home() {
  const [activeView, setActiveViewState] = useState<ViewType>('datasets')
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)
  const [imageCache, setImageCache] = useState<ImageCache>({})
  const [annotateInitialImageId, setAnnotateInitialImageId] = useState<string | null>(null)
  const [gpuStatus, setGpuStatus] = useState<{ state: string; message: string } | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)

  // Sync activeView from URL after hydration, and on back/forward navigation
  useEffect(() => {
    const syncFromUrl = () => {
      const segment = window.location.pathname.replace(/^\//, '') as ViewType
      if (VALID_VIEWS.has(segment)) {
        setActiveViewState(segment)
      }
    }
    syncFromUrl()
    window.addEventListener('popstate', syncFromUrl)
    return () => window.removeEventListener('popstate', syncFromUrl)
  }, [])

  // Navigate to a view — updates React state AND the browser URL without remounting
  const setActiveView = useCallback((view: ViewType) => {
    setActiveViewState(view)
    window.history.pushState(null, '', `/${view}`)
  }, [])

  // apiUrl now comes from the shared settings context
  const { settings } = useSettings()
  const apiUrl = settings.apiUrl

  // Poll GPU/CUDA install status every 4 s — hide banner once state is "ready"
  useEffect(() => {
    let timer: ReturnType<typeof setTimeout>
    const check = async () => {
      try {
        const r = await fetch(`${apiUrl}/api/device-info`)
        if (r.ok) {
          const d = await r.json()
          const s = d.gpu_status
          if (s && s.state !== 'ready' && s.state !== 'no_gpu' && s.state !== 'unknown') {
            setGpuStatus(s)
          } else {
            setGpuStatus(null)
          }
          // Stop polling once ready or no GPU
          if (s?.state === 'ready' || s?.state === 'no_gpu') return
        }
      } catch { /* backend may be restarting */ }
      timer = setTimeout(check, 4000)
    }
    timer = setTimeout(check, 3000)
    return () => clearTimeout(timer)
  }, [apiUrl])

  const updateImageCache = useCallback((datasetId: string, images: ImageData[]) => {
    setImageCache(prev => ({ ...prev, [datasetId]: images }))
  }, [])

  const invalidateImageCache = useCallback((datasetId: string) => {
    setImageCache(prev => {
      const next = { ...prev }
      delete next[datasetId]
      return next
    })
  }, [])

  const handleSelectDataset = (dataset: Dataset) => {
    setSelectedDataset(dataset)
  }

  const renderView = () => {
    let viewNode: React.ReactNode = null
    switch (activeView) {
      case 'datasets':
        viewNode = (
          <DatasetsView
            datasets={datasets}
            setDatasets={setDatasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            onDatasetLoaded={invalidateImageCache}
            onNavigateToVideo={() => setActiveView('video-extraction')}
            apiUrl={apiUrl}
          />
        )
        break
      case 'dashboard':
        viewNode = (
          <DashboardView
            datasets={datasets}
            selectedDataset={selectedDataset}
            onSelectDataset={handleSelectDataset}
            setActiveView={setActiveView}
            apiUrl={apiUrl}
          />
        )
        break
      case 'gallery':
        viewNode = (
          <GalleryView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
            onOpenAnnotator={(imageId: string) => { setAnnotateInitialImageId(imageId); setActiveView('annotate') }}
          />
        )
        break
      case 'health':
        viewNode = (
          <HealthView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
          />
        )
        break
      case 'compare':
        viewNode = (
          <CompareView
            datasets={datasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'snapshots':
        viewNode = (
          <SnapshotView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'sorting':
        viewNode = (
          <SortingView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
          />
        )
        break
      case 'duplicate-detection':
        viewNode = (
          <DuplicateDetectionView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'annotate':
        viewNode = (
          <AnnotationView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
            imageCache={imageCache}
            updateImageCache={updateImageCache}
            initialImageId={annotateInitialImageId}
            onInitialImageConsumed={() => setAnnotateInitialImageId(null)}
          />
        )
        break
      case 'classes':
        viewNode = (
          <ClassManagementView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'augmentation':
        viewNode = (
          <AugmentationView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={setSelectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'video-extraction':
        viewNode = (
          <VideoExtractionView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={(dataset) => {
              setSelectedDataset(dataset)
              invalidateImageCache(dataset.id)
            }}
            apiUrl={apiUrl}
          />
        )
        break
      case 'split':
        viewNode = (
          <SplitView
            selectedDataset={selectedDataset}
            datasets={datasets}
            setDatasets={setDatasets}
            onDatasetCreated={setSelectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'convert':
        viewNode = (
          <ConvertView
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'merge':
        viewNode = (
          <MergeView
            datasets={datasets}
            setDatasets={setDatasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'training':
        viewNode = (
          <TrainingView
            datasets={datasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'inference':
        viewNode = (
          <InferenceView
            apiUrl={apiUrl}
          />
        )
        break
      case 'evaluate':
        viewNode = (
          <EvaluationView
            datasets={datasets}
            apiUrl={apiUrl}
          />
        )
        break
      case 'models':
        viewNode = (
          <ModelsView
            apiUrl={apiUrl}
          />
        )
        break
      case 'batch-jobs':
        viewNode = (
          <BatchJobsView
            datasets={datasets}
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'yaml-wizard':
        viewNode = (
          <YamlWizardView
            selectedDataset={selectedDataset}
            apiUrl={apiUrl}
          />
        )
        break
      case 'settings':
        return <SettingsView />
      default:
        return null
    }

    // Wrap views that require a dataset with the inline dataset picker
    if (DATASET_REQUIRED_VIEWS.has(activeView)) {
      return (
        <DatasetRequiredGuard
          datasets={datasets}
          selectedDataset={selectedDataset}
          onSelectDataset={handleSelectDataset}
        >
          {viewNode}
        </DatasetRequiredGuard>
      )
    }
    return viewNode
  }

  return (
    <div className="flex h-screen bg-background flex-col overflow-hidden">
      {/* GPU status banner */}
      {gpuStatus && (
        <div className={`flex items-center gap-2.5 px-4 py-2 text-xs font-medium z-50 shrink-0 ${
          gpuStatus.state === 'failed'
            ? 'bg-destructive/90 text-destructive-foreground'
            : 'bg-warning/90 text-warning-foreground'
        }`}>
          {gpuStatus.state === 'installing' ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
          ) : (
            <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          )}
          <span className="flex-1 truncate">{gpuStatus.message}</span>
          {gpuStatus.state === 'failed' && (
            <button
              className="shrink-0 opacity-70 hover:opacity-100 transition-opacity"
              onClick={() => setGpuStatus(null)}
              aria-label="Dismiss"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      )}

      {/* Mobile top bar — hidden on md+ */}
      <div className="md:hidden flex items-center gap-3 px-4 py-3 border-b border-border bg-sidebar shrink-0">
        <button
          onClick={() => setSidebarOpen(true)}
          className="p-1.5 rounded-lg text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <span className="text-sm font-semibold text-foreground capitalize">
          {activeView.replace(/-/g, ' ')}
        </span>
        {selectedDataset && (
          <span className="ml-auto text-[10px] font-mono text-primary truncate max-w-[140px]">
            {selectedDataset.name}
          </span>
        )}
      </div>

      <div className="flex flex-1 min-h-0 overflow-hidden">
        <Sidebar
          activeView={activeView}
          setActiveView={setActiveView}
          selectedDataset={selectedDataset}
          isOpen={sidebarOpen}
          onClose={() => setSidebarOpen(false)}
        />
        <main className="flex-1 overflow-y-auto min-w-0 bg-background">
          {renderView()}
        </main>
      </div>
    </div>
  )
}
