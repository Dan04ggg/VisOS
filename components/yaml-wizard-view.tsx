'use client'

import { useState, useEffect, useRef, useCallback } from 'react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import {
  FileCode2, ChevronRight, ChevronLeft, CheckCircle2, AlertCircle,
  Copy, Download, RefreshCw, Eye, Pencil, ArrowRight, Layers,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'
import { toast } from 'sonner'

// ── Types ─────────────────────────────────────────────────────────────────────

interface SampleAnnotation {
  type: 'bbox' | 'polygon'
  class_id: number
  x_center?: number; y_center?: number; width?: number; height?: number
  points?: number[]
  normalized?: boolean
}

interface ClassSample {
  image_id: string
  image_path: string
  annotations: SampleAnnotation[]
}

interface ClassEntry {
  class_id: number
  existing_name: string | null
  samples: ClassSample[]
}

interface WizardData {
  classes: ClassEntry[]
  splits: { train: string | null; val: string | null; test: string | null }
  dataset_path: string
  existing_names: string[]
}

type Phase = 'loading' | 'labeling' | 'verifying' | 'done'

// ── Annotation canvas ─────────────────────────────────────────────────────────

const COLORS = [
  '#ef4444','#f97316','#eab308','#22c55e','#06b6d4',
  '#6366f1','#ec4899','#14b8a6','#f59e0b','#8b5cf6',
]

function AnnotationCanvas({
  apiUrl, datasetId, imagePath, annotations, classNames, highlightClassId,
}: {
  apiUrl: string
  datasetId: string
  imagePath: string
  annotations: SampleAnnotation[]
  classNames: Record<number, string>
  highlightClassId?: number
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [imgSize, setImgSize] = useState<{ w: number; h: number } | null>(null)

  const imgUrl = `${apiUrl}/api/datasets/${datasetId}/image-file/${imagePath}`

  const draw = useCallback((img: HTMLImageElement) => {
    const canvas = canvasRef.current
    if (!canvas) return
    const cw = canvas.width
    const ch = canvas.height
    const scaleX = cw / img.naturalWidth
    const scaleY = ch / img.naturalHeight
    const ctx = canvas.getContext('2d')!
    ctx.clearRect(0, 0, cw, ch)
    ctx.drawImage(img, 0, 0, cw, ch)

    for (const ann of annotations) {
      const isHighlight = highlightClassId === undefined || ann.class_id === highlightClassId
      const color = COLORS[ann.class_id % COLORS.length]
      ctx.globalAlpha = isHighlight ? 1 : 0.25

      if (ann.type === 'bbox' && ann.normalized) {
        const iw = img.naturalWidth, ih = img.naturalHeight
        const x = (ann.x_center! - ann.width! / 2) * iw * scaleX
        const y = (ann.y_center! - ann.height! / 2) * ih * scaleY
        const w = ann.width! * iw * scaleX
        const h = ann.height! * ih * scaleY
        ctx.strokeStyle = color
        ctx.lineWidth = isHighlight ? 2.5 : 1.5
        ctx.strokeRect(x, y, w, h)
        if (isHighlight && classNames[ann.class_id]) {
          ctx.fillStyle = color
          ctx.globalAlpha = 0.85
          const label = classNames[ann.class_id]
          ctx.font = 'bold 11px sans-serif'
          const tw = ctx.measureText(label).width
          ctx.fillRect(x, y - 16, tw + 8, 16)
          ctx.fillStyle = '#fff'
          ctx.globalAlpha = 1
          ctx.fillText(label, x + 4, y - 3)
        }
      } else if (ann.type === 'polygon' && ann.points?.length) {
        const iw = img.naturalWidth, ih = img.naturalHeight
        const pts = ann.points
        ctx.beginPath()
        ctx.moveTo(pts[0] * iw * scaleX, pts[1] * ih * scaleY)
        for (let i = 2; i < pts.length - 1; i += 2) {
          ctx.lineTo(pts[i] * iw * scaleX, pts[i + 1] * ih * scaleY)
        }
        ctx.closePath()
        ctx.strokeStyle = color
        ctx.lineWidth = isHighlight ? 2.5 : 1.5
        ctx.stroke()
        ctx.fillStyle = color
        ctx.globalAlpha = isHighlight ? 0.15 : 0.06
        ctx.fill()
        if (isHighlight && classNames[ann.class_id] && pts.length >= 2) {
          ctx.globalAlpha = 0.85
          ctx.fillStyle = color
          const label = classNames[ann.class_id]
          ctx.font = 'bold 11px sans-serif'
          const tw = ctx.measureText(label).width
          ctx.fillRect(pts[0] * iw * scaleX, pts[1] * ih * scaleY - 16, tw + 8, 16)
          ctx.fillStyle = '#fff'
          ctx.globalAlpha = 1
          ctx.fillText(label, pts[0] * iw * scaleX + 4, pts[1] * ih * scaleY - 3)
        }
      }
      ctx.globalAlpha = 1
    }
  }, [annotations, classNames, highlightClassId])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const img = new Image()
    img.onload = () => {
      setImgSize({ w: img.naturalWidth, h: img.naturalHeight })
      draw(img)
    }
    img.src = imgUrl
  }, [imgUrl, draw])

  // Re-draw when annotations/classNames change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !imgSize) return
    const img = new Image()
    img.onload = () => draw(img)
    img.src = imgUrl
  }, [classNames, highlightClassId, draw, imgUrl, imgSize])

  return (
    <canvas
      ref={canvasRef}
      width={320}
      height={240}
      className="w-full rounded-lg border border-border object-contain bg-muted/30"
      style={{ aspectRatio: '4/3', maxHeight: '220px' }}
    />
  )
}

// ── Main wizard ───────────────────────────────────────────────────────────────

export function YamlWizardView({ selectedDataset, apiUrl }: { selectedDataset: Dataset | null; apiUrl: string }) {
  const [phase, setPhase] = useState<Phase>('loading')
  const [wizardData, setWizardData] = useState<WizardData | null>(null)
  const [classNames, setClassNames] = useState<Record<number, string>>({})
  const [currentClassIdx, setCurrentClassIdx] = useState(0)
  const [currentSampleIdx, setCurrentSampleIdx] = useState(0)
  const [splits, setSplits] = useState<{ train: string; val: string; test: string }>({ train: '', val: '', test: '' })
  const [yamlPreview, setYamlPreview] = useState('')
  const [isSaving, setIsSaving] = useState(false)
  const [nameInput, setNameInput] = useState('')

  const load = useCallback(async () => {
    if (!selectedDataset) return
    setPhase('loading')
    try {
      const r = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/class-samples`)
      if (!r.ok) throw new Error('Failed to load samples')
      const data: WizardData = await r.json()
      setWizardData(data)
      setSplits({
        train: data.splits.train ?? '',
        val: data.splits.val ?? '',
        test: data.splits.test ?? '',
      })
      // Pre-fill names from existing YAML
      const pre: Record<number, string> = {}
      data.classes.forEach(c => {
        if (c.existing_name) pre[c.class_id] = c.existing_name
      })
      setClassNames(pre)
      setCurrentClassIdx(0)
      setCurrentSampleIdx(0)
      setNameInput(pre[data.classes[0]?.class_id] ?? '')
      setPhase('labeling')
    } catch (e) {
      toast.error(e instanceof Error ? e.message : 'Failed to load dataset samples')
      setPhase('labeling')
    }
  }, [selectedDataset, apiUrl])

  useEffect(() => { load() }, [load])

  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <div className="text-center space-y-2">
          <FileCode2 className="w-10 h-10 mx-auto opacity-30" />
          <p>Select a dataset to generate its YAML config</p>
        </div>
      </div>
    )
  }

  if (phase === 'loading') {
    return (
      <div className="h-full flex items-center justify-center">
        <RefreshCw className="w-6 h-6 animate-spin text-muted-foreground" />
      </div>
    )
  }

  if (!wizardData || wizardData.classes.length === 0) {
    return (
      <div className="h-full flex items-center justify-center text-muted-foreground">
        <div className="text-center space-y-3">
          <AlertCircle className="w-10 h-10 mx-auto opacity-40" />
          <p className="font-medium">No annotations found in this dataset</p>
          <p className="text-sm">Annotate some images first, then come back to generate the YAML.</p>
          <Button variant="outline" onClick={load}><RefreshCw className="w-4 h-4 mr-2" />Refresh</Button>
        </div>
      </div>
    )
  }

  const currentClassEntry = wizardData.classes[currentClassIdx]
  const allLabeled = wizardData.classes.every(c => !!classNames[c.class_id]?.trim())

  // ── Labeling phase ─────────────────────────────────────────────────────────
  if (phase === 'labeling') {
    const sample = currentClassEntry?.samples[currentSampleIdx]

    const commitLabel = () => {
      const trimmed = nameInput.trim()
      if (!trimmed) return
      const newNames = { ...classNames, [currentClassEntry.class_id]: trimmed }
      setClassNames(newNames)

      const nextIdx = currentClassIdx + 1
      if (nextIdx < wizardData.classes.length) {
        const nextClass = wizardData.classes[nextIdx]
        setCurrentClassIdx(nextIdx)
        setCurrentSampleIdx(0)
        setNameInput(newNames[nextClass.class_id] ?? nextClass.existing_name ?? '')
      } else {
        // Last class — advance to verify if all are named
        const allDone = wizardData.classes.every(c => !!newNames[c.class_id]?.trim())
        if (allDone) setPhase('verifying')
      }
    }

    return (
      <div className="p-6 space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold">YAML Wizard</h1>
            <p className="text-muted-foreground mt-1">Label each class by inspecting sample annotations</p>
          </div>
          <div className="flex items-center gap-3">
            <Badge variant="secondary">
              {Object.values(classNames).filter(Boolean).length} / {wizardData.classes.length} labeled
            </Badge>
            <Button
              disabled={!allLabeled}
              onClick={() => setPhase('verifying')}
            >
              <Eye className="w-4 h-4 mr-2" />
              Review & Save
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </div>
        </div>

        <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
          {/* Class list */}
          <Card className="xl:col-span-1">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Classes ({wizardData.classes.length} unique IDs)</CardTitle>
              <CardDescription className="text-xs">Click any class to label it</CardDescription>
            </CardHeader>
            <CardContent className="space-y-1 max-h-[500px] overflow-y-auto">
              {wizardData.classes.map((c, idx) => {
                const labeled = !!classNames[c.class_id]?.trim()
                const isCurrent = idx === currentClassIdx
                return (
                  <button
                    key={c.class_id}
                    onClick={() => {
                      setCurrentClassIdx(idx)
                      setCurrentSampleIdx(0)
                      setNameInput(classNames[c.class_id] ?? c.existing_name ?? '')
                    }}
                    className={cn(
                      'w-full flex items-center gap-3 px-3 py-2 rounded-lg text-sm text-left transition-colors',
                      isCurrent ? 'bg-primary/15 border border-primary/30' : 'hover:bg-muted',
                    )}
                  >
                    <span
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ background: COLORS[c.class_id % COLORS.length] }}
                    />
                    <span className="flex-1 font-mono text-xs">ID {c.class_id}</span>
                    {labeled
                      ? <span className="text-xs text-green-500 truncate max-w-[100px]">{classNames[c.class_id]}</span>
                      : <span className="text-xs text-muted-foreground italic">unlabeled</span>
                    }
                    {labeled
                      ? <CheckCircle2 className="w-3.5 h-3.5 text-green-500 flex-shrink-0" />
                      : <AlertCircle className="w-3.5 h-3.5 text-amber-500 flex-shrink-0" />
                    }
                  </button>
                )
              })}
            </CardContent>
          </Card>

          {/* Image + labeling */}
          <Card className="xl:col-span-2">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <span
                  className="w-3 h-3 rounded-full"
                  style={{ background: COLORS[currentClassEntry.class_id % COLORS.length] }}
                />
                Labeling Class ID {currentClassEntry.class_id}
                <span className="text-muted-foreground font-normal text-xs ml-1">
                  — highlighted in image
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              {sample ? (
                <>
                  <AnnotationCanvas
                    apiUrl={apiUrl}
                    datasetId={selectedDataset.id}
                    imagePath={sample.image_path}
                    annotations={sample.annotations}
                    classNames={classNames}
                    highlightClassId={currentClassEntry.class_id}
                  />
                  {/* Sample navigation */}
                  {currentClassEntry.samples.length > 1 && (
                    <div className="flex items-center justify-center gap-2">
                      {currentClassEntry.samples.map((_, i) => (
                        <button
                          key={i}
                          onClick={() => setCurrentSampleIdx(i)}
                          className={cn(
                            'w-2 h-2 rounded-full transition-colors',
                            i === currentSampleIdx ? 'bg-primary' : 'bg-muted-foreground/30 hover:bg-muted-foreground/60'
                          )}
                        />
                      ))}
                      <span className="text-xs text-muted-foreground ml-2">
                        Image {currentSampleIdx + 1} of {currentClassEntry.samples.length}
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div className="flex items-center justify-center h-40 rounded-lg bg-muted/30 text-muted-foreground text-sm">
                  No sample images for this class
                </div>
              )}

              {/* Label input */}
              <div className="space-y-2">
                <label className="text-sm font-medium">
                  What are these highlighted annotations called?
                </label>
                <div className="flex gap-2">
                  <Input
                    value={nameInput}
                    onChange={e => setNameInput(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && commitLabel()}
                    placeholder={`Class name for ID ${currentClassEntry.class_id}…`}
                    className="flex-1"
                    autoFocus
                  />
                  <Button onClick={commitLabel} disabled={!nameInput.trim()}>
                    {currentClassIdx < wizardData.classes.length - 1 ? (
                      <>Confirm <ChevronRight className="w-4 h-4 ml-1" /></>
                    ) : (
                      <>Done <CheckCircle2 className="w-4 h-4 ml-1" /></>
                    )}
                  </Button>
                </div>
                {currentClassEntry.existing_name && currentClassEntry.existing_name !== nameInput && (
                  <p className="text-xs text-muted-foreground">
                    Existing name: <button className="underline" onClick={() => setNameInput(currentClassEntry.existing_name!)}>
                      {currentClassEntry.existing_name}
                    </button>
                  </p>
                )}
              </div>

              {/* Previous / Next nav */}
              <div className="flex justify-between pt-1">
                <Button
                  variant="ghost" size="sm"
                  disabled={currentClassIdx === 0}
                  onClick={() => {
                    const prev = wizardData.classes[currentClassIdx - 1]
                    setCurrentClassIdx(currentClassIdx - 1)
                    setCurrentSampleIdx(0)
                    setNameInput(classNames[prev.class_id] ?? prev.existing_name ?? '')
                  }}
                >
                  <ChevronLeft className="w-4 h-4 mr-1" />Prev Class
                </Button>
                <Button
                  variant="ghost" size="sm"
                  disabled={currentClassIdx >= wizardData.classes.length - 1}
                  onClick={() => {
                    const next = wizardData.classes[currentClassIdx + 1]
                    setCurrentClassIdx(currentClassIdx + 1)
                    setCurrentSampleIdx(0)
                    setNameInput(classNames[next.class_id] ?? next.existing_name ?? '')
                  }}
                >
                  Next Class<ChevronRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  // ── Verification phase — one card per class ────────────────────────────────
  if (phase === 'verifying') {
    const save = async () => {
      if (!selectedDataset) return
      setIsSaving(true)
      try {
        const orderedNames = wizardData.classes.map(c => classNames[c.class_id] ?? `class_${c.class_id}`)
        const body = {
          class_names: orderedNames,
          train_path: splits.train || null,
          val_path:   splits.val   || null,
          test_path:  splits.test  || null,
        }
        const r = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/generate-yaml`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        })
        if (!r.ok) throw new Error('Save failed')
        const data = await r.json()
        setYamlPreview(data.preview)
        setPhase('done')
        toast.success(`data.yaml saved to dataset — training is now ready`)
      } catch (e) {
        toast.error(e instanceof Error ? e.message : 'Save failed')
      } finally {
        setIsSaving(false)
      }
    }

    return (
      <div className="p-6 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-2xl font-semibold">Verify Each Class</h1>
            <p className="text-muted-foreground mt-1">
              Each card shows a sample for that class — confirm the name matches what you see highlighted
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={() => setPhase('labeling')}>
              <Pencil className="w-4 h-4 mr-2" />Edit Names
            </Button>
            <Button onClick={save} disabled={isSaving}>
              {isSaving
                ? <><RefreshCw className="w-4 h-4 mr-2 animate-spin" />Saving…</>
                : <><CheckCircle2 className="w-4 h-4 mr-2" />Confirm & Save YAML</>
              }
            </Button>
          </div>
        </div>

        {/* Split paths row */}
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Split Paths (editable)</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-3 gap-3">
              {(['train', 'val', 'test'] as const).map(k => (
                <div key={k} className="space-y-1">
                  <label className="text-xs text-muted-foreground capitalize">{k}</label>
                  <Input
                    value={splits[k]}
                    onChange={e => setSplits(prev => ({ ...prev, [k]: e.target.value }))}
                    placeholder={`../${k}/images`}
                    className="text-xs h-7"
                  />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* One card per class — unique sample image */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {wizardData.classes.map(c => {
            const sample = c.samples[0]
            const name   = classNames[c.class_id] ?? `class_${c.class_id}`
            const color  = COLORS[c.class_id % COLORS.length]
            return (
              <Card key={c.class_id} className="overflow-hidden">
                <CardHeader className="pb-2">
                  <CardTitle className="text-sm flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full flex-shrink-0" style={{ background: color }} />
                    <span>ID {c.class_id}</span>
                    <span className="ml-auto font-normal text-foreground">{name}</span>
                  </CardTitle>
                  <CardDescription className="text-xs">
                    Highlighted annotations should represent "{name}"
                  </CardDescription>
                </CardHeader>
                <CardContent className="pt-0">
                  {sample ? (
                    <AnnotationCanvas
                      apiUrl={apiUrl}
                      datasetId={selectedDataset.id}
                      imagePath={sample.image_path}
                      annotations={sample.annotations}
                      classNames={classNames}
                      highlightClassId={c.class_id}
                    />
                  ) : (
                    <div className="h-32 flex items-center justify-center bg-muted/30 rounded-lg text-muted-foreground text-xs">
                      No sample image
                    </div>
                  )}
                </CardContent>
              </Card>
            )
          })}
        </div>
      </div>
    )
  }

  // ── Done phase ─────────────────────────────────────────────────────────────
  return (
    <div className="p-6 space-y-4">
      {/* Prominent success banner */}
      <div className="rounded-xl border border-green-500/30 bg-green-500/10 p-4 flex items-start gap-3">
        <CheckCircle2 className="w-6 h-6 text-green-500 flex-shrink-0 mt-0.5" />
        <div>
          <p className="font-semibold text-green-500">data.yaml saved to dataset</p>
          <p className="text-sm text-muted-foreground mt-0.5">
            The file has been written into <strong>{selectedDataset.name}</strong>'s directory.
            Training jobs will now find it automatically.
          </p>
          <p className="text-xs text-muted-foreground mt-1 font-mono">{wizardData.dataset_path}/data.yaml</p>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">Generated data.yaml</h1>
        <div className="flex gap-2">
          <Button variant="outline" onClick={load}>
            <RefreshCw className="w-4 h-4 mr-2" />Re-run Wizard
          </Button>
        </div>
      </div>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center justify-between">
            <span className="flex items-center gap-2"><FileCode2 className="w-4 h-4" />data.yaml</span>
            <div className="flex gap-2">
              <Button variant="ghost" size="sm" onClick={() => { navigator.clipboard.writeText(yamlPreview); toast.success('Copied') }}>
                <Copy className="w-4 h-4 mr-1" />Copy
              </Button>
              <Button variant="ghost" size="sm" onClick={() => {
                const blob = new Blob([yamlPreview], { type: 'text/yaml' })
                const a = document.createElement('a')
                a.href = URL.createObjectURL(blob)
                a.download = 'data.yaml'
                a.click()
              }}>
                <Download className="w-4 h-4 mr-1" />Download
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <pre className="bg-muted rounded-lg p-4 text-sm font-mono whitespace-pre-wrap overflow-x-auto">
            {yamlPreview}
          </pre>
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Layers className="w-4 h-4" />Class Mapping
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
            {wizardData.classes.map(c => (
              <div key={c.class_id} className="flex items-center gap-2 p-2 rounded-lg bg-muted/50 text-sm">
                <span className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                  style={{ background: COLORS[c.class_id % COLORS.length] }} />
                <span className="font-mono text-xs text-muted-foreground">{c.class_id}</span>
                <span className="font-medium truncate">{classNames[c.class_id]}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
