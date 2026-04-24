'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { Input } from '@/components/ui/input'
import { 
  Shuffle,
  CheckCircle,
  AlertCircle,
  Loader2,
  FolderOpen,
  Image
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

interface SplitViewProps {
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  selectedDataset: Dataset | null
  onDatasetCreated?: (dataset: Dataset) => void
  apiUrl: string
}

interface SplitConfig {
  train: number
  val: number
  test: number
}

interface SplitResult {
  success: boolean
  message: string
  train_count: number
  val_count: number
  test_count: number
}

export function SplitView({ datasets, setDatasets, selectedDataset, onDatasetCreated, apiUrl }: SplitViewProps) {
  const [splitConfig, setSplitConfig] = useState<SplitConfig>({ train: 70, val: 20, test: 10 })
  const [outputName, setOutputName] = useState('')
  const [isSplitting, setIsSplitting] = useState(false)
  const [result, setResult] = useState<SplitResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [seed, setSeed] = useState(42)
  const [stratify, setStratify] = useState(true)
  const [currentSplits, setCurrentSplits] = useState<Record<string, number> | null>(null)

  useEffect(() => {
    if (!selectedDataset) { setCurrentSplits(null); return }
    fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/stats`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.splits && Object.keys(data.splits).length > 0) {
          setCurrentSplits(data.splits)
        } else {
          setCurrentSplits(null)
        }
      })
      .catch(() => setCurrentSplits(null))
  }, [selectedDataset?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // Calculate counts based on percentages
  const totalImages = selectedDataset?.num_images || 0
  const trainCount = Math.round((splitConfig.train / 100) * totalImages)
  const valCount = Math.round((splitConfig.val / 100) * totalImages)
  const testCount = totalImages - trainCount - valCount

  const handleTrainChange = (value: number[]) => {
    const train = value[0]
    const remaining = 100 - train
    const valRatio = splitConfig.val / (splitConfig.val + splitConfig.test) || 0.67
    const val = Math.round(remaining * valRatio)
    const test = remaining - val
    setSplitConfig({ train, val, test })
  }

  const handleValChange = (value: number[]) => {
    const val = value[0]
    const maxVal = 100 - splitConfig.train
    const clampedVal = Math.min(val, maxVal)
    const test = maxVal - clampedVal
    setSplitConfig({ ...splitConfig, val: clampedVal, test })
  }

  const handleSplit = async () => {
    if (!selectedDataset) return

    setIsSplitting(true)
    setError(null)
    setResult(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/split`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          train_ratio: splitConfig.train / 100,
          val_ratio: splitConfig.val / 100,
          test_ratio: splitConfig.test / 100,
          output_name: outputName || `${selectedDataset.name}_split`,
          seed,
          stratify
        })
      })

      if (!response.ok) {
        const err = await response.json()
        const detail = err.detail
        const msg = Array.isArray(detail)
          ? detail.map((e: { msg?: string }) => e.msg || JSON.stringify(e)).join('; ')
          : (typeof detail === 'string' ? detail : 'Split failed')
        throw new Error(msg)
      }

      const data = await response.json()

      if (data.success) {
        setResult({
          success: true,
          message: 'Dataset split successfully!',
          train_count: data.train_count || trainCount,
          val_count: data.val_count || valCount,
          test_count: data.test_count || testCount
        })

        // Refresh datasets list
        const datasetsResponse = await fetch(`${apiUrl}/api/datasets`)
        if (datasetsResponse.ok) {
          const datasetsData = await datasetsResponse.json()
          setDatasets(datasetsData.datasets || [])
        }

        if (data.new_dataset && onDatasetCreated) {
          onDatasetCreated(data.new_dataset)
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Split failed')
    } finally {
      setIsSplitting(false)
    }
  }

  if (!selectedDataset) {
    return (
      <div className="h-full flex items-center justify-center">
        <div className="text-center">
          <AlertCircle className="w-12 h-12 text-muted-foreground/50 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-muted-foreground">No dataset selected</h3>
          <p className="text-sm text-muted-foreground/70 mt-1">
            Select a dataset from the Datasets view to create a train/val/test split
          </p>
        </div>
      </div>
    )
  }

  return (
    <div className="p-6 space-y-6">
      <div className="mb-6">
        <h2 className="text-2xl font-semibold text-foreground">Train/Val/Test Split</h2>
        <p className="text-muted-foreground text-sm mt-1">
          Split your dataset into training, validation, and test sets
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 mb-6">
        {/* Source Dataset Info */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Source Dataset</CardTitle>
            <CardDescription>The dataset to split</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="p-4 bg-muted/50 rounded-lg space-y-3">
              <div className="flex items-center gap-3">
                <FolderOpen className="w-5 h-5 text-primary" />
                <div>
                  <p className="font-medium">{selectedDataset.name}</p>
                  <p className="text-sm text-muted-foreground">{selectedDataset.format.toUpperCase()}</p>
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Image className="w-4 h-4 text-muted-foreground" />
                <span>{totalImages.toLocaleString()} images</span>
              </div>
              {selectedDataset.classes && selectedDataset.classes.length > 0 && (
                <div className="pt-2 border-t">
                  <p className="text-sm text-muted-foreground mb-2">Classes ({selectedDataset.classes.length})</p>
                  <div className="flex flex-wrap gap-1">
                    {selectedDataset.classes.slice(0, 6).map((cls, idx) => (
                      <span key={idx} className="px-2 py-0.5 bg-muted rounded text-xs">{cls}</span>
                    ))}
                    {selectedDataset.classes.length > 6 && (
                      <span className="px-2 py-0.5 text-muted-foreground text-xs">
                        +{selectedDataset.classes.length - 6} more
                      </span>
                    )}
                  </div>
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Current Split Breakdown */}
        {currentSplits && (
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Current Split</CardTitle>
              <CardDescription>Existing train/val/test distribution in this dataset</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {Object.entries(currentSplits).map(([split, count]) => {
                  const total = Object.values(currentSplits).reduce((a, b) => a + b, 0)
                  const pct = total > 0 ? Math.round((count / total) * 100) : 0
                  const color = split === 'train' ? 'bg-primary' : split === 'val' || split === 'valid' ? 'bg-amber-500' : 'bg-blue-500'
                  const textColor = split === 'train' ? 'text-primary' : split === 'val' || split === 'valid' ? 'text-amber-500' : 'text-blue-500'
                  return (
                    <div key={split} className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="capitalize font-medium">{split}</span>
                        <span className={cn('font-medium', textColor)}>{count} images ({pct}%)</span>
                      </div>
                      <div className="h-2 bg-secondary rounded-full overflow-hidden">
                        <div className={cn('h-full rounded-full', color)} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  )
                })}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Split Configuration */}
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Split Configuration</CardTitle>
            <CardDescription>Adjust the split ratios</CardDescription>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Train Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Training Set</label>
                <span className="text-sm text-primary font-medium">{splitConfig.train}% ({trainCount} images)</span>
              </div>
              <Slider
                value={[splitConfig.train]}
                onValueChange={handleTrainChange}
                min={10}
                max={90}
                step={5}
                className="[&>span:first-child]:bg-primary"
              />
            </div>

            {/* Val Slider */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Validation Set</label>
                <span className="text-sm text-amber-500 font-medium">{splitConfig.val}% ({valCount} images)</span>
              </div>
              <Slider
                value={[splitConfig.val]}
                onValueChange={handleValChange}
                min={0}
                max={100 - splitConfig.train}
                step={5}
                className="[&>span:first-child]:bg-amber-500"
              />
            </div>

            {/* Test (calculated) */}
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <label className="text-sm font-medium">Test Set</label>
                <span className="text-sm text-blue-500 font-medium">{splitConfig.test}% ({testCount} images)</span>
              </div>
              <div className="h-2 bg-secondary rounded-full overflow-hidden">
                <div className="h-full bg-blue-500 rounded-full" style={{ width: `${splitConfig.test}%` }} />
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Visual Split Preview */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">Split Preview</CardTitle>
          <CardDescription>Visual representation of the split</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-8 rounded-lg overflow-hidden flex">
            <div 
              className="bg-primary flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.train}%` }}
            >
              {splitConfig.train >= 20 && `Train ${splitConfig.train}%`}
            </div>
            <div 
              className="bg-amber-500 flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.val}%` }}
            >
              {splitConfig.val >= 15 && `Val ${splitConfig.val}%`}
            </div>
            <div 
              className="bg-blue-500 flex items-center justify-center text-white text-xs font-medium transition-all"
              style={{ width: `${splitConfig.test}%` }}
            >
              {splitConfig.test >= 10 && `Test ${splitConfig.test}%`}
            </div>
          </div>
          
          <div className="grid grid-cols-3 gap-4 mt-4 text-center">
            <div className="p-3 bg-primary/10 rounded-lg">
              <p className="text-2xl font-bold text-primary">{trainCount}</p>
              <p className="text-xs text-muted-foreground">Training Images</p>
            </div>
            <div className="p-3 bg-amber-500/10 rounded-lg">
              <p className="text-2xl font-bold text-amber-500">{valCount}</p>
              <p className="text-xs text-muted-foreground">Validation Images</p>
            </div>
            <div className="p-3 bg-blue-500/10 rounded-lg">
              <p className="text-2xl font-bold text-blue-500">{testCount}</p>
              <p className="text-xs text-muted-foreground">Test Images</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Advanced Options */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">Options</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">Output Name</label>
              <Input
                value={outputName}
                onChange={(e) => setOutputName(e.target.value)}
                placeholder={`${selectedDataset.name}_split`}
              />
            </div>
            <div>
              <label className="text-sm text-muted-foreground mb-2 block">Random Seed</label>
              <Input
                type="number"
                value={seed}
                onChange={(e) => setSeed(parseInt(e.target.value) || 42)}
                placeholder="42"
              />
            </div>
          </div>
          
          <div className="flex items-center gap-3">
            <input
              type="checkbox"
              id="stratify"
              checked={stratify}
              onChange={(e) => setStratify(e.target.checked)}
              className="rounded border-border"
            />
            <label htmlFor="stratify" className="text-sm">
              Stratified split (maintain class distribution in each split)
            </label>
          </div>
        </CardContent>
      </Card>

      {/* Action */}
      <div className="flex items-center justify-center gap-4 mb-6">
        <Button
          size="lg"
          onClick={handleSplit}
          disabled={isSplitting || totalImages === 0}
        >
          {isSplitting ? (
            <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          ) : (
            <Shuffle className="w-4 h-4 mr-2" />
          )}
          {isSplitting ? 'Splitting...' : 'Create Split'}
        </Button>
      </div>

      {/* Result */}
      {result && (
        <Card className="border-primary/50 bg-primary/5 mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-primary" />
              <div>
                <p className="font-medium">{result.message}</p>
                <p className="text-sm text-muted-foreground mt-1">
                  Train: {result.train_count} | Val: {result.val_count} | Test: {result.test_count}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {error && (
        <Card className="border-destructive/50 bg-destructive/5 mb-6">
          <CardContent className="p-4">
            <div className="flex items-center gap-3">
              <AlertCircle className="w-5 h-5 text-destructive" />
              <p className="text-destructive">{error}</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Common Presets */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Quick Presets</CardTitle>
          <CardDescription>Common split ratios used in machine learning</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {[
              { name: '70/20/10', train: 70, val: 20, test: 10 },
              { name: '80/10/10', train: 80, val: 10, test: 10 },
              { name: '60/20/20', train: 60, val: 20, test: 20 },
              { name: '90/5/5', train: 90, val: 5, test: 5 },
            ].map((preset) => (
              <Button
                key={preset.name}
                variant="outline"
                className={cn(
                  'h-auto py-3 flex-col',
                  splitConfig.train === preset.train && 
                  splitConfig.val === preset.val && 
                  'border-primary bg-primary/5'
                )}
                onClick={() => setSplitConfig({ train: preset.train, val: preset.val, test: preset.test })}
              >
                <span className="font-medium">{preset.name}</span>
                <span className="text-xs text-muted-foreground">Train/Val/Test</span>
              </Button>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
