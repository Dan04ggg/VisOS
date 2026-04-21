'use client'

import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Checkbox } from '@/components/ui/checkbox'
import {
  Layers,
  Trash2,
  GitMerge,
  Copy,
  Plus,
  AlertTriangle,
  CheckCircle,
  ArrowRight,
  Search,
  RefreshCw,
  Pencil,
  ImageMinus
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

const EXPORT_FORMATS = [
  { id: 'yolo',           name: 'YOLO' },
  { id: 'coco',           name: 'COCO JSON' },
  { id: 'pascal-voc',     name: 'Pascal VOC' },
  { id: 'labelme',        name: 'LabelMe' },
  { id: 'tensorflow-csv', name: 'TensorFlow CSV' },
  { id: 'createml',       name: 'CreateML' },
]

interface ClassManagementViewProps {
  selectedDataset: Dataset | null
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  apiUrl: string
}

interface ClassInfo {
  name: string
  count: number
  selected: boolean
}

export function ClassManagementView({ 
  selectedDataset, 
  datasets,
  setDatasets,
  apiUrl 
}: ClassManagementViewProps) {
  const [classes, setClasses] = useState<ClassInfo[]>([])
  const [loading, setLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState('')
  const [activeTab, setActiveTab] = useState<'extract' | 'delete' | 'merge' | 'add' | 'rename' | 'remove-unannotated'>('extract')
  const [newClassName, setNewClassName] = useState('')
  const [mergeTargetName, setMergeTargetName] = useState('')
  const [extractOutputName, setExtractOutputName] = useState('')
  const [extractOutputFormat, setExtractOutputFormat] = useState('')
  const [renameNewName, setRenameNewName] = useState('')
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)

  useEffect(() => {
    if (selectedDataset) {
      loadClasses()
      setExtractOutputName(`${selectedDataset.name}_extracted`)
      setExtractOutputFormat(selectedDataset.format || '')
    }
  }, [selectedDataset])

  const loadClasses = async () => {
    if (!selectedDataset) return
    setLoading(true)
    setMessage(null)
    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/classes`)
      if (!response.ok) {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || `Server returned ${response.status}`)
      }
      const data = await response.json()
      const classData: Record<string, number> = data.classes || {}

      if (Object.keys(classData).length === 0 && selectedDataset.classes?.length > 0) {
        // Backend returned empty dict but dataset has known classes — use them as fallback
        setClasses(selectedDataset.classes.map(name => ({ name, count: 0, selected: false })))
      } else {
        setClasses(
          Object.entries(classData).map(([name, count]) => ({
            name,
            count: count as number,
            selected: false,
          }))
        )
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Unknown error'
      setMessage({ type: 'error', text: `Failed to load classes: ${msg}` })
      // Still show whatever the dataset metadata has so the panel is not blank
      if (selectedDataset.classes?.length > 0) {
        setClasses(selectedDataset.classes.map(name => ({ name, count: 0, selected: false })))
      }
    } finally {
      setLoading(false)
    }
  }

  const toggleClass = (className: string) => {
    setClasses(classes.map(c => 
      c.name === className ? { ...c, selected: !c.selected } : c
    ))
  }

  const selectAll = () => {
    setClasses(classes.map(c => ({ ...c, selected: true })))
  }

  const deselectAll = () => {
    setClasses(classes.map(c => ({ ...c, selected: false })))
  }

  const selectedClasses = classes.filter(c => c.selected)
  const filteredClasses = classes.filter(c => 
    c.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const handleExtractClasses = async () => {
    if (!selectedDataset || selectedClasses.length === 0) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/extract-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          classes_to_extract: selectedClasses.map(c => c.name),
          output_name: extractOutputName || `${selectedDataset.name}_extracted`,
          output_format: extractOutputFormat || undefined
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.new_dataset) {
          setDatasets([...datasets, data.new_dataset])
        }
        setMessage({
          type: 'success',
          text: `Extracted ${selectedClasses.length} class(es) to new dataset with ${data.extracted_images} images`
        })
        deselectAll()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Failed to extract classes')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to extract classes' })
    }
    setLoading(false)
  }

  const handleRenameClass = async () => {
    if (!selectedDataset || selectedClasses.length !== 1 || !renameNewName.trim()) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/rename-class`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          old_name: selectedClasses[0].name,
          new_name: renameNewName.trim()
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          setDatasets(datasets.map(d =>
            d.id === selectedDataset.id ? data.updated_dataset : d
          ))
        }
        setMessage({
          type: 'success',
          text: `Renamed "${selectedClasses[0].name}" to "${renameNewName}" (${data.renamed_annotations} annotations updated)`
        })
        setRenameNewName('')
        deselectAll()
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Failed to rename class')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to rename class' })
    }
    setLoading(false)
  }

  const handleDeleteClasses = async () => {
    if (!selectedDataset || selectedClasses.length === 0) return
    
    if (!confirm(`Are you sure you want to delete ${selectedClasses.length} class(es)? This action cannot be undone.`)) {
      return
    }

    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/delete-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          classes_to_delete: selectedClasses.map(c => c.name)
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          setDatasets(datasets.map(d =>
            d.id === selectedDataset.id ? data.updated_dataset : d
          ))
        }
        setMessage({
          type: 'success',
          text: `Deleted ${selectedClasses.length} class(es) (${data.deleted_annotations} annotations removed)`
        })
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Failed to delete classes')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to delete classes' })
    }
    setLoading(false)
  }

  const handleMergeClasses = async () => {
    if (!selectedDataset || selectedClasses.length < 2 || !mergeTargetName) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/merge-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          source_classes: selectedClasses.map(c => c.name),
          target_class: mergeTargetName
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          setDatasets(datasets.map(d =>
            d.id === selectedDataset.id ? data.updated_dataset : d
          ))
        }
        setMessage({
          type: 'success',
          text: `Merged ${selectedClasses.length} class(es) into "${mergeTargetName}" (${data.merged_annotations} annotations updated)`
        })
        setMergeTargetName('')
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Failed to merge classes')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to merge classes' })
    }
    setLoading(false)
  }

  const handleAddClass = async () => {
    if (!selectedDataset || !newClassName.trim()) return
    setLoading(true)
    setMessage(null)

    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/add-classes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_id: selectedDataset.id,
          new_classes: [newClassName.trim()],
          use_model: false
        })
      })

      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          setDatasets(datasets.map(d => 
            d.id === selectedDataset.id ? data.updated_dataset : d
          ))
        }
        setMessage({ type: 'success', text: `Added class "${newClassName}"` })
        setNewClassName('')
        loadClasses()
      } else {
        throw new Error('Failed to add class')
      }
    } catch (err) {
      setMessage({ type: 'error', text: 'Failed to add class' })
    }
    setLoading(false)
  }

  const handleRemoveUnannotated = async () => {
    if (!selectedDataset) return
    setLoading(true)
    setMessage(null)
    try {
      const response = await fetch(`${apiUrl}/api/datasets/${selectedDataset.id}/remove-unannotated`, {
        method: 'POST',
      })
      if (response.ok) {
        const data = await response.json()
        if (data.updated_dataset) {
          setDatasets(datasets.map(d =>
            d.id === selectedDataset.id ? data.updated_dataset : d
          ))
        }
        setMessage({ type: 'success', text: `Removed ${data.removed} unannotated image(s)` })
        loadClasses()
      } else {
        const body = await response.json().catch(() => ({}))
        throw new Error(body.detail || 'Failed to remove unannotated images')
      }
    } catch (err) {
      setMessage({ type: 'error', text: err instanceof Error ? err.message : 'Failed to remove unannotated images' })
    }
    setLoading(false)
  }

  if (!selectedDataset) {
    return (
      <div className="h-full flex flex-col items-center justify-center p-6 text-center">
        <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
          <Layers className="w-10 h-10 text-muted-foreground" />
        </div>
        <h2 className="text-2xl font-semibold text-foreground mb-2">No Dataset Selected</h2>
        <p className="text-muted-foreground max-w-md">
          Select a dataset to manage its classes
        </p>
      </div>
    )
  }

  return (
    <div className="h-full flex flex-col p-6 overflow-y-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-foreground">Class Management</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Extract, delete, merge, or add classes in {selectedDataset.name}
          </p>
        </div>
      </div>

      {message && (
        <div className={cn(
          'mb-4 p-3 rounded-lg flex items-center gap-2',
          message.type === 'success' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 'bg-destructive/10 text-destructive border border-destructive/20'
        )}>
          {message.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          <span className="text-sm">{message.text}</span>
          <button onClick={() => setMessage(null)} className="ml-auto text-xs underline">Dismiss</button>
        </div>
      )}

      <div className="flex gap-6 flex-1 min-h-0">
        <Card className="w-80 flex-shrink-0 flex flex-col">
          <CardHeader className="pb-3">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Classes ({classes.length})</CardTitle>
              <Button
                variant="ghost"
                size="icon"
                className="h-7 w-7"
                onClick={loadClasses}
                disabled={loading}
                title="Refresh class list"
              >
                <RefreshCw className={cn("w-3.5 h-3.5", loading && "animate-spin")} />
              </Button>
            </div>
            <CardDescription>
              {selectedClasses.length} selected
            </CardDescription>
          </CardHeader>
          <CardContent className="flex-1 overflow-hidden flex flex-col">
            <div className="relative mb-3">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input 
                placeholder="Search classes..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>
            
            <div className="flex items-center gap-2 mb-3">
              <Button variant="outline" size="sm" onClick={selectAll}>Select All</Button>
              <Button variant="outline" size="sm" onClick={deselectAll}>Deselect All</Button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-1">
              {filteredClasses.map((cls) => (
                <div 
                  key={cls.name}
                  className={cn(
                    'flex items-center gap-3 p-2 rounded-lg cursor-pointer transition-colors',
                    cls.selected ? 'bg-primary/10' : 'hover:bg-muted'
                  )}
                  onClick={() => toggleClass(cls.name)}
                >
                  <Checkbox checked={cls.selected} />
                  <span className="flex-1 text-sm truncate">{cls.name}</span>
                  <span className="text-xs text-muted-foreground">{cls.count}</span>
                </div>
              ))}
              {filteredClasses.length === 0 && (
                <p className="text-sm text-muted-foreground text-center py-4">No classes found</p>
              )}
            </div>
          </CardContent>
        </Card>

        <div className="flex-1 overflow-y-auto">
          <div className="flex flex-wrap gap-2 mb-4">
            {(['extract', 'delete', 'merge', 'rename', 'add', 'remove-unannotated'] as const).map((tab) => (
              <Button
                key={tab}
                variant={activeTab === tab ? 'default' : 'outline'}
                size="sm"
                onClick={() => setActiveTab(tab)}
              >
                {tab === 'extract' && <Copy className="w-4 h-4 mr-2" />}
                {tab === 'delete' && <Trash2 className="w-4 h-4 mr-2" />}
                {tab === 'merge' && <GitMerge className="w-4 h-4 mr-2" />}
                {tab === 'rename' && <Pencil className="w-4 h-4 mr-2" />}
                {tab === 'add' && <Plus className="w-4 h-4 mr-2" />}
                {tab === 'remove-unannotated' && <ImageMinus className="w-4 h-4 mr-2" />}
                {tab === 'remove-unannotated' ? 'Remove Unannotated' : tab.charAt(0).toUpperCase() + tab.slice(1)}
              </Button>
            ))}
          </div>

          {activeTab === 'extract' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Copy className="w-5 h-5" />
                  Extract Classes to New Dataset
                </CardTitle>
                <CardDescription>
                  Create a new dataset containing only the selected classes. Only images that have at least one annotation of those classes are included.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Output Dataset Name</Label>
                  <Input
                    value={extractOutputName}
                    onChange={(e) => setExtractOutputName(e.target.value)}
                    placeholder="New dataset name"
                  />
                </div>

                <div className="space-y-2">
                  <Label>Output Format</Label>
                  <select
                    className="w-full border rounded-md px-3 py-2 text-sm bg-background"
                    value={extractOutputFormat}
                    onChange={(e) => setExtractOutputFormat(e.target.value)}
                  >
                    {EXPORT_FORMATS.map(f => (
                      <option key={f.id} value={f.id}>{f.name}</option>
                    ))}
                  </select>
                  <p className="text-xs text-muted-foreground">
                    Currently: {selectedDataset?.format || '—'}. Change to convert the extracted dataset.
                  </p>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">Classes to extract:</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedClasses.map(c => (
                        <span key={c.name} className="px-2 py-0.5 bg-background rounded text-xs">
                          {c.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <Button
                  onClick={handleExtractClasses}
                  disabled={loading || selectedClasses.length === 0}
                  className="w-full"
                >
                  {loading ? 'Extracting...' : `Extract ${selectedClasses.length} Class(es)`}
                  <ArrowRight className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'delete' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Trash2 className="w-5 h-5 text-destructive" />
                  Delete Classes
                </CardTitle>
                <CardDescription>
                  Remove selected classes and their annotations from the dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">Warning</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        This action will permanently remove all annotations for the selected classes. 
                        Images will be kept but their labels for these classes will be deleted.
                      </p>
                    </div>
                  </div>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">Classes to delete:</p>
                    <div className="flex flex-wrap gap-1">
                      {selectedClasses.map(c => (
                        <span key={c.name} className="px-2 py-0.5 bg-destructive/20 text-destructive rounded text-xs">
                          {c.name}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                <Button 
                  variant="destructive"
                  onClick={handleDeleteClasses} 
                  disabled={loading || selectedClasses.length === 0}
                  className="w-full"
                >
                  {loading ? 'Deleting...' : `Delete ${selectedClasses.length} Class(es)`}
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'merge' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <GitMerge className="w-5 h-5" />
                  Merge Classes
                </CardTitle>
                <CardDescription>
                  Combine multiple classes into a single class
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Target Class Name</Label>
                  <Input 
                    value={mergeTargetName}
                    onChange={(e) => setMergeTargetName(e.target.value)}
                    placeholder="Name for merged class"
                  />
                  <p className="text-xs text-muted-foreground">
                    All selected classes will be renamed to this class
                  </p>
                </div>

                {selectedClasses.length > 0 && (
                  <div className="p-3 bg-muted rounded-lg">
                    <p className="text-sm font-medium mb-2">Classes to merge:</p>
                    <div className="flex flex-wrap gap-1 items-center">
                      {selectedClasses.map((c, idx) => (
                        <span key={c.name}>
                          <span className="px-2 py-0.5 bg-background rounded text-xs">{c.name}</span>
                          {idx < selectedClasses.length - 1 && (
                            <span className="mx-1 text-muted-foreground">+</span>
                          )}
                        </span>
                      ))}
                      {mergeTargetName && (
                        <>
                          <ArrowRight className="w-4 h-4 mx-2 text-muted-foreground" />
                          <span className="px-2 py-0.5 bg-primary text-primary-foreground rounded text-xs">
                            {mergeTargetName}
                          </span>
                        </>
                      )}
                    </div>
                  </div>
                )}

                <Button 
                  onClick={handleMergeClasses} 
                  disabled={loading || selectedClasses.length < 2 || !mergeTargetName}
                  className="w-full"
                >
                  {loading ? 'Merging...' : `Merge ${selectedClasses.length} Classes`}
                </Button>
                
                {selectedClasses.length < 2 && (
                  <p className="text-xs text-muted-foreground text-center">
                    Select at least 2 classes to merge
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {activeTab === 'rename' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Pencil className="w-5 h-5" />
                  Rename Class
                </CardTitle>
                <CardDescription>
                  Rename a class across all annotations in the dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {selectedClasses.length === 1 ? (
                  <>
                    <div className="p-3 bg-muted rounded-lg">
                      <p className="text-sm text-muted-foreground mb-1">Renaming class:</p>
                      <p className="font-medium">{selectedClasses[0].name}</p>
                      <p className="text-xs text-muted-foreground mt-1">{selectedClasses[0].count} annotation(s)</p>
                    </div>
                    <div className="space-y-2">
                      <Label>New Class Name</Label>
                      <Input
                        value={renameNewName}
                        onChange={(e) => setRenameNewName(e.target.value)}
                        placeholder="Enter new name"
                        onKeyDown={(e) => e.key === 'Enter' && handleRenameClass()}
                      />
                    </div>
                    {renameNewName && (
                      <div className="flex items-center gap-2 text-sm p-2 bg-muted rounded-lg">
                        <span className="px-2 py-0.5 bg-background rounded text-xs">{selectedClasses[0].name}</span>
                        <ArrowRight className="w-4 h-4 text-muted-foreground" />
                        <span className="px-2 py-0.5 bg-primary text-primary-foreground rounded text-xs">{renameNewName}</span>
                      </div>
                    )}
                    <Button
                      onClick={handleRenameClass}
                      disabled={loading || !renameNewName.trim() || renameNewName.trim() === selectedClasses[0].name}
                      className="w-full"
                    >
                      {loading ? 'Renaming...' : 'Rename Class'}
                      <Pencil className="w-4 h-4 ml-2" />
                    </Button>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground text-center py-4">
                    Select exactly 1 class to rename
                  </p>
                )}
              </CardContent>
            </Card>
          )}

          {activeTab === 'add' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <Plus className="w-5 h-5" />
                  Add New Class
                </CardTitle>
                <CardDescription>
                  Add a new class to the dataset for manual annotation
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <Label>Class Name</Label>
                  <Input
                    value={newClassName}
                    onChange={(e) => setNewClassName(e.target.value)}
                    placeholder="Enter new class name"
                    onKeyDown={(e) => e.key === 'Enter' && handleAddClass()}
                  />
                </div>

                <Button
                  onClick={handleAddClass}
                  disabled={loading || !newClassName.trim()}
                  className="w-full"
                >
                  {loading ? 'Adding...' : 'Add Class'}
                  <Plus className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}

          {activeTab === 'remove-unannotated' && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base flex items-center gap-2">
                  <ImageMinus className="w-5 h-5 text-destructive" />
                  Remove Unannotated Images
                </CardTitle>
                <CardDescription>
                  Permanently delete all images that have no annotations from this dataset
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-destructive flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-destructive">Warning</p>
                      <p className="text-sm text-muted-foreground mt-1">
                        This will permanently delete every image file that has zero annotations.
                        Label files for those images are also removed. This action cannot be undone.
                      </p>
                    </div>
                  </div>
                </div>
                <p className="text-sm text-muted-foreground">
                  Use this to clean up a dataset after deleting classes or to remove images
                  that were never annotated before training.
                </p>
                <Button
                  variant="destructive"
                  onClick={handleRemoveUnannotated}
                  disabled={loading}
                  className="w-full"
                >
                  {loading ? 'Removing...' : 'Remove Unannotated Images'}
                  <ImageMinus className="w-4 h-4 ml-2" />
                </Button>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  )
}
