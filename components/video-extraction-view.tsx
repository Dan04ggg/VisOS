'use client'

import { useState, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Slider } from '@/components/ui/slider'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import {
  Video,
  Upload,
  Film,
  Clock,
  Image as ImageIcon,
  Play,
  Pause,
  SkipBack,
  SkipForward,
  Scissors,
  CheckCircle,
  AlertTriangle,
  Layers,
  Grid3X3,
  Timer,
  ZoomIn,
  RefreshCcw
} from 'lucide-react'
import { cn } from '@/lib/utils'
import type { Dataset } from '@/app/page'

interface VideoExtractionViewProps {
  selectedDataset: Dataset | null
  datasets: Dataset[]
  setDatasets: (datasets: Dataset[]) => void
  onDatasetCreated?: (dataset: Dataset) => void
  apiUrl: string
}

interface VideoFile {
  id: string
  name: string
  duration: number
  fps: number
  totalFrames: number
  width: number
  height: number
  url: string
  thumbnail: string | null
}

type ExtractionMode = 'interval' | 'uniform' | 'keyframes' | 'manual'

export function VideoExtractionView({
  selectedDataset,
  datasets,
  setDatasets,
  onDatasetCreated,
  apiUrl
}: VideoExtractionViewProps) {
  const [videos, setVideos] = useState<VideoFile[]>([])
  const [selectedVideo, setSelectedVideo] = useState<VideoFile | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState(0)
  
  // Extraction settings
  const [extractionMode, setExtractionMode] = useState<ExtractionMode>('interval')
  const [frameInterval, setFrameInterval] = useState(30)
  const [uniformCount, setUniformCount] = useState(100)
  const [manualFrames, setManualFrames] = useState<number[]>([])
  const [startTime, setStartTime] = useState(0)
  const [endTime, setEndTime] = useState(0)
  const [outputName, setOutputName] = useState('')
  
  // Video player state
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const videoRef = useRef<HTMLVideoElement>(null)
  
  // Extraction state
  const [isExtracting, setIsExtracting] = useState(false)
  const [extractionProgress, setExtractionProgress] = useState(0)
  const [extractionFrames, setExtractionFrames] = useState(0)
  const [message, setMessage] = useState<{ type: 'success' | 'error', text: string } | null>(null)
  const [existingDatasetId, setExistingDatasetId] = useState<string>('new')

  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (!files || files.length === 0) return

    setUploading(true)
    setUploadProgress(0)

    const uploadNext = (index: number) => {
      if (index >= files.length) {
        setUploading(false)
        setUploadProgress(0)
        if (fileInputRef.current) fileInputRef.current.value = ''
        return
      }

      const file = files[index]
      const formData = new FormData()
      formData.append('video', file)

      const xhr = new XMLHttpRequest()
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) {
          setUploadProgress(Math.round(ev.loaded / ev.total * 100))
        }
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText)
            const newVideo: VideoFile = {
              id: data.id || `vid_${Date.now()}`,
              name: file.name,
              duration: data.duration || 60,
              fps: data.fps || 30,
              totalFrames: data.total_frames || 1800,
              width: data.width || 1920,
              height: data.height || 1080,
              url: data.url || URL.createObjectURL(file),
              thumbnail: data.thumbnail || null,
            }
            setVideos(prev => [...prev, newVideo])
            setSelectedVideo(sv => {
              if (!sv) {
                setEndTime(newVideo.duration)
                setOutputName(`frames_${file.name.replace(/\.[^/.]+$/, '')}`)
                return newVideo
              }
              return sv
            })
          } catch (_) {}
        }
        uploadNext(index + 1)
      }
      xhr.onerror = () => uploadNext(index + 1)
      xhr.open('POST', `${apiUrl}/api/videos/upload`)
      xhr.send(formData)
    }

    uploadNext(0)
  }

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    const ms = Math.floor((seconds % 1) * 100)
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}.${ms.toString().padStart(2, '0')}`
  }

  const calculateEstimatedFrames = () => {
    if (!selectedVideo) return 0
    const duration = endTime - startTime
    switch (extractionMode) {
      case 'interval':
        return Math.floor(duration * selectedVideo.fps / frameInterval)
      case 'uniform':
        return uniformCount
      case 'keyframes':
        return Math.floor(duration / 2) // Rough estimate
      case 'manual':
        return manualFrames.length
    }
  }

  const togglePlayPause = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause()
      } else {
        videoRef.current.play()
      }
      setIsPlaying(!isPlaying)
    }
  }

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime)
    }
  }

  const seekTo = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time
      setCurrentTime(time)
    }
  }

  const addManualFrame = () => {
    const frameNumber = Math.floor(currentTime * (selectedVideo?.fps || 30))
    if (!manualFrames.includes(frameNumber)) {
      setManualFrames(prev => [...prev, frameNumber].sort((a, b) => a - b))
    }
  }

  const handleExtract = async () => {
    if (!selectedVideo) return
    setIsExtracting(true)
    setExtractionProgress(0)
    setExtractionFrames(0)
    setMessage(null)

    try {
      const config: Record<string, unknown> = {
        video_id: selectedVideo.id,
        mode: extractionMode,
        start_time: startTime,
        end_time: endTime,
        nth_frame: frameInterval,
        frame_interval: frameInterval,
        uniform_count: uniformCount,
        manual_frames: manualFrames,
        max_frames: extractionMode === 'uniform' ? uniformCount : undefined,
        output_name: outputName || `frames_${selectedVideo.name.replace(/\.[^/.]+$/, '')}`,
      }
      if (existingDatasetId && existingDatasetId !== 'new') {
        config.existing_dataset_id = existingDatasetId
      }

      const startResp = await fetch(`${apiUrl}/api/videos/extract-frames`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config),
      })

      if (!startResp.ok) {
        const err = await startResp.json().catch(() => ({ detail: 'Extraction failed' }))
        setMessage({ type: 'error', text: err.detail || 'Frame extraction failed.' })
        setIsExtracting(false)
        return
      }

      const { job_id } = await startResp.json()

      // Poll for real progress
      await new Promise<void>((resolve) => {
        const poll = setInterval(async () => {
          try {
            const statusResp = await fetch(`${apiUrl}/api/videos/extract/${job_id}/status`)
            if (!statusResp.ok) return
            const job = await statusResp.json()
            setExtractionProgress(job.progress ?? 0)
            setExtractionFrames(job.extracted_frames ?? 0)

            if (job.status === 'done') {
              clearInterval(poll)
              setExtractionProgress(100)
              if (job.new_dataset && !job.is_existing) {
                setDatasets([...datasets, job.new_dataset])
                onDatasetCreated?.(job.new_dataset)
              } else if (job.new_dataset && job.is_existing) {
                setDatasets(datasets.map(d => d.id === job.dataset_id ? job.new_dataset : d))
              }
              const name = job.new_dataset?.name || outputName
              setMessage({ type: 'success', text: `Extracted ${job.extracted_frames} frames to dataset "${name}"` })
              resolve()
            } else if (job.status === 'error') {
              clearInterval(poll)
              setMessage({ type: 'error', text: job.error || 'Extraction failed.' })
              setExtractionProgress(0)
              resolve()
            }
          } catch (_) {}
        }, 500)
      })
    } catch (err) {
      setMessage({ type: 'error', text: `Extraction failed: ${err instanceof Error ? err.message : 'Unknown error'}` })
      setExtractionProgress(0)
    }
    setIsExtracting(false)
  }

  return (
    <div className="flex flex-col p-6 overflow-y-auto min-h-full">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-semibold text-foreground">Video Frame Extraction</h2>
          <p className="text-muted-foreground text-sm mt-1">
            Extract frames from videos to create training datasets
          </p>
        </div>
        <div className="flex items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept="video/*"
            multiple
            onChange={handleFileSelect}
            className="hidden"
          />
          <Button 
            variant="outline"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
          >
            <Upload className="w-4 h-4 mr-2" />
            Upload Video
          </Button>
          <Button 
            onClick={handleExtract}
            disabled={isExtracting || !selectedVideo}
          >
            {isExtracting ? (
              <>
                <RefreshCcw className="w-4 h-4 mr-2 animate-spin" />
                Extracting...
              </>
            ) : (
              <>
                <Scissors className="w-4 h-4 mr-2" />
                Extract Frames
              </>
            )}
          </Button>
        </div>
      </div>

      {uploading && (
        <Card className="mb-4">
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <Upload className="w-5 h-5 text-muted-foreground" />
              <Progress value={uploadProgress} className="flex-1" />
              <span className="text-sm text-muted-foreground w-12">{uploadProgress}%</span>
            </div>
          </CardContent>
        </Card>
      )}

      {isExtracting && (
        <Card className="mb-4">
          <CardContent className="py-4">
            <div className="flex items-center gap-4">
              <Scissors className="w-5 h-5 text-muted-foreground" />
              <Progress value={extractionProgress} className="flex-1" />
              <span className="text-sm text-muted-foreground w-32 text-right">
                {extractionFrames > 0 ? `${extractionFrames} frames` : `${extractionProgress}%`}
              </span>
            </div>
          </CardContent>
        </Card>
      )}

      {message && (
        <div className={cn(
          'mb-4 p-3 rounded-lg flex items-center gap-2 text-sm',
          message.type === 'success' ? 'bg-emerald-500/10 text-emerald-500 border border-emerald-500/20' : 'bg-destructive/10 text-destructive border border-destructive/20'
        )}>
          {message.type === 'success' ? <CheckCircle className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}
          {message.text}
        </div>
      )}

      <div className="flex gap-6 flex-1">
        {/* Video List */}
        <Card className="w-64 flex-shrink-0 flex flex-col">
          <CardHeader className="pb-3">
            <CardTitle className="text-base">Videos ({videos.length})</CardTitle>
          </CardHeader>
          <CardContent className="flex-1 overflow-y-auto">
            {videos.length === 0 ? (
              <div className="text-center py-8">
                <Film className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
                <p className="text-sm text-muted-foreground">
                  Upload videos to extract frames
                </p>
              </div>
            ) : (
              <div className="space-y-2">
                {videos.map((video) => (
                  <div
                    key={video.id}
                    className={cn(
                      'p-3 rounded-lg cursor-pointer transition-colors border',
                      selectedVideo?.id === video.id 
                        ? 'border-primary bg-primary/5' 
                        : 'border-transparent hover:bg-muted'
                    )}
                    onClick={() => {
                      setSelectedVideo(video)
                      setEndTime(video.duration)
                      setOutputName(`frames_${video.name.replace(/\.[^/.]+$/, '')}`)
                    }}
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-12 h-8 bg-muted rounded flex items-center justify-center overflow-hidden">
                        {video.thumbnail ? (
                          <img src={video.thumbnail} alt="" className="w-full h-full object-cover" />
                        ) : (
                          <Video className="w-4 h-4 text-muted-foreground" />
                        )}
                      </div>
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium truncate">{video.name}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatTime(video.duration)} • {video.fps}fps
                        </p>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </CardContent>
        </Card>

        {/* Video Player & Settings */}
        <div className="flex-1 flex flex-col min-w-0">
          {selectedVideo ? (
            <>
              {/* Video Player */}
              <Card className="mb-4">
                <CardContent className="p-4">
                  <div className="relative aspect-video bg-black rounded-lg overflow-hidden mb-4">
                    <video
                      ref={videoRef}
                      src={selectedVideo.url.startsWith('http') ? selectedVideo.url : `${apiUrl}${selectedVideo.url}`}
                      className="w-full h-full object-contain"
                      onTimeUpdate={handleTimeUpdate}
                      onEnded={() => setIsPlaying(false)}
                    />
                    {!isPlaying && (
                      <div 
                        className="absolute inset-0 flex items-center justify-center cursor-pointer bg-black/30"
                        onClick={togglePlayPause}
                      >
                        <div className="w-16 h-16 rounded-full bg-white/90 flex items-center justify-center">
                          <Play className="w-8 h-8 text-black ml-1" />
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Video Controls */}
                  <div className="space-y-3">
                    <div className="flex items-center gap-3">
                      <Button variant="outline" size="icon" onClick={() => seekTo(Math.max(0, currentTime - 5))}>
                        <SkipBack className="w-4 h-4" />
                      </Button>
                      <Button variant="outline" size="icon" onClick={togglePlayPause}>
                        {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                      </Button>
                      <Button variant="outline" size="icon" onClick={() => seekTo(Math.min(selectedVideo.duration, currentTime + 5))}>
                        <SkipForward className="w-4 h-4" />
                      </Button>
                      <div className="flex-1">
                        <Slider
                          value={[currentTime]}
                          onValueChange={([v]) => seekTo(v)}
                          min={0}
                          max={selectedVideo.duration}
                          step={0.1}
                        />
                      </div>
                      <span className="text-sm text-muted-foreground font-mono w-28 text-right">
                        {formatTime(currentTime)} / {formatTime(selectedVideo.duration)}
                      </span>
                    </div>

                    {extractionMode === 'manual' && (
                      <Button variant="outline" size="sm" onClick={addManualFrame}>
                        <ImageIcon className="w-4 h-4 mr-2" />
                        Mark Frame at {formatTime(currentTime)}
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* Extraction Settings */}
              <Card className="flex flex-col">
                <CardHeader className="pb-3">
                  <CardTitle className="text-base">Extraction Settings</CardTitle>
                  <CardDescription>
                    Configure how frames are extracted
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <Tabs value={extractionMode} onValueChange={(v) => setExtractionMode(v as ExtractionMode)}>
                    <TabsList className="mb-4">
                      <TabsTrigger value="interval">
                        <Timer className="w-4 h-4 mr-2" />
                        Interval
                      </TabsTrigger>
                      <TabsTrigger value="uniform">
                        <Grid3X3 className="w-4 h-4 mr-2" />
                        Uniform
                      </TabsTrigger>
                      <TabsTrigger value="keyframes">
                        <Layers className="w-4 h-4 mr-2" />
                        Keyframes
                      </TabsTrigger>
                      <TabsTrigger value="manual">
                        <ZoomIn className="w-4 h-4 mr-2" />
                        Manual
                      </TabsTrigger>
                    </TabsList>

                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div className="space-y-2">
                        <Label>Start Time</Label>
                        <div className="flex gap-2">
                          <Input
                            type="number"
                            value={startTime.toFixed(1)}
                            onChange={(e) => setStartTime(parseFloat(e.target.value) || 0)}
                            min={0}
                            max={selectedVideo.duration}
                            step={0.1}
                          />
                          <Button variant="outline" size="icon" onClick={() => setStartTime(currentTime)}>
                            <Clock className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                      <div className="space-y-2">
                        <Label>End Time</Label>
                        <div className="flex gap-2">
                          <Input
                            type="number"
                            value={endTime.toFixed(1)}
                            onChange={(e) => setEndTime(parseFloat(e.target.value) || selectedVideo.duration)}
                            min={0}
                            max={selectedVideo.duration}
                            step={0.1}
                          />
                          <Button variant="outline" size="icon" onClick={() => setEndTime(currentTime)}>
                            <Clock className="w-4 h-4" />
                          </Button>
                        </div>
                      </div>
                    </div>

                    <TabsContent value="interval" className="space-y-4 mt-0">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label>Extract every Nth frame</Label>
                          <span className="text-sm text-muted-foreground">
                            Every {frameInterval} frames ({(frameInterval / selectedVideo.fps).toFixed(2)}s apart)
                          </span>
                        </div>
                        <div className="flex gap-3 items-center">
                          <Slider
                            value={[frameInterval]}
                            onValueChange={([v]) => setFrameInterval(v)}
                            min={1}
                            max={300}
                            step={1}
                            className="flex-1"
                          />
                          <Input
                            type="number"
                            value={frameInterval}
                            onChange={(e) => setFrameInterval(Math.max(1, parseInt(e.target.value) || 1))}
                            min={1}
                            className="w-20"
                          />
                        </div>
                        <p className="text-xs text-muted-foreground">
                          N=1 extracts every frame, N=30 extracts 1 frame per second at 30fps
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="uniform" className="space-y-4 mt-0">
                      <div className="space-y-2">
                        <div className="flex items-center justify-between">
                          <Label>Number of Frames</Label>
                          <span className="text-sm text-muted-foreground">{uniformCount} frames</span>
                        </div>
                        <Slider
                          value={[uniformCount]}
                          onValueChange={([v]) => setUniformCount(v)}
                          min={10}
                          max={500}
                          step={10}
                        />
                        <p className="text-xs text-muted-foreground">
                          Frames will be evenly distributed across the selected time range
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="keyframes" className="space-y-4 mt-0">
                      <div className="p-4 bg-muted rounded-lg">
                        <p className="text-sm">
                          Automatically detect scene changes and extract keyframes.
                          This uses motion detection to find frames with significant changes.
                        </p>
                      </div>
                    </TabsContent>

                    <TabsContent value="manual" className="space-y-4 mt-0">
                      <div className="p-4 bg-muted rounded-lg">
                        <p className="text-sm mb-3">
                          Manually select frames by navigating the video and clicking &quot;Mark Frame&quot;.
                        </p>
                        {manualFrames.length > 0 ? (
                          <div className="flex flex-wrap gap-1">
                            {manualFrames.map((frame) => (
                              <span 
                                key={frame} 
                                className="px-2 py-0.5 bg-background rounded text-xs cursor-pointer hover:bg-destructive/20"
                                onClick={() => setManualFrames(prev => prev.filter(f => f !== frame))}
                              >
                                Frame {frame}
                              </span>
                            ))}
                          </div>
                        ) : (
                          <p className="text-xs text-muted-foreground">No frames selected</p>
                        )}
                      </div>
                    </TabsContent>
                  </Tabs>

                  <div className="mt-4 space-y-4">
                    <div className="space-y-2">
                      <Label>Extract Into</Label>
                      <Select value={existingDatasetId} onValueChange={setExistingDatasetId}>
                        <SelectTrigger>
                          <SelectValue placeholder="New dataset" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="new">New dataset</SelectItem>
                          {datasets.map(d => (
                            <SelectItem key={d.id} value={d.id}>{d.name}</SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    {existingDatasetId === 'new' && (
                    <div className="space-y-2">
                      <Label>Output Dataset Name</Label>
                      <Input
                        value={outputName}
                        onChange={(e) => setOutputName(e.target.value)}
                        placeholder="Dataset name"
                      />
                    </div>
                    )}

                    <div className="p-4 bg-muted/50 rounded-lg">
                      <div className="flex items-center justify-between text-sm">
                        <span>Estimated Frames:</span>
                        <span className="font-medium">{calculateEstimatedFrames()}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm mt-1">
                        <span>Duration:</span>
                        <span className="font-medium">{formatTime(endTime - startTime)}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm mt-1">
                        <span>Resolution:</span>
                        <span className="font-medium">{selectedVideo.width}×{selectedVideo.height}</span>
                      </div>
                      <div className="flex items-center justify-between text-sm mt-1">
                        <span>Video FPS:</span>
                        <span className="font-medium">{selectedVideo.fps} fps</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          ) : (
            <div className="flex-1 flex flex-col items-center justify-center text-center">
              <div className="w-20 h-20 rounded-full bg-muted flex items-center justify-center mb-6">
                <Video className="w-10 h-10 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-semibold text-foreground mb-2">No Video Selected</h3>
              <p className="text-muted-foreground max-w-md mb-6">
                Upload a video file to extract frames for your training dataset
              </p>
              <Button onClick={() => fileInputRef.current?.click()}>
                <Upload className="w-4 h-4 mr-2" />
                Upload Video
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
