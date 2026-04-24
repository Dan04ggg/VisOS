'use client'

import { useEffect } from 'react'
import { cn } from '@/lib/utils'
import { VisosLogo } from "@/components/ui/visos-logo"
import {
  Settings,
  Database,
  BarChart3,
  Images,
  ArrowUpDown,
  Copy,
  PenTool,
  Tag,
  Sparkles,
  Film,
  Scissors,
  ArrowLeftRight,
  GitMerge,
  FileCode,
  Cpu,
  Bot,
  List,
  Activity,
  BarChart2,
  Archive,
  ScanSearch,
  FlaskConical,
  X,
  type LucideIcon,
} from 'lucide-react'
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip'
import type { ViewType, Dataset } from '@/app/page'

interface SidebarProps {
  activeView: ViewType
  setActiveView: (view: ViewType) => void
  selectedDataset: Dataset | null
  isOpen?: boolean
  onClose?: () => void
}

interface NavItem {
  id: ViewType
  label: string
  icon: LucideIcon
  description: string
}

interface NavGroup {
  label: string
  items: NavItem[]
}

const navGroups: NavGroup[] = [
  {
    label: 'Data',
    items: [
      {
        id: 'datasets',
        label: 'Datasets',
        icon: Database,
        description: 'Load and manage your image collections. Upload a ZIP file or browse your local folders.',
      },
      {
        id: 'dashboard',
        label: 'Dashboard',
        icon: BarChart3,
        description: 'View statistics, class distributions, and insights for your active dataset.',
      },
      {
        id: 'gallery',
        label: 'Gallery',
        icon: Images,
        description: 'Browse all images in a visual grid. Click any image to open it in the annotator.',
      },
      {
        id: 'sorting',
        label: 'Sort & Filter',
        icon: ArrowUpDown,
        description: 'Review images one-by-one using keyboard shortcuts to quickly organise your data.',
      },
      {
        id: 'duplicate-detection',
        label: 'Duplicates',
        icon: Copy,
        description: 'Automatically find and remove duplicate or near-identical images from your dataset.',
      },
    ],
  },
  {
    label: 'Annotate',
    items: [
      {
        id: 'annotate',
        label: 'Annotate',
        icon: PenTool,
        description: 'Draw bounding boxes and labels on images to teach the AI what objects to look for.',
      },
      {
        id: 'classes',
        label: 'Classes',
        icon: Tag,
        description: 'Define the object categories your AI will learn to detect (e.g. "car", "person").',
      },
    ],
  },
  {
    label: 'Process',
    items: [
      {
        id: 'split',
        label: 'Train / Val / Test',
        icon: Scissors,
        description: 'Divide your dataset into training, validation, and test sets for proper model evaluation.',
      },
      {
        id: 'augmentation',
        label: 'Augmentation',
        icon: Sparkles,
        description: 'Multiply your training data by automatically applying flips, rotations, and colour changes.',
      },
      {
        id: 'video-extraction',
        label: 'Video Frames',
        icon: Film,
        description: 'Extract individual frames from video files to use as training images.',
      },
      {
        id: 'convert',
        label: 'Convert Format',
        icon: ArrowLeftRight,
        description: 'Convert your dataset between formats — YOLO, COCO, Pascal VOC, and more.',
      },
      {
        id: 'merge',
        label: 'Merge',
        icon: GitMerge,
        description: 'Combine two or more separate datasets into one unified dataset.',
      },
      {
        id: 'yaml-wizard',
        label: 'YAML Config',
        icon: FileCode,
        description: 'Generate the configuration file required to start model training.',
      },
    ],
  },
  {
    label: 'Train',
    items: [
      {
        id: 'training',
        label: 'Training',
        icon: Cpu,
        description: 'Train an AI model on your annotated dataset with real-time loss and accuracy tracking.',
      },
      {
        id: 'inference',
        label: 'Inference',
        icon: ScanSearch,
        description: 'Test trained models on images, video files, or live webcam with tracking and ReID.',
      },
      {
        id: 'evaluate',
        label: 'Evaluate',
        icon: FlaskConical,
        description: 'Head-to-head model evaluation — compare mAP, precision, recall across multiple models.',
      },
      {
        id: 'models',
        label: 'Models',
        icon: Bot,
        description: 'View and manage your trained models. Run inference to test their performance.',
      },
      {
        id: 'batch-jobs',
        label: 'Batch Jobs',
        icon: List,
        description: 'Monitor and manage long-running background operations like export or augmentation.',
      },
    ],
  },
  {
    label: 'Analyze',
    items: [
      {
        id: 'health',
        label: 'Health Check',
        icon: Activity,
        description: 'Scan your dataset for quality issues such as missing labels or corrupted images.',
      },
      {
        id: 'compare',
        label: 'Compare',
        icon: BarChart2,
        description: 'Side-by-side statistical comparison between two or more datasets.',
      },
      {
        id: 'snapshots',
        label: 'Snapshots',
        icon: Archive,
        description: 'Save a snapshot of your dataset at any point and restore it later if needed.',
      },
    ],
  },
]

export function Sidebar({ activeView, setActiveView, selectedDataset, isOpen = false, onClose }: SidebarProps) {
  // Close sidebar on Escape key on mobile
  useEffect(() => {
    const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') onClose?.() }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  const handleNavigate = (view: ViewType) => {
    setActiveView(view)
    onClose?.()
  }

  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/50 md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside className={cn(
        'relative flex flex-col bg-sidebar border-r border-sidebar-border overflow-hidden select-none shrink-0',
        // Desktop: always visible, fixed width
        'md:w-56 md:h-full md:static md:translate-x-0 md:z-auto',
        // Mobile: overlay slide-in
        'max-md:fixed max-md:inset-y-0 max-md:left-0 max-md:w-72 max-md:z-50 max-md:h-full max-md:transition-transform max-md:duration-300',
        isOpen ? 'max-md:translate-x-0' : 'max-md:-translate-x-full',
      )}>

      {/* Ambient glow */}
      <div className="ambient-orb w-48 h-48 bg-primary -top-16 -left-16 absolute" />

      {/* Brand */}
      <div className="relative px-5 pt-5 pb-4 flex items-center justify-between">
        <VisosLogo size={100} showText={true} />
        {/* Mobile close button */}
        <button
          className="md:hidden p-1.5 rounded-lg text-sidebar-foreground/60 hover:text-sidebar-foreground hover:bg-sidebar-accent transition-colors"
          onClick={onClose}
          aria-label="Close menu"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Active dataset pill */}
      {selectedDataset && (
        <div className="mx-3 mb-3 px-3 py-2.5 rounded-xl border border-primary/20 bg-primary/8 relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-transparent pointer-events-none" />
          <p className="text-[9px] font-mono uppercase tracking-[0.18em] text-primary/60 mb-1">
            Active Dataset
          </p>
          <p className="text-xs font-semibold text-sidebar-foreground truncate leading-tight">
            {selectedDataset.name}
          </p>
          <div className="flex items-center gap-2 mt-1.5">
            <span className="text-[9px] px-1.5 py-0.5 bg-primary/15 text-primary rounded-md font-mono tracking-wider border border-primary/20">
              {selectedDataset.format?.toUpperCase()}
            </span>
            <span className="text-[9px] font-mono text-muted-foreground">
              {selectedDataset.num_images?.toLocaleString()} images
            </span>
          </div>
        </div>
      )}

      {/* Divider */}
      <div className="h-px mx-3 bg-sidebar-border mb-2" />

      {/* Navigation */}
      <nav className="flex-1 overflow-y-auto px-2 pb-2 space-y-4">
        {navGroups.map((group) => (
          <div key={group.label}>
            <p className="px-2 mb-1 text-[9px] font-mono font-semibold uppercase tracking-[0.2em] text-muted-foreground/35 select-none">
              {group.label}
            </p>
            <div className="space-y-px">
              {group.items.map((item) => {
                const isActive = activeView === item.id
                const Icon = item.icon
                return (
                  <Tooltip key={item.id} delayDuration={700}>
                    <TooltipTrigger asChild>
                      <button
                        onClick={() => handleNavigate(item.id)}
                        className={cn(
                          'group relative w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left',
                          'transition-all duration-150',
                          isActive
                            ? 'bg-primary/10 text-primary'
                            : 'text-sidebar-foreground/55 hover:text-sidebar-foreground hover:bg-sidebar-accent'
                        )}
                      >
                        {isActive && (
                          <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-primary rounded-r-full" />
                        )}
                        <Icon
                          className={cn(
                            'w-3.5 h-3.5 shrink-0 transition-colors',
                            isActive
                              ? 'text-primary'
                              : 'text-muted-foreground/50 group-hover:text-sidebar-foreground/70'
                          )}
                          strokeWidth={1.75}
                        />
                        <span className={cn(
                          'text-[12px] leading-none transition-colors',
                          isActive ? 'font-semibold' : 'font-medium'
                        )}>
                          {item.label}
                        </span>
                      </button>
                    </TooltipTrigger>
                    <TooltipContent side="right" sideOffset={10} className="max-w-[210px] py-2.5 px-3">
                      <p className="font-semibold text-[11px] mb-0.5">{item.label}</p>
                      <p className="text-[11px] text-background/65 leading-snug">{item.description}</p>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
          </div>
        ))}
      </nav>

      {/* Footer — Settings */}
      <div className="px-2 py-2 border-t border-sidebar-border">
        <Tooltip delayDuration={700}>
          <TooltipTrigger asChild>
            <button
              onClick={() => handleNavigate('settings')}
              className={cn(
                'group relative w-full flex items-center gap-2.5 px-2.5 py-1.5 rounded-lg text-left',
                'transition-all duration-150',
                activeView === 'settings'
                  ? 'bg-primary/10 text-primary'
                  : 'text-sidebar-foreground/55 hover:text-sidebar-foreground hover:bg-sidebar-accent'
              )}
            >
              {activeView === 'settings' && (
                <span className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-4 bg-primary rounded-r-full" />
              )}
              <Settings
                className={cn(
                  'w-3.5 h-3.5 shrink-0',
                  activeView === 'settings'
                    ? 'text-primary'
                    : 'text-muted-foreground/50 group-hover:text-sidebar-foreground/70'
                )}
                strokeWidth={1.75}
              />
              <span className={cn(
                'text-[12px] leading-none',
                activeView === 'settings' ? 'font-semibold' : 'font-medium'
              )}>
                Settings
              </span>
            </button>
          </TooltipTrigger>
          <TooltipContent side="right" sideOffset={10} className="max-w-[210px] py-2.5 px-3">
            <p className="font-semibold text-[11px] mb-0.5">Settings</p>
            <p className="text-[11px] text-background/65 leading-snug">Configure API connection, file paths, GPU device, and appearance.</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </aside>
    </>
  )
}
