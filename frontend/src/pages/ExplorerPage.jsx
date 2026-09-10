import React, { useState } from 'react';
import VignetteBloomCanvas from '../components/VignetteBloomCanvas';
import { Compass, Eye, Layers, Sliders, Sparkles, MapPin, Grid, Maximize2 } from 'lucide-react';

const EXPLORER_DATASETS = [
  {
    id: 'exp-1',
    title: 'Rotterdam Commercial Harbor',
    location: '51.9244° N, 4.4777° E',
    category: 'Port Infrastructure',
    resolution: '0.5m High Resolution',
    image: 'https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=1200&q=80',
    description: 'Busiest port in Europe showcasing heavy container shipping logistics, dock yards, and industrial coastal perimeter.'
  },
  {
    id: 'exp-2',
    title: 'Kansas Circular Irrigation Fields',
    location: '38.5266° N, 96.7265° W',
    category: 'Center-Pivot Agriculture',
    resolution: '10m Sentinel-2 MSI',
    image: 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80',
    description: 'Distinctive circular cropland geometry produced by center-pivot irrigation rigs across the Great Plains.'
  },
  {
    id: 'exp-3',
    title: 'Great Barrier Reef Shoreline',
    location: '16.9186° S, 145.7781° E',
    category: 'Coastal Ecology',
    resolution: '3m PlanetScope',
    image: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80',
    description: 'Shallow marine coral formations and tropical vegetation boundaries sensitive to sea level & temperature shifts.'
  }
];

export default function ExplorerPage() {
  const [selectedDataset, setSelectedDataset] = useState(EXPLORER_DATASETS[0]);
  const [renderMode, setRenderMode] = useState('mosaic');
  const [cellSize, setCellSize] = useState(16);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 lg:p-8 font-sans">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 pb-4 border-b border-slate-800">
          <div>
            <div className="flex items-center gap-2 text-sky-400 text-xs font-semibold uppercase tracking-wider mb-1">
              <Compass className="w-4 h-4" />
              <span>Geospatial Sandbox</span>
            </div>
            <h1 className="text-3xl font-extrabold text-white">Satellite Imagery Explorer</h1>
          </div>
          <p className="text-xs text-slate-400 max-w-md mt-2 md:mt-0">
            Compare original satellite optical telemetry against 21st.dev Vignette Bloom raster primitives in real-time.
          </p>
        </div>

        {/* Dataset Selector Tabs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {EXPLORER_DATASETS.map((ds) => (
            <button
              key={ds.id}
              onClick={() => setSelectedDataset(ds)}
              className={`p-4 rounded-2xl border text-left transition-all ${
                selectedDataset.id === ds.id 
                  ? 'bg-sky-500/10 border-sky-400 text-sky-200 shadow-lg shadow-sky-500/10'
                  : 'bg-slate-900 border-slate-800 text-slate-300 hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs text-sky-400 font-mono mb-1">
                <span>{ds.category}</span>
                <span className="flex items-center gap-1"><MapPin className="w-3 h-3" /> {ds.location}</span>
              </div>
              <h3 className="font-bold text-base text-white mb-1">{ds.title}</h3>
              <p className="text-xs text-slate-400 line-clamp-2">{ds.description}</p>
            </button>
          ))}
        </div>

        {/* Side-by-Side Comparison Container */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Panel 1: Original Telemetry */}
          <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-sm text-sky-300 flex items-center gap-2">
                <Eye className="w-4 h-4 text-sky-400" />
                Raw Optical Telemetry (RGB)
              </h3>
              <span className="text-xs font-mono text-slate-400">{selectedDataset.resolution}</span>
            </div>
            <div className="relative h-[440px] rounded-2xl overflow-hidden bg-black border border-slate-800">
              <img src={selectedDataset.image} alt={selectedDataset.title} className="w-full h-full object-cover" />
            </div>
          </div>

          {/* Panel 2: Vignette Bloom Raster View */}
          <div className="bg-slate-900/80 border border-sky-500/30 rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-bold text-sm text-sky-300 flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-sky-400" />
                Vignette Bloom Primitive Filter
              </h3>
              <div className="flex items-center gap-2">
                <select
                  value={renderMode}
                  onChange={(e) => setRenderMode(e.target.value)}
                  className="bg-slate-950 border border-slate-700 rounded-lg px-2.5 py-1 text-xs text-sky-300 font-mono"
                >
                  <option value="mosaic">Mosaic</option>
                  <option value="characters">ASCII Characters</option>
                  <option value="matrix">Matrix Green Rain</option>
                  <option value="hexagons">Hexagon Honeycomb</option>
                  <option value="dither">Dither Primitive</option>
                  <option value="triangles">Low-Poly Triangles</option>
                </select>
              </div>
            </div>

            <div className="relative h-[440px] rounded-2xl overflow-hidden border border-slate-800">
              <VignetteBloomCanvas
                config={{
                  renderMode: renderMode,
                  cellSize: cellSize,
                  pfx: { vignette: { enabled: true, intensity: 45 }, bloom: { enabled: true, intensity: 30 } },
                  animated: true,
                  animStyle: 'wave'
                }}
                imageSrc={selectedDataset.image}
                width={600}
                height={440}
                className="w-full h-full"
              />
            </div>
          </div>

        </div>

      </div>
    </div>
  );
}
