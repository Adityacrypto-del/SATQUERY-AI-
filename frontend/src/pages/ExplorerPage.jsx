import React, { useState } from 'react';
import VignetteBloomCanvas from '../components/VignetteBloomCanvas';
import { Compass, Eye, Sparkles, MapPin } from 'lucide-react';

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
  const [cellSize] = useState(16);

  return (
    <div className="min-h-[calc(100vh-65px)] bg-[#08090a] text-white p-4 lg:p-8 font-sans selection:bg-white selection:text-black">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="flex flex-col md:flex-row md:items-center justify-between mb-8 pb-4 border-b border-white/[0.08]">
          <div>
            <div className="flex items-center gap-2 text-white/50 text-xs font-semibold uppercase tracking-wider mb-1">
              <Compass className="w-4 h-4 text-white" />
              <span>Geospatial Sandbox</span>
            </div>
            <h1 className="text-3xl font-bold text-white">Satellite Imagery Explorer</h1>
          </div>
          <p className="text-xs text-white/50 max-w-md mt-2 md:mt-0">
            Compare original satellite optical telemetry against 21st.dev Vignette Bloom raster primitives in real-time.
          </p>
        </div>

        {/* Dataset Selector Tabs */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
          {EXPLORER_DATASETS.map((ds) => (
            <button
              key={ds.id}
              onClick={() => setSelectedDataset(ds)}
              className={`p-4 rounded-2xl border text-left transition-all cursor-pointer ${
                selectedDataset.id === ds.id 
                  ? 'bg-white text-black border-white shadow-lg'
                  : 'bg-white/[0.02] border-white/[0.08] text-white/80 hover:border-white/25 hover:bg-white/[0.05]'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono mb-1">
                <span className={selectedDataset.id === ds.id ? 'text-black font-semibold' : 'text-white/60'}>{ds.category}</span>
                <span className={`flex items-center gap-1 ${selectedDataset.id === ds.id ? 'text-neutral-700' : 'text-white/40'}`}><MapPin className="w-3 h-3" /> {ds.location}</span>
              </div>
              <h3 className={`font-bold text-base mb-1 ${selectedDataset.id === ds.id ? 'text-black' : 'text-white'}`}>{ds.title}</h3>
              <p className={`text-xs line-clamp-2 ${selectedDataset.id === ds.id ? 'text-neutral-700' : 'text-white/45'}`}>{ds.description}</p>
            </button>
          ))}
        </div>

        {/* Side-by-Side Comparison Container */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Panel 1: Original Telemetry */}
          <div className="bg-white/[0.02] border border-white/[0.08] rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-sm text-white flex items-center gap-2">
                <Eye className="w-4 h-4 text-white" />
                Raw Optical Telemetry (RGB)
              </h3>
              <span className="text-xs font-mono text-white/40">{selectedDataset.resolution}</span>
            </div>
            <div className="relative h-[440px] rounded-2xl overflow-hidden bg-black border border-white/10">
              <img src={selectedDataset.image} alt={selectedDataset.title} className="w-full h-full object-cover" />
            </div>
          </div>

          {/* Panel 2: Vignette Bloom Raster View */}
          <div className="bg-white/[0.02] border border-white/[0.12] rounded-3xl p-5 shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-sm text-white flex items-center gap-2">
                <Sparkles className="w-4 h-4 text-white" />
                Vignette Bloom Primitive Filter
              </h3>
              <div className="flex items-center gap-2">
                <select
                  value={renderMode}
                  onChange={(e) => setRenderMode(e.target.value)}
                  className="bg-black border border-white/20 rounded-lg px-2.5 py-1 text-xs text-white font-mono"
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

            <div className="relative h-[440px] rounded-2xl overflow-hidden border border-white/10 bg-black">
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
