import React, { useState } from 'react';
import VignetteBloomCanvas, { DEFAULT_VIGNETTE_PARAMS } from '../components/VignetteBloomCanvas';
import { 
  Sparkles, MessageSquare, ArrowRight, Play, Sliders, Layers, 
  Eye, Zap, Shield, Cpu, Image as ImageIcon, CheckCircle 
} from 'lucide-react';

const DEMO_PRESET_IMAGES = [
  {
    name: "Ref-008 WebP (Default)",
    url: "/ascii-editor/demos/generated/ref-008.webp"
  },
  {
    name: "Rotterdam Harbor (Urban)",
    url: "https://images.unsplash.com/photo-1578575437130-527eed3abbec?auto=format&fit=crop&w=1200&q=80"
  },
  {
    name: "Kansas Crop Fields (Agriculture)",
    url: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?auto=format&fit=crop&w=1200&q=80"
  },
  {
    name: "Great Barrier Reef (Coastal)",
    url: "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80"
  }
];

export default function LandingPage({ onNavigateToChat }) {
  const [canvasConfig, setCanvasConfig] = useState(DEFAULT_VIGNETTE_PARAMS);
  const [selectedImage, setSelectedImage] = useState(DEMO_PRESET_IMAGES[0].url);
  const [showControlPanel, setShowControlPanel] = useState(true);

  const RENDER_MODES = [
    "mosaic", "characters", "matrix", "hexagons", "dither", 
    "dots", "triangles", "lines", "hexdump", "lego", "hatch", "halfblocks"
  ];

  const ANIM_STYLES = ["wave", "pulse", "shimmer", "ripple", "flicker"];

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans selection:bg-sky-500 selection:text-white">
      
      {/* Background Glow Orbs */}
      <div className="absolute top-1/4 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[700px] h-[700px] bg-sky-500/10 rounded-full blur-[140px] pointer-events-none" />
      <div className="absolute bottom-10 right-10 w-[500px] h-[500px] bg-indigo-500/10 rounded-full blur-[140px] pointer-events-none" />

      {/* Hero Section */}
      <section className="relative pt-12 pb-20 px-4 lg:px-8 max-w-7xl mx-auto">
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-sky-500/10 border border-sky-500/30 text-sky-300 text-xs font-semibold mb-6 shadow-lg shadow-sky-500/10 backdrop-blur-md">
            <Sparkles className="w-3.5 h-3.5 text-sky-400 animate-pulse" />
            <span>Branch: ayushFRONTEND &bull; 21st.dev Vignette Bloom Canvas</span>
          </div>

          <h1 className="text-4xl sm:text-6xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-sky-200 via-indigo-100 to-white mb-6 leading-tight">
            SatQuery AI &mdash; Advanced Geospatial Assistant
          </h1>

          <p className="text-lg text-slate-300 font-normal leading-relaxed mb-8">
            Experience next-generation satellite image visual question answering (VQA), multi-spectral scene description, and real-time ASCII Vignette Bloom raster processing.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-4">
            <button
              onClick={onNavigateToChat}
              className="group relative inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-sky-500 via-indigo-600 to-cyan-500 text-white font-bold text-sm shadow-xl shadow-sky-500/25 hover:shadow-sky-500/40 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200"
            >
              <MessageSquare className="w-5 h-5 text-sky-100 group-hover:rotate-12 transition-transform" />
              <span>Launch AI Satellite Chatbot</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <button
              onClick={() => setShowControlPanel(!showControlPanel)}
              className="inline-flex items-center gap-2.5 px-6 py-4 rounded-2xl bg-slate-900/90 border border-slate-800 text-slate-200 hover:text-sky-300 hover:border-sky-500/40 font-semibold text-sm transition-all"
            >
              <Sliders className="w-4 h-4 text-sky-400" />
              <span>{showControlPanel ? 'Hide Controls' : 'Tweak Vignette Bloom'}</span>
            </button>
          </div>
        </div>

        {/* Main Vignette Bloom Canvas Showcase */}
        <div className="relative rounded-3xl p-2 bg-gradient-to-b from-sky-500/20 via-indigo-500/10 to-slate-900/50 border border-sky-500/30 shadow-2xl backdrop-blur-xl">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
            
            {/* Canvas Container */}
            <div className={`${showControlPanel ? 'lg:col-span-8' : 'lg:col-span-12'} transition-all duration-300`}>
              <div className="relative w-full h-[520px] rounded-2xl overflow-hidden shadow-inner">
                <VignetteBloomCanvas
                  config={canvasConfig}
                  imageSrc={selectedImage}
                  width={showControlPanel ? 760 : 1180}
                  height={520}
                  className="w-full h-full"
                />
                
                {/* Mode Overlay Tag */}
                <div className="absolute top-4 left-4 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/80 border border-sky-500/30 text-xs font-mono text-sky-300 backdrop-blur-md shadow-lg">
                  <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
                  <span>Render: {canvasConfig.renderMode.toUpperCase()}</span>
                  <span>|</span>
                  <span className="text-indigo-300">Anim: {canvasConfig.animStyle}</span>
                </div>
              </div>
            </div>

            {/* Interactive Controls Drawer */}
            {showControlPanel && (
              <div className="lg:col-span-4 bg-slate-900/90 border border-slate-800/80 rounded-2xl p-5 flex flex-col justify-between overflow-y-auto max-h-[520px] custom-scrollbar">
                <div>
                  <div className="flex items-center justify-between pb-3 border-b border-slate-800 mb-4">
                    <h3 className="font-bold text-sm text-sky-300 flex items-center gap-2">
                      <Sliders className="w-4 h-4 text-sky-400" />
                      Vignette Bloom Parameters
                    </h3>
                    <span className="text-[10px] font-mono text-slate-400">JSON Spec Live</span>
                  </div>

                  {/* Image Selector */}
                  <div className="mb-4">
                    <label className="block text-xs font-medium text-slate-300 mb-2 flex items-center gap-1.5">
                      <ImageIcon className="w-3.5 h-3.5 text-sky-400" /> Source Photo
                    </label>
                    <select
                      value={selectedImage}
                      onChange={(e) => setSelectedImage(e.target.value)}
                      className="w-full bg-slate-950 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500"
                    >
                      {DEMO_PRESET_IMAGES.map((img, idx) => (
                        <option key={idx} value={img.url}>{img.name}</option>
                      ))}
                    </select>
                  </div>

                  {/* Render Mode */}
                  <div className="mb-4">
                    <label className="block text-xs font-medium text-slate-300 mb-2">Render Mode</label>
                    <div className="grid grid-cols-3 gap-1.5">
                      {RENDER_MODES.map((mode) => (
                        <button
                          key={mode}
                          onClick={() => setCanvasConfig({ ...canvasConfig, renderMode: mode })}
                          className={`px-2 py-1.5 rounded-lg text-[11px] font-mono transition-all ${
                            canvasConfig.renderMode === mode
                              ? 'bg-sky-500 text-white font-bold shadow-md shadow-sky-500/20'
                              : 'bg-slate-950 text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                          }`}
                        >
                          {mode}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Cell Size Slider */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Cell Size</span>
                      <span className="font-mono text-sky-400">{canvasConfig.cellSize}px</span>
                    </div>
                    <input
                      type="range"
                      min="6"
                      max="32"
                      step="2"
                      value={canvasConfig.cellSize}
                      onChange={(e) => setCanvasConfig({ ...canvasConfig, cellSize: parseInt(e.target.value) })}
                      className="w-full accent-sky-400 bg-slate-950 rounded-lg h-1.5 cursor-pointer"
                    />
                  </div>

                  {/* Vignette Intensity Slider */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Vignette Intensity</span>
                      <span className="font-mono text-sky-400">{canvasConfig.pfx.vignette.intensity}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={canvasConfig.pfx.vignette.intensity}
                      onChange={(e) => setCanvasConfig({
                        ...canvasConfig,
                        pfx: {
                          ...canvasConfig.pfx,
                          vignette: { ...canvasConfig.pfx.vignette, intensity: parseInt(e.target.value) }
                        }
                      })}
                      className="w-full accent-sky-400 bg-slate-950 rounded-lg h-1.5 cursor-pointer"
                    />
                  </div>

                  {/* Bloom Intensity Slider */}
                  <div className="mb-4">
                    <div className="flex justify-between text-xs text-slate-300 mb-1">
                      <span>Bloom Glow Intensity</span>
                      <span className="font-mono text-sky-400">{canvasConfig.pfx.bloom.intensity}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={canvasConfig.pfx.bloom.intensity}
                      onChange={(e) => setCanvasConfig({
                        ...canvasConfig,
                        pfx: {
                          ...canvasConfig.pfx,
                          bloom: { ...canvasConfig.pfx.bloom, intensity: parseInt(e.target.value) }
                        }
                      })}
                      className="w-full accent-sky-400 bg-slate-950 rounded-lg h-1.5 cursor-pointer"
                    />
                  </div>

                  {/* Animation Style */}
                  <div className="mb-4">
                    <label className="block text-xs font-medium text-slate-300 mb-2">Animation Motion</label>
                    <div className="flex flex-wrap gap-1.5">
                      {ANIM_STYLES.map((style) => (
                        <button
                          key={style}
                          onClick={() => setCanvasConfig({ ...canvasConfig, animStyle: style })}
                          className={`px-2.5 py-1 rounded-lg text-[11px] font-mono transition-all ${
                            canvasConfig.animStyle === style
                              ? 'bg-indigo-600 text-white font-bold'
                              : 'bg-slate-950 text-slate-400 hover:text-slate-200'
                          }`}
                        >
                          {style}
                        </button>
                      ))}
                    </div>
                  </div>

                </div>

                {/* Quick CTA inside Panel */}
                <button
                  onClick={onNavigateToChat}
                  className="w-full mt-4 py-2.5 rounded-xl bg-sky-500/20 hover:bg-sky-500/30 border border-sky-400/40 text-sky-300 font-semibold text-xs transition-all flex items-center justify-center gap-2"
                >
                  <MessageSquare className="w-4 h-4" />
                  <span>Analyze Satellite Images in AI Chat</span>
                </button>
              </div>
            )}

          </div>
        </div>

      </section>

      {/* Capabilities & Feature Grid */}
      <section className="py-16 px-4 lg:px-8 max-w-7xl mx-auto border-t border-slate-800/60">
        <div className="text-center max-w-2xl mx-auto mb-12">
          <h2 className="text-3xl font-extrabold text-white mb-4">
            Engineered for High-Resolution Remote Sensing
          </h2>
          <p className="text-slate-400 text-sm">
            Powered by Qwen2-VL-7B-Instruct fine-tuning, GeoTIFF multispectral band preprocessing, and deterministic query routing.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          
          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 hover:border-sky-500/30 transition-all group">
            <div className="w-12 h-12 rounded-xl bg-sky-500/10 border border-sky-500/30 flex items-center justify-center text-sky-400 mb-4 group-hover:scale-110 transition-transform">
              <Cpu className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Qwen2-VL 7B Vision Backbone</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Variable-resolution input support with dynamic patch size allocation. Quantized to 4-bit for rapid local inference.
            </p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 hover:border-sky-500/30 transition-all group">
            <div className="w-12 h-12 rounded-xl bg-indigo-500/10 border border-indigo-500/30 flex items-center justify-center text-indigo-400 mb-4 group-hover:scale-110 transition-transform">
              <Layers className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">GeoTIFF Band Extraction</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Preserves exact spatial CRS metadata and constructs balanced RGB representations for multi-spectral remote sensing layers.
            </p>
          </div>

          <div className="bg-slate-900/60 border border-slate-800 rounded-2xl p-6 hover:border-sky-500/30 transition-all group">
            <div className="w-12 h-12 rounded-xl bg-cyan-500/10 border border-cyan-500/30 flex items-center justify-center text-cyan-400 mb-4 group-hover:scale-110 transition-transform">
              <Zap className="w-6 h-6" />
            </div>
            <h3 className="text-lg font-bold text-white mb-2">Deterministic VQA Router</h3>
            <p className="text-slate-400 text-xs leading-relaxed">
              Automatically routes scene description prompts to captioning pipelines and specific queries to visual question answering modes.
            </p>
          </div>

        </div>
      </section>

    </div>
  );
}
