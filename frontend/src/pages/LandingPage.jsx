import React, { useState, useRef, useEffect } from 'react';
import VignetteBloomCanvas, { DEFAULT_VIGNETTE_PARAMS } from '../components/VignetteBloomCanvas';
import {
  Sparkles, MessageSquare, ArrowRight, Sliders, Layers,
  Zap, Cpu, Image as ImageIcon, ChevronDown, ChevronUp, RotateCcw
} from 'lucide-react';

const DEMO_IMAGES = [
  { name: 'Ref-008 (Default)', url: '/ascii-editor/demos/generated/ref-008.webp' },
  { name: 'Rotterdam Harbor',  url: 'https://images.unsplash.com/photo-1578575437130-527eed3abbec?w=1200&q=80' },
  { name: 'Pivot Cropland',    url: 'https://images.unsplash.com/photo-1500382017468-9049fed747ef?w=1200&q=80' },
  { name: 'Great Barrier Reef', url: 'https://images.unsplash.com/photo-1507525428034-b723cf961d3e?w=1200&q=80' },
  { name: 'City Aerial',        url: 'https://images.unsplash.com/photo-1477959858617-67f85cf4f1df?w=1200&q=80' },
];

const RENDER_MODES = [
  'mosaic', 'characters', 'matrix', 'hexagons', 'dither',
  'dots', 'triangles', 'lego', 'hatch', 'diagonal', 'disco',
  'contour', 'halfblocks', 'rings', 'stars', 'hearts', 'cross',
  'braille', 'hexdump', 'voxel', 'pixel', 'diamond', 'mixed', 'lines', 'bubbles',
];

const ANIM_STYLES = ['wave', 'pulse', 'shimmer', 'ripple', 'flicker'];

export default function LandingPage({ onNavigateToChat }) {
  const containerRef = useRef(null);
  const [canvasW, setCanvasW] = useState(900);
  const [canvasH] = useState(520);

  const [cfg, setCfg] = useState(DEFAULT_VIGNETTE_PARAMS);
  const [selectedImage, setSelectedImage] = useState(DEMO_IMAGES[0].url);
  const [showPanel, setShowPanel] = useState(true);
  const [panelTab, setPanelTab] = useState('render');

  // Responsive canvas width
  useEffect(() => {
    const obs = new ResizeObserver(entries => {
      for (const e of entries) {
        const avail = showPanel
          ? Math.floor(e.contentRect.width * 0.635)
          : e.contentRect.width - 32;
        setCanvasW(Math.max(320, avail));
      }
    });
    if (containerRef.current) obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, [showPanel]);

  const set = (patch) => setCfg(prev => ({ ...prev, ...patch }));
  const setPfx = (key, patch) =>
    setCfg(prev => ({ ...prev, pfx: { ...prev.pfx, [key]: { ...prev.pfx[key], ...patch } } }));

  const resetCfg = () => setCfg(DEFAULT_VIGNETTE_PARAMS);

  const Slider = ({ label, value, min, max, step = 1, onChange }) => (
    <div className="mb-3">
      <div className="flex justify-between text-[11px] text-slate-300 mb-1">
        <span>{label}</span>
        <span className="font-mono text-sky-400 tabular-nums">{value}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full accent-sky-400 cursor-pointer bg-slate-800" />
    </div>
  );

  const Toggle = ({ label, checked, onChange }) => (
    <button onClick={() => onChange(!checked)}
      className={`flex items-center justify-between w-full px-3 py-2 rounded-xl text-xs mb-1 transition-all border ${
        checked ? 'bg-sky-500/20 border-sky-400/40 text-sky-200' : 'bg-slate-950 border-slate-800 text-slate-400'
      }`}>
      <span>{label}</span>
      <div className={`w-8 h-4 rounded-full relative transition-colors ${checked ? 'bg-sky-500' : 'bg-slate-700'}`}>
        <div className={`absolute top-0.5 w-3 h-3 rounded-full bg-white transition-all ${checked ? 'left-4' : 'left-0.5'}`} />
      </div>
    </button>
  );

  return (
    <div className="relative min-h-screen bg-slate-950 text-slate-100 overflow-hidden font-sans selection:bg-sky-500 selection:text-white">

      {/* Ambient background orbs */}
      <div className="pointer-events-none fixed inset-0 z-0 overflow-hidden">
        <div className="absolute -top-32 left-1/2 -translate-x-1/2 w-[900px] h-[600px] bg-sky-500/8 rounded-full blur-[160px]" />
        <div className="absolute bottom-0 right-0 w-[600px] h-[500px] bg-indigo-600/7 rounded-full blur-[140px]" />
      </div>

      <div className="relative z-10 max-w-[1400px] mx-auto px-4 lg:px-8 pt-10 pb-20">

        {/* ── Hero Header ────────────────────────────────────────────────── */}
        <div className="text-center max-w-3xl mx-auto mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-sky-500/10 border border-sky-500/25 text-sky-300 text-xs font-semibold mb-5 shadow-lg shadow-sky-500/10 backdrop-blur-sm">
            <Sparkles className="w-3.5 h-3.5 animate-pulse" />
            <span>21st.dev Vignette Bloom · Canvas2D · ayushFRONTEND branch</span>
          </div>

          <h1 className="text-5xl sm:text-6xl font-extrabold tracking-tight leading-tight mb-5">
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-300 via-cyan-200 to-white">
              SatQuery&nbsp;AI
            </span>
            <br />
            <span className="text-slate-300 text-3xl sm:text-4xl font-bold">
              Advanced Geospatial Intelligence
            </span>
          </h1>

          <p className="text-base text-slate-400 leading-relaxed mb-8 max-w-2xl mx-auto">
            Real-time ASCII raster art engine over satellite imagery — powered by
            Canvas2D, Qwen2-VL-7B, and deterministic VQA routing.
          </p>

          <div className="flex flex-wrap items-center justify-center gap-3">
            <button onClick={onNavigateToChat}
              className="group inline-flex items-center gap-3 px-8 py-4 rounded-2xl bg-gradient-to-r from-sky-500 via-indigo-600 to-cyan-500 text-white font-bold text-sm shadow-xl shadow-sky-500/25 hover:shadow-sky-500/45 hover:scale-[1.02] active:scale-[0.98] transition-all duration-200">
              <MessageSquare className="w-5 h-5 group-hover:rotate-12 transition-transform" />
              Launch AI Satellite Chatbot
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <button onClick={() => setShowPanel(v => !v)}
              className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl bg-slate-900 border border-slate-800 hover:border-sky-500/40 text-slate-200 hover:text-sky-300 font-semibold text-sm transition-all">
              <Sliders className="w-4 h-4 text-sky-400" />
              {showPanel ? 'Hide' : 'Show'} Controls
              {showPanel ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
            </button>
            <button onClick={resetCfg}
              className="inline-flex items-center gap-2 px-4 py-4 rounded-2xl bg-slate-900 border border-slate-800 hover:border-rose-500/30 text-slate-400 hover:text-rose-300 transition-all">
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* ── Main Canvas + Controls ────────────────────────────────────── */}
        <div ref={containerRef} className="relative rounded-3xl p-[2px] bg-gradient-to-b from-sky-500/25 via-indigo-500/15 to-slate-900 border border-sky-500/25 shadow-2xl shadow-sky-500/10">
          <div className="rounded-[22px] bg-slate-900/80 backdrop-blur-2xl overflow-hidden">
            <div className={`flex flex-col lg:flex-row ${showPanel ? '' : ''}`}>

              {/* Canvas view */}
              <div className={`${showPanel ? 'lg:w-[63%]' : 'w-full'} relative transition-all duration-300`}>
                <VignetteBloomCanvas
                  config={cfg}
                  imageSrc={selectedImage}
                  width={canvasW}
                  height={canvasH}
                  className="w-full border-b lg:border-b-0 lg:border-r border-slate-800/60 !rounded-none"
                />

                {/* Info overlay */}
                <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-950/85 border border-sky-500/25 text-[11px] font-mono text-sky-300 backdrop-blur-md shadow-lg pointer-events-none">
                  <span className="w-2 h-2 rounded-full bg-sky-400 animate-ping" />
                  <span>{cfg.renderMode.toUpperCase()}</span>
                  <span className="opacity-50">|</span>
                  <span className="text-indigo-300">{cfg.animStyle}</span>
                  <span className="opacity-50">|</span>
                  <span className="text-emerald-300">{cfg.cellSize}px</span>
                </div>
              </div>

              {/* Controls Panel */}
              {showPanel && (
                <div className="lg:w-[37%] flex flex-col bg-slate-950/60 border-t lg:border-t-0 border-slate-800/60">

                  {/* Tab bar */}
                  <div className="flex border-b border-slate-800/60">
                    {[
                      { id: 'render', label: 'Render' },
                      { id: 'color',  label: 'Color' },
                      { id: 'pfx',    label: 'Post-FX' },
                      { id: 'anim',   label: 'Anim' },
                    ].map(t => (
                      <button key={t.id} onClick={() => setPanelTab(t.id)}
                        className={`flex-1 py-3 text-xs font-semibold transition-all border-b-2 ${
                          panelTab === t.id
                            ? 'text-sky-300 border-sky-400 bg-sky-500/5'
                            : 'text-slate-400 border-transparent hover:text-slate-200'
                        }`}>
                        {t.label}
                      </button>
                    ))}
                  </div>

                  <div className="flex-1 overflow-y-auto p-4 custom-scrollbar space-y-4">

                    {/* ── Render Tab ── */}
                    {panelTab === 'render' && (<>
                      {/* Image source */}
                      <div>
                        <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                          Source Photo
                        </label>
                        <select value={selectedImage} onChange={e => setSelectedImage(e.target.value)}
                          className="w-full bg-slate-900 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-sky-500">
                          {DEMO_IMAGES.map(img => (
                            <option key={img.url} value={img.url}>{img.name}</option>
                          ))}
                        </select>
                      </div>

                      {/* Render mode grid */}
                      <div>
                        <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                          Render Mode
                        </label>
                        <div className="grid grid-cols-4 gap-1">
                          {RENDER_MODES.map(m => (
                            <button key={m} onClick={() => set({ renderMode: m })}
                              className={`px-1 py-1.5 rounded-lg text-[10px] font-mono transition-all truncate ${
                                cfg.renderMode === m
                                  ? 'bg-sky-500 text-white font-bold shadow-md shadow-sky-500/25'
                                  : 'bg-slate-900 text-slate-400 hover:text-slate-100 hover:bg-slate-800'
                              }`}>
                              {m}
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* Background mode */}
                      <div>
                        <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                          Background Mode
                        </label>
                        <div className="grid grid-cols-4 gap-1">
                          {['solid', 'blurred', 'photo', 'none'].map(m => (
                            <button key={m} onClick={() => set({ bgMode: m })}
                              className={`py-1.5 rounded-lg text-[10px] font-mono transition-all ${
                                cfg.bgMode === m
                                  ? 'bg-indigo-600 text-white font-bold'
                                  : 'bg-slate-900 text-slate-400 hover:text-slate-100'
                              }`}>
                              {m}
                            </button>
                          ))}
                        </div>
                      </div>

                      <Slider label="Cell Size (px)" value={cfg.cellSize} min={4} max={32} step={2} onChange={v => set({ cellSize: v })} />
                      <Slider label="Coverage (%)" value={cfg.coverage} min={10} max={100} onChange={v => set({ coverage: v })} />
                      <Slider label="Edge Emphasis" value={cfg.edgeEmphasis} min={0} max={100} onChange={v => set({ edgeEmphasis: v })} />
                      <Toggle label="Invert Luminance" checked={cfg.invert} onChange={v => set({ invert: v })} />
                    </>)}

                    {/* ── Color Tab ── */}
                    {panelTab === 'color' && (<>
                      <Slider label="Brightness" value={cfg.brightness} min={-50} max={100} onChange={v => set({ brightness: v })} />
                      <Slider label="Contrast (%)" value={cfg.contrast} min={50} max={200} onChange={v => set({ contrast: v })} />
                      <Slider label="Saturation (%)" value={cfg.saturation} min={0} max={200} onChange={v => set({ saturation: v })} />
                      <Slider label="Grayscale (%)" value={cfg.grayscale} min={0} max={100} onChange={v => set({ grayscale: v })} />
                      <div className="mb-3">
                        <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2">
                          Tint Color
                        </label>
                        <div className="flex items-center gap-3">
                          <input type="color" value={cfg.tint}
                            onChange={e => set({ tint: e.target.value })}
                            className="w-10 h-8 rounded-lg cursor-pointer border-0 bg-transparent" />
                          <span className="font-mono text-xs text-slate-400">{cfg.tint}</span>
                        </div>
                      </div>
                      <Slider label="Tint Opacity (%)" value={cfg.tintOpacity} min={0} max={100} onChange={v => set({ tintOpacity: v })} />
                    </>)}

                    {/* ── Post-FX Tab ── */}
                    {panelTab === 'pfx' && (
                      <div className="space-y-1">
                        {Object.entries(cfg.pfx).map(([key, fx]) => (
                          <div key={key} className="bg-slate-900/60 rounded-xl p-3 border border-slate-800">
                            <Toggle
                              label={key.charAt(0).toUpperCase() + key.slice(1)}
                              checked={fx.enabled}
                              onChange={v => setPfx(key, { enabled: v })}
                            />
                            {fx.enabled && (
                              <div className="mt-2">
                                <Slider label="Intensity" value={fx.intensity} min={0} max={100}
                                  onChange={v => setPfx(key, { intensity: v })} />
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {/* ── Anim Tab ── */}
                    {panelTab === 'anim' && (<>
                      <Toggle label="Animated" checked={cfg.animated} onChange={v => set({ animated: v })} />
                      {cfg.animated && (<>
                        <div className="mt-3">
                          <label className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block mb-2">
                            Animation Style
                          </label>
                          <div className="grid grid-cols-3 gap-1.5">
                            {ANIM_STYLES.map(s => (
                              <button key={s} onClick={() => set({ animStyle: s })}
                                className={`py-2 rounded-xl text-xs font-mono font-semibold transition-all ${
                                  cfg.animStyle === s
                                    ? 'bg-indigo-600 text-white shadow-md shadow-indigo-500/20'
                                    : 'bg-slate-900 text-slate-400 hover:text-white'
                                }`}>
                                {s}
                              </button>
                            ))}
                          </div>
                        </div>
                        <Slider label="Speed" value={cfg.animSpeed.intensity} min={10} max={200}
                          onChange={v => setCfg(p => ({ ...p, animSpeed: { ...p.animSpeed, intensity: v } }))} />
                        <Slider label="Intensity" value={cfg.animIntensity.intensity} min={0} max={100}
                          onChange={v => setCfg(p => ({ ...p, animIntensity: { ...p.animIntensity, intensity: v } }))} />
                      </>)}
                    </>)}

                  </div>

                  {/* Bottom CTA */}
                  <div className="p-4 border-t border-slate-800/60">
                    <button onClick={onNavigateToChat}
                      className="w-full py-2.5 rounded-xl bg-gradient-to-r from-sky-500/20 to-indigo-500/20 hover:from-sky-500/30 hover:to-indigo-500/30 border border-sky-400/30 text-sky-300 font-semibold text-xs flex items-center justify-center gap-2 transition-all">
                      <MessageSquare className="w-4 h-4" />
                      Analyze Satellite Images in AI Chat →
                    </button>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* ── Feature Grid ─────────────────────────────────────────────── */}
        <div className="mt-20 pt-12 border-t border-slate-800/50">
          <div className="text-center max-w-2xl mx-auto mb-10">
            <h2 className="text-3xl font-extrabold text-white mb-3">
              Engineered for High-Resolution Remote Sensing
            </h2>
            <p className="text-slate-400 text-sm">
              Powered by Qwen2-VL-7B-Instruct, GeoTIFF multispectral preprocessing, and deterministic query routing.
            </p>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {[
              { icon: Cpu, color: 'sky', title: 'Qwen2-VL 7B Vision Backbone',
                body: 'Variable-resolution dynamic patching. 4-bit quantized for rapid local inference on consumer GPUs.' },
              { icon: Layers, color: 'indigo', title: 'GeoTIFF Band Extraction',
                body: 'Preserves spatial CRS metadata and constructs balanced RGB representations from multispectral layers.' },
              { icon: Zap, color: 'cyan', title: 'Deterministic VQA Router',
                body: 'Automatically routes scene description prompts to captioning pipelines and specific queries to VQA modes.' },
            ].map(({ icon: Icon, color, title, body }) => (
              <div key={title}
                className={`bg-slate-900/60 border border-slate-800 rounded-2xl p-6 hover:border-${color}-500/30 transition-all group`}>
                <div className={`w-12 h-12 rounded-xl bg-${color}-500/10 border border-${color}-500/25 flex items-center justify-center text-${color}-400 mb-4 group-hover:scale-110 transition-transform`}>
                  <Icon className="w-6 h-6" />
                </div>
                <h3 className="text-base font-bold text-white mb-2">{title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{body}</p>
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
