import React, { useState, useEffect } from 'react';
import { OrbitalHeroSection } from '../components/ui/orbital-hero-section';
import {
  MessageSquare, ArrowRight, Sparkles, Satellite,
  Layers, Zap, Cpu, Compass, Orbit, FileUp, CheckCircle2, Globe2
} from 'lucide-react';

/** Custom hook to handle responsive canvas framing */
function useNarrow(query = '(max-width: 767px)') {
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const m = window.matchMedia(query);
    const sync = () => setNarrow(m.matches);
    sync();
    m.addEventListener('change', sync);
    return () => m.removeEventListener('change', sync);
  }, [query]);
  return narrow;
}

const CAPABILITIES = [
  {
    icon: Cpu,
    title: 'Qwen2-VL 7B Vision Model',
    body: 'Domain-adapted for Earth Observation. Variable-resolution dynamic patching with 4-bit LoRA quantization for rapid local inference.',
    tag: 'Vision Backbone'
  },
  {
    icon: Layers,
    title: 'Multispectral GeoTIFF Support',
    body: 'Extracts coordinate reference systems (CRS), bounding coordinates, and reconstructs calibrated TrueColor RGB from raw multi-band TIFFs.',
    tag: 'Spectral Preprocessing'
  },
  {
    icon: Zap,
    title: 'Dual Task Pipeline Router',
    body: 'Deterministic intent classification automatically branches queries between deep visual question answering (VQA) and high-fidelity scene captioning.',
    tag: 'VQA + Captioning'
  },
];

const SAMPLE_QUERIES = [
  {
    title: 'Maritime & Port Logistics',
    query: 'Count container vessels, cargo ships, and dock infrastructure along the harbor.',
    badge: 'Object Counting'
  },
  {
    title: 'Agricultural Health & Water',
    query: 'Identify center-pivot irrigation circles and summarize crop vigor distribution.',
    badge: 'Vegetation Analysis'
  },
  {
    title: 'Urban Expansion & Land Cover',
    query: 'Classify high-density residential zones, road networks, and natural water bodies.',
    badge: 'Land Cover'
  }
];

export default function LandingPage({ onNavigateToChat }) {
  const narrow = useNarrow();

  return (
    <div className="relative min-h-screen bg-[#08090a] text-white overflow-x-hidden font-sans selection:bg-sky-500 selection:text-white">

      {/* ── Orbital Keplerian Hero Section ──────────────────────────── */}
      <section className="relative min-h-[92svh] w-full md:min-h-[740px] border-b border-white/[0.07]">
        <OrbitalHeroSection
          focus={narrow ? [0.5, 0.86] : [0.74, 0.44]}
          scrim={narrow ? 'top' : 'left'}
          scrimStrength={narrow ? 0.94 : 0.92}
          viewRadius={narrow ? 2.2 : 3.2}
          lead={narrow ? 0.05 : 0.12}
          glow={narrow ? 0.5 : 1}
          driftSpeed={1.4}
          trailYears={2.8}
          showSunTrack={true}
          interactive={true}
          className="min-h-[92svh] md:min-h-[740px]"
        >
          <div className="relative z-10 flex h-full min-h-[92svh] md:min-h-[740px] items-center px-6 pt-12 pb-16 sm:px-10 lg:px-20">
            <div className="max-w-[38rem] text-left">
              
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full mb-6
                bg-sky-500/10 border border-sky-500/25 text-sky-300 text-xs font-semibold tracking-wider uppercase
                backdrop-blur-md shadow-lg shadow-sky-500/5"
              >
                <Orbit className="w-3.5 h-3.5 text-sky-400 animate-spin" style={{ animationDuration: '10s' }} />
                <span>AyushFrontend · Earth Observation AI</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-4xl sm:text-6xl lg:text-[4.2rem] font-extrabold tracking-[-0.04em] leading-[1.02] mb-6"
                style={{
                  background: 'linear-gradient(135deg, #ffffff 0%, #d8e5ff 50%, #8ac2ff 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}
              >
                Every orbit.
                <br />
                <span style={{
                  background: 'linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #22d3ee 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  backgroundClip: 'text',
                }}>
                  Every detail.
                </span>
              </h1>

              {/* Sub-text */}
              <p className="text-base sm:text-lg text-slate-300/80 max-w-xl leading-relaxed mb-8 font-normal">
                Upload raw satellite passes or multispectral GeoTIFFs. Ask natural language questions about infrastructure, terrain, and environmental change with grounded AI answers.
              </p>

              {/* Call-to-action buttons redirecting to other sections & chatbot */}
              <div className="flex flex-wrap items-center gap-3.5">
                <button
                  onClick={onNavigateToChat}
                  className="group relative inline-flex items-center gap-3 px-8 py-4 rounded-2xl font-bold text-sm text-white
                    shadow-2xl hover:scale-[1.02] active:scale-[0.98] transition-all duration-200 cursor-pointer"
                  style={{
                    background: 'linear-gradient(135deg, #0ea5e9 0%, #4f46e5 50%, #06b6d4 100%)',
                    boxShadow: '0 0 35px rgba(56,189,248,0.25), 0 4px 20px rgba(0,0,0,0.5)',
                  }}
                >
                  <MessageSquare className="w-4 h-4 group-hover:rotate-12 transition-transform" />
                  <span>Launch AI Satellite Chatbot</span>
                  <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
                </button>

                <a
                  href="#capabilities"
                  className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl text-sm text-slate-300 font-medium
                    bg-white/[0.05] border border-white/10 hover:border-white/20 hover:bg-white/[0.08] hover:text-white
                    backdrop-blur-sm transition-all"
                >
                  <span>Explore Capabilities</span>
                </a>
              </div>

              {/* Micro specs */}
              <div className="mt-10 flex items-center gap-5 text-xs text-slate-400/70 font-mono">
                <span className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-sky-400" /> GeoTIFF CRS Extraction</span>
                <span className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-sky-400" /> 4-Bit LoRA Quantized</span>
              </div>

            </div>
          </div>
        </OrbitalHeroSection>
      </section>

      {/* ── Key Metrics Strip ───────────────────────────────────────── */}
      <div className="relative z-20 border-b border-white/[0.06] bg-slate-950/60 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-6 py-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { value: 'Qwen2-VL 7B', label: 'Vision-Language Model' },
            { value: '4-Bit NF4', label: 'Quantized Local Weights' },
            { value: 'GeoTIFF / PNG', label: 'Full Raster Support' },
            { value: 'Dual Router', label: 'VQA & Scene Captioning' },
          ].map((item) => (
            <div key={item.label} className="text-center md:text-left md:pl-4 border-l border-white/[0.06] first:border-l-0">
              <div className="text-xl font-bold text-sky-300 tracking-tight">{item.value}</div>
              <div className="text-xs text-slate-400 font-medium mt-0.5">{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Capabilities Section ────────────────────────────────────── */}
      <section id="capabilities" className="relative z-20 py-24 px-6 max-w-6xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full
            bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold tracking-wider uppercase mb-4"
          >
            <Sparkles className="w-3.5 h-3.5" />
            Core Capabilities
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white tracking-tight mb-4">
            Engineered for Remote Sensing Workflows
          </h2>
          <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
            From raw multispectral bands to fine-grained visual question answering, SatQuery connects Earth observation imagery with state-of-the-art vision models.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {CAPABILITIES.map((cap) => {
            const Icon = cap.icon;
            return (
              <div
                key={cap.title}
                className="group relative p-7 rounded-3xl border border-white/[0.08] bg-slate-900/50
                  hover:bg-slate-900/90 hover:border-sky-500/40 transition-all duration-300
                  backdrop-blur-md flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div className="w-12 h-12 rounded-2xl flex items-center justify-center
                      bg-sky-500/10 border border-sky-500/25 text-sky-400 group-hover:scale-110 transition-transform duration-300"
                    >
                      <Icon className="w-6 h-6" />
                    </div>
                    <span className="text-[10px] font-mono uppercase tracking-widest text-slate-400 px-2 py-0.5 rounded-md bg-white/[0.04]">
                      {cap.tag}
                    </span>
                  </div>
                  <h3 className="text-lg font-bold text-white mb-2.5">{cap.title}</h3>
                  <p className="text-slate-400 text-xs sm:text-sm leading-relaxed">{cap.body}</p>
                </div>

                <div className="mt-6 pt-4 border-t border-white/[0.05] flex items-center gap-2 text-xs font-semibold text-sky-400">
                  <span>Optimized Pipeline</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Sample Questions / Quick Redirects ────────────────────────── */}
      <section className="relative z-20 py-20 px-6 border-t border-white/[0.06] bg-slate-950/80">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
            <div>
              <div className="text-xs font-semibold text-sky-400 uppercase tracking-widest mb-2">Example Investigations</div>
              <h2 className="text-2xl sm:text-3xl font-extrabold text-white">Ask Anything in Natural Language</h2>
            </div>
            <p className="text-xs sm:text-sm text-slate-400 max-w-md">
              Click any investigation below to immediately open the satellite chatbot workspace with your query ready.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {SAMPLE_QUERIES.map((item, idx) => (
              <button
                key={item.title}
                onClick={onNavigateToChat}
                className="group p-6 rounded-2xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.05] hover:border-sky-500/40 text-left transition-all duration-200 cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs text-sky-400 font-mono mb-3">
                  <span className="px-2 py-0.5 rounded bg-sky-500/10 border border-sky-500/20">{item.badge}</span>
                  <span className="text-slate-500">0{idx + 1}</span>
                </div>
                <h3 className="font-bold text-base text-white mb-2 group-hover:text-sky-300 transition-colors">
                  {item.title}
                </h3>
                <p className="text-xs text-slate-400 leading-relaxed mb-4">
                  “{item.query}”
                </p>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-sky-400 group-hover:underline">
                  <span>Open in Chatbot</span>
                  <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-1 transition-transform" />
                </div>
              </button>
            ))}
          </div>
        </div>
      </section>

      {/* ── Call To Action Banner ───────────────────────────────────── */}
      <section className="relative z-20 py-20 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="relative overflow-hidden rounded-3xl border border-sky-500/25 p-10 sm:p-14 text-center bg-gradient-to-b from-sky-950/40 via-slate-900/60 to-slate-950">
            <div className="relative z-10 max-w-2xl mx-auto">
              <div className="w-12 h-12 rounded-2xl bg-sky-500/10 border border-sky-500/30 text-sky-400 flex items-center justify-center mx-auto mb-6">
                <Satellite className="w-6 h-6" />
              </div>
              <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-4 tracking-tight">
                Ready to Analyze Your Satellite Scenes?
              </h2>
              <p className="text-slate-300/80 text-sm sm:text-base leading-relaxed mb-8">
                Upload GeoTIFFs or standard satellite images, toggle spectral false-color visualizations, and interact with the AI assistant.
              </p>
              <button
                onClick={onNavigateToChat}
                className="inline-flex items-center gap-3 px-8 py-4 rounded-2xl font-bold text-sm text-white
                  hover:scale-[1.03] active:scale-[0.98] transition-all duration-200 cursor-pointer shadow-xl shadow-sky-500/20"
                style={{
                  background: 'linear-gradient(135deg, #0ea5e9, #6366f1)',
                }}
              >
                <MessageSquare className="w-4 h-4" />
                <span>Start Satellite Analysis Workspace</span>
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="relative z-20 border-t border-white/[0.06] py-8 text-center text-slate-500 text-xs font-mono tracking-wider">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <span>SATQUERY AI · AYUSHFRONTEND BRANCH</span>
          <span>POWERED BY QWEN2-VL & KEPLER ORBITAL TELEMETRY</span>
        </div>
      </footer>

    </div>
  );
}
