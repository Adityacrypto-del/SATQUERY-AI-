import React, { useState, useEffect } from 'react';
import { OrbitalHeroSection } from '../components/ui/orbital-hero-section';
import {
  MessageSquare, ArrowRight, Sparkles, Satellite,
  Layers, Zap, Cpu, Orbit, CheckCircle2
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
    <div className="relative min-h-screen bg-[#08090a] text-white overflow-x-hidden font-sans selection:bg-white selection:text-black">

      {/* ── Orbital Keplerian Hero Section ──────────────────────────── */}
      <section className="relative min-h-[92svh] w-full md:min-h-[740px] border-b border-white/[0.08]">
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
          sunColor="#FFFFFF"
          className="min-h-[92svh] md:min-h-[740px]"
        >
          <div className="relative z-10 flex h-full min-h-[92svh] md:min-h-[740px] items-center px-6 pt-12 pb-16 sm:px-10 lg:px-20">
            <div className="max-w-[38rem] text-left">
              
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full mb-7
                bg-white/[0.05] border border-white/15 text-white/80 text-xs font-semibold tracking-wider uppercase
                backdrop-blur-md shadow-lg"
              >
                <Orbit className="w-3.5 h-3.5 text-white animate-spin" style={{ animationDuration: '12s' }} />
                <span>Earth Observation Intelligence</span>
              </div>

              {/* Main Heading */}
              <h1 className="text-4xl sm:text-6xl lg:text-[4.25rem] font-light tracking-[-0.04em] leading-[1.03] mb-6 text-white"
                style={{
                  textShadow: '0 2px 20px rgba(0,0,0,0.8)'
                }}
              >
                Nothing here
                <br />
                <span className="font-semibold text-white/95">
                  stands still.
                </span>
              </h1>

              {/* Sub-text */}
              <p className="text-base sm:text-lg text-white/60 max-w-xl leading-relaxed mb-9 font-normal">
                Upload raw satellite passes or multispectral GeoTIFFs. Ask natural language questions about infrastructure, terrain, and environmental change with grounded AI answers.
              </p>

              {/* Black & White CTAs */}
              <div className="flex flex-wrap items-center gap-3.5">
                <button
                  onClick={onNavigateToChat}
                  className="inline-flex items-center gap-2.5 rounded-full bg-white px-7 py-3.5 text-sm font-semibold text-black transition-all hover:bg-white/90 active:scale-[0.98] shadow-lg cursor-pointer"
                >
                  <MessageSquare className="w-4 h-4 text-black" />
                  <span>Start analysing</span>
                  <ArrowRight className="w-4 h-4 text-black" />
                </button>

                <a
                  href="#capabilities"
                  className="inline-flex items-center gap-2 rounded-full border border-white/20 bg-black/40 px-6 py-3.5 text-sm font-medium text-white/85 transition-all hover:border-white/40 hover:bg-white/[0.08] hover:text-white backdrop-blur-sm cursor-pointer"
                >
                  <span>How it works</span>
                </a>
              </div>

              {/* Micro specs */}
              <div className="mt-11 flex items-center gap-6 text-xs text-white/40 font-mono">
                <span className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-white/70" /> GeoTIFF CRS Extraction</span>
                <span className="flex items-center gap-1.5"><CheckCircle2 className="w-3.5 h-3.5 text-white/70" /> 4-Bit LoRA Quantized</span>
              </div>

            </div>
          </div>
        </OrbitalHeroSection>
      </section>

      {/* ── Key Metrics Strip ───────────────────────────────────────── */}
      <div className="relative z-20 border-b border-white/[0.08] bg-[#050505]">
        <div className="max-w-6xl mx-auto px-6 py-6 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { value: 'Qwen2-VL 7B', label: 'Vision Backbone' },
            { value: '4-Bit NF4', label: 'Quantized LoRA Weights' },
            { value: 'GeoTIFF / PNG', label: 'Spectral Raster Support' },
            { value: 'Dual Router', label: 'VQA & Scene Captioning' },
          ].map((item) => (
            <div key={item.label} className="text-center md:text-left md:pl-4 border-l border-white/[0.08] first:border-l-0">
              <div className="text-xl font-bold text-white tracking-tight">{item.value}</div>
              <div className="text-xs text-white/45 font-medium mt-0.5">{item.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Capabilities Section ────────────────────────────────────── */}
      <section id="capabilities" className="relative z-20 py-24 px-6 max-w-6xl mx-auto">
        <div className="text-center max-w-2xl mx-auto mb-16">
          <div className="inline-flex items-center gap-2 px-3.5 py-1 rounded-full
            bg-white/[0.05] border border-white/15 text-white/70 text-xs font-semibold tracking-wider uppercase mb-4"
          >
            <Sparkles className="w-3.5 h-3.5 text-white" />
            Core Capabilities
          </div>
          <h2 className="text-3xl sm:text-4xl font-bold text-white tracking-tight mb-4">
            Built for Remote Sensing Workflows
          </h2>
          <p className="text-white/50 text-sm sm:text-base leading-relaxed">
            From raw multispectral bands to fine-grained visual question answering, SatQuery connects Earth observation imagery with state-of-the-art vision models.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {CAPABILITIES.map((cap) => {
            const Icon = cap.icon;
            return (
              <div
                key={cap.title}
                className="group relative p-7 rounded-3xl border border-white/[0.08] bg-white/[0.02]
                  hover:bg-white/[0.05] hover:border-white/25 transition-all duration-300
                  backdrop-blur-md flex flex-col justify-between"
              >
                <div>
                  <div className="flex items-center justify-between mb-6">
                    <div className="w-12 h-12 rounded-2xl flex items-center justify-center
                      bg-white/[0.06] border border-white/15 text-white group-hover:scale-105 transition-transform duration-300"
                    >
                      <Icon className="w-5 h-5 text-white" />
                    </div>
                    <span className="text-[10px] font-mono uppercase tracking-widest text-white/50 px-2 py-0.5 rounded-md bg-white/[0.04]">
                      {cap.tag}
                    </span>
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2.5">{cap.title}</h3>
                  <p className="text-white/50 text-xs sm:text-sm leading-relaxed">{cap.body}</p>
                </div>

                <div className="mt-6 pt-4 border-t border-white/[0.06] flex items-center gap-2 text-xs font-semibold text-white/80">
                  <span>Optimized Pipeline</span>
                  <ArrowRight className="w-3 h-3 group-hover:translate-x-1 transition-transform" />
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Sample Questions / Quick Redirects ────────────────────────── */}
      <section className="relative z-20 py-20 px-6 border-t border-white/[0.08] bg-[#050505]">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row md:items-end justify-between mb-12 gap-4">
            <div>
              <div className="text-xs font-semibold text-white/45 uppercase tracking-widest mb-2">Example Investigations</div>
              <h2 className="text-2xl sm:text-3xl font-bold text-white">Ask Anything in Plain Language</h2>
            </div>
            <p className="text-xs sm:text-sm text-white/50 max-w-md">
              Click any investigation below to immediately open the satellite chatbot workspace with your query ready.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {SAMPLE_QUERIES.map((item, idx) => (
              <button
                key={item.title}
                onClick={onNavigateToChat}
                className="group p-6 rounded-2xl border border-white/[0.08] bg-white/[0.02] hover:bg-white/[0.05] hover:border-white/25 text-left transition-all duration-200 cursor-pointer"
              >
                <div className="flex items-center justify-between text-xs font-mono mb-3">
                  <span className="px-2 py-0.5 rounded bg-white/[0.06] border border-white/10 text-white/80">{item.badge}</span>
                  <span className="text-white/35">0{idx + 1}</span>
                </div>
                <h3 className="font-semibold text-base text-white mb-2 group-hover:text-white/90 transition-colors">
                  {item.title}
                </h3>
                <p className="text-xs text-white/50 leading-relaxed mb-4">
                  “{item.query}”
                </p>
                <div className="flex items-center gap-1.5 text-xs font-semibold text-white/80 group-hover:underline">
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
          <div className="relative overflow-hidden rounded-3xl border border-white/15 p-10 sm:p-14 text-center bg-white/[0.02] backdrop-blur-md">
            <div className="relative z-10 max-w-2xl mx-auto">
              <div className="w-12 h-12 rounded-2xl bg-white/[0.06] border border-white/15 text-white flex items-center justify-center mx-auto mb-6">
                <Satellite className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-3xl sm:text-4xl font-bold text-white mb-4 tracking-tight">
                Ready to Analyze Your Satellite Scenes?
              </h2>
              <p className="text-white/50 text-sm sm:text-base leading-relaxed mb-8">
                Upload GeoTIFFs or standard satellite images, inspect spectral visual representations, and interact with the AI assistant.
              </p>
              <button
                onClick={onNavigateToChat}
                className="inline-flex items-center gap-2.5 rounded-full bg-white px-8 py-3.5 text-sm font-semibold text-black hover:bg-white/90 active:scale-[0.98] transition-all cursor-pointer shadow-lg"
              >
                <MessageSquare className="w-4 h-4 text-black" />
                <span>Open Satellite Chatbot</span>
                <ArrowRight className="w-4 h-4 text-black" />
              </button>
            </div>
          </div>
        </div>
      </section>

      {/* ── Footer ──────────────────────────────────────────────────── */}
      <footer className="relative z-20 border-t border-white/[0.08] py-8 text-center text-white/40 text-xs font-mono tracking-wider">
        <div className="max-w-6xl mx-auto px-6 flex flex-col sm:flex-row items-center justify-between gap-3">
          <span>SATQUERY AI · AYUSHFRONTEND</span>
          <span>REMOTE SENSING INTELLIGENCE</span>
        </div>
      </footer>

    </div>
  );
}
