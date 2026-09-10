import React, { useState } from 'react';
import GlobeStudy from '../components/ui/globe-study';
import {
  MessageSquare, ArrowRight, Sparkles, Satellite,
  Layers, Zap, Cpu, MousePointer2, Orbit,
} from 'lucide-react';

const FEATURES = [
  {
    icon: Cpu,
    color: 'sky',
    title: 'Qwen2-VL 7B Vision',
    body: 'Variable-resolution dynamic patching with 4-bit LoRA quantization for rapid local inference.',
  },
  {
    icon: Layers,
    color: 'indigo',
    title: 'GeoTIFF Multispectral',
    body: 'Preserves CRS metadata and constructs balanced RGB from multispectral band stacks.',
  },
  {
    icon: Zap,
    color: 'cyan',
    title: 'Deterministic VQA Router',
    body: 'Automatically classifies queries into captioning vs VQA pipelines before inference.',
  },
];

export default function LandingPage({ onNavigateToChat }) {
  const [globeHue, setGlobeHue] = useState(200);

  return (
    <div className="relative min-h-screen bg-[#08090a] text-white overflow-hidden font-sans selection:bg-sky-500 selection:text-white">

      {/* ── Full-viewport Globe Hero ───────────────────────────────────── */}
      <div className="relative w-full" style={{ height: '100vh', minHeight: 600 }}>

        {/* Globe fills the entire viewport */}
        <div className="absolute inset-0 z-0">
          <GlobeStudy
            mode="dark"
            scale={1}
            opacity={1}
            hue={globeHue - 200}
            saturation={1.15}
            brightness={1}
          />
        </div>

        {/* Gradient fade to dark at the bottom so content section blends in */}
        <div className="absolute bottom-0 left-0 right-0 h-64 pointer-events-none z-10"
          style={{ background: 'linear-gradient(to bottom, transparent, #08090a)' }} />

        {/* Radial glow behind text */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
          <div className="w-[700px] h-[500px] rounded-full"
            style={{ background: 'radial-gradient(ellipse at center, rgba(56,189,248,0.06) 0%, transparent 70%)' }} />
        </div>

        {/* ── Hero Overlay Text ─────────────────────────────────────────── */}
        <div className="absolute inset-0 z-20 flex flex-col items-center justify-center px-4 text-center pointer-events-none">

          {/* Branch badge */}
          <div className="pointer-events-auto inline-flex items-center gap-2 px-4 py-1.5 rounded-full mb-8
            bg-white/5 border border-white/10 text-white/60 text-xs font-semibold tracking-widest uppercase
            backdrop-blur-md shadow-lg"
          >
            <Orbit className="w-3.5 h-3.5 text-sky-400 animate-spin" style={{ animationDuration: '12s' }} />
            <span>SatQuery AI · ayushFRONTEND · Branch</span>
          </div>

          {/* Title */}
          <h1 className="text-5xl sm:text-7xl font-extrabold tracking-tight leading-none mb-6"
            style={{
              background: 'linear-gradient(135deg, #e2e4e9 0%, #a0c8ff 45%, #ffffff 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
              textShadow: 'none',
            }}
          >
            Geospatial&nbsp;AI
            <br />
            <span style={{
              background: 'linear-gradient(135deg, #38bdf8 0%, #818cf8 60%, #22d3ee 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              backgroundClip: 'text',
            }}>
              Intelligence
            </span>
          </h1>

          {/* Sub-copy */}
          <p className="text-base sm:text-lg text-white/45 max-w-xl leading-relaxed mb-10 font-light">
            Upload satellite imagery. Ask natural-language questions.
            Get structured scene analysis, land-cover breakdowns, and geo-metadata — instantly.
          </p>

          {/* CTAs */}
          <div className="pointer-events-auto flex flex-col sm:flex-row items-center gap-3">
            <button
              onClick={onNavigateToChat}
              className="group relative inline-flex items-center gap-3 px-8 py-4 rounded-2xl font-bold text-sm text-white
                shadow-2xl hover:scale-[1.03] active:scale-[0.98] transition-all duration-200"
              style={{
                background: 'linear-gradient(135deg, #0ea5e9, #6366f1, #06b6d4)',
                boxShadow: '0 0 40px rgba(56,189,248,0.25), 0 4px 24px rgba(0,0,0,0.4)',
              }}
            >
              <MessageSquare className="w-5 h-5 group-hover:rotate-12 transition-transform" />
              Launch AI Satellite Chatbot
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>

            <div className="inline-flex items-center gap-2 px-5 py-3.5 rounded-2xl text-sm text-white/50 font-medium
              bg-white/5 border border-white/8 backdrop-blur-sm select-none"
            >
              <MousePointer2 className="w-4 h-4 text-white/30" />
              <span>Drag · Scroll · Click to pin</span>
            </div>
          </div>
        </div>

        {/* Bottom scroll hint */}
        <div className="absolute bottom-10 left-1/2 -translate-x-1/2 z-20 pointer-events-none
          flex flex-col items-center gap-2 text-white/25 text-[11px] font-medium tracking-widest uppercase"
        >
          <div className="w-px h-10 bg-gradient-to-b from-transparent to-white/20 rounded-full" />
          <span>Scroll</span>
        </div>
      </div>

      {/* ── Stats Bar ─────────────────────────────────────────────────── */}
      <div className="relative z-20 border-t border-b border-white/5 bg-white/[0.02] backdrop-blur-sm">
        <div className="max-w-5xl mx-auto px-6 py-5 grid grid-cols-2 md:grid-cols-4 gap-6">
          {[
            { value: 'Qwen2-VL', label: '7B Vision Backbone' },
            { value: '4-bit', label: 'LoRA Quantized' },
            { value: 'GeoTIFF', label: 'Native Band Support' },
            { value: 'VQA + Cap', label: 'Dual Task Routing' },
          ].map(stat => (
            <div key={stat.label} className="text-center">
              <div className="text-lg font-extrabold text-sky-300 tracking-tight mb-0.5">{stat.value}</div>
              <div className="text-[11px] text-white/35 font-medium uppercase tracking-wider">{stat.label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Feature Cards ────────────────────────────────────────────── */}
      <div className="relative z-20 py-20 px-4 max-w-5xl mx-auto">
        <div className="text-center mb-14">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full
            bg-sky-500/10 border border-sky-500/20 text-sky-400 text-xs font-semibold tracking-wider uppercase mb-5"
          >
            <Sparkles className="w-3.5 h-3.5 animate-pulse" />
            Core Capabilities
          </div>
          <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-4 tracking-tight">
            Built for Remote Sensing
          </h2>
          <p className="text-white/40 text-sm max-w-lg mx-auto leading-relaxed">
            Every component of the pipeline is optimised for satellite and aerial imagery — from spectral preprocessing to domain-adapted inference.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, color, title, body }) => (
            <div
              key={title}
              className="group relative p-6 rounded-3xl border border-white/6 bg-white/[0.03]
                hover:bg-white/[0.06] hover:border-white/10 transition-all duration-300
                backdrop-blur-sm overflow-hidden"
            >
              {/* Glow accent */}
              <div className="absolute -top-10 -right-10 w-40 h-40 rounded-full opacity-0
                group-hover:opacity-100 transition-opacity duration-500 pointer-events-none"
                style={{ background: `radial-gradient(circle, rgba(56,189,248,0.12), transparent 70%)` }}
              />

              <div className="w-11 h-11 rounded-xl flex items-center justify-center mb-5
                bg-sky-500/10 border border-sky-500/20 text-sky-400
                group-hover:scale-110 transition-transform duration-300"
              >
                <Icon className="w-5 h-5" />
              </div>
              <h3 className="text-base font-bold text-white mb-2">{title}</h3>
              <p className="text-white/40 text-xs leading-relaxed">{body}</p>
            </div>
          ))}
        </div>
      </div>

      {/* ── Full-width CTA Banner ────────────────────────────────────── */}
      <div className="relative z-20 mb-20 mx-4 sm:mx-8 lg:mx-16">
        <div className="relative overflow-hidden rounded-3xl border border-white/8 p-10 sm:p-14 text-center"
          style={{
            background: 'linear-gradient(135deg, rgba(14,165,233,0.12), rgba(99,102,241,0.12), rgba(6,182,212,0.08))',
          }}
        >
          {/* Background globe glow */}
          <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
            <div className="w-[600px] h-[300px] rounded-full"
              style={{ background: 'radial-gradient(ellipse, rgba(56,189,248,0.08), transparent 70%)' }} />
          </div>

          <div className="relative z-10">
            <Satellite className="w-10 h-10 text-sky-400/60 mx-auto mb-5" />
            <h2 className="text-3xl sm:text-4xl font-extrabold text-white mb-4 tracking-tight">
              Start Querying Satellite Imagery
            </h2>
            <p className="text-white/40 text-sm max-w-md mx-auto leading-relaxed mb-8">
              Upload any PNG, JPEG, TIFF, or GeoTIFF. Ask in plain English. Get structured answers with land-cover analysis, execution traces, and spectral metadata.
            </p>
            <button
              onClick={onNavigateToChat}
              className="group inline-flex items-center gap-3 px-10 py-4 rounded-2xl font-bold text-sm text-white
                hover:scale-[1.03] active:scale-[0.97] transition-all duration-200 shadow-2xl"
              style={{
                background: 'linear-gradient(135deg, #0ea5e9, #6366f1, #06b6d4)',
                boxShadow: '0 0 40px rgba(56,189,248,0.2)',
              }}
            >
              <MessageSquare className="w-5 h-5 group-hover:rotate-12 transition-transform" />
              Open AI Satellite Chatbot
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
          </div>
        </div>
      </div>

      {/* Footer micro-text */}
      <div className="relative z-20 pb-8 text-center text-white/20 text-xs font-mono tracking-widest">
        SATQUERY AI · AYUSHFRONTEND · REMOTE SENSING INTELLIGENCE
      </div>

    </div>
  );
}
