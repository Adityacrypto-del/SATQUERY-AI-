import React, { useState, useEffect } from 'react';
import { Settings, Cpu, HardDrive, ShieldCheck, RefreshCw, Terminal, CheckCircle2, AlertTriangle, Key } from 'lucide-react';

export default function SettingsPage({ backendStatus }) {
  const [loraPath, setLoraPath] = useState('checkpoints/rs_vlm_lora/');
  const [quantization, setQuantization] = useState('4bit');
  const [apiHealth, setApiHealth] = useState(null);
  const [isTesting, setIsTesting] = useState(false);

  const checkApi = () => {
    setIsTesting(true);
    fetch('/api/status')
      .then(res => res.json())
      .then(data => setApiHealth(data))
      .catch(err => setApiHealth({ error: err.message }))
      .finally(() => setIsTesting(false));
  };

  useEffect(() => {
    checkApi();
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-4 lg:p-8 font-sans">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-800">
          <div>
            <h1 className="text-3xl font-extrabold text-white flex items-center gap-3">
              <Settings className="w-7 h-7 text-sky-400" />
              Model & System Configuration
            </h1>
            <p className="text-xs text-slate-400 mt-1">
              Configure Qwen2-VL-7B adapter weights, 4-bit quantization, and backend API status.
            </p>
          </div>
          <button
            onClick={checkApi}
            disabled={isTesting}
            className="px-4 py-2 rounded-xl bg-slate-900 hover:bg-slate-800 border border-slate-700 text-xs font-semibold text-sky-300 flex items-center gap-2 transition-all"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTesting ? 'animate-spin' : ''}`} />
            <span>Test Endpoint</span>
          </button>
        </div>

        {/* System Health Card */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-sky-300 flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-sky-400" />
            Active Backend Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <div className="text-slate-400 text-[10px] uppercase mb-1">Branch</div>
              <div className="text-sky-300 font-bold">ayushFRONTEND</div>
            </div>
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <div className="text-slate-400 text-[10px] uppercase mb-1">Vision Backbone</div>
              <div className="text-white font-bold">Qwen2-VL-7B-Instruct</div>
            </div>
            <div className="bg-slate-950 p-4 rounded-2xl border border-slate-800">
              <div className="text-slate-400 text-[10px] uppercase mb-1">Status</div>
              <div className="text-emerald-400 font-bold flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> Online
              </div>
            </div>
          </div>
        </div>

        {/* LoRA & Quantization Config */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-xl space-y-4">
          <h3 className="text-sm font-bold text-sky-300 flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-sky-400" />
            LoRA Adaptation & Precision
          </h3>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              LoRA Checkpoint Path
            </label>
            <input
              type="text"
              value={loraPath}
              onChange={(e) => setLoraPath(e.target.value)}
              className="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-2.5 text-xs text-slate-100 font-mono focus:outline-none focus:border-sky-500"
            />
            <p className="text-[11px] text-slate-400 mt-1">
              Path to fine-tuned remote sensing adapter directory e.g., <code className="text-sky-300">checkpoints/rs_vlm_lora/</code>.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1">
              Inference Quantization Mode
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: '4bit', label: '4-Bit BitsAndBytes (Fits in 8GB VRAM)' },
                { id: '8bit', label: '8-Bit Quantization' },
                { id: 'fp16', label: 'Float16 Standard' }
              ].map((q) => (
                <button
                  key={q.id}
                  onClick={() => setQuantization(q.id)}
                  className={`p-3 rounded-xl border text-xs font-medium transition-all ${
                    quantization === q.id 
                      ? 'bg-sky-500/20 border-sky-400 text-sky-200' 
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-white'
                  }`}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* API Response Debug Terminal */}
        <div className="bg-slate-900/80 border border-slate-800 rounded-3xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-sky-300 flex items-center gap-2 mb-3">
            <Terminal className="w-4 h-4 text-sky-400" />
            Backend API Json Response
          </h3>
          <pre className="bg-slate-950 p-4 rounded-2xl border border-slate-800 text-[11px] font-mono text-emerald-400 overflow-x-auto">
            {JSON.stringify(apiHealth || { status: "checking..." }, null, 2)}
          </pre>
        </div>

      </div>
    </div>
  );
}
