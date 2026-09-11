import React, { useState, useEffect } from 'react';
import { Settings, Cpu, HardDrive, RefreshCw, Terminal, CheckCircle2 } from 'lucide-react';

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
    <div className="min-h-[calc(100vh-65px)] bg-[#08090a] text-white p-4 lg:p-8 font-sans selection:bg-white selection:text-black">
      <div className="max-w-4xl mx-auto space-y-6">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-4 border-b border-white/[0.08]">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Settings className="w-6 h-6 text-white" />
              Model & System Configuration
            </h1>
            <p className="text-xs text-white/50 mt-1">
              Configure Qwen2-VL-7B adapter weights, 4-bit quantization, and backend API status.
            </p>
          </div>
          <button
            onClick={checkApi}
            disabled={isTesting}
            className="px-4 py-2 rounded-xl bg-white text-black hover:bg-white/90 text-xs font-semibold flex items-center gap-2 transition-all cursor-pointer shadow-sm"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isTesting ? 'animate-spin' : ''}`} />
            <span>Test Endpoint</span>
          </button>
        </div>

        {/* System Health Card */}
        <div className="bg-white/[0.02] border border-white/[0.08] rounded-3xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-4">
            <Cpu className="w-4 h-4 text-white" />
            Active Backend Status
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 font-mono text-xs">
            <div className="bg-black/60 p-4 rounded-2xl border border-white/[0.08]">
              <div className="text-white/40 text-[10px] uppercase mb-1">Branch</div>
              <div className="text-white font-bold">ayushFRONTEND</div>
            </div>
            <div className="bg-black/60 p-4 rounded-2xl border border-white/[0.08]">
              <div className="text-white/40 text-[10px] uppercase mb-1">Vision Backbone</div>
              <div className="text-white font-bold">Qwen2-VL-7B-Instruct</div>
            </div>
            <div className="bg-black/60 p-4 rounded-2xl border border-white/[0.08]">
              <div className="text-white/40 text-[10px] uppercase mb-1">Status</div>
              <div className="text-white font-bold flex items-center gap-1.5">
                <CheckCircle2 className="w-3.5 h-3.5 text-white" /> Online
              </div>
            </div>
          </div>
        </div>

        {/* LoRA & Quantization Config */}
        <div className="bg-white/[0.02] border border-white/[0.08] rounded-3xl p-6 shadow-xl space-y-4">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <HardDrive className="w-4 h-4 text-white" />
            LoRA Adaptation & Precision
          </h3>

          <div>
            <label className="block text-xs font-medium text-white/70 mb-1.5">
              LoRA Checkpoint Path
            </label>
            <input
              type="text"
              value={loraPath}
              onChange={(e) => setLoraPath(e.target.value)}
              className="w-full bg-black/60 border border-white/[0.1] rounded-xl px-4 py-2.5 text-xs text-white font-mono focus:outline-none focus:border-white/40"
            />
            <p className="text-[11px] text-white/40 mt-1">
              Path to fine-tuned remote sensing adapter directory e.g., <code className="text-white/80">checkpoints/rs_vlm_lora/</code>.
            </p>
          </div>

          <div>
            <label className="block text-xs font-medium text-white/70 mb-1.5">
              Inference Quantization Mode
            </label>
            <div className="grid grid-cols-3 gap-2">
              {[
                { id: '4bit', label: '4-Bit BitsAndBytes (Recommended)' },
                { id: '8bit', label: '8-Bit Quantization' },
                { id: 'fp16', label: 'Float16 Standard' }
              ].map((q) => (
                <button
                  key={q.id}
                  onClick={() => setQuantization(q.id)}
                  className={`p-3 rounded-xl border text-xs font-medium transition-all cursor-pointer ${
                    quantization === q.id 
                      ? 'bg-white text-black border-white font-semibold' 
                      : 'bg-black/60 border-white/[0.08] text-white/60 hover:text-white hover:border-white/20'
                  }`}
                >
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* API Response Debug Terminal */}
        <div className="bg-white/[0.02] border border-white/[0.08] rounded-3xl p-6 shadow-xl">
          <h3 className="text-sm font-bold text-white flex items-center gap-2 mb-3">
            <Terminal className="w-4 h-4 text-white" />
            Backend API Json Response
          </h3>
          <pre className="bg-black/70 p-4 rounded-2xl border border-white/[0.08] text-[11px] font-mono text-white/90 overflow-x-auto">
            {JSON.stringify(apiHealth || { status: "checking..." }, null, 2)}
          </pre>
        </div>

      </div>
    </div>
  );
}
