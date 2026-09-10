import React from 'react';
import { Sparkles, MessageSquare, Compass, Settings, Orbit, CheckCircle2, RefreshCw } from 'lucide-react';

export default function Navbar({ activePage, setActivePage, backendStatus }) {
  const navItems = [
    { id: 'landing', label: 'Landing & Studio', icon: Sparkles },
    { id: 'chatbot', label: 'AI Satellite Chatbot', icon: MessageSquare, badge: 'VQA / Caption' },
    { id: 'explorer', label: 'Geo-Explorer', icon: Compass },
    { id: 'settings', label: 'Model Settings', icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-slate-950/80 border-b border-slate-800/80 px-4 lg:px-8 py-3 transition-all duration-300">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div 
          onClick={() => setActivePage('landing')}
          className="flex items-center gap-3 cursor-pointer group"
        >
          <div className="relative w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 via-indigo-500 to-cyan-400 p-[1.5px] shadow-lg shadow-sky-500/20 group-hover:shadow-sky-500/40 transition-all duration-300">
            <div className="w-full h-full bg-slate-950 rounded-[10.5px] flex items-center justify-center">
              <Orbit className="w-5 h-5 text-sky-400 animate-spin-slow group-hover:scale-110 transition-transform" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-extrabold text-lg text-transparent bg-clip-text bg-gradient-to-r from-sky-300 via-indigo-200 to-white tracking-wider">
                SatQuery
              </span>
              <span className="px-2 py-0.5 text-[10px] font-mono tracking-widest uppercase bg-sky-500/10 text-sky-400 border border-sky-500/30 rounded-full">
                ayushFRONTEND
              </span>
            </div>
            <p className="text-[11px] text-slate-400 font-medium">Remote Sensing AI Assistant</p>
          </div>
        </div>

        {/* Navigation Buttons */}
        <nav className="flex items-center bg-slate-900/90 border border-slate-800 p-1 rounded-2xl shadow-inner">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-semibold transition-all duration-200 ${
                  isActive
                    ? 'bg-gradient-to-r from-sky-500/20 via-indigo-500/20 to-cyan-500/20 text-sky-300 border border-sky-400/30 shadow-md shadow-sky-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-sky-400' : 'text-slate-400'}`} />
                <span>{item.label}</span>
                {item.badge && (
                  <span className="hidden md:inline-block px-1.5 py-0.2 text-[9px] font-mono bg-sky-500/20 text-sky-300 rounded border border-sky-500/30">
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Backend Status Pill */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800 text-xs">
          <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-slate-300 font-mono text-[11px]">
            {backendStatus?.status === 'online' ? 'Qwen2-VL Ready' : 'System Active'}
          </span>
        </div>

      </div>
    </header>
  );
}
