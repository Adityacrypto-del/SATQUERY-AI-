import React from 'react';
import { Sparkles, MessageSquare, Compass, Settings, Orbit } from 'lucide-react';

export default function Navbar({ activePage, setActivePage, backendStatus }) {
  const navItems = [
    { id: 'landing', label: 'Landing & Studio', icon: Sparkles },
    { id: 'chatbot', label: 'AI Satellite Chatbot', icon: MessageSquare, badge: 'VQA / Caption' },
    { id: 'explorer', label: 'Geo-Explorer', icon: Compass },
    { id: 'settings', label: 'Model Settings', icon: Settings },
  ];

  return (
    <header className="sticky top-0 z-50 backdrop-blur-xl bg-[#08090a]/90 border-b border-white/[0.08] px-4 lg:px-8 py-3.5 transition-all duration-300">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        
        {/* Brand / Logo */}
        <div 
          onClick={() => setActivePage('landing')}
          className="flex items-center gap-3 cursor-pointer group select-none"
        >
          <div className="relative w-9 h-9 rounded-xl bg-white/[0.08] border border-white/20 p-[1px] shadow-md group-hover:border-white/40 transition-all duration-300">
            <div className="w-full h-full bg-[#08090a] rounded-[10px] flex items-center justify-center">
              <Orbit className="w-4 h-4 text-white group-hover:scale-110 transition-transform" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base text-white tracking-wide">
                SatQuery
              </span>
              <span className="px-2 py-0.5 text-[9px] font-mono tracking-widest uppercase bg-white/[0.06] text-white/70 border border-white/15 rounded-full">
                ayushFRONTEND
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 font-normal">Remote Sensing AI Assistant</p>
          </div>
        </div>

        {/* Navigation Buttons */}
        <nav className="flex items-center bg-white/[0.03] border border-white/[0.08] p-1 rounded-2xl shadow-inner">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`relative flex items-center gap-2 px-4 py-2 rounded-xl text-xs font-medium transition-all duration-200 cursor-pointer ${
                  isActive
                    ? 'bg-white text-black font-semibold shadow-sm'
                    : 'text-neutral-400 hover:text-white hover:bg-white/[0.06]'
                }`}
              >
                <Icon className={`w-3.5 h-3.5 ${isActive ? 'text-black' : 'text-neutral-400'}`} />
                <span>{item.label}</span>
                {item.badge && (
                  <span className={`hidden md:inline-block px-1.5 py-0.2 text-[9px] font-mono rounded border ${
                    isActive 
                      ? 'bg-black/10 text-black border-black/20' 
                      : 'bg-white/[0.06] text-white/60 border-white/10'
                  }`}>
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>

        {/* Backend Status Pill */}
        <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/[0.04] border border-white/[0.08] text-xs">
          <div className="w-1.5 h-1.5 rounded-full bg-white animate-pulse" />
          <span className="text-neutral-300 font-mono text-[11px]">
            {backendStatus?.status === 'online' ? 'Qwen2-VL Ready' : 'System Active'}
          </span>
        </div>

      </div>
    </header>
  );
}
