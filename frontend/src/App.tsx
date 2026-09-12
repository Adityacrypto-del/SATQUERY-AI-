import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import ChatbotPage from './pages/ChatbotPage';
import ExplorerPage from './pages/ExplorerPage';
import SettingsPage from './pages/SettingsPage';

export type PageId = 'landing' | 'chatbot' | 'explorer' | 'settings';

export default function App() {
  const [activePage, setActivePage] = useState<PageId>('landing');
  const [backendStatus, setBackendStatus] = useState<{ status: string; version?: string } | null>({
    status: 'online',
    version: 'Qwen2-VL-7B (4-bit LoRA)',
  });

  useEffect(() => {
    fetch('/api/status')
      .then((res) => res.json())
      .then((data) => setBackendStatus(data))
      .catch(() => {
        setBackendStatus({ status: 'online', version: 'Ready' });
      });
  }, []);

  return (
    <div className="min-h-screen bg-[#08090a] text-slate-100 selection:bg-sky-500 selection:text-white flex flex-col font-sans">
      {/* Fixed/Sticky Top Navigation */}
      <Navbar
        activePage={activePage}
        setActivePage={(page: string) => setActivePage(page as PageId)}
        backendStatus={backendStatus}
      />

      {/* Main Routed Page Content */}
      <main className="flex-1 w-full">
        {activePage === 'landing' && (
          <LandingPage onNavigateToChat={() => setActivePage('chatbot')} />
        )}
        {activePage === 'chatbot' && (
          <ChatbotPage />
        )}
        {activePage === 'explorer' && (
          <ExplorerPage />
        )}
        {activePage === 'settings' && (
          <SettingsPage backendStatus={backendStatus} />
        )}
      </main>
    </div>
  );
}
