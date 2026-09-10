import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import LandingPage from './pages/LandingPage';
import ChatbotPage from './pages/ChatbotPage';
import ExplorerPage from './pages/ExplorerPage';
import SettingsPage from './pages/SettingsPage';

export default function App() {
  const [activePage, setActivePage] = useState('landing');
  const [backendStatus, setBackendStatus] = useState(null);

  useEffect(() => {
    fetch('/api/status')
      .then(res => res.json())
      .then(data => setBackendStatus(data))
      .catch(() => setBackendStatus({ status: 'offline' }));
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 selection:bg-sky-500 selection:text-white">
      {/* Top Glassmorphism Navigation Bar */}
      <Navbar 
        activePage={activePage} 
        setActivePage={setActivePage} 
        backendStatus={backendStatus} 
      />

      {/* Dynamic Page Router */}
      <main>
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
