import React, { useState } from 'react';
import './App.css';
import GenerateMode from './components/GenerateMode';
import DetectMode from './components/DetectMode';

function App() {
    const [mode, setMode] = useState('generate');

    return (
        <div className="app">
            <header className="app-header">
                <h1>Poketrix</h1>
                <nav>
                    <button onClick={() => setMode('generate')} className={mode === 'generate' ? 'active' : ''}>Generate</button>
                    <button onClick={() => setMode('detect')} className={mode === 'detect' ? 'active' : ''}>Detect</button>
                </nav>
            </header>
            <main>
                {mode === 'generate' ? <GenerateMode /> : <DetectMode />}
            </main>
        </div>
    );
}

export default App;