import React, { useState } from 'react';
import axios from 'axios';

function GenerateMode() {
    const [type, setType] = useState('Fire');
    const [color, setColor] = useState('Red');
    const [generatedImage, setGeneratedImage] = useState(null);
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleGenerate = async () => {
        setLoading(true);
        try {
            const response = await axios.post('http://localhost:8000/generate', { type, color });
            setGeneratedImage(`data:image/png;base64,${response.data.image}`);
            setResult(response.data);
        } catch (error) {
            console.error('Error generating image:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="generate-mode">
            <h2>Generate Pokémon</h2>
            <div>
                <label>Type:</label>
                <select value={type} onChange={(e) => setType(e.target.value)}>
                    <option>Fire</option>
                    <option>Water</option>
                    <option>Grass</option>
                    <option>Electric</option>
                </select>
            </div>
            <div>
                <label>Color:</label>
                <select value={color} onChange={(e) => setColor(e.target.value)}>
                    <option>Red</option>
                    <option>Blue</option>
                    <option>Green</option>
                    <option>Yellow</option>
                </select>
            </div>
            <button onClick={handleGenerate} disabled={loading}>{loading ? 'Generating...' : 'Generate'}</button>
            {generatedImage && (
                <div>
                    <img src={generatedImage} alt="Generated Pokémon" />
                    {result && result.real_probability !== undefined && (
                        <div style={{ marginTop: '15px' }}>
                            <p><strong>Discriminator Target Score:</strong></p>
                            <p>Real Probability: {(result.real_probability * 100).toFixed(2)}%</p>
                            <p>Fake Probability: {(result.fake_probability * 100).toFixed(2)}%</p>
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}

export default GenerateMode;