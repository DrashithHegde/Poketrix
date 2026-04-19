import React, { useState } from 'react';
import axios from 'axios';

function DetectMode() {
    const [file, setFile] = useState(null);
    const [type, setType] = useState('Fire');
    const [color, setColor] = useState('Red');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleDetect = async () => {
        if (!file) return;
        setLoading(true);
        const formData = new FormData();
        formData.append('file', file);
        formData.append('type', type);
        formData.append('color', color);

        try {
            const response = await axios.post('http://localhost:8000/detect', formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setResult(response.data);
        } catch (error) {
            console.error('Error detecting image:', error);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="detect-mode">
            <h2>Detect Real vs Fake</h2>

            <div style={{ marginBottom: '15px' }}>
                <label>Claimed Type:</label>
                <select value={type} onChange={(e) => setType(e.target.value)}>
                    <option>Fire</option>
                    <option>Water</option>
                    <option>Grass</option>
                    <option>Electric</option>
                </select>
            </div>
            <div style={{ marginBottom: '15px' }}>
                <label>Claimed Color:</label>
                <select value={color} onChange={(e) => setColor(e.target.value)}>
                    <option>Red</option>
                    <option>Blue</option>
                    <option>Green</option>
                    <option>Yellow</option>
                </select>
            </div>

            <input type="file" onChange={handleFileChange} />
            <button onClick={handleDetect} disabled={loading || !file}>{loading ? 'Detecting...' : 'Detect'}</button>
            {result && (
                <div style={{ marginTop: '20px', padding: '10px', border: '2px solid #ccc', borderRadius: '5px' }}>
                    <p><strong>Discriminator Verdict:</strong></p>
                    <p>Real Probability: {(result.real_probability * 100).toFixed(2)}%</p>
                    <p>Fake Probability: {(result.fake_probability * 100).toFixed(2)}%</p>
                </div>
            )}
        </div>
    );
}

export default DetectMode;