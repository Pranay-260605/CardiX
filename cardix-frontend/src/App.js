import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    if (selected) {
      setFile(selected);
      setPreview(URL.createObjectURL(selected));
      setResult('');
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    const formData = new FormData();
    formData.append('file', file);

    try {
      setLoading(true);
      const response = await axios.post('http://localhost:8000/predict/', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data.prediction);
    } catch (error) {
      console.error('Upload error:', error);
      setResult('Prediction failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      style={{
        backgroundColor: '#22150b',
        color: '#e9e8e3',
        minHeight: '100vh',
        padding: '20px',
        fontFamily: 'Arial, sans-serif',
        position: 'relative',
      }}
    >
      {/* Logo */}
      <img
        src="/cardix-logo.png"
        alt="Cardix Logo"
        style={{ width: '180px', position: 'absolute', top: '20px', left: '20px' }}
      />

      {/* Initial screen */}
      {!file && (
        <div
          style={{
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'center',
            alignItems: 'center',
          }}
        >
          <h1 style={{ fontSize: '1.8rem', fontWeight: 'bold', marginBottom: '30px' }}>
            Please Upload Your X-Ray Scans
          </h1>

          <div
            style={{
              backgroundColor: '#c9c7be',
              borderRadius: '50px',
              display: 'flex',
              alignItems: 'center',
              padding: '12px 20px',
              width: '400px',
              justifyContent: 'space-between',
              gap: '20px',
            }}
          >
            <label
              htmlFor="file-upload"
              style={{
                cursor: 'pointer',
                fontSize: '1.25rem',
                fontWeight: 'bold',
                color: '#22150b',
              }}
            >
              +
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            <button
              disabled
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                fontSize: '1rem',
                fontWeight: 'bold',
                color: '#22150b',
                cursor: 'not-allowed',
              }}
            >
              Diagnose
            </button>
          </div>
        </div>
      )}

      {/* After upload */}
      {file && (
        <>
          <div
            style={{
              marginTop: '100px',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'flex-start',
            }}
          >
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-start' }}>
              {/* Uploaded image */}
              {preview && (
                <img
                  src={preview}
                  alt="Preview"
                  style={{
                    maxWidth: '244px',
                    borderRadius: '10px',
                    marginLeft:'400px',
                    border: '2px solid #deddd6',
                  }}
                />
              )}

              {/* Prediction below image */}
              {result && (
                <div
                  style={{
                    marginTop: '20px',
                    backgroundColor: '#c9c7be',
                    color: '#22150b',
                    padding: '15px 15px',
                    borderRadius: '40px',
                    fontSize: '1rem',
                    fontWeight: 'bold',
                  }}
                >
                  The diagnosis says : {result}
                </div>
              )}
            </div>
          </div>

          {/* Bottom upload bar */}
          <div
            style={{
              position: 'fixed',
              bottom: '100px',
              left: '50%',
              transform: 'translateX(-50%)',
              backgroundColor: '#c9c7be',
              borderRadius: '50px',
              display: 'flex',
              alignItems: 'center',
              padding: '12px 20px',
              width: '500px',
              justifyContent: 'space-between',
              gap: '20px',
            }}
          >
            <label
              htmlFor="file-upload"
              style={{
                cursor: 'pointer',
                fontSize: '1.25rem',
                fontWeight: 'bold',
                color: '#22150b',
              }}
            >
              +
            </label>
            <input
              id="file-upload"
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />

            <button
              onClick={handleUpload}
              disabled={loading}
              style={{
                backgroundColor: 'transparent',
                border: 'none',
                fontSize: '1rem',
                fontWeight: 'bold',
                color: '#22150b',
                cursor: 'pointer',
              }}
            >
              Diagnose
            </button>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
