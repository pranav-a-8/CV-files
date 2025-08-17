// frontend/src/App.js
import React, { useState } from 'react';
import InputForm from './components/InputForm';
import Preview from './components/Preview';
import DownloadButton from './components/DownloadButton';

function App() {
  const [projectId, setProjectId] = useState(null);
  const [error, setError] = useState(null);

  const handleGenerate = (id) => {
    setProjectId(id);
    setError(null);
  };

  return (
    <div>
      <div className="navbar">WebGenAI - Create Websites with AI</div>
      <div className="container">
        <InputForm onGenerate={handleGenerate} />
        {projectId && <Preview projectId={projectId} />}
        {projectId && <DownloadButton projectId={projectId} />}
        {error && <div className="error">{error}</div>}
      </div>
    </div>
  );
}

export default App;