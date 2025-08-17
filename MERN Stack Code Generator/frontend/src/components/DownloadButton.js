// frontend/src/components/DownloadButton.js
import React from 'react';
import { downloadZip } from '../services/api';

const DownloadButton = ({ projectId }) => {
  const handleDownload = () => {
    if (projectId) {
      downloadZip(projectId);
    }
  };

  return (
    <button onClick={handleDownload} disabled={!projectId}>
      Download Project ZIP
    </button>
  );
};

export default DownloadButton;