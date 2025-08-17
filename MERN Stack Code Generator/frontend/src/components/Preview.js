// frontend/src/components/Preview.js
import React, { useState, useEffect } from 'react';
import { checkStatus, stopPreview } from '../services/api';

const Preview = ({ projectId }) => {
  const [previewUrl, setPreviewUrl] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!projectId) return;

    // Clean up previous preview if projectId changes
    const cleanupPreviousPreview = async () => {
      try {
        await stopPreview(projectId);
      } catch (err) {
        console.warn(`Failed to stop previous preview: ${err.message}`);
      }
    };

    const fetchPreview = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await checkStatus(projectId);
        setPreviewUrl(response.preview_url);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    cleanupPreviousPreview().then(fetchPreview);

    // Cleanup on unmount or projectId change
    return () => {
      cleanupPreviousPreview();
    };
  }, [projectId]);

  if (loading) return <div>Loading preview...</div>;
  if (error) return <div>Error: {error}</div>;
  if (!previewUrl) return <div>No preview available</div>;

  return (
    <iframe
      src={previewUrl}
      style={{ width: '100%', height: '500px', border: 'none' }}
      title="Project Preview"
    />
  );
};

export default Preview;