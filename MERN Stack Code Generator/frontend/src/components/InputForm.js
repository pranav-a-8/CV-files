import React, { useState } from 'react';
import { generateProject } from '../services/api';

const InputForm = ({ onGenerate }) => {
  const [description, setDescription] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleGenerate = async () => {
    if (!description.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const { id } = await generateProject(description);
      onGenerate(id); // Pass project_id to parent for polling
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <textarea
        value={description}
        onChange={(e) => setDescription(e.target.value)}
        placeholder="Describe your website (e.g., 'Create a simple blog with posts and comments')"
      />
      <button onClick={handleGenerate} disabled={loading}>
        {loading ? 'Generating...' : 'Generate'}
      </button>
      {error && <div className="error">{error}</div>}
    </div>
  );
};

export default InputForm;