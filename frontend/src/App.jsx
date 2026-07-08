import { useEffect, useState } from 'react'
import './App.css'
import { analyzeShelf, fetchImages } from './api'

function App() {
  const [images, setImages] = useState([])
  const [selectedImageId, setSelectedImageId] = useState('')
  const [loadingImages, setLoadingImages] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [analysisResult, setAnalysisResult] = useState(null)

  useEffect(() => {
    const loadImages = async () => {
      try {
        const availableImages = await fetchImages()
        setImages(availableImages)
        if (availableImages.length > 0) {
          setSelectedImageId(availableImages[0].image_id)
        }
      } catch (err) {
        setError(err.message || 'Unable to load shelf images')
      } finally {
        setLoadingImages(false)
      }
    }

    loadImages()
  }, [])

  const handleAnalyze = async () => {
    if (!selectedImageId) {
      setError('Please select a shelf image first')
      return
    }

    setAnalyzing(true)
    setError('')
    setAnalysisResult(null)

    try {
      const result = await analyzeShelf(selectedImageId)
      setAnalysisResult(result)
    } catch (err) {
      setError(err.message || 'Analysis failed')
    } finally {
      setAnalyzing(false)
    }
  }

  return (
    <main className="app-shell">
      <section className="panel">
        <h1>Shelf Recognition Demo</h1>
        <p className="intro">Select a demo shelf image and run the recognition workflow.</p>

        <div className="controls">
          <label htmlFor="image-select" className="field-label">
            Available shelf images
          </label>
          <select
            id="image-select"
            value={selectedImageId}
            onChange={(event) => setSelectedImageId(event.target.value)}
            disabled={loadingImages || images.length === 0}
          >
            {loadingImages ? (
              <option value="">Loading images...</option>
            ) : images.length === 0 ? (
              <option value="">No images available</option>
            ) : (
              images.map((image) => (
                <option key={image.image_id} value={image.image_id}>
                  {image.image_id}
                </option>
              ))
            )}
          </select>

          <button type="button" onClick={handleAnalyze} disabled={analyzing || !selectedImageId}>
            {analyzing ? 'Analyzing…' : 'Analyze Shelf'}
          </button>
        </div>

        {error ? <div className="message error">{error}</div> : null}

        <div className="response-panel">
          <h2>Response</h2>
          {analysisResult ? (
            <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
          ) : (
            <p className="placeholder">No analysis run yet.</p>
          )}
        </div>
      </section>
    </main>
  )
}

export default App
