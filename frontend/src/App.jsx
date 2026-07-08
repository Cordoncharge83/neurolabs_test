import { useEffect, useState } from 'react'
import './App.css'
import { analyzeShelf, fetchImages } from './api'

function formatSimilarity(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return '—'
  }

  return `${(value * 100).toFixed(2)}%`
}

function App() {
  const [images, setImages] = useState([])
  const [selectedImageId, setSelectedImageId] = useState('')
  const [loadingImages, setLoadingImages] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [error, setError] = useState('')
  const [analysisResult, setAnalysisResult] = useState(null)

  const selectedImage = images.find((image) => image.image_id === selectedImageId) ?? null
  const brandDistribution = Object.entries(analysisResult?.summary?.brand_counts ?? {})
    .map(([brand, count]) => ({ brand: brand || 'Unknown', count: Number(count) || 0 }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
  const maxBrandCount = brandDistribution[0]?.count ?? 0

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

        {selectedImage ? (
          <div className="preview-card">
            <div className="preview-label">Selected shelf image</div>
            {selectedImage.image_url ? (
              <img src={selectedImage.image_url} alt={selectedImage.image_id} className="preview-image" />
            ) : (
              <div className="preview-placeholder">Preview unavailable</div>
            )}
            <div className="preview-caption">{selectedImage.filename}</div>
          </div>
        ) : null}

        {error ? <div className="message error">{error}</div> : null}

        <div className="response-panel">
          <h2>Results</h2>
          {analysisResult ? (
            <>
              <div className="kpi-grid">
                <div className="kpi-card">
                  <span className="kpi-label">Total products</span>
                  <strong>{analysisResult.summary?.total_products ?? 0}</strong>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">Unique SKUs</span>
                  <strong>{analysisResult.summary?.unique_skus ?? 0}</strong>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">Average similarity</span>
                  <strong>{formatSimilarity(analysisResult.summary?.average_similarity)}</strong>
                </div>
                <div className="kpi-card">
                  <span className="kpi-label">Top brand</span>
                  <strong>{analysisResult.summary?.brand_counts && Object.keys(analysisResult.summary.brand_counts).length > 0 ? Object.entries(analysisResult.summary.brand_counts).sort((a, b) => b[1] - a[1])[0][0] : '—'}</strong>
                </div>
              </div>

              {brandDistribution.length > 0 ? (
                <div className="brand-chart">
                  <div className="brand-chart-header">
                    <h3>Brand distribution</h3>
                    <span>Top 10 by product count</span>
                  </div>
                  <div className="brand-chart-rows">
                    {brandDistribution.map(({ brand, count }) => (
                      <div className="brand-chart-row" key={brand}>
                        <div className="brand-chart-meta">
                          <span className="brand-name">{brand}</span>
                          <span className="brand-count">{count}</span>
                        </div>
                        <div className="brand-bar-track" aria-hidden="true">
                          <div className="brand-bar" style={{ width: `${maxBrandCount > 0 ? (count / maxBrandCount) * 100 : 0}%` }} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              <div className="table-wrapper">
                <table>
                  <thead>
                    <tr>
                      <th>box_id</th>
                      <th>pred_sku_id</th>
                      <th>name</th>
                      <th>brand</th>
                      <th>similarity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(analysisResult.predictions ?? []).map((prediction) => (
                      <tr key={prediction.box_id}>
                        <td>{prediction.box_id}</td>
                        <td>{prediction.pred_sku_id}</td>
                        <td>{prediction.name}</td>
                        <td>{prediction.brand}</td>
                        <td>{formatSimilarity(prediction.similarity)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="placeholder">No analysis run yet.</p>
          )}
        </div>
      </section>
    </main>
  )
}

export default App
