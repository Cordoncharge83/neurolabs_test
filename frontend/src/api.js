const API_BASE_URL = 'http://localhost:8000'

export async function fetchImages() {
  const response = await fetch(`${API_BASE_URL}/images`)
  if (!response.ok) {
    throw new Error('Failed to load shelf images')
  }
  const data = await response.json()
  return data.images ?? []
}

export async function analyzeShelf(imageId) {
  const response = await fetch(`${API_BASE_URL}/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ image_id: imageId }),
  })

  const data = await response.json()
  if (!response.ok) {
    throw new Error(data.detail || 'Analysis failed')
  }

  return data
}
