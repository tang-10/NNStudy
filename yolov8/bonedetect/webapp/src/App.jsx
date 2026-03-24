import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import ResultDisplay from './components/ResultDisplay'

const API_BASE = '/api'  // 通过 vite proxy 转发到 http://localhost:5000

function App() {
  const [result, setResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [error, setError] = useState(null)

  const handleImageUpload = async (imageFile, previewUrl) => {
    setUploadedImage(previewUrl)
    setIsAnalyzing(true)
    setResult(null)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('file', imageFile)

      const response = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`服务器错误: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err.message || '检测失败，请重试')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleClear = () => {
    setResult(null)
    setUploadedImage(null)
    setIsAnalyzing(false)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                    d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z" />
                </svg>
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">BoneDetect</h1>
                <p className="text-xs text-gray-400">手部骨龄检测系统</p>
              </div>
            </div>
            <span className="text-sm text-gray-400 hidden sm:block">基于 YOLOv8 关节检测</span>
          </div>
        </div>
      </header>

      {/* Main */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">

        {/* 错误提示 */}
        {error && (
          <div className="mb-6 bg-red-50 border border-red-200 rounded-xl p-4 flex items-start space-x-3">
            <svg className="w-5 h-5 text-red-500 mt-0.5 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-red-700">检测失败</p>
              <p className="text-sm text-red-600 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* 左：上传 */}
          <ImageUploader
            onUpload={handleImageUpload}
            onClear={handleClear}
            isAnalyzing={isAnalyzing}
          />

          {/* 右：结果 */}
          <ResultDisplay
            result={result}
            isAnalyzing={isAnalyzing}
            originalImage={uploadedImage}
          />
        </div>

        {/* 使用说明 */}
        <div className="mt-8 card">
          <h2 className="text-sm font-semibold text-gray-700 mb-3">使用说明</h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-500">
            {[
              { step: '1', text: '上传手部正位 X 光图像（JPG / PNG）' },
              { step: '2', text: '系统自动定位 13 个关键关节并标记' },
              { step: '3', text: '查看标记结果与各关节置信度' },
            ].map(({ step, text }) => (
              <div key={step} className="flex items-start space-x-2">
                <span className="flex-shrink-0 w-6 h-6 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-xs font-semibold">
                  {step}
                </span>
                <p>{text}</p>
              </div>
            ))}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-white border-t mt-12">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <p className="text-center text-xs text-gray-400">
            BoneDetect © 2024 · 仅供研究使用，不作为临床诊断依据
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
