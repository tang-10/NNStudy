import { useState } from 'react'
import ImageUploader from './components/ImageUploader'
import ResultDisplay from './components/ResultDisplay'

const API_BASE = '/api'

function App() {
  const [result, setResult] = useState(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [uploadedImage, setUploadedImage] = useState(null)
  const [error, setError] = useState(null)
  const [activeTab, setActiveTab] = useState('upload')  // 侧边栏活跃标签

  const handleImageUpload = async (imageFile, previewUrl, sex) => {
    setUploadedImage(previewUrl)
    setIsAnalyzing(true)
    setResult(null)
    setError(null)
    setActiveTab('result')  // 自动切换到结果标签

    try {
      const formData = new FormData()
      formData.append('file', imageFile)
      formData.append('sex', sex)

      const response = await fetch(`${API_BASE}/detect`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`服务器错误: ${response.status}`)
      }

      const data = await response.json()
      setResult(data)
      setActiveTab('report')  // 检测完成后自动切换到报告
    } catch (err) {
      setError(err.message || '检测失败，请重试')
      setActiveTab('result')
    } finally {
      setIsAnalyzing(false)
    }
  }

  const handleClear = () => {
    setResult(null)
    setUploadedImage(null)
    setIsAnalyzing(false)
    setError(null)
    setActiveTab('upload')
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-gray-100">
      {/* Header */}
      <header className="bg-white shadow-sm sticky top-0 z-40">
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

      {/* Main Layout */}
      <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
          
          {/* 左侧：主内容区 (3列) */}
          <div className="lg:col-span-3 space-y-6">
            
            {/* 错误提示 */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start space-x-3">
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

            {/* 上传区 */}
            <div id="upload-section">
              <ImageUploader
                onUpload={handleImageUpload}
                onClear={handleClear}
                isAnalyzing={isAnalyzing}
              />
            </div>

            {/* 检测结果 */}
            {(uploadedImage || result) && (
              <div id="result-section">
                <ResultDisplay
                  result={result}
                  isAnalyzing={isAnalyzing}
                  originalImage={uploadedImage}
                  showImagesOnly={true}
                />
              </div>
            )}
          </div>

          {/* 右侧：侧边栏 (1列) */}
          <aside className="lg:col-span-1">
            <div className="sticky top-24 space-y-4">
              
              {/* 导航菜单 */}
              <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                <div className="p-4 bg-primary-600 text-white">
                  <h3 className="font-semibold text-sm">📍 导航</h3>
                </div>
                <nav className="divide-y">
                  <button
                    onClick={() => {
                      setActiveTab('upload')
                      document.getElementById('upload-section')?.scrollIntoView({ behavior: 'smooth' })
                    }}
                    className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors ${
                      activeTab === 'upload'
                        ? 'bg-primary-50 text-primary-600 border-l-4 border-primary-600'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    📤 上传图片
                  </button>
                  <button
                    onClick={() => {
                      setActiveTab('result')
                      document.getElementById('result-section')?.scrollIntoView({ behavior: 'smooth' })
                    }}
                    disabled={!uploadedImage}
                    className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors ${
                      activeTab === 'result'
                        ? 'bg-primary-50 text-primary-600 border-l-4 border-primary-600'
                        : uploadedImage
                        ? 'text-gray-700 hover:bg-gray-50'
                        : 'text-gray-300 cursor-not-allowed'
                    }`}
                  >
                    🦴 检测结果
                  </button>
                  <button
                    onClick={() => setActiveTab('report')}
                    disabled={!result?.success}
                    className={`w-full text-left px-4 py-3 text-sm font-medium transition-colors ${
                      activeTab === 'report'
                        ? 'bg-primary-50 text-primary-600 border-l-4 border-primary-600'
                        : result?.success
                        ? 'text-gray-700 hover:bg-gray-50'
                        : 'text-gray-300 cursor-not-allowed'
                    }`}
                  >
                    📋 诊断报告
                  </button>
                </nav>
              </div>

              {/* 诊断报告卡片 */}
              {result?.success && result.data.report && (
                <div className="bg-white rounded-xl shadow-lg overflow-hidden">
                  <div className="p-4 bg-green-600 text-white">
                    <h3 className="font-semibold text-sm">✅ 诊断报告</h3>
                  </div>
                  <div className="p-4 space-y-3">
                    <div className="bg-green-50 rounded-lg p-3 border border-green-200">
                      <p className="text-xs text-green-700 leading-relaxed whitespace-pre-wrap font-mono">
                        {result.data.report}
                      </p>
                    </div>
                    <button
                      onClick={() => {
                        const text = result.data.report
                        navigator.clipboard.writeText(text).then(() => {
                          alert('报告已复制到剪贴板')
                        })
                      }}
                      className="w-full py-2 px-3 bg-green-50 hover:bg-green-100 text-green-600 rounded-lg text-xs font-medium transition-colors border border-green-200"
                    >
                      📋 复制报告
                    </button>
                  </div>
                </div>
              )}

              {/* 处理时间 */}
              {result?.success && (
                <div className="bg-white rounded-xl shadow-lg p-4">
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-600">⏱️ 处理耗时</span>
                    <span className="text-sm font-bold text-primary-600">{result.data.processing_time}</span>
                  </div>
                </div>
              )}

              {/* 使用提示 */}
              <div className="bg-blue-50 rounded-xl p-4 border border-blue-200">
                <h4 className="text-xs font-semibold text-blue-900 mb-2">💡 使用提示</h4>
                <ul className="text-xs text-blue-800 space-y-1">
                  <li>✓ 选择性别后上传</li>
                  <li>✓ 点击图片可放大查看</li>
                  <li>✓ 报告可直接复制</li>
                </ul>
              </div>
            </div>
          </aside>

        </div>

        {/* 使用说明 */}
        <div className="mt-12 bg-white rounded-xl shadow-lg p-6">
          <h2 className="text-sm font-semibold text-gray-700 mb-4">📖 使用说明</h2>
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm text-gray-600">
            {[
              { step: '1', text: '选择性别（男/女）' },
              { step: '2', text: '上传手部正位 X 光图像' },
              { step: '3', text: '系统自动定位关键关节' },
              { step: '4', text: '查看检测结果和骨龄报告' },
            ].map(({ step, text }) => (
              <div key={step} className="flex items-start space-x-3">
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
