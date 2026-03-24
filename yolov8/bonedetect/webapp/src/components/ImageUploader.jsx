import { useState, useRef } from 'react'

export default function ImageUploader({ onUpload, onClear, isAnalyzing }) {
  const [preview, setPreview] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const inputRef = useRef(null)

  const handleFile = (file) => {
    if (!file.type.startsWith('image/')) {
      alert('请上传图片文件（JPG / PNG）')
      return
    }
    const reader = new FileReader()
    reader.onload = (e) => {
      setPreview(e.target.result)
      onUpload(file, e.target.result)
    }
    reader.readAsDataURL(file)
  }

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(e.type === 'dragenter' || e.type === 'dragover')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    if (e.dataTransfer.files?.[0]) handleFile(e.dataTransfer.files[0])
  }

  const handleClear = () => {
    setPreview(null)
    if (inputRef.current) inputRef.current.value = ''
    onClear()
  }

  return (
    <div className="card flex flex-col">
      <h2 className="text-lg font-semibold text-gray-900 mb-4 flex items-center space-x-2">
        <span>📤</span><span>上传 X 光图像</span>
      </h2>

      {!preview ? (
        /* 拖拽上传区 */
        <div
          className={`flex-1 border-2 border-dashed rounded-xl p-10 text-center transition-all duration-200 cursor-pointer ${
            dragActive
              ? 'border-primary-500 bg-primary-50'
              : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
        >
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
          />
          <div className="space-y-4">
            <div className="flex justify-center">
              <div className="w-20 h-20 bg-primary-100 rounded-full flex items-center justify-center">
                <svg className="w-10 h-10 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                    d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                </svg>
              </div>
            </div>
            <div>
              <p className="text-gray-700 font-medium">点击或拖拽图片到此处</p>
              <p className="text-sm text-gray-400 mt-1">支持 JPG、PNG 格式</p>
            </div>
            <button
              type="button"
              className="btn-primary"
              onClick={(e) => { e.stopPropagation(); inputRef.current?.click() }}
            >
              选择图片
            </button>
          </div>
        </div>
      ) : (
        /* 预览区 */
        <div className="space-y-4">
          <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
            <img
              src={preview}
              alt="原始图像"
              className="w-full h-72 object-contain"
            />
            {isAnalyzing && (
              <div className="absolute inset-0 bg-white/75 flex flex-col items-center justify-center space-y-3">
                <div className="animate-spin rounded-full h-12 w-12 border-4 border-primary-500 border-t-transparent" />
                <p className="text-primary-600 font-medium text-sm">正在检测关节...</p>
              </div>
            )}
            {/* 图片标签 */}
            <div className="absolute top-2 left-2 bg-black/50 text-white text-xs px-2 py-1 rounded">
              原始图像
            </div>
          </div>

          <button
            type="button"
            onClick={handleClear}
            disabled={isAnalyzing}
            className="w-full py-2 px-4 border border-gray-300 rounded-lg text-sm text-gray-600 hover:bg-gray-50 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            🗑️ 清除，重新上传
          </button>
        </div>
      )}
    </div>
  )
}
