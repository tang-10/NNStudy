import { useState, useRef, useEffect } from 'react'

export default function ResultDisplay({ result, isAnalyzing, originalImage, showImagesOnly }) {
  const [preview, setPreview] = useState(null)
  const [scale, setScale] = useState(1)
  const [position, setPosition] = useState({ x: 0, y: 0 })
  const [dragging, setDragging] = useState(false)
  const dragStart = useRef({ x: 0, y: 0 })
  const imgRef = useRef(null)

  const openPreview = (imgData) => {
    setPreview(imgData)
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }

  const closePreview = () => {
    setPreview(null)
    setScale(1)
    setPosition({ x: 0, y: 0 })
  }

  const handleWheel = (e) => {
    e.preventDefault()
    const delta = e.deltaY > 0 ? -0.1 : 0.1
    setScale(prev => Math.min(Math.max(0.5, prev + delta), 5))
  }

  const handleMouseDown = (e) => {
    if (e.button !== 0) return
    setDragging(true)
    dragStart.current = {
      x: e.clientX - position.x,
      y: e.clientY - position.y
    }
  }

  const handleMouseMove = (e) => {
    if (!dragging) return
    setPosition({
      x: e.clientX - dragStart.current.x,
      y: e.clientY - dragStart.current.y
    })
  }

  const handleMouseUp = () => {
    setDragging(false)
  }

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && preview) {
        closePreview()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [preview])

  /* ── 空状态 ── */
  if (!originalImage && !result) {
    return (
      <div className="card min-h-[400px] flex items-center justify-center">
        <div className="text-center text-gray-400 space-y-3">
          <div className="w-20 h-20 bg-gray-100 rounded-full flex items-center justify-center mx-auto">
            <svg className="w-10 h-10 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
          </div>
          <p className="font-medium">等待上传图片</p>
          <p className="text-sm">上传 X 光图像后，检测结果将显示在此处</p>
        </div>
      </div>
    )
  }

  /* ── 分析中 ── */
  if (isAnalyzing) {
    return (
      <div className="card min-h-[400px] flex items-center justify-center">
        <div className="text-center space-y-4">
          <div className="w-24 h-24 bg-primary-100 rounded-full flex items-center justify-center mx-auto animate-pulse">
            <svg className="w-12 h-12 text-primary-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5}
                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <p className="text-gray-700 font-medium">YOLOv8 正在检测...</p>
          <p className="text-sm text-gray-400">正在计算骨龄，请稍候</p>
        </div>
      </div>
    )
  }

  /* ── 有结果 ── */
  if (result?.success) {
    const { annotated_all, annotated_selected, processing_time } = result.data

    return (
      <>
        <div className="card space-y-5">
          <h2 className="text-lg font-semibold text-gray-900 flex items-center space-x-2">
            <span>🦴</span><span>检测结果</span>
          </h2>

          {/* 两张图并排展示 */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* 全量检测图 */}
            <div className="space-y-2">
              <div
                className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50 cursor-zoom-in hover:border-primary-400 transition-colors"
                onClick={() => openPreview(annotated_all)}
              >
                <img
                  src={`data:image/png;base64,${annotated_all}`}
                  alt="全量关节标记"
                  className="w-full object-contain"
                />
                <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
                  全量检测
                </div>
                <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded opacity-0 hover:opacity-100 transition-opacity">
                  点击放大
                </div>
              </div>
              <p className="text-xs text-gray-400 text-center">所有检测到的关节</p>
            </div>

            {/* 筛选检测图 */}
            <div className="space-y-2">
              <div
                className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50 cursor-zoom-in hover:border-primary-400 transition-colors"
                onClick={() => openPreview(annotated_selected)}
              >
                <img
                  src={`data:image/png;base64,${annotated_selected}`}
                  alt="筛选关节标记"
                  className="w-full object-contain"
                />
                <div className="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded">
                  筛选结果
                </div>
                <div className="absolute bottom-2 right-2 bg-black/60 text-white text-xs px-2 py-1 rounded opacity-0 hover:opacity-100 transition-opacity">
                  点击放大
                </div>
              </div>
              <p className="text-xs text-gray-400 text-center">筛选后的关键关节</p>
            </div>
          </div>

          {/* 处理时间 */}
          <div className="flex items-center justify-end text-sm text-gray-400 space-x-1 pt-2 border-t border-gray-100">
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>处理耗时 <span className="font-medium text-primary-600">{processing_time}</span></span>
          </div>
        </div>

        {/* 图片预览弹窗 */}
        {preview && (
          <div
            className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
            onClick={closePreview}
            onWheel={handleWheel}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={handleMouseUp}
          >
            {/* 关闭按钮 */}
            <button
              className="absolute top-4 right-4 text-white/80 hover:text-white z-10"
              onClick={closePreview}
            >
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>

            {/* 操作提示 */}
            <div className="absolute top-4 left-1/2 -translate-x-1/2 text-white/60 text-sm space-x-4">
              <span>滚轮缩放</span>
              <span>·</span>
              <span>拖拽移动</span>
              <span>·</span>
              <span>ESC 或点击背景关闭</span>
            </div>

            {/* 缩放指示 */}
            <div className="absolute bottom-4 left-1/2 -translate-x-1/2 text-white/60 text-sm">
              {Math.round(scale * 100)}%
            </div>

            {/* 图片 */}
            <img
              ref={imgRef}
              src={`data:image/png;base64,${preview}`}
              alt="预览"
              className="max-w-full max-h-full select-none"
              style={{
                transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
                cursor: dragging ? 'grabbing' : scale > 1 ? 'grab' : 'default',
                transition: dragging ? 'none' : 'transform 0.1s ease-out'
              }}
              onClick={(e) => e.stopPropagation()}
              onMouseDown={handleMouseDown}
              draggable={false}
            />
          </div>
        )}
      </>
    )
  }

  /* ── 检测失败 ── */
  return (
    <div className="card min-h-[400px] flex items-center justify-center">
      <div className="text-center space-y-2">
        <p className="font-medium text-red-500">检测未返回结果</p>
        <p className="text-sm text-gray-400">请检查后端服务是否正常运行</p>
      </div>
    </div>
  )
}
