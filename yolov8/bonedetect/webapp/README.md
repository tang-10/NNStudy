# BoneDetect - 手部骨龄检测系统

## 目录结构

```
bonedetect/
├── api.py              # FastAPI 后端（对接 YOLOv8）
├── detect.py           # 原始检测脚本
├── requirements.txt    # Python 依赖
├── weights/
│   └── yolov8m_detect.pt   # 模型权重（放这里）
└── webapp/             # React 前端
    ├── package.json
    ├── vite.config.js
    ├── index.html
    └── src/
        ├── App.jsx
        ├── index.css
        ├── main.jsx
        └── components/
            ├── ImageUploader.jsx
            └── ResultDisplay.jsx
```

## 启动方式

### 1. 后端

```bash
cd D:\projects\yolov8\bonedetect

# 安装依赖
pip install -r requirements.txt

# 启动 API 服务
python api.py
# → http://localhost:5000
# → API 文档: http://localhost:5000/docs
```

### 2. 前端

```bash
cd D:\projects\yolov8\bonedetect\webapp

# 安装依赖
npm install

# 启动开发服务器
npm run dev
# → http://localhost:3000
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/health` | 健康检查 |
| POST | `/api/detect` | 上传图片，返回标记图 + 关节列表 |

### POST /api/detect 响应格式

```json
{
  "success": true,
  "data": {
    "annotated_image": "<base64 PNG>",
    "joints": [
      { "name": "关节名", "confidence": 0.98, "bbox": [x1, y1, x2, y2] }
    ],
    "summary": "检测到 13 个关节，平均置信度 95.2%",
    "processing_time": "0.83s"
  }
}
```

## 数据流

```
用户上传 X 光图片
    ↓ POST /api/detect
FastAPI 接收图片
    ↓
YOLOv8 检测 13 个关节
    ↓
OpenCV 绘制检测框
    ↓
返回 base64 标记图 + 关节列表
    ↓
前端展示标记图 + 置信度列表
```
