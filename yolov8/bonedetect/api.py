from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import base64
import time
import tempfile
import os
from pathlib import Path

# 导入 detect_and_classfiy.py 中的方法
from detect_and_classfiy import (
    draw_arthrosis,
    draw_arthrosis_keyjoint,
    get_keyjoint,
    get_score,
    calcBoneAge,
    export,
)

app = FastAPI(title="BoneDetect API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path(__file__).parent / "weights" / "yolov8m_detect.pt"


@app.get("/")
def root():
    return {"message": "BoneDetect API 运行中", "docs": "/docs"}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...), sex: str = Form(...)):
    """
    接收一张手部 X 光图像和性别，返回：
    - annotated_all: base64 编码的全量标记图
    - annotated_selected: base64 编码的筛选标记图
    - report: 诊断报告文本
    - processing_time: 处理耗时
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件")

    if sex not in ["boy", "girl"]:
        raise HTTPException(status_code=400, detail="性别必须为 'boy' 或 'girl'")

    start = time.time()

    # 把上传的图片写入临时文件
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_in:
        tmp_in.write(await file.read())
        input_path = tmp_in.name

    output_all = input_path.replace(suffix, f"_all{suffix}")
    output_selected = input_path.replace(suffix, f"_selected{suffix}")

    try:
        # 调用两个绘制方法
        draw_arthrosis(input_path, output_all)
        draw_arthrosis_keyjoint(input_path, output_selected)

        # 读取并编码两张图
        def read_b64(path):
            img = cv2.imread(path)
            if img is None:
                return None
            _, buf = cv2.imencode(".png", img)
            return base64.b64encode(buf).decode("utf-8")

        annotated_all = read_b64(output_all)
        annotated_selected = read_b64(output_selected)

        if not annotated_all or not annotated_selected:
            raise HTTPException(status_code=500, detail="标记图生成失败")

        # ── 计算骨龄和生成报告 ──
        _, _, images_keyjoint = get_keyjoint(input_path)
        total_score, scores = get_score(sex, images_keyjoint)
        bone_age = calcBoneAge(total_score, sex)
        report = export(scores, total_score, bone_age)

    finally:
        # 清理临时文件
        for p in [input_path, output_all, output_selected]:
            if os.path.exists(p):
                os.remove(p)

    elapsed = time.time() - start

    return {
        "success": True,
        "data": {
            "annotated_all": annotated_all,
            "annotated_selected": annotated_selected,
            "report": report,
            "processing_time": f"{elapsed:.2f}s",
        },
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=5000, reload=True)
