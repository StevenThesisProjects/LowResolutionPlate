"""Request and response models; FastAPI turns them into the OpenAPI spec."""
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class Candidate(BaseModel):
    plate: str = Field(description="Chuỗi ứng viên")
    prob: float = Field(description="p(chuỗi | track), tính đúng bằng thuật toán forward của CTC")


class FrameInfo(BaseModel):
    index: int
    name: str = Field(description="Tên file, hoặc tên trong file zip")
    width: int
    height: int
    weight: float = Field(description="Trọng số attention của frame trong fusion (cộng các frame = 1)")
    sharpness: float = Field(description="Phương sai Laplacian của ảnh xám 32x128 (cao = nét)")


class Triage(BaseModel):
    needs_review: bool
    reasons: List[str] = Field(description="low_confidence | invalid_layout | not_calibrated")
    score_name: str = Field(description="Điểm dùng để phân loại (chọn trên val 999)")
    score: float
    threshold: Optional[float] = Field(description="Ngưỡng cho đúng số frame của track")


class Timing(BaseModel):
    read_ms: float
    model_ms: float
    decode_ms: float


class PredictResponse(BaseModel):
    request_id: str
    plate: str = Field(description="Biển số đọc được")
    confidence: float = Field(description="Trung bình độ tin cậy từng ký tự (định nghĩa của cuộc thi)")
    sequence_prob: float = Field(description="Xác suất CTC của cả chuỗi")
    char_conf: List[float] = Field(description="Độ tin cậy từng ký tự")
    top3: List[Candidate] = Field(description="Phần tử đầu luôn là `plate`; sau đó là các chuỗi "
                                              "khả dĩ nhất tiếp theo (số phần tử = top_k)")
    layout: Optional[str] = Field(description="Mercosur | Brazilian | null nếu sai cú pháp")
    n_frames: int
    frame_weights: List[float]
    sharpness: List[float]
    frames: List[FrameInfo]
    needs_review: bool
    review_reasons: List[str]
    triage: Triage
    warnings: List[str]
    expected_accuracy: Optional[float] = Field(
        description="Accuracy trung bình trên lock-box với đúng số frame này (E3, 4 seed)")
    reference_plate: Optional[str] = Field(
        default=None, description="Nhãn đúng, chỉ có khi zip chứa annotations.json")
    decoder: str
    latency_ms: float
    timing: Timing
    model_version: str


class ErrorBody(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    error: ErrorBody
    request_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    request_id: str = Field(min_length=8, max_length=64)
    correct: bool
    corrected_plate: Optional[str] = Field(
        default=None, pattern=r"^[A-Za-z0-9]{1,10}$",
        description="Biển số đúng nếu kết quả sai (chữ và số, tối đa 10 ký tự)")


class FeedbackResponse(BaseModel):
    feedback_id: int
    request_id: str


class HistoryItem(BaseModel):
    request_id: str
    created_at: str
    n_frames: int
    plate: str
    confidence: float
    needs_review: bool
    decoder: str
    latency_ms: float
    feedback_correct: Optional[bool] = None
    corrected_plate: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = Field(description="ok | unavailable")
    model_loaded: bool
    model_version: Optional[str] = None
    device: Optional[str] = None
    uptime_s: float
    detail: Optional[str] = None


class ModelInfo(BaseModel):
    model_version: str
    checkpoint: str
    sha256: str
    sha256_verified: bool
    parameters: int
    default_decoder: str
    decoders: List[str]
    triage_score: str
    thresholds_by_frames: Dict[str, Optional[float]]
    expected_accuracy_by_frames: Dict[str, float]
    limits: Dict[str, int]
    scope: str
    selection_rule: str
