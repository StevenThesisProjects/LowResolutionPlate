export const pct = (x, digits = 1) => (x == null ? '—' : `${(x * 100).toFixed(digits)}%`);

export const confLevel = (c) => (c >= 0.95 ? 'ok' : c >= 0.8 ? 'mid' : 'low');

export const REASONS = {
  low_confidence: (n) => `Độ tin cậy thấp hơn ngưỡng dành cho ${n} frame`,
  invalid_layout: () => 'Chuỗi đọc được không theo mẫu biển Brazil (LLLNNNN) hay Mercosur (LLLNLNN)',
  not_calibrated: () => 'Chưa có ngưỡng cho số frame này',
};

export const DECODERS = [
  ['default', 'Mặc định'],
  ['greedy', 'Greedy'],
  ['beam', 'Beam'],
  ['constrained', 'Theo cú pháp'],
];

export function sampleLabel(name) {
  if (name.startsWith('easy')) return 'Dễ đọc';
  if (name.startsWith('blurry')) return 'Ảnh rất mờ';
  if (name.startsWith('review')) return 'Cần kiểm tra';
  return name.replace(/\.zip$/i, '');
}

export function timeOf(iso) {
  try {
    return new Date(iso).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch {
    return iso;
  }
}
