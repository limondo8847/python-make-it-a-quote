# File: make_quote.py
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import budoux

# ===== 画像サイズとレイアウト =====
W, H = 1200, 630

# アイコン（calibrate.pyの結果）
ICON_SIZE = 630
ICON_X = -25
ICON_Y = 0

# 斜めのグラデーション
#   y=0 の行では GRAD_START〜GRAD_END の範囲でフェードする
#   1px下がるごとに、フェードの範囲が GRAD_SLOPE px 右にずれる
GRAD_START, GRAD_END = 150, 438
GRAD_SLOPE = 0.220
GRAD_CURVE = "smoothstep"  # "smoothstep" または "linear"

# 本文
TEXT_CX = 855
TEXT_MAX_W = 632
TEXT_MAX_H = 460
BLOCK_CY = 328
FONT_MAX, FONT_MIN = 60, 14
LINE_SPACING = 1.25

# 表示名・ユーザー名・透かし
NAME_SIZE = 29
USER_SIZE = 20
WATERMARK_SIZE = 19
NAME_GAP = 18
NAME_H = 38
USER_H = 28
ITALIC_SHEAR = 0.2

COLOR_TEXT = (255, 255, 255, 255)
COLOR_USER = (180, 180, 180, 255)
COLOR_WATERMARK = (165, 165, 165, 255)

# ===== フォント（Rounded M+ 1c） =====
FONT_DIR = os.path.dirname(os.path.abspath(__file__))

FONT_CANDIDATES = {
    "regular": [
        "MPLUSRounded1c-Regular.ttf",
        "rounded-mplus-1c-regular.ttf",
    ],
    "medium": [
        "MPLUSRounded1c-Medium.ttf",
        "rounded-mplus-1c-medium.ttf",
    ],
    "light": [
        "MPLUSRounded1c-Light.ttf",
        "rounded-mplus-1c-light.ttf",
    ],
}

BODY_WEIGHT = "regular"
NAME_WEIGHT = "regular"
USER_WEIGHT = "regular"
WATERMARK_WEIGHT = "regular"

PARSER = budoux.load_default_japanese_parser()
_font_cache = {}


def find_font_path(weight):
    for w in (weight, "regular"):
        for name in FONT_CANDIDATES.get(w, []):
            path = os.path.join(FONT_DIR, name)
            if os.path.exists(path):
                return path
    raise FileNotFoundError(
        "Rounded M+ 1c のフォントファイルが見つかりません。"
        f"{FONT_DIR} に置いてください。"
    )


def load_font(weight, size):
    key = (weight, size)
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(find_font_path(weight), size)
    return _font_cache[key]


def text_w(text, font):
    return font.getlength(text)


def wrap_text(text, font, max_w):
    """BudouXで文節に分けて、指定した幅で折り返す"""
    lines = []
    for para in text.split("\n"):
        phrases = PARSER.parse(para) or [""]
        cur = ""
        for ph in phrases:
            if text_w(cur + ph, font) <= max_w:
                cur += ph
                continue
            if cur:
                lines.append(cur)
                cur = ""
            for ch in ph:
                if text_w(cur + ch, font) <= max_w:
                    cur += ch
                else:
                    if cur:
                        lines.append(cur)
                    cur = ch
        lines.append(cur)
    return lines


def fit_text(text):
    """枠に収まる一番大きい文字サイズを探す"""
    for size in range(FONT_MAX, FONT_MIN - 1, -1):
        font = load_font(BODY_WEIGHT, size)
        lines = wrap_text(text, font, TEXT_MAX_W)
        lh = int(size * LINE_SPACING)
        if lh * len(lines) <= TEXT_MAX_H:
            return font, lines, lh
    font = load_font(BODY_WEIGHT, FONT_MIN)
    return font, wrap_text(text, font, TEXT_MAX_W), int(FONT_MIN * LINE_SPACING)


def load_square_icon(icon_path):
    """中央を正方形に切り抜き、透明部分を黒で埋めたRGB画像を返す"""
    icon = Image.open(icon_path).convert("RGBA")
    w, h = icon.size
    s = min(w, h)
    icon = icon.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    bg = Image.new("RGBA", icon.size, (0, 0, 0, 255))
    return Image.alpha_composite(bg, icon).convert("RGB")


def gradient_t(xs, ys, start, end, slope, curve):
    """各ピクセルの黒さ（0=アイコンのまま、1=真っ黒）を返す"""
    t = (xs - start - slope * ys) / max(1e-6, end - start)
    t = np.clip(t, 0.0, 1.0)
    if curve == "smoothstep":
        t = t * t * (3 - 2 * t)
    return t


def gradient_mask(start=GRAD_START, end=GRAD_END, slope=GRAD_SLOPE, curve=GRAD_CURVE):
    ys = np.arange(H, dtype=np.float32)[:, None]
    xs = np.arange(W, dtype=np.float32)[None, :]
    t = gradient_t(xs, ys, start, end, slope, curve)
    return Image.fromarray((t * 255 + 0.5).astype(np.uint8), "L")


def make_background(icon_path, icon_x=ICON_X, icon_y=ICON_Y, icon_size=ICON_SIZE,
                    grad_start=GRAD_START, grad_end=GRAD_END,
                    grad_slope=GRAD_SLOPE, curve=GRAD_CURVE):
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    icon = load_square_icon(icon_path).resize((icon_size, icon_size), Image.LANCZOS)
    canvas.paste(icon, (icon_x, icon_y))
    black = Image.new("RGB", (W, H), (0, 0, 0))
    mask = gradient_mask(grad_start, grad_end, grad_slope, curve)
    canvas = Image.composite(black, canvas, mask)
    return canvas.convert("RGBA")


def draw_italic_center(canvas, center, text, font, fill, shear=ITALIC_SHEAR):
    """Rounded M+ 1cには斜体がないので、文字を斜めに変形して疑似的な斜体を作る"""
    l, t, r, b = font.getbbox(text)
    tw, th = r - l, b - t
    pad = 10
    lw = int(tw + th * shear) + pad * 2
    lh = th + pad * 2
    layer = Image.new("RGBA", (lw, lh), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((lw / 2, lh / 2), text, font=font, fill=fill, anchor="mm")
    hc = lh / 2
    layer = layer.transform(
        layer.size,
        Image.AFFINE,
        (1, shear, -shear * hc, 0, 1, 0),
        resample=Image.BICUBIC,
    )
    cx, cy = center
    canvas.alpha_composite(layer, (int(cx - lw / 2), int(cy - lh / 2)))


def make_quote(icon_path, text, display_name, username,
               watermark="Make it a Quote#6660", out_path="quote.png"):
    canvas = make_background(icon_path)
    draw = ImageDraw.Draw(canvas)

    # 本文
    font, lines, lh = fit_text(text)
    quote_h = lh * len(lines)
    total_h = quote_h + NAME_GAP + NAME_H + USER_H
    top = BLOCK_CY - total_h / 2

    for i, line in enumerate(lines):
        y = top + lh * (i + 0.5)
        draw.text((TEXT_CX, y), line, font=font, fill=COLOR_TEXT, anchor="mm")

    # 表示名（斜体風）
    name_font = load_font(NAME_WEIGHT, NAME_SIZE)
    name_cy = top + quote_h + NAME_GAP + NAME_H / 2
    draw_italic_center(canvas, (TEXT_CX, name_cy), f"- {display_name}", name_font, COLOR_TEXT)

    # ユーザー名
    draw = ImageDraw.Draw(canvas)
    user_font = load_font(USER_WEIGHT, USER_SIZE)
    user_cy = name_cy + NAME_H / 2 + USER_H / 2
    draw.text((TEXT_CX, user_cy), f"@{username}", font=user_font, fill=COLOR_USER, anchor="mm")

    # 右下の透かし
    wm_font = load_font(WATERMARK_WEIGHT, WATERMARK_SIZE)
    draw.text((W - 10, H - 15), watermark, font=wm_font, fill=COLOR_WATERMARK, anchor="rs")

    canvas.convert("RGB").save(out_path)
    return out_path


if __name__ == "__main__":
    make_quote("miq.png", "このように、Make it a Quoteのような画像を生成することができます。", "Make it a Quote Demo", "miq_demo", out_path="quote_short.png")
    make_quote(
        "icon.png",
        "長文テスト。吾輩は猫である。名前はまだ無い。どこで生れたかとんと見当がつかぬ。"
        "何でも薄暗いじめじめした所でニャーニャー泣いていた事だけは記憶している。"
        "吾輩はここで始めて人間というものを見た。",
        "li mondo",
        "li_mondo",
        out_path="quote_long.png",
    )