"""
prepare_dataset7_more_data.py
===============================================================================
prepare_dataset6_final.py 에서 "데이터 양 확대"를 위해 수정한 버전입니다.

★ 이번 버전에서 바뀐 것 (v6 → v7)
  1) MAX_PER_CATEGORY: 800 → 3000
     - 원본 데이터에 라벨이 58만 개나 있는데 카테고리당 800장만 쓰는 건
       너무 아깝습니다. 처음부터 학습(from scratch)하는 CNN은 데이터 양에
       가장 민감하므로, 데이터를 늘리는 것이 정확도를 올리는 가장 확실한
       방법입니다.
  2) MAX_PER_SOURCE_VIDEO: 5 → 8
     - 영상 하나당 뽑는 장수를 조금 늘려서 전체 수집량을 확보합니다.
       train/val 분리는 여전히 "영상(그룹) 단위"로 하기 때문에, 같은 영상의
       사진이 늘어나도 데이터 유출은 생기지 않습니다.
  3) MIN_ACCEPTABLE_COUNT: 200 → 1200
     - 수집 목표가 커진 만큼, 완화 기준(fallback)을 적용할지 판단하는
       기준도 같이 올렸습니다.

===============================================================================
"""

import json
from pathlib import Path

from PIL import Image

# ── 설정  ──────────────────────────────────────────
DATA_ROOT = Path("./데이터")          # Training/Validation을 모두 담은 최상위 폴더
OUTPUT_ROOT = Path("./dataset")       # 크롭된 결과물을 저장할 폴더

# 우리가 실제로 쓸 대분류만 남기고 나머지(쓰레기, 공사현장 등)는 자동으로 무시됩니다.
KEEP_CATEGORIES = {"불법주정차", "현수막", "보행방해물"}

MAX_PER_CATEGORY = 3000  # ★ v7: 800 → 3000 (데이터 확대가 이번 버전의 핵심)
MIN_BOX_SIZE = 40        # 원본 박스가 이 픽셀보다 작으면 애초에 후보에서 제외 (노이즈 컷)

# 맥락 포함 크롭 설정 (v6에서 정한 카테고리별 배수 그대로 유지)
CONTEXT_MULTIPLIER_BY_CATEGORY = {
    "불법주정차": 3.0,
    "보행방해물": 2.5,
    "현수막": 1.5,   # 현수막은 객체 자체가 식별 대상이라 배경 비중을 줄임
}
MIN_CONTEXT_EDGE = 150       # 1차 기준: 크롭 결과 한 변이 이 이상이면 "고화질"로 우선 사용
MIN_CONTEXT_EDGE_FALLBACK = 90   # 카테고리 하나가 유독 데이터가 적을 때 이 정도까지 완화
MIN_ACCEPTABLE_COUNT = 1200      # ★ v7: 고화질 기준으로 이 개수를 못 채운 카테고리만 완화 기준 적용

# 같은 CCTV 영상(같은 차가 몇 시간 동안 계속 잡힌 경우 등)에서 너무 많이
# 뽑히지 않도록, 영상 하나당 최대 이만큼만 사용합니다 (다양성 확보용).
MAX_PER_SOURCE_VIDEO = 8     # ★ v7: 5 → 8
# ──────────────────────────────────────────────────────────────────────────


def build_image_index(image_root: Path) -> dict:
    """image_root 아래 모든 jpg/jpeg/png 파일을 파일명 기준으로 인덱싱합니다."""
    index = {}
    exts = {".jpg", ".jpeg", ".png"}
    for p in image_root.rglob("*"):
        if p.suffix.lower() in exts:
            index[p.name] = p
    print(f"[이미지 인덱스] {len(index):,}개 파일 발견")
    return index


def extract_top_category(atch_file_path: str) -> str | None:
    """
    atchFilePath 예: "../../../원천데이터/보행방해물/간이의자(낮)/"
    -> "원천데이터" 바로 다음 폴더명을 대분류로 사용
    """
    parts = [p for p in atch_file_path.replace("\\", "/").split("/") if p and p != ".."]
    for i, part in enumerate(parts):
        if part in ("원천데이터", "원본데이터", "01.원천데이터"):
            if i + 1 < len(parts):
                return parts[i + 1]
    for part in parts:
        if part in KEEP_CATEGORIES or part == "쓰레기":
            return part
    return None


def collect_label_files(label_root: Path) -> list[Path]:
    files = list(label_root.rglob("*.json"))
    print(f"[라벨 인덱스] {len(files):,}개 JSON 파일 발견")
    return files


def compute_context_crop(x, y, w, h, img_w, img_h, multiplier):
    """
    박스 (x,y,w,h)를 중심으로 multiplier배 크기의 정사각형 영역을 계산합니다.
    이미지 경계에 걸리면, 크기를 그대로 유지한 채 안쪽으로 밀어서(shift) 최대한
    원하는 크기를 확보합니다 (단순히 잘라내면 경계 근처 객체는 계속 작게 나옴).
    반환: (left, top, right, bottom) 정수 좌표
    """
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * multiplier
    half = side / 2

    left, top = cx - half, cy - half
    right, bottom = cx + half, cy + half

    # 이미지 밖으로 나가면 크기는 유지한 채 안쪽으로 밀어넣기
    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > img_w:
        left -= (right - img_w)
        right = img_w
    if bottom > img_h:
        top -= (bottom - img_h)
        bottom = img_h

    # 그래도 이미지 자체보다 크게 요청된 경우 최종 클램프
    left = max(0, left)
    top = max(0, top)
    right = min(img_w, right)
    bottom = min(img_h, bottom)

    return int(left), int(top), int(right), int(bottom)


def scan_candidates(label_files: list[Path]) -> dict:
    """1단계: JSON만 읽어서 카테고리별 (박스면적, 파일명, box, box_idx) 후보를 모으고,
    박스가 큰 순서대로 정렬합니다."""
    candidates = {cat: [] for cat in KEEP_CATEGORIES}
    raw_box_count = {cat: 0 for cat in KEEP_CATEGORIES}
    parse_errors = 0

    for label_path in label_files:
        try:
            with open(label_path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            parse_errors += 1
            continue

        anno = data.get("annotations", {}).get("Bbox Annotation")
        if not anno:
            continue

        top_category = extract_top_category(anno.get("atchFilePath", ""))
        if top_category not in KEEP_CATEGORIES:
            continue

        image_filename = anno.get("atchFileName")
        boxes = anno.get("Box", [])
        # 같은 영상(카메라+녹화본)에서 나온 프레임인지 구분하기 위한 키.
        # meta.resource(영상 파일명) 기준 — 없으면 이미지 파일명으로 대체.
        source_id = data.get("meta", {}).get("resource") or image_filename

        for i, box in enumerate(boxes):
            raw_box_count[top_category] += 1
            w, h = box.get("w", 0), box.get("h", 0)
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue
            area = w * h
            candidates[top_category].append((area, image_filename, box, i, source_id))

    if parse_errors:
        print(f"[경고] JSON 파싱 실패 {parse_errors:,}건 (건너뜀)")

    for cat in candidates:
        candidates[cat].sort(key=lambda t: t[0], reverse=True)
        print(f"[{cat}] 전체 박스: {raw_box_count[cat]:,}개 | {MIN_BOX_SIZE}px 이상(후보): {len(candidates[cat]):,}개")

    return candidates


def crop_selected(candidates: dict, image_index: dict) -> dict:
    """2단계: 카테고리별로 큰 것부터 맥락 포함 정사각형으로 크롭·저장합니다.
    같은 영상에서 너무 많이 뽑히지 않도록 MAX_PER_SOURCE_VIDEO로 제한합니다."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    counts = {cat: 0 for cat in KEEP_CATEGORIES}
    quality_counts = {cat: {"high": 0, "fallback": 0} for cat in KEEP_CATEGORIES}
    source_used = {cat: {} for cat in KEEP_CATEGORIES}  # cat -> {source_id: 사용횟수}
    skipped_no_image = 0
    skipped_source_cap = 0
    errors = 0
    image_cache = {}

    def edge_ok(left, top, right, bottom, threshold):
        return (right - left) >= threshold and (bottom - top) >= threshold

    for cat, items in candidates.items():
        target = MAX_PER_CATEGORY or len(items)

        for pass_name, threshold in [("high", MIN_CONTEXT_EDGE), ("fallback", MIN_CONTEXT_EDGE_FALLBACK)]:
            if counts[cat] >= target:
                break
            if pass_name == "fallback" and counts[cat] >= MIN_ACCEPTABLE_COUNT:
                break  # 고화질만으로 이미 충분함

            for area, image_filename, box, box_idx, source_id in items:
                if counts[cat] >= target:
                    break

                # 파일명에 영상(source) ID를 앞에 붙여서, 학습 스크립트가 "같은 영상은
                # 절대 train/val에 나눠 담기지 않도록" 그룹 단위로 분리할 수 있게 함.
                safe_source = str(source_id).replace(" ", "_").replace("/", "-")
                out_name = f"{safe_source}__{Path(image_filename).stem}_{box_idx}.jpg"
                out_dir = OUTPUT_ROOT / cat
                if (out_dir / out_name).exists():
                    continue  # high 패스에서 이미 처리된 항목 중복 방지

                if source_used[cat].get(source_id, 0) >= MAX_PER_SOURCE_VIDEO:
                    skipped_source_cap += 1
                    continue

                image_path = image_index.get(image_filename)
                if not image_path:
                    if pass_name == "high":
                        skipped_no_image += 1
                    continue

                try:
                    if image_path not in image_cache:
                        image_cache[image_path] = Image.open(image_path).convert("RGB")
                        if len(image_cache) > 50:
                            image_cache.pop(next(iter(image_cache)))
                    img = image_cache[image_path]
                except Exception:
                    if pass_name == "high":
                        errors += 1
                    continue

                img_w, img_h = img.size
                x, y, w, h = box.get("x", 0), box.get("y", 0), box.get("w", 0), box.get("h", 0)
                left, top, right, bottom = compute_context_crop(x, y, w, h, img_w, img_h, CONTEXT_MULTIPLIER_BY_CATEGORY[cat])
                if right <= left or bottom <= top:
                    continue

                if pass_name == "high":
                    if not edge_ok(left, top, right, bottom, MIN_CONTEXT_EDGE):
                        continue
                else:  # fallback
                    if not edge_ok(left, top, right, bottom, MIN_CONTEXT_EDGE_FALLBACK):
                        continue
                    if edge_ok(left, top, right, bottom, MIN_CONTEXT_EDGE):
                        continue  # high 패스 대상이었어야 할 것은 건너뜀(중복 방지)

                crop = img.crop((left, top, right, bottom))
                out_dir.mkdir(parents=True, exist_ok=True)
                crop.save(out_dir / out_name, quality=90)
                counts[cat] += 1
                quality_counts[cat][pass_name] += 1
                source_used[cat][source_id] = source_used[cat].get(source_id, 0) + 1

        if quality_counts[cat]["fallback"] > 0:
            print(
                f"[{cat}] 고화질만으로는 부족해서 완화 기준({MIN_CONTEXT_EDGE_FALLBACK}px)까지 사용함 "
                f"→ 고화질 {quality_counts[cat]['high']}장 + 완화 {quality_counts[cat]['fallback']}장"
            )
        print(f"[{cat}] 서로 다른 영상(카메라) 수: {len(source_used[cat]):,}개에서 골고루 수집됨")

    print(f"\n건너뜀 - 이미지 못찾음: {skipped_no_image}, 같은 영상 상한 초과: {skipped_source_cap}, 오류: {errors}")
    return counts


def main():
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        print(f"⚠️  경고: OUTPUT_ROOT({OUTPUT_ROOT})에 기존 파일이 있습니다!")
        print("   예전 크롭 결과(다른 배수/설정)와 섞이면 안 되므로,")
        print("   기존 dataset 폴더를 지우거나 이름을 바꾼 뒤 다시 실행해주세요.")
        print("   (예: mv dataset dataset_v6_backup)")
        return

    if not DATA_ROOT.exists():
        print(f"❌ DATA_ROOT({DATA_ROOT})를 찾을 수 없습니다.")
        print("   스크립트 상단의 경로를 실제 다운로드 위치(Training/Validation을 담은 최상위 폴더)로 수정해주세요.")
        return

    image_index = build_image_index(DATA_ROOT)
    label_files = collect_label_files(DATA_ROOT)

    print("\n[1단계] 라벨만 훑어서 카테고리별로 큰 박스 순으로 정렬 중...")
    candidates = scan_candidates(label_files)

    print(f"\n[2단계] 큰 박스부터 맥락 포함(카테고리별 배수 {CONTEXT_MULTIPLIER_BY_CATEGORY}) 정사각형으로 크롭 중...")
    counts = crop_selected(candidates, image_index)

    print("\n=== 완료 ===")
    for cat, n in counts.items():
        print(f"  {cat}: {n}장")
    print(f"\n결과물 위치: {OUTPUT_ROOT.resolve()}")


if __name__ == "__main__":
    main()