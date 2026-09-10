#!/bin/bash

# 사용법: ./unzip_all.sh [압축해제_대상_폴더] [압축해제_위치]
# 예: ./unzip_all.sh /path/to/zipfiles /path/to/output

ZIP_DIR="$1"       # zip 파일들이 있는 폴더
OUTPUT_DIR="$2"    # 압축을 풀 위치

# 인자 확인
if [ -z "$ZIP_DIR" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "사용법: $0 [압축해제_대상_폴더] [압축해제_위치]"
    exit 1
fi

# 출력 경로 없으면 생성
mkdir -p "$OUTPUT_DIR"

# 폴더 안의 모든 zip 파일 압축 해제
for file in "$ZIP_DIR"/*.zip; do
    if [ -f "$file" ]; then
        echo "압축 해제 중: $file"
        unzip -o "$file" -d "$OUTPUT_DIR"
    else
        echo "zip 파일이 없습니다: $ZIP_DIR"
    fi
done

echo "모든 zip 파일 압축 해제 완료!"
