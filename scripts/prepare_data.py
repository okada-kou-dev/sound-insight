"""指定MIMII原版だけを取得・検証・抽出して固定manifestを作る。"""

import argparse
import json
from pathlib import Path
import sys

from src.dataset import DatasetError, prepare_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="公式6_dB_pump.zipの取得を許可する（最大3試行）")
    args = parser.parse_args()
    try:
        summary = prepare_dataset(Path(__file__).resolve().parents[1], download=args.download)
    except (DatasetError, OSError, ValueError) as exc:
        print(f"データ準備失敗: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
