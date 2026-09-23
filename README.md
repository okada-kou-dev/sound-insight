# SOUND INSIGHT

ポンプの運転音から正常・異常を判定するPythonアプリです。正常音との近傍距離で判定し、公開元の正解ラベルと照合します。音声に同期した波形・スペクトログラム・周波数帯別寄与の表示と、結果のCSV出力に対応します。

## 起動

Windows / Python 3.14.6<br>
リポジトリ直下で実行します。

```powershell
python -m venv .venv
$appPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
& $appPython -m pip install -r requirements.txt
& $appPython -m streamlit run streamlit_app.py --server.address 127.0.0.1
```

「全サンプルを一括検査」で検査・再生し、「1件だけ検査」で個別に確認します。学習済みモデルと音声50件（正常25・異常25）を同梱しています。

## 検出精度

開発100件で99件正解（見逃し1件・誤警報0件）。同梱50件はその一部で、全件正解です。100件は方式・しきい値の調整にも使用しており、未知音声の精度を示す独立評価ではありません。

## 資料

[設計](ARCHITECTURE.md) / [モデル・データ](MODEL_CARD.md) / [学習・評価手順](training/README.md) / [検出精度・評価条件](docs/accuracy_results.md)

## ライセンス

Copyright (C) 2026 岡田 耕（Okada Kou）<br>
コード・文書：[MIT](LICENSE)<br>
音声・音声由来の可視化・配布モデル：[CC BY-SA 4.0](ASSET_LICENSE.md)<br>
[第三者ソフトウェアの表記](THIRD_PARTY_NOTICES.md)
