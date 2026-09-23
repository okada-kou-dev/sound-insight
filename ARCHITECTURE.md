# 設計

WAV読込み → ログメル特徴 → 正常音との近傍距離 → しきい値判定 → 正解との照合 → 音声同期表示の順に処理します。
判定はPythonで計算し、ブラウザの再生位置に合わせて結果を表示します。

| 処理 | 実装 |
| --- | --- |
| 起動・サンプル読込み | `streamlit_app.py`、`src/public_ui.py` |
| 音声・特徴量 | `src/audio.py` |
| 正常近傍モデル・判定 | `src/nearest.py`、`src/service.py` |
| 検査・正解照合・CSV | `src/ui.py`、`src/evaluation.py` |
| 再生・グラフ・周波数帯別寄与 | `src/playback.py`、`src/frontend/` |
| 音声・モデルの整合性確認 | `src/public_bundle.py` |
| データ準備・学習 | `scripts/prepare_data.py`、`scripts/train_nearest.py` |

正解ラベルとファイル名は推論に使いません。再生速度・表示設定を変えても判定は変わりません。
同梱サンプルを検査し、結果はセッション内に保持します。任意音声のアップロードや外部AI APIへの送信は行いません。

## テスト

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py' -v
& .\.venv\Scripts\python.exe -m scripts.verify_public_demo
```

単体テストと、同梱50件の音声・スコア・判定・検出区間の照合を行います。
