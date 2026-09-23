# 学習・評価

Windows / Python 3.14.6 / scikit-learn 1.9.0<br>
依存関係はリポジトリ直下の requirements.txt を使用します。

正常音だけで標準化と近傍モデルを学習し、全体判定と区間表示のしきい値を別々に校正します。学習用の原音は同梱していません。

## データ

[MIMII public 1.0](https://zenodo.org/records/3384388)の `6_dB_pump.zip`（約7.7 GB）から pump / id_00 を使用します。
MD5：`a09ba6060c10fc09cd4c8770213b0b9f`。[利用条件](../ASSET_LICENSE.md)

| 分割 | 正常 | 異常 | 用途 |
| --- | ---: | ---: | --- |
| train | 604 | 0 | 正常パターン学習 |
| calibration | 201 | 0 | 区間しきい値 |
| development | 101 | 72 | 全体しきい値・評価 |
| final_test | 100 | 71 | 同梱モデルでは未評価 |

分割seedは42。クリップ単位で分け、重複を検査します。同梱50件・校正100件の固定リストとPCMハッシュは `configs/` に収録しています。

## 環境

リポジトリ直下で実行します。取得・展開には20 GiB以上の空きが必要です。

```powershell
$appPython = Join-Path (Get-Location) '.venv\Scripts\python.exe'
```

## 学習

各コマンドが正常終了してから次へ進みます。

```powershell
& $appPython -m scripts.prepare_data --download
& $appPython -m scripts.freeze_demo
& $appPython -m scripts.train_nearest --activate
```

全体しきい値は固定100件の誤判定数を最小化し、同数なら誤警報数、境界との距離の順で決めます。
モデルは新規IDで `models/`、評価は `outputs/evaluations/` に保存します。同梱モデルは変更しません。再学習の結果は実行環境により変わります。

## 評価

```powershell
# 再学習したモデルを固定100件で検証
& $appPython -m scripts.verify_fixed_demo
# 同梱モデル・50音声を保存済みの判定と照合
& $appPython -m scripts.verify_public_demo
```

音声・正解ラベル・スコア・判定・区間を照合します。校正用データの成績と独立評価を区別します。[検出精度・評価条件](../docs/accuracy_results.md)
