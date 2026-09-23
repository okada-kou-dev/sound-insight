# 音源・モデルの利用条件

**MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection**
public 1.0（2019）、提供：Hitachi, Ltd.

著作者：Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, Yohei Kawaguchi

[出典 DOI](https://doi.org/10.5281/zenodo.3384388) / [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) / [条文](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)

## 使用・加工

`6_dB_pump.zip / pump / id_00 / +6 dB` から正常25件・異常25件を同梱しています。
先頭チャンネルを16 kHz・16 bit PCMのモノラルWAVへ抽出し、振幅を保持しています。再サンプリング・ノイズ除去・音量正規化は行っていません。
ファイル名を変更し、特徴量抽出・モデル学習・可視化を行っています。

加工・モデル作成：okada-kou-dev、2026年。
音声・音声由来の可視化・配布モデルはCC BY-SA 4.0で提供します。出典・著作者・変更内容を表示し、改変物は同じ条件で共有してください。
提供元による本アプリの承認・保証を意味しません。コードのMITライセンスとは区別します。

参考文献：H. Purohit et al., “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection,” DCASE Workshop, 2019。
[公式論文情報](https://github.com/MIMII-hitachi/mimii_baseline)
