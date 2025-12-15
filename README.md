# AI Necklace (Gemini版)

Raspberry Pi 5で動作する音声AIアシスタント。Google Gemini APIを使用。

## 特徴

- **音声認識**: Gemini 2.5 Flashで音声をテキストに変換
- **AI応答**: Gemini 2.5 Flashでインテリジェントな応答を生成
- **音声合成**: Gemini TTSで自然な音声を生成（24言語対応）
- **画像認識**: カメラで撮影してGeminiで画像を解析
- **翻訳機能**: 日本語↔英語などの双方向翻訳（70言語対応）
- **Gmail連携**: メールの確認・送信・返信
- **アラーム**: 音声でアラーム設定
- **音声メッセージ**: スマホとの音声メッセージ送受信（Firebase連携）

## 必要なもの

### ハードウェア
- Raspberry Pi 5
- USBマイク/スピーカー
- プッシュボタン（GPIO5）
- カメラモジュール（オプション）

### ソフトウェア
- Python 3.11+
- ffmpeg

## セットアップ

### 1. リポジトリをクローン

```bash
git clone https://github.com/hotaka0908/raspi-voice2.git
cd raspi-voice2
```

### 2. 依存関係をインストール

```bash
pip install -r requirements.txt
```

### 3. Google API キーを設定

1. [Google AI Studio](https://aistudio.google.com/apikey) でAPIキーを取得
2. `.env` ファイルを作成:

```bash
cp env_template .env
# .env を編集して GOOGLE_API_KEY を設定
```

### 4. Gmail設定（オプション）

Gmail機能を使用する場合:
1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを作成
2. Gmail APIを有効化
3. OAuth 2.0クライアントIDを作成
4. `~/.ai-necklace/credentials.json` に保存

### 5. 実行

```bash
python ai_necklace.py
```

## 使用方法

### 基本操作
- ボタンを押しながら話す（トランシーバー方式）
- ボタンを離すと応答が再生される

### 音声コマンド例

| コマンド | 動作 |
|---------|------|
| 「今日の天気は？」 | AIが応答 |
| 「通訳モードにして」 | 翻訳モード開始 |
| 「通訳モード終了」 | 通常モードに戻る |
| 「写真を撮って」 | カメラで撮影して説明 |
| 「メールを確認して」 | 未読メール一覧 |
| 「7時にアラームをセットして」 | アラーム設定 |

## 設定

`ai_necklace.py` の `CONFIG` で設定を変更できます:

```python
CONFIG = {
    "gemini_model": "gemini-2.5-flash",      # 使用するモデル
    "tts_voice": "Aoede",                     # TTS音声
    "sample_rate": 16000,                     # 入力サンプルレート
    "output_sample_rate": 24000,              # 出力サンプルレート
    "button_pin": 5,                          # GPIOピン番号
}
```

### 利用可能なTTS音声
- Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr

## systemdサービスとして実行

```bash
sudo cp ai-necklace.service /etc/systemd/system/
sudo systemctl enable ai-necklace
sudo systemctl start ai-necklace
```

## OpenAI版との違い

| 機能 | raspi-voice (OpenAI) | raspi-voice2 (Gemini) |
|-----|---------------------|----------------------|
| 音声認識 | Whisper API | Gemini 2.5 Flash |
| AI応答 | GPT-4o-mini | Gemini 2.5 Flash |
| 音声合成 | OpenAI TTS | Gemini TTS |
| 画像認識 | GPT-4o Vision | Gemini 2.5 Flash |
| 翻訳 | - | ネイティブ対応 |
| 入力レート | 44.1kHz | 16kHz |
| 出力レート | 44.1kHz | 24kHz |

## ライセンス

MIT License
