#!/usr/bin/env python3
"""
AI Necklace (Gemini版) - Raspberry Pi 5 スタンドアロン音声AIクライアント

マイクから音声を録音し、Google Gemini APIで:
- 音声認識（Speech-to-Text）
- AI応答生成
- 音声合成（Text-to-Speech）
- 画像認識（Vision）
- 音声翻訳（Translation）

ボタン操作: GPIO5に接続したボタンを押している間録音（トランシーバー方式）

Gmail機能:
- 「メールを確認」「メールを読んで」→ 未読メール一覧
- 「○○からのメール」→ 特定の送信者のメール
- 「メールに返信して」→ 返信作成
- 「メールを送って」→ 新規メール作成

アラーム機能:
- 「7時にアラームをセットして」→ アラーム設定
- 「アラームを確認して」→ 一覧表示
- 「アラームを削除して」→ 削除

カメラ機能:
- 「写真を撮って」「何が見える？」→ カメラで撮影してAIが説明
- 「これは何？」「目の前にあるものを教えて」→ 画像認識

翻訳機能:
- 「通訳モードにして」→ 日本語↔英語の同時通訳を開始
- 「通訳モード終了」→ 通常モードに戻る
"""

import os
import io
import wave
import tempfile
import time
import signal
import sys
import threading
import json
import base64
import re
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime, timedelta
import subprocess

import pyaudio
import numpy as np
from dotenv import load_dotenv

# Google Gemini API
from google import genai
from google.genai import types

# Gmail API
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Firebase Voice Messenger
try:
    from firebase_voice import FirebaseVoiceMessenger
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("警告: firebase_voiceモジュールが見つかりません。音声メッセージ機能は無効です。")

# GPIOライブラリ
try:
    from gpiozero import Button
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    print("警告: gpiozeroが使用できません。ボタン操作は無効です。")

# systemdで実行時にprint出力をリアルタイムで表示するため
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# 環境変数の読み込み
load_dotenv()

# Gmail APIスコープ
GMAIL_SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.modify'
]

# 設定
CONFIG = {
    # オーディオ設定
    "sample_rate": 44100,  # Gemini Native Audio入力は16kHz
    "output_sample_rate": 24000,  # Gemini Native Audio出力は24kHz
    "channels": 1,
    "chunk_size": 1024,
    "max_record_seconds": 30,
    "silence_threshold": 500,

    # デバイス設定
    "input_device_index": None,
    "output_device_index": None,

    # GPIO設定
    "button_pin": 5,
    "use_button": True,

    # Gemini AI設定
    "gemini_model": "gemini-2.5-flash",
    "gemini_tts_model": "gemini-2.5-flash-preview-tts",
    "tts_voice": "Aoede",  # 利用可能: Puck, Charon, Kore, Fenrir, Aoede, Leda, Orus, Zephyr

    # Gmail設定
    "gmail_credentials_path": os.path.expanduser("~/.ai-necklace/credentials.json"),
    "gmail_token_path": os.path.expanduser("~/.ai-necklace/token.json"),

    # 翻訳設定
    "translation_mode": False,
    "source_language": "ja",
    "target_language": "en",

    # システムプロンプト
    "system_prompt": """あなたは親切なAIアシスタントです。
ユーザーの質問に簡潔に答えてください。
音声で読み上げられるため、1-2文程度の短い応答を心がけてください。
日本語で回答してください。

利用可能なツールがある場合は適切に使用してください。
ユーザーが「メールを確認」と言ったらgmail_listを使ってください。
ユーザーが「通訳モードにして」と言ったらtranslation_mode_onを使ってください。
ユーザーが「通訳モード終了」と言ったらtranslation_mode_offを使ってください。
ユーザーが「写真を撮って」と言ったらcamera_captureを使ってください。
ユーザーが「アラームをセット」と言ったらalarm_setを使ってください。
ユーザーが「スマホにメッセージを送って」と言ったらvoice_record_sendを使ってください。
""",
}

# ==================== Function Calling ツール定義 ====================
from google.genai import types as genai_types

TOOL_DECLARATIONS = [
    genai_types.FunctionDeclaration(
        name="gmail_list",
        description="メール一覧を取得する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "query": {"type": "STRING", "description": "検索クエリ（例: is:unread）"},
                "max_results": {"type": "INTEGER", "description": "取得件数"}
            }
        }
    ),
    genai_types.FunctionDeclaration(
        name="gmail_read",
        description="メール本文を読み取る",
        parameters={
            "type": "OBJECT",
            "properties": {
                "message_id": {"type": "INTEGER", "description": "メールID（番号）"}
            },
            "required": ["message_id"]
        }
    ),
    genai_types.FunctionDeclaration(
        name="gmail_send",
        description="新規メールを送信する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "to": {"type": "STRING", "description": "宛先メールアドレス"},
                "subject": {"type": "STRING", "description": "件名"},
                "body": {"type": "STRING", "description": "本文"}
            },
            "required": ["to", "subject", "body"]
        }
    ),
    genai_types.FunctionDeclaration(
        name="gmail_reply",
        description="メールに返信する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "message_id": {"type": "INTEGER", "description": "返信するメールの番号"},
                "body": {"type": "STRING", "description": "返信本文"},
                "attach_photo": {"type": "BOOLEAN", "description": "写真を添付するか"}
            },
            "required": ["message_id", "body"]
        }
    ),
    genai_types.FunctionDeclaration(
        name="alarm_set",
        description="アラームを設定する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "time": {"type": "STRING", "description": "時刻（HH:MM形式）"},
                "label": {"type": "STRING", "description": "ラベル"},
                "message": {"type": "STRING", "description": "読み上げメッセージ"}
            },
            "required": ["time"]
        }
    ),
    genai_types.FunctionDeclaration(
        name="alarm_list",
        description="アラーム一覧を取得する",
        parameters={"type": "OBJECT", "properties": {}}
    ),
    genai_types.FunctionDeclaration(
        name="alarm_delete",
        description="アラームを削除する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "alarm_id": {"type": "INTEGER", "description": "アラームID"}
            },
            "required": ["alarm_id"]
        }
    ),
    genai_types.FunctionDeclaration(
        name="camera_capture",
        description="カメラで撮影して画像を説明する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "prompt": {"type": "STRING", "description": "画像に対する質問"}
            }
        }
    ),
    genai_types.FunctionDeclaration(
        name="gmail_send_photo",
        description="写真を撮影してメールで送信する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "to": {"type": "STRING", "description": "宛先"},
                "subject": {"type": "STRING", "description": "件名"},
                "body": {"type": "STRING", "description": "本文"}
            }
        }
    ),
    genai_types.FunctionDeclaration(
        name="voice_record_send",
        description="スマホに音声メッセージを録音して送信する",
        parameters={"type": "OBJECT", "properties": {}}
    ),
    genai_types.FunctionDeclaration(
        name="translation_mode_on",
        description="翻訳モードを開始する",
        parameters={
            "type": "OBJECT",
            "properties": {
                "source_lang": {"type": "STRING", "description": "元の言語（デフォルト: ja）"},
                "target_lang": {"type": "STRING", "description": "翻訳先言語（デフォルト: en）"}
            }
        }
    ),
    genai_types.FunctionDeclaration(
        name="translation_mode_off",
        description="翻訳モードを終了する",
        parameters={"type": "OBJECT", "properties": {}}
    ),
]

TOOLS = [genai_types.Tool(function_declarations=TOOL_DECLARATIONS)]


# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY = 2

def retry_on_error(func):
    """503エラー時にリトライするデコレータ"""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if '503' in error_str or 'overloaded' in error_str.lower():
                    last_error = e
                    if attempt < MAX_RETRIES - 1:
                        print(f"サーバー混雑中... {attempt + 1}/{MAX_RETRIES} 回目リトライ")
                        import time
                        time.sleep(RETRY_DELAY * (attempt + 1))
                    continue
                raise e
        raise last_error
    return wrapper


# グローバル変数
running = True
gemini_client = None
audio = None
button = None
is_recording = False
record_lock = threading.Lock()
gmail_service = None
conversation_history = []
last_email_list = []

# アラーム関連
alarms = []
alarm_next_id = 1
alarm_thread = None
alarm_file_path = os.path.expanduser("~/.ai-necklace/alarms.json")

# Firebase Voice Messenger
firebase_messenger = None


def signal_handler(sig, frame):
    """Ctrl+C で終了"""
    global running, firebase_messenger
    print("\n終了します...")
    running = False
    if firebase_messenger:
        firebase_messenger.stop_listening()


# ==================== Firebase Voice Messenger ====================

def init_firebase_messenger():
    """Firebase Voice Messengerを初期化"""
    global firebase_messenger

    if not FIREBASE_AVAILABLE:
        print("Firebase Voice Messenger: 無効（モジュールなし）")
        return False

    try:
        firebase_messenger = FirebaseVoiceMessenger(
            device_id="raspi",
            on_message_received=on_voice_message_received
        )
        firebase_messenger.start_listening(poll_interval=1.5)
        print("Firebase Voice Messenger: 有効")
        return True
    except Exception as e:
        print(f"Firebase初期化エラー: {e}")
        return False


def on_voice_message_received(message):
    """スマホからの音声メッセージを受信したときの処理"""
    global firebase_messenger

    print(f"\n📱 スマホから音声メッセージ受信!")

    # 通知音を再生
    notification = generate_notification_sound()
    if notification:
        play_audio(notification)

    try:
        audio_url = message.get("audio_url")
        if not audio_url:
            print("音声URLがありません")
            return

        audio_data = firebase_messenger.download_audio(audio_url)
        if not audio_data:
            print("音声ダウンロードに失敗")
            return

        filename = message.get("filename", "audio.webm")
        wav_data = convert_webm_to_wav(audio_data, filename)
        if wav_data:
            play_audio(wav_data)
        else:
            print("音声変換に失敗")

        firebase_messenger.mark_as_played(message.get("id"))

    except Exception as e:
        print(f"音声メッセージ処理エラー: {e}")


def convert_webm_to_wav(audio_data, filename="audio.webm"):
    """WebM音声をWAV形式に変換"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as webm_file:
            webm_file.write(audio_data)
            webm_path = webm_file.name

        wav_path = webm_path.replace(".webm", ".wav")

        result = subprocess.run([
            "ffmpeg", "-y", "-i", webm_path,
            "-ar", str(CONFIG["output_sample_rate"]), "-ac", "1", "-f", "wav", wav_path
        ], capture_output=True, timeout=30)

        if result.returncode != 0:
            print(f"ffmpeg変換エラー: {result.stderr.decode()}")
            return None

        with open(wav_path, "rb") as f:
            wav_data = f.read()

        os.unlink(webm_path)
        os.unlink(wav_path)

        return wav_data

    except Exception as e:
        print(f"音声変換エラー: {e}")
        return None


def send_voice_to_phone(audio_buffer, text=None):
    """音声をスマホに送信"""
    global firebase_messenger

    if not firebase_messenger:
        print("Firebase未初期化")
        return False

    try:
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()
        return firebase_messenger.send_message(audio_data, text=text)
    except Exception as e:
        print(f"音声送信エラー: {e}")
        return False


# ==================== アラーム機能 ====================

def load_alarms():
    """保存されたアラームを読み込み"""
    global alarms, alarm_next_id
    try:
        if os.path.exists(alarm_file_path):
            with open(alarm_file_path, 'r') as f:
                data = json.load(f)
                alarms = data.get('alarms', [])
                alarm_next_id = data.get('next_id', 1)
                print(f"アラーム読み込み: {len(alarms)}件")
    except Exception as e:
        print(f"アラーム読み込みエラー: {e}")
        alarms = []
        alarm_next_id = 1


def save_alarms():
    """アラームを保存"""
    global alarms, alarm_next_id
    try:
        os.makedirs(os.path.dirname(alarm_file_path), exist_ok=True)
        with open(alarm_file_path, 'w') as f:
            json.dump({'alarms': alarms, 'next_id': alarm_next_id}, f, ensure_ascii=False)
    except Exception as e:
        print(f"アラーム保存エラー: {e}")


def alarm_set(time_str, label="アラーム", message=""):
    """アラームを設定"""
    global alarms, alarm_next_id

    try:
        hour, minute = map(int, time_str.split(':'))
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return "時刻が不正です。00:00〜23:59の形式で指定してください。"
    except:
        return "時刻の形式が不正です。HH:MM形式（例: 07:00）で指定してください。"

    alarm = {
        "id": alarm_next_id,
        "time": time_str,
        "label": label,
        "message": message or f"{label}の時間です",
        "enabled": True,
        "created_at": datetime.now().isoformat()
    }

    alarms.append(alarm)
    alarm_next_id += 1
    save_alarms()

    return f"{time_str}に「{label}」のアラームを設定しました。"


def alarm_list():
    """アラーム一覧を取得"""
    global alarms

    if not alarms:
        return "設定されているアラームはありません。"

    result = "アラーム一覧:\n"
    for alarm in alarms:
        status = "有効" if alarm.get("enabled", True) else "無効"
        result += f"{alarm['id']}. {alarm['time']} - {alarm['label']} ({status})\n"

    return result.strip()


def alarm_delete(alarm_id):
    """アラームを削除"""
    global alarms

    try:
        alarm_id = int(alarm_id)
    except:
        return "アラームIDは数字で指定してください。"

    for i, alarm in enumerate(alarms):
        if alarm['id'] == alarm_id:
            deleted = alarms.pop(i)
            save_alarms()
            return f"「{deleted['label']}」({deleted['time']})のアラームを削除しました。"

    return f"ID {alarm_id} のアラームが見つかりません。"


def check_alarms_and_notify():
    """アラームをチェックして通知（バックグラウンドスレッド用）"""
    global running, alarms

    last_triggered = {}

    while running:
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")

            for alarm in alarms:
                if not alarm.get("enabled", True):
                    continue

                alarm_id = alarm['id']
                alarm_time = alarm['time']

                trigger_key = f"{alarm_id}_{current_time}"
                if trigger_key in last_triggered:
                    continue

                if alarm_time == current_time:
                    print(f"アラーム発動: {alarm['label']} ({alarm_time})")
                    last_triggered[trigger_key] = True

                    with record_lock:
                        if not is_recording:
                            try:
                                message = alarm.get('message', f"{alarm['label']}の時間です")
                                speech_audio = text_to_speech(f"アラームです。{message}")
                                if speech_audio:
                                    play_audio(speech_audio)
                            except Exception as e:
                                print(f"アラーム通知エラー: {e}")

            current_minute = now.strftime("%H:%M")
            keys_to_remove = [k for k in last_triggered if not k.endswith(current_minute)]
            for k in keys_to_remove:
                del last_triggered[k]

        except Exception as e:
            print(f"アラームチェックエラー: {e}")

        time.sleep(10)


def start_alarm_thread():
    """アラーム監視スレッドを開始"""
    global alarm_thread
    alarm_thread = threading.Thread(target=check_alarms_and_notify, daemon=True)
    alarm_thread.start()
    print("アラーム監視スレッド開始")


# ==================== カメラ機能 ====================

def camera_capture():
    """カメラで写真を撮影"""
    try:
        image_path = "/tmp/ai_necklace_capture.jpg"

        result = subprocess.run(
            ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            print(f"カメラエラー: {result.stderr}")
            return None, "カメラでの撮影に失敗しました"

        with open(image_path, "rb") as f:
            image_data = f.read()

        print(f"撮影成功: {image_path}")
        return image_data, None

    except subprocess.TimeoutExpired:
        return None, "カメラの撮影がタイムアウトしました"
    except FileNotFoundError:
        return None, "rpicam-stillコマンドが見つかりません"
    except Exception as e:
        return None, f"カメラエラー: {str(e)}"


def camera_describe(prompt="この画像に何が写っていますか？簡潔に説明してください。"):
    """カメラで撮影してGeminiで画像を解析"""
    global gemini_client

    print("カメラで撮影中...")
    image_data, error = camera_capture()

    if error:
        return error

    print("画像を解析中...")

    try:
        response = gemini_client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=[
                prompt + "\n\n日本語で回答してください。音声で読み上げるため、1-2文程度の簡潔な説明をお願いします。",
                types.Part.from_bytes(data=image_data, mime_type="image/jpeg")
            ]
        )

        return response.text

    except Exception as e:
        return f"画像解析エラー: {str(e)}"


# ==================== Gmail機能 ====================

def init_gmail():
    """Gmail API初期化"""
    global gmail_service

    creds = None
    token_path = CONFIG["gmail_token_path"]
    credentials_path = CONFIG["gmail_credentials_path"]

    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, GMAIL_SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_path):
                print(f"警告: Gmail認証情報が見つかりません: {credentials_path}")
                print("Gmail機能は無効です。")
                return False

            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, GMAIL_SCOPES)
            creds = flow.run_local_server(port=0)

        os.makedirs(os.path.dirname(token_path), exist_ok=True)
        with open(token_path, 'w') as token:
            token.write(creds.to_json())

    try:
        gmail_service = build('gmail', 'v1', credentials=creds)
        print("Gmail API初期化完了")
        return True
    except Exception as e:
        print(f"Gmail API初期化エラー: {e}")
        return False


def gmail_list(query="is:unread", max_results=5):
    """メール一覧を取得"""
    global gmail_service, last_email_list

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        results = gmail_service.users().messages().list(
            userId='me',
            q=query,
            maxResults=max_results
        ).execute()

        messages = results.get('messages', [])

        if not messages:
            return "該当するメールはありません"

        email_list = []
        last_email_list = []

        for i, msg in enumerate(messages, 1):
            msg_detail = gmail_service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()

            headers = {h['name']: h['value'] for h in msg_detail.get('payload', {}).get('headers', [])}

            from_header = headers.get('From', '不明')
            from_match = re.match(r'(.+?)\s*<', from_header)
            from_name = from_match.group(1).strip() if from_match else from_header.split('@')[0]

            email_info = {
                'id': msg['id'],
                'from': from_name,
                'from_email': from_header,
                'subject': headers.get('Subject', '(件名なし)'),
                'date': headers.get('Date', ''),
            }
            last_email_list.append(email_info)
            email_list.append(f"{i}. {from_name}さんから: {email_info['subject']}")

        return "メール一覧:\n" + "\n".join(email_list)

    except HttpError as e:
        return f"メール取得エラー: {e}"


def gmail_read(message_id):
    """メール本文を読み取り"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        msg = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='full'
        ).execute()

        headers = {h['name']: h['value'] for h in msg.get('payload', {}).get('headers', [])}

        body = ""
        payload = msg.get('payload', {})

        if 'body' in payload and payload['body'].get('data'):
            body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8')
        elif 'parts' in payload:
            for part in payload['parts']:
                if part.get('mimeType') == 'text/plain' and part.get('body', {}).get('data'):
                    body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    break

        if len(body) > 500:
            body = body[:500] + "...(以下省略)"

        from_header = headers.get('From', '不明')
        from_match = re.match(r'(.+?)\s*<', from_header)
        from_name = from_match.group(1).strip() if from_match else from_header

        return f"送信者: {from_name}\n件名: {headers.get('Subject', '(件名なし)')}\n\n本文:\n{body}"

    except HttpError as e:
        return f"メール読み取りエラー: {e}"


def gmail_send(to, subject, body):
    """新規メール送信"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        message = MIMEText(body)
        message['to'] = to
        message['subject'] = subject

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        return f"{to}にメールを送信しました"

    except HttpError as e:
        return f"メール送信エラー: {e}"


def extract_email_address(email_str):
    """メールアドレス部分を抽出"""
    if not email_str:
        return None
    match = re.search(r'<([^>]+)>', email_str)
    if match:
        return match.group(1)
    if '@' in email_str:
        return email_str.strip()
    return None


def gmail_send_photo(to=None, subject="写真を送ります", body="", take_photo=True):
    """写真付きメール送信"""
    global gmail_service, last_email_list

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    if not to:
        if not last_email_list:
            return "送信先が指定されていません。先に「メールを確認して」と言うか、宛先を指定してください。"
        to = extract_email_address(last_email_list[0].get('from_email', ''))
        if not to:
            return "直前のメール送信者のアドレスが取得できませんでした"

    try:
        if take_photo:
            print("写真を撮影中...")
            image_path = "/tmp/ai_necklace_capture.jpg"
            result = subprocess.run(
                ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return f"写真の撮影に失敗しました: {result.stderr}"
        else:
            image_path = "/tmp/ai_necklace_capture.jpg"
            if not os.path.exists(image_path):
                return "送信する写真がありません。"

        message = MIMEMultipart()
        message['to'] = to
        message['subject'] = subject

        message.attach(MIMEText(body or "写真を送ります。", 'plain'))

        with open(image_path, 'rb') as f:
            img_data = f.read()

        img_part = MIMEBase('image', 'jpeg')
        img_part.set_payload(img_data)
        encoders.encode_base64(img_part)

        filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        img_part.add_header('Content-Disposition', 'attachment', filename=filename)
        message.attach(img_part)

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw}
        ).execute()

        to_match = re.match(r'(.+?)\s*<', to)
        to_name = to_match.group(1).strip() if to_match else to.split('@')[0]

        return f"{to_name}さんに写真付きメールを送信しました"

    except subprocess.TimeoutExpired:
        return "カメラの撮影がタイムアウトしました"
    except Exception as e:
        return f"写真付きメール送信エラー: {str(e)}"


def gmail_reply(message_id, body, to_email=None, attach_photo=False):
    """メール返信"""
    global gmail_service

    if not gmail_service:
        return "Gmail機能が初期化されていません"

    try:
        image_path = None
        if attach_photo:
            print("写真を撮影中...")
            image_path = "/tmp/ai_necklace_capture.jpg"
            result = subprocess.run(
                ["rpicam-still", "-o", image_path, "-t", "500", "--width", "1280", "--height", "960"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                return f"写真の撮影に失敗しました: {result.stderr}"

        original = gmail_service.users().messages().get(
            userId='me',
            id=message_id,
            format='metadata',
            metadataHeaders=['From', 'Subject', 'Message-ID', 'References', 'Reply-To']
        ).execute()

        headers = {h['name']: h['value'] for h in original.get('payload', {}).get('headers', [])}

        to_raw = to_email or headers.get('Reply-To') or headers.get('From', '')
        to = extract_email_address(to_raw)

        if not to:
            return "返信先のメールアドレスが取得できませんでした"

        subject = headers.get('Subject', '')
        if not subject.startswith('Re:'):
            subject = 'Re: ' + subject

        thread_id = original.get('threadId')
        message_id_header = headers.get('Message-ID', '')
        references = headers.get('References', '')

        if attach_photo and image_path:
            message = MIMEMultipart()
            message['to'] = to
            message['subject'] = subject
            if message_id_header:
                message['In-Reply-To'] = message_id_header
                message['References'] = f"{references} {message_id_header}".strip()

            message.attach(MIMEText(body or "写真を送ります。", 'plain'))

            with open(image_path, 'rb') as f:
                img_data = f.read()
            img_part = MIMEBase('image', 'jpeg')
            img_part.set_payload(img_data)
            encoders.encode_base64(img_part)
            filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            img_part.add_header('Content-Disposition', 'attachment', filename=filename)
            message.attach(img_part)
        else:
            message = MIMEText(body)
            message['to'] = to
            message['subject'] = subject
            if message_id_header:
                message['In-Reply-To'] = message_id_header
                message['References'] = f"{references} {message_id_header}".strip()

        raw = base64.urlsafe_b64encode(message.as_bytes()).decode()

        gmail_service.users().messages().send(
            userId='me',
            body={'raw': raw, 'threadId': thread_id}
        ).execute()

        to_match = re.match(r'(.+?)\s*<', to)
        to_name = to_match.group(1).strip() if to_match else to.split('@')[0]

        if attach_photo:
            return f"{to_name}さんに写真付きで返信しました"
        return f"{to_name}さんに返信を送信しました"

    except subprocess.TimeoutExpired:
        return "カメラの撮影がタイムアウトしました"
    except HttpError as e:
        return f"返信エラー: {e}"


# ==================== 翻訳モード ====================

def translation_mode_on(source_lang="ja", target_lang="en"):
    """翻訳モードを開始"""
    CONFIG["translation_mode"] = True
    CONFIG["source_language"] = source_lang
    CONFIG["target_language"] = target_lang

    lang_names = {
        "ja": "日本語", "en": "英語", "zh": "中国語",
        "ko": "韓国語", "es": "スペイン語", "fr": "フランス語", "de": "ドイツ語",
    }

    source_name = lang_names.get(source_lang, source_lang)
    target_name = lang_names.get(target_lang, target_lang)

    return f"翻訳モードを開始しました。{source_name}から{target_name}に翻訳します。「通訳モード終了」で終了できます。"


def translation_mode_off():
    """翻訳モードを終了"""
    CONFIG["translation_mode"] = False
    return "翻訳モードを終了しました。通常モードに戻ります。"


def detect_language(text):
    """テキストの言語を判定（簡易版）"""
    # 日本語文字が含まれているかチェック
    japanese_chars = sum(1 for c in text if '぀' <= c <= 'ゟ' or '゠' <= c <= 'ヿ' or '一' <= c <= '鿿')
    if japanese_chars > len(text) * 0.1:
        return "ja"
    return "en"


@retry_on_error
def translate_text(text):
    """テキストを翻訳（翻訳結果と出力言語のタプルを返す）"""
    global gemini_client

    source_lang = CONFIG["source_language"]
    target_lang = CONFIG["target_language"]
    
    # 入力言語を判定して出力言語を決定
    detected = detect_language(text)
    if detected == source_lang:
        output_lang = target_lang
    else:
        output_lang = source_lang

    prompt = f"""以下のテキストを翻訳してください。

入力言語が{source_lang}の場合は{target_lang}に翻訳してください。
入力言語が{target_lang}の場合は{source_lang}に翻訳してください。

翻訳結果のみを出力してください。説明や注釈は不要です。

テキスト: {text}"""

    try:
        response = gemini_client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=prompt
        )
        translated = response.text.strip()
        return (translated, output_lang)
    except Exception as e:
        return (f"翻訳エラー: {str(e)}", "ja")


# ==================== ツール実行 ====================

def execute_tool(tool_call):
    """ツール呼び出しを実行"""
    global last_email_list

    tool_name = tool_call.get('tool')
    params = tool_call.get('params', {})

    if tool_name == 'gmail_list':
        return gmail_list(
            query=params.get('query', 'is:unread'),
            max_results=params.get('max_results', 5)
        )
    elif tool_name == 'gmail_read':
        msg_id = params.get('message_id')
        if isinstance(msg_id, int) or (isinstance(msg_id, str) and msg_id.isdigit()):
            idx = int(msg_id) - 1
            if 0 <= idx < len(last_email_list):
                msg_id = last_email_list[idx]['id']
            else:
                return "指定されたメールが見つかりません"
        return gmail_read(msg_id)
    elif tool_name == 'gmail_send':
        return gmail_send(
            to=params.get('to'),
            subject=params.get('subject'),
            body=params.get('body')
        )
    elif tool_name == 'gmail_reply':
        msg_id = params.get('message_id')
        to_email = None
        attach_photo = params.get('attach_photo', False)
        if isinstance(msg_id, int) or (isinstance(msg_id, str) and msg_id.isdigit()):
            idx = int(msg_id) - 1
            if 0 <= idx < len(last_email_list):
                msg_id = last_email_list[idx]['id']
                to_email = last_email_list[idx].get('from_email')
            else:
                return "指定されたメールが見つかりません。先に「メールを確認して」と言ってください。"
        return gmail_reply(msg_id, params.get('body'), to_email, attach_photo)
    elif tool_name == 'alarm_set':
        return alarm_set(
            time_str=params.get('time'),
            label=params.get('label', 'アラーム'),
            message=params.get('message', '')
        )
    elif tool_name == 'alarm_list':
        return alarm_list()
    elif tool_name == 'alarm_delete':
        return alarm_delete(params.get('alarm_id'))
    elif tool_name == 'camera_capture':
        prompt = params.get('prompt', 'この画像に何が写っていますか？簡潔に説明してください。')
        return camera_describe(prompt)
    elif tool_name == 'gmail_send_photo':
        return gmail_send_photo(
            to=params.get('to'),
            subject=params.get('subject', '写真を送ります'),
            body=params.get('body', ''),
            take_photo=params.get('take_photo', True)
        )
    elif tool_name == 'voice_record_send':
        if not firebase_messenger:
            return "音声メッセージ機能が無効です"
        return "VOICE_RECORD_SEND"
    elif tool_name == 'translation_mode_on':
        return translation_mode_on(
            source_lang=params.get('source_lang', 'ja'),
            target_lang=params.get('target_lang', 'en')
        )
    elif tool_name == 'translation_mode_off':
        return translation_mode_off()
    else:
        return f"不明なツール: {tool_name}"


# ==================== オーディオ処理 ====================

def find_audio_device(p, device_type="input"):
    """オーディオデバイスを自動検出"""
    target_names = ["USB PnP Sound", "USB Audio", "USB PnP Audio", "UACDemoV1.0"]

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        name = info.get("name", "")

        if device_type == "input" and info.get("maxInputChannels", 0) > 0:
            for target in target_names:
                if target in name:
                    print(f"入力デバイス検出: [{i}] {name}")
                    return i
        elif device_type == "output" and info.get("maxOutputChannels", 0) > 0:
            for target in target_names:
                if target in name:
                    print(f"出力デバイス検出: [{i}] {name}")
                    return i

    if device_type == "input":
        return p.get_default_input_device_info()["index"]
    else:
        return p.get_default_output_device_info()["index"]


def record_audio_while_pressed():
    """ボタンを押している間録音（トランシーバー方式）"""
    global audio, button, is_recording

    input_device = CONFIG["input_device_index"]
    if input_device is None:
        input_device = find_audio_device(audio, "input")

    print("録音中... (ボタンを離すと停止)")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CONFIG["channels"],
        rate=CONFIG["sample_rate"],
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CONFIG["chunk_size"],
        stream_callback=None
    )

    frames = []
    max_chunks = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * CONFIG["max_record_seconds"])
    recording_timeout = 60
    start_time = time.time()

    with record_lock:
        is_recording = True

    while True:
        if not running:
            break

        elapsed_time = time.time() - start_time
        if elapsed_time > recording_timeout:
            print(f"録音タイムアウト ({recording_timeout}秒経過)、録音終了")
            break

        if button and not button.is_pressed:
            print("ボタンが離されました、録音終了")
            break

        if len(frames) >= max_chunks:
            print("最大録音時間に達しました、録音終了")
            break

        try:
            available = stream.get_read_available()
            if available >= CONFIG["chunk_size"]:
                data = stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
                frames.append(data)
            else:
                time.sleep(0.001)
        except Exception as e:
            print(f"録音中にエラー: {e}")
            break

    with record_lock:
        is_recording = False

    stream.stop_stream()
    stream.close()

    if len(frames) < 5:
        print("録音が短すぎます")
        return None

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CONFIG["channels"])
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(CONFIG["sample_rate"])
        wf.writeframes(b''.join(frames))

    wav_buffer.seek(0)
    return wav_buffer


def record_audio_auto():
    """自動録音（ボタンなしモード、無音検出で停止）"""
    global audio

    input_device = CONFIG["input_device_index"]
    if input_device is None:
        input_device = find_audio_device(audio, "input")

    print("録音開始... 話しかけてください")

    stream = audio.open(
        format=pyaudio.paInt16,
        channels=CONFIG["channels"],
        rate=CONFIG["sample_rate"],
        input=True,
        input_device_index=input_device,
        frames_per_buffer=CONFIG["chunk_size"]
    )

    frames = []
    silent_chunks = 0
    has_sound = False
    max_chunks = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * 5)
    silence_duration = 1.5
    silence_chunks_threshold = int(CONFIG["sample_rate"] / CONFIG["chunk_size"] * silence_duration)

    for i in range(max_chunks):
        if not running:
            break

        data = stream.read(CONFIG["chunk_size"], exception_on_overflow=False)
        frames.append(data)

        audio_data = np.frombuffer(data, dtype=np.int16)
        volume = np.abs(audio_data).mean()

        if volume > CONFIG["silence_threshold"]:
            has_sound = True
            silent_chunks = 0
        else:
            silent_chunks += 1

        if has_sound and silent_chunks > silence_chunks_threshold:
            print("無音検出、録音終了")
            break

    stream.stop_stream()
    stream.close()

    if not has_sound:
        print("音声が検出されませんでした")
        return None

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(CONFIG["channels"])
        wf.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
        wf.setframerate(CONFIG["sample_rate"])
        wf.writeframes(b''.join(frames))

    wav_buffer.seek(0)
    return wav_buffer


@retry_on_error
def transcribe_audio(audio_data):
    """音声をテキストに変換（Gemini API）"""
    global gemini_client

    print("音声認識中...")

    audio_data.seek(0)
    wav_bytes = audio_data.read()

    try:
        response = gemini_client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=[
                "この音声を正確に文字起こししてください。日本語または英語で話されています。文字起こし結果のみを出力してください。",
                types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")
            ]
        )
        return response.text.strip()
    except Exception as e:
        print(f"音声認識エラー: {e}")
        return None


@retry_on_error
def get_ai_response(text):
    """AIからの応答を取得（Gemini Function Calling対応）"""
    global gemini_client, conversation_history

    print(f"AI処理中... (入力: {text})")

    # 翻訳モードの場合
    if CONFIG["translation_mode"]:
        translated, output_lang = translate_text(text)
        return (translated, output_lang)

    conversation_history.append({"role": "user", "content": text})

    if len(conversation_history) > 10:
        conversation_history = conversation_history[-10:]

    # Gemini用のcontentsを構築
    contents = []
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])]
        ))

    try:
        # Function Callingを有効にしてリクエスト
        response = gemini_client.models.generate_content(
            model=CONFIG["gemini_model"],
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CONFIG["system_prompt"],
                max_output_tokens=500,
                tools=TOOLS
            )
        )

        # Function Callがあるかチェック
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    func_call = part.function_call
                    tool_name = func_call.name
                    tool_params = dict(func_call.args) if func_call.args else {}
                    
                    print(f"Function Call: {tool_name}({tool_params})")
                    
                    # ツールを実行
                    tool_call = {"tool": tool_name, "params": tool_params}
                    tool_result = execute_tool(tool_call)
                    print(f"ツール結果: {tool_result}")

                    if tool_result == "VOICE_RECORD_SEND":
                        return "VOICE_RECORD_SEND"

                    # ツール結果をfunction_responseとして返す
                    function_response_content = types.Content(
                        role="user",
                        parts=[types.Part.from_function_response(
                            name=tool_name,
                            response={"result": str(tool_result)}
                        )]
                    )
                    
                    # 元のcontentsにfunction_callとresponseを追加
                    contents.append(types.Content(
                        role="model",
                        parts=[part]
                    ))
                    contents.append(function_response_content)

                    # 要約を取得
                    summary_response = gemini_client.models.generate_content(
                        model=CONFIG["gemini_model"],
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction="ツールの実行結果を音声で読み上げるために、簡潔に日本語で要約してください。",
                            max_output_tokens=300
                        )
                    )

                    final_response = summary_response.text
                    # Function Callingを使った会話は履歴に含めない（エラー防止）
                    # 追加したユーザー入力を削除
                    if conversation_history and conversation_history[-1]["role"] == "user":
                        conversation_history.pop()
                    return final_response

        # 通常のテキスト応答
        ai_response = response.text
        print(f"Gemini応答: {ai_response}")
        conversation_history.append({"role": "assistant", "content": ai_response})
        return ai_response

    except Exception as e:
        error_str = str(e)
        print(f"AI応答エラー: {error_str}")
        # 503エラーの場合はリトライ
        if "503" in error_str or "overloaded" in error_str.lower():
            raise e
        return "申し訳ありません。エラーが発生しました。"




def text_to_speech(text, lang="ja"):
    """テキストを音声に変換（Google Cloud TTS REST API）"""
    import requests
    import base64
    
    api_key = os.getenv('GOOGLE_TTS_API_KEY')
    print(f"音声合成中... (テキスト: {text[:30]}..., 言語: {lang})")

    # 言語に応じた音声設定
    voice_config = {
        'ja': {'languageCode': 'ja-JP', 'name': 'ja-JP-Neural2-B', 'ssmlGender': 'FEMALE'},
        'en': {'languageCode': 'en-US', 'name': 'en-US-Neural2-F', 'ssmlGender': 'FEMALE'},
        'zh': {'languageCode': 'zh-CN', 'name': 'zh-CN-Neural2-A', 'ssmlGender': 'FEMALE'},
        'ko': {'languageCode': 'ko-KR', 'name': 'ko-KR-Neural2-A', 'ssmlGender': 'FEMALE'},
        'fr': {'languageCode': 'fr-FR', 'name': 'fr-FR-Neural2-A', 'ssmlGender': 'FEMALE'},
        'de': {'languageCode': 'de-DE', 'name': 'de-DE-Neural2-A', 'ssmlGender': 'FEMALE'},
        'es': {'languageCode': 'es-ES', 'name': 'es-ES-Neural2-A', 'ssmlGender': 'FEMALE'},
    }
    
    voice = voice_config.get(lang, voice_config['ja'])

    try:
        url = f'https://texttospeech.googleapis.com/v1/text:synthesize?key={api_key}'
        
        payload = {
            'input': {'text': text},
            'voice': voice,
            'audioConfig': {
                'audioEncoding': 'LINEAR16',
                'sampleRateHertz': 24000
            }
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        audio_content = base64.b64decode(response.json()['audioContent'])
        return audio_content

    except Exception as e:
        print(f"音声合成エラー: {e}")
        return None


def generate_notification_sound():
    """通知音を生成（スマホと同じピンポン音）"""
    import numpy as np
    
    sample_rate = 24000
    
    # 1音目: 880Hz, 0.5秒
    duration1 = 0.5
    t1 = np.linspace(0, duration1, int(sample_rate * duration1), False)
    envelope1 = np.exp(-t1 * 6)
    tone1 = envelope1 * np.sin(2 * np.pi * 880 * t1) * 0.3
    
    # 間隔: 150ms
    gap = np.zeros(int(sample_rate * 0.15))
    
    # 2音目: 1320Hz, 0.3秒
    duration2 = 0.3
    t2 = np.linspace(0, duration2, int(sample_rate * duration2), False)
    envelope2 = np.exp(-t2 * 8)
    tone2 = envelope2 * np.sin(2 * np.pi * 1320 * t2) * 0.2
    
    # 結合
    audio = np.concatenate([tone1, gap, tone2])
    audio_data = (audio * 32767).astype(np.int16).tobytes()
    
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)
    
    wav_buffer.seek(0)
    return wav_buffer.read()



def play_audio(audio_data):
    """音声を再生"""
    global audio

    if audio_data is None:
        print("音声データがありません")
        return

    output_device = CONFIG["output_device_index"]
    if output_device is None:
        output_device = find_audio_device(audio, "output")

    print("再生中...")

    wav_buffer = io.BytesIO(audio_data)
    with wave.open(wav_buffer, 'rb') as wf:
        original_rate = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        target_rate = 48000

        # 全フレームを読み込み
        frames = wf.readframes(wf.getnframes())

        # 48000Hz以外の場合はリサンプリング
        if original_rate != target_rate:
            import numpy as np
            # バイトデータをnumpy配列に変換
            audio_array = np.frombuffer(frames, dtype=np.int16)

            # リサンプリング（線形補間）
            original_length = len(audio_array)
            target_length = int(original_length * target_rate / original_rate)
            indices = np.linspace(0, original_length - 1, target_length)
            resampled = np.interp(indices, np.arange(original_length), audio_array)
            frames = resampled.astype(np.int16).tobytes()
            print(f"リサンプリング: {original_rate}Hz → {target_rate}Hz")

        stream = audio.open(
            format=audio.get_format_from_width(sampwidth),
            channels=channels,
            rate=target_rate,
            output=True,
            output_device_index=output_device
        )

        chunk_size = 1024 * sampwidth * channels
        for i in range(0, len(frames), chunk_size):
            if not running:
                break
            stream.write(frames[i:i+chunk_size])

        stream.stop_stream()
        stream.close()


def process_voice():
    """音声処理のメインフロー"""
    global button

    if CONFIG["use_button"] and button:
        audio_data = record_audio_while_pressed()
    else:
        audio_data = record_audio_auto()

    if audio_data is None:
        return

    try:
        text = transcribe_audio(audio_data)
        if not text or text.strip() == "":
            print("テキストが認識できませんでした")
            return

        print(f"\n[あなた] {text}")

        response = get_ai_response(text)
        
        # 翻訳モードの場合はタプル(翻訳結果, 言語)が返る
        if isinstance(response, tuple):
            response_text, output_lang = response
            print(f"[AI] {response_text} ({output_lang})")
            speech_audio = text_to_speech(response_text, output_lang)
        else:
            print(f"[AI] {response}")
            if response == "VOICE_RECORD_SEND":
                record_and_send_voice_message()
                return
            speech_audio = text_to_speech(response)
        if speech_audio:
            play_audio(speech_audio)
        else:
            print("音声合成に失敗しました")

    except Exception as e:
        print(f"⚠️ 処理エラー: {e}")
        import traceback
        traceback.print_exc()


def record_and_send_voice_message():
    """音声を録音してスマホに送信"""
    global button, firebase_messenger, conversation_history

    # 音声メッセージ送信後は会話をリセット（次のリクエストに影響しないように）
    conversation_history = []

    announce = text_to_speech("了解です。押しながら話してください。")
    if announce:
        play_audio(announce)

    print("📢 メッセージを録音中...")
    if CONFIG["use_button"] and button:
        print("ボタンを押して録音を開始してください...")
        while not button.is_pressed and running:
            time.sleep(0.05)
        if not running:
            return
        audio_data = record_audio_while_pressed()
    else:
        audio_data = record_audio_auto()

    if audio_data is None:
        print("録音に失敗しました")
        error_msg = text_to_speech("録音に失敗しました")
        if error_msg:
            play_audio(error_msg)
        return

    print("🔤 音声をテキストに変換中...")
    audio_data.seek(0)
    transcribed_text = None
    try:
        transcribed_text = transcribe_audio(audio_data)
        if transcribed_text:
            print(f"変換されたテキスト: {transcribed_text}")
    except Exception as e:
        print(f"テキスト変換エラー: {e}")

    print("📤 スマホに送信中...")
    audio_data.seek(0)
    if send_voice_to_phone(audio_data, text=transcribed_text):
        success_msg = text_to_speech("メッセージをスマホに送信しました")
        if success_msg:
            play_audio(success_msg)
    else:
        error_msg = text_to_speech("送信に失敗しました")
        if error_msg:
            play_audio(error_msg)


def main():
    """メインループ"""
    global running, gemini_client, audio, button

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Google API キーの確認
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("エラー: GOOGLE_API_KEY が設定されていません")
        print(".env ファイルに GOOGLE_API_KEY=... を設定してください")
        sys.exit(1)

    # Gemini クライアント初期化
    gemini_client = genai.Client(api_key=api_key)
    audio = pyaudio.PyAudio()

    # Gmail初期化
    gmail_available = init_gmail()

    # アラーム初期化
    load_alarms()
    start_alarm_thread()

    # Firebase Voice Messenger初期化
    firebase_available = init_firebase_messenger()

    # ボタン初期化
    if CONFIG["use_button"] and GPIO_AVAILABLE:
        try:
            button = Button(CONFIG["button_pin"], pull_up=True, bounce_time=0.1)
            print(f"ボタン初期化完了: GPIO{CONFIG['button_pin']}")
        except Exception as e:
            print(f"ボタン初期化エラー: {e}")
            print("ボタンなしモードで動作します")
            button = None
            CONFIG["use_button"] = False
    else:
        button = None
        if CONFIG["use_button"]:
            print("GPIOが使用できないため、ボタンなしモードで動作します")
            CONFIG["use_button"] = False

    print("=" * 50)
    print("AI Necklace (Gemini版) 起動")
    print("=" * 50)
    print(f"Gemini Model: {CONFIG['gemini_model']}")
    print(f"TTS Voice: {CONFIG['tts_voice']}")
    print(f"Gmail: {'有効' if gmail_available else '無効'}")
    print(f"Voice Messenger: {'有効' if firebase_available else '無効'}")
    if CONFIG["use_button"]:
        print(f"操作方法: GPIO{CONFIG['button_pin']}のボタンを押している間録音")
    else:
        print("操作方法: 自動録音（無音検出で停止）")
    print("Ctrl+C で終了")
    print("=" * 50)

    print("\n翻訳コマンド例:")
    print("  - 「通訳モードにして」")
    print("  - 「通訳モード終了」")
    print("=" * 50)

    try:
        if CONFIG["use_button"] and button:
            print("\n--- ボタンを押して話しかけてください ---")
            while running:
                try:
                    if button.is_pressed:
                        process_voice()
                        if running:
                            print("\n--- ボタンを押して話しかけてください ---")
                    time.sleep(0.05)
                except Exception as e:
                    print(f"⚠️ ループ内エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    print("処理を継続します...")
                    time.sleep(1)
        else:
            while running:
                try:
                    print("\n--- 待機中 (話しかけてください) ---")
                    process_voice()
                except Exception as e:
                    print(f"⚠️ ループ内エラー: {e}")
                    import traceback
                    traceback.print_exc()
                    print("処理を継続します...")
                    time.sleep(1)

    except KeyboardInterrupt:
        print("\n終了シグナルを受信しました")
    except Exception as e:
        print(f"致命的エラー: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if audio:
            audio.terminate()
        print("終了しました")


if __name__ == "__main__":
    main()
