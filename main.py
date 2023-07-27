import langid
import openai
import os
import tiktoken
from flask import Flask, request, render_template
from langchain.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from google.cloud import texttospeech
from google.cloud import storage

# Flaskアプリケーションを初期化
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return app.send_static_file('index.html')


@app.route('/translate', methods=['POST'])
def summarize():
    # OpenAI APIキーを環境変数に設定
    os.environ["OPENAI_API_KEY"] = 'xxxx'

    # YoutubeのURLから動画を読み込む
    loader = YoutubeLoader.from_youtube_url(
        youtube_url=request.form['url'],
        language=["ja", 'en']
    )
    docs = loader.load()
    print(f"字幕：{docs[0].page_content}")

    # 字幕言語を取得
    video_lang = detect_language(docs[0].page_content)
    print(f"字幕言語：{video_lang}")

    # OpenAI API を実行
    result = call_ai(docs, video_lang)

    # テキストをHTML用に整形
    translate_html = result["output_text"].replace('\n', '<br>')

    bucket_name = "aiit-programing-report"
    # テキストを音声に変換し、ファイル名を取得
    filename = text_to_speech(result["output_text"], bucket_name, video_lang)
    print(f"音声ファイル名：{filename}")

    # 音声ファイルのパブリックなURL生成
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.make_public()
    print(f"ファイルURL：{blob.public_url}")

    return render_template('translate.html', translate=translate_html, audio_url=blob.public_url)


def call_ai(docs, video_lang):
    # GPT-3.5-turboモデルのためのエンコーディングを取得
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    tokens = encoding.encode(docs[0].page_content)
    tokens_count = len(tokens)
    print(f"トークン数{tokens_count}")

    # チャットモデルを初期化
    chat_model = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0)

    # 動画の言語が英語の場合、問い合わせ文を英語に変更します
    query = '内容を日本語のラップにして'
    if video_lang == 'English':
        query = 'write a rap song about it'

    chain_type = "stuff"

    # トークン数が4096を超える場合、テキストを分割
    if tokens_count > 4096:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        docs = text_splitter.create_documents([docs[0].page_content])
        chain_type = "map_reduce"

    # チェーンを実行
    chain = load_qa_chain(chat_model, chain_type=chain_type)
    result = chain({"input_documents": docs, "question": query})
    return result


def detect_language(input_text):
    # 言語を検出
    language, _ = langid.classify(input_text)
    if language == 'en':
        return 'English'
    elif language == 'ja':
        return 'Japanese'
    else:
        return 'Other'


def text_to_speech(text, bucket_name, video_lang):
    client = texttospeech.TextToSpeechClient()

    # 読み上げるテキストを設定
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # 声のリクエストを作成
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.MALE,
    )

    # 動画の言語が日本語の場合、声の設定を日本語に変更
    if video_lang == 'Japanese':
        voice = texttospeech.VoiceSelectionParams(
            language_code="ja-JP",
            ssml_gender=texttospeech.SsmlVoiceGender.MALE,
        )

    # 音声ファイルのタイプを選択
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # テキストを音声に変換
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    return upload_file(text, bucket_name, response)


def upload_file(text, bucket_name, response):
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)

    filename = f"audio_{hash(text)}.mp3"

    blob = bucket.blob(filename)

    # ファイルをアップロード
    blob.upload_from_string(response.audio_content)

    print(f'ファイルの保存先： gs://{bucket_name}/{filename}')

    return filename


if __name__ == "__main__":
    app.run(port=8080)
