The near-continuous speech recognition demo using [OpenAI whisper](https://github.com/openai/whisper), built using [Gradio](https://gradio.app/).

### How to run?
Install openai/whisper

    pip install git+https://github.com/openai/whisper.git

Install requirements

    pip install -r requirements.txt

Start the Gradio app

    python whisper_demo.py

### Simple Notes
1. The near-continuous recognition is implemented by incrementally recognizing all historical audio streaming every N seconds. The config is `REC_INTERVAL_IN_SECONDS`

2. The near-continuous recognition is in fact quite broken(slow) and only used for demo purpose. You should try a web socket way for real time recognition by referring to https://github.com/shirayu/whispering

