import datetime
import os
os.system('pip install git+https://github.com/openai/whisper.git')
import gradio as gr
import wave
import whisper
import logging
import torchaudio
import torchaudio.functional as F

LOGGING_FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=LOGGING_FORMAT,level=logging.INFO)

REC_INTERVAL_IN_SECONDS = 3

# tmp dir to store audio files.
if not os.path.isdir('./tmp/'):
    os.mkdir('./tmp')

class WhisperStreaming():
    def __init__(self, model_name='base', language='en', fp16=False):
        self.model_name = model_name
        self.language = language
        self.fp16 = fp16
        self.whisper_model = whisper.load_model(f'{model_name}.{language}')
        self.decode_option = whisper.DecodingOptions(language=self.language,
                                                     without_timestamps=True,
                                                     fp16=self.fp16)
        self.whisper_sample_rate = 16000

    def transcribe_audio_file(self, wave_file_path):
        waveform, sample_rate = torchaudio.load(wave_file_path)
        resampled_waveform = F.resample(waveform, sample_rate, self.whisper_sample_rate, lowpass_filter_width=6)
        audio_tmp = whisper.pad_or_trim(resampled_waveform[0])
        mel = whisper.log_mel_spectrogram(audio_tmp)
        results = self.whisper_model.decode(mel, self.decode_option)
        return results

def concat_multiple_wav_files(wav_files):
    logging.info(f'Concat {wav_files}')
    concat_audio = []
    for wav_file in wav_files:
        w = wave.open(wav_file, 'rb')
        concat_audio.append([w.getparams(), w.readframes(w.getnframes())])
        w.close()
        logging.info(f'Delete audio file {wav_file}')
        os.remove(wav_file)

    output_file_name = f'{datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")}.wav'
    output_file_path = os.path.join('./tmp', output_file_name)
    output = wave.open(output_file_path, 'wb')
    output.setparams(concat_audio[0][0])

    for i in range(len(concat_audio)):
        output.writeframes(concat_audio[i][1])
    output.close()
    logging.info(f'Concat past {len(wav_files)} wav files into {output_file_path}')
    return output_file_path


# fp16 indicates whether using Float16 or Float32. Normally, PyTorch does not support fp16 when run on CPU
whisper_model = WhisperStreaming(model_name='base', language='en', fp16=False)


def transcribe(audio, state={}):
    logging.info(f'Transcribe audio file {audio}')
    print('=====================')
    logging.info(state)

    if not state:
        state['concated_audio'] = audio
        state['result_text'] = 'Waitting...'
        state['count'] = 0
    else:
        state['concated_audio'] = concat_multiple_wav_files([state['concated_audio'], audio])
        state['count'] += 1

    if state['count'] % REC_INTERVAL_IN_SECONDS == 0 and state['count'] > 0:
        logging.info('start to transcribe.......')
        result = whisper_model.transcribe_audio_file(state['concated_audio'])
        logging.info('complete transcribe.......')
        state['result_text'] = result.text
        logging.info('The text is:' + state['result_text'])
    else:
        logging.info(f'The count of streaming is {state["count"]}, and skip speech recognition')

    return state['result_text'], state


gr.Interface(fn=transcribe,
             inputs=[gr.Audio(source="microphone", type='filepath', streaming=True), 'state'],
             outputs = ['text', 'state'],
             live=True).launch()