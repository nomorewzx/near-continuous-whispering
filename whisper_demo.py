import datetime
import os
os.system('pip install git+https://github.com/openai/whisper.git')
from whisper.audio import N_SAMPLES
import gradio as gr
import wave
import whisper
import logging
import torchaudio
import torchaudio.functional as F

LOGGING_FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=LOGGING_FORMAT,level=logging.INFO)

RECOGNITION_INTERVAL = 6
CNT_PER_CHUNK = 12
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
        audio_tmp = whisper.pad_or_trim(resampled_waveform[0], length=N_SAMPLES)
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
    # Whisper only take maximum 30s of audio as input.
    # And the gradio streaming does not guarantee each callback is 1s, And I set CNT_PER_CHUNK as 6, it's just a rough guess that 6 callbacks does not sum up an audio longer than 30s.
    # The logic of chunk splitting could be improved by reading exact how many samples in audio files.
    # After count reach CNT_PER_CHUNK * n, a new audio file is created.
    # However the text should not change.

    if not state:
        state['all_chunk_texts'] = 'Waitting...'
        state['count'] = 0
        state['chunks'] = {}
        return state['all_chunk_texts'], state

    chunk = state['count'] // CNT_PER_CHUNK
    chunk_offset = state['count'] % CNT_PER_CHUNK

    if chunk_offset == 0:
        state['chunks'][chunk] = {}
        state['chunks'][chunk]['concated_audio'] = audio
        state['chunks'][chunk]['result_text'] = ''
    else:
        state['chunks'][chunk]['concated_audio'] = concat_multiple_wav_files([state['chunks'][chunk]['concated_audio'], audio])

    state['count'] += 1

    # Determin if recognizes current chunk.
    if (chunk_offset + 1) % RECOGNITION_INTERVAL == 0 and chunk_offset > 0:
        logging.info(f'start to transcribe chunk: {chunk}, offset: {chunk_offset}')
        result = whisper_model.transcribe_audio_file(state['chunks'][chunk]['concated_audio'])
        logging.info('complete transcribe.......')
        state['chunks'][chunk]['result_text'] = result.text
        logging.info('The text is:' + state['chunks'][chunk]['result_text'])
    else:
        logging.info(f'The offset of streaming chunk is {chunk_offset}, and skip speech recognition')

    # Concat result_texts of all chunks
    result_texts = ''

    for tmp_chunk_idx, tmp_chunk_values in state['chunks'].items():
        result_texts += tmp_chunk_values['result_text'] + ' '

    state['all_chunk_texts'] = result_texts

    return state['all_chunk_texts'], state

# Make sure not missing any audio clip.
assert CNT_PER_CHUNK % RECOGNITION_INTERVAL == 0

STEP_ONE_DESCRIPTION = '''
    Model: base
    Language: en
<div>
    <h3>
        Step1. Click button <i>"Record from microphone"</i> and allow this site to use your microphone.
    </h3>
    <note>Right now the continuous Speech to text transcription is lag and sometimes missing some sentences...</note>
</div>
'''

STEP_TWO_DESCRIPTION = '''
<div align=center>
    <h3 style="font-weight: 900; margin-bottom: 7px;">
        Step2. Try to play the video and see how Whisper transcribe!
    </h3>
    <p>
        Note: make sure using speaker that your computer microphone is able to hear! i.e. computer default speaker
    </p>
    <video id="video" width=50% controls="" preload="none">
        <source id="mp4" src="https://nomorewzx.github.io/near-continuous-whispering/demo_video/whisper_demo.mp4" type="video/mp4">
    </videos>
</div>
'''

gr.Interface(fn=transcribe,
             inputs=[gr.Audio(source="microphone", type='filepath', streaming=True), 'state'],
             outputs = ['text', 'state'],
             description=STEP_ONE_DESCRIPTION,
             article=STEP_TWO_DESCRIPTION,
             live=True).queue(concurrency_count=5).launch()

