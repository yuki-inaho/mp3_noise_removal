from random import sample
import numpy as np
import noisereduce as nr
from pydub import AudioSegment

audio_data = AudioSegment.from_mp3("data/iPauJ_T8OMQCRwtm.mp3")
audio_data_array = np.asarray(audio_data.get_array_of_samples()).reshape(-1, 1)
audio_data_array_cleaned = nr.reduce_noise(
    y=audio_data_array.T,
    sr=audio_data.frame_rate,
    n_std_thresh_stationary=1.0,
    stationary=True
).T

audio_data_cleaned = AudioSegment(
    data=audio_data_array_cleaned,
    sample_width=audio_data.sample_width,
    frame_rate=audio_data.frame_rate,
    channels=audio_data.channels
)
audio_data_cleaned.export("data/iPauJ_T8OMQCRwtm_cleaned.mp3", format="mp3")
