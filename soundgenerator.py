from preprocess import MinMaxNormaliser
import librosa

class SoundGenerator:
    """Generate audio files from spectrograms"""

    def __init__(self,vae,hop_length):
        self.vae=vae
        self.hop_length=hop_length
        self._min_max_normaliser=MinMaxNormaliser(0,1)

    def generate(self,spectrograms,min_max_values):
        generated_specs,latent_reps=self.vae.reconstruct(spectrograms)
        signals=self.convert_specs_to_audio(generated_specs,min_max_values)

        return signals,latent_reps

    def convert_specs_to_audio(self,spectrograms,min_max_values):
        signals=[]
        for spec,min_max_value in zip(spectrograms,min_max_values):
            # Reshape log-spec
            log_spec=spec[:,:,0]

            #Apply denormalisation
            denorm_log_spec=self._min_max_normaliser.denormalise(log_spec,min_max_value["min"],min_max_value["max"])

            #log-spec to spec
            spec=librosa.db_to_amplitude(denorm_log_spec)

            #Apply Griffin-Lim (ISTFT)
            signal=librosa.istft(spec,hop_length=self.hop_length)

            #append to signals
            signals.append(signal)

        return signals


