from src.config import EEG_BANDS


def get_dynamic_diffusion(time_step):
    """Dynamically adjust diffusion constant based on EEG bands."""
    if time_step < 500:
        return EEG_BANDS["delta"]
    elif time_step < 1000:
        return EEG_BANDS["theta"]
    elif time_step < 1500:
        return EEG_BANDS["alpha"]
    elif time_step < 1800:
        return EEG_BANDS["beta"]
    else:
        return EEG_BANDS["gamma"]
