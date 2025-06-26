from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    left_lim: int
    right_lim: int
    time_disc_left: int
    time_disc_right: int
    peak_left: int
    peak_right: int
    IS_LOG: bool
    all_peaks: bool
    derivative_slope: float

    class Config:
        env_file = ".env"

settings = Settings()