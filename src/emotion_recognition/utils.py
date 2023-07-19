import numpy as np
import pandas as pd

merge_columns = ["person_id", "image_id", "valence_rating", "arousal_rating", "valence_norm", "arousal_norm"]


def calculate_basic_stats(df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
    kurtosis = (
        df.groupby(merge_columns, group_keys=True)[feature_column]
        .apply(pd.DataFrame.kurt)
        .reset_index(name=f"kurtosis_{feature_column}")
    )

    return (
        df.groupby(merge_columns, group_keys=True)[feature_column]
        .agg(["mean", "median", "skew"])
        .reset_index()
        .rename(
            columns={
                "mean": f"mean_{feature_column}",
                "median": f"median_{feature_column}",
                "skew": f"skew_{feature_column}",
            }
        )
        .merge(kurtosis, how="outer", on=merge_columns)
    )


def detect_saccades(pupil_positions, eye, velocity_threshold=1, min_duration=0.02, max_duration=0.3):
    pupil_positions = pupil_positions.reset_index(drop=True)
    pupil_positions["velocity_x"] = np.abs(pupil_positions["norm_pos_x"].diff()) / pupil_positions["timestamp"].diff()
    pupil_positions["velocity_y"] = np.abs(pupil_positions["norm_pos_y"].diff()) / pupil_positions["timestamp"].diff()

    pupil_positions.loc[
        (pupil_positions["velocity_x"] > velocity_threshold) | (pupil_positions["velocity_y"] > velocity_threshold),
        "velocity_threshold_surpassed",
    ] = True
    pupil_positions["velocity_threshold_surpassed"] = pupil_positions["velocity_threshold_surpassed"].fillna(False)

    saccades = {
        "start_time": list(),
        "saccade_duration": list(),
        "saccade_amplitude_x": list(),
        "saccade_amplitude_y": list(),
        "saccade_amplitude_cartesian": list(),
    }
    i = 0

    while i < len(pupil_positions.index):
        if pupil_positions.loc[i, "velocity_threshold_surpassed"]:
            start_time = pupil_positions.loc[i, "timestamp"]
            end_time = pupil_positions.loc[i, "timestamp"]
            start_pos_x = pupil_positions.loc[i, "norm_pos_x"]
            end_pos_x = pupil_positions.loc[i, "norm_pos_x"]
            start_pos_y = pupil_positions.loc[i, "norm_pos_y"]
            end_pos_y = pupil_positions.loc[i, "norm_pos_y"]

            while pupil_positions.loc[i, "velocity_threshold_surpassed"] and i < len(pupil_positions.index) - 1:
                i += 1
                end_time = pupil_positions.loc[i, "timestamp"]
                end_pos_x = pupil_positions.loc[i, "norm_pos_x"]
                end_pos_y = pupil_positions.loc[i, "norm_pos_y"]

            if min_duration < end_time - start_time < max_duration:
                saccades["start_time"].append(start_time)
                saccades["saccade_duration"].append(end_time - start_time)
                saccades["saccade_amplitude_x"].append(np.abs(end_pos_x - start_pos_x))
                saccades["saccade_amplitude_y"].append(np.abs(end_pos_y - start_pos_y))

                saccades["saccade_amplitude_cartesian"].append(
                    np.sqrt((end_pos_x - start_pos_x) ** 2 + (end_pos_y - start_pos_y) ** 2)
                )
        i += 1

    saccades = pd.DataFrame(saccades)
    saccades["eye_id"] = 0.0 if eye == "left" else 1.0

    return pd.DataFrame(saccades)
