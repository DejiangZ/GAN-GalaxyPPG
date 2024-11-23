import os
import numpy as np
import pandas as pd
from scipy import signal
from filters import normalize, filter_ppg
from models import evaluate
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from collections import OrderedDict


class HeartRateComparison:
    def __init__(self, window_data_dir):
        self.window_data_dir = window_data_dir
        self.input_sample_rate = 25  # Galaxy Watch Rate
        self.target_sample_rate = 128  # Model Expected is 128HZ
        self.window_size = 8  # Galaxy PPG Window Segment

        # Dataset Activity Label
        self.activity_labels = OrderedDict({
            'adaptation': 'Adaptation',
            'baseline': 'Baseline',
            'tsst-prep': 'TSST\nPrep',
            'tsst-speech': 'TSST\nSpeech',
            'meditation-1': 'Meditation 1',
            'screen-reading': 'Reading',
            'ssst-prep': 'SSST\nPrep',
            'ssst-sing': 'SSST\nSing',
            'meditation-2': 'Meditation 2',
            'keyboard-typing': 'Keyboard',
            'rest-1': 'Rest 1',
            'mobile-typing': 'Mobile',
            'rest-2': 'Rest 2',
            'standing': 'Standing',
            'rest-3': 'Rest 3',
            'walking': 'Walking',
            'rest-4': 'Rest 4',
            'jogging': 'Jogging',
            'rest-5': 'Rest 5',
            'running': 'Running'
        })

    def preprocess_signal(self, signal_str):
        if pd.isna(signal_str):
            return None

        try:
            original_signal = np.array([float(x) for x in signal_str.split(';')])
            if len(original_signal) < self.input_sample_rate * self.window_size * 0.5:
                return None

            target_length = 1024
            resampled_signal = signal.resample(original_signal, target_length)

            filtered_signal = filter_ppg(resampled_signal, self.target_sample_rate)

            normalized_signal = normalize(filtered_signal)

            if len(normalized_signal) >= 512:
                start_idx = (len(normalized_signal) - 512) // 2
                model_input = normalized_signal[start_idx:start_idx + 512]

                if np.isnan(model_input).any() or np.isinf(model_input).any():
                    return None

                return model_input
            return None

        except Exception as e:
            print(f"Error preprocessing signal: {str(e)}")
            return None

    def validate_signal_quality(self, signal):
        if signal is None:
            return False

        if len(signal) != 512:
            return False

        if np.isnan(signal).any() or np.isinf(signal).any():
            return False

        if np.min(signal) < -1.1 or np.max(signal) > 1.1:
            return False

        if np.std(signal) < 0.01:
            return False

        return True

    def compare_heart_rates(self, participant_id):
        try:
            input_file = os.path.join(self.window_data_dir, f"{participant_id}_processed_GD.csv")
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                return None, None

            df = pd.read_csv(input_file)
            results = []

            valid_windows = []
            valid_indices = []

            for session_name in self.activity_labels.keys():
                session_data = df[df['session'] == session_name]

                for idx, row in session_data.iterrows():
                    if pd.notna(row['galaxyPPG']) and pd.notna(row['gdHR']):
                        ppg_signal = self.preprocess_signal(row['galaxyPPG'])

                        if ppg_signal is not None and self.validate_signal_quality(ppg_signal):
                            valid_windows.append(ppg_signal)
                            valid_indices.append(idx)

            if not valid_windows:
                print(f"No valid windows found for participant {participant_id}")
                return None, None

            ppg_windows = np.array(valid_windows)
            print(f"Processing {len(ppg_windows)} valid windows for {participant_id}")

            if ppg_windows.shape[1] != 512:
                print(f"Invalid input shape: {ppg_windows.shape}")
                return None, None

            generated_ppg, heart_rates = evaluate(ppg_windows)

            for i, idx in enumerate(valid_indices):
                row = df.iloc[idx]
                results.append({
                    'participant': participant_id,
                    'session': row['session'],
                    'windowNumber': row['windowNumber'],
                    'startTime': row['startTime'],
                    'endTime': row['endTime'],
                    'gdHR': row['gdHR'],
                    'modelHR': heart_rates[i],
                    'absError': abs(row['gdHR'] - heart_rates[i])
                })

            results_df = pd.DataFrame(results)

            session_stats = {}
            for session in self.activity_labels.keys():
                session_data = results_df[results_df['session'] == session]
                if not session_data.empty:
                    mae = mean_absolute_error(session_data['gdHR'], session_data['modelHR'])
                    rmse = np.sqrt(mean_squared_error(session_data['gdHR'], session_data['modelHR']))
                    correlation = np.corrcoef(session_data['gdHR'], session_data['modelHR'])[0, 1]

                    session_stats[session] = {
                        'MAE': mae,
                        'RMSE': rmse,
                        'Correlation': correlation,
                        'Count': len(session_data),
                        'Mean_gdHR': session_data['gdHR'].mean(),
                        'Mean_modelHR': session_data['modelHR'].mean()
                    }

            return results_df, session_stats

        except Exception as e:
            print(f"Error comparing heart rates for participant {participant_id}: {str(e)}")
            return None, None

    def analyze_all_participants(self):
        files = [f for f in os.listdir(self.window_data_dir) if f.endswith('_processed_GD.csv')]
        all_results = []
        all_session_stats = {}

        for session in self.activity_labels.keys():
            all_session_stats[session] = {
                'MAE_list': [],
                'RMSE_list': [],
                'Correlation_list': [],
                'Total_windows': 0,
                'Valid_participants': []
            }

        for file in files:
            participant_id = file.split('_')[0]
            print(f"\nAnalyzing {participant_id}...")
            results, session_stats = self.compare_heart_rates(participant_id)

            if results is not None and session_stats is not None:
                all_results.append(results)

                for session, stats in session_stats.items():
                    if stats['Count'] > 0:
                        all_session_stats[session]['MAE_list'].append(stats['MAE'])
                        all_session_stats[session]['RMSE_list'].append(stats['RMSE'])
                        all_session_stats[session]['Correlation_list'].append(stats['Correlation'])
                        all_session_stats[session]['Total_windows'] += stats['Count']
                        all_session_stats[session]['Valid_participants'].append(participant_id)

        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)

            activity_summary = []
            for session in self.activity_labels.keys():
                stats = all_session_stats[session]
                if stats['MAE_list']:
                    summary = {
                        'Activity': self.activity_labels[session],
                        'Mean_MAE': np.mean(stats['MAE_list']),
                        'Std_MAE': np.std(stats['MAE_list']),
                        'Mean_RMSE': np.mean(stats['RMSE_list']),
                        'Mean_Correlation': np.mean(stats['Correlation_list']),
                        'Total_Windows': stats['Total_windows'],
                        'Participants_Count': len(stats['Valid_participants']),
                        'Valid_Participants': ', '.join(stats['Valid_participants'])
                    }
                    activity_summary.append(summary)

            summary_df = pd.DataFrame(activity_summary)
            summary_df = summary_df.round(3)

            output_dir = os.path.join(self.window_data_dir, "analysis_results")
            os.makedirs(output_dir, exist_ok=True)

            combined_results.to_csv(os.path.join(output_dir, "all_heart_rate_results.csv"), index=False)
            summary_df.to_csv(os.path.join(output_dir, "activity_summary.csv"), index=False)

            print("\nActivity-wise Statistics:")
            print(summary_df.to_string())

            return combined_results, summary_df

        return None, None



def main():
    window_data_dir = '/Users/dejiang.zheng/Library/CloudStorage/GoogleDrive-dejiang.jeong@gmail.com/其他计算机/Home/Dataset/ppg/WindowData'

    analyzer = HeartRateComparison(window_data_dir)

    combined_results, summary_df = analyzer.analyze_all_participants()



if __name__ == "__main__":
    main()