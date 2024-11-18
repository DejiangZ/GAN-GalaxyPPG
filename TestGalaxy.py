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
        self.input_sample_rate = 25  # Galaxy Watch的采样率
        self.target_sample_rate = 128  # 模型期望的采样率
        self.window_size = 8  # 窗口大小（秒）

        # 定义活动顺序
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
        """按照原模型要求处理信号"""
        if pd.isna(signal_str):
            return None

        try:
            # 1. 转换原始信号
            original_signal = np.array([float(x) for x in signal_str.split(';')])
            if len(original_signal) < self.input_sample_rate * self.window_size * 0.5:  # 确保至少有一半的数据点
                return None

            # 2. 重采样到128Hz
            target_length = int(self.window_size * self.target_sample_rate)
            resampled_signal = signal.resample(original_signal, target_length)

            # 3. 使用原模型的filter_ppg进行带通滤波
            filtered_signal = filter_ppg(resampled_signal, self.target_sample_rate)

            # 4. 归一化
            normalized_signal = normalize(filtered_signal)

            # 5. 提取中间512点
            if len(normalized_signal) >= 512:
                start_idx = (len(normalized_signal) - 512) // 2
                model_input = normalized_signal[start_idx:start_idx + 512]

                # 6. 验证信号质量
                if np.isnan(model_input).any() or np.isinf(model_input).any():
                    return None

                return model_input
            return None

        except Exception as e:
            print(f"Error preprocessing signal: {str(e)}")
            return None

    def validate_signal_quality(self, signal):
        """验证信号质量"""
        if signal is None:
            return False

        # 检查信号长度
        if len(signal) != 512:
            return False

        # 检查是否包含无效值
        if np.isnan(signal).any() or np.isinf(signal).any():
            return False

        # 检查信号范围（归一化后应该在[-1,1]范围内）
        if np.min(signal) < -1.1 or np.max(signal) > 1.1:
            return False

        # 检查信号方差（避免完全平坦的信号）
        if np.std(signal) < 0.01:
            return False

        return True

    def compare_heart_rates(self, participant_id):
        """比较模型生成的心率与gdHR，并按活动分类统计"""
        try:
            input_file = os.path.join(self.window_data_dir, f"{participant_id}_processed_GD.csv")
            if not os.path.exists(input_file):
                print(f"File not found: {input_file}")
                return None, None

            df = pd.read_csv(input_file)
            results = []

            valid_windows = []
            valid_indices = []

            # 按session处理数据，确保时序性
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

            # 批量处理有效窗口
            ppg_windows = np.array(valid_windows)
            print(f"Processing {len(ppg_windows)} valid windows for {participant_id}")

            # 确保输入形状正确
            if ppg_windows.shape[1] != 512:
                print(f"Invalid input shape: {ppg_windows.shape}")
                return None, None

            generated_ppg, heart_rates = evaluate(ppg_windows)

            # 收集每个窗口的结果
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

            # 按活动顺序计算统计数据
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
        """分析所有参与者的数据，并按活动统计"""
        files = [f for f in os.listdir(self.window_data_dir) if f.endswith('_processed_GD.csv')]
        all_results = []
        all_session_stats = {}

        # 初始化所有活动的统计数据结构
        for session in self.activity_labels.keys():
            all_session_stats[session] = {
                'MAE_list': [],
                'RMSE_list': [],
                'Correlation_list': [],
                'Total_windows': 0,
                'Valid_participants': []
            }

        # 处理每个参与者的数据
        for file in files:
            participant_id = file.split('_')[0]
            print(f"\nAnalyzing {participant_id}...")
            results, session_stats = self.compare_heart_rates(participant_id)

            if results is not None and session_stats is not None:
                all_results.append(results)

                # 累积每个活动的统计数据
                for session, stats in session_stats.items():
                    if stats['Count'] > 0:  # 只记录有有效数据的session
                        all_session_stats[session]['MAE_list'].append(stats['MAE'])
                        all_session_stats[session]['RMSE_list'].append(stats['RMSE'])
                        all_session_stats[session]['Correlation_list'].append(stats['Correlation'])
                        all_session_stats[session]['Total_windows'] += stats['Count']
                        all_session_stats[session]['Valid_participants'].append(participant_id)

        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)

            # 计算并保存每个活动的总体统计结果
            activity_summary = []
            for session in self.activity_labels.keys():
                stats = all_session_stats[session]
                if stats['MAE_list']:  # 如果有数据
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

            # 创建并保存活动统计摘要
            summary_df = pd.DataFrame(activity_summary)
            summary_df = summary_df.round(3)

            # 保存结果
            output_dir = os.path.join(self.window_data_dir, "analysis_results")
            os.makedirs(output_dir, exist_ok=True)

            # 保存详细结果
            combined_results.to_csv(os.path.join(output_dir, "all_heart_rate_results.csv"), index=False)
            # 保存活动统计摘要
            summary_df.to_csv(os.path.join(output_dir, "activity_summary.csv"), index=False)

            # 打印活动统计摘要
            print("\nActivity-wise Statistics:")
            print(summary_df.to_string())

            return combined_results, summary_df

        return None, None

    def visualize_activity_results(self, summary_df):
        """可视化各个活动的MAE结果"""
        plt.figure(figsize=(15, 8))

        activities = summary_df['Activity']
        mae_values = summary_df['Mean_MAE']
        mae_std = summary_df['Std_MAE']

        # 创建条形图
        bars = plt.bar(range(len(activities)), mae_values, yerr=mae_std, capsize=5)

        # 设置图表属性
        plt.xticks(range(len(activities)), activities, rotation=45, ha='right')
        plt.ylabel('Mean Absolute Error (MAE)')
        plt.title('Heart Rate Estimation MAE by Activity')

        # 添加数值标签
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{mae_values[i]:.2f}\nn={summary_df.iloc[i]["Participants_Count"]}',
                     ha='center', va='bottom')

        plt.tight_layout()

        # 保存图表
        output_dir = os.path.join(self.window_data_dir, "analysis_results")
        plt.savefig(os.path.join(output_dir, "activity_mae_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()


def main():
    window_data_dir = '/Users/dejiang.zheng/Library/CloudStorage/GoogleDrive-dejiang.jeong@gmail.com/其他计算机/Home/Dataset/ppg/WindowData'

    analyzer = HeartRateComparison(window_data_dir)

    # 分析所有参与者
    combined_results, summary_df = analyzer.analyze_all_participants()

    if combined_results is not None and summary_df is not None:
        # 创建可视化
        analyzer.visualize_activity_results(summary_df)
        print("\nAnalysis completed. Results have been saved to the analysis_results directory.")
    else:
        print("Analysis failed: No valid results generated.")


if __name__ == "__main__":
    main()