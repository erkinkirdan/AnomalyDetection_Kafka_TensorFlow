import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from dateutil.parser import parse
import matplotlib as mpl
from scipy.stats import sem, t

# Set up Matplotlib parameters
mpl.rcParams['font.weight'] = 'bold'
mpl.rcParams['font.size'] = 12

def read_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def evaluate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    confusion = confusion_matrix(y_true, y_pred)

    print("Performance Evaluation:")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"Confusion Matrix:\n{confusion}")

def calculate_latencies(pred_data):
    stream_latency = [(parse(row[6]) - parse(row[5])).total_seconds() * 1000 for row in pred_data[1:]]
    prediction_latency = [(parse(row[7]) - parse(row[6])).total_seconds() * 1000 for row in pred_data[1:]]
    total_latency = [(parse(row[7]) - parse(row[5])).total_seconds() * 1000 for row in pred_data[1:]]
    return stream_latency, prediction_latency, total_latency

def evaluate_latencies(latencies):
    for latency_name, latency_values in zip(["Stream Latency", "Prediction Latency", "Total Latency"], latencies):
        mean = np.mean(latency_values)
        standard_error = sem(latency_values)
        confidence_interval = t.ppf((1 + 0.95) / 2, len(latency_values) - 1) * standard_error

        print(f"{latency_name}: Mean = {mean}, Confidence Interval = ({mean - confidence_interval}, {mean + confidence_interval})")

def plot_latencies(latencies):
    stream_latency, prediction_latency, total_latency = latencies

    stream_latency_sorted, prediction_latency_sorted, total_latency_sorted = map(np.sort, latencies)

    # Cumulative Distribution Function (CDF)
    cdf_stream_latency = np.arange(len(stream_latency)) / float(len(stream_latency))
    cdf_prediction_latency = np.arange(len(prediction_latency)) / float(len(prediction_latency))
    cdf_total_latency = np.arange(len(total_latency)) / float(len(total_latency))

    # Create a figure with two subplots (vertical arrangement)
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))

    # Cumulative Distribution Function (CDF) - First Subplot
    axs[0].plot(stream_latency_sorted, cdf_stream_latency, label='Stream Latency', linewidth=2.0)
    axs[0].plot(prediction_latency_sorted, cdf_prediction_latency, label='Prediction Latency', linewidth=2.0)
    axs[0].plot(total_latency_sorted, cdf_total_latency, label='Total Latency', linewidth=2.0)
    axs[0].set_xscale('log')  # Set the x-axis to a logarithmic scale

    # 50th Percentile
    y_line_50 = 0.5
    axs[0].axhline(y=y_line_50, color='r', linestyle='--', linewidth=2.0)
    axs[0].text(0.9, y_line_50, 'P50', color='r', va='center', ha='left', backgroundcolor='w', fontweight='bold')
    x_stream_latency_50 = np.interp(y_line_50, cdf_stream_latency, stream_latency_sorted)
    x_prediction_latency_50 = np.interp(y_line_50, cdf_prediction_latency, prediction_latency_sorted)
    x_total_latency_50 = np.interp(y_line_50, cdf_total_latency, total_latency_sorted)
    axs[0].annotate(f'{x_stream_latency_50:.2f}', xy=(x_stream_latency_50, y_line_50), xycoords='data',
                     xytext=(-30, -30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))
    axs[0].annotate(f'{x_prediction_latency_50:.2f}', xy=(x_prediction_latency_50, y_line_50), xycoords='data',
                     xytext=(30, 30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))
    axs[0].annotate(f'{x_total_latency_50:.2f}', xy=(x_total_latency_50, y_line_50), xycoords='data',
                     xytext=(30, 30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))

    axs[0].set_title('CDF of Latencies', fontweight='bold')
    axs[0].set_xlabel('Latency (ms)', fontweight='bold')
    axs[0].set_ylabel('CDF', fontweight='bold')
    axs[0].grid(True)
    axs[0].legend()

    ccdf_stream_latency = 1 - np.arange(len(stream_latency)) / len(stream_latency)
    ccdf_prediction_latency = 1 - np.arange(len(prediction_latency)) / len(prediction_latency)
    ccdf_total_latency = 1 - np.arange(len(total_latency)) / len(total_latency)

    # Plot the CCDF - Second Subplot
    axs[1].loglog(stream_latency_sorted, ccdf_stream_latency, label='Stream Latency', linewidth=2.0)
    axs[1].loglog(prediction_latency_sorted, ccdf_prediction_latency, label='Prediction Latency', linewidth=2.0)
    axs[1].loglog(total_latency_sorted, ccdf_total_latency, label='Total Latency', linewidth=2.0)

    # 99th Percentile
    y_line_99 = 10 ** -2
    plt.axhline(y=y_line_99, color='r', linestyle='--', linewidth=2.0)
    plt.text(0.9, y_line_99, 'P99', color='r', va='center', ha='left', backgroundcolor='w', fontweight='bold')
    x_stream_latency_99 = np.interp(y_line_99, ccdf_stream_latency[::-1], stream_latency_sorted[::-1])
    x_prediction_latency_99 = np.interp(y_line_99, ccdf_prediction_latency[::-1], prediction_latency_sorted[::-1])
    x_total_latency_99 = np.interp(y_line_99, ccdf_total_latency[::-1], total_latency_sorted[::-1])
    plt.annotate(f'{x_stream_latency_99:.2f}', xy=(x_stream_latency_99, y_line_99), xycoords='data',
                 xytext=(-30, -30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_prediction_latency_99:.2f}', xy=(x_prediction_latency_99, y_line_99), xycoords='data',
                 xytext=(30, 30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_total_latency_99:.2f}', xy=(x_total_latency_99, y_line_99), xycoords='data',
                 xytext=(30, 30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))

    axs[1].set_title('CCDF of Latencies', fontweight='bold')
    axs[1].set_xlabel('Latency (ms)', fontweight='bold')
    axs[1].set_ylabel('CCDF', fontweight='bold')
    axs[1].grid(True)
    axs[1].legend()

    # Save the entire figure with both subplots as a PDF
    plt.savefig('Latency_Plots.pdf')

def eval_performance(pred_file):
    pred_data = read_csv(pred_file)
    y_true = [int(row[4]) for row in pred_data[1:]]
    y_pred = [int(row[3]) for row in pred_data[1:]]

    evaluate_classification_metrics(y_true, y_pred)

    latencies = calculate_latencies(pred_data)
    evaluate_latencies(latencies)
    plot_latencies(latencies)

if __name__ == "__main__":
    eval_performance("labels.csv")
