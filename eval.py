import csv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from dateutil.parser import parse
import matplotlib as mpl

mpl.rcParams['font.weight'] = 'bold'  # Make all text bold
mpl.rcParams['font.size'] = 12  # Increase font size

def read_csv(file_name):
    with open(file_name, newline='') as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    return data

def eval_performance(pred_file):
    pred_data = read_csv(pred_file)

    y_true = [int(row[4]) for row in pred_data[1:]]  # Skip the header row
    y_pred = [int(row[3]) for row in pred_data[1:]]  # Skip the header row

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

    stream_latency = [(parse(row[6]) - parse(row[5])).total_seconds() * 1000 for row in pred_data[1:]]  # Skip the header row
    prediction_latency = [(parse(row[7]) - parse(row[6])).total_seconds() * 1000 for row in pred_data[1:]]  # Skip the header row
    total_latency = [(parse(row[7]) - parse(row[5])).total_seconds() * 1000 for row in pred_data[1:]]  # Skip the header row

    stream_latency_sorted = np.sort(stream_latency)
    prediction_latency_sorted = np.sort(prediction_latency)
    total_latency_sorted = np.sort(total_latency)

    ccdf_stream_latency = 1 - np.arange(len(stream_latency)) / len(stream_latency)
    ccdf_prediction_latency = 1 - np.arange(len(prediction_latency)) / len(prediction_latency)
    ccdf_total_latency = 1 - np.arange(len(total_latency)) / len(total_latency)

    # Plot the CCDF
    plt.figure(figsize=(10, 7))
    plt.loglog(stream_latency_sorted, ccdf_stream_latency, label='Stream Latency', linewidth=2.0)
    plt.loglog(prediction_latency_sorted, ccdf_prediction_latency, label='Prediction Latency', linewidth=2.0)
    plt.loglog(total_latency_sorted, ccdf_total_latency, label='Total Latency', linewidth=2.0)

    # Add vertical lines at specified percentiles
    percentiles = [(10 ** -2, '99th Percentile'), (10 ** -4, '99.99th Percentile'), (10 ** -0.3, '50th Percentile')]
    for y_line, percentile_label in percentiles:
        plt.axhline(y=y_line, color='r', linestyle='--', linewidth=2.0)
        plt.text(0.9, y_line, percentile_label, color='r', va='center', ha='right', backgroundcolor='w', fontweight='bold')

        # Get the x values where the line crosses the ccdf plots
        x_stream_latency = np.interp(y_line, ccdf_stream_latency[::-1], stream_latency_sorted[::-1])
        x_prediction_latency = np.interp(y_line, ccdf_prediction_latency[::-1], prediction_latency_sorted[::-1])
        x_total_latency = np.interp(y_line, ccdf_total_latency[::-1], total_latency_sorted[::-1])

        # Annotate the intersection points
        plt.annotate(f'{x_stream_latency:.2f}', xy=(x_stream_latency, y_line), xycoords='data',
                     xytext=(-30, -30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))
        plt.annotate(f'{x_prediction_latency:.2f}', xy=(x_prediction_latency, y_line), xycoords='data',
                     xytext=(30, 30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))
        plt.annotate(f'{x_total_latency:.2f}', xy=(x_total_latency, y_line), xycoords='data',
                     xytext=(30, 30), textcoords='offset points', fontweight='bold',
                     arrowprops=dict(arrowstyle="->"))

    plt.title('CCDF of Latencies', fontweight='bold')
    plt.xlabel('Latency (ms)', fontweight='bold')
    plt.ylabel('CCDF', fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.savefig('Latency.pdf')

if __name__ == "__main__":
    eval_performance("labels.csv")
