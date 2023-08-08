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
    
    # Cumulative Distribution Function (CDF)
    cdf_stream_latency = np.arange(len(stream_latency)) / float(len(stream_latency))
    cdf_prediction_latency = np.arange(len(prediction_latency)) / float(len(prediction_latency))
    cdf_total_latency = np.arange(len(total_latency)) / float(len(total_latency))

    # Cumulative Distribution Function (CDF)
    plt.figure(figsize=(10, 7))
    plt.plot(stream_latency_sorted, cdf_stream_latency, label='Stream Latency', linewidth=2.0)
    plt.plot(prediction_latency_sorted, cdf_prediction_latency, label='Prediction Latency', linewidth=2.0)
    plt.plot(total_latency_sorted, cdf_total_latency, label='Total Latency', linewidth=2.0)
    plt.xscale('log')  # Set the x-axis to a logarithmic scale

    # Add a vertical line for the 50th percentile
    plt.axhline(y=0.5, color='r', linestyle='--', linewidth=2.0)
    plt.text(0.9, 0.5, '50th Percentile', color='r', va='center', ha='left', backgroundcolor='w', fontweight='bold')

    # Get the x values where the line crosses the cdf plots for 50th percentile
    x_stream_latency_50 = np.interp(0.5, cdf_stream_latency, stream_latency_sorted)
    x_prediction_latency_50 = np.interp(0.5, cdf_prediction_latency, prediction_latency_sorted)
    x_total_latency_50 = np.interp(0.5, cdf_total_latency, total_latency_sorted)

    # Annotate the intersection points for 50th percentile
    plt.annotate(f'{x_stream_latency_50:.2f}', xy=(x_stream_latency_50, 0.5), xycoords='data',
                 xytext=(-30, -30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_prediction_latency_50:.2f}', xy=(x_prediction_latency_50, 0.5), xycoords='data',
                 xytext=(30, 30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_total_latency_50:.2f}', xy=(x_total_latency_50, 0.5), xycoords='data',
                 xytext=(30, 30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))

    plt.title('CDF of Latencies', fontweight='bold')
    plt.xlabel('Latency (ms)', fontweight='bold')
    plt.ylabel('CDF', fontweight='bold')
    plt.grid(True)
    plt.legend()
    plt.savefig('Latency_CDF.pdf')

    ccdf_stream_latency = 1 - np.arange(len(stream_latency)) / len(stream_latency)
    ccdf_prediction_latency = 1 - np.arange(len(prediction_latency)) / len(prediction_latency)
    ccdf_total_latency = 1 - np.arange(len(total_latency)) / len(total_latency)

    # Plot the CCDF
    plt.figure(figsize=(10, 7))
    plt.loglog(stream_latency_sorted, ccdf_stream_latency, label='Stream Latency', linewidth=2.0)
    plt.loglog(prediction_latency_sorted, ccdf_prediction_latency, label='Prediction Latency', linewidth=2.0)
    plt.loglog(total_latency_sorted, ccdf_total_latency, label='Total Latency', linewidth=2.0)

    # 99th Percentile
    y_line_99 = 10 ** -2
    plt.axhline(y=y_line_99, color='r', linestyle='--', linewidth=2.0)
    plt.text(0.9, y_line_99, '99th Percentile', color='r', va='center', ha='left', backgroundcolor='w', fontweight='bold')
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

    # 99.99th Percentile
    y_line_9999 = 10 ** -4
    plt.axhline(y=y_line_9999, color='r', linestyle='--', linewidth=2.0)
    plt.text(0.9, y_line_9999, '99.99th Percentile', color='r', va='center', ha='right', backgroundcolor='w', fontweight='bold')
    x_stream_latency_9999 = np.interp(y_line_9999, ccdf_stream_latency[::-1], stream_latency_sorted[::-1])
    x_prediction_latency_9999 = np.interp(y_line_9999, ccdf_prediction_latency[::-1], prediction_latency_sorted[::-1])
    x_total_latency_9999 = np.interp(y_line_9999, ccdf_total_latency[::-1], total_latency_sorted[::-1])
    plt.annotate(f'{x_stream_latency_9999:.2f}', xy=(x_stream_latency_9999, y_line_9999), xycoords='data',
                 xytext=(-30, -30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_prediction_latency_9999:.2f}', xy=(x_prediction_latency_9999, y_line_9999), xycoords='data',
                 xytext=(30, 30), textcoords='offset points', fontweight='bold',
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'{x_total_latency_9999:.2f}', xy=(x_total_latency_9999, y_line_9999), xycoords='data',
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
