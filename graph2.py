import pandas as pd
import matplotlib.pyplot as plt

# read the CSV file into a pandas DataFrame
df = pd.read_csv("G:/My Drive/SR/statistics/srf_4_train_results.csv")

cols = ["Epoch", "Loss_D", "Loss_G", "PSNR", "SSIM"]
data = df[cols]

# set the x axis to be the "Epoch" column
x = data["Epoch"]

# Create a larger figure for the plot
plt.figure(figsize=(12, 8))  # Adjust width and height as needed

# plot each column as a separate line graph
for col in cols[1:]:
    y = data[col]
    plt.plot(x, y, label=col)


# Adding labels and title for the combined plot
plt.xlabel("Epoch")
plt.ylabel("Values")
plt.title("Model Metrics vs Epoch")
plt.legend()

# Adjusting y-axis ticks to show an interval of 1
y_min, y_max = plt.ylim()  # Get the current y-axis limits
plt.yticks(range(int(y_min), int(y_max) + 1, 1))  # Set ticks from min to max with a step size of 1

# display the graphs
plt.show()
