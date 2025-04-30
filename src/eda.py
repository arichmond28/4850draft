# Importing All Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_csv('../data/ai_insights.csv')

# Function To Create A Bar Chart Displaying Growth Frequencies
def growth_plot():
  # Extracting The Last Column From Data
  growth = data.iloc[:, -1]

  # Creating A BarChart Displaying Growth Frequencies
  plt.figure(figsize=(10, 6))
  sns.countplot(x=growth, palette='viridis')
  plt.title('Growth Frequencies')
  plt.xlabel('Growth Projection')
  plt.ylabel('Frequency')

  # Saving The Plot
  plt.savefig('../results/eda_plots/growth_frequencies.png')

# Function To Create A Box Plot Displaying Salary Distribution
def salary_plot():
  # Extracting The Salary Column From Data
  salary = data['Salary_USD']

  # Creating A Box Plot Displaying Salary Distribution
  plt.figure(figsize=(10, 6))
  sns.boxplot(y=salary, palette='viridis')
  plt.title('Salary Distribution')
  plt.ylabel('Salary')

  # Saving The Plot
  plt.savefig('../results/eda_plots/salary_distribution.png')


growth_plot()
salary_plot()