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

# Function To Create A Bar Chart Displaying AI Adoption Frequencies
def ai_adoption_plot():
  # Extracting The AI Adoption Column From Data
  ai_adoption = data['AI_Adoption_Level']

  # Creating A BarChart Displaying AI Adoption Frequencies
  plt.figure(figsize=(10, 6))
  sns.countplot(x=ai_adoption, palette='viridis')
  plt.title('AI Adoption Frequencies')
  plt.xlabel('AI Adoption')
  plt.ylabel('Frequency')

  # Saving The Plot
  plt.savefig('../results/eda_plots/ai_adoption_frequencies.png')

# Function To Create A Cross Tabulation Displaying AI Adoption Frequencies By Growth
def ai_adoption_by_growth():
  # Creating A Cross Tabulation Displaying AI Adoption Frequencies By Growth
  crosstab = pd.crosstab(data['AI_Adoption_Level'], data['Job_Growth_Projection'])
  plt.figure(figsize=(10, 6))
  sns.heatmap(crosstab, annot=True, cmap='viridis')
  plt.title('AI Adoption Frequencies By Growth')
  plt.xlabel('Growth Projection')
  plt.ylabel('AI Adoption Level')

  # Saving The Plot
  plt.savefig('../results/eda_plots/ai_adoption_by_growth.png')

# Function To Create A Plot Displaying Salary Distribution By AI Adoption
def salary_by_ai_adoption():
  # Creating A Box Plot Displaying Salary Distribution By AI Adoption
  plt.figure(figsize=(10, 6))
  sns.boxplot(x=data['AI_Adoption_Level'], y=data['Salary_USD'], palette='viridis')
  plt.title('Salary Distribution By AI Adoption')
  plt.xlabel('AI Adoption Level')
  plt.ylabel('Salary')

  # Saving The Plot
  plt.savefig('../results/eda_plots/salary_by_ai_adoption.png')

# Function To Create A Plot Displaying AI Adoption By Automation Risk
def automating_risk_by_ai_adoption():
  # Creating A Count Plot Displaying Automating Risk By AI Adoption
  plt.figure(figsize=(10, 6))
  sns.countplot(x=data['AI_Adoption_Level'], hue=data['Automation_Risk'], palette='viridis')
  plt.title('Automating Risk By AI Adoption')
  plt.xlabel('AI Adoption Level')
  plt.ylabel('Automating Risk')

  # Saving The Plot
  plt.savefig('../results/eda_plots/automating_risk_by_ai_adoption.png')

# Function To Create A Plot Displaying Growth Projection By Remote Work
def growth_by_remote_work():
  # Creating A Count Plot Displaying Growth Projection By Remote Work
  plt.figure(figsize=(10, 6))
  sns.countplot(x=data['Remote_Friendly'], hue=data['Job_Growth_Projection'], palette='viridis')
  plt.title('Growth Projection By Remote Work')
  plt.xlabel('Remote Friendly')
  plt.ylabel('Growth Projection')

  # Saving The Plot
  plt.savefig('../results/eda_plots/growth_by_remote_work.png')

growth_plot()
salary_plot()
ai_adoption_plot()
ai_adoption_by_growth()
salary_by_ai_adoption()
automating_risk_by_ai_adoption()
growth_by_remote_work()