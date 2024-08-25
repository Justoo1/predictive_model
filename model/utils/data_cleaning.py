import pandas as pd

def cleaning():
    # Loading dataset
    df = pd.read_csv(r'D:\predictive_model\model\utils\world_energy_data.csv')

    # Remove the first row
    df = df.iloc[1:].reset_index(drop=True)

    # Handle missing values
    df = df.ffill()

    # Convert Co2 Emissions to numeric, forcing errors to NaN and then forward fill
    df['Co2 Emissions'] = pd.to_numeric(df['Co2 Emissions'], errors='coerce')
    df['Co2 Emissions'] = df['Co2 Emissions'].ffill()

    # Normalize numerical data
    df['Co2 Emissions'] = (df['Co2 Emissions'] - df['Co2 Emissions'].mean()) / df['Co2 Emissions'].std()

    # Selecting relevant features
    features = df[['year', 'Coal Data', 'Natural Gas', 'Electricity']]
    target = df['Co2 Emissions']

    return features, target

if __name__ == "__main__":
    features, target = cleaning()
    print("Features:")
    print(features.head())
    print("Target:")
    print(target.head())
