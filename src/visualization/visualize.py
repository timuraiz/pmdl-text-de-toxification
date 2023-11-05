import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src import ROOT_DIR

output_directory = f'{ROOT_DIR}/reports/'


def plot_toxicity_distributions(df):
    # Distribution of reference toxicity levels
    sns.histplot(df['ref_tox'], kde=True, stat="density", linewidth=0)
    plt.title('Distribution of Reference Text Toxicity Levels')
    plt.xlabel('Toxicity Level')
    plt.ylabel('Density')
    plt.savefig(output_directory + 'Reference_Text_Toxicity_Levels.png')
    plt.close()

    # Distribution of translated toxicity levels
    sns.histplot(df['trn_tox'], kde=True, stat="density", linewidth=0, color='orange')
    plt.title('Distribution of Translated Text Toxicity Levels')
    plt.xlabel('Toxicity Level')
    plt.ylabel('Density')
    plt.savefig(output_directory + 'Translated_Text_Toxicity_Levels.png')
    plt.close()


def plot_toxicity_comparison_scatter(df):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='ref_tox', y='trn_tox', data=df)
    plt.title('Toxicity Level: Reference vs. Translation')
    plt.xlabel('Reference Toxicity Level')
    plt.ylabel('Translated Toxicity Level')
    plt.axline((0, 0), (1, 1), linewidth=2, color='r', linestyle='--')  # y=x line
    plt.savefig(output_directory + 'Toxicity_Comparison_Scatter.png')
    plt.close()


def plot_text_length_distributions(df):
    df['ref_length'] = df['reference'].apply(len)
    df['trn_length'] = df['translation'].apply(len)

    sns.histplot(df['ref_length'], kde=False, color='blue')
    plt.title('Reference Text Length Distribution')
    plt.xlabel('Length of Text')
    plt.ylabel('Frequency')
    plt.savefig(output_directory + 'Reference_Text_Length_Distributions.png')
    plt.close()

    sns.histplot(df['trn_length'], kde=False, color='green')
    plt.title('Translated Text Length Distribution')
    plt.xlabel('Length of Text')
    plt.ylabel('Frequency')
    plt.savefig(output_directory + 'Translated_Text_Length_Distributions.png')
    plt.close()


def plot_cosine_similarity_distribution(df):
    sns.histplot(df['similarity'], kde=True)
    plt.title('Cosine Similarity Distribution')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Density')
    plt.savefig(output_directory + 'Cosine_Similarity_Distribution.png')
    plt.close()


def main(path=f'{ROOT_DIR}/data/filtered.tsv'):
    df = pd.read_csv(path, sep='\t', header=0)

    # Step: 1.
    plot_toxicity_distributions(df)

    # Step: 2.
    plot_toxicity_comparison_scatter(df)

    # Step: 3.
    plot_text_length_distributions(df)

    # Step: 4.
    plot_cosine_similarity_distribution(df)


if __name__ == '__main__':
    main()
