import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .parsing import dict_to_prob_vector, convert_to_array


class EnsembleSingleArchitecture:
    """
    Encapsulates the Stage 1 (Single Architecture) ensemble logic extracted
    from the original notebook 'G.SubtVision etapa 1 - ensemble multiarquitetura.ipynb'.
    """

    def __init__(self, base_path: str, classes: list):
        self.base_path = base_path
        self.classes = classes

    @staticmethod
    def load_and_rename(file_path: str, prefix: str, fold: int) -> pd.DataFrame:
        # Carregar o arquivo CSV
        df = pd.read_csv(file_path)

        # Definir o sufixo com base no prefixo e no fold
        suffix = f'_{prefix}{fold}'

        # Renomear as colunas de predição
        df.rename(columns={
            'predicted_label': f'predicted_label{suffix}',
            'predicted_probability': f'predicted_probability{suffix}',
            'probability_vector': f'probability_vector{suffix}',
            'probability_std_dev': f'probability_std_dev{suffix}'
        }, inplace=True)

        return df

    def generate_merged_table(self, folders: list, model_names: list) -> pd.DataFrame:
        # Etapa para carregar os arquivos e renomear as colunas
        all_data = []

        for model_name, folder in zip(model_names, folders):
            for fold in range(10):  # Para cada fold de 0 a 9
                # Caminho do arquivo (modifique o caminho conforme sua estrutura de pastas)
                file_path = f'{self.base_path}/{model_name}_fold{fold}_results.csv'  # Ajuste o caminho real
                if os.path.exists(file_path):
                    # Carregar e renomear as colunas
                    df_renamed = self.load_and_rename(file_path, folder, fold)
                    all_data.append(df_renamed)
                else:
                    print(f'Arquivo não encontrado: {file_path}')

        # Mesclar os arquivos pela chave 'image_path' e 'true_label'
        merged_df = all_data[0]
        for df in all_data[1:]:
            merged_df = merged_df.merge(df, on=['image_path', 'true_label', 'true_label_one_hot'], how='outer')

        return merged_df

    @staticmethod
    def hard_voting(df: pd.DataFrame, model_prefix: str, network: str, print_flag: bool = True) -> pd.DataFrame:
        # Extrair as colunas de predicted_label
        cols_hard = [f'predicted_label_{model_prefix}{i}' for i in range(10)]

        # Função para fazer a votação majoritária
        def get_majority_vote(row):
            votes = [row[col] for col in cols_hard]
            return max(set(votes), key=votes.count)

        # Aplicar a função para calcular a predição do Hard Voting
        df['hard_voting_label'] = df.apply(get_majority_vote, axis=1)

        if print_flag:
            print(f"===== Hard Voting 10 folds nível de TILE - {network}=====")
            print(classification_report(df['true_label'], df['hard_voting_label']))

        return df

    def soft_voting(self, df: pd.DataFrame, model_prefix: str, network: str, classes: list, print_flag: bool = True) -> pd.DataFrame:
        # Extrair as colunas de probability_vector
        cols_soft = [f'probability_vector_{model_prefix}{i}' for i in range(10)]

        # Função para calcular a média das probabilidades
        def get_soft_vote(row):
            prob_vectors = [dict_to_prob_vector(ast.literal_eval(row[col]), classes) for col in cols_soft]
            prob_vectors = np.array(prob_vectors)  # Converter para um array numpy
            avg_probs = prob_vectors.mean(axis=0)  # Média das probabilidades para cada classe
            return avg_probs

        # Aplicar a função para calcular a probabilidade média
        df['soft_voting_probs'] = df.apply(get_soft_vote, axis=1)

        # Escolher a classe com maior probabilidade média
        df['soft_voting_label'] = df['soft_voting_probs'].apply(lambda x: classes[np.argmax(x)])

        if print_flag:
            print(f"\n===== Soft Voting 10 folds nível de TILE - {network} =====")
            print(classification_report(df['true_label'], df['soft_voting_label']))

        return df

    @staticmethod
    def aggregate_to_image_level(df: pd.DataFrame, classes: list) -> pd.DataFrame:
        # extrair image_id do image_path
        df['image_id'] = df['image_path'].apply(lambda x: x.split('/')[-3].split('_files')[0])

        # função auxiliar para calcular o hard_voting_label por imagem
        def majority_vote(labels):
            return labels.value_counts().idxmax()

        # média dos vetores por imagem
        def mean_vec(series: pd.Series):
            return np.mean(np.vstack(series.values), axis=0)

        df['soft_voting_probs'] = df['soft_voting_probs'].apply(lambda p: convert_to_array(p, num_classes=len(classes)))

        # agregação por image_id
        agg_df = df.groupby('image_id').agg({
            'hard_voting_label': majority_vote,
            'soft_voting_probs': mean_vec,
            'true_label': 'first'
        }).reset_index()

        # calcular o soft_voting_label a partir do vetor médio
        agg_df['soft_voting_label'] = agg_df['soft_voting_probs'].apply(
            lambda probs: classes[np.argmax(probs)]
        )

        return agg_df[['image_id', 'true_label', 'hard_voting_label', 'soft_voting_probs', 'soft_voting_label']]

    @staticmethod
    def evaluate_ensembles_image(agg_df: pd.DataFrame, network: str) -> None:
        print(f"===== Hard Voting 10 folds nível de IMAGEM - {network} =====")
        print(classification_report(agg_df['true_label'], agg_df['hard_voting_label']))

        print(f"\n===== Soft Voting 10 folds nível de IMAGEM - {network} =====")
        print(classification_report(agg_df['true_label'], agg_df['soft_voting_label']))

    @staticmethod
    def aggregate_to_patient_level(df: pd.DataFrame, classes: list) -> pd.DataFrame:
        df['patient_id'] = df['image_path'].apply(lambda x: x.split('/')[-3].split('_files')[0][:12])

        def majority_vote(labels):
            return labels.value_counts().idxmax()

        def mean_vec(series: pd.Series):
            return np.mean(np.vstack(series.values), axis=0)

        df['soft_voting_probs'] = df['soft_voting_probs'].apply(lambda p: convert_to_array(p, num_classes=len(classes)))

        agg_df = df.groupby('patient_id').agg({
            'hard_voting_label': majority_vote,
            'soft_voting_probs': mean_vec,
            'true_label': 'first'
        }).reset_index()

        agg_df['soft_voting_label'] = agg_df['soft_voting_probs'].apply(
            lambda probs: classes[np.argmax(probs)]
        )

        return agg_df[['patient_id', 'true_label', 'hard_voting_label', 'soft_voting_probs', 'soft_voting_label']]

    @staticmethod
    def evaluate_ensembles_patient(agg_df: pd.DataFrame, network: str) -> None:
        print(f"===== Hard Voting 10 folds nível de PATIENT - {network} =====")
        print(classification_report(agg_df['true_label'], agg_df['hard_voting_label']))

        print(f"\n===== Soft Voting 10 folds nível de PATIENT - {network} =====")
        print(classification_report(agg_df['true_label'], agg_df['soft_voting_label']))

    @staticmethod
    def plot_sodt_roc_curve(df: pd.DataFrame, level: str, fig_name: str, file_name: str, classes: list, save_output_dir: str = 'outputs/auc_roc_sa_plots') -> None:
        # Binarizar os rótulos verdadeiros (true_label) para o cálculo da curva ROC
        y_true = label_binarize(df['true_label'], classes=classes)

        # Soft Voting: Convertendo as probabilidades para matriz
        y_pred_soft = np.vstack(df['soft_voting_probs'].apply(lambda p: convert_to_array(p, num_classes=len(classes))).values)

        # Preparando o gráfico
        plt.figure(figsize=(10, 8))

        # Loop sobre cada classe para gerar a curva ROC
        roc_auc = {}
        for i, cls in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_soft[:, i])
            roc_auc[cls] = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Classe {cls} (AUC = {roc_auc[cls]:.2f})')

        # Macro-average ROC curve e AUC
        fpr_macro, tpr_macro, _ = roc_curve(y_true.ravel(), y_pred_soft.ravel())
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        plt.plot(fpr_macro, tpr_macro, color='purple', linestyle='--', lw=2, label=f'Macro-average (AUC = {roc_auc_macro:.2f})')

        # Plotar linha de aleatoriedade
        plt.plot([0, 1], [0, 1], 'k--', lw=2)

        # Definir títulos e rótulos
        plt.xlabel('FPR (False Positive Rate)')
        plt.ylabel('TPR (True Positive Rate)')
        plt.title(fig_name)
        plt.legend(loc='lower right')
        plt.grid(True)

        plt.tight_layout()
        if save_output_dir:
            if not os.path.exists(save_output_dir):
                os.makedirs(save_output_dir, exist_ok=True)
            plt.savefig(os.path.join(save_output_dir, file_name))
        plt.show()