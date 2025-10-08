import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.preprocessing import label_binarize

from .parsing import dict_to_prob_vector, convert_to_array


class EnsembleMultiArchitecture:
    """
    Encapsulates the Stage 2 (Multi-Architecture) ensemble logic extracted
    from the original notebook 'G.SubtVision etapa 2 - ensemble multiarquitetura.ipynb'.
    """

    def __init__(self, base_path: str, classes: list):
        self.base_path = base_path
        self.classes = classes

    @staticmethod
    def load_and_rename(file_path: str, prefix: str, fold: int) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        suffix = f'_{prefix}{fold}'
        df.rename(columns={
            'predicted_label': f'predicted_label{suffix}',
            'predicted_probability': f'predicted_probability{suffix}',
            'probability_vector': f'probability_vector{suffix}',
            'probability_std_dev': f'probability_std_dev{suffix}'
        }, inplace=True)
        return df

    def merge_three_architectures(self, folders: list, model_names: list) -> pd.DataFrame:
        # Etapa para carregar os arquivos e renomear as colunas
        all_data = []

        for model_name, folder in zip(model_names, folders):
            for fold in range(10):
                file_path = f'{self.base_path}/{model_name}_fold{fold}_results.csv'
                if os.path.exists(file_path):
                    df_renamed = self.load_and_rename(file_path, folder, fold)
                    all_data.append(df_renamed)
                else:
                    print(f'Arquivo não encontrado: {file_path}')

        print(f"Nomes das colunas da primeira tabela mesclada: \n {list(all_data[0].columns)}")

        merged_df = all_data[0]
        for df in all_data[1:]:
            merged_df = merged_df.merge(df, on=['image_path', 'true_label', 'true_label_one_hot'], how='outer')

        return merged_df

    @staticmethod
    def hard_voting(df: pd.DataFrame, print_flag: bool = True) -> pd.DataFrame:
        cols_hard = [f'predicted_label_m{i}' for i in range(10)] + \
                    [f'predicted_label_s{i}' for i in range(10)] + \
                    [f'predicted_label_g{i}' for i in range(10)]

        def get_majority_vote(row):
            votes = [row[col] for col in cols_hard]
            return max(set(votes), key=votes.count)

        df['hard_voting_label'] = df.apply(get_majority_vote, axis=1)

        if print_flag:
            print("===== HARD VOTING MÉTRICAS =====")
            print(classification_report(df['true_label'], df['hard_voting_label']))

        return df

    def soft_voting(self, df: pd.DataFrame, print_flag: bool = True) -> pd.DataFrame:
        cols_soft = [f'probability_vector_m{i}' for i in range(10)] + \
                    [f'probability_vector_s{i}' for i in range(10)] + \
                    [f'probability_vector_g{i}' for i in range(10)]

        def get_soft_vote(row):
            prob_vectors = [dict_to_prob_vector(ast.literal_eval(row[col]), self.classes) for col in cols_soft]
            prob_vectors = np.array(prob_vectors)
            avg_probs = prob_vectors.mean(axis=0)
            return avg_probs

        df['soft_voting_probs'] = df.apply(get_soft_vote, axis=1)
        df['soft_voting_label'] = df['soft_voting_probs'].apply(lambda x: self.classes[np.argmax(x)])

        if print_flag:
            print("\n===== SOFT VOTING MÉTRICAS =====")
            print(classification_report(df['true_label'], df['soft_voting_label']))

        return df

    def plot_sodt_roc_curve(self, df: pd.DataFrame, level: str, fig_name: str, file_name: str, save_output_dir: str = 'outputs/auc_roc_sa_plots') -> None:
        y_true = label_binarize(df['true_label'], classes=self.classes)
        y_pred_soft = np.vstack(df['soft_voting_probs'].apply(lambda p: convert_to_array(p, num_classes=len(self.classes))).values)

        plt.figure(figsize=(10, 8))
        roc_auc = {}
        for i, cls in enumerate(self.classes):
            fpr, tpr, _ = roc_curve(y_true[:, i], y_pred_soft[:, i])
            roc_auc[cls] = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Classe {cls} (AUC = {roc_auc[cls]:.2f})')

        fpr_macro, tpr_macro, _ = roc_curve(y_true.ravel(), y_pred_soft.ravel())
        roc_auc_macro = auc(fpr_macro, tpr_macro)
        plt.plot(fpr_macro, tpr_macro, color='purple', linestyle='--', lw=2, label=f'Macro-average (AUC = {roc_auc_macro:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
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

    # =============================
    # Aggregations (Image / Patient)
    # =============================
    def aggregate_to_image_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega as predições por imagem:
        - hard_voting_label: votação majoritária dos tiles de uma mesma imagem.
        - soft_voting_probs: média dos vetores de probabilidade (soft) por imagem.
        - soft_voting_label: classe com maior probabilidade média.

        A extração de image_id replica a lógica dos notebooks originais:
        image_id = parte do caminho no índice [-3], removendo o sufixo '_files'.
        """

        # Derivar image_id do image_path (mesma regra usada nos notebooks originais)
        df['image_id'] = df['image_path'].apply(lambda x: x.split('/')[-3].split('_files')[0])

        # Funções auxiliares
        def majority_vote(labels: pd.Series) -> str:
            return labels.value_counts().idxmax()

        def mean_vec(series: pd.Series) -> np.ndarray:
            return np.mean(np.vstack(series.values), axis=0)

        # Garantir vetor numpy para cada linha (suporta string/list/ndarray)
        df['soft_voting_probs'] = df['soft_voting_probs'].apply(
            lambda p: convert_to_array(p, num_classes=len(self.classes))
        )

        # Agregação por imagem
        agg_df = df.groupby('image_id').agg({
            'hard_voting_label': majority_vote,
            'soft_voting_probs': mean_vec,
            'true_label': 'first'
        }).reset_index()

        # Classe final via soft voting médio
        agg_df['soft_voting_label'] = agg_df['soft_voting_probs'].apply(
            lambda probs: self.classes[np.argmax(probs)]
        )

        return agg_df[['image_id', 'true_label', 'hard_voting_label', 'soft_voting_probs', 'soft_voting_label']]

    def evaluate_ensembles_image(self, agg_df: pd.DataFrame) -> None:
        """Imprime classification_report para hard e soft em nível de IMAGEM."""
        print("===== Hard Voting 10 folds nível de IMAGEM - Multi-Arquitetura =====")
        print(classification_report(agg_df['true_label'], agg_df['hard_voting_label']))

        print("\n===== Soft Voting 10 folds nível de IMAGEM - Multi-Arquitetura =====")
        print(classification_report(agg_df['true_label'], agg_df['soft_voting_label']))

    def aggregate_to_patient_level(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega as predições por paciente:
        - patient_id: derivado do image_id (primeiros 12 caracteres)
        - hard_voting_label: votação majoritária por paciente.
        - soft_voting_probs: média das probabilidades por paciente.
        - soft_voting_label: classe com maior probabilidade média.
        """

        # Derivar patient_id com a mesma regra dos notebooks
        df['patient_id'] = df['image_path'].apply(lambda x: x.split('/')[-3].split('_files')[0][:12])

        def majority_vote(labels: pd.Series) -> str:
            return labels.value_counts().idxmax()

        def mean_vec(series: pd.Series) -> np.ndarray:
            return np.mean(np.vstack(series.values), axis=0)

        df['soft_voting_probs'] = df['soft_voting_probs'].apply(
            lambda p: convert_to_array(p, num_classes=len(self.classes))
        )

        agg_df = df.groupby('patient_id').agg({
            'hard_voting_label': majority_vote,
            'soft_voting_probs': mean_vec,
            'true_label': 'first'
        }).reset_index()

        agg_df['soft_voting_label'] = agg_df['soft_voting_probs'].apply(
            lambda probs: self.classes[np.argmax(probs)]
        )

        return agg_df[['patient_id', 'true_label', 'hard_voting_label', 'soft_voting_probs', 'soft_voting_label']]

    def evaluate_ensembles_patient(self, agg_df: pd.DataFrame) -> None:
        """Imprime classification_report para hard e soft em nível de PACIENTE."""
        print("===== Hard Voting 10 folds nível de PATIENT - Multi-Arquitetura =====")
        print(classification_report(agg_df['true_label'], agg_df['hard_voting_label']))

        print("\n===== Soft Voting 10 folds nível de PATIENT - Multi-Arquitetura =====")
        print(classification_report(agg_df['true_label'], agg_df['soft_voting_label']))