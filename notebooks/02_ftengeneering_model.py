#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from scipy.optimize import differential_evolution
import joblib


# In[2]:


train = pd.read_csv('../data/train_renomeado.csv')
test = pd.read_csv('../data/test_renomeado.csv')

train = train.rename(columns={'ENTRADAS (v3)': 'Id'})
test = test.rename(columns={'ENTRADAS (v3)': 'Id'})


# In[3]:


target_cols = train.columns[-11:].tolist()
print(f"Targets identificados: {len(target_cols)} colunas")


# In[4]:


def create_features(df):
    df_new = df.copy()

    # 1. Tratamento de outliers e transformações
    skewed_cols = ['Emissões pesticidas', 'Urea', 'Macronutrientes']
    for col in skewed_cols:
        if col in df_new.columns:
            df_new[col] = np.log1p(np.abs(df_new[col]))

    # 2. Flags binárias para features com excesso de zeros
    zero_flag_cols = ['Energia da biomassa', 'Energia elétrica (kwh)', 
                      'Esterco animal (kg)', 'Esterco verde (kg)', 
                      'Micronutrientes', 'Transformation total, to and from']

    for col in zero_flag_cols:
        if col in df_new.columns:
            df_new[f'flag_{col}'] = (df_new[col] > 0).astype(int)

    # 3. Combinação de features correlacionadas
    if 'Calcário e gesso' in df_new.columns and 'Ocuppation, total' in df_new.columns:
        df_new['intensidade_calcario'] = df_new['Calcário e gesso'] / (df_new['Ocuppation, total'] + 1e-6)

    if 'Transformation total, to and from' in df_new.columns and 'Energia da biomassa' in df_new.columns:
        df_new['eficiencia_energetica'] = df_new['Transformation total, to and from'] / (df_new['Energia da biomassa'] + 1e-6)

    # 4. PCA para features de pesticidas
    pesticidas_cols = ['Fungicida, herbicida e pesticida', 'Emissões pesticidas']
    if all(col in df_new.columns for col in pesticidas_cols):
        pca = PCA(n_components=1)
        pesticidas_pca = pca.fit_transform(df_new[pesticidas_cols])
        df_new['pesticidas_pca'] = pesticidas_pca

    # 5. Criação de índices compostos
    nutrientes_cols = ['Macronutrientes', 'Micronutrientes']
    if all(col in df_new.columns for col in nutrientes_cols):
        df_new['nutrientes_total'] = df_new['Macronutrientes'] + df_new['Micronutrientes']

    if 'Urea' in df_new.columns and 'Ammonia e afins' in df_new.columns:
        df_new['impacto_fertilizantes'] = df_new['Urea'] + 0.7 * df_new['Ammonia e afins']

    # 6. Interações com a cultura (Seed)
    if 'Seed' in df_new.columns:
        df_new['biomassa_por_seed'] = df_new.groupby('Seed')['Energia da biomassa'].transform('mean')
        df_new['calcario_por_seed'] = df_new.groupby('Seed')['Calcário e gesso'].transform('mean')

    # 7. Transformações polinomiais para top features
    top_features = [
        'Transformation total, to and from',
        'Energia da biomassa',
        'Micronutrientes',
        'Esterco animal (kg)',
        'Fungicida, herbicida e pesticida'
    ]

    for col in top_features:
        if col in df_new.columns:
            df_new[f'{col}_sq'] = df_new[col] ** 2
            df_new[f'{col}_sqrt'] = np.sqrt(np.abs(df_new[col]))

    return df_new


# In[5]:


train_featured = create_features(train)
test_featured = create_features(test)


# In[6]:


def encode_categorical(train_df, test_df):
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    if 'Seed' in train_encoded.columns:
        # Target Encoding
        target_encoder = {}
        for target in target_cols:
            for seed in train_encoded['Seed'].unique():
                mask = (train_encoded['Seed'] == seed)
                target_encoder[(seed, target)] = train_encoded.loc[mask, target].mean()

        # Aplicar encoding
        for col in target_cols:
            train_encoded[f'Seed_encoded_{col}'] = train_encoded['Seed'].apply(
                lambda x: target_encoder.get((x, col), 0)
            )
            test_encoded[f'Seed_encoded_{col}'] = test_encoded['Seed'].apply(
                lambda x: target_encoder.get((x, col), np.nan)
            )

            # Preencher valores faltantes
            global_mean = train_encoded[col].mean()
            test_encoded[f'Seed_encoded_{col}'] = test_encoded[f'Seed_encoded_{col}'].fillna(global_mean)

        # Remover coluna original
        train_encoded = train_encoded.drop(columns=['Seed'])
        test_encoded = test_encoded.drop(columns=['Seed'])

    return train_encoded, test_encoded


# In[7]:


train_encoded, test_encoded = encode_categorical(train_featured, test_featured)


# In[8]:


train_ids = train_encoded['Id']
test_ids = test_encoded['Id']

# Separar features e targets
X_train = train_encoded.drop(columns=['Id'] + target_cols)
y_train = train_encoded[target_cols]
X_test = test_encoded.drop(columns=['Id'])

# Garantir mesmas colunas no treino e teste
for col in X_train.columns:
    if col not in X_test.columns:
        X_test[col] = 0
X_test = X_test[X_train.columns]

# Normalização
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[9]:


models = {
    "RandomForest": MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            min_samples_split=5,
            max_features=0.8,
            random_state=42,
            n_jobs=-1
        )
    ),
    "XGBoost": MultiOutputRegressor(
        XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
    ),
    "SVR": MultiOutputRegressor(
        SVR(C=3.0, epsilon=0.1, kernel='rbf')
    )
}

# Treinar modelos e fazer previsões OOF (Out-of-Fold)
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
oof_predictions = {name: np.zeros_like(y_train.values) for name in models}
test_predictions = {name: np.zeros((X_test.shape[0], len(target_cols))) for name in models}

for name, model in models.items():
    print(f"\nTreinando {name}...")
    fold_test_preds = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
        print(f"  Fold {fold+1}/{n_splits}")
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train.values[train_idx], y_train.values[val_idx]

        model.fit(X_tr, y_tr)

        # Prever no validation set
        val_preds = model.predict(X_val)
        oof_predictions[name][val_idx] = val_preds

        # Calcular MSE do fold
        fold_mse = mean_squared_error(y_val, val_preds)
        print(f"  Fold MSE: {fold_mse:.5f}")

        # Prever no teste
        fold_test_preds.append(model.predict(X_test_scaled))

    # Média das previsões do teste entre os folds
    test_predictions[name] = np.mean(fold_test_preds, axis=0)

    # Calcular MSE completo
    full_mse = mean_squared_error(y_train, oof_predictions[name])
    print(f"{name} OOF MSE: {full_mse:.5f}")


# In[10]:


all_oof_preds = [oof_predictions[name] for name in models]
all_test_preds = [test_predictions[name] for name in models]

# Função para calcular MSE com pesos
def ensemble_mse(weights, preds_list, y_true):
    weighted_preds = np.zeros_like(y_true)
    for i, w in enumerate(weights):
        weighted_preds += w * preds_list[i]
    return mean_squared_error(y_true, weighted_preds)

# Otimizar pesos para cada target separadamente
optimal_weights = np.zeros((len(target_cols), len(models)))

for target_idx in range(len(target_cols)):
    print(f"Otimizando target {target_idx+1}/{len(target_cols)}")
    target_oof_preds = [preds[:, target_idx] for preds in all_oof_preds]

    # Definir função objetivo para este target
    def target_objective(weights):
        return ensemble_mse(weights, target_oof_preds, y_train.values[:, target_idx])

    # Otimizar pesos
    bounds = [(0, 1)] * len(models)
    result = differential_evolution(
        target_objective,
        bounds,
        strategy='best1bin',
        maxiter=100,
        popsize=15,
        tol=1e-4,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=42
    )

    # Normalizar pesos para soma=1
    weights = result.x
    weights /= weights.sum()
    optimal_weights[target_idx] = weights
    print(f"  Pesos otimizados: {weights}")


# In[11]:


print("\nCriando previsões finais...")
final_predictions = np.zeros((X_test_scaled.shape[0], len(target_cols)))

for target_idx in range(len(target_cols)):
    for model_idx, model_name in enumerate(models):
        weight = optimal_weights[target_idx, model_idx]
        final_predictions[:, target_idx] += weight * all_test_preds[model_idx][:, target_idx]


# In[13]:


print("Preparando arquivo de submissão...")
submission = pd.DataFrame(final_predictions, columns=[f"target{i+1}" for i in range(11)])
submission.insert(0, 'Id', test_ids.values)

# Garantir que não há valores negativos
submission.iloc[:, 1:] = submission.iloc[:, 1:].clip(lower=0)

# Salvar arquivo
submission_file = '../submissions/ensemble_predictions.csv'
submission.to_csv(submission_file, index=False)
print(f"Submissão salva em: {submission_file}")

# 10. ===== Salvar modelos =====
print("Salvando modelos para uso futuro...")
for name, model in models.items():
    joblib.dump(model, f'../models/{name}_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
np.save('../models/optimal_weights.npy', optimal_weights)

print("Processo completo!")

