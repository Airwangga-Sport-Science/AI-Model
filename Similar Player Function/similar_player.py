import pickle
import numpy as np

def find_top_min_variance_rows(input_dict, dataframe, top_n=3, min_international_reputation=4):
    """
    Mencari baris dengan varians paling rendah antara posisi pemain dalam input dan posisi pemain dalam dataframe.
    
    Parameters:
    - input_dict (dict): Dict yang berisi posisi pemain input dengan nama posisi sebagai kunci dan nilai numerik sebagai nilai.
    - dataframe (pd.DataFrame): DataFrame yang berisi data pemain dengan kolom-kolom termasuk 'international_reputation' dan posisi pemain.
    - top_n (int): Jumlah baris teratas dengan varians paling rendah yang ingin diambil. Default: 3.
    - min_international_reputation (int): Nilai reputasi internasional minimum yang dibutuhkan untuk mempertimbangkan pemain. Default: 4.
    
    Returns:
    - pd.Series: Seri berisi nama-nama pemain dengan varians paling rendah untuk posisi yang diberikan.
                Jika tidak ada pemain yang memenuhi kriteria, kembalikan DataFrame kosong.
    """
    input_values = np.array(list(input_dict.values()))

    filtered_dataframe = dataframe[dataframe['international_reputation'] >= min_international_reputation]

    if filtered_dataframe.empty:
        return pd.DataFrame()
    
    positions = filtered_dataframe[['rwb', 'lwb', 'rb', 'lb', 'cb', 'cdm', 'lm', 'rm', 'lw', 'rw', 'cf', 'st']]

    # Minimalisasi varians antara input dan players_data
    variances = positions.apply(lambda row: np.var(row.values - input_values), axis=1)

    # Cari varians paling minimum
    top_indices = variances.nsmallest(top_n).index

    # Cari top N varians paling minimum
    top_rows = filtered_dataframe.loc[top_indices]

    return top_rows['long_name']

def infer(user_skills, players_data):

    """
    Memperkirakan kemampuan pemain berdasarkan model prediksi yang telah dilatih sebelumnya.

    Parameters:
    - user_skills (pd.Series): Pandas Series berisi atribut kemampuan pemain input.
    - players_data (pd.DataFrame): DataFrame berisi data pemain lengkap termasuk atribut kemampuan dan posisi.

    Returns:
    - dict: Kamus berisi prediksi kemampuan pemain untuk setiap posisi dan daftar pemain yang serupa.
    """
    # Daftar atribut untuk setiap model
    atribut_model_1 = [
      'movement_sprint_speed',
      'movement_acceleration',
      'mentality_positioning',
      'mentality_interceptions',
      'mentality_aggression',
      'attacking_finishing',
      'power_shot_power',
      'power_long_shots',
      'attacking_volleys',
      'mentality_penalties',
      'mentality_vision',
      'attacking_crossing',
      'skill_fk_accuracy',
      'attacking_short_passing',
      'skill_long_passing',
      'skill_curve',
      'movement_agility',
      'movement_balance',
      'movement_reactions',
      'skill_ball_control',
      'skill_dribbling',
      'mentality_composure',
      'attacking_heading_accuracy',
      'defending_marking_awareness',
      'defending_standing_tackle',
      'defending_sliding_tackle',
      'power_jumping',
      'power_stamina',
      'power_strength'
    ]
    
    atribut_model_2 = ['movement_sprint_speed',
      'mentality_positioning',
      'mentality_interceptions',
      'mentality_aggression',
      'attacking_finishing',
      'power_shot_power',
      'power_long_shots',
      'attacking_volleys',
      'mentality_penalties',
      'mentality_vision',
      'attacking_crossing',
      'skill_fk_accuracy',
      'attacking_short_passing',
      'skill_long_passing',
      'skill_curve',
      'movement_agility',
      'movement_balance',
      'movement_reactions',
      'skill_ball_control',
      'skill_dribbling',
      'mentality_composure',
      'attacking_heading_accuracy',
      'defending_marking_awareness',
      'defending_standing_tackle',
      'defending_sliding_tackle',
      'power_jumping',
      'power_stamina',
      'power_strength'
    ]
    
    atribut_model_3 = [
      'movement_sprint_speed',
      'movement_acceleration',
      'mentality_positioning',
      'mentality_interceptions',
      'mentality_aggression',
      'attacking_finishing',
      'power_shot_power',
      'power_long_shots',
      'attacking_volleys',
      'mentality_penalties',
      'mentality_vision',
      'attacking_crossing',
      'skill_fk_accuracy',
      'attacking_short_passing',
      'skill_long_passing',
      'skill_curve',
      'movement_agility',
      'movement_balance',
      'movement_reactions',
      'skill_ball_control',
      'skill_dribbling',
      'mentality_composure',
      'attacking_heading_accuracy',
      'defending_marking_awareness',
      'defending_standing_tackle',
      'defending_sliding_tackle',
      'power_jumping',
      'power_stamina'
    ]

    # Dict untuk menyimpan prediksi kemampuan pemain untuk setiap posisi
    skill_pred ={
        'rwb': 0,
        'lwb': 0,
        'rb': 0,
        'lb': 0,
        'cb': 0,
        'cdm': 0,
        'lm': 0,
        'rm': 0,
        'lw': 0,
        'rw': 0,
        'cf': 0,
        'st': 0
    }

    # Membaca model yang telah dilatih sebelumnya
    with open('../Model/model_pertama.pkl', 'rb') as file:
        loaded_model_1 = pickle.load(file)

    with open('../Model/model_kedua.pkl', 'rb') as file:
        loaded_model_2 = pickle.load(file)

    with open('../Model/model_ketiga.pkl', 'rb') as file:
        loaded_model_3 = pickle.load(file)
        
    # Melakukan prediksi menggunakan setiap model
    predictions_1 =np.round(loaded_model_1.predict(user_skills[atribut_model_1].values.reshape(1, -1))).astype(int)
    predictions_2 =np.round(loaded_model_2.predict(user_skills[atribut_model_2].values.reshape(1, -1))).astype(int)
    predictions_3 =np.round(loaded_model_3.predict(user_skills[atribut_model_3].values.reshape(1, -1))).astype(int)
    
    #Model 1
    keys_to_save_1 = ['rwb', 'lwb', 'rb', 'lb', 'cdm', 'lw', 'rw', 'st']

    for i, key in enumerate(keys_to_save_1):
        skill_pred[key] = predictions_1[0][i]

    #Model 2
    keys_to_save_2 = ['cb']
    skill_pred.update({key: prediction for key, prediction in zip(keys_to_save_2, predictions_2)})

    #Model 3
    keys_to_save_3 = ['lm', 'rm', 'cf']
    for i, key in enumerate(keys_to_save_3):
        skill_pred[key] = predictions_3[0][i]
    
    # Mencari pemain-pemain serupa berdasarkan prediksi kemampuan
    players_alike = find_top_min_variance_rows(skill_pred, players_data, top_n=3)
    
    infer = skill_pred
    
    infer['similar_players'] = []

    # Menambahkan informasi pemain-pemain serupa ke dalam kamus prediksi
    for i, player in enumerate(players_alike):
        infer['similar_players'].append(player)
        
    return infer

# contoh pemakaian 
# player_skills = attributes.drop('long_name', axis=1)
# infer(player_skills.loc[143050], player_position_transformed)