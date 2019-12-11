# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def make_dictionnary_MCA(df_mca):
    df_mca["Unnamed: 6"].fillna("D", inplace=True)
    df_mca["Unnamed: 5"] = df_mca["Unnamed: 5"].replace(["8"], "L")
    dim = 4
    vec = np.zeros((len(df_mca), dim))

    # first letter weight
    values_weight = df_mca.apply(lambda row: row["Weight"].strip(" kg"), axis=1)
    vec[:, 0] = values_weight
    # seocnd letter gravity center
    values_longi = df_mca.apply(lambda row: float(row["Longi CG"].strip(" m")), axis=1)
    values_lateral = df_mca.apply(lambda row: float(row["Lateral CG"].strip(" m")) if row["Lateral CG"] != "neutral" else 0, axis=1)
    vec[:, 1] = values_longi
    vec[:, 2] = values_lateral
    # third letter altitude
    values_altitude = df_mca.apply(lambda row: "".join(row["Altitude"].strip("ft").split()[-2:]) if row["Altitude"] != "Zp max" else 15000, axis=1)
    vec[:, 3] = values_altitude

    return (df_mca["Code MCA"] + df_mca["Unnamed: 5"] + df_mca["Unnamed: 6"]).values, vec


def make_map_family_L(df_manoeuvers):
    dim = 13
    vec = np.zeros((len(df_manoeuvers), dim))
    # first letter L
    vec[:, 0] = 1
    # second letter R L T M
    bool_idx_R = df_manoeuvers['letter_2'] == "R"
    vec[bool_idx_R, 1] = 1
    bool_idx_L = df_manoeuvers['letter_2'] == "L"
    vec[bool_idx_L, 2] = 1
    bool_idx_T = df_manoeuvers['letter_2'] == "T"
    vec[bool_idx_T, 3] = 1
    bool_idx_M = df_manoeuvers['letter_2'] == "M"
    vec[bool_idx_M, 4] = 1
    # third letter A
    bool_idx_A = df_manoeuvers['letter_3'] == "A"
    vec[bool_idx_A, 5] = 1
    # third letter B
    bool_idx_B = df_manoeuvers['letter_3'] == "B"
    values_B = df_manoeuvers[bool_idx_B].apply(lambda row: row["letter_4_value"].strip(" *"), axis=1)
    vec[bool_idx_B, 6] = values_B
    # third letter C
    bool_idx_C = df_manoeuvers["letter_3"] == "C"
    idx_C = np.arange(len(bool_idx_C))[bool_idx_C]
    # fourth letter E
    bool_idx_CE = df_manoeuvers.iloc[idx_C]["letter_4"] == "E"
    vec[idx_C[bool_idx_CE], 7] = 1
    # fourth letter B
    bool_idx_CB = df_manoeuvers.iloc[idx_C]["letter_4"] == "B"
    vec[idx_C[bool_idx_CB], 8] = 1
    # fifth letter F L R B
    bool_idx_F = df_manoeuvers['letter_5'] == "F"
    vec[bool_idx_F, 9] = 1
    bool_idx_L = df_manoeuvers['letter_5'] == "L"
    vec[bool_idx_L, 10] = 1
    bool_idx_R = df_manoeuvers['letter_5'] == "R"
    vec[bool_idx_R, 11] = 1
    bool_idx_B = df_manoeuvers['letter_5'] == "B"
    vec[bool_idx_B, 12] = 1

    return df_manoeuvers["code"].values, vec


def make_map_family_dummy(df_manoeuvers):

    nb = 30
    dim = 25
    return np.array([str(elm)[:4] for elm in np.random.rand(nb)]), np.ones((nb, dim))



def create_mapping_code_one_hot(manoeuvers_code_file, mca_code_file, families=None):
    """

    :param manoeuvers_code_file:
    :param mca_code_files:
    :return: a function that takes code string as input and gives a 1D array as output (one hot encoding of the code)
    """

    # for each family: a function that returns a dict: code -> vector
    dct_family_dict_maker = {
        "L": make_map_family_L,
        "DUMMY": make_map_family_dummy
    }

    names_manoeuvers = ["code",
                        "letter_1",
                        "letter_1_value",
                        "letter_2",
                        "letter_2_value",
                        "letter_3",
                        "letter_3_value",
                        "letter_4",
                        "letter_4_value",
                        "letter_5",
                        "letter_5_value",
                        "1", "desc"]
    df_manoeuvers = pd.read_csv(open(manoeuvers_code_file, 'r', errors="ignore"), delimiter=";", names=names_manoeuvers)
    df_manoeuvers = df_manoeuvers.dropna(how="all")
    if families is None:
        families = tuple(np.unique(df_manoeuvers["letter_1"].values))

    dict_flight_configuration_parameters = {}

    total_nb_manoeuvers = 0
    dim_total_encoding = 0

    for family in families:
        bool_index_family = df_manoeuvers["code"].str.startswith(family, na=False)
        df_family = df_manoeuvers[bool_index_family]
        codes_family, encodings_family = dct_family_dict_maker[family](df_family)
        dict_flight_configuration_parameters[family] = (codes_family, encodings_family)
        total_nb_manoeuvers += codes_family.shape[0]
        dim_total_encoding += encodings_family.shape[1]

    df_mca = pd.read_csv(open(mca_code_file, 'r', errors="ignore"), delimiter=";")
    df_mca = df_mca.dropna(how="all", axis=0)
    df_mca = df_mca.dropna(how="all", axis=1)

    codes_mca, encodings_mca = make_dictionnary_MCA(df_mca)
    nb_mca, dim_mca = encodings_mca.shape
    # total_nb_manoeuvers *= codes_mca.shape[0]
    # dim_total_encoding += encodings_mca.shape[1]

    families_codes = np.empty((total_nb_manoeuvers,), dtype="object")
    families_encodings = np.zeros((total_nb_manoeuvers, dim_total_encoding))
    i_pos, j_pos = 0, 0
    for family in sorted(families):
        codes_family, encodings_family = dict_flight_configuration_parameters[family]
        nb_encodings_to_add = encodings_family.shape[0]
        nb_dim_encoding_to_add = encodings_family.shape[1]
        i_pos_next, j_pos_next = i_pos+nb_encodings_to_add, j_pos+nb_dim_encoding_to_add
        families_encodings[i_pos:i_pos_next, j_pos:j_pos_next] = encodings_family
        families_codes[i_pos:i_pos_next] = codes_family
        i_pos, j_pos = i_pos_next, j_pos_next

    repeated_families_encodings = np.tile(families_encodings.T, nb_mca).T
    repeated_mca = np.tile(encodings_mca.T, families_encodings.shape[0]).T

    repeated_family_codes = np.tile(families_codes.T, nb_mca).T
    repeated_mca_codes = np.tile(codes_mca.T, families_codes.shape[0]).T

    final_encodings = np.hstack([repeated_families_encodings, repeated_mca])
    final_codes = repeated_family_codes + repeated_mca_codes
    dict_final_mapping = dict((code, final_encodings[idx]) for idx, code in enumerate(final_codes))
    return dict_final_mapping
    # return dict_final_mapping.get


def encode_data(data_file, mapping_code_one_hot, families=None):
    """

    :param data_file:
    :param mapping_code_one_hot:
    :return: a ND array of the encoded data read from file (X) and 1D array of Y
    """
    df_data = pd.read_csv(data_file, delimiter=";", header=None)
    if families is None:
        families = tuple(np.unique(df_data[0].apply(lambda row: row[0][0])))

    for family in families:
        df_data_family = df_data[df_data[0].str.startswith(family)]
    df_data_family.apply(lambda row: mapping_code_one_hot(row[0]))

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    manoeuvers_code_file = Path(os.environ["project_dir"]) / "data/raw/manoeuver_code.csv"
    mca_code_file = Path(os.environ["project_dir"]) / "data/raw/MCA_code.csv"
    data_main_rotor = Path(os.environ["project_dir"]) / "data/raw/Data_Main_rotor.csv"

    families = ("L")

    mapping_code_one_hot = create_mapping_code_one_hot(manoeuvers_code_file, mca_code_file, families=families)
    encode_data(data_main_rotor, mapping_code_one_hot, families=families)
    print(mapping_code_one_hot("LLAXXGIB"))