# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

def create_mapping_code_one_hot(manoeuvers_code_file, mca_code_files):
    """

    :param manoeuvers_code_file:
    :param mca_code_files:
    :return: a function that takes code string as input and gives a 1D array as output (one hot encoding of the code)
    """
    pass

def encode_data(data_file, mapping_code_one_hot):
    """

    :param data_file:
    :param mapping_code_one_hot:
    :return: a ND array of the encoded data read from file (X) and 1D array of Y
    """
    pass

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

    main()
