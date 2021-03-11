"""
Method by Thakkar N. from
"Balancing sensitivity and specificity in distinguishing TCR groups by CDR sequence similarity"
The method is based on the Smith Watermann algorithm (SW) with Blosum Scoring and Gap penalty of 10.
"""
import swalign
import math
import Utils.Configurations as Dirs


def get_cdr_dist_value(sequence_1, sequence_2):
    """
    Calculates the distances between to CDR3 regions based on method in paper above
    :param sequence_1: Sequence of the CDR3 region of TCR 1
    :param sequence_2: Sequence of the CDR3 region of TCR 2
    :return: scoring value [0, 1] indicating the distance between both sequences
    """
    blosum_scoring = swalign.ScoringMatrix(Dirs.PATH_BLOSUM_45)
    sw_setup = swalign.LocalAlignment(blosum_scoring, -10)

    unnormalized_score = calculate_swaglin_score(sequence_1, sequence_2, sw_setup)

    normalizing_1 = calculate_swaglin_score(sequence_1, sequence_1, sw_setup)
    normalizing_2 = calculate_swaglin_score(sequence_2, sequence_2, sw_setup)

    similarity = unnormalized_score**2 / (normalizing_1 * normalizing_2)
    similarity = math.sqrt(similarity)

    distance = 1 - similarity
    return distance


def calculate_swaglin_score(sequence_1, sequence_2, sw_setup):
    """
    Calculate the Smith Waterman score between to CDR3 sequences.
    :param sequence_1: str representing a first amino acid with single letter code
    :param sequence_2: str representing a second amino acid with single letter code
    :param sw_setup: swalign object containing scoring matrix and gap penalty
    :return: float representing the similarity between both sequences
    """
    similarity = sw_setup.align(sequence_1, sequence_2)
    similarity = similarity.score
    return similarity
